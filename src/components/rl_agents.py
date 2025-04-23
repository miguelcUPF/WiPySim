from src.sim_params import SimParams as sparams
from src.user_config import UserConfig as cfg
from src.utils.event_logger import get_logger
from src.components.network import AP

import numpy as np
import simpy
import random
import wandb


# https://arxiv.org/pdf/1003.0146
class LinUcbCMAB:
    def __init__(
        self,
        name: str,
        n_actions: int,
        context_dim: int,
        strategy: str = "linucb",
        weights_r: dict[str, float] = None,
        alpha: float = 1.0,
    ):
        self.name = name
        self.n_actions = n_actions
        self.context_dim = context_dim

        self.strategy = strategy
        self.alpha = alpha  # Confidence bound parameter for LinUCB
        self.weights_r = weights_r or {}

        self.A = [np.identity(context_dim) for _ in range(n_actions)]
        self.b = [np.zeros(context_dim) for _ in range(n_actions)]

    def _linucb(self, context, valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))

        p = np.full(self.n_actions, -np.inf)
        for a in valid_actions:
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p[a] = context @ theta + self.alpha * np.sqrt(context @ A_inv @ context)
        return np.argmax(p)

    def select_action(self, context, valid_actions=None):
        if self.strategy == "linucb":
            return self._linucb(context, valid_actions)
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

    def update(self, context, action, reward):
        x = context
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

    def reset(self):
        self.A = [np.identity(self.context_dim) for _ in range(self.n_actions)]
        self.b = [np.zeros(self.context_dim) for _ in range(self.n_actions)]


class EpsCMAB:
    def __init__(
        self,
        name: str,
        n_actions: int,
        context_dim: int,
        strategy: str = "epsilon_greedy",
        weights_r: dict[str, float] = None,
        epsilon: float = 0.1,
        decay_rate: float = 0.99,
        alpha_q: float = 0.1,
        alpha_r: float = 0.9,
    ):

        self.name = name

        self.n_actions = n_actions
        self.context_dim = context_dim

        self.strategy = strategy

        # epsilon-greedy
        self.epsilon = epsilon
        # decay epsilon-greedy
        self.decay_rate = decay_rate

        self.alpha_q = (
            alpha_q  # linear gradient descent step size for weight matrix update
        )
        self.alpha_r = (
            alpha_r  # for exponential moving average (EMA) factor reward normalization
        )

        self.reward_mean = (
            None  # lazy initialization: done once the first reward is observed
        )
        self.reward_std = None

        self.weights_r = weights_r or {}  # Used if decomposition enabled

        # Weight matrix, i.e., one weight vector per action for linear reward estimation
        self.weight_matrix = np.zeros((self.n_actions, self.context_dim))

    def _epsilon_greedy(self, context, valid_actions=None):
        """
        Epsilon-greedy algorithm:
            with probability 1-ε, choose the action with the highest Q-value (exploitation)
            with probability ε, choose a random action (exploration)
        """
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))

        if random.random() < self.epsilon:
            action = random.choice(valid_actions)  # Explore
        else:
            preds = self.weight_matrix @ context
            # Create a masked array with -inf for invalid actions to prevent them from being selected
            masked_preds = np.full_like(preds, -np.inf)
            for a in valid_actions:
                masked_preds[a] = preds[a]
            action = np.argmax(masked_preds)  # Exploit
        return action

    def _decay_epsilon_greedy(self, context, valid_actions=None):
        self.epsilon *= self.decay_rate
        return self._epsilon_greedy(context, valid_actions)

    def select_action(self, context, valid_actions=None):
        if self.strategy == "epsilon_greedy":
            return self._epsilon_greedy(context, valid_actions)
        elif self.strategy == "decay_epsilon_greedy":
            return self._decay_epsilon_greedy(context, valid_actions)
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

    def update(self, context, action, reward):
        """
        Updates the weight matrix based on the observed reward and context using linear approximation.
        """
        if self.reward_mean is None:
            self.reward_mean = reward
            self.reward_std = 0
        # EMA Z-score normalization https://arxiv.org/pdf/2101.08482
        self.reward_mean = self.alpha_r * reward + (1 - self.alpha_r) * self.reward_mean
        self.reward_std = (
            self.alpha_r * (reward - self.reward_mean) ** 2
            + (1 - self.alpha_r) * self.reward_std
        )

        normalized_reward = (reward - self.reward_mean) / (
            np.sqrt(self.reward_std) + 1e-8
        )  # Added a small constant to avoid division by zero

        # Update weights using linear SGD
        pred = self.weight_matrix[action] @ context
        error = normalized_reward - pred
        self.weight_matrix[action] += self.alpha_q * error * context

    def reset(self):
        self.weight_matrix = np.zeros((self.n_actions, self.context_dim))
        self.reward_mean = None
        self.reward_std = None


class MARLAgentController:
    def __init__(
        self,
        sparams: sparams,
        cfg: cfg,
        env: simpy.Environment,
        node: AP,
        settings: dict,
    ):
        self.cfg = cfg
        self.sparams = sparams
        self.env = env

        self.name = "MARL"
        self.logger = get_logger(
            self.name,
            cfg,
            sparams,
            env,
            True if node.id in self.cfg.EXCLUDED_IDS else False,
        )

        self.node = node

        self.settings = settings

        # Select agent types based on the strategy setting
        strategy = settings.get("strategy", "linucb")

        if strategy == "linucb":
            agent_class = LinUcbCMAB
        elif strategy in ["epsilon_greedy", "decay_epsilon_greedy"]:
            agent_class = EpsCMAB
        else:
            raise ValueError(f"Unknown strategy {strategy}")

        channel_params = {
            "name": "channel_agent",
            "n_actions": 7,  # 0: {1}, 1: {2}, 2: {3}, 3: {4}, 4: {1, 2}, 5: {3, 4}, 6: {1, 2, 3, 4}
            "context_dim": 10,  # 1x current channel (mapped idx) + 4x channel contenders + 4x channel busy flags + 1x queue size
            "strategy": strategy,
            "weights_r": settings.get("channel_weights", {}),
        }

        primary_params = {
            "name": "primary_agent",
            "n_actions": 4,  # 0: {1}, 1: {2}, 2: {3}, 3: {4} (depending on channel)
            "context_dim": 10,  # 1x current primary (mapped idx) + 1x current channel (mapped idx) + 4x channel contenders + 4x channel busy flags
            "strategy": strategy,
            "weights_r": settings.get("primary_weights", {}),
        }

        cw_params = {
            "name": "cw_agent",
            "n_actions": 3,  # 0: decrease, 1: maintain, 2: increase
            "context_dim": 12,  # 1x current cw (mapped idx) + 1x current primary (mapped idx) + 1x current channel (mapped idx) + 4x channel contenders + 4x channel busy flags + 1x queue size
            "strategy": strategy,
            "weights_r": settings.get("cw_weights", {}),
        }

        if agent_class == EpsCMAB:
            channel_params.update(
                {
                    "epsilon": settings.get("epsilon", 0.1),
                    "decay_rate": settings.get("decay_rate", 0.99),
                    "alpha_q": settings.get("alpha_q", 0.1),
                    "alpha_r": settings.get("alpha_r", 0.9),
                }
            )
            primary_params.update(
                {
                    "epsilon": settings.get("epsilon", 0.1),
                    "decay_rate": settings.get("decay_rate", 0.99),
                    "alpha_q": settings.get("alpha_q", 0.1),
                    "alpha_r": settings.get("alpha_r", 0.9),
                }
            )
            cw_params.update(
                {
                    "epsilon": settings.get("epsilon", 0.1),
                    "decay_rate": settings.get("decay_rate", 0.99),
                    "alpha_q": settings.get("alpha_q", 0.1),
                    "alpha_r": settings.get("alpha_r", 0.9),
                }
            )
        elif agent_class == LinUcbCMAB:
            channel_params.update(
                {
                    "alpha": settings.get("alpha", 1.0),
                }
            )
            primary_params.update(
                {
                    "alpha": settings.get("alpha", 1.0),
                }
            )
            cw_params.update(
                {
                    "alpha": settings.get("alpha", 1.0),
                }
            )

        self.channel_agent = agent_class(**channel_params)
        self.primary_agent = agent_class(**primary_params)
        self.cw_agent = agent_class(**cw_params)

        self.last_channel_action = None
        self.last_primary_action = None
        self.last_cw_action = None

        self.last_channel_context = None
        self.last_primary_context = None
        self.last_cw_context = None

    def decide_channel(self, context):
        self.last_channel_context = context
        action = self.channel_agent.select_action(context)
        self.last_channel_action = action
        self.logger.debug(
            f"{self.node.type} {self.node.id} -> Channel action: {action}"
        )
        return action

    def decide_primary(self, context, allocated_channels):
        self.last_primary_context = context
        valid_actions = [c - 1 for c in allocated_channels]
        action = self.primary_agent.select_action(context, valid_actions)
        self.last_primary_action = action
        self.logger.debug(
            f"{self.node.type} {self.node.id} -> Primary action: {action}"
        )
        return action

    def decide_cw(self, context):
        self.last_cw_context = context
        action = self.cw_agent.select_action(context)
        self.last_cw_action = action
        self.logger.debug(f"{self.node.type} {self.node.id} -> CW action: {action}")
        return action

    def _compute_weighted_reward(self, delay_components: dict, weights: dict):
        return -sum(weights.get(k, 0) * delay_components[k] for k in weights)

    def update_agents(self, delay_components: dict):
        if self.cfg.ENABLE_REWARD_DECOMPOSITION:
            # Per-agent weighted reward
            r_ch = self._compute_weighted_reward(
                delay_components, self.channel_agent.weights_r
            )
            r_pr = self._compute_weighted_reward(
                delay_components, self.primary_agent.weights_r
            )
            r_cw = self._compute_weighted_reward(
                delay_components, self.cw_agent.weights_r
            )
        else:
            # Shared reward
            r_ch = r_pr = r_cw = -(sum(delay_components.values()))  # (minimize delay)

        self.channel_agent.update(
            self.last_channel_context, self.last_channel_action, r_ch
        )
        self.primary_agent.update(
            self.last_primary_context, self.last_primary_action, r_pr
        )
        self.cw_agent.update(self.last_cw_context, self.last_cw_action, r_cw)

        self._log_to_wandb(
            delay_components, {"channel": r_ch, "primary": r_pr, "cw": r_cw}
        )

    def _log_to_wandb(self, delay_components: dict, reward: dict):
        if wandb.run:
            wandb.log(
                {
                    f"node_{self.node.id}/action/channel": self.last_channel_action,
                    f"node_{self.node.id}/action/primary": self.last_primary_action,
                    f"node_{self.node.id}/action/cw": self.last_cw_action,
                    f"node_{self.node.id}/action/cw_current": self.node.mac_layer.cw_current,
                    f"node_{self.node.id}/reward/channel": reward["channel"],
                    f"node_{self.node.id}/reward/primary": reward["primary"],
                    f"node_{self.node.id}/reward/cw": reward["cw"],
                    f"node_{self.node.id}/delay/sensing": delay_components["sensing_delay"],
                    f"node_{self.node.id}/delay/backoff": delay_components["backoff_delay"],
                    f"node_{self.node.id}/delay/tx": delay_components["tx_delay"],
                    f"node_{self.node.id}/delay/residual": delay_components[
                        "residual_delay"
                    ],
                    f"node_{self.node.id}/delay/total": sum(delay_components.values()),
                    "env_time_us": self.env.now,
                }
            )
