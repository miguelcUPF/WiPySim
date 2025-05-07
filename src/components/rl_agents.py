from src.sim_params import SimParams as sparams
from src.user_config import UserConfig as cfg
from src.utils.event_logger import get_logger
from src.components.network import AP

from collections import deque

import numpy as np
import simpy
import random
import wandb


# https://arxiv.org/pdf/1003.0146
# https://dl.acm.org/doi/abs/10.1145/3297280.3297440?casa_token=eoZgPNBt-AUAAAAA:o80ERr_mN7BeM9GFgjH801INiTUf31_9OYERVQfAnnHPYEC6K9i00knEYUwMpcR_ZQeGwNq6yn9tOMU
class SWLinUCB:
    def __init__(
        self,
        name: str,
        n_actions: int,
        context_dim: int,
        marl_controller,
        strategy: str = "sw_linucb",
        weights_r: dict[str, float] = None,
        alpha: float = 1.0,
        window_size: int | None = None,
    ):
        self.name = name
        self.n_actions = n_actions
        self.context_dim = context_dim

        self.marl_controller = marl_controller

        self.strategy = strategy
        self.alpha = alpha
        self.weights_r = weights_r or {}

        self.A = [np.identity(context_dim) for _ in range(n_actions)]
        self.b = [np.zeros(context_dim) for _ in range(n_actions)]

        # SW-LinUCB
        self.time_step = 0
        self.window_size = window_size if window_size is not None else n_actions
        self.E = [deque(maxlen=self.window_size) for _ in range(n_actions)]

    def _linucb(self, context, valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))

        p = np.full(self.n_actions, -np.inf)
        for a in valid_actions:
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p[a] = context @ theta + self.alpha * np.sqrt(context @ A_inv @ context)
        action = np.argmax(p)
        return action

    def _sw_linucb(self, context, valid_actions=None):
        self.time_step += 1
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))

        p = np.full(self.n_actions, -np.inf)
        for a in valid_actions:
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]

            if self.window_size == 0:
                # Act like LinUCB: ignore gamma_t
                p[a] = context @ theta + self.alpha * np.sqrt(context @ A_inv @ context)
            else:
                occ = sum(self.E[a]) if self.time_step > self.window_size else 0
                gamma_t = occ / self.window_size
                # since our rewards are negative, we consider (occ / self.window_size) rather than (1 - occ / self.window_size) to penalize frequently selected actions

                p[a] = gamma_t * (context @ theta) + self.alpha * np.sqrt(
                    context @ A_inv @ context
                )
        action = np.argmax(p)
        for a in range(self.n_actions):
            self.E[a].append(1 if a == action else 0)
        return action

    def select_action(self, context, valid_actions=None):
        if self.strategy == "linucb":
            return self._linucb(context, valid_actions)
        elif self.strategy == "sw_linucb":
            return self._sw_linucb(context, valid_actions)
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

    def update(self, context, action, reward):
        x = context
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

    def reset(self):
        self.A = [np.identity(self.context_dim) for _ in range(self.n_actions)]
        self.b = [np.zeros(self.context_dim) for _ in range(self.n_actions)]
        self.E = [deque(maxlen=self.window_size) for _ in range(self.n_actions)]
        self.time_step = 0


class EpsRMSProp:
    """
    Epsilon-greedy Contextual Multi-Armed Bandit with linear reward approximation
    and RMSProp-based weight updates.
    """

    def __init__(
        self,
        name: str,
        n_actions: int,
        context_dim: int,
        marl_controller,
        strategy: str = "epsilon_greedy",
        weights_r: dict[str, float] = None,
        epsilon: float = 0.1,
        decay_rate: float = 0.99,
        eta: float = 0.1,
        gamma: float = 0.9,
        alpha_ema: float = 0.1,  # EMA factor
    ):

        self.name = name

        self.n_actions = n_actions
        self.context_dim = context_dim

        self.marl_controller = marl_controller

        self.strategy = strategy

        self.weights_r = weights_r or {}  # Used if decomposition enabled

        # (decay) epsilon-greedy
        self.epsilon = epsilon
        self.decay_rate = decay_rate

        self.eta = eta  # Learning rate
        self.gamma = gamma  # RMSProp decay factor
        self.epsilon_rms = 1e-8  # for numerical stability

        self.alpha_ema = alpha_ema

        # Linear model: one weight vector per action
        self.weight_matrix = np.zeros((n_actions, context_dim))
        self.weight_matrix_ema = np.zeros((n_actions, context_dim))  # EMA of weights

        # RMSProp: moving average of squared gradients
        self.grad_squared_avg = np.zeros((n_actions, context_dim))

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
            preds = self.weight_matrix_ema @ context
            # Create a masked array with -inf for invalid actions to prevent them from being selected
            masked_preds = np.full_like(preds, -np.inf)
            for a in valid_actions:
                masked_preds[a] = preds[a]
            return np.argmax(masked_preds)  # Exploit
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
        Updates weights using RMSProp and maintains EMA of weights.
        """
        pred = self.weight_matrix[action] @ context
        error = pred - reward
        gradient = error * context

        # Update RMSProp memory
        self.grad_squared_avg[action] = self.gamma * self.grad_squared_avg[action] + (
            1 - self.gamma
        ) * (gradient**2)

        # Parameter update
        self.weight_matrix[action] -= (
            self.eta
            / (np.sqrt(self.grad_squared_avg[action] + self.epsilon_rms))
            * gradient
        )

        # EMA update
        self.weight_matrix_ema[action] = (
            self.alpha_ema * self.weight_matrix_ema[action]
            + (1 - self.alpha_ema) * self.weight_matrix[action]
        )

    def reset(self):
        self.weight_matrix = np.zeros((self.n_actions, self.context_dim))
        self.weight_matrix_ema = np.zeros((self.n_actions, self.context_dim))
        self.grad_squared_avg = np.zeros((self.n_actions, self.context_dim))


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
        strategy = settings.get("strategy", "sw_linucb")

        if strategy in ["sw_linucb", "linucb"]:
            agent_class = SWLinUCB
        elif strategy in ["epsilon_greedy", "decay_epsilon_greedy"]:
            agent_class = EpsRMSProp
        else:
            raise ValueError(f"Unknown strategy {strategy}")

        padding_ctxt = 1 if settings.get("include_prev_decision", False) else 0

        channel_params = {
            "name": "channel_agent",
            "n_actions": 7,  # 0: {1}, 1: {2}, 2: {3}, 3: {4}, 4: {1, 2}, 5: {3, 4}, 6: {1, 2, 3, 4}
            "context_dim": 9
            + padding_ctxt,  # 4x channel contenders + 4x channel busy flags + 1x queue size + (prev decision: current channel (mapped idx))
            "strategy": strategy,
            "weights_r": settings.get("channel_weights", {}),
        }

        primary_params = {
            "name": "primary_agent",
            "n_actions": 4,  # 0: {1}, 1: {2}, 2: {3}, 3: {4} (depending on channel)
            "context_dim": 9
            + padding_ctxt,  # 1x current channel (mapped idx) + 4x channel contenders + 4x channel busy flags + (prev decision: current primary (mapped idx))
            "strategy": strategy,
            "weights_r": settings.get("primary_weights", {}),
        }

        cw_params = {
            "name": "cw_agent",
            "n_actions": 7,  # 0: {16}, 1: {32}, 2: {64}, 3: {128}, 4: {256}, 5: {512}, 6: {1024} (i.e., 2**(x+4))
            "context_dim": 11
            + padding_ctxt,  #  1x current channel (mapped idx) + 1x current primary (mapped idx) + 4x channel contenders + 4x channel busy flags + 1x queue size + (prev decision: current cw)
            "strategy": strategy,
            "weights_r": settings.get("cw_weights", {}),
        }

        if agent_class == EpsRMSProp:
            channel_params.update(
                {
                    "epsilon": settings.get("epsilon", 0.1),
                    "decay_rate": settings.get("decay_rate", 0.99),
                    "eta": settings.get("eta", 0.1),
                    "gamma": settings.get("gamma", 0.9),
                    "alpha_ema": settings.get("alpha_ema", 0.1),
                }
            )
            primary_params.update(
                {
                    "epsilon": settings.get("epsilon", 0.1),
                    "decay_rate": settings.get("decay_rate", 0.99),
                    "eta": settings.get("eta", 0.1),
                    "gamma": settings.get("gamma", 0.9),
                    "alpha_ema": settings.get("alpha_ema", 0.1),
                }
            )
            cw_params.update(
                {
                    "epsilon": settings.get("epsilon", 0.1),
                    "decay_rate": settings.get("decay_rate", 0.99),
                    "eta": settings.get("eta", 0.1),
                    "gamma": settings.get("gamma", 0.9),
                    "alpha_ema": settings.get("alpha_ema", 0.1),
                }
            )
        elif agent_class == SWLinUCB:
            channel_params.update(
                {
                    "alpha": settings.get("alpha", 1.0),
                    "window_size": settings.get("window_size", None),
                }
            )
            primary_params.update(
                {
                    "alpha": settings.get("alpha", 1.0),
                    "window_size": settings.get("window_size", None),
                }
            )
            cw_params.update(
                {
                    "alpha": settings.get("alpha", 1.0),
                    "window_size": settings.get("window_size", None),
                }
            )

        self.channel_agent = agent_class(**channel_params, marl_controller=self)
        self.primary_agent = agent_class(**primary_params, marl_controller=self)
        self.cw_agent = agent_class(**cw_params, marl_controller=self)

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
                    f"node_{self.node.id}/delay/sensing": delay_components[
                        "sensing_delay"
                    ],
                    f"node_{self.node.id}/delay/backoff": delay_components[
                        "backoff_delay"
                    ],
                    f"node_{self.node.id}/delay/tx": delay_components["tx_delay"],
                    f"node_{self.node.id}/delay/residual": delay_components[
                        "residual_delay"
                    ],
                    f"node_{self.node.id}/delay/total": sum(delay_components.values()),
                    "env_time_us": self.env.now,
                }
            )
