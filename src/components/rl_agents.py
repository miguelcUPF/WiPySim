from src.sim_params import SimParams as sparams
from src.user_config import UserConfig as cfg
from src.utils.event_logger import get_logger
from src.components.network import AP

import numpy as np
import simpy
import random


class ContextualMAB:
    def __init__(
        self,
        name: str,
        n_actions: int,
        context_dim: int,
        strategy: str = "epsilon_greedy",
        epsilon: float = 0.1,
        decay_rate: float = 0.99,
        alpha_q: float = 0.1,
        alpha_r: float = 0.9,
        weights: dict[str, float] = None,
    ):

        self.name = name

        self.n_actions = n_actions
        self.context_dim = context_dim

        self.strategy = strategy

        # epsilon-greedy
        self.epsilon = epsilon
        # decay epsilon-greedy
        self.decay_rate = decay_rate

        # exponential moving average (EMA) factor
        self.alpha_q = alpha_q
        self.alpha_r = alpha_r

        self.reward_mean = (
            None  # lazy initialization: done once the first reward is observed
        )
        self.reward_std = None

        self.weights = weights or {}  # Used if decomposition enabled

        self.q_values = np.zeros((self.n_actions, context_dim))

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
            preds = self.q_values @ context
            action = np.argmax(preds)  # Exploit
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

        # EMA for non-stationary environments
        self.q_values[action] += (
            self.alpha_q * (normalized_reward - self.q_values[action]) * context
        )

    def reset(self):
        self.q_values = [np.zeros(self.context_dim) for _ in range(self.n_actions)]
        self.step = 1
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
        self.logger = get_logger(self.name, cfg, sparams, env)

        self.node = node

        self.settings = settings

        self.channel_agent = ContextualMAB(
            name="channel_agent",
            n_actions=7,  # 0: {1}, 1: {2}, 2: {3}, 3: {4}, 4: {1, 2}, 5: {3, 4}, 6: {1, 2, 3, 4}
            context_dim=10,  # 1x current channel (mapped idx) + 4x channel contenders + 4x channel busy flags + 1x queue size
            strategy=settings.get("strategy", "epsilon_greedy"),
            epsilon=settings.get("epsilon", 0.1),
            decay_rate=settings.get("decay_rate", 0.99),
            alpha_q=settings.get("alpha_q", 0.1),
            alpha_r=settings.get("alpha_r", 0.9),
            weights=settings.get("channel_weights", {}),
        )
        self.primary_agent = ContextualMAB(
            n_actions=4,  # 0: {1}, 1: {2}, 2: {3}, 3: {4} (depending on channel)
            context_dim=10,  # 1x current primary (mapped idx) + 1x current channel (mapped idx) + 4x channel contenders + 4x channel busy flags
            strategy=settings.get("strategy", "epsilon_greedy"),
            epsilon=settings.get("epsilon", 0.1),
            decay_rate=settings.get("decay_rate", 0.99),
            alpha_q=settings.get("alpha_q", 0.1),
            alpha_r=settings.get("alpha_r", 0.9),
            weights=settings.get("primary_weights", {}),
        )
        self.cw_agent = ContextualMAB(
            n_actions=3,  # 0: decrease, 1: maintain, 2: increase
            context_dim=12,  # 1x current cw (mapped idx) + 1x current primary (mapped idx) + 1x current channel (mapped idx) + 4x channel contenders + 4x channel busy flags  + 1x queue size
            strategy=settings.get("strategy", "epsilon_greedy"),
            epsilon=settings.get("epsilon", 0.1),
            decay_rate=settings.get("decay_rate", 0.99),
            alpha_q=settings.get("alpha_q", 0.1),
            alpha_r=settings.get("alpha_r", 0.9),
            weights=settings.get("cw_weights", {}),
        )

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
                delay_components, self.channel_agent.weights
            )
            r_pr = self._compute_weighted_reward(
                delay_components, self.primary_agent.weights
            )
            r_cw = self._compute_weighted_reward(
                delay_components, self.cw_agent.weights
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
