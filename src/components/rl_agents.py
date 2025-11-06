from src.sim_params import SimParams as sparams_module
from src.user_config import UserConfig as cfg_module
from src.utils.event_logger import get_logger
from src.components.network import AP

from collections import deque
from codecarbon import EmissionsTracker

import numpy as np
import simpy
import random
import wandb


CHANNEL_MAP = {
    0: {1},
    1: {2},
    2: {3},
    3: {4},
    4: {1, 2},
    5: {3, 4},
    6: {1, 2, 3, 4},
}
PRIMARY_CHANNEL_MAP = {0: {1}, 1: {2}, 2: {3}, 3: {4}}
CW_MAP = {i: 2 ** (4 + i) for i in range(7)}

META_MAP = {i: 2**i for i in range(4)}  # joint freq for CSA, PCSA, CWSA
META_MAP_multifreq = {
    0: (1, 1, 1),
    1: (2, 2, 1),
    2: (2, 2, 2),
    3: (4, 4, 1),
    4: (4, 4, 2),
    5: (4, 4, 4),
    6: (8, 8, 1),
    7: (8, 8, 2),
    8: (8, 8, 4),
    9: (8, 8, 8),
}  # (CSA_freq, PCSA_freq, CWSA_freq)


MIN_REWARD = -10e3  # equal to CH_ACCESS_TIMEOUT_us
MAX_REWARD = 0


class UCB:
    def __init__(
        self,
        name: str,
        n_actions: int,
        marl_controller,
        weights_r: dict[str, float] = None,
        alpha: float = 4.0,  # alpha = 4 for UCB-1
        min_val: float = MIN_REWARD,
        max_val: float = MAX_REWARD,
        rng: random.Random | None = None,
    ):
        self.name = name
        self.n_actions = n_actions

        self.marl_controller = marl_controller

        self.weights_r = weights_r or {}  # Used if decomposition enabled

        self.alpha = alpha

        self.counts = np.zeros(n_actions)  # Number of pulls per arm
        self.values = np.zeros(n_actions)  # Average reward per arm
        self.time_step = 0

        # Normalization
        self.min_val = min_val
        self.max_val = max_val

        self.rng = rng

    def _normalize_reward(self, reward):
        # Clipping
        clipped_reward = max(min(reward, self.max_val), self.min_val)

        # Normalize the reward to the range [0, 1]
        normalized_reward = (clipped_reward - self.min_val) / (
            self.max_val - self.min_val
        )

        return normalized_reward

    def select_action(self, valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))

        self.time_step += 1

        # Pull each arm once if not pulled yet
        for a in valid_actions:
            if self.counts[a] == 0:
                return a

        ucb_values = np.full(self.n_actions, -np.inf)
        for a in valid_actions:
            ucb_values[a] = self.values[a] + np.sqrt(
                self.alpha * np.log(self.time_step) / (2 * self.counts[a])
            )

        max_ucb = np.max(ucb_values)
        candidate_actions = [a for a in valid_actions if ucb_values[a] == max_ucb]

        action = (
            self.rng.choice(candidate_actions)
            if self.rng
            else np.random.choice(candidate_actions)
        )
        return action

    def update(self, action, reward):
        reward = self._normalize_reward(reward)
        self.counts[action] += 1
        n = self.counts[action]

        # Update running average
        self.values[action] = ((n - 1) / n) * self.values[action] + (1 / n) * reward

    def reset(self):
        self.counts = np.zeros(self.n_actions)
        self.values = np.zeros(self.n_actions)
        self.time_step = 0


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
        min_val: float = MIN_REWARD,
        max_val: float = MAX_REWARD,
        window_size: int | None = None,
        rng: random.Random | None = None,
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

        # Normalization
        self.min_val = min_val
        self.max_val = max_val

        self.rng = rng

    def _linucb(self, context, valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))

        p = np.full(self.n_actions, -np.inf)
        for a in valid_actions:
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p[a] = context @ theta + self.alpha * np.sqrt(context @ A_inv @ context)
        max_p = np.max(p)
        candidate_actions = np.where(p == max_p)[0]
        action = np.random.choice(candidate_actions)
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

                gamma_t = 1 - (occ / self.window_size)

                p[a] = gamma_t * (context @ theta) + self.alpha * np.sqrt(
                    context @ A_inv @ context
                )

        max_p = np.max(p)
        candidate_actions = np.where(p == max_p)[0]
        action = np.random.choice(candidate_actions)
        for a in range(self.n_actions):
            self.E[a].append(1 if a == action else 0)
        return action

    def _normalize_reward(self, reward):
        # Clipping
        clipped_reward = max(min(reward, self.max_val), self.min_val)

        # Normalize the reward to the range [0, 1]
        normalized_reward = (clipped_reward - self.min_val) / (
            self.max_val - self.min_val
        )

        return normalized_reward

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
        self.b[action] += self._normalize_reward(reward) * x

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
        min_val: float = MIN_REWARD,
        max_val: float = MAX_REWARD,
        rng: random.Random | None = None,
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

        self.min_val = min_val
        self.max_val = max_val

        self.rng = rng

    def _epsilon_greedy(self, context, valid_actions=None):
        """
        Epsilon-greedy algorithm:
            with probability 1-ε, choose the action with the highest Q-value (exploitation)
            with probability ε, choose a random action (exploration)
        """
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))

        if (self.rng.random() if self.rng else np.random.random()) < self.epsilon:
            action = (
                self.rng.choice(valid_actions)
                if self.rng
                else np.random.choice(valid_actions)
            )  # Explore
        else:
            preds = self.weight_matrix_ema @ context
            # Create a masked array with -inf for invalid actions to prevent them from being selected
            masked_preds = np.full_like(preds, -np.inf)
            for a in valid_actions:
                masked_preds[a] = preds[a]

            max_p = np.max(masked_preds)
            candidate_actions = np.where(masked_preds == max_p)[0]
            action = np.random.choice(candidate_actions)
            return action  # Exploit
        return action

    def _normalize_reward(self, reward):
        # Clipping
        clipped_reward = max(min(reward, self.max_val), self.min_val)

        # Normalize the reward to the range [0, 1]
        normalized_reward = (clipped_reward - self.min_val) / (
            self.max_val - self.min_val
        )

        return normalized_reward

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
        reward = self._normalize_reward(reward)
        error = reward - pred
        gradient = error * context

        # Update RMSProp memory
        self.grad_squared_avg[action] = self.gamma * self.grad_squared_avg[action] + (
            1 - self.gamma
        ) * (gradient**2)

        # Parameter update
        self.weight_matrix[action] += (
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


class E2TC:  # only for single agent
    def __init__(
        self,
        name: str,
        actions: np.ndarray,
        marl_controller,
        weights_r: dict[str, float] = None,
        T=2 * 10**4,
        sigma2=1 / 4,
        alpha=1,  # recommended between 1 and 3; best alpha=1
        min_val: float = MIN_REWARD,
        max_val: float = MAX_REWARD,
        rng: random.Random | None = None,
    ):
        def select_diverse_base(actions, d, rng):
            selected = []
            used_set = set()

            unique_channels = list(set(tuple(a[:4]) for a in actions))
            num_channels = len(unique_channels)

            cycles = d // num_channels

            for _ in range(cycles):
                rng.shuffle(unique_channels)
                used_cws_in_cycle = set()

                for ch_vec in unique_channels:
                    candidates = [
                        a
                        for a in actions
                        if tuple(a[:4]) == ch_vec
                        and a[8] not in used_cws_in_cycle
                        and tuple(a) not in used_set
                    ]
                    if not candidates:
                        candidates = [
                            a
                            for a in actions
                            if tuple(a[:4]) == ch_vec and tuple(a) not in used_set
                        ]

                    if candidates:
                        choice = rng.choice(candidates)
                        selected.append(choice)
                        used_set.add(tuple(choice))
                        used_cws_in_cycle.add(choice[8])

            # Fill remaining slots randomly
            remaining_slots = d - len(selected)
            if remaining_slots > 0:
                remaining = [a for a in actions if tuple(a) not in used_set]
                rng.shuffle(remaining)
                selected.extend(remaining[:remaining_slots])

            return np.stack(selected)

        d = len(actions[0])

        self.name = name

        self.marl_controller = marl_controller

        self.weights_r = weights_r or {}  # Used if decomposition enabled

        self.actions = np.array([a for a in actions])
        self.base = select_diverse_base(self.actions, d, rng)

        self.sigma2 = sigma2
        self.alpha = alpha
        self.T = T

        self.d = d
        self.X = np.zeros((T, self.d))
        self.Y = np.zeros(T)

        self.t = 0

        self.hattheta = np.zeros(self.d)

        self.phase_1_end_t = None
        self.phase_2_end_t = None

        # phase params
        self.i = 0

        self.deltai = None
        self.Ui = None

        self.a = None
        self.b = None

        self.Ne = None

        # Normalization
        self.min_val = min_val
        self.max_val = max_val

        self.rng = rng

    def _get_phase_1_action(self):
        self.X[self.t, :] = self.base[self.t % self.d, :]
        return self.X[self.t, :]

    def _get_phase_2_action(self):
        self.X[self.t, :] = self.base[self.t % self.d, :]
        return self.X[self.t, :]

    def _get_phase_3_action(self):
        # Return the action with the highest estimated reward
        estimates = {}
        for i, action in enumerate(self.actions):
            estimates[tuple(action)] = action @ self.hattheta
        return self.actions[np.argmax(list(estimates.values()))]

    def _normalize_reward(self, reward):
        # Clipping
        clipped_reward = max(min(reward, self.max_val), self.min_val)

        # Normalize the reward to the range [0, 1]
        normalized_reward = (clipped_reward - self.min_val) / (
            self.max_val - self.min_val
        )
        return normalized_reward

    def select_action(self):
        if self.phase_1_end_t is None:
            if self.a is None:
                self.update_phase_1()

            action = self._get_phase_1_action()

            if self.t == self.b:
                self.update_hattheta(self.a, self.b)
                norm = np.linalg.norm(self.hattheta)

                if norm > self.alpha * self.Ui:
                    self.phase_1_end_t = self.b
                else:
                    self.i += 1
                    self.update_phase_1()
            self.t += 1
            return np.where(np.all(self.actions == action, axis=1))[0][0]

        elif self.phase_2_end_t is None:
            if self.Ne is None:
                self.update_phase_2()

            action = self._get_phase_2_action()

            if self.t == self.b:
                self.phase_2_end_t = self.b
                self.update_hattheta(self.a, self.b)

            self.t += 1
            return np.where(np.all(self.actions == action, axis=1))[0][0]
        else:
            action = self._get_phase_3_action()

            self.t += 1

            return np.where(np.all(self.actions == action, axis=1))[0][0]

    def update(self, y):
        if self.t == 0 or self.t > self.T:
            return
        self.Y[self.t - 1] = self._normalize_reward(y)

    def update_hattheta(self, a, b):
        self.hattheta = (
            np.linalg.inv(
                self.X[a:b, :].T @ self.X[a:b, :] + 1e-6 * np.eye(self.X.shape[1])
            )  # add regularization to avoid singular matrix
            @ self.X[a:b, :].T
            @ self.Y[a:b]
        )

    def update_phase_1(self):
        self.a = self.d * (2**self.i - 1)
        self.b = self.d * (2 ** (self.i + 1) - 1)
        ni = self.d * 2 ** (self.i - 1)
        self.deltai = min(1, self.d * ni / self.T)
        self.Ui = (self.sigma2 * self.d * self.d / ni) * (
            1
            + 2
            * np.sqrt(
                (1 / self.d) * np.log(1 / self.deltai)
                + (2 / self.d) * np.log(1 / self.deltai)
            )
        )

    def update_phase_2(self):
        self.Ne = (
            self.d
            * np.sqrt(self.sigma2)
            * np.ceil(np.sqrt(self.T) / np.linalg.norm(self.hattheta))
        )
        a = self.phase_1_end_t + 1
        self.b = int(a + self.Ne)


class OSUB:  # only for multi agent
    def __init__(
        self,
        name: str,
        n_actions: int,
        marl_controller,
        strategy: str = "sw_osub",
        weights_r: dict[str, float] = None,
        actions: dict = None,
        is_linear: bool = True,
        prob: float = 0.05,
        min_val: float = MIN_REWARD,
        max_val: float = MAX_REWARD,
        window_size: int | None = None,
        rng: random.Random | None = None,
        is_sa: bool = False,  # single-agent (joint) flag
    ):
        self.name = name
        self.n_actions = n_actions

        self.marl_controller = marl_controller

        self.strategy = strategy

        self.weights_r = weights_r or {}

        self.is_linear = is_linear
        self.actions = actions
        self.prob = prob  # to avoid strong correlation across agents

        self.counts = np.zeros(n_actions)
        self.rewards = np.zeros(n_actions)

        self.t = 1

        self.window_size = window_size if window_size is not None else n_actions
        if self.strategy == "sw_osub":
            self.history = deque(
                maxlen=self.window_size + 1
            )  # store (action, reward) tuples

        # Normalization
        self.min_val = min_val
        self.max_val = max_val

        self.rng = rng

        # ---- single-agent joint action support ----
        self.is_sa = is_sa
        if self.is_sa:
            self._precomputed_neighbors = [
                self.get_neighbors_sa(i) for i in range(self.n_actions)
            ]

    def get_linear_neighbors(self, k):
        # assuming a linear action space
        if k == 0:
            return [0, 1]
        elif k == self.n_actions - 1:
            return [self.n_actions - 2, self.n_actions - 1]
        else:
            return [k - 1, k, k + 1]

    def get_action_neighbors(self, k):
        # considering each action sharing a subset of actions as neighbors
        target_actions = self.actions[k]
        neighbors = [k]

        for other_k, other_actions in self.actions.items():
            if other_k != k and target_actions & other_actions:
                neighbors.append(other_k)

        return neighbors

    def get_neighbors_sa(self, k):
        """
        Single-agent joint-action neighbor generation.
        - neighbors share a channel region
        - and have primary, cw within ±1 of leader
        """
        leader = self.actions[k]  # (c_id, p_idx, cw_id)
        c1, p1, cw1 = leader

        neighbors = set()
        neighbors.add(k)

        c1_set = set(CHANNEL_MAP.get(c1, set()))

        for j_idx, (c2, p2, cw2) in enumerate(self.actions):
            if j_idx == k:
                continue

            # must share a channel region
            c2_set = set(CHANNEL_MAP.get(c2, set()))
            if not (c1_set & c2_set):
                continue

            # primary within ±1 and valid for that channel
            if abs(p1 - p2) > 1:
                continue
            if (p2 + 1) not in CHANNEL_MAP.get(
                c2, set()
            ):  # this is not necessary but it's a sanity check
                continue

            # CW within ±1
            if abs(cw1 - cw2) > 1:
                continue
            neighbors.add(j_idx)

        return sorted(neighbors)

    def get_neighbors(self, k):
        if self.is_sa:
            return self._precomputed_neighbors[k]
        if self.is_linear:
            return self.get_linear_neighbors(k)
        else:
            return self.get_action_neighbors(k)

    def kl_ucb_index(self, m, n, t):
        if n == 0:
            return 1.0
        if m == 1.0:
            return 1.0
        if m == 0.0:
            return 1.0 - (t ** (-1 / n))

        epsilon = 1e-8
        max_iter = 1000
        threshold = 1e-5

        q = m
        found = False

        for _ in range(max_iter):
            kl = n * (m * np.log(m / q) + (1 - m) * np.log((1 - m) / (1 - q)))
            if kl >= np.log(t):
                found = True
                break
            q = (q + 1.0) / 2.0
            q = min(max(q, epsilon), 1 - epsilon)

        if not found:
            return 1.0  # fallback

        # Newton's method
        for _ in range(max_iter):
            q = min(max(q, epsilon), 1 - epsilon)
            f = n * (m * np.log(m / q) + (1 - m) * np.log((1 - m) / (1 - q))) - np.log(
                t
            )
            fprime = n * (-m / q + (1 - m) / (1 - q))

            if fprime == 0:
                break

            step = f / fprime
            if abs(step) < threshold:
                break
            q -= step
        return q

    def _sliding_window_stats(self):
        counts = np.zeros(self.n_actions)
        rewards = np.zeros(self.n_actions)

        for a, r in self.history:
            counts[a] += 1
            rewards[a] += r

        mu_hat = rewards / np.maximum(counts, 1)

        return mu_hat, counts

    def select_action(self, valid_actions=None):
        if self.strategy in "osub" or "sw_osub":
            return self._select_action(valid_actions)
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

    def _select_action(self, valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))

        if self.strategy == "sw_osub":
            mu_hat, counts = self._sliding_window_stats()
        else:
            mu_hat = self.rewards / np.maximum(self.counts, 1)
            counts = self.counts

        # Mask invalid actions for leader selection
        masked_means = np.full(self.n_actions, -np.inf)
        masked_means[valid_actions] = mu_hat[valid_actions]
        leader = np.argmax(masked_means)

        # neighbors of leader that are valid
        N = self.get_neighbors(leader)
        N = [a for a in N if a in valid_actions]

        if self.rng.random() < self.prob:
            return self.rng.choice(N)

        indices = [self.kl_ucb_index(mu_hat[k], counts[k], self.t) for k in N]

        max_idx = np.argmax(indices)
        action = N[max_idx]

        return action

    def _normalize_reward(self, reward):
        clipped_reward = max(min(reward, self.max_val), self.min_val)
        return (clipped_reward - self.min_val) / (self.max_val - self.min_val)

    def update(self, action, reward):
        norm_reward = self._normalize_reward(reward)

        if self.strategy == "sw_osub":
            self.history.append((action, norm_reward))
        else:
            self.counts[action] += 1
            self.rewards[action] += norm_reward
        self.t += 1

    def reset(self):
        self.counts = np.zeros(self.n_actions)
        self.rewards = np.zeros(self.n_actions)
        self.t = 1
        if self.strategy == "sw_osub":
            self.history.clear()


class MARLController:  # only for multi agent
    def __init__(
        self,
        sparams: sparams_module,
        cfg: cfg_module,
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
        elif strategy in ["ucb"]:
            agent_class = UCB
        elif strategy in ["osub", "sw_osub"]:
            agent_class = OSUB
        else:
            raise ValueError(f"Unknown strategy {strategy}")

        channel_params = {
            "name": "channel_agent",
            "n_actions": len(
                CHANNEL_MAP
            ),  # 0: {1}, 1: {2}, 2: {3}, 3: {4}, 4: {1, 2}, 5: {3, 4}, 6: {1, 2, 3, 4}
            "context_dim": 9
            + (
                1 if settings.get("enable_meta_agent", False) else 0
            ),  # 4x channel occupation ratio + 4x channel busy flags + 1x queue size + (1x holdtime if meta-controller enabled)
            "strategy": strategy,
            "weights_r": settings.get("channel_weights", {}),
        }

        primary_params = {
            "name": "primary_agent",
            "n_actions": len(
                PRIMARY_CHANNEL_MAP
            ),  # 0: {1}, 1: {2}, 2: {3}, 3: {4} (depending on channel)
            "context_dim": 12
            + (
                1 if settings.get("enable_meta_agent", False) else 0
            ),  # 4x current selected channels (one hot encoded) + 4x channel occupation ratio + 4x channel busy flags + (1x holdtime if meta-controller enabled)
            "strategy": strategy,
            "weights_r": settings.get("primary_weights", {}),
        }

        cw_params = {
            "name": "cw_agent",
            "n_actions": len(
                CW_MAP
            ),  # 0: {16}, 1: {32}, 2: {64}, 3: {128}, 4: {256}, 5: {512}, 6: {1024} (i.e., 2**(x+4))
            "context_dim": 17
            + (
                1 if settings.get("enable_meta_agent", False) else 0
            ),  #  4x current selected channels (one hot encoded) + 4x current selected primary (one hot encoded) + 4x channel occupation ratio + 4x channel busy flags + 1x queue size + (1x holdtime if meta-controller enabled)
            "strategy": strategy,
            "weights_r": settings.get("cw_weights", {}),
        }

        meta_params = {
            "name": "meta_agent",
            "n_actions": (
                len(META_MAP)
                if not settings.get("enable_meta_agent_multifreq", False)
                else len(META_MAP_multifreq)
            ),  # 0: {1}, 1: {2}, 2: {4}, 3: {8} (i.e., 2**(x))
            "context_dim": 17,  #  4x current selected channels (one hot encoded) + 4x current selected primary (one hot encoded) + 1x current cw (mapped idx) + 4x channel occupation ratio + 4x channel busy flags
            "strategy": strategy,
        }

        if agent_class == EpsRMSProp:
            for param in [channel_params, primary_params, cw_params, meta_params]:
                param.update(
                    {
                        "epsilon": settings.get("epsilon", 0.1),
                        "decay_rate": settings.get("decay_rate", 0.99),
                        "eta": settings.get("eta", 0.1),
                        "gamma": settings.get("gamma", 0.9),
                        "alpha_ema": settings.get("alpha_ema", 0.1),
                        "min_val": settings.get("min_val", MIN_REWARD),
                        "max_val": settings.get("max_val", MAX_REWARD),
                    }
                )
        elif agent_class == SWLinUCB:
            for param in [channel_params, primary_params, cw_params, meta_params]:
                param.update(
                    {
                        "alpha": settings.get("alpha", 1.0),
                        "min_val": settings.get("min_val", MIN_REWARD),
                        "max_val": settings.get("max_val", MAX_REWARD),
                        "window_size": settings.get("window_size", None),
                    }
                )
        elif agent_class == UCB:
            for param in [channel_params, primary_params, cw_params, meta_params]:
                param.update(
                    {
                        "alpha": settings.get("alpha", 4.0),
                        "min_val": settings.get("min_val", MIN_REWARD),
                        "max_val": settings.get("max_val", MAX_REWARD),
                    }
                )
                param.pop("context_dim", None)
                param.pop("strategy", None)
        elif agent_class == OSUB:
            for param in [channel_params, primary_params, cw_params, meta_params]:
                param.update(
                    {
                        "min_val": settings.get("min_val", MIN_REWARD),
                        "max_val": settings.get("max_val", MAX_REWARD),
                        "window_size": settings.get("window_size", None),
                    }
                )
                if param["name"] == "channel_agent":
                    param.update({"is_linear": False, "actions": CHANNEL_MAP})
                param.pop("context_dim", None)

        self.channel_agent = agent_class(
            **channel_params, marl_controller=self, rng=env.rng
        )
        self.primary_agent = agent_class(
            **primary_params, marl_controller=self, rng=env.rng
        )
        self.cw_agent = agent_class(**cw_params, marl_controller=self, rng=env.rng)

        self.meta_agent = (
            agent_class(**meta_params, marl_controller=self, rng=env.rng)
            if settings.get("enable_meta_agent", False)
            else None
        )

        self.last_channel_action = None
        self.last_primary_action = None
        self.last_cw_action = None
        self.last_meta_action = None

        self.last_channel_context = None
        self.last_primary_context = None
        self.last_cw_context = None
        self.last_meta_context = None

        self.channel_emissions_tracker = None
        self.primary_emissions_tracker = None
        self.cw_emissions_tracker = None
        self.meta_emissions_tracker = None

        if cfg.USE_CODECARBON:
            self.channel_emissions_tracker = EmissionsTracker(
                project_name="channel_agent"
            )
            self.primary_emissions_tracker = EmissionsTracker(
                project_name="primary_agent"
            )
            self.cw_emissions_tracker = EmissionsTracker(project_name="cw_agent")
            self.meta_emissions_tracker = (
                EmissionsTracker(project_name="meta_agent") if self.meta_agent else None
            )

        self.results = []

    def decide_channel(self, context):
        if self.channel_emissions_tracker:
            self.channel_emissions_tracker.start()

        self.last_channel_context = context

        if isinstance(self.channel_agent, UCB) or isinstance(self.channel_agent, OSUB):
            action = self.channel_agent.select_action()
        else:
            action = self.channel_agent.select_action(context)

        self.last_channel_action = action
        self.logger.debug(
            f"{self.node.type} {self.node.id} -> Channel action: {action}"
        )

        if self.channel_emissions_tracker:
            self.channel_emissions_tracker.stop()

        return action

    def decide_primary(self, context, allocated_channels):
        if self.primary_emissions_tracker:
            self.primary_emissions_tracker.start()

        self.last_primary_context = context
        valid_actions = [c - 1 for c in allocated_channels]

        if isinstance(self.primary_agent, UCB) or isinstance(self.primary_agent, OSUB):
            action = self.primary_agent.select_action(valid_actions)
        else:
            action = self.primary_agent.select_action(context, valid_actions)

        self.last_primary_action = action
        self.logger.debug(
            f"{self.node.type} {self.node.id} -> Primary action: {action}"
        )

        if self.primary_emissions_tracker:
            self.primary_emissions_tracker.stop()

        return action

    def decide_cw(self, context):
        if self.cw_emissions_tracker:
            self.cw_emissions_tracker.start()

        self.last_cw_context = context

        if isinstance(self.cw_agent, UCB) or isinstance(self.cw_agent, OSUB):
            action = self.cw_agent.select_action()
        else:
            action = self.cw_agent.select_action(context)

        self.last_cw_action = action
        self.logger.debug(f"{self.node.type} {self.node.id} -> CW action: {action}")

        if self.cw_emissions_tracker:
            self.cw_emissions_tracker.stop()

        return action

    def decide_meta(self, context):
        if self.meta_emissions_tracker:
            self.meta_emissions_tracker.start()

        self.last_meta_context = context

        if isinstance(self.meta_agent, UCB) or isinstance(self.meta_agent, OSUB):
            action = self.meta_agent.select_action()
        else:
            action = self.meta_agent.select_action(context)

        self.last_meta_action = action
        self.logger.debug(f"{self.node.type} {self.node.id} -> Meta action: {action}")

        if self.meta_emissions_tracker:
            self.meta_emissions_tracker.stop()

        return action

    def _compute_weighted_reward(self, delay_components: dict, weights: dict):
        return -sum(weights.get(k, 0) * delay_components[k] for k in weights)

    def update_channel_agent(self, delay_components: dict = None):
        if delay_components is None:
            reward = MIN_REWARD
        elif self.cfg.ENABLE_REWARD_DECOMPOSITION:
            reward = self._compute_weighted_reward(
                delay_components, self.channel_agent.weights_r
            )
        else:
            reward = -sum(
                delay_components.values()
            )  # (minimize delay) sum of component means equals mean of per-sample sums since each component has the same number of samples

        if self.channel_emissions_tracker:
            self.channel_emissions_tracker.start()
        if isinstance(self.channel_agent, UCB) or isinstance(self.channel_agent, OSUB):
            self.channel_agent.update(self.last_channel_action, reward)
        else:
            self.channel_agent.update(
                self.last_channel_context, self.last_channel_action, reward
            )  # update
        if self.channel_emissions_tracker:
            self.channel_emissions_tracker.stop()

        self._log_agent_data("channel", reward)

    def update_primary_agent(self, delay_components: dict = None):
        if delay_components is None:
            reward = MIN_REWARD
        elif self.cfg.ENABLE_REWARD_DECOMPOSITION:
            reward = self._compute_weighted_reward(
                delay_components, self.primary_agent.weights_r
            )
        else:
            reward = -sum(delay_components.values())

        if self.primary_emissions_tracker:
            self.primary_emissions_tracker.start()
        if isinstance(self.primary_agent, UCB) or isinstance(self.primary_agent, OSUB):
            self.primary_agent.update(self.last_primary_action, reward)
        else:
            self.primary_agent.update(
                self.last_primary_context, self.last_primary_action, reward
            )  # update
        if self.primary_emissions_tracker:
            self.primary_emissions_tracker.stop()

        self._log_agent_data("primary", reward)

    def update_cw_agent(self, delay_components: dict = None):
        if delay_components is None:
            reward = MIN_REWARD

        elif self.cfg.ENABLE_REWARD_DECOMPOSITION:
            reward = self._compute_weighted_reward(
                delay_components, self.cw_agent.weights_r
            )
        else:
            reward = -sum(delay_components.values())

        if self.cw_emissions_tracker:
            self.cw_emissions_tracker.start()
        if isinstance(self.cw_agent, UCB) or isinstance(self.cw_agent, OSUB):
            self.cw_agent.update(self.last_cw_action, reward)
        else:
            self.cw_agent.update(
                self.last_cw_context, self.last_cw_action, reward
            )  # update
        if self.cw_emissions_tracker:
            self.cw_emissions_tracker.stop()

        self._log_agent_data("cw", reward)

    def update_meta_agent(self, delay_components: dict = None):
        if not self.meta_agent:
            return
        if delay_components is None:
            reward = MIN_REWARD
        elif self.cfg.ENABLE_REWARD_DECOMPOSITION:
            reward = self._compute_weighted_reward(
                delay_components, self.meta_agent.weights_r
            )
        else:
            reward = -sum(delay_components.values())

        if self.meta_emissions_tracker:
            self.meta_emissions_tracker.start()
        if isinstance(self.meta_agent, UCB) or isinstance(self.meta_agent, OSUB):
            self.meta_agent.update(self.last_meta_action, reward)
        else:
            self.meta_agent.update(
                self.last_meta_context, self.last_meta_action, reward
            )  # update
        if self.meta_emissions_tracker:
            self.meta_emissions_tracker.stop()

        self._log_agent_data("meta", reward)

    def update_tx_duration(self, delay_components: dict):
        self.results.append(sum(delay_components.values()))

    def log_eps_weight_matrix(self):
        if wandb.run:
            wandb.run.summary["channel_weight_matrix"] = (
                self.channel_agent.weight_matrix.tolist()
            )
            wandb.run.summary["primary_weight_matrix"] = (
                self.primary_agent.weight_matrix.tolist()
            )
            wandb.run.summary["cw_weight_matrix"] = self.cw_agent.weight_matrix.tolist()
            if self.meta_agent:
                wandb.run.summary["meta_weight_matrix"] = (
                    self.meta_agent.weight_matrix.tolist()
                )

    def log_emissions_data(self):
        if wandb.run:
            wandb.run.summary["channel_emissions"] = (
                self.channel_emissions_tracker.final_emissions_data.__dict__
            )
            wandb.run.summary["primary_emissions"] = (
                self.primary_emissions_tracker.final_emissions_data.__dict__
            )
            wandb.run.summary["cw_emissions"] = (
                self.cw_emissions_tracker.final_emissions_data.__dict__
            )
            if self.meta_emissions_tracker:
                wandb.run.summary["meta_emissions"] = (
                    self.meta_emissions_tracker.final_emissions_data.__dict__
                )

    def get_emissions_data(self):
        emissions_data = {
            "channel": self.channel_emissions_tracker.final_emissions_data.__dict__,
            "primary": self.primary_emissions_tracker.final_emissions_data.__dict__,
            "cw": self.cw_emissions_tracker.final_emissions_data.__dict__,
        }

        if self.meta_emissions_tracker:
            emissions_data["meta"] = (
                self.meta_emissions_tracker.final_emissions_data.__dict__
            )
        return emissions_data

    def _log_agent_data(self, agent: str, reward: float):
        if wandb.run:
            wandb.log(
                {
                    f"node_{self.node.id}/action/{agent}": getattr(
                        self, f"last_{agent}_action"
                    ),
                    f"node_{self.node.id}/reward/{agent}": reward,
                    "env_time_us": self.env.now,
                }
            )


class SARLController:
    def __init__(
        self,
        sparams: sparams_module,
        cfg: cfg_module,
        env: simpy.Environment,
        node: AP,
        settings: dict,
    ):

        def action_to_onehot_vector(
            channel_id, primary_index, cw_index, n_channels=4, n_primaries=4
        ):
            channel_bits = [0] * n_channels
            for ch in CHANNEL_MAP[channel_id]:
                channel_bits[ch - 1] = 1

            primary_bits = [0] * n_primaries
            primary_bits[primary_index] = 1

            cw_min = min(CW_MAP.values())
            cw_max = max(CW_MAP.values())

            normalized_cw = (CW_MAP[cw_index] - cw_min) / (cw_max - cw_min)

            vec = channel_bits + primary_bits + [normalized_cw]
            return vec

        self.cfg = cfg
        self.sparams = sparams
        self.env = env

        self.name = "SARL"

        self.logger = get_logger(
            self.name,
            cfg,
            sparams,
            env,
            True if node.id in self.cfg.EXCLUDED_IDS else False,
        )

        self.node = node

        self.settings = settings

        strategy = settings.get("strategy", "sw_linucb")

        if strategy in ["sw_linucb", "linucb"]:
            agent_class = SWLinUCB
        elif strategy in ["epsilon_greedy", "decay_epsilon_greedy"]:
            agent_class = EpsRMSProp
        elif strategy in ["ucb"]:
            agent_class = UCB
        elif strategy in ["e2tc"]:
            agent_class = E2TC
        elif strategy in ["osub", "sw_osub"]:
            agent_class = OSUB
        else:
            raise ValueError(f"Unknown strategy {strategy}")

        valid_actions = []
        valid_actions_onehot = (
            []
        )  # [channel1, channel2, channel3, channel4, primary1, primary2, primary3, primary4, cw]; d=9
        for c_id, pset in CHANNEL_MAP.items():
            for p in pset:
                for cw_id, cw_value in CW_MAP.items():
                    valid_actions.append(
                        (c_id, p - 1, cw_id)
                    )  # p-1 because primaries are 0-indexed
                    valid_actions_onehot.append(
                        action_to_onehot_vector(c_id, p - 1, cw_id)
                    )

        self.valid_joint_actions = valid_actions  # (channel, primary, cw)
        self.n_actions = len(self.valid_joint_actions)

        agent_params = {
            "name": "joint_agent",
            "n_actions": self.n_actions,
            "context_dim": 9,  # 4x channel occupation ratio + 4x channel busy flags + 1x queue size
            "strategy": strategy,
        }

        if agent_class == EpsRMSProp:
            agent_params.update(
                {
                    "epsilon": settings.get("epsilon", 0.1),
                    "decay_rate": settings.get("decay_rate", 0.99),
                    "eta": settings.get("eta", 0.1),
                    "gamma": settings.get("gamma", 0.9),
                    "alpha_ema": settings.get("alpha_ema", 0.1),
                    "min_val": settings.get("min_val", MIN_REWARD),
                    "max_val": settings.get("max_val", MAX_REWARD),
                }
            )
        elif agent_class == SWLinUCB:
            agent_params.update(
                {
                    "alpha": settings.get("alpha", 1.0),
                    "min_val": settings.get("min_val", MIN_REWARD),
                    "max_val": settings.get("max_val", MAX_REWARD),
                    "window_size": settings.get("window_size", None),
                }
            )
        elif agent_class == UCB:
            agent_params.update(
                {
                    "alpha": settings.get("alpha", 4.0),
                    "min_val": settings.get("min_val", MIN_REWARD),
                    "max_val": settings.get("max_val", MAX_REWARD),
                }
            )
            agent_params.pop("context_dim", None)
            agent_params.pop("strategy", None)
        elif agent_class == E2TC:
            agent_params = {
                "name": "joint_agent",
                "actions": valid_actions_onehot,  # one-hot vectors
                "T": settings.get("T", 2 * 10**4),
                "alpha": settings.get("alpha", 1.0),
                "min_val": settings.get("min_val", MIN_REWARD),
                "max_val": settings.get("max_val", MAX_REWARD),
            }
        elif agent_class == OSUB:
            agent_params.update(
                {
                    "name": "joint_agent",
                    "min_val": settings.get("min_val", MIN_REWARD),
                    "max_val": settings.get("max_val", MAX_REWARD),
                    "window_size": settings.get("window_size", None),
                    "is_linear": False,  # joint actions are not linear overall
                    "actions": self.valid_joint_actions,
                    "is_sa": True,
                }
            )
            agent_params.pop("context_dim", None)
            agent_params.pop("strategy", None)

        self.joint_agent = agent_class(
            **agent_params, marl_controller=self, rng=env.rng
        )

        self.last_context = None

        self.last_action_idx = None
        self.last_action_tuple = None

        self.emissions_tracker = (
            EmissionsTracker(project_name="joint_agent") if cfg.USE_CODECARBON else None
        )

        self.results = []

    def decide_joint_action(self, context):
        if self.emissions_tracker:
            self.emissions_tracker.start()

        if isinstance(self.joint_agent, E2TC) or isinstance(self.joint_agent, UCB) or isinstance(self.joint_agent, OSUB):
            action_idx = self.joint_agent.select_action()
        else:
            self.last_context = context
            action_idx = self.joint_agent.select_action(context)

        self.last_action_idx = action_idx
        self.last_action_tuple = self.valid_joint_actions[action_idx]

        self.logger.debug(
            f"{self.node.type} {self.node.id} -> Joint action: {self.last_action_tuple}"
        )

        if self.emissions_tracker:
            self.emissions_tracker.stop()

        return self.last_action_tuple

    def update_single_agent(self, delay_components: dict = None):
        if delay_components is None:
            reward = MIN_REWARD
        else:
            reward = -sum(delay_components.values())

        if self.emissions_tracker:
            self.emissions_tracker.start()

        if isinstance(self.joint_agent, E2TC):
            self.joint_agent.update(reward)
        elif isinstance(self.joint_agent, UCB) or isinstance(self.joint_agent, OSUB):
            self.joint_agent.update(self.last_action_idx, reward)
        else:
            self.joint_agent.update(self.last_context, self.last_action_idx, reward)

        if self.emissions_tracker:
            self.emissions_tracker.stop()

        self._log_agent_data(reward)

    def update_tx_duration(self, delay_components: dict):
        self.results.append(sum(delay_components.values()))

    def log_eps_weight_matrix(self):
        if wandb.run:
            wandb.run.summary["joint_weight_matrix"] = (
                self.joint_agent.weight_matrix.tolist()
            )

    def log_emissions_data(self):
        if wandb.run:
            wandb.run.summary["joint_emissions"] = (
                self.emissions_tracker.final_emissions_data.__dict__
            )

    def get_emissions_data(self):
        if not self.emissions_tracker:
            return {}
        return {"joint": self.emissions_tracker.final_emissions_data.__dict__}

    def _log_agent_data(self, reward: float):
        if wandb.run:
            wandb.log(
                {
                    f"node_{self.node.id}/action/channel": self.last_action_tuple[0],
                    f"node_{self.node.id}/action/primary": self.last_action_tuple[1],
                    f"node_{self.node.id}/action/cw": self.last_action_tuple[2],
                    f"node_{self.node.id}/reward": reward,
                    "env_time_us": self.env.now,
                }
            )
