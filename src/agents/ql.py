import numpy as np
from src.utils import ReplayBuffer, Discretizer


class QLearning:
    def __init__(self, discretizer: Discretizer, alpha: float, gamma: float) -> None:
        self.alpha = alpha
        self.gamma = gamma

        self.buffer = ReplayBuffer(1)
        self.discretizer = discretizer
        self.Q = np.zeros(
            np.concatenate([discretizer.bucket_states, discretizer.bucket_actions])
        )

    def select_random_action(self) -> np.ndarray:
        a_idx = tuple(np.random.randint(self.discretizer.bucket_actions).tolist())
        return self.discretizer.get_action_from_index(a_idx)

    def select_greedy_action(self, s: np.ndarray, h: int) -> np.ndarray:
        s_idx = tuple(self.discretizer.get_state_index(s))
        q = self.Q[s_idx]
        a_idx = np.unravel_index(q.argmax(), q.shape)
        return self.discretizer.get_action_from_index(a_idx)

    def select_action(self, s: np.ndarray, h: int, epsilon: float) -> np.ndarray:
        if np.random.rand() < epsilon:
            return self.select_random_action()
        return self.select_greedy_action(s, h)

    def update(self) -> None:
        _, s, a, sp, r, _ = self.buffer.sample()

        s_idx = tuple(self.discretizer.get_state_index(s))
        sp_idx = tuple(self.discretizer.get_state_index(sp))
        a_idx = tuple(self.discretizer.get_action_index(a))

        q_target = r + self.gamma * self.Q[sp_idx].max()
        q_hat = self.Q[s_idx + a_idx]

        self.Q[s_idx + a_idx] += self.alpha * (q_target - q_hat)


class FHQLearning:
    def __init__(self, discretizer: Discretizer, alpha: float, H: int) -> None:
        self.alpha = alpha
        self.H = H

        self.buffer = ReplayBuffer(1)
        self.discretizer = discretizer
        self.Q = np.zeros(
            np.concatenate([[H], discretizer.bucket_states, discretizer.bucket_actions])
        )

    def select_random_action(self) -> np.ndarray:
        a_idx = tuple(np.random.randint(self.discretizer.bucket_actions).tolist())
        return self.discretizer.get_action_from_index(a_idx)

    def select_greedy_action(self, s: np.ndarray, h: int) -> np.ndarray:
        s_idx = self.discretizer.get_state_index(s)
        q = self.Q[tuple(np.concatenate([[h], s_idx]))]
        a_idx = np.unravel_index(q.argmax(), q.shape)
        return self.discretizer.get_action_from_index(a_idx)

    def select_action(self, s: np.ndarray, h: int, epsilon: float) -> np.ndarray:
        if np.random.rand() < epsilon:
            return self.select_random_action()
        return self.select_greedy_action(s, h)

    def update(self) -> None:
        h, s, a, sp, r, d = self.buffer.sample()

        s_idx = tuple(np.concatenate([[h], self.discretizer.get_state_index(s)]))
        sp_idx = tuple(np.concatenate([[h + 1], self.discretizer.get_state_index(sp)]))
        a_idx = self.discretizer.get_action_index(a)

        q_target = r
        if not d:
            q_target += self.Q[sp_idx].max()
        q_hat = self.Q[tuple(np.concatenate([s_idx, a_idx]))]

        self.Q[tuple(np.concatenate([s_idx, a_idx]))] += self.alpha * (q_target - q_hat)
