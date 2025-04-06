import numpy as np

from torch import tensor, double
from torch.nn import MSELoss
from torch.optim import Adamax

from src.models import RBFmodel,FHRBFmodel
from src.utils import ReplayBuffer, Discretizer


class RBF:
    def __init__(
        self, discretizer: Discretizer, alpha: float, gamma: float, buffer_size: int
    ) -> None:
        self.gamma = gamma

        self.buffer = ReplayBuffer(buffer_size)
        self.discretizer = discretizer
        self.Q = RBFmodel(
            len(discretizer.bucket_states),50, np.prod(discretizer.bucket_actions)
        ).double()
        self.opt = Adamax(self.Q.parameters(), lr=alpha)

    def select_random_action(self) -> np.ndarray:
        a_idx = tuple(np.random.randint(self.discretizer.bucket_actions).tolist())
        return self.discretizer.get_action_from_index(a_idx)

    def select_greedy_action(self, s: np.ndarray, h: int) -> np.ndarray:
        a_idx_flat = self.Q(s).argmax().detach().item()
        a_idx = np.unravel_index(a_idx_flat, self.discretizer.bucket_actions)
        return self.discretizer.get_action_from_index(a_idx)

    def select_action(self, s: np.ndarray, h: int, epsilon: float) -> np.ndarray:
        if np.random.rand() < epsilon:
            return self.select_random_action()
        return self.select_greedy_action(s, h)

    def update(self) -> None:
        _, s, a, sp, r, _ = self.buffer.sample()

        a_multi_idx = self.discretizer.get_action_index(a)
        a_idx = np.ravel_multi_index(a_multi_idx, self.discretizer.bucket_actions)

        q_target = r + self.gamma * self.Q(sp).max().detach()
        q_hat = self.Q(s)[a_idx]

        self.opt.zero_grad()
        loss = MSELoss()
        loss(q_hat, q_target).backward()
        self.opt.step()

class FHRBF:
    def __init__(
        self, discretizer: Discretizer, alpha: float, H: int, buffer_size: int
    ) -> None:
        self.alpha = alpha
        self.H = H

        self.buffer = ReplayBuffer(buffer_size)
        self.discretizer = discretizer
        self.Q = FHRBFmodel(
            len(discretizer.bucket_states),5, np.prod(discretizer.bucket_actions), H
        ).double()
        self.opt = Adamax(self.Q.parameters(), lr=alpha)

    def select_random_action(self) -> np.ndarray:
        a_idx = tuple(np.random.randint(self.discretizer.bucket_actions).tolist())
        return self.discretizer.get_action_from_index(a_idx)

    def select_greedy_action(self, s: np.ndarray, h: int) -> np.ndarray:
        a_idx_flat = self.Q(s, h).argmax().detach().item()
        a_idx = np.unravel_index(a_idx_flat, self.discretizer.bucket_actions)
        return self.discretizer.get_action_from_index(a_idx)

    def select_action(self, s: np.ndarray, h: int, epsilon: float) -> np.ndarray:
        if np.random.rand() < epsilon:
            return self.select_random_action()
        return self.select_greedy_action(s, h)

    def update(self) -> None:
        h, s, a, sp, r, d = self.buffer.sample()

        a_multi_idx = self.discretizer.get_action_index(a)
        a_idx = np.ravel_multi_index(a_multi_idx, self.discretizer.bucket_actions)

        q_target = tensor(r, dtype=double)
        if not d:
            q_target += self.Q(sp, h + 1).max().detach()
        q_hat = self.Q(s, h)[a_idx]

        self.opt.zero_grad()
        loss = MSELoss()
        loss(q_hat, q_target).backward()
        self.opt.step()
