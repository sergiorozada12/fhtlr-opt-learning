import numpy as np

from torch import tensor, double
from torch import no_grad
from torch.nn import MSELoss
from torch.optim import Adamax
from torch.optim.lr_scheduler import StepLR

from src.models import PARAFAC
from src.utils import ReplayBuffer, Discretizer


class FHTlr:
    def __init__(
        self,
        discretizer: Discretizer,
        alpha: float,
        H: int,
        k: int,
        scale: float,
        lr_decayment: float = None,
        lr_decayment_step: int = None,
        w_decay: float = 0.0,
        buffer_size: int = 1,
    ) -> None:
        self.alpha = alpha
        self.H = H

        self.buffer = ReplayBuffer(buffer_size)
        self.discretizer = discretizer
        self.Q = PARAFAC(
            np.concatenate(
                [[H], discretizer.bucket_states, discretizer.bucket_actions]
            ),
            k=k,
            scale=scale,
            nA=len(discretizer.bucket_actions),
        ).double()
        self.opt = Adamax(self.Q.parameters(), lr=alpha, weight_decay=w_decay)

        if lr_decayment:
            self.scheduler = StepLR(
                self.opt, step_size=lr_decayment_step, gamma=lr_decayment
            )

    def select_random_action(self) -> np.ndarray:
        a_idx = tuple(np.random.randint(self.discretizer.bucket_actions).tolist())
        return self.discretizer.get_action_from_index(a_idx)

    def select_greedy_action(self, s: np.ndarray, h: int) -> np.ndarray:
        s_idx = np.concatenate([[h], self.discretizer.get_state_index(s)])
        a_idx_flat = self.Q(s_idx).argmax().detach().item()
        a_idx = np.unravel_index(a_idx_flat, self.discretizer.bucket_actions)
        return self.discretizer.get_action_from_index(a_idx)

    def select_action(self, s: np.ndarray, h: int, epsilon: float) -> np.ndarray:
        if np.random.rand() < epsilon:
            return self.select_random_action()
        return self.select_greedy_action(s, h)

    def update(self) -> None:
        h, s, a, sp, r, d = self.buffer.sample()

        s_idx = np.concatenate([[h], self.discretizer.get_state_index(s)])
        sp_idx = np.concatenate([[h + 1], self.discretizer.get_state_index(sp)])
        a_idx = self.discretizer.get_action_index(a)

        for factor in self.Q.factors:
            r = tensor(r, dtype=double)
            q_target = tensor(0.0, dtype=double)
            if not d:
                ap = self.select_greedy_action(sp, h + 1)
                ap_idx = self.discretizer.get_action_index(ap)
                q_target += self.Q(np.concatenate([sp_idx, ap_idx]))
            q_hat = self.Q(np.concatenate([s_idx, a_idx]))

            self.opt.zero_grad()
            loss = MSELoss()
            loss(r, q_hat - q_target).backward()

            with no_grad():
                for frozen_factor in self.Q.factors:
                    if frozen_factor is not factor:
                        frozen_factor.grad = None
            self.opt.step()


class FHMaxTlr:
    def __init__(
        self,
        discretizer: Discretizer,
        alpha: float,
        H: int,
        k: int,
        scale: float,
        lr_decayment: float = None,
        lr_decayment_step: int = None,
        w_decay: float = 0.0,
        buffer_size: int = 1,
    ) -> None:
        self.alpha = alpha
        self.H = H

        self.buffer = ReplayBuffer(buffer_size)
        self.discretizer = discretizer
        self.Q = PARAFAC(
            np.concatenate(
                [[H], discretizer.bucket_states, discretizer.bucket_actions]
            ),
            k=k,
            scale=scale,
            nA=len(discretizer.bucket_actions),
        ).double()
        self.opt = Adamax(self.Q.parameters(), lr=alpha, weight_decay=w_decay)

        if lr_decayment:
            self.scheduler = StepLR(
                self.opt, step_size=lr_decayment_step, gamma=lr_decayment
            )

    def select_random_action(self) -> np.ndarray:
        a_idx = tuple(np.random.randint(self.discretizer.bucket_actions).tolist())
        return self.discretizer.get_action_from_index(a_idx)

    def select_greedy_action(self, s: np.ndarray, h: int) -> np.ndarray:
        s_idx = np.concatenate([[h], self.discretizer.get_state_index(s)])
        a_idx_flat = self.Q(s_idx).argmax().detach().item()
        a_idx = np.unravel_index(a_idx_flat, self.discretizer.bucket_actions)
        return self.discretizer.get_action_from_index(a_idx)

    def select_action(self, s: np.ndarray, h: int, epsilon: float) -> np.ndarray:
        if np.random.rand() < epsilon:
            return self.select_random_action()
        return self.select_greedy_action(s, h)

    def update(self) -> None:
        h, s, a, sp, r, d = self.buffer.sample()

        s_idx = np.concatenate([[h], self.discretizer.get_state_index(s)])
        sp_idx = np.concatenate([[h + 1], self.discretizer.get_state_index(sp)])
        a_idx = self.discretizer.get_action_index(a)

        for factor in self.Q.factors:
            q_target = tensor(r, dtype=double)
            if not d:
                q_target += self.Q(sp_idx).max().detach()
            q_hat = self.Q(np.concatenate([s_idx, a_idx]))

            self.opt.zero_grad()
            loss = MSELoss()
            loss(q_hat, q_target).backward()

            with no_grad():
                for frozen_factor in self.Q.factors:
                    if frozen_factor is not factor:
                        frozen_factor.grad = None
            self.opt.step()