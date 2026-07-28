import numpy as np

from src.utils import Discretizer, ReplayBuffer


class FHLSVIUCB:
    """Finite-horizon LSVI-UCB with action-block RBF features."""

    def __init__(self, discretizer: Discretizer, H: int,
                 num_rbf_features: int = 5, regularization: float = 1.0,
                 bonus_coef: float = 1.0, buffer_size: int = 1,
                 feature_seed: int = 0, normalize_states: bool = False,
                 action_candidates: int | None = None) -> None:
        self.discretizer = discretizer
        self.H = H
        self.num_rbf_features = num_rbf_features
        self.block_dim = num_rbf_features + 1
        self.n_actions = int(np.prod(discretizer.bucket_actions))
        self.feature_dim = self.n_actions * self.block_dim
        self.bonus_coef = bonus_coef
        self.normalize_states = normalize_states
        self.action_candidates = action_candidates
        self.buffer = ReplayBuffer(buffer_size)
        rng = np.random.default_rng(feature_seed)
        self.centers = rng.normal(
            size=(H, num_rbf_features, len(discretizer.bucket_states))
        )
        self.actions = self._enumerate_actions()
        identity = np.eye(self.block_dim) / regularization
        self.covariance_inv = np.broadcast_to(
            identity, (H, self.n_actions, self.block_dim, self.block_dim)
        ).copy()
        self.target_sum = np.zeros((H, self.n_actions, self.block_dim))
        self.theta = np.zeros((H, self.n_actions, self.block_dim))

    
    @property
    def n_parameters(self):
        return self.H * self.feature_dim

    def _enumerate_actions(self):
        indices = np.indices(tuple(self.discretizer.bucket_actions)).reshape(
            len(self.discretizer.bucket_actions), -1
        ).T
        return np.asarray([
            self.discretizer.get_action_from_index(index) for index in indices
        ], dtype=float)

    def _basis(self, state, h):
        state = np.asarray(state, dtype=float)
        if self.normalize_states:
            scale = np.where(self.discretizer.range_states == 0, 1, self.discretizer.range_states)
            state = 2 * (state - self.discretizer.min_points_states) / scale - 1
        squared_distance = np.sum((self.centers[h] - state) ** 2, axis=1)
        return np.concatenate((np.exp(-squared_distance), np.ones(1)))

    def _action_index(self, action):
        multi_index = self.discretizer.get_action_index(np.asarray(action))
        return int(np.ravel_multi_index(multi_index, self.discretizer.bucket_actions))

    def _candidate_indices(self):
        if self.action_candidates is None or self.action_candidates >= self.n_actions:
            return np.arange(self.n_actions)
        return np.random.choice(
            self.n_actions, size=self.action_candidates, replace=False
        )

    def _values(self, state, h, optimistic, action_indices=None):
        basis = self._basis(state, h)
        theta = self.theta[h]
        covariance_inv = self.covariance_inv[h]
        if action_indices is not None:
            theta = theta[action_indices]
            covariance_inv = covariance_inv[action_indices]
        values = theta @ basis
        if optimistic:
            uncertainty = np.sqrt(np.maximum(
                np.einsum("i,aij,j->a", basis, covariance_inv, basis,
                          optimize=True),
                0.0,
            ))
            values = values + self.bonus_coef * uncertainty
        return values

    
    @staticmethod
    def _random_argmax(values):
        best = np.max(values)
        candidates = np.flatnonzero(np.isclose(values, best, rtol=1e-12, atol=1e-12))
        return int(np.random.choice(candidates))

    def select_action(self, state, h, epsilon=None):
        del epsilon
        candidates = self._candidate_indices()
        local_index = self._random_argmax(
            self._values(state, h, optimistic=True, action_indices=candidates)
        )
        index = int(candidates[local_index])
        return self.actions[index].copy()

    def select_greedy_action(self, state, h):
        index = self._random_argmax(self._values(state, h, optimistic=False))
        return self.actions[index].copy()

    def update(self):
        h, state, action, next_state, reward, done = self.buffer.sample()
        action_index = self._action_index(action)
        basis = self._basis(state, h)
        target = float(reward)
        if not done and h + 1 < self.H:
            candidates = self._candidate_indices()
            target += float(np.max(self._values(
                next_state, h + 1, optimistic=True, action_indices=candidates
            )))
        inverse = self.covariance_inv[h, action_index]
        inverse_basis = inverse @ basis
        denominator = 1.0 + basis @ inverse_basis
        self.covariance_inv[h, action_index] = inverse - np.outer(
            inverse_basis, inverse_basis
        ) / denominator
        self.target_sum[h, action_index] += basis * target
        self.theta[h, action_index] = (
            self.covariance_inv[h, action_index]
            @ self.target_sum[h, action_index]
        )
