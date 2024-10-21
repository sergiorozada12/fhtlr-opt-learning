from typing import List
import random
from dataclasses import dataclass
import numpy as np
import torch


class Discretizer:
    def __init__(
        self,
        min_points_states: List[float],
        max_points_states: List[float],
        bucket_states: List[int],
        min_points_actions: List[float],
        max_points_actions: List[float],
        bucket_actions: List[int],
    ) -> None:
        self.min_points_states = np.array(min_points_states)
        self.max_points_states = np.array(max_points_states)
        self.bucket_states = np.array(bucket_states)
        self.range_states = self.max_points_states - self.min_points_states

        self.min_points_actions = np.array(min_points_actions)
        self.max_points_actions = np.array(max_points_actions)
        self.bucket_actions = np.array(bucket_actions)
        self.spacing_actions = (self.max_points_actions - self.min_points_actions) / (
            self.bucket_actions - 1
        )

        self.range_actions = self.max_points_actions - self.min_points_actions

        self.n_states = np.round(self.bucket_states).astype(int)
        self.n_actions = np.round(self.bucket_actions).astype(int)
        self.dimensions = np.concatenate((self.n_states, self.n_actions))

    def get_state_index(self, state: np.ndarray) -> List[int]:
        state = np.clip(
            state, a_min=self.min_points_states, a_max=self.max_points_states
        )
        scaling = (state - self.min_points_states) / self.range_states
        state_idx = np.round(scaling * (self.bucket_states - 1)).astype(int)
        return state_idx.tolist()

    def get_action_index(self, action: np.ndarray) -> List[int]:
        action = np.clip(
            action, a_min=self.min_points_actions, a_max=self.max_points_actions
        )
        scaling = (action - self.min_points_actions) / self.range_actions
        action_idx = np.round(scaling * (self.bucket_actions - 1)).astype(int)
        return action_idx.tolist()

    def get_action_from_index(self, action_idx: np.ndarray) -> np.ndarray:
        return self.min_points_actions + action_idx * self.spacing_actions


@dataclass
class Transition:
    timestep: int
    state: torch.Tensor
    action: np.array
    next_state: torch.Tensor
    reward: float
    done: bool


# From OpenAI baselines
class ReplayBuffer(object):
    def __init__(self, size: int) -> None:
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def append(self, *args):
        data = Transition(*args)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self):
        sample = self._storage[random.randint(0, len(self._storage) - 1)]
        return (
            sample.timestep,
            sample.state,
            sample.action,
            sample.next_state,
            sample.reward,
            sample.done,
        )
