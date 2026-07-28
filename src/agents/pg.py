import numpy as np
import torch
from torch.distributions import Categorical
from torch.optim import Adam

from src.utils import Discretizer

class ActorCritic(torch.nn.Module):
    def __init__(self, state_dim, n_actions, hidden_layers):
        super().__init__()
        layers = []
        in_features = state_dim + 1
        for width in hidden_layers:
            layers.extend([torch.nn.Linear(in_features, width), torch.nn.Tanh()])
            in_features = width
        self.body = torch.nn.Sequential(*layers)
        self.policy = torch.nn.Linear(in_features, n_actions)
        self.value = torch.nn.Linear(in_features, 1)
        torch.nn.init.orthogonal_(self.policy.weight, gain=0.01)
        torch.nn.init.zeros_(self.policy.bias)
        torch.nn.init.orthogonal_(self.value.weight, gain=1.0)
        torch.nn.init.zeros_(self.value.bias)

    def forward(self, states, timesteps):
        x = torch.cat((states, timesteps.unsqueeze(-1)), dim=-1)
        features = self.body(x)
        return self.policy(features), self.value(features).squeeze(-1)



class PG:
    """Finite-horizon vanilla policy gradient with a learned value baseline."""

    def __init__(self, discretizer: Discretizer, H: int, alpha: float = 3e-4,
                 value_alpha: float = 1e-3, gamma: float = 1.0,
                 entropy_coef: float = 0.01, rollout_episodes: int = 32,
                 value_epochs: int = 10, minibatch_size: int = 64,
                 hidden_layers=(64, 64)):
        self.discretizer = discretizer
        self.H = H
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.rollout_steps = rollout_episodes * H
        self.value_epochs = value_epochs
        self.minibatch_size = minibatch_size
        n_actions = int(np.prod(discretizer.bucket_actions))
        self.network = ActorCritic(
            len(discretizer.bucket_states), n_actions, hidden_layers
        ).double()
        self.policy_parameters = (
            list(self.network.body.parameters()) + list(self.network.policy.parameters())
        )
        self.policy_opt = Adam(self.policy_parameters, lr=alpha)
        self.value_opt = Adam(self.network.value.parameters(), lr=value_alpha)
        self.rollout = []

    def _inputs(self, state, h):
        state_t = torch.as_tensor(state, dtype=torch.double)
        h_t = torch.as_tensor(h / max(self.H - 1, 1), dtype=torch.double)
        return state_t, h_t

    def select_action(self, state, h):
        state_t, h_t = self._inputs(state, h)
        with torch.no_grad():
            logits, value = self.network(state_t.unsqueeze(0), h_t.unsqueeze(0))
            distribution = Categorical(logits=logits.squeeze(0))
            action_idx_flat = distribution.sample()
        action_idx = np.unravel_index(
            action_idx_flat.item(), self.discretizer.bucket_actions
        )
        action = self.discretizer.get_action_from_index(action_idx)
        return action, action_idx_flat.item(), value.item()

    def select_greedy_action(self, state, h):
        state_t, h_t = self._inputs(state, h)
        with torch.no_grad():
            logits, _ = self.network(state_t.unsqueeze(0), h_t.unsqueeze(0))
            action_idx_flat = logits.argmax(dim=-1).item()
        action_idx = np.unravel_index(action_idx_flat, self.discretizer.bucket_actions)
        return self.discretizer.get_action_from_index(action_idx)

    def store(self, h, state, action_idx, reward, done, value):
        self.rollout.append((h, np.asarray(state), action_idx, reward, done, value))

    def maybe_update(self, force=False):
        if len(self.rollout) < self.rollout_steps and not force:
            return
        if not self.rollout:
            return
        timesteps, states, actions, rewards, dones, values = zip(*self.rollout)
        returns = np.zeros(len(rewards), dtype=np.float64)
        future_return = 0.0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                future_return = 0.0
            future_return = rewards[t] + self.gamma * future_return
            returns[t] = future_return
        advantages = returns - np.asarray(values, dtype=np.float64)

        states_t = torch.as_tensor(np.asarray(states), dtype=torch.double)
        timesteps_t = torch.as_tensor(
            np.asarray(timesteps) / max(self.H - 1, 1), dtype=torch.double
        )
        actions_t = torch.as_tensor(actions, dtype=torch.long)
        returns_t = torch.as_tensor(returns, dtype=torch.double)
        advantages_t = torch.as_tensor(advantages, dtype=torch.double)
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std(unbiased=False) + 1e-8
        )

        logits, _ = self.network(states_t, timesteps_t)
        distribution = Categorical(logits=logits)
        policy_loss = -(
            distribution.log_prob(actions_t) * advantages_t
        ).mean() - self.entropy_coef * distribution.entropy().mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_parameters, 0.5)
        self.policy_opt.step()

        batch_size = len(self.rollout)
        for _ in range(self.value_epochs):
            for indices in torch.randperm(batch_size).split(self.minibatch_size):
                with torch.no_grad():
                    features = self.network.body(torch.cat(
                        (states_t[indices], timesteps_t[indices].unsqueeze(-1)), dim=-1
                    ))
                predicted_values = self.network.value(features).squeeze(-1)
                value_loss = 0.5 * (
                    predicted_values - returns_t[indices]
                ).square().mean()
                self.value_opt.zero_grad()
                value_loss.backward()
                self.value_opt.step()
        self.rollout.clear()

    @property
    def num_parameters(self):
        return sum(parameter.numel() for parameter in self.network.parameters())
