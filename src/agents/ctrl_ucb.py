import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from src.utils import Discretizer


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super().__init__()
        layers = []
        for width in hidden_layers:
            layers.extend([torch.nn.Linear(input_dim, width), torch.nn.ReLU()])
            input_dim = width
        layers.append(torch.nn.Linear(input_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CTRLUCBM:
    """Finite-horizon practical CTRL-UCBM with categorical actions."""

    def __init__(self, discretizer: Discretizer, H: int, gamma: float = 1.0,
                 representation_dim: int = 32, hidden_layers=(64, 64),
                 representation_lr: float = 3e-4, actor_lr: float = 3e-4,
                 critic_lr: float = 1e-3, batch_size: int = 128,
                 buffer_size: int = 50_000, learning_starts: int = 256,
                 n_negatives: int = 16, temperature: float = 0.2,
                 bonus_coef: float = 1.0, covariance_reg: float = 1.0,
                 entropy_coef: float = 0.01, tau: float = 0.005,
                 covariance_refresh: int = 100,
                 covariance_samples: int = 4096,
                 exact_action_threshold: int = 128):
        self.discretizer = discretizer
        self.H = H
        self.gamma = gamma
        self.representation_dim = representation_dim
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.n_negatives = n_negatives
        self.temperature = temperature
        self.bonus_coef = bonus_coef
        self.covariance_reg = covariance_reg
        self.entropy_coef = entropy_coef
        self.tau = tau
        self.covariance_refresh = covariance_refresh
        self.covariance_samples = covariance_samples
        self.exact_action_threshold = exact_action_threshold
        self.n_actions = int(np.prod(discretizer.bucket_actions))
        self.state_dim = len(discretizer.bucket_states)
        self.action_dim = len(discretizer.bucket_actions)

        transition_input = self.state_dim + self.action_dim + 1
        next_state_input = self.state_dim + 1
        policy_input = self.state_dim + 1
        self.phi = MLP(transition_input, hidden_layers, representation_dim).double()
        self.mu = MLP(next_state_input, hidden_layers, representation_dim).double()
        self.actor = MLP(policy_input, hidden_layers, self.n_actions).double()
        self.critic = MLP(representation_dim, hidden_layers, 1).double()
        self.target_critic = MLP(representation_dim, hidden_layers, 1).double()
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.representation_opt = Adam(
            list(self.phi.parameters()) + list(self.mu.parameters()),
            lr=representation_lr,
        )
        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=critic_lr)
        self.buffer = deque(maxlen=buffer_size)
        self.update_steps = 0
        self.covariance_inv = torch.eye(representation_dim, dtype=torch.double) / covariance_reg

    def _time(self, h, batch_size=None):
        value = h / max(self.H - 1, 1)
        if batch_size is None:
            return torch.as_tensor([value], dtype=torch.double)
        return torch.full((batch_size, 1), value, dtype=torch.double)

    def _policy_input(self, states, timesteps):
        return torch.cat((states, timesteps.unsqueeze(-1)), dim=-1)

    def _action_values(self, flat_indices):
        indices = np.array([
            np.unravel_index(int(i), self.discretizer.bucket_actions)
            for i in flat_indices
        ])
        values = np.array([
            self.discretizer.get_action_from_index(index) for index in indices
        ])
        scale = np.where(self.discretizer.range_actions == 0, 1, self.discretizer.range_actions)
        return torch.as_tensor(
            2 * (values - self.discretizer.min_points_actions) / scale - 1,
            dtype=torch.double,
        )

    def _phi(self, states, action_values, timesteps):
        x = torch.cat((states, action_values, timesteps.unsqueeze(-1)), dim=-1)
        return F.normalize(self.phi(x), dim=-1)

    def _mu(self, next_states, next_timesteps):
        x = torch.cat((next_states, next_timesteps.unsqueeze(-1)), dim=-1)
        return F.normalize(self.mu(x), dim=-1)

    def select_action(self, state, h):
        state_t = torch.as_tensor(state, dtype=torch.double).unsqueeze(0)
        h_t = torch.as_tensor([h / max(self.H - 1, 1)], dtype=torch.double)
        with torch.no_grad():
            logits = self.actor(self._policy_input(state_t, h_t)).squeeze(0)
            index = Categorical(logits=logits).sample().item()
        multi_index = np.unravel_index(index, self.discretizer.bucket_actions)
        return self.discretizer.get_action_from_index(multi_index), index

    def select_greedy_action(self, state, h):
        state_t = torch.as_tensor(state, dtype=torch.double).unsqueeze(0)
        h_t = torch.as_tensor([h / max(self.H - 1, 1)], dtype=torch.double)
        with torch.no_grad():
            index = self.actor(self._policy_input(state_t, h_t)).argmax(-1).item()
        multi_index = np.unravel_index(index, self.discretizer.bucket_actions)
        return self.discretizer.get_action_from_index(multi_index)

    def store(self, h, state, action_idx, next_state, reward, done):
        self.buffer.append((h, np.asarray(state), action_idx,
                            np.asarray(next_state), reward, done))

    def _sample(self, count):
        batch = random.sample(self.buffer, count)
        h, state, action, next_state, reward, done = zip(*batch)
        return (
            torch.as_tensor(h, dtype=torch.double),
            torch.as_tensor(np.asarray(state), dtype=torch.double),
            torch.as_tensor(action, dtype=torch.long),
            torch.as_tensor(np.asarray(next_state), dtype=torch.double),
            torch.as_tensor(reward, dtype=torch.double),
            torch.as_tensor(done, dtype=torch.double),
        )

    def _all_action_features(self, states, timesteps):
        batch = states.shape[0]
        states_all = states[:, None, :].expand(-1, self.n_actions, -1).reshape(-1, self.state_dim)
        times_all = timesteps[:, None].expand(-1, self.n_actions).reshape(-1)
        action_values = self._action_values(np.tile(np.arange(self.n_actions), batch))
        features = self._phi(states_all, action_values, times_all)
        return features.reshape(batch, self.n_actions, self.representation_dim)

    def _refresh_covariance(self):
        count = min(len(self.buffer), self.covariance_samples)
        h, states, actions, _, _, _ = self._sample(count)
        action_values = self._action_values(actions.numpy())
        with torch.no_grad():
            features = self._phi(states, action_values, h / max(self.H - 1, 1))
            covariance = features.T @ features
            covariance += self.covariance_reg * torch.eye(
                self.representation_dim, dtype=torch.double
            )
            self.covariance_inv = torch.linalg.inv(covariance)

    def update(self):
        if len(self.buffer) < max(self.learning_starts, self.batch_size):
            return
        h, states, actions, next_states, rewards, dones = self._sample(self.batch_size)
        times = h / max(self.H - 1, 1)
        next_times = torch.clamp(h + 1, max=self.H - 1) / max(self.H - 1, 1)
        action_values = self._action_values(actions.numpy())

        features = self._phi(states, action_values, times)
        next_embeddings = self._mu(next_states, next_times)
        negative_indices = torch.randint(
            0, self.batch_size, (self.batch_size, self.n_negatives)
        )
        negative_embeddings = next_embeddings[negative_indices]
        positive_scores = (features * next_embeddings).sum(-1, keepdim=True)
        negative_scores = torch.einsum("bd,bkd->bk", features, negative_embeddings)
        logits = torch.cat((positive_scores, negative_scores), dim=1) / self.temperature
        nce_loss = F.cross_entropy(
            logits, torch.zeros(self.batch_size, dtype=torch.long)
        )
        self.representation_opt.zero_grad()
        nce_loss.backward()
        self.representation_opt.step()

        self.update_steps += 1
        if self.update_steps == 1 or self.update_steps % self.covariance_refresh == 0:
            self._refresh_covariance()

        with torch.no_grad():
            features = self._phi(states, action_values, times)
            uncertainty = torch.sqrt(torch.clamp(
                torch.einsum("bi,ij,bj->b", features, self.covariance_inv, features),
                min=0.0,
            ))
            bonus = torch.clamp(self.bonus_coef * uncertainty, max=2.0)
            next_logits = self.actor(self._policy_input(next_states, next_times))
            if self.n_actions <= self.exact_action_threshold:
                next_features = self._all_action_features(next_states, next_times)
                next_q = self.target_critic(next_features).squeeze(-1)
                next_log_probs = F.log_softmax(next_logits, dim=-1)
                next_probs = next_log_probs.exp()
                soft_next_value = (
                    next_probs * (next_q - self.entropy_coef * next_log_probs)
                ).sum(-1)
            else:
                next_distribution = Categorical(logits=next_logits)
                next_actions = next_distribution.sample()
                next_action_values = self._action_values(next_actions.numpy())
                next_features = self._phi(next_states, next_action_values, next_times)
                next_q = self.target_critic(next_features).squeeze(-1)
                soft_next_value = (
                    next_q - self.entropy_coef
                    * next_distribution.log_prob(next_actions)
                )
            target = rewards + bonus + self.gamma * (1 - dones) * soft_next_value

        predicted = self.critic(features.detach()).squeeze(-1)
        critic_loss = F.mse_loss(predicted, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_logits = self.actor(self._policy_input(states, times))
        if self.n_actions <= self.exact_action_threshold:
            actor_features = self._all_action_features(states, times).detach()
            with torch.no_grad():
                actor_q = self.critic(actor_features).squeeze(-1)
            log_probs = F.log_softmax(actor_logits, dim=-1)
            probs = log_probs.exp()
            actor_loss = (
                probs * (self.entropy_coef * log_probs - actor_q)
            ).sum(-1).mean()
        else:
            distribution = Categorical(logits=actor_logits)
            sampled_actions = distribution.sample()
            sampled_action_values = self._action_values(sampled_actions.detach().numpy())
            actor_features = self._phi(states, sampled_action_values, times).detach()
            with torch.no_grad():
                actor_q = self.critic(actor_features).squeeze(-1)
            actor_loss = -(
                distribution.log_prob(sampled_actions) * actor_q
            ).mean() - self.entropy_coef * distribution.entropy().mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_parameter.mul_(1 - self.tau).add_(parameter, alpha=self.tau)

    @property
    def num_parameters(self):
        modules = (self.phi, self.mu, self.actor, self.critic)
        return sum(p.numel() for module in modules for p in module.parameters())
