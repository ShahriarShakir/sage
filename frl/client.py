"""
frl/client.py — Federated Client: Local Actor-Critic Training
Created: 2026-02-26

Each client runs K local PPO-style updates on its own partition of the
multi-agent environment, then returns parameter deltas to the server.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List, Any

from frl.models import ActorCritic


# ---------------------------------------------------------------------------
# Rollout buffer (on-policy)
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Simple buffer for on-policy data collection."""

    def __init__(self):
        self.obs: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[bool] = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and discounted returns."""
        T = len(self.rewards)
        advantages = torch.zeros(T)
        returns = torch.zeros(T)
        gae = 0.0

        values = [v.item() if isinstance(v, torch.Tensor) else v for v in self.values]
        values.append(last_value)

        for t in reversed(range(T)):
            mask = 0.0 if self.dones[t] else 1.0
            delta = self.rewards[t] + gamma * values[t + 1] * mask - values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return returns, advantages

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = torch.stack(self.obs)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        return obs, actions, old_log_probs

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class FRLClient:
    """
    A federated RL client that:
    1. Receives global model from server
    2. Runs K local PPO updates
    3. Returns parameter deltas
    """

    def __init__(
        self,
        client_id: int,
        env,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = (128, 128),
        continuous: bool = False,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        local_epochs: int = 4,
        rollout_steps: int = 256,
        minibatch_size: int = 64,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.env = env
        self.device = torch.device(device)

        # Hyperparams
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.local_epochs = local_epochs
        self.rollout_steps = rollout_steps
        self.minibatch_size = minibatch_size

        # Model
        self.model = ActorCritic(
            obs_dim, act_dim, hidden_dims, continuous
        ).to(self.device)
        self.optimizer = torch.optim.Adam([
            {"params": self.model.actor.parameters(), "lr": lr_actor},
            {"params": self.model.critic.parameters(), "lr": lr_critic},
        ])

        # Track the reference state for computing deltas
        self._reference_state: Optional[Dict[str, torch.Tensor]] = None

    # ---- interface with server -----------------------------------------

    def receive_global_model(self, state_dict: Dict[str, torch.Tensor]):
        """Load global model weights and save as reference."""
        self.model.load_state_dict(
            {k: v.clone().to(self.device) for k, v in state_dict.items()}
        )
        self._reference_state = {k: v.clone() for k, v in self.model.state_dict().items()}

    def get_update_delta(self) -> Dict[str, torch.Tensor]:
        """Return parameter delta since last receive_global_model."""
        assert self._reference_state is not None
        return self.model.get_state_dict_delta(self._reference_state)

    # ---- local training ------------------------------------------------

    def collect_rollout(self) -> RolloutBuffer:
        """Collect rollout_steps of experience from the environment."""
        buffer = RolloutBuffer()
        obs, info = self.env.reset()
        obs_t = torch.FloatTensor(obs).to(self.device)

        for _ in range(self.rollout_steps):
            with torch.no_grad():
                action, log_prob, value = self.model.act(obs_t)

            action_np = action.cpu().numpy()
            if isinstance(action_np, np.ndarray) and action_np.ndim == 0:
                action_np = action_np.item()

            # Clip continuous actions to action space bounds
            if hasattr(self.env, 'action_space') and hasattr(self.env.action_space, 'low'):
                action_np = np.clip(action_np, self.env.action_space.low, self.env.action_space.high)

            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            buffer.add(obs_t, action, log_prob, float(reward), value, done)

            if done:
                obs, info = self.env.reset()
                obs_t = torch.FloatTensor(obs).to(self.device)
            else:
                obs_t = torch.FloatTensor(next_obs).to(self.device)

        # Compute last value for GAE
        with torch.no_grad():
            last_value = self.model.critic(obs_t).item()

        return buffer

    def local_update(self) -> Dict[str, float]:
        """
        Run K local PPO updates. Returns training stats.
        """
        buffer = self.collect_rollout()

        with torch.no_grad():
            # Get last obs value for GAE
            last_obs = buffer.obs[-1]
            last_value = self.model.critic(last_obs).item()

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs, actions, old_log_probs = buffer.get_tensors()
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _epoch in range(self.local_epochs):
            # Shuffle and create minibatches
            indices = np.arange(len(buffer))
            np.random.shuffle(indices)

            for start in range(0, len(buffer), self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_lp = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_adv = advantages[mb_idx]

                new_log_prob, entropy, values = self.model.evaluate(mb_obs, mb_actions)

                # PPO clipped objective
                ratio = (new_log_prob - mb_old_lp).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, mb_returns)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        stats = {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "n_updates": n_updates,
            "rollout_reward_mean": np.mean(buffer.rewards),
        }
        return stats

    def run_round(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Full federated round: local update → return delta + stats."""
        stats = self.local_update()
        delta = self.get_update_delta()
        return delta, stats

    # ---- evaluation ----------------------------------------------------

    def evaluate(self, n_episodes: int = 10, seed: Optional[int] = None) -> Dict[str, float]:
        """Run deterministic evaluation episodes."""
        returns = []
        for ep in range(n_episodes):
            if seed is not None:
                obs, info = self.env.reset(seed=seed + ep)
            else:
                obs, info = self.env.reset()
            obs_t = torch.FloatTensor(obs).to(self.device)
            ep_return = 0.0
            done = False
            while not done:
                with torch.no_grad():
                    action, _, _ = self.model.act(obs_t, deterministic=True)
                action_np = action.cpu().numpy()
                if isinstance(action_np, np.ndarray) and action_np.ndim == 0:
                    action_np = action_np.item()
                obs, reward, terminated, truncated, info = self.env.step(action_np)
                obs_t = torch.FloatTensor(obs).to(self.device)
                ep_return += reward
                done = terminated or truncated
            returns.append(ep_return)

        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)),
        }
