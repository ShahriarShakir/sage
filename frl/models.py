"""
frl/models.py — Actor-Critic Neural Network Architectures
Created: 2026-02-26

Provides MLP-based actor and critic networks for multi-agent RL,
supporting both continuous (Gaussian) and discrete action spaces.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Tuple, Optional, Dict, List


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _make_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: str = "relu",
    output_activation: Optional[str] = None,
) -> nn.Sequential:
    """Construct a simple feedforward MLP."""
    act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[activation]
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(act_fn())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    if output_activation == "tanh":
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Actor (Policy) network
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """
    Policy network.
    - Discrete actions  → outputs logits, samples via Categorical.
    - Continuous actions → outputs (mu, log_std), samples via Normal.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = (128, 128),
        continuous: bool = False,
        activation: str = "relu",
    ):
        super().__init__()
        self.continuous = continuous
        self.act_dim = act_dim

        if continuous:
            self.net = _make_mlp(obs_dim, list(hidden_dims), act_dim, activation)
            self.log_std = nn.Parameter(torch.zeros(act_dim))
        else:
            self.net = _make_mlp(obs_dim, list(hidden_dims), act_dim, activation)

    def forward(self, obs: torch.Tensor):
        """Return distribution parameters (logits or mu)."""
        return self.net(obs)

    def get_distribution(self, obs: torch.Tensor):
        if self.continuous:
            mu = self.net(obs)
            std = self.log_std.exp().expand_as(mu)
            return Normal(mu, std)
        else:
            logits = self.net(obs)
            return Categorical(logits=logits)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return (action, log_prob)."""
        dist = self.get_distribution(obs)
        if deterministic:
            if self.continuous:
                action = dist.mean
            else:
                action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        if self.continuous:
            log_prob = log_prob.sum(dim=-1)
        return action, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log_prob and entropy for given actions."""
        dist = self.get_distribution(obs)
        log_prob = dist.log_prob(actions)
        if self.continuous:
            log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy()
        if self.continuous:
            entropy = entropy.sum(dim=-1)
        return log_prob, entropy


# ---------------------------------------------------------------------------
# Critic (Value) network
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """State-value function V(s)."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: List[int] = (128, 128),
        activation: str = "relu",
    ):
        super().__init__()
        self.net = _make_mlp(obs_dim, list(hidden_dims), 1, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ---------------------------------------------------------------------------
# Combined Actor-Critic
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Wraps actor + critic together for convenience."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = (128, 128),
        continuous: bool = False,
        activation: str = "relu",
    ):
        super().__init__()
        self.actor = Actor(obs_dim, act_dim, hidden_dims, continuous, activation)
        self.critic = Critic(obs_dim, hidden_dims, activation)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action, log_prob = self.actor.act(obs, deterministic)
        value = self.critic(obs)
        return action, log_prob, value

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_prob, entropy = self.actor.evaluate_actions(obs, actions)
        value = self.critic(obs)
        return log_prob, entropy, value

    def get_state_dict_delta(self, reference: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute parameter delta relative to a reference state dict."""
        current = self.state_dict()
        delta = {}
        for k in current:
            delta[k] = current[k] - reference[k]
        return delta

    def apply_delta(self, delta: Dict[str, torch.Tensor]):
        """Apply a parameter delta to current weights."""
        current = self.state_dict()
        for k in delta:
            current[k] = current[k] + delta[k]
        self.load_state_dict(current)
