"""
frl/envs/gym_wrapper.py — Standard Gymnasium Environment Wrapper
Created: 2026-02-26

Simple wrapper that makes any Gymnasium environment compatible
with our FRLClient interface. Useful for debugging and single-agent
baselines (CartPole, LunarLander, etc.).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Optional


def make_gym_env(
    env_id: str = "CartPole-v1",
    seed: Optional[int] = None,
) -> gym.Env:
    """Create a standard Gymnasium environment."""
    env = gym.make(env_id)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_gym_env_factory(
    env_id: str = "CartPole-v1",
    seed: Optional[int] = None,
):
    """Return a factory callable."""
    def factory():
        return make_gym_env(env_id, seed)
    return factory
