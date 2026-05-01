"""
frl/envs/smac_wrapper.py — SMACv2 Environment Wrapper (Plug-in Interface)
Created: 2026-02-26

Provides a Gymnasium-compatible interface for StarCraft Multi-Agent Challenge v2.
Wraps SMACv2 as a single-agent env per client for federated training.

NOTE: Requires StarCraft II binary and smacv2 package.
      Install: pip install smacv2
      SC2 binary: see https://github.com/oxwhirl/smacv2
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SMACClientEnv(gym.Env):
    """
    Wraps a SMACv2 environment as a single-agent Gym env.

    In the federated MARL setting, each client controls one agent
    in the StarCraft battle scenario. The observation is the local
    observation of that agent, and the action is the individual
    agent's action.

    For centralized value function, the global state is available
    through the info dict.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        map_name: str = "3m",
        agent_idx: int = 0,
        max_steps: int = 200,
        other_agents_policy=None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.map_name = map_name
        self.agent_idx = agent_idx
        self.max_steps = max_steps
        self.other_agents_policy = other_agents_policy

        self._env = None
        self._n_agents = None
        self._obs_dim = None
        self._n_actions = None
        self._step_count = 0

        try:
            self._create_env(seed)
        except ImportError:
            logger.warning(
                "smacv2 not installed. SMACClientEnv will use a dummy env. "
                "Install: pip install smacv2"
            )
            self._setup_dummy()

    def _create_env(self, seed=None):
        """Try to create the actual SMACv2 environment."""
        try:
            from smacv2.env import StarCraft2Env
            self._env = StarCraft2Env(map_name=self.map_name, seed=seed or 42)
            env_info = self._env.get_env_info()
            self._n_agents = env_info["n_agents"]
            self._obs_dim = env_info["obs_shape"]
            self._n_actions = env_info["n_actions"]

            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
            )
            self.action_space = gym.spaces.Discrete(self._n_actions)
            self._is_dummy = False
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to create SMACv2 env: {e}")
            self._setup_dummy()

    def _setup_dummy(self):
        """Setup a dummy env for testing without StarCraft II."""
        self._is_dummy = True
        self._n_agents = 3
        self._obs_dim = 30
        self._n_actions = 9

        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(self._obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self._n_actions)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        self._step_count = 0

        if self._is_dummy:
            obs = np.random.randn(self._obs_dim).astype(np.float32)
            return obs, {"global_state": np.random.randn(self._obs_dim * 2).astype(np.float32)}

        obs_list, state = self._env.reset()
        obs = obs_list[self.agent_idx] if self.agent_idx < len(obs_list) else np.zeros(self._obs_dim)
        return obs.astype(np.float32), {"global_state": state}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1

        if self._is_dummy:
            obs = np.random.randn(self._obs_dim).astype(np.float32)
            reward = float(np.random.randn() * 0.1)
            done = self._step_count >= self.max_steps
            return obs, reward, done, False, {}

        # Build joint action
        actions = []
        avail_actions = self._env.get_avail_actions()
        for i in range(self._n_agents):
            if i == self.agent_idx:
                actions.append(int(action))
            elif self.other_agents_policy is not None:
                obs_i = self._env.get_obs()[i]
                act_i = self.other_agents_policy(obs_i, i, avail_actions[i])
                actions.append(act_i)
            else:
                # Random action from available
                avail = avail_actions[i]
                valid = np.where(avail)[0]
                actions.append(int(np.random.choice(valid)) if len(valid) > 0 else 0)

        reward, done, info = self._env.step(actions)
        obs_list = self._env.get_obs()
        obs = obs_list[self.agent_idx] if self.agent_idx < len(obs_list) else np.zeros(self._obs_dim)

        truncated = self._step_count >= self.max_steps
        return obs.astype(np.float32), float(reward), done, truncated, info

    def close(self):
        if self._env is not None and not self._is_dummy:
            self._env.close()


def make_smac_env(
    map_name: str = "3m",
    agent_idx: int = 0,
    max_steps: int = 200,
    seed: Optional[int] = None,
) -> SMACClientEnv:
    """Factory function for creating SMAC environments."""
    return SMACClientEnv(map_name=map_name, agent_idx=agent_idx, max_steps=max_steps, seed=seed)


def make_smac_env_factory(
    map_name: str = "3m",
    agent_idx: int = 0,
    max_steps: int = 200,
    seed: Optional[int] = None,
):
    """Return a factory callable."""
    def factory():
        return make_smac_env(map_name, agent_idx, max_steps, seed)
    return factory
