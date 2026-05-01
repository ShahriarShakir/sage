"""
frl/envs/mpe_wrapper.py — PettingZoo MPE Environment Wrapper
Created: 2026-02-26

Wraps PettingZoo MPE (Multi-agent Particle Environment) tasks
into a single-agent Gymnasium-compatible interface for each client.

Supported scenarios:
  - simple_spread_v3: cooperative navigation
  - simple_adversary_v3: adversary pursuit
  - simple_tag_v3: predator-prey
  - simple_reference_v3: cooperative communication
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any


class MPEClientEnv(gym.Env):
    """
    Wraps a PettingZoo MPE environment as a single-agent Gym env
    for one specific agent (client) in the multi-agent game.

    Other agents use a fixed (or slowly updated) policy.
    This is the standard approach in federated MARL:
    each client trains its own policy while other agents' policies
    are held fixed to the latest global model.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        scenario: str = "simple_spread_v3",
        agent_idx: int = 0,
        max_cycles: int = 25,
        continuous_actions: bool = True,
        other_agents_policy=None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.scenario_name = scenario
        self.agent_idx = agent_idx
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.other_agents_policy = other_agents_policy
        self.render_mode = render_mode

        # Create the parallel env
        self._create_env()

    def _create_env(self):
        """Create the PettingZoo parallel environment."""
        from pettingzoo.mpe import (
            simple_spread_v3,
            simple_adversary_v3,
            simple_tag_v3,
            simple_reference_v3,
        )

        env_map = {
            "simple_spread_v3": simple_spread_v3,
            "simple_adversary_v3": simple_adversary_v3,
            "simple_tag_v3": simple_tag_v3,
            "simple_reference_v3": simple_reference_v3,
        }

        if self.scenario_name not in env_map:
            raise ValueError(
                f"Unknown MPE scenario: {self.scenario_name}. "
                f"Choose from {list(env_map.keys())}"
            )

        env_module = env_map[self.scenario_name]
        self.par_env = env_module.parallel_env(
            max_cycles=self.max_cycles,
            continuous_actions=self.continuous_actions,
            render_mode=self.render_mode,
        )
        self.par_env.reset()

        self.agents = self.par_env.possible_agents
        assert self.agent_idx < len(self.agents), (
            f"agent_idx={self.agent_idx} but env has {len(self.agents)} agents"
        )
        self.my_agent = self.agents[self.agent_idx]

        # Get observation and action spaces for our agent
        obs_space = self.par_env.observation_space(self.my_agent)
        act_space = self.par_env.action_space(self.my_agent)

        self.observation_space = obs_space
        self.action_space = act_space

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)
        observations, infos = self.par_env.reset(seed=seed)

        if self.my_agent in observations:
            obs = observations[self.my_agent]
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs.astype(np.float32), {}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step the environment. Our agent takes the given action;
        other agents use their policy (or random if not set).
        """
        # Build action dict for all agents
        actions = {}
        for agent in self.par_env.agents:
            if agent == self.my_agent:
                if isinstance(action, np.ndarray):
                    act = action
                else:
                    act = np.array(action)
                # Clip to action space bounds to suppress warnings
                act = np.clip(act, self.action_space.low, self.action_space.high)
                actions[agent] = act
            else:
                # Other agents: use provided policy or sample random
                if self.other_agents_policy is not None:
                    obs = self._last_observations.get(agent)
                    if obs is not None:
                        actions[agent] = self.other_agents_policy(obs, agent)
                    else:
                        actions[agent] = self.par_env.action_space(agent).sample()
                else:
                    actions[agent] = self.par_env.action_space(agent).sample()

        observations, rewards, terminations, truncations, infos = self.par_env.step(actions)

        # Store observations for other agents' policies next step
        self._last_observations = observations

        if self.my_agent in observations:
            obs = observations[self.my_agent]
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        reward = rewards.get(self.my_agent, 0.0)
        terminated = terminations.get(self.my_agent, True)
        truncated = truncations.get(self.my_agent, False)
        info = infos.get(self.my_agent, {})

        # Check if all agents are done
        if not self.par_env.agents:
            terminated = True

        return obs.astype(np.float32), float(reward), terminated, truncated, info

    @property
    def _last_observations(self):
        if not hasattr(self, "_stored_obs"):
            self._stored_obs = {}
        return self._stored_obs

    @_last_observations.setter
    def _last_observations(self, val):
        self._stored_obs = val if val else {}

    def close(self):
        self.par_env.close()


def make_mpe_env(
    scenario: str = "simple_spread_v3",
    agent_idx: int = 0,
    max_cycles: int = 25,
    continuous_actions: bool = True,
) -> MPEClientEnv:
    """Factory function for creating MPE environments."""
    return MPEClientEnv(
        scenario=scenario,
        agent_idx=agent_idx,
        max_cycles=max_cycles,
        continuous_actions=continuous_actions,
    )


def make_mpe_env_factory(
    scenario: str = "simple_spread_v3",
    agent_idx: int = 0,
    max_cycles: int = 25,
    continuous_actions: bool = True,
):
    """Return a factory callable for creating MPE environments."""
    def factory():
        return make_mpe_env(scenario, agent_idx, max_cycles, continuous_actions)
    return factory
