"""
frl/attacks.py — Attack Implementations for Federated MARL
Created: 2026-02-26
Updated: 2026-02-27 — Added Normalized Attack (Fang et al., WebConf 2025),
                       Inner-Product Manipulation Attack, Adaptive Strategic Attack

Comprehensive attack taxonomy:
  1. Sign-flip attack
  2. Scaling attack
  3. Gaussian noise injection
  4. Directional / angle-based attack (normalized)
  5. Reward poisoning (bias & sparse trigger)
  6. Stale / delayed updates
  7. Sybil clients (identity cloning)
  8. Normalized attack (Fang et al. 2025) — maximizes angular deviation
  9. Inner-product manipulation — minimizes cosine similarity to honest aggregate
  10. Adaptive strategic attack — slow poison that evades detection
"""

from __future__ import annotations

import copy
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque


# ---------------------------------------------------------------------------
# Base attack class
# ---------------------------------------------------------------------------

class Attack:
    """Base class for all attacks."""

    name: str = "base"

    def perturb_delta(
        self,
        delta: Dict[str, torch.Tensor],
        round_idx: int,
        client_id: int,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Perturb a client's parameter delta."""
        return delta

    def perturb_reward(self, reward: float, obs: Any, action: Any, **kwargs) -> float:
        """Perturb a reward signal during local rollout."""
        return reward

    def should_participate(self, round_idx: int, client_id: int) -> bool:
        """Whether this client participates in this round (for dropout/stale)."""
        return True


# ---------------------------------------------------------------------------
# 1. Sign-Flip Attack
# ---------------------------------------------------------------------------

class SignFlipAttack(Attack):
    """
    Flips the sign of all parameter deltas, optionally scaled.

    Maximally opposes the honest gradient direction.
    """

    name = "sign_flip"

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def perturb_delta(self, delta, round_idx, client_id, **kwargs):
        return {k: -self.scale * v for k, v in delta.items()}


# ---------------------------------------------------------------------------
# 2. Scaling Attack
# ---------------------------------------------------------------------------

class ScalingAttack(Attack):
    """
    Scales the delta by a large factor to dominate aggregation.
    """

    name = "scaling"

    def __init__(self, scale_factor: float = 10.0):
        self.scale_factor = scale_factor

    def perturb_delta(self, delta, round_idx, client_id, **kwargs):
        return {k: self.scale_factor * v for k, v in delta.items()}


# ---------------------------------------------------------------------------
# 3. Gaussian Noise Attack
# ---------------------------------------------------------------------------

class GaussianNoiseAttack(Attack):
    """
    Adds Gaussian noise to parameter deltas.
    """

    name = "gaussian_noise"

    def __init__(self, noise_std: float = 1.0, relative: bool = True):
        """
        Args:
            noise_std: standard deviation of noise
            relative: if True, noise_std is relative to delta norm
        """
        self.noise_std = noise_std
        self.relative = relative

    def perturb_delta(self, delta, round_idx, client_id, **kwargs):
        result = {}
        for k, v in delta.items():
            if self.relative:
                std = self.noise_std * v.abs().mean().clamp(min=1e-8)
            else:
                std = self.noise_std
            noise = torch.randn_like(v) * std
            result[k] = v + noise
        return result


# ---------------------------------------------------------------------------
# 4. Directional / Angle-Based Attack
# ---------------------------------------------------------------------------

class DirectionalAttack(Attack):
    """
    Sends a normalized update in the opposite direction of the honest
    aggregate, with a configurable attack magnitude.

    If honest_aggregate is not available, falls back to sending
    unit-norm noise in a random direction.
    """

    name = "directional"

    def __init__(self, magnitude: float = 5.0):
        self.magnitude = magnitude

    def perturb_delta(self, delta, round_idx, client_id, honest_aggregate=None, **kwargs):
        if honest_aggregate is not None:
            # Attack in opposite direction of honest aggregate
            result = {}
            for k in delta:
                v = honest_aggregate[k].float()
                norm = v.norm().clamp(min=1e-8)
                result[k] = -self.magnitude * v / norm
            return result
        else:
            # Random direction with fixed magnitude
            result = {}
            for k, v in delta.items():
                rand_dir = torch.randn_like(v)
                norm = rand_dir.norm().clamp(min=1e-8)
                result[k] = self.magnitude * rand_dir / norm
            return result


# ---------------------------------------------------------------------------
# 5. Reward Poisoning
# ---------------------------------------------------------------------------

class RewardBiasPoisoning(Attack):
    """
    Adds a constant bias to all rewards during local rollout.
    Causes the agent to learn a distorted value function.
    """

    name = "reward_bias"

    def __init__(self, bias: float = 5.0):
        self.bias = bias

    def perturb_reward(self, reward, obs, action, **kwargs):
        return reward + self.bias


class SparseTriggerRewardPoisoning(Attack):
    """
    Occasionally injects large reward signals (sparse triggers)
    to create backdoor-like behavior patterns.
    """

    name = "reward_sparse_trigger"

    def __init__(
        self,
        trigger_prob: float = 0.05,
        trigger_reward: float = 20.0,
        trigger_condition: Optional[callable] = None,
    ):
        self.trigger_prob = trigger_prob
        self.trigger_reward = trigger_reward
        self.trigger_condition = trigger_condition

    def perturb_reward(self, reward, obs, action, **kwargs):
        if self.trigger_condition is not None:
            if self.trigger_condition(obs, action):
                return reward + self.trigger_reward
        elif np.random.random() < self.trigger_prob:
            return reward + self.trigger_reward
        return reward


# ---------------------------------------------------------------------------
# 6. Stale / Delayed Updates
# ---------------------------------------------------------------------------

class StaleUpdateAttack(Attack):
    """
    Sends stale (old) updates instead of current ones.
    Simulates delayed or dropped communication.
    """

    name = "stale_update"

    def __init__(self, delay_rounds: int = 5, dropout_prob: float = 0.0):
        self.delay_rounds = delay_rounds
        self.dropout_prob = dropout_prob
        self._buffer: deque = deque(maxlen=delay_rounds + 1)

    def perturb_delta(self, delta, round_idx, client_id, **kwargs):
        self._buffer.append({k: v.clone() for k, v in delta.items()})
        if len(self._buffer) > self.delay_rounds:
            return self._buffer[0]  # Return delayed update
        else:
            # Not enough history yet; return a zero delta (essentially dropping)
            return {k: torch.zeros_like(v) for k, v in delta.items()}

    def should_participate(self, round_idx, client_id):
        if self.dropout_prob > 0 and np.random.random() < self.dropout_prob:
            return False
        return True


# ---------------------------------------------------------------------------
# 7. Sybil Clients
# ---------------------------------------------------------------------------

class SybilAttack(Attack):
    """
    Sybil attack: multiple colluding clients send identical
    (or near-identical) malicious updates.

    This class wraps another attack and ensures Sybil clients
    coordinate their updates.

    Usage: assign the same SybilAttack instance to multiple client IDs.
    """

    name = "sybil"

    def __init__(
        self,
        inner_attack: Attack,
        n_sybils: int = 3,
        noise_std: float = 0.01,  # small noise to avoid exact duplicates
    ):
        self.inner_attack = inner_attack
        self.n_sybils = n_sybils
        self.noise_std = noise_std
        self._leader_delta: Optional[Dict[str, torch.Tensor]] = None
        self._current_round: int = -1

    def perturb_delta(self, delta, round_idx, client_id, **kwargs):
        # First Sybil in this round becomes the leader
        if round_idx != self._current_round:
            self._current_round = round_idx
            self._leader_delta = self.inner_attack.perturb_delta(
                delta, round_idx, client_id, **kwargs
            )

        # All Sybils send the leader's delta + small noise
        result = {}
        for k, v in self._leader_delta.items():
            noise = torch.randn_like(v) * self.noise_std
            result[k] = v.clone() + noise
        return result


# ---------------------------------------------------------------------------
# Combined / Multi-Attack
# ---------------------------------------------------------------------------

class CombinedAttack(Attack):
    """Apply multiple attacks in sequence."""

    name = "combined"

    def __init__(self, attacks: List[Attack]):
        self.attacks = attacks

    def perturb_delta(self, delta, round_idx, client_id, **kwargs):
        for atk in self.attacks:
            delta = atk.perturb_delta(delta, round_idx, client_id, **kwargs)
        return delta

    def perturb_reward(self, reward, obs, action, **kwargs):
        for atk in self.attacks:
            reward = atk.perturb_reward(reward, obs, action, **kwargs)
        return reward

    def should_participate(self, round_idx, client_id):
        return all(a.should_participate(round_idx, client_id) for a in self.attacks)


# ---------------------------------------------------------------------------
# 8. Normalized Attack (Fang et al., Web Conference 2025)
# ---------------------------------------------------------------------------

class NormalizedAttack(Attack):
    """
    From "Provably Robust Federated Reinforcement Learning" (Fang et al., 2025).

    Unlike traditional attacks that maximize the *distance* of the poisoned
    update from the honest aggregate, the Normalized attack maximizes the
    *angle of deviation*. This breaks many existing Byzantine-robust
    aggregation methods (Krum, Trimmed Mean, etc.) because they focus on
    filtering outliers by norm or coordinate-wise distance.

    The attack:
    1. Computes the opposite direction of the honest mean delta
    2. Normalizes it to have the same norm as the honest mean
    3. This maximizes angular deviation while staying within the norm envelope

    This is significantly harder to detect than sign-flip or scaling attacks.
    """

    name = "normalized"

    def __init__(self, perturbation_factor: float = 1.0):
        """
        Args:
            perturbation_factor: multiplier on the honest mean norm.
                1.0 = same norm as honest (hardest to detect by norm filtering).
        """
        self.perturbation_factor = perturbation_factor

    def perturb_delta(self, delta, round_idx, client_id, honest_aggregate=None, **kwargs):
        if honest_aggregate is not None:
            # Compute flat honest aggregate
            all_tensors = []
            target_norms = []
            result = {}
            for k in delta:
                v = honest_aggregate[k].float()
                target_norms.append(v.norm().item())
                # Opposite direction, normalized to same magnitude
                norm = v.norm().clamp(min=1e-10)
                result[k] = -self.perturbation_factor * v  # opposite direction, same scale
            return result
        else:
            # Without honest aggregate: random direction with matched norm
            result = {}
            for k, v in delta.items():
                rand_dir = torch.randn_like(v)
                rand_norm = rand_dir.norm().clamp(min=1e-10)
                orig_norm = v.norm().clamp(min=1e-10)
                result[k] = -self.perturbation_factor * orig_norm * rand_dir / rand_norm
            return result


# ---------------------------------------------------------------------------
# 9. Inner-Product Manipulation Attack
# ---------------------------------------------------------------------------

class InnerProductManipulationAttack(Attack):
    """
    Crafts updates that minimize the inner product (cosine similarity)
    with the honest aggregate while maintaining a plausible norm.
    Specifically designed to test trust mechanisms that rely on
    cosine similarity tracking (like our temporal direction tracker).

    Uses a projected gradient approach: takes the honest delta,
    projects out the component aligned with the aggregate, and amplifies
    the orthogonal component.
    """

    name = "inner_product_manipulation"

    def __init__(self, orthogonal_scale: float = 2.0):
        self.orthogonal_scale = orthogonal_scale

    def perturb_delta(self, delta, round_idx, client_id, honest_aggregate=None, **kwargs):
        if honest_aggregate is None:
            return delta  # Can't manipulate without reference

        result = {}
        for k in delta:
            v = delta[k].float()
            ref = honest_aggregate[k].float().reshape(-1)
            v_flat = v.reshape(-1)

            ref_norm = ref.norm().clamp(min=1e-10)
            ref_dir = ref / ref_norm

            # Project out the aligned component
            projection = torch.dot(v_flat, ref_dir) * ref_dir
            orthogonal = v_flat - projection

            # Amplify orthogonal, reverse aligned
            poisoned = -projection + self.orthogonal_scale * orthogonal
            result[k] = poisoned.reshape(v.shape)

        return result


# ---------------------------------------------------------------------------
# 10. Adaptive Strategic Attack
# ---------------------------------------------------------------------------

class AdaptiveStrategicAttack(Attack):
    """
    A "slow poison" attack that starts with small perturbations and
    gradually increases the attack magnitude over rounds. Designed to
    evade temporal trust mechanisms by building up a history of
    "good behavior" before striking.

    Phase 1 (warmup_rounds): behave honestly (small perturbation)
    Phase 2 (ramp_rounds): gradually increase perturbation
    Phase 3 (attack rounds): full attack strength
    """

    name = "adaptive_strategic"

    def __init__(
        self,
        warmup_rounds: int = 20,
        ramp_rounds: int = 30,
        max_perturbation: float = 3.0,
        attack_type: str = "sign_flip",
    ):
        self.warmup_rounds = warmup_rounds
        self.ramp_rounds = ramp_rounds
        self.max_perturbation = max_perturbation
        self.attack_type = attack_type

    def _get_strength(self, round_idx: int) -> float:
        if round_idx < self.warmup_rounds:
            return 0.0  # Honest
        elif round_idx < self.warmup_rounds + self.ramp_rounds:
            progress = (round_idx - self.warmup_rounds) / max(self.ramp_rounds, 1)
            return progress * self.max_perturbation
        else:
            return self.max_perturbation

    def perturb_delta(self, delta, round_idx, client_id, **kwargs):
        strength = self._get_strength(round_idx)
        if strength < 1e-6:
            return delta  # Honest during warmup

        if self.attack_type == "sign_flip":
            return {k: -strength * v for k, v in delta.items()}
        elif self.attack_type == "noise":
            return {
                k: v + strength * torch.randn_like(v) for k, v in delta.items()
            }
        else:
            # Default: scaled opposite
            return {k: -strength * v for k, v in delta.items()}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ATTACKS = {
    "none": Attack,
    "sign_flip": SignFlipAttack,
    "scaling": ScalingAttack,
    "gaussian_noise": GaussianNoiseAttack,
    "directional": DirectionalAttack,
    "normalized": NormalizedAttack,
    "inner_product_manipulation": InnerProductManipulationAttack,
    "adaptive_strategic": AdaptiveStrategicAttack,
    "reward_bias": RewardBiasPoisoning,
    "reward_sparse_trigger": SparseTriggerRewardPoisoning,
    "stale_update": StaleUpdateAttack,
    "sybil": SybilAttack,
    "combined": CombinedAttack,
}


def get_attack(name: str, **kwargs) -> Attack:
    """Instantiate an attack by name."""
    if name not in ATTACKS:
        raise ValueError(f"Unknown attack '{name}'. Choose from {list(ATTACKS.keys())}")
    return ATTACKS[name](**kwargs)
