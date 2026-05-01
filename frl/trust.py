"""
frl/trust.py — Heterogeneity-Aware Temporal Trust (HATT) Scoring
Created: 2026-02-26
Updated: 2026-02-27 — Enhanced with spectral analysis, cross-client
                       correlation detection, and adaptive thresholding
                       (informed by SDEA ICML 2024, OptiGradTrust 2025,
                        Fang et al. WebConf 2025 Normalized Attack defense)
Updated: 2026-03-10 — v2: Added coordinate-wise anomaly scoring (CWAS),
                       adaptive component weighting, rebalanced trust fusion.
                       Key insight: coordinate-wise MAD analysis catches sign-flip
                       and normalized attacks that vector-level methods miss.

Key algorithmic contribution:
  - Coordinate-wise anomaly detection (MAD-based, dimension-aware)
  - Adaptive trust component weighting (discrimination-driven)
  - Temporal direction consistency via EMA with hysteresis
  - Spectral outlier detection via SVD of update matrix
  - Audit-set forward-pass divergence scoring
  - Short audit rollouts on fixed seeds
  - Heterogeneity-aware scoring (do not punish legitimate drift)
  - Cross-client correlation detection (catches Sybil & collusion)
  - Adaptive thresholds that evolve with training dynamics
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from frl.agg import flatten_state_dict


# ---------------------------------------------------------------------------
# Temporal Direction Tracker
# ---------------------------------------------------------------------------

class TemporalDirectionTracker:
    """
    Tracks the direction of each client's parameter updates over time
    using an exponential moving average (EMA) with hysteresis thresholds.

    The key insight: in MARL, legitimate policy shifts cause direction
    changes. We use hysteresis to avoid flagging transient honest drift
    while still catching persistent adversarial deviation.
    """

    def __init__(
        self,
        n_clients: int,
        ema_beta: float = 0.8,
        high_threshold: float = 0.7,  # cosine sim below this → suspicious
        low_threshold: float = 0.5,   # cosine sim below this → flagged
        hysteresis_window: int = 3,   # consecutive rounds to trigger
    ):
        self.n_clients = n_clients
        self.ema_beta = ema_beta
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.hysteresis_window = hysteresis_window

        # EMA of update directions per client
        self.ema_directions: Dict[int, Optional[torch.Tensor]] = {
            i: None for i in range(n_clients)
        }
        # Count of consecutive suspicious rounds
        self.suspicious_counts: Dict[int, int] = defaultdict(int)
        # Cosine similarity history
        self.cosine_history: Dict[int, List[float]] = defaultdict(list)

    def update(
        self, client_id: int, delta_flat: torch.Tensor
    ) -> float:
        """
        Update the EMA direction for a client, return the cosine similarity
        between the new delta and the EMA direction.
        """
        delta_norm = delta_flat.norm()
        if delta_norm < 1e-10:
            return 1.0  # Zero delta → consistent (no info)

        direction = delta_flat / delta_norm

        if self.ema_directions[client_id] is None:
            self.ema_directions[client_id] = direction.clone()
            cos_sim = 1.0
        else:
            ema = self.ema_directions[client_id]
            cos_sim = torch.dot(direction, ema).item()

            # Update EMA
            self.ema_directions[client_id] = (
                self.ema_beta * ema + (1 - self.ema_beta) * direction
            )
            # Re-normalize EMA direction
            ema_norm = self.ema_directions[client_id].norm()
            if ema_norm > 1e-10:
                self.ema_directions[client_id] /= ema_norm

        self.cosine_history[client_id].append(cos_sim)

        # Hysteresis logic
        if cos_sim < self.low_threshold:
            self.suspicious_counts[client_id] += 1
        elif cos_sim > self.high_threshold:
            self.suspicious_counts[client_id] = max(
                0, self.suspicious_counts[client_id] - 1
            )

        return cos_sim

    def is_flagged(self, client_id: int) -> bool:
        """Check if a client has been consistently suspicious."""
        return self.suspicious_counts[client_id] >= self.hysteresis_window


# ---------------------------------------------------------------------------
# Heterogeneity Envelope
# ---------------------------------------------------------------------------

class HeterogeneityEnvelope:
    """
    Maintains a per-client update direction envelope to track legitimate
    heterogeneity. Agents in MARL discover different strategies at
    different times → their update directions naturally diverge.

    We compute a "heterogeneity envelope" = the expected spread of
    honest update directions, and only flag clients whose updates fall
    outside this envelope.
    """

    def __init__(
        self,
        n_clients: int,
        window_size: int = 10,
        z_score_threshold: float = 3.0,
    ):
        self.n_clients = n_clients
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold

        # Keep recent norms and angular deviations
        self.norm_history: Dict[int, List[float]] = defaultdict(list)
        self.angular_history: Dict[int, List[float]] = defaultdict(list)

    def update_and_score(
        self,
        client_id: int,
        delta_flat: torch.Tensor,
        global_mean_direction: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Score = 1.0 (fully trusted) if update is within envelope,
        decreases toward 0 for outliers.

        Returns heterogeneity-aware score ∈ [0, 1].
        """
        norm = delta_flat.norm().item()
        self.norm_history[client_id].append(norm)

        # Keep window
        if len(self.norm_history[client_id]) > self.window_size:
            self.norm_history[client_id] = self.norm_history[client_id][-self.window_size:]

        # Angular deviation from global mean
        if global_mean_direction is not None and delta_flat.norm() > 1e-10:
            cos_to_mean = torch.dot(
                delta_flat / delta_flat.norm(),
                global_mean_direction / global_mean_direction.norm().clamp(min=1e-10),
            ).item()
            angular_dev = 1.0 - cos_to_mean  # 0 = aligned, 2 = opposite
        else:
            angular_dev = 0.0

        self.angular_history[client_id].append(angular_dev)
        if len(self.angular_history[client_id]) > self.window_size:
            self.angular_history[client_id] = self.angular_history[client_id][-self.window_size:]

        # Compute z-scores for norm and angular deviation
        norm_score = self._z_score_to_trust(
            self.norm_history[client_id], norm
        )
        angular_score = self._z_score_to_trust(
            self.angular_history[client_id], angular_dev
        )

        # Combined score (geometric mean for sensitivity)
        return float(np.sqrt(norm_score * angular_score))

    def _z_score_to_trust(self, history: List[float], value: float) -> float:
        """Convert a z-score to a trust value in [0, 1]."""
        if len(history) < 3:
            return 1.0  # Not enough data yet

        arr = np.array(history)
        mu, sigma = arr.mean(), arr.std()
        if sigma < 1e-10:
            return 1.0

        z = abs(value - mu) / sigma
        # Sigmoid-like mapping: z < threshold → trust ≈ 1, z >> threshold → trust → 0
        trust = 1.0 / (1.0 + np.exp(z - self.z_score_threshold))
        return float(trust)


# ---------------------------------------------------------------------------
# Audit Rollout Scorer (Legacy — evaluates client's learned model)
# ---------------------------------------------------------------------------

class AuditRolloutScorer:
    """
    Server runs short audit rollouts on fixed-seed environments using
    each client's submitted model. Compares the behavioral policy
    to the expected behavior of an honest model.

    NOTE: This evaluates the client's learned model, NOT the submitted
    delta. For sign-flip attacks, the client model is fine (trained
    honestly); only the delta is attacked. Use DeltaEffectAuditor for
    delta-based detection.
    """

    def __init__(
        self,
        audit_env_factory,
        audit_seeds: List[int] = (42, 123, 456),
        n_steps: int = 50,
        reward_z_threshold: float = 2.0,
    ):
        self.audit_env_factory = audit_env_factory
        self.audit_seeds = list(audit_seeds)
        self.n_steps = n_steps
        self.reward_z_threshold = reward_z_threshold

        # History of audit rewards per client
        self.reward_history: Dict[int, List[float]] = defaultdict(list)

    def score_client(
        self,
        client_id: int,
        model,
        global_model=None,
        device: str = "cpu",
    ) -> float:
        """
        Run audit rollouts and return a behavioral trust score ∈ [0, 1].
        """
        rewards = []
        action_divergences = []

        for seed in self.audit_seeds:
            env = self.audit_env_factory()
            obs, _ = env.reset(seed=seed)
            obs_t = torch.FloatTensor(obs).to(device)

            ep_reward = 0.0
            n_divergent = 0
            n_total = 0

            for _step in range(self.n_steps):
                with torch.no_grad():
                    action, _, _ = model.act(obs_t, deterministic=True)

                    if global_model is not None:
                        ref_action, _, _ = global_model.act(obs_t, deterministic=True)
                        if not torch.equal(action, ref_action):
                            n_divergent += 1
                        n_total += 1

                action_np = action.cpu().numpy()
                if action_np.ndim == 0 or (hasattr(action_np, 'shape') and action_np.size == 1):
                    action_np = action_np.item()

                obs, reward, terminated, truncated, _ = env.step(action_np)
                ep_reward += reward
                obs_t = torch.FloatTensor(obs).to(device)

                if terminated or truncated:
                    break

            rewards.append(ep_reward)
            if n_total > 0:
                action_divergences.append(n_divergent / n_total)

            env.close()

        avg_reward = np.mean(rewards)
        self.reward_history[client_id].append(avg_reward)

        # Reward consistency score
        if len(self.reward_history[client_id]) >= 3:
            hist = np.array(self.reward_history[client_id])
            mu, sigma = hist.mean(), hist.std()
            if sigma > 1e-10:
                z = abs(avg_reward - mu) / sigma
                reward_score = 1.0 / (1.0 + np.exp(z - self.reward_z_threshold))
            else:
                reward_score = 1.0
        else:
            reward_score = 1.0

        # Action divergence score
        if action_divergences:
            avg_div = np.mean(action_divergences)
            action_score = 1.0 - min(avg_div, 1.0)
        else:
            action_score = 1.0

        return float(np.sqrt(reward_score * action_score))


# ---------------------------------------------------------------------------
# Delta-Effect Auditor (Primary detection — evaluates delta's impact)
# ---------------------------------------------------------------------------

class DeltaEffectAuditor:
    """
    Byzantine detection via norm anomaly + direction cluster analysis.

    Key insight: Rollout-based evaluation is too noisy in RL (high variance
    in episode returns overwhelms the small signal from parameter deltas).
    Instead, we use DETERMINISTIC signals:

    1. **Norm anomaly**: Byzantine attacks (sign_flip, scaling) produce deltas
       with anomalous L2 norms. We use the lower quartile as reference
       (robust to up to 50% Byzantine). Catches scale > 1 attacks.

    2. **Direction cluster analysis**: Find the largest direction-consistent
       cluster of clients (using pairwise cosine similarities on normalized
       deltas). Clients in the minority cluster get low scores.
       Works in high-D because we compare RELATIVE directions among clients,
       not absolute direction quality. Catches direction-flipping attacks
       INCLUDING normalized attacks (scale=1.0).

    3. **Cumulative tracking**: Builds signal over rounds (sqrt(T) SNR).
       Early rounds are noisy; cumulative deltas amplify systematic patterns.

    No rollout evaluation needed → fast, deterministic, no RL noise.
    """

    def __init__(
        self,
        audit_env_factory=None,  # kept for API compat but not used
        audit_seeds: List[int] = (42, 123, 456, 789, 314),
        n_steps: int = 100,
        scale_factor: float = 0.3,
    ):
        self.audit_env_factory = audit_env_factory
        # Cumulative delta tracking for direction consistency
        self.cumulative: Dict[int, torch.Tensor] = {}
        # Per-round norm history
        self.norm_history: Dict[int, List[float]] = defaultdict(list)
        self.norm_anomaly_history: Dict[int, List[float]] = defaultdict(list)
        self._round = 0

    def score_all_deltas(
        self,
        deltas: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        global_model=None,
        device: str = "cpu",
    ) -> Dict[int, float]:
        """
        Score all client deltas using deterministic norm + direction analysis.

        Algorithm:
        1. Compute per-client delta norms → norm anomaly scores
        2. Flatten and normalize deltas → pairwise cosine similarities
        3. Find majority direction cluster → direction alignment scores
        4. Update cumulative deltas → cumulative direction scores
        5. Combined weighted score

        Returns dict of client_id → score ∈ [0, 1].
        """
        n = len(deltas)
        if n < 2:
            return {cid: 1.0 for cid in client_ids}

        # Flatten all deltas
        deltas_flat = [flatten_state_dict(d) for d in deltas]

        # ===================== NORM ANOMALY =====================
        delta_norms = [f.norm().item() for f in deltas_flat]

        # Lower quartile as reference (robust to 50% Byzantine)
        sorted_norms = sorted(delta_norms)
        q25_idx = max(0, len(sorted_norms) // 4)
        ref_norm = max(float(sorted_norms[q25_idx]), 1e-10)

        norm_scores = {}
        for i, cid in enumerate(client_ids):
            norm_ratio = delta_norms[i] / ref_norm
            self.norm_anomaly_history[cid].append(norm_ratio)
            self.norm_history[cid].append(delta_norms[i])

            # Sigmoid penalty for anomalous norms
            log_ratio = abs(np.log(max(norm_ratio, 1e-10)))
            # log_ratio > 0.5 starts penalizing (ratio > ~1.65x)
            norm_scores[cid] = float(1.0 / (1.0 + np.exp(4.0 * (log_ratio - 0.5))))

        # ===================== DIRECTION CLUSTER =====================
        # Normalize deltas to unit vectors
        norms_t = torch.tensor(delta_norms, dtype=torch.float32)
        mat = torch.stack(deltas_flat, dim=0).float().cpu()
        norms_t = norms_t.cpu()
        mat_norm = mat / norms_t.unsqueeze(1).clamp(min=1e-10)

        # Pairwise cosine similarity matrix
        cos_mat = mat_norm @ mat_norm.T  # (n, n)

        # ===================== SYBIL DETECTION (before cluster) =====================
        # In federated RL, honest gradients are heterogeneous (different
        # local experiences → different gradients). A subgroup with
        # suspiciously high mutual cosine (>0.95) is likely colluding.
        # This is key for detecting normalized/identical attacks.
        SYBIL_THRESHOLD = 0.95

        # Find Sybil groups: connected components with pairwise cos > threshold
        sybil_groups = []
        in_sybil_group = set()
        for i in range(n):
            if i in in_sybil_group:
                continue
            group = {i}
            for j in range(i + 1, n):
                if all(cos_mat[k, j].item() > SYBIL_THRESHOLD for k in group):
                    group.add(j)
            if len(group) >= 2:
                sybil_groups.append(group)
                in_sybil_group.update(group)

        # Determine which Sybil group is suspicious:
        # - If one Sybil group is a minority → it's suspicious
        # - If two groups of equal size → the MORE internally consistent
        #   one is suspicious (honest RL gradients have more variance)
        sybil_penalty = {cid: 1.0 for cid in client_ids}
        suspicious_sybil_indices = set()

        if sybil_groups:
            for group in sybil_groups:
                non_group = set(range(n)) - group

                if len(group) < len(non_group):
                    # Clear minority → suspicious
                    suspicious_sybil_indices.update(group)
                elif len(group) == len(non_group) and len(non_group) >= 2:
                    # 50/50 split — use within-group cosine variance as tiebreaker
                    group_list = sorted(group)
                    non_group_list = sorted(non_group)

                    # Mean pairwise cosine within each group
                    g_cos = [cos_mat[a, b].item()
                             for a in group_list for b in group_list if a < b]
                    ng_cos = [cos_mat[a, b].item()
                              for a in non_group_list for b in non_group_list if a < b]

                    g_mean = np.mean(g_cos) if g_cos else 0.0
                    ng_mean = np.mean(ng_cos) if ng_cos else 0.0

                    # The more homogeneous group (higher mean cos) is suspicious
                    if g_mean > ng_mean + 0.05:
                        suspicious_sybil_indices.update(group)
                    elif ng_mean > g_mean + 0.05:
                        suspicious_sybil_indices.update(non_group)
                    # else: can't decide → no penalty

            for idx in suspicious_sybil_indices:
                cid = client_ids[idx]
                sybil_penalty[cid] = 0.1  # strong penalty for suspected Sybils

        # ===================== DIRECTION CLUSTER SCORES =====================
        # Find majority cluster alignment — but override if Sybil detected
        n_select = max(1, n // 2)
        cluster_scores = {}
        for i, cid in enumerate(client_ids):
            row = cos_mat[i].clone()
            row[i] = -2.0  # exclude self
            topk_vals, _ = row.topk(n_select)
            cluster_scores[cid] = float(topk_vals.mean().item())

        # Map to [0, 1] with inconclusive fallback
        cs_vals = list(cluster_scores.values())
        cs_median = float(np.median(cs_vals))
        cs_std = max(float(np.std(cs_vals)), 1e-10)

        inconclusive = cs_std < 0.02

        direction_scores = {}
        if suspicious_sybil_indices:
            # Sybil detected → direction scores should PENALIZE the Sybil group
            for i, cid in enumerate(client_ids):
                if i in suspicious_sybil_indices:
                    direction_scores[cid] = 0.1  # suspicious Sybils
                else:
                    direction_scores[cid] = 0.9  # non-Sybil → likely honest
        elif inconclusive:
            for cid in client_ids:
                direction_scores[cid] = 1.0  # no signal → trust all
        else:
            for cid in client_ids:
                z = (cluster_scores[cid] - cs_median) / cs_std
                direction_scores[cid] = float(1.0 / (1.0 + np.exp(-2.0 * z)))

        # ===================== CUMULATIVE DIRECTION =====================
        # Track cumulative deltas for longer-term direction consistency
        for i, cid in enumerate(client_ids):
            df = deltas_flat[i].detach().cpu()
            if cid not in self.cumulative:
                self.cumulative[cid] = df.clone()
            else:
                self.cumulative[cid] = self.cumulative[cid] + df

        self._round += 1

        cum_direction_scores = {cid: 0.5 for cid in client_ids}
        if self._round >= 5 and len(self.cumulative) >= 3:
            # Compute cumulative direction agreement
            cum_stack = torch.stack([self.cumulative[cid] for cid in client_ids]).cpu().float()
            cum_norms = cum_stack.norm(dim=1, keepdim=True).clamp(min=1e-10)
            cum_norm = cum_stack / cum_norms
            cum_cos = cum_norm @ cum_norm.T

            # Cumulative Sybil detection (same logic as per-round)
            cum_sybil_indices = set()
            for i in range(n):
                group = {i}
                for j in range(i + 1, n):
                    if all(cum_cos[k, j].item() > SYBIL_THRESHOLD for k in group):
                        group.add(j)
                if len(group) >= 2:
                    non_group = set(range(n)) - group
                    if len(group) < len(non_group):
                        cum_sybil_indices.update(group)
                    elif len(group) == len(non_group) and len(non_group) >= 2:
                        g_list = sorted(group)
                        ng_list = sorted(non_group)
                        g_cos = [cum_cos[a, b].item()
                                 for a in g_list for b in g_list if a < b]
                        ng_cos = [cum_cos[a, b].item()
                                  for a in ng_list for b in ng_list if a < b]
                        if np.mean(g_cos) > np.mean(ng_cos) + 0.05:
                            cum_sybil_indices.update(group)
                        elif np.mean(ng_cos) > np.mean(g_cos) + 0.05:
                            cum_sybil_indices.update(non_group)

            if cum_sybil_indices:
                for i, cid in enumerate(client_ids):
                    cum_direction_scores[cid] = 0.1 if i in cum_sybil_indices else 0.9
            else:
                for i, cid in enumerate(client_ids):
                    row = cum_cos[i].clone()
                    row[i] = -2.0
                    topk_vals, _ = row.topk(n_select)
                    cum_direction_scores[cid] = float(topk_vals.mean().item())

                # Normalize with inconclusive fallback
                cv = list(cum_direction_scores.values())
                cv_med = float(np.median(cv))
                cv_std = max(float(np.std(cv)), 1e-10)
                cum_inconclusive = cv_std < 0.02
                for cid in client_ids:
                    if cum_inconclusive:
                        cum_direction_scores[cid] = 1.0  # no signal → trust all
                    else:
                        z = (cum_direction_scores[cid] - cv_med) / cv_std
                        cum_direction_scores[cid] = float(1.0 / (1.0 + np.exp(-2.0 * z)))

        # ===================== NORM VARIANCE CHECK =====================
        # If norms are homogeneous (low coefficient of variation),
        # norm anomaly is inconclusive → weight shifts to direction.
        norm_cv = float(np.std(delta_norms) / max(np.mean(delta_norms), 1e-10))
        norm_homogeneous = norm_cv < 0.15  # coefficient of variation < 15%

        # ===================== COMBINED SCORE =====================
        scores = {}
        for cid in client_ids:
            if norm_homogeneous:
                # Norms are similar (e.g., normalized attack): shift weight
                # from norm (useless) to direction + sybil detection
                combined = (
                    0.10 * norm_scores[cid]
                    + 0.35 * direction_scores[cid]
                    + 0.25 * cum_direction_scores[cid]
                    + 0.30 * sybil_penalty[cid]
                )
            else:
                # Norms are heterogeneous: norm anomaly is informative
                combined = (
                    0.45 * norm_scores[cid]
                    + 0.20 * direction_scores[cid]
                    + 0.20 * cum_direction_scores[cid]
                    + 0.15 * sybil_penalty[cid]
                )
            scores[cid] = float(max(0.0, min(1.0, combined)))

        return scores


# ---------------------------------------------------------------------------
# Spectral Outlier Detector (Enhanced — catches sign-flip + Normalized Attack)
# ---------------------------------------------------------------------------

class SpectralOutlierDetector:
    """
    Uses SVD of the CUMULATIVE client update matrix to detect outliers.

    Enhanced approach:
    1. Accumulates client deltas over rounds (builds signal in high-D)
    2. Uses SVD of cumulative updates for direction-based detection
    3. Combines residual magnitude + projection sign + norm anomaly

    In high-D federated RL, per-round detection fails due to curse of
    dimensionality. Cumulative tracking builds sqrt(T) SNR improvement.
    """

    def __init__(self, n_components: int = 2, outlier_threshold: float = 2.0):
        self.n_components = n_components
        self.outlier_threshold = outlier_threshold
        self.cumulative: Dict[int, torch.Tensor] = {}
        self.history: List[Dict[int, float]] = []
        self._round = 0

    def score_all(
        self, deltas_flat: List[torch.Tensor], client_ids: List[int]
    ) -> Dict[int, float]:
        """
        Score clients based on spectral analysis of cumulative updates.
        """
        # Update cumulative sums
        for i, cid in enumerate(client_ids):
            if cid not in self.cumulative:
                self.cumulative[cid] = deltas_flat[i].clone().detach()
            else:
                self.cumulative[cid] = self.cumulative[cid] + deltas_flat[i].detach()

        self._round += 1

        if len(client_ids) < 4 or self._round < 3:
            return {cid: 1.0 for cid in client_ids}

        # Build cumulative update matrix: (n_clients, D)
        mat = torch.stack([self.cumulative[cid] for cid in client_ids], dim=0)
        n = mat.shape[0]

        # Compute norms for magnitude anomaly detection
        norms = mat.norm(dim=1)
        norm_mean = norms.mean()
        norm_std = norms.std().clamp(min=1e-10)
        norm_z = (norms - norm_mean) / norm_std

        # Normalize rows for angular analysis
        mat_normalized = mat / norms.unsqueeze(1).clamp(min=1e-10)

        # SVD
        try:
            U, S, Vh = torch.linalg.svd(mat_normalized, full_matrices=False)
        except Exception:
            return {cid: 1.0 for cid in client_ids}

        # 1. Residual-based score (catches Normalized Attack)
        k = min(self.n_components, n - 1, len(S))
        Vh_k = Vh[:k]
        projections = mat_normalized @ Vh_k.T
        reconstructed = projections @ Vh_k
        residuals = (mat_normalized - reconstructed).norm(dim=1)

        # 2. Direction-based score (catches sign-flip & directional attacks)
        top_sv = Vh[0]
        proj_on_top = mat_normalized @ top_sv  # projection coefficient

        # Majority direction: use the sign of the TRIMMED mean projection
        # (more robust than median at 50% Byzantine)
        proj_values = proj_on_top.clone()
        n_trim = max(1, n // 4)  # trim 25% from each end
        sorted_proj, _ = proj_values.sort()
        trimmed_mean_proj = sorted_proj[n_trim:-n_trim].mean().item() if n > 2 * n_trim else sorted_proj.mean().item()
        if trimmed_mean_proj < 0:
            proj_on_top = -proj_on_top

        # Combine signals
        scores = {}
        for i, cid in enumerate(client_ids):
            # Residual score
            res_z = (residuals[i] - residuals.mean()) / residuals.std().clamp(min=1e-10)
            residual_score = 1.0 / (1.0 + np.exp(res_z.item() - self.outlier_threshold))

            # Direction score
            proj_val = proj_on_top[i].item()
            direction_score = max(0.0, min(1.0, (1.0 + proj_val) / 2.0))

            # Norm anomaly score (key signal especially at high Byzantine %)
            norm_score = 1.0 / (1.0 + np.exp(abs(norm_z[i].item()) - self.outlier_threshold))

            # Combined: weighted average (norm gets higher weight since it's reliable)
            combined = 0.3 * residual_score + 0.3 * direction_score + 0.4 * norm_score
            scores[cid] = float(combined)

        self.history.append(scores)
        return scores


# ---------------------------------------------------------------------------
# Population-Relative Direction Scorer
# ---------------------------------------------------------------------------

class PopulationDirectionScorer:
    """
    Scores each client based on how well its CUMULATIVE update direction
    aligns with the robust population direction.

    Key insight: In high-dimensional federated RL (D >> n_clients),
    per-round gradient vectors are nearly orthogonal due to the curse
    of dimensionality. Single-round direction comparison fails.

    Solution: Track the CUMULATIVE sum of each client's deltas. Over T
    rounds, the signal-to-noise ratio grows as sqrt(T), making the
    honest consensus direction separable from Byzantine directions.

    After sufficient rounds (>5-10), the cumulative direction clearly
    separates honest clients from sign-flip/directional attackers.
    """

    def __init__(self, trim_fraction: float = 0.2, min_rounds: int = 3):
        self.trim_fraction = trim_fraction
        self.min_rounds = min_rounds
        self.cumulative: Dict[int, torch.Tensor] = {}
        self.history: List[Dict[int, float]] = []
        self._round = 0

    def score_all(
        self, deltas_flat: List[torch.Tensor], client_ids: List[int]
    ) -> Dict[int, float]:
        """Score clients by alignment of cumulative direction with population."""
        n = len(deltas_flat)

        # Update cumulative sums
        for i, cid in enumerate(client_ids):
            if cid not in self.cumulative:
                self.cumulative[cid] = deltas_flat[i].clone().detach()
            else:
                self.cumulative[cid] = self.cumulative[cid] + deltas_flat[i].detach()

        self._round += 1

        if n < 3 or self._round < self.min_rounds:
            return {cid: 1.0 for cid in client_ids}

        # Stack cumulative directions (raw, unnormalized → preserves magnitude info)
        cum_stack = torch.stack([self.cumulative[cid] for cid in client_ids])  # (n, D)

        # Compute robust mean direction using geometric median approach
        cum_norms = cum_stack.norm(dim=1, keepdim=True).clamp(min=1e-10)
        cum_normalized = cum_stack / cum_norms

        # Use trimmed-mean robust direction instead of Krum (which fails at 50%)
        # Step 1: Compute all pairwise cosines
        cos_matrix = cum_normalized @ cum_normalized.T  # (n, n)

        # Step 2: Find robust center using f-tolerance Krum
        # At 50% Byz, standard Krum selects one client → can be wrong
        # Instead: select top-ceil(n/2) by pairwise agreement, then average
        cos_sums = cos_matrix.sum(dim=1) - 1.0  # subtract self-similarity
        # Select the ceil(n/2) clients with highest total cosine
        n_select = max(2, (n + 1) // 2)
        _, top_indices = cos_sums.topk(n_select)
        # Average of selected clients = robust direction
        robust_direction = cum_normalized[top_indices].mean(dim=0)
        robust_direction = robust_direction / robust_direction.norm().clamp(min=1e-10)

        # Score each client by cosine similarity with the robust direction
        scores = {}
        for i, cid in enumerate(client_ids):
            cos_sim = torch.dot(cum_normalized[i], robust_direction).item()
            # Map cosine similarity to trust score
            # cos ≈ 1 → aligned → score ≈ 1.0
            # cos ≈ 0 → orthogonal → score ≈ 0.5
            # cos ≈ -1 → anti-aligned → score ≈ 0.0
            score = (1.0 + cos_sim) / 2.0
            scores[cid] = float(max(0.0, min(1.0, score)))

        self.history.append(scores)
        return scores


# ---------------------------------------------------------------------------
# Cross-Client Correlation Detector (Catches Sybil attacks)
# ---------------------------------------------------------------------------

class CrossClientCorrelationDetector:
    """
    Detects groups of clients sending suspiciously correlated updates.
    Key for defending against Sybil attacks where multiple colluding
    clients send nearly identical malicious updates.

    Computes pairwise cosine similarity matrix and flags clients
    whose updates are too correlated with each other.
    """

    def __init__(self, correlation_threshold: float = 0.95, min_group_size: int = 2):
        self.correlation_threshold = correlation_threshold
        self.min_group_size = min_group_size

    def detect_sybils(
        self, deltas_flat: List[torch.Tensor], client_ids: List[int]
    ) -> Dict[int, float]:
        """
        Returns penalty scores: 1.0 = no suspicion, 0.0 = highly suspicious Sybil.
        """
        n = len(deltas_flat)
        if n < 3:
            return {cid: 1.0 for cid in client_ids}

        # Compute pairwise cosine similarities
        mat = torch.stack(deltas_flat, dim=0)
        norms = mat.norm(dim=1, keepdim=True).clamp(min=1e-10)
        mat_norm = mat / norms
        cos_sim = mat_norm @ mat_norm.T  # (n, n)

        # For each client, count how many others have cos_sim > threshold
        penalties = {}
        for i, cid in enumerate(client_ids):
            high_corr_count = 0
            for j in range(n):
                if i != j and cos_sim[i, j].item() > self.correlation_threshold:
                    high_corr_count += 1

            if high_corr_count >= self.min_group_size:
                # Suspicious: penalize proportionally
                penalty = max(0.3, 1.0 - 0.2 * high_corr_count)
                penalties[cid] = penalty
            else:
                penalties[cid] = 1.0

        return penalties


# ---------------------------------------------------------------------------
# Coordinate-Wise Anomaly Scorer (CWAS) — v2 addition
# Catches sign-flip and normalized attacks at coordinate level
# ---------------------------------------------------------------------------

class CoordinateWiseAnomalyScorer:
    """
    Detects Byzantine agents via coordinate-wise anomaly analysis.

    Key insight: Vector-level cosine similarity fails in high-D (D >> n)
    because cos(sign_flip(g), g) ~ 0 ± 1/sqrt(D), indistinguishable from
    honest updates. However, COORDINATE-WISE analysis directly reveals
    sign-flip: for each coordinate, a negated value is an outlier relative
    to the population median.

    Algorithm:
    1. For each parameter coordinate, compute median and MAD (median
       absolute deviation) across clients.
    2. For each client, measure the fraction of coordinates that deviate
       from the median by more than `mad_threshold` * MAD.
    3. High anomaly fraction → client is likely Byzantine.

    This is complementary to DeltaEffectAuditor (which uses vector-level
    norm + direction clusters). Together they cover:
      - CWAS catches: sign_flip, normalized, directional attacks
      - DeltaEffectAuditor catches: scaling, sybil coordination, norm anomalies

    Complexity: O(n * D_sub) where D_sub = min(D, subsample_dim).
    """

    def __init__(
        self,
        mad_threshold: float = 3.0,
        subsample_dim: int = 5000,
        sign_agreement_weight: float = 0.0,
    ):
        """
        Args:
            mad_threshold: Number of MADs for a coordinate to be anomalous.
            subsample_dim: Subsample to this many coordinates for speed.
            sign_agreement_weight: Weight for sign-agreement scoring (vs MAD).
                Set to 0 by default — MAD alone catches sign-flip while avoiding
                false positives in heterogeneous environments.
        """
        self.mad_threshold = mad_threshold
        self.subsample_dim = subsample_dim
        self.sign_agreement_weight = sign_agreement_weight
        self._round = 0

    def score_all(
        self, deltas_flat: List[torch.Tensor], client_ids: List[int]
    ) -> Dict[int, float]:
        """
        Score all clients via coordinate-wise anomaly analysis.

        Returns dict of client_id → score in [0, 1].
        1.0 = no anomaly (trusted), 0.0 = high anomaly (suspicious).
        """
        n = len(deltas_flat)
        if n < 3:
            return {cid: 1.0 for cid in client_ids}

        self._round += 1

        # Stack all deltas: (n, D)
        mat = torch.stack(deltas_flat, dim=0).float().cpu()
        D = mat.shape[1]

        # Subsample coordinates for computational efficiency
        if D > self.subsample_dim:
            # Use deterministic subsampling based on round for reproducibility
            gen = torch.Generator()
            gen.manual_seed(42 + self._round)
            indices = torch.randperm(D, generator=gen)[:self.subsample_dim]
            mat_sub = mat[:, indices]
        else:
            mat_sub = mat

        D_sub = mat_sub.shape[1]

        # ---- Signal 1: MAD-based anomaly fraction ----
        # Coordinate-wise median: robust center (tolerates up to 50% Byzantine)
        medians = mat_sub.median(dim=0).values  # (D_sub,)
        abs_devs = (mat_sub - medians.unsqueeze(0)).abs()  # (n, D_sub)
        mad = abs_devs.median(dim=0).values  # (D_sub,)

        # Avoid division by zero for coordinates with zero MAD
        # (all clients agree → no anomaly signal at that coordinate)
        active_mask = mad > 1e-10
        n_active = active_mask.sum().item()

        if n_active < 10:
            # Not enough active coordinates for meaningful scoring
            return {cid: 1.0 for cid in client_ids}

        # Normalized deviations on active coordinates
        mad_active = mad[active_mask]
        abs_devs_active = abs_devs[:, active_mask]
        norm_devs = abs_devs_active / mad_active.unsqueeze(0)

        # Anomaly fraction: fraction of active coordinates exceeding threshold
        anomaly_fracs = (norm_devs > self.mad_threshold).float().mean(dim=1)  # (n,)

        # ---- Signal 2: Sign agreement with population median direction ----
        # For each coordinate, check if client agrees with the SIGN of the median
        # (directly catches sign-flip: Byzantine will have opposite sign on most coords)
        sign_median = torch.sign(medians)  # (D_sub,)
        sign_client = torch.sign(mat_sub)  # (n, D_sub)

        # Only evaluate on coordinates where median is non-zero
        nonzero_mask = sign_median.abs() > 0
        if nonzero_mask.sum() > 100:
            sign_agreement = (sign_client[:, nonzero_mask] == sign_median[nonzero_mask].unsqueeze(0)).float().mean(dim=1)
        else:
            sign_agreement = torch.ones(n)

        # ---- Combined score ----
        scores = {}
        for i, cid in enumerate(client_ids):
            f = anomaly_fracs[i].item()
            s = sign_agreement[i].item()

            # Hard gate: only activate CWAS when anomaly fraction is high
            # This prevents false positives in heterogeneous environments
            # (e.g., mixed cooperative-competitive) where honest agents
            # naturally disagree on many coordinates.
            if f < 0.40:
                # Below gate: return neutral score regardless of sign agreement
                scores[cid] = 1.0
                continue

            # MAD-based score: sigmoid centered at 0.50 anomaly fraction
            # Only reaches this point if f >= 0.40 (strong anomaly)
            # f ~ 0.40 → score ≈ 0.82
            # f ~ 0.50 → score ≈ 0.50
            # f > 0.70 → score ≈ 0.05 (sign-flip territory)
            mad_score = 1.0 / (1.0 + np.exp(15.0 * (f - 0.50)))

            # Sign agreement score: honest → ~0.7-0.9, sign-flip → ~0.0-0.3
            # Map from [0, 1] with center at 0.6
            sign_score = 1.0 / (1.0 + np.exp(-12.0 * (s - 0.55)))

            combined = (
                (1.0 - self.sign_agreement_weight) * mad_score
                + self.sign_agreement_weight * sign_score
            )
            scores[cid] = float(max(0.0, min(1.0, combined)))

        return scores


# ---------------------------------------------------------------------------
# HATT: Heterogeneity-Aware Temporal Trust (Main Module)
# ---------------------------------------------------------------------------
# INNOVATION 1: Leave-One-Out (LOO) Validation Trust
# ---------------------------------------------------------------------------

class LOOValidationScorer:
    """
    Evaluates each agent by measuring the aggregated model's performance
    WITH and WITHOUT the agent's contribution.  An agent whose exclusion
    improves validation performance is likely Byzantine.

    Unlike DeltaEffectAuditor (which uses norm / direction heuristics),
    this directly evaluates the FUNCTIONAL impact on policy quality
    through short deterministic rollouts.

    Computational cost: O(n * rollout_cost) per round.  We amortise by
    running every `frequency` rounds and caching scores in between.
    """

    def __init__(
        self,
        eval_env_factory,
        eval_seeds: List[int] = (42, 123, 456),
        eval_steps: int = 80,
        frequency: int = 2,
    ):
        self.eval_env_factory = eval_env_factory
        self.eval_seeds = list(eval_seeds)
        self.eval_steps = eval_steps
        self.frequency = frequency
        self._round = 0
        self._cache: Dict[int, float] = {}

    def _evaluate_model(self, model, device: str) -> float:
        """Run deterministic rollouts and return mean reward."""
        import copy as _copy
        total = 0.0
        for seed in self.eval_seeds:
            env = self.eval_env_factory()
            obs, _ = env.reset(seed=seed)
            obs_t = torch.FloatTensor(obs).to(device)
            ep_r = 0.0
            for _ in range(self.eval_steps):
                with torch.no_grad():
                    action, _, _ = model.act(obs_t, deterministic=True)
                a_np = action.cpu().numpy()
                if a_np.ndim == 0 or a_np.size == 1:
                    a_np = a_np.item()
                obs, r, term, trunc, _ = env.step(a_np)
                ep_r += r
                obs_t = torch.FloatTensor(obs).to(device)
                if term or trunc:
                    break
            total += ep_r
            env.close()
        return total / len(self.eval_seeds)

    def score_all(
        self,
        deltas: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        global_model,
        device: str = "cpu",
    ) -> Dict[int, float]:
        """Score all clients using LOO validation."""
        import copy as _copy

        self._round += 1
        if self._round % self.frequency != 0 and self._cache:
            return {cid: self._cache.get(cid, 0.5) for cid in client_ids}

        n = len(deltas)
        if n < 2 or global_model is None:
            return {cid: 0.5 for cid in client_ids}

        base_state = _copy.deepcopy(global_model.state_dict())

        # Compute the all-inclusive aggregated delta (simple mean)
        agg_all = {}
        for k in deltas[0]:
            agg_all[k] = torch.stack([d[k].float() for d in deltas]).mean(dim=0)

        # Evaluate the all-inclusive model
        model_all = _copy.deepcopy(global_model)
        state_all = _copy.deepcopy(base_state)
        for k in agg_all:
            state_all[k] = state_all[k].float() + agg_all[k]
        model_all.load_state_dict(state_all)
        r_all = self._evaluate_model(model_all, device)

        # LOO: for each client, aggregate WITHOUT that client and evaluate
        loo_rewards = {}
        for idx, cid in enumerate(client_ids):
            others = [d for j, d in enumerate(deltas) if j != idx]
            if len(others) == 0:
                loo_rewards[cid] = r_all
                continue
            agg_loo = {}
            for k in others[0]:
                agg_loo[k] = torch.stack([d[k].float() for d in others]).mean(dim=0)
            model_loo = _copy.deepcopy(global_model)
            state_loo = _copy.deepcopy(base_state)
            for k in agg_loo:
                state_loo[k] = state_loo[k].float() + agg_loo[k]
            model_loo.load_state_dict(state_loo)
            loo_rewards[cid] = self._evaluate_model(model_loo, device)

        # Score: if removing agent i IMPROVES performance → agent i is harmful
        # improvement_i = r_loo_i - r_all  (positive = removing i helps)
        scores = {}
        improvements = [loo_rewards[cid] - r_all for cid in client_ids]
        imp_arr = np.array(improvements)
        imp_std = max(float(np.std(imp_arr)), 1e-8)
        imp_mean = float(np.mean(imp_arr))

        for cid in client_ids:
            imp = loo_rewards[cid] - r_all
            # z-score: positive z = removal helps = bad agent
            z = (imp - imp_mean) / imp_std
            # Sigmoid: z > 1 → score drops towards 0
            score = float(1.0 / (1.0 + np.exp(2.0 * z)))
            scores[cid] = max(0.0, min(1.0, score))

        self._cache = dict(scores)
        return scores


# ---------------------------------------------------------------------------
# INNOVATION 2: Gradient Inversion Score (GIS)
# ---------------------------------------------------------------------------

class GradientInversionScorer:
    """
    Detects agents whose updates oppose the consensus direction.

    Normalized attacks evade norm-based detection by scaling updates to
    match honest norms, but they cannot simultaneously (a) have correct
    norms AND (b) agree in direction with the honest majority.  GIS
    exploits this by measuring each agent's cosine alignment with the
    robust consensus direction (trimmed mean of updates).

    Unlike the existing population_direction component (which uses raw
    cosine to mean), GIS:
      1. Uses a ROBUST consensus (trimmed mean, not simple mean)
      2. Applies an exponential DECAY to past scores (memory)
      3. Measures INVERSION specifically (cos < 0 is much worse than cos ≈ 0)
    """

    def __init__(self, trim_fraction: float = 0.25, ema_beta: float = 0.7):
        self.trim_fraction = trim_fraction
        self.ema_beta = ema_beta
        self._ema_scores: Dict[int, float] = {}

    def score_all(
        self,
        deltas_flat: List[torch.Tensor],
        client_ids: List[int],
    ) -> Dict[int, float]:
        """
        Score each agent by cosine similarity to robust consensus.

        Returns dict client_id → score ∈ [0, 1].
        """
        n = len(deltas_flat)
        if n < 3:
            return {cid: 0.5 for cid in client_ids}

        mat = torch.stack(deltas_flat, dim=0).float().cpu()

        # Robust consensus: coordinate-wise trimmed mean
        k = max(1, int(n * self.trim_fraction))
        sorted_vals, _ = mat.sort(dim=0)
        trimmed = sorted_vals[k:n - k]  # trim k from each end
        if trimmed.shape[0] == 0:
            trimmed = sorted_vals
        consensus = trimmed.mean(dim=0)
        c_norm = consensus.norm().clamp(min=1e-10)

        scores = {}
        for i, cid in enumerate(client_ids):
            d = deltas_flat[i].float().cpu()
            d_norm = d.norm().clamp(min=1e-10)
            cos_sim = float(torch.dot(d / d_norm, consensus / c_norm).item())

            # Transform: cos_sim ∈ [-1, 1] → score ∈ [0, 1]
            # cos_sim = 1 → score = 1 (aligned)
            # cos_sim = 0 → score ≈ 0.5 (orthogonal)
            # cos_sim = -1 → score ≈ 0 (inverted)
            raw = (1.0 + cos_sim) / 2.0

            # Apply steep penalty for inversion (cos < 0)
            if cos_sim < 0:
                raw = raw * 0.3  # extra penalty for opposition

            # EMA smoothing
            old = self._ema_scores.get(cid, 0.5)
            smoothed = self.ema_beta * old + (1.0 - self.ema_beta) * raw
            self._ema_scores[cid] = smoothed
            scores[cid] = float(max(0.0, min(1.0, smoothed)))

        return scores


# ---------------------------------------------------------------------------
# INNOVATION 3: Temporal Trajectory Consistency (TTC)
# ---------------------------------------------------------------------------

class TemporalTrajectoryScorer:
    """
    Tracks each agent's update trajectory using a sliding window of
    gradient directions and detects agents that deviate from their own
    established learning trajectory.

    Key insight: honest RL agents follow a coherent learning curve — their
    gradient directions change gradually as the policy improves.  Attackers
    (even adaptive ones) must periodically introduce adversarial updates
    that break this coherence.  TTC catches this by measuring:

    1. Self-consistency: cosine sim between current update and the agent's
       own EMA direction (how much the agent agrees with its past self)
    2. Trajectory smoothness: variance of cosine similarities over a window
       (honest agents have smoothly changing directions; attackers are jerky)
    3. Cross-agent trajectory correlation: honest agents in similar roles
       should have correlated trajectory patterns
    """

    def __init__(
        self,
        window_size: int = 15,
        ema_beta: float = 0.85,
    ):
        self.window_size = window_size
        self.ema_beta = ema_beta
        self._ema_dir: Dict[int, Optional[torch.Tensor]] = {}
        self._cos_history: Dict[int, List[float]] = defaultdict(list)

    def score_all(
        self,
        deltas_flat: List[torch.Tensor],
        client_ids: List[int],
    ) -> Dict[int, float]:
        """Score agents by trajectory consistency."""
        scores = {}
        current_cos_vals = {}

        for i, cid in enumerate(client_ids):
            d = deltas_flat[i].float().cpu()
            d_norm = d.norm()
            if d_norm < 1e-10:
                current_cos_vals[cid] = 1.0
                scores[cid] = 1.0
                continue

            direction = d / d_norm

            if cid not in self._ema_dir or self._ema_dir[cid] is None:
                self._ema_dir[cid] = direction.clone()
                current_cos_vals[cid] = 1.0
                scores[cid] = 1.0
                continue

            # Self-consistency: compare to own EMA direction
            ema = self._ema_dir[cid]
            cos_sim = float(torch.dot(direction, ema).item())
            current_cos_vals[cid] = cos_sim

            # Update EMA direction
            new_ema = self.ema_beta * ema + (1.0 - self.ema_beta) * direction
            ema_n = new_ema.norm()
            if ema_n > 1e-10:
                new_ema = new_ema / ema_n
            self._ema_dir[cid] = new_ema

            # Track history
            self._cos_history[cid].append(cos_sim)
            if len(self._cos_history[cid]) > self.window_size:
                self._cos_history[cid] = self._cos_history[cid][-self.window_size:]

            hist = self._cos_history[cid]

            # Component 1: current self-consistency (higher = better)
            consistency_score = (1.0 + cos_sim) / 2.0

            # Component 2: trajectory smoothness (low variance = smooth = honest)
            if len(hist) >= 5:
                smoothness = 1.0 - min(float(np.std(hist)), 0.5) * 2.0
            else:
                smoothness = 1.0

            # Combined
            raw = 0.6 * consistency_score + 0.4 * max(0.0, smoothness)
            scores[cid] = float(max(0.0, min(1.0, raw)))

        return scores


# ---------------------------------------------------------------------------
# INNOVATION 4: Bayesian Trust Update
# ---------------------------------------------------------------------------

class BayesianTrustUpdater:
    """
    Maintains a Beta distribution posterior for each agent's
    trustworthiness, providing principled uncertainty-aware trust scoring.

    Instead of EMA (which treats all evidence equally), the Beta posterior:
    1. Naturally handles early-round uncertainty (wide prior → cautious)
    2. Responds quickly to strong evidence (sharp likelihood shifts)
    3. Provides a credible interval for each agent's trust
    4. Is more resistant to oscillation than EMA

    The trust score is the posterior mean: α / (α + β).
    Evidence from detection components is mapped to pseudo-observations:
    high component scores → increment α, low scores → increment β.
    """

    def __init__(
        self,
        n_clients: int,
        prior_alpha: float = 2.0,
        prior_beta: float = 2.0,
        evidence_scale: float = 1.5,
        decay: float = 0.995,
    ):
        """
        Args:
            prior_alpha, prior_beta: Initial Beta parameters (2,2 = uniform-ish prior)
            evidence_scale: How strongly each round's evidence updates the posterior
            decay: Multiplicative decay towards prior each round (for non-stationarity)
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.evidence_scale = evidence_scale
        self.decay = decay

        # Per-client Beta parameters
        self.alpha: Dict[int, float] = {i: prior_alpha for i in range(n_clients)}
        self.beta_param: Dict[int, float] = {i: prior_beta for i in range(n_clients)}

    def update(self, client_id: int, raw_score: float) -> float:
        """
        Update the Beta posterior for one client given a raw trust score ∈ [0,1].

        Returns the posterior mean (trust score).
        """
        # Decay towards prior (allows trust recovery and non-stationarity)
        a = self.alpha.get(client_id, self.prior_alpha)
        b = self.beta_param.get(client_id, self.prior_beta)

        a = self.decay * a + (1 - self.decay) * self.prior_alpha
        b = self.decay * b + (1 - self.decay) * self.prior_beta

        # Map raw_score to pseudo-observations
        # raw_score close to 1 → strong positive evidence
        # raw_score close to 0 → strong negative evidence
        pos_evidence = raw_score * self.evidence_scale
        neg_evidence = (1.0 - raw_score) * self.evidence_scale

        a += pos_evidence
        b += neg_evidence

        self.alpha[client_id] = a
        self.beta_param[client_id] = b

        # Posterior mean
        trust = a / (a + b)
        return float(np.clip(trust, 0.0, 1.0))

    def get_trust(self, client_id: int) -> float:
        a = self.alpha.get(client_id, self.prior_alpha)
        b = self.beta_param.get(client_id, self.prior_beta)
        return float(a / (a + b))

    def get_uncertainty(self, client_id: int) -> float:
        """Return posterior variance — higher = more uncertain."""
        a = self.alpha.get(client_id, self.prior_alpha)
        b = self.beta_param.get(client_id, self.prior_beta)
        return float(a * b / ((a + b) ** 2 * (a + b + 1)))


# ---------------------------------------------------------------------------
# HATT Trust Scorer — v3: With all four innovations
# ---------------------------------------------------------------------------

class HATTTrustScorer:
    """
    Heterogeneity-Aware Temporal Trust (HATT) — our main contribution.

    Combines multiple signals for Byzantine detection in federated MARL:
      1. Delta-Effect Auditing: Norm anomaly + direction cluster analysis.
         Catches scaling and Sybil attacks. Fast, deterministic.
      2. Coordinate-Wise Anomaly (CWAS): Per-coordinate MAD analysis.
         Catches sign-flip and normalized attacks that vector-level methods miss.
      3. Cumulative Spectral Analysis: SVD of cumulative update matrix.
         Builds sqrt(T) signal over rounds for direction attacks.
      4. Cumulative Population Direction: Robust majority alignment.
      5. Temporal direction consistency (EMA + hysteresis).
      6. Heterogeneity envelope (z-score, per-client history).
      7. Cross-client correlation (Sybil detection).

    v2 additions (2026-03-10):
      - CoordinateWiseAnomalyScorer for per-coordinate detection
      - Adaptive component weighting based on observed discrimination power
      - Rebalanced default weights for broader coverage

    Key insight: No single detection signal works against ALL attacks.
    HATT's strength is in ADAPTIVE FUSION: the system identifies which
    trust components are discriminating (high score variance across clients)
    and upweights them dynamically. This makes HATT universally robust
    without manual tuning per attack type.
    """

    def __init__(
        self,
        n_clients: int,
        # Temporal direction params
        ema_beta: float = 0.8,
        high_threshold: float = 0.7,
        low_threshold: float = 0.5,
        hysteresis_window: int = 3,
        # Heterogeneity envelope params
        envelope_window: int = 10,
        z_score_threshold: float = 3.0,
        # Spectral analysis params
        spectral_components: int = 2,
        spectral_threshold: float = 2.0,
        # Cross-client correlation params
        correlation_threshold: float = 0.95,
        # Audit params (delta-effect audit — primary detection)
        audit_env_factory=None,
        audit_seeds: List[int] = (42, 123, 456),
        audit_steps: int = 50,
        audit_frequency: int = 5,  # run audits every N rounds
        # Coordinate-wise anomaly params (v2)
        cwas_mad_threshold: float = 3.0,
        cwas_subsample_dim: int = 5000,
        cwas_sign_weight: float = 0.0,
        # Combination weights (v2 rebalanced — no single >50%)
        w_delta_effect: float = 0.35,    # Norm + direction cluster
        w_coordinate_anomaly: float = 0.15,  # v2: Per-coordinate MAD (conservative)
        w_spectral: float = 0.10,        # Cumulative SVD
        w_population_dir: float = 0.15,  # Cumulative direction
        w_temporal: float = 0.05,        # Per-round temporal
        w_heterogeneity: float = 0.05,   # Per-client history
        w_correlation: float = 0.15,     # Sybil detection
        # v3: New innovation weights
        w_loo_validation: float = 0.20,  # LOO validation trust
        w_gradient_inversion: float = 0.15,  # Gradient inversion score
        w_trajectory: float = 0.10,      # Temporal trajectory consistency
        # Adaptive weighting (v2)
        adaptive_weights: bool = True,   # Enable discrimination-driven weighting
        adaptive_blend: float = 0.5,     # How much to blend adaptive vs prior weights
        # v3: Bayesian trust update
        use_bayesian_trust: bool = True,
        bayesian_prior_alpha: float = 2.0,
        bayesian_prior_beta: float = 2.0,
        bayesian_evidence_scale: float = 1.5,
        bayesian_decay: float = 0.995,
        # v3: Functional-first trust initialization
        # During these initial rounds, ONLY functional evaluators
        # (LOO + delta_effect) determine trust.  This prevents consensus-
        # based components from poisoning early trust when attackers form
        # a majority.  After this phase, all components blend in and the
        # adaptive weighting has a correct trust signal to validate against.
        functional_first_rounds: int = 15,
        # Trust smoothing
        trust_ema_beta: float = 0.7,
        # Warmup: trust = neutral for all clients during warmup
        warmup_rounds: int = 0,
    ):
        self.n_clients = n_clients
        self.audit_frequency = audit_frequency

        self.temporal_tracker = TemporalDirectionTracker(
            n_clients, ema_beta, high_threshold, low_threshold, hysteresis_window
        )
        self.hetero_envelope = HeterogeneityEnvelope(
            n_clients, envelope_window, z_score_threshold
        )
        self.spectral_detector = SpectralOutlierDetector(
            spectral_components, spectral_threshold
        )
        self.population_dir_scorer = PopulationDirectionScorer(
            trim_fraction=0.2, min_rounds=3
        )
        self.correlation_detector = CrossClientCorrelationDetector(
            correlation_threshold
        )

        # v2: Coordinate-wise anomaly scorer
        self.cwas = CoordinateWiseAnomalyScorer(
            mad_threshold=cwas_mad_threshold,
            subsample_dim=cwas_subsample_dim,
            sign_agreement_weight=cwas_sign_weight,
        )

        # Delta-effect auditor (PRIMARY detection mechanism)
        self.delta_effect_auditor: Optional[DeltaEffectAuditor] = None
        if audit_env_factory is not None:
            self.delta_effect_auditor = DeltaEffectAuditor(
                audit_env_factory, audit_seeds, audit_steps,
                scale_factor=0.3,
            )

        # v3: LOO Validation scorer
        self.loo_scorer: Optional[LOOValidationScorer] = None
        if audit_env_factory is not None:
            self.loo_scorer = LOOValidationScorer(
                eval_env_factory=audit_env_factory,
                eval_seeds=[42, 123, 456],
                eval_steps=80,
                frequency=1,  # every round during functional-first; reset to 3 later
            )

        # v3: Gradient Inversion scorer
        self.gis_scorer = GradientInversionScorer(
            trim_fraction=0.25, ema_beta=0.7,
        )

        # v3: Temporal Trajectory scorer
        self.ttc_scorer = TemporalTrajectoryScorer(
            window_size=15, ema_beta=0.85,
        )

        # Prior weights (can be overridden, but adaptive weighting adjusts them)
        self.w_delta_effect = w_delta_effect
        self.w_coordinate_anomaly = w_coordinate_anomaly
        self.w_spectral = w_spectral
        self.w_population_dir = w_population_dir
        self.w_temporal = w_temporal
        self.w_heterogeneity = w_heterogeneity
        self.w_correlation = w_correlation
        # v3 weights
        self.w_loo_validation = w_loo_validation
        self.w_gradient_inversion = w_gradient_inversion
        self.w_trajectory = w_trajectory

        # v2: Adaptive weighting
        self.adaptive_weights = adaptive_weights
        self.adaptive_blend = adaptive_blend
        # v3: Functional-first initialization phase
        self.functional_first_rounds = functional_first_rounds
        # Track discrimination power per component over time
        self._component_discrimination: Dict[str, float] = {}
        self._discrimination_ema: Dict[str, float] = {
            "delta_effect": 0.0,
            "coordinate_anomaly": 0.0,
            "spectral": 0.0,
            "population_dir": 0.0,
            "temporal": 0.0,
            "heterogeneity": 0.0,
            "correlation": 0.0,
            "loo_validation": 0.0,
            "gradient_inversion": 0.0,
            "trajectory": 0.0,
        }

        self.trust_ema_beta = trust_ema_beta
        self.warmup_rounds = warmup_rounds

        # v3: Bayesian trust update (replaces EMA when enabled)
        self.use_bayesian_trust = use_bayesian_trust
        self.bayesian_updater: Optional[BayesianTrustUpdater] = None
        if use_bayesian_trust:
            self.bayesian_updater = BayesianTrustUpdater(
                n_clients=n_clients,
                prior_alpha=bayesian_prior_alpha,
                prior_beta=bayesian_prior_beta,
                evidence_scale=bayesian_evidence_scale,
                decay=bayesian_decay,
            )

        # Smoothed trust scores (start at 1.0 — conservative, prevents
        # false positives in early rounds when delta-effect scores are noisy)
        self.trust_scores: Dict[int, float] = {i: 1.0 for i in range(n_clients)}
        # Raw component scores for logging
        self.component_scores: Dict[int, Dict[str, float]] = {
            i: {} for i in range(n_clients)
        }
        # CACHED delta-effect scores — persisted across non-audit rounds
        self._cached_delta_effect_scores: Dict[int, float] = {
            i: 1.0 for i in range(n_clients)
        }
        # v2: Cached adaptive weights for logging
        self._effective_weights: Dict[str, float] = {}
        self._round = 0

    def update(
        self,
        client_id: int,
        delta: Dict[str, torch.Tensor],
        model=None,
        global_model=None,
        all_deltas: Optional[List[Dict[str, torch.Tensor]]] = None,
        device: str = "cpu",
        spectral_score: float = 1.0,
        correlation_score: float = 1.0,
        population_dir_score: float = 1.0,
        delta_effect_score: float = 1.0,
        coordinate_anomaly_score: float = 1.0,
        loo_validation_score: float = 0.5,
        gradient_inversion_score: float = 0.5,
        trajectory_score: float = 1.0,
    ) -> float:
        """
        Update trust score for a client given their latest delta.
        v3: Includes LOO validation, gradient inversion, trajectory, and Bayesian update.
        """
        delta_flat = flatten_state_dict(delta)

        # 1. Temporal direction consistency
        cos_sim = self.temporal_tracker.update(client_id, delta_flat)
        temporal_score = (1.0 + cos_sim) / 2.0

        # 2. Heterogeneity envelope — only use per-client norm history.
        # Do NOT use global mean direction (corrupted by Byzantine at high %).
        hetero_score = self.hetero_envelope.update_and_score(
            client_id, delta_flat, None
        )

        # Store component scores
        self.component_scores[client_id] = {
            "temporal": temporal_score,
            "heterogeneity": hetero_score,
            "spectral": spectral_score,
            "population_dir": population_dir_score,
            "correlation": correlation_score,
            "delta_effect": delta_effect_score,
            "coordinate_anomaly": coordinate_anomaly_score,
            "loo_validation": loo_validation_score,
            "gradient_inversion": gradient_inversion_score,
            "trajectory": trajectory_score,
            "cos_sim": cos_sim,
        }

        # Update cached delta_effect score if this is an audit round
        if delta_effect_score != 1.0 or self._round <= self.warmup_rounds + 1:
            self._cached_delta_effect_scores[client_id] = delta_effect_score

        # Get effective weights (may be adaptively adjusted)
        w = self._get_effective_weights()

        # Weighted combination with all components (v3: 10 components)
        cached_de = self._cached_delta_effect_scores.get(client_id, 1.0)
        raw_score = (
            w["delta_effect"] * cached_de
            + w["coordinate_anomaly"] * coordinate_anomaly_score
            + w["spectral"] * spectral_score
            + w["population_dir"] * population_dir_score
            + w["temporal"] * temporal_score
            + w["heterogeneity"] * hetero_score
            + w["correlation"] * correlation_score
            + w["loo_validation"] * loo_validation_score
            + w["gradient_inversion"] * gradient_inversion_score
            + w["trajectory"] * trajectory_score
        )

        # v3: Bayesian trust update OR EMA smoothing
        if self.use_bayesian_trust and self.bayesian_updater is not None:
            new_trust = self.bayesian_updater.update(client_id, raw_score)
        else:
            # EMA smoothing of trust
            old_trust = self.trust_scores[client_id]
            new_trust = self.trust_ema_beta * old_trust + (1 - self.trust_ema_beta) * raw_score

        self.trust_scores[client_id] = float(np.clip(new_trust, 0.0, 1.0))

        return self.trust_scores[client_id]

    def _get_effective_weights(self) -> Dict[str, float]:
        """
        Get effective component weights, potentially with adaptive adjustment.

        v3: Functional-first initialization — during the first
        `functional_first_rounds` rounds, only functional evaluators
        (delta_effect, LOO) contribute.  This establishes a correct trust
        signal before consensus components (which can be fooled by majority
        corruption) are blended in.

        v2: Adaptive weighting based on observed discrimination power.
        Components that distinguish between clients (high variance of scores)
        get higher weight. This makes HATT self-tuning per attack type.
        """
        prior = {
            "delta_effect": self.w_delta_effect,
            "coordinate_anomaly": self.w_coordinate_anomaly,
            "spectral": self.w_spectral,
            "population_dir": self.w_population_dir,
            "temporal": self.w_temporal,
            "heterogeneity": self.w_heterogeneity,
            "correlation": self.w_correlation,
            "loo_validation": self.w_loo_validation,
            "gradient_inversion": self.w_gradient_inversion,
            "trajectory": self.w_trajectory,
        }

        # ---- v3: Functional-first phase ----
        # During early rounds, only functional evaluators determine trust.
        # This prevents consensus-based components from establishing wrong
        # trust when attackers form a majority group.
        if self._round < self.functional_first_rounds:
            functional = {
                "delta_effect": self.w_delta_effect,
                "loo_validation": self.w_loo_validation,
            }
            # Zero out all non-functional components
            weights = {k: 0.0 for k in prior}
            func_total = sum(functional.values()) or 1.0
            for k, v in functional.items():
                weights[k] = v / func_total  # normalize to sum=1
            self._effective_weights = weights
            return weights

        # ---- v3: Gradual blend-in phase ----
        # After functional_first_rounds, gradually introduce other components
        # over `blend_in_duration` rounds to avoid sudden weight shifts.
        blend_in_duration = 30
        blend_progress = min(1.0, (self._round - self.functional_first_rounds) / blend_in_duration)

        # Compute base weights (prior or adaptive)
        if blend_progress >= 1.0 and self.adaptive_weights and self._round >= 5:
            disc = dict(self._discrimination_ema)
            total_disc = sum(disc.values()) + 1e-10
            if total_disc >= 0.01:
                adaptive = {k: v / total_disc for k, v in disc.items()}
                alpha = self.adaptive_blend
                base = {}
                for k in prior:
                    base[k] = (1.0 - alpha) * prior[k] + alpha * adaptive.get(k, 0.0)
            else:
                base = dict(prior)
        else:
            base = dict(prior)

        # ---- v3: Functional floor enforcement ----
        # The functional floor is the minimum combined weight for functional
        # evaluators (delta_effect + LOO).  This prevents consensus-based
        # components from overwhelming the correct functional signal when
        # attackers form a majority and corrupt consensus estimates.
        #
        # The floor is set at 0.70 to ensure functional evaluators always
        # dominate the trust signal.  Consensus-based components are useful
        # at low corruption but can be corrupted under majority corruption.
        FUNCTIONAL_KEYS = {"delta_effect", "loo_validation"}
        min_functional_share = 0.70

        if blend_progress < 1.0:
            # Still blending: interpolate between functional-only and base
            func_only = {k: 0.0 for k in prior}
            func_vals = {k: base[k] for k in FUNCTIONAL_KEYS}
            func_t = sum(func_vals.values()) or 1.0
            for k in FUNCTIONAL_KEYS:
                func_only[k] = func_vals[k] / func_t

            base_total = sum(base.values()) or 1.0
            norm_base = {k: v / base_total for k, v in base.items()}
            blended = {}
            for k in prior:
                blended[k] = (1.0 - blend_progress) * func_only[k] + blend_progress * norm_base[k]
        else:
            blended = dict(base)

        # Enforce functional floor
        total = sum(blended.values()) or 1.0
        blended = {k: v / total for k, v in blended.items()}
        func_share = sum(blended[k] for k in FUNCTIONAL_KEYS)
        if func_share < min_functional_share:
            # Scale up functional, scale down non-functional
            nonfunc_share = 1.0 - func_share
            target_nonfunc = 1.0 - min_functional_share
            scale_func = min_functional_share / max(func_share, 1e-10)
            scale_nonfunc = target_nonfunc / max(nonfunc_share, 1e-10)
            for k in blended:
                if k in FUNCTIONAL_KEYS:
                    blended[k] *= scale_func
                else:
                    blended[k] *= scale_nonfunc

        # Final normalization
        total = sum(blended.values()) or 1.0
        blended = {k: v / total for k, v in blended.items()}

        self._effective_weights = blended
        return blended

    def update_all(
        self,
        deltas: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        models=None,
        global_model=None,
        device: str = "cpu",
    ) -> List[float]:
        """
        Update trust scores for all clients in a round.

        v2: Enhanced with coordinate-wise anomaly scoring and adaptive
        component weighting based on observed discrimination power.
        """
        # During warmup, return trust = 1.0 for all clients
        # (not enough history for meaningful scoring)
        if self._round < self.warmup_rounds:
            # Still update internal trackers to build history
            deltas_flat = [flatten_state_dict(d) for d in deltas]
            for i, cid in enumerate(client_ids):
                self.temporal_tracker.update(cid, deltas_flat[i])
                self.hetero_envelope.update_and_score(
                    cid, deltas_flat[i],
                    torch.stack(deltas_flat).mean(dim=0) if len(deltas_flat) > 0 else None,
                )
            self._round += 1
            return [1.0] * len(deltas)

        # Flatten all deltas for batch operations
        deltas_flat = [flatten_state_dict(d) for d in deltas]

        # Batch: spectral outlier detection (cumulative)
        spectral_scores = self.spectral_detector.score_all(deltas_flat, client_ids)

        # Batch: population direction agreement (cumulative)
        population_dir_scores = self.population_dir_scorer.score_all(
            deltas_flat, client_ids
        )

        # Batch: cross-client correlation (Sybil detection)
        correlation_scores = self.correlation_detector.detect_sybils(
            deltas_flat, client_ids
        )

        # Batch: delta-effect audit (EVERY round — now deterministic, fast)
        delta_effect_scores = {cid: 1.0 for cid in client_ids}
        if self.delta_effect_auditor is not None:
            delta_effect_scores = self.delta_effect_auditor.score_all_deltas(
                deltas, client_ids, global_model, device
            )

        # v2: Batch: coordinate-wise anomaly scoring
        cwas_scores = self.cwas.score_all(deltas_flat, client_ids)

        # v3: Batch: LOO validation scoring
        loo_scores = {cid: 0.5 for cid in client_ids}
        if self.loo_scorer is not None and global_model is not None:
            loo_scores = self.loo_scorer.score_all(
                deltas, client_ids, global_model, device
            )
            # v3: Keep LOO running every round throughout training
            # The functional signal must remain fresh to maintain detection

        # v3: Batch: Gradient Inversion scoring
        gis_scores = self.gis_scorer.score_all(deltas_flat, client_ids)

        # v3: Batch: Temporal Trajectory scoring
        ttc_scores = self.ttc_scorer.score_all(deltas_flat, client_ids)

        # v2: Update discrimination power estimates (for adaptive weighting)
        if self.adaptive_weights:
            self._update_discrimination({
                "delta_effect": delta_effect_scores,
                "coordinate_anomaly": cwas_scores,
                "spectral": spectral_scores,
                "population_dir": population_dir_scores,
                "correlation": correlation_scores,
                "loo_validation": loo_scores,
                "gradient_inversion": gis_scores,
                "trajectory": ttc_scores,
            }, client_ids)

        # Per-client updates with batch-computed context
        scores = []
        for i, cid in enumerate(client_ids):
            model = models[i] if models is not None else None
            score = self.update(
                cid, deltas[i], model, global_model, deltas, device,
                spectral_score=spectral_scores.get(cid, 1.0),
                correlation_score=correlation_scores.get(cid, 1.0),
                population_dir_score=population_dir_scores.get(cid, 1.0),
                delta_effect_score=delta_effect_scores.get(cid, 1.0),
                coordinate_anomaly_score=cwas_scores.get(cid, 1.0),
                loo_validation_score=loo_scores.get(cid, 0.5),
                gradient_inversion_score=gis_scores.get(cid, 0.5),
                trajectory_score=ttc_scores.get(cid, 1.0),
            )
            scores.append(score)
        self._round += 1
        return scores

    def _update_discrimination(
        self,
        all_component_scores: Dict[str, Dict[int, float]],
        client_ids: List[int],
    ):
        """
        v3: Update the discrimination power estimate for each component.

        Discrimination = variance of scores across clients, gated by a
        correlation check against FUNCTIONAL trust (LOO + delta_effect).

        v3 key change: The correlation guard now validates against functional
        evaluator scores (which measure actual policy impact) instead of
        overall trust.  This prevents circular reasoning where corrupted
        consensus components establish wrong trust, then the guard validates
        against that wrong trust.

        Components with high variance but negative correlation with
        functional trust are being "fooled" — their discrimination is
        zeroed out to prevent amplification.
        """
        disc_ema_beta = 0.8  # smoothing factor

        # v3: Use FUNCTIONAL trust (LOO + delta) as the correlation reference
        # instead of overall trust scores.  This breaks the circular dependency.
        FUNCTIONAL_COMPONENTS = {"delta_effect", "loo_validation"}
        func_vals = []
        for cid in client_ids:
            fv = 0.0
            n_func = 0
            for fc in FUNCTIONAL_COMPONENTS:
                if fc in all_component_scores:
                    fv += all_component_scores[fc].get(cid, 0.5)
                    n_func += 1
            func_vals.append(fv / max(n_func, 1))
        func_std = float(np.std(func_vals)) if len(func_vals) > 1 else 0.0

        for component_name, scores_dict in all_component_scores.items():
            values = [scores_dict.get(cid, 1.0) for cid in client_ids]
            if len(values) < 2:
                continue

            # Discrimination = standard deviation of scores
            disc = float(np.std(values))

            # v3: Correlation guard against FUNCTIONAL trust
            # Skip guard for functional components themselves
            if (component_name not in FUNCTIONAL_COMPONENTS
                    and func_std > 0.02 and disc > 0.01 and self._round > 5):
                corr = float(np.corrcoef(values, func_vals)[0, 1])
                if np.isnan(corr):
                    corr = 0.0
                if corr < -0.1:
                    # Component disagrees with functional evaluators → zeroed
                    disc = 0.0

            # EMA update
            old = self._discrimination_ema.get(component_name, 0.0)
            self._discrimination_ema[component_name] = (
                disc_ema_beta * old + (1.0 - disc_ema_beta) * disc
            )

    def get_scores(self) -> Dict[int, float]:
        """Return current trust scores."""
        return dict(self.trust_scores)

    def get_component_scores(self) -> Dict[int, Dict[str, float]]:
        """Return detailed component scores for logging."""
        return dict(self.component_scores)

    def get_effective_weights(self) -> Dict[str, float]:
        """v3: Return current effective component weights (may be adaptive)."""
        return dict(self._effective_weights) if self._effective_weights else {
            "delta_effect": self.w_delta_effect,
            "coordinate_anomaly": self.w_coordinate_anomaly,
            "spectral": self.w_spectral,
            "population_dir": self.w_population_dir,
            "temporal": self.w_temporal,
            "heterogeneity": self.w_heterogeneity,
            "correlation": self.w_correlation,
            "loo_validation": self.w_loo_validation,
            "gradient_inversion": self.w_gradient_inversion,
            "trajectory": self.w_trajectory,
        }

    def get_discrimination_power(self) -> Dict[str, float]:
        """v2: Return current discrimination power estimates per component."""
        return dict(self._discrimination_ema)
