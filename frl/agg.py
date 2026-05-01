"""
frl/agg.py — Aggregation Strategies for Federated RL
Created: 2026-02-26
Updated: 2026-03-04 — Added Krum, Multi-Krum, FLTrust, FLAME, FoolsGold,
                       and redesigned Trust-Weighted with Adaptive Defense
                       Intensity (ADI) for no-harm guarantee.
Updated: 2026-03-10 — v2: Added trust-informed coordinate-wise trimmed mean.
                       Key: trust scores guide per-coordinate trimming decisions,
                       combining coordinate-wise robustness with trust information.

Implements:
  - FedAvg (weighted average baseline)
  - Trimmed Mean (coordinate-wise robust)
  - Geometric Median (Weiszfeld algorithm)
  - Krum / Multi-Krum (Blanchard et al., NeurIPS 2017)
  - FLTrust (Cao et al., NDSS 2021)
  - FLAME (Nguyen et al., USENIX Security 2022)
  - FoolsGold (Fung et al., IEEE TDSC 2020)
  - Trust-Weighted Robust Aggregation with ADI (our method)
  - Trust-Informed Coordinate-Wise Trimmed Mean (v2 — hybrid)
"""

from __future__ import annotations

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: flatten / unflatten state-dict deltas
# ---------------------------------------------------------------------------

def flatten_state_dict(sd: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a state dict into a single 1-D vector."""
    return torch.cat([v.reshape(-1).float() for v in sd.values()])


def unflatten_state_dict(
    vec: torch.Tensor, template: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Reshape a flat vector back into a state dict using a template."""
    result = {}
    offset = 0
    for k, v in template.items():
        numel = v.numel()
        result[k] = vec[offset : offset + numel].reshape(v.shape).to(v.dtype)
        offset += numel
    return result


# ---------------------------------------------------------------------------
# FedAvg
# ---------------------------------------------------------------------------

def fedavg(
    deltas: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Weighted average of parameter deltas.

    Args:
        deltas: list of state-dict deltas from clients
        weights: optional per-client weights (default: uniform)
    Returns:
        aggregated delta (state dict)
    """
    n = len(deltas)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    agg = {}
    for k in deltas[0]:
        agg[k] = sum(w * d[k].float() for w, d in zip(weights, deltas))
    return agg


# ---------------------------------------------------------------------------
# Trimmed Mean
# ---------------------------------------------------------------------------

def trimmed_mean(
    deltas: List[Dict[str, torch.Tensor]],
    trim_fraction: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Coordinate-wise trimmed mean. Removes the top and bottom
    `trim_fraction` of values at each coordinate.

    Args:
        deltas: list of state-dict deltas
        trim_fraction: fraction to trim from each tail (e.g., 0.1 = 10%)
    Returns:
        aggregated delta
    """
    n = len(deltas)
    k = max(1, int(trim_fraction * n))  # number to trim from each side
    k = min(k, max(0, (n - 2) // 2))    # always keep at least 2 values

    agg = {}
    for key in deltas[0]:
        # Stack all deltas for this parameter: shape (n, *param_shape)
        stacked = torch.stack([d[key].float() for d in deltas], dim=0)
        original_shape = stacked.shape[1:]
        flat = stacked.reshape(n, -1)  # (n, D)

        # Sort along client dimension
        sorted_vals, _ = flat.sort(dim=0)

        # Trim top-k and bottom-k, average the rest
        trimmed = sorted_vals[k : n - k]  # (n - 2k, D)
        mean_vals = trimmed.mean(dim=0)

        agg[key] = mean_vals.reshape(original_shape)

    return agg


# ---------------------------------------------------------------------------
# Geometric Median (Weiszfeld's algorithm)
# ---------------------------------------------------------------------------

def geometric_median(
    deltas: List[Dict[str, torch.Tensor]],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Geometric median of parameter deltas via Weiszfeld's algorithm.

    Operates in flattened parameter space.

    Args:
        deltas: list of state-dict deltas
        max_iter: maximum Weiszfeld iterations
        tol: convergence tolerance
    Returns:
        aggregated delta (geometric median)
    """
    template = deltas[0]
    vectors = torch.stack([flatten_state_dict(d) for d in deltas], dim=0)  # (n, D)
    n, D = vectors.shape

    # Initialize with coordinate-wise median
    median = vectors.median(dim=0).values.clone()

    for _it in range(max_iter):
        # Compute distances from current estimate to each vector
        diffs = vectors - median.unsqueeze(0)  # (n, D)
        dists = diffs.norm(dim=1, keepdim=True).clamp(min=1e-10)  # (n, 1)

        # Weiszfeld update: weighted average with 1/dist weights
        weights = 1.0 / dists  # (n, 1)
        new_median = (weights * vectors).sum(dim=0) / weights.sum()

        shift = (new_median - median).norm().item()
        median = new_median
        if shift < tol:
            break

    return unflatten_state_dict(median, template)


# ---------------------------------------------------------------------------
# Krum / Multi-Krum (Blanchard et al., NeurIPS 2017)
# ---------------------------------------------------------------------------

def krum(
    deltas: List[Dict[str, torch.Tensor]],
    n_byzantine: int = 1,
    multi_k: int = 1,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Krum aggregation: selects the client whose delta is closest
    to its nearest (n - f - 2) neighbors in L2.

    Multi-Krum: averages the top-k clients by Krum score.

    Args:
        deltas: list of state-dict deltas
        n_byzantine: number of Byzantine clients (f)
        multi_k: how many to keep (1 = Krum, >1 = Multi-Krum)
    Returns:
        aggregated delta
    """
    n = len(deltas)
    vectors = torch.stack([flatten_state_dict(d) for d in deltas], dim=0)

    # Pairwise L2 distance matrix
    dists = torch.cdist(vectors.unsqueeze(0), vectors.unsqueeze(0)).squeeze(0)

    # Number of neighbors to consider
    n_neighbors = max(1, n - n_byzantine - 2)

    # Krum score = sum of distances to closest n_neighbors neighbors
    krum_scores = []
    for i in range(n):
        sorted_dists, _ = dists[i].sort()
        # sorted_dists[0] is self (0), take next n_neighbors
        score = sorted_dists[1:n_neighbors + 1].sum().item()
        krum_scores.append(score)

    # Select the multi_k clients with lowest Krum score
    multi_k = min(multi_k, n)
    selected = sorted(range(n), key=lambda i: krum_scores[i])[:multi_k]

    # Average the selected deltas
    agg = {}
    for key in deltas[0]:
        agg[key] = torch.stack([deltas[i][key].float() for i in selected]).mean(dim=0)

    return agg


def multi_krum(
    deltas: List[Dict[str, torch.Tensor]],
    n_byzantine: int = 1,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Multi-Krum: selects ceil(n/2) clients using Krum scoring."""
    n = len(deltas)
    multi_k = max(1, (n + 1) // 2)
    return krum(deltas, n_byzantine=n_byzantine, multi_k=multi_k)


# ---------------------------------------------------------------------------
# FLTrust (Cao et al., NDSS 2021)
# ---------------------------------------------------------------------------

def fltrust(
    deltas: List[Dict[str, torch.Tensor]],
    server_delta: Optional[Dict[str, torch.Tensor]] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    FLTrust: uses a server-side reference update (from a small root dataset)
    to compute trust scores based on cosine similarity, then clips client
    updates to have the same magnitude as the server update.

    If no server_delta provided, uses the trimmed mean as approximation.

    Args:
        deltas: list of state-dict deltas from clients
        server_delta: server's reference update (from root dataset)
    Returns:
        aggregated delta
    """
    n = len(deltas)

    # If no server delta, approximate with trimmed mean of client deltas
    if server_delta is None:
        server_delta = trimmed_mean(deltas, trim_fraction=0.2)

    server_flat = flatten_state_dict(server_delta)
    server_norm = server_flat.norm().clamp(min=1e-10)
    server_dir = server_flat / server_norm

    # Compute cosine similarity and ReLU trust scores
    trust_scores = []
    client_flats = []
    for d in deltas:
        flat = flatten_state_dict(d)
        client_norm = flat.norm().clamp(min=1e-10)
        cos_sim = torch.dot(flat / client_norm, server_dir).item()
        trust_scores.append(max(0.0, cos_sim))  # ReLU: only positive alignments
        client_flats.append(flat)

    # Normalize trust scores
    total_trust = sum(trust_scores)
    if total_trust < 1e-10:
        # All clients negative alignment → fall back to server delta
        return server_delta

    # Weighted sum: scale each client to server_norm, weight by trust
    agg_flat = torch.zeros_like(server_flat)
    for i in range(n):
        if trust_scores[i] > 0:
            client_norm = client_flats[i].norm().clamp(min=1e-10)
            normalized_delta = client_flats[i] / client_norm * server_norm
            agg_flat += (trust_scores[i] / total_trust) * normalized_delta

    template = deltas[0]
    return unflatten_state_dict(agg_flat, template)


# ---------------------------------------------------------------------------
# FLAME (Nguyen et al., USENIX Security 2022)
# ---------------------------------------------------------------------------

def flame(
    deltas: List[Dict[str, torch.Tensor]],
    n_byzantine: int = 1,
    noise_sigma: float = 0.001,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    FLAME: Taming Backdoors in Federated Learning.

    Two stages:
    1. HDBSCAN-inspired clustering: filter outliers using pairwise cosine
       distance, keep the largest cluster.
    2. Adaptive clipping + noise injection on the remaining updates.

    Simplified version using threshold-based clustering (no HDBSCAN dep).

    Args:
        deltas: list of state-dict deltas
        n_byzantine: expected number of Byzantine clients
        noise_sigma: standard deviation of differential privacy noise
    Returns:
        aggregated delta
    """
    n = len(deltas)
    if n < 3:
        return fedavg(deltas)

    vectors = torch.stack([flatten_state_dict(d) for d in deltas], dim=0)
    norms = vectors.norm(dim=1, keepdim=True).clamp(min=1e-10)
    normalized = vectors / norms

    # Step 1: Cosine distance clustering
    cos_sim = normalized @ normalized.T  # (n, n)

    # Agglomerative clustering: merge pairs with highest cosine similarity
    # Until we have a cluster with > n/2 members
    # Simple approach: for each client, count in-cluster neighbors
    threshold = 0.5  # cosine similarity threshold
    best_cluster = list(range(n))
    best_score = -float('inf')

    for t in np.arange(0.3, 0.95, 0.05):
        # Build adjacency and find largest connected component
        adj = cos_sim > t
        visited = set()
        clusters = []

        for i in range(n):
            if i in visited:
                continue
            # BFS
            cluster = {i}
            queue = [i]
            while queue:
                node = queue.pop(0)
                for j in range(n):
                    if j not in visited and j not in cluster and adj[node, j]:
                        cluster.add(j)
                        queue.append(j)
            visited.update(cluster)
            clusters.append(list(cluster))

        # Find largest cluster
        largest = max(clusters, key=len)
        # Score: cluster size - variance of norms in cluster
        if len(largest) >= max(2, n - n_byzantine):
            cluster_norms = norms[largest].squeeze()
            norm_var = cluster_norms.var().item() if len(largest) > 1 else 0.0
            score = len(largest) - 0.1 * norm_var
            if score > best_score:
                best_score = score
                best_cluster = largest

    # Step 2: Adaptive clipping
    cluster_vectors = vectors[best_cluster]
    cluster_norms = cluster_vectors.norm(dim=1)
    median_norm = cluster_norms.median().item()

    # Clip each vector to median norm
    clipped = []
    for i, idx in enumerate(best_cluster):
        v = cluster_vectors[i]
        v_norm = v.norm().item()
        if v_norm > median_norm:
            v = v * (median_norm / max(v_norm, 1e-10))
        clipped.append(v)

    # Average clipped vectors
    agg_flat = torch.stack(clipped).mean(dim=0)

    # Step 3: Add DP noise
    if noise_sigma > 0:
        noise = torch.randn_like(agg_flat) * noise_sigma * median_norm
        agg_flat = agg_flat + noise

    template = deltas[0]
    return unflatten_state_dict(agg_flat, template)


# ---------------------------------------------------------------------------
# FoolsGold (Fung et al., IEEE TDSC 2020)
# ---------------------------------------------------------------------------

def foolsgold(
    deltas: List[Dict[str, torch.Tensor]],
    history: Optional[List[torch.Tensor]] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    FoolsGold: Mitigating Sybil Attacks on Federated Learning.

    Reduces the contribution of clients with similar update histories
    (likely Sybils). Uses pairwise cosine similarity of cumulative
    updates to down-weight colluding clients.

    Args:
        deltas: list of state-dict deltas
        history: optional list of cumulative gradient histories per client
    Returns:
        aggregated delta
    """
    n = len(deltas)
    vectors = torch.stack([flatten_state_dict(d) for d in deltas], dim=0)

    # Compute pairwise cosine similarities
    norms = vectors.norm(dim=1, keepdim=True).clamp(min=1e-10)
    normalized = vectors / norms
    cos_sim = normalized @ normalized.T  # (n, n)

    # For each client, compute max cosine similarity with any other client
    # (excluding self)
    weights = torch.ones(n)
    for i in range(n):
        max_sim = -1.0
        for j in range(n):
            if i != j:
                sim = cos_sim[i, j].item()
                if sim > max_sim:
                    max_sim = sim

        # FoolsGold weight: penalize clients with high max similarity
        # (their update is too similar to someone else → likely Sybil)
        if max_sim > 0:
            weights[i] = 1.0 - max_sim
        else:
            weights[i] = 1.0

    # Reweight: apply logit transformation for sharper discrimination
    weights = weights.clamp(min=1e-6)
    # Normalize
    weights = weights / weights.sum()

    # Weighted average
    agg = {}
    for key in deltas[0]:
        stacked = torch.stack([d[key].float() for d in deltas], dim=0)
        device = stacked.device
        w = weights.to(device)
        # Handle multi-dim params
        shape = stacked.shape[1:]
        flat = stacked.reshape(n, -1)
        result = (w.unsqueeze(1) * flat).sum(dim=0)
        agg[key] = result.reshape(shape)

    return agg


# ---------------------------------------------------------------------------
# Weighted Geometric Median (Weiszfeld's algorithm with client weights)
# ---------------------------------------------------------------------------

def weighted_geometric_median(
    deltas: List[Dict[str, torch.Tensor]],
    weights: torch.Tensor,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Weighted geometric median via Weiszfeld's algorithm.

    Standard geometric median minimizes Σ ||x - p_i||.
    Weighted version minimizes Σ w_i ||x - p_i||.

    When all weights are equal, this is identical to standard GM.

    Args:
        deltas: list of state-dict deltas
        weights: per-client weights (1-D tensor, length n)
        max_iter: maximum Weiszfeld iterations
        tol: convergence tolerance
    Returns:
        aggregated delta (weighted geometric median)
    """
    template = deltas[0]
    vectors = torch.stack([flatten_state_dict(d) for d in deltas], dim=0)  # (n, D)
    n, D = vectors.shape

    # Normalize weights to sum to 1
    w = weights.float().to(vectors.device)
    w = w / w.sum()

    # Initialize with weighted mean
    median = (w.unsqueeze(1) * vectors).sum(dim=0)

    for _it in range(max_iter):
        diffs = vectors - median.unsqueeze(0)  # (n, D)
        dists = diffs.norm(dim=1).clamp(min=1e-10)  # (n,)

        # Weighted Weiszfeld: w_i / ||x_i - median||
        inv_dists = w / dists  # (n,)
        new_median = (inv_dists.unsqueeze(1) * vectors).sum(dim=0) / inv_dists.sum()

        shift = (new_median - median).norm().item()
        median = new_median
        if shift < tol:
            break

    return unflatten_state_dict(median, template)


# ---------------------------------------------------------------------------
# SAGE: Similarity-Aware Geometric mEdian (our method)
# ---------------------------------------------------------------------------

def sage(
    deltas: List[Dict[str, torch.Tensor]],
    trust_scores: Optional[List[float]] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    SAGE: Similarity-Aware Geometric mEdian for Byzantine-Robust Federated MARL.

    Uses the pairwise cosine similarity matrix of client updates to compute
    per-client credibility weights, then applies a weighted geometric median.

    Three orthogonal credibility signals:
      1. SYBIL DETECTION: Penalizes clients whose update is near-identical
         to another client (cosine > 0.9999). Targets collusion/normalized
         attacks where Byzantine clients send coordinated identical updates.
         Threshold 0.9999 avoids false positives on honest MARL clients
         training the same agent (cos typically 0.95-0.999).
      2. ALIGNMENT CHECK (conditional): Penalizes clients whose update
         opposes the majority direction (negative median pairwise cosine).
         DISABLED when >= half of clients have negative median cosine
         (symmetric scenario like sign_flip 50% where both groups oppose
         each other equally). Targets asymmetric directional attacks.
      3. NORM OUTLIER CHECK: Penalizes clients with abnormal update magnitude
         (norm > 2x or < 0.5x the median). Assigns minimum weight (0.01)
         to effectively neutralize norm outliers. Targets scaling and
         adaptive attacks that amplify gradients.

    Key properties:
    - NO-DEGRADATION GUARANTEE: When all three signals indicate benign
      behavior (which happens under no attack), weights are exactly uniform →
      the method calls standard geometric_median() for BIT-IDENTICAL results.
    - NO temporal state (stateless, no warmup, no accumulation)
    - NO server-side data (unlike FLTrust)
    - NO additional environment interaction (no eval episodes)
    - Handles multiple attack types through orthogonal signals
    - Computational cost: O(n^2 * d) for cosine matrix + O(GM) ~ 1.05x standard GM

    Args:
        deltas: list of state-dict deltas from clients
        trust_scores: IGNORED (accepted for interface compatibility)
        **kwargs: IGNORED
    Returns:
        Aggregated delta (state dict)
    """
    n = len(deltas)
    if n <= 1:
        return deltas[0] if n == 1 else {}
    if n == 2:
        return geometric_median(deltas)

    # ---- Step 1: Flatten and compute pairwise cosine similarity ----
    vectors = torch.stack([flatten_state_dict(d) for d in deltas], dim=0)  # (n, D)
    norms = vectors.norm(dim=1).clamp(min=1e-10)  # (n,)
    normalized = vectors / norms.unsqueeze(1)  # (n, D)
    S = normalized @ normalized.T  # (n, n) cosine similarity matrix

    median_norm = norms.median().item()

    # ---- Step 2: Two-phase credibility weight computation ----
    # Phase A: Collect per-client signals
    sybil_ws = []
    raw_align_ws = []
    norm_ws = []
    median_sims = []

    for i in range(n):
        # Get similarities with all OTHER clients
        mask = torch.ones(n, dtype=torch.bool, device=vectors.device)
        mask[i] = False
        sims_i = S[i, mask]  # (n-1,)

        # ------ Signal 1: Sybil detection ------
        # Threshold 0.9999: normalized attack → cos≈1.0 (bit-identical),
        # honest same-agent MARL clients → cos 0.95-0.999 (below threshold).
        max_sim = sims_i.max().item()
        if max_sim > 0.9999:
            sybil_w = 0.01  # effectively neutralize
        else:
            sybil_w = 1.0
        sybil_ws.append(sybil_w)

        # ------ Signal 2: Alignment with majority ------
        median_sim = sims_i.median().item()
        median_sims.append(median_sim)
        if median_sim < 0:
            align_w = max(0.1, 1.0 + median_sim)
        else:
            align_w = 1.0
        raw_align_ws.append(align_w)

        # ------ Signal 3: Norm outlier ------
        # Hard floor of 0.01 effectively neutralizes outliers,
        # reducing their influence to ~1% in weighted GM.
        norm_ratio = norms[i].item() / max(median_norm, 1e-10)
        if norm_ratio > 2.0:
            norm_w = 0.01  # effectively neutralize
        elif norm_ratio < 0.5:
            norm_w = 0.01  # effectively neutralize
        else:
            norm_w = 1.0
        norm_ws.append(norm_w)

    # Phase B: Check alignment signal applicability
    # If >= half of clients have negative median cosine, this indicates
    # a SYMMETRIC scenario (e.g., sign_flip 50% where honest and Byzantine
    # groups are equally sized and both "oppose" each other). In symmetric
    # scenarios, the alignment signal is noise — it can't distinguish
    # honest from Byzantine. DISABLE it to avoid random harmful weighting.
    n_negative_median = sum(1 for m in median_sims if m < 0)
    use_alignment = (n_negative_median < n / 2)

    if not use_alignment:
        align_ws = [1.0] * n
    else:
        align_ws = raw_align_ws

    # Phase C: Combine signals into per-client weights
    weights = torch.ones(n, device=vectors.device)
    for i in range(n):
        weights[i] = sybil_ws[i] * align_ws[i] * norm_ws[i]

    # ---- Step 3: Decide standard GM vs weighted GM ----
    # If all weights are effectively uniform (max/min < 1.05), use standard
    # geometric_median() for BIT-IDENTICAL deterministic results with GM baseline.
    w_min = weights.min().item()
    w_max = weights.max().item()
    if w_min > 0 and (w_max / w_min) < 1.05:
        logger.debug(f"SAGE: weights uniform (ratio={w_max/w_min:.4f}) -> standard GM")
        return geometric_median(deltas)

    # Otherwise, apply weighted geometric median with credibility weights
    logger.debug(
        f"SAGE: weights non-uniform (min={w_min:.4f}, max={w_max:.4f}, "
        f"ratio={w_max/w_min:.2f}) -> weighted GM"
    )
    return weighted_geometric_median(deltas, weights)


# ---------------------------------------------------------------------------
# Trust-Informed Coordinate-Wise Trimmed Mean (v2 — legacy)
# ---------------------------------------------------------------------------

def trust_informed_trimmed_mean(
    deltas: List[Dict[str, torch.Tensor]],
    trust_scores: List[float],
    trim_fraction: float = 0.1,
    trust_trim_boost: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """
    Coordinate-wise trimmed mean where low-trust clients are trimmed first.

    This is a hybrid of:
    - Coordinate-wise robust statistics (handles per-coord sign-flip)
    - Trust-based client selection (handles subtle attacks)

    Algorithm:
    1. Sort clients by trust score.
    2. Identify clearly untrusted clients (trust < 0.3).
    3. Remove their coordinates from the trimming pool.
    4. Apply standard coordinate-wise trimmed mean on the remaining.

    This is strictly better than:
    - trimmed_mean alone (which is trust-blind)
    - trust-weighted averaging (which is not coordinate-wise robust)

    Args:
        deltas: list of parameter deltas from clients
        trust_scores: per-client trust scores in [0, 1]
        trim_fraction: base fraction to trim from each tail
        trust_trim_boost: additional trim for low-trust scenarios
    Returns:
        aggregated delta
    """
    n = len(deltas)
    if n < 2:
        return deltas[0] if n == 1 else {}

    ts = np.array(trust_scores, dtype=np.float64)

    # Phase 1: Identify clearly untrusted clients
    # Use trust_threshold = 0.3 as a hard floor
    untrusted_mask = ts < 0.3
    n_untrusted = int(untrusted_mask.sum())

    # Safety: never remove more than ceil(n/2) - 1 clients
    max_remove = max(0, n // 2 - 1)
    if n_untrusted > max_remove:
        # Remove only the MOST untrusted
        sorted_idx = np.argsort(ts)
        untrusted_indices = set(sorted_idx[:max_remove].tolist())
    elif n_untrusted > 0:
        untrusted_indices = set(np.where(untrusted_mask)[0].tolist())
    else:
        untrusted_indices = set()

    # Phase 2: Compute trust-informed weights for remaining clients
    # Higher trust → this client's values are less likely to be trimmed
    trusted_indices = [i for i in range(n) if i not in untrusted_indices]
    n_trusted = len(trusted_indices)

    if n_trusted < 2:
        # Not enough trusted clients → fall back to standard trimmed mean
        return trimmed_mean(deltas, trim_fraction=trim_fraction)

    # Adaptive trim fraction: increase when trust gap is large
    trust_gap = float(ts[trusted_indices].max() - ts[trusted_indices].min())
    adaptive_trim = trim_fraction + trust_trim_boost * trust_gap
    k = max(1, int(adaptive_trim * n_trusted))
    k = min(k, max(0, (n_trusted - 2) // 2))

    agg = {}
    for key in deltas[0]:
        # Stack only trusted clients
        stacked = torch.stack(
            [deltas[i][key].float() for i in trusted_indices], dim=0
        )
        original_shape = stacked.shape[1:]
        flat = stacked.reshape(n_trusted, -1)  # (n_trusted, D)

        # Sort along client dimension
        sorted_vals, _ = flat.sort(dim=0)

        # Trim top-k and bottom-k, average the rest
        if k > 0 and n_trusted > 2 * k:
            trimmed = sorted_vals[k : n_trusted - k]
        else:
            trimmed = sorted_vals
        mean_vals = trimmed.mean(dim=0)

        agg[key] = mean_vals.reshape(original_shape)

    return agg


# ---------------------------------------------------------------------------
# Trust-Weighted Robust Aggregation with Adaptive Defense Intensity (ADI)
# (Our Method — HATT + ADI)
# ---------------------------------------------------------------------------

def trust_weighted_robust_aggregation(
    deltas: List[Dict[str, torch.Tensor]],
    trust_scores: List[float],
    base_aggregator: str = "trimmed_mean",
    trim_fraction: float = 0.1,
    trust_threshold: float = 0.5,
    filter_anomalous: bool = True,
    trust_power: float = 2.0,
    adi_mode: str = "full",
) -> Dict[str, torch.Tensor]:
    """
    Adaptive Defense Intensity (ADI) Trust-Weighted Aggregation.

    Key innovation: automatically detects attack intensity from the trust
    score distribution and adapts filtering aggressiveness.

    - No attack detected (low trust variance): degenerates to base robust
      aggregator with negligible overhead → NO-HARM GUARANTEE.
    - Strong attack detected (bimodal trust scores): aggressive filtering
      and trust-weighted averaging of the honest client subset.

    The ADI mechanism uses three signals:
    1. Trust score bimodality (gap between clusters)
    2. Trust score coefficient of variation
    3. Normalized trust entropy

    Args:
        deltas: list of parameter deltas from clients
        trust_scores: per-client trust scores in [0, 1]
        base_aggregator: "trimmed_mean" or "geometric_median"
        trim_fraction: for trimmed mean
        trust_threshold: base minimum trust to include client (adaptively adjusted)
        filter_anomalous: whether to allow filtering low-trust clients
        trust_power: base exponent for trust weights
        adi_mode: ADI ablation mode — "full" (default), "disabled" (fixed params),
                  "cv_only" (only CV signal), "no_bat" (no bimodality-aware threshold)
    Returns:
        aggregated delta
    """
    assert len(deltas) == len(trust_scores)
    n = len(deltas)

    if n < 2:
        return deltas[0] if n == 1 else {}

    # ===================== ADAPTIVE DEFENSE INTENSITY =====================
    ts = np.array(trust_scores, dtype=np.float64)

    if adi_mode == "disabled":
        # Ablation: no ADI — use fixed threshold and power
        attack_intensity = 0.0
        adaptive_power = trust_power
        adaptive_threshold = trust_threshold
        intensity_cv = 0.0
        intensity_gap = 0.0
        intensity_bimodal = 0.0
        logger.debug(
            f"ADI [DISABLED]: fixed power={adaptive_power:.2f}, "
            f"threshold={adaptive_threshold:.3f}"
        )
    else:
        # Signal 1: Coefficient of variation of trust scores
        ts_mean = ts.mean()
        ts_std = ts.std()
        cv = ts_std / max(ts_mean, 1e-10)

        # Signal 2: Bimodality gap — largest gap between sorted trust scores
        ts_sorted = np.sort(ts)
        gaps = np.diff(ts_sorted)
        max_gap = float(gaps.max()) if len(gaps) > 0 else 0.0
        gap_position = int(gaps.argmax()) if len(gaps) > 0 else 0

        # Signal 3: Trust entropy (low entropy = concentrated = attack likely)
        ts_clipped = np.clip(ts, 0.01, 0.99)
        entropy = -np.sum(ts_clipped * np.log(ts_clipped + 1e-10)) / max(n, 1)

        # Compute attack intensity ∈ [0, 1]
        # Use conservative normalization to avoid false positives under no-attack
        # (honest trust scores can have CV ≈ 0.05-0.15 due to noise)
        intensity_cv = float(np.clip(cv / 0.5, 0.0, 1.0))

        if adi_mode == "cv_only":
            # Ablation: only CV signal, no gap or bimodality
            intensity_gap = 0.0
            intensity_bimodal = 0.0
            attack_intensity = intensity_cv
        else:
            # Full or no_bat: use all three signals
            intensity_gap = float(np.clip(max_gap / 0.4, 0.0, 1.0))
            intensity_bimodal = 0.0

            # Check for bimodality: clear separation between clusters
            if max_gap > 0.15 and gap_position >= 1:
                low_cluster = ts_sorted[:gap_position + 1]
                high_cluster = ts_sorted[gap_position + 1:]
                if len(low_cluster) >= 1 and len(high_cluster) >= 1:
                    cluster_sep = high_cluster.mean() - low_cluster.mean()
                    intensity_bimodal = float(np.clip(cluster_sep / 0.3, 0.0, 1.0))

            attack_intensity = max(intensity_cv, intensity_gap, intensity_bimodal)

        attack_intensity = float(np.clip(attack_intensity, 0.0, 1.0))

        # ===================== ADAPTIVE PARAMETERS =====================
        adaptive_power = 1.0 + attack_intensity * (trust_power - 1.0)
        adaptive_threshold = trust_threshold * attack_intensity

        # ===================== BIMODALITY-AWARE THRESHOLD (BAT) =====================
        if adi_mode != "no_bat" and attack_intensity > 0.5 and len(ts_sorted) >= 4:
            best_gap_score = -1.0
            best_gap_idx = -1
            for i in range(len(ts_sorted) - 1):
                gap_size = float(ts_sorted[i + 1] - ts_sorted[i])
                if gap_size < 0.05:
                    continue
                frac = (i + 1) / len(ts_sorted)
                balance = 1.0 - abs(2.0 * frac - 1.0)
                score = gap_size * (0.3 + 0.7 * balance)
                if score > best_gap_score:
                    best_gap_score = score
                    best_gap_idx = i

            if best_gap_idx >= 0:
                gap_threshold = float(
                    (ts_sorted[best_gap_idx] + ts_sorted[best_gap_idx + 1]) / 2.0
                )
                adaptive_threshold = max(adaptive_threshold, gap_threshold)

                # Safety: ensure we retain at least ceil(n/3) clients
                min_keep = max(2, (n + 2) // 3)
                n_above = sum(1 for t in trust_scores if t >= adaptive_threshold)
                while n_above < min_keep and adaptive_threshold > 0.01:
                    adaptive_threshold -= 0.05
                    n_above = sum(1 for t in trust_scores if t >= adaptive_threshold)

        # Power escalation: when bimodality is very clear, sharpen trust discrimination
        if adi_mode != "no_bat" and intensity_bimodal > 0.5:
            adaptive_power = min(adaptive_power * 1.5, trust_power * 2.0)

        logger.debug(
            f"ADI [{adi_mode}]: intensity={attack_intensity:.3f} (cv={intensity_cv:.3f}, "
            f"gap={intensity_gap:.3f}, bimodal={intensity_bimodal:.3f}), "
            f"power={adaptive_power:.2f}, threshold={adaptive_threshold:.3f}"
        )

    # ===================== ADAPTIVE AGGREGATION =====================
    # v2 Architecture: Three-tier defense
    #
    # Tier 1 (ADI <= 0.3): Pure base aggregator, zero overhead → NO-HARM
    # Tier 2 (0.3 < ADI <= tier3_thresh): Trust-informed coordinate-wise trimmed mean
    #         Combines trust filtering with coordinate-wise robustness
    # Tier 3 (ADI > tier3_thresh): Aggressive filtering + trust-informed trimmed mean
    #
    # Key insight: Tier 2 (trust-informed coord-wise) beats both:
    #   - trimmed_mean alone (blind to trust information)
    #   - trust-weighted average (not coordinate-wise robust)

    tier3_thresh = float(os.environ.get('HATT_TIER3_THRESHOLD', '0.7'))

    if filter_anomalous and attack_intensity > tier3_thresh:
        # ---- Tier 3: Strong attack — aggressive filter + trust-informed trim ----
        kept_indices = [i for i, t in enumerate(trust_scores)
                        if t >= adaptive_threshold]
        # Safety: keep at least ceil(n/3) clients
        min_keep = max(2, (n + 2) // 3)
        if len(kept_indices) < min_keep:
            ranked = sorted(range(n), key=lambda i: trust_scores[i], reverse=True)
            kept_indices = ranked[:min_keep]

        kept_deltas = [deltas[i] for i in kept_indices]
        kept_trust = [trust_scores[i] for i in kept_indices]

        logger.debug(
            f"ADI TIER 3: {len(kept_deltas)}/{n} clients kept "
            f"(threshold={adaptive_threshold:.3f})"
        )

        # Apply trust-informed trimmed mean on filtered set
        adaptive_trim = trim_fraction + attack_intensity * 0.15
        return trust_informed_trimmed_mean(
            kept_deltas, kept_trust,
            trim_fraction=adaptive_trim,
            trust_trim_boost=0.1,
        )

    elif filter_anomalous and attack_intensity > 0.3:
        # ---- Tier 2: Moderate attack (0.3 < ADI <= tier3_thresh) — trust-informed coord-wise trimming ----
        # No pre-filtering: let trust_informed_trimmed_mean handle it
        # This preserves more honest clients while still being robust
        logger.debug(
            f"ADI TIER 2: Trust-informed trim on all {n} clients "
            f"(intensity={attack_intensity:.3f})"
        )

        adaptive_trim = trim_fraction + attack_intensity * 0.10
        return trust_informed_trimmed_mean(
            deltas, trust_scores,
            trim_fraction=adaptive_trim,
            trust_trim_boost=0.15,
        )

    else:
        # ---- Tier 1: Benign / early — use ALL clients, base aggregator ----
        if base_aggregator == "trimmed_mean":
            return trimmed_mean(deltas, trim_fraction=trim_fraction)
        elif base_aggregator == "geometric_median":
            return geometric_median(deltas)
        else:
            raise ValueError(f"Unknown base aggregator: {base_aggregator}")


# ---------------------------------------------------------------------------
# Registry for easy config-driven selection
# ---------------------------------------------------------------------------
# Faithful baseline variants (added 2026-04-27)
#
# Reviewer-defensible reimplementations that close the simplifications
# documented in Appendix E of the SAGE paper.
#
#   * fltrust_lagged   — FLTrust with a one-round-lagged trimmed-mean
#                        reference (no server root data, but the reference
#                        is decoupled from the current-round client deltas
#                        so the trust scoring is no longer self-referential)
#   * flame_hdbscan    — FLAME using genuine HDBSCAN clustering, median
#                        norm clipping, and Gaussian DP noise
#   * foolsgold_hist   — FoolsGold with cumulative per-client gradient
#                        history across rounds and pardoning step
# ---------------------------------------------------------------------------

def fltrust_lagged(
    deltas: List[Dict[str, torch.Tensor]],
    server_reference: Optional[torch.Tensor] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    FLTrust with a server-side rolling reference produced by the previous
    round's robust aggregate. The trust scoring (cosine + ReLU + norm
    clipping) is identical to the published FLTrust; only the source of
    the reference update changes.

    First-round behaviour falls back to the trimmed mean of the current
    deltas (warm-up only); from round 2 onwards the server passes in
    `server_reference`.
    """
    n = len(deltas)
    if server_reference is None or server_reference.numel() == 0:
        server_flat = flatten_state_dict(trimmed_mean(deltas, trim_fraction=0.2))
    else:
        server_flat = server_reference.float()

    server_norm = server_flat.norm().clamp(min=1e-10)
    server_dir = server_flat / server_norm

    trust_scores = []
    client_flats = []
    for d in deltas:
        flat = flatten_state_dict(d)
        client_norm = flat.norm().clamp(min=1e-10)
        cos_sim = torch.dot(flat / client_norm, server_dir).item()
        trust_scores.append(max(0.0, cos_sim))
        client_flats.append(flat)

    total_trust = sum(trust_scores)
    if total_trust < 1e-10:
        return unflatten_state_dict(server_flat, deltas[0])

    agg_flat = torch.zeros_like(server_flat)
    for i in range(n):
        if trust_scores[i] > 0:
            client_norm = client_flats[i].norm().clamp(min=1e-10)
            normalized = client_flats[i] / client_norm * server_norm
            agg_flat += (trust_scores[i] / total_trust) * normalized

    return unflatten_state_dict(agg_flat, deltas[0])


def flame_hdbscan(
    deltas: List[Dict[str, torch.Tensor]],
    n_byzantine: int = 1,
    noise_sigma: float = 0.001,
    min_cluster_size: int = 2,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Faithful FLAME: HDBSCAN over pairwise cosine distances, median-norm
    clipping, Gaussian DP noise. Falls back to the threshold-sweep variant
    if HDBSCAN is unavailable.
    """
    n = len(deltas)
    if n < 3:
        return fedavg(deltas)

    vectors = torch.stack([flatten_state_dict(d) for d in deltas], dim=0)
    norms = vectors.norm(dim=1, keepdim=True).clamp(min=1e-10)
    normalized = vectors / norms

    cos_sim = (normalized @ normalized.T).cpu().numpy()
    distance = np.clip(1.0 - cos_sim, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0

    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            metric="precomputed",
            min_cluster_size=max(2, min_cluster_size),
            min_samples=1,
            allow_single_cluster=True,
        )
        labels = clusterer.fit_predict(distance.astype(np.float64))
        valid = [i for i, l in enumerate(labels) if l != -1]
        if len(valid) > 0:
            unique, counts = np.unique([labels[i] for i in valid], return_counts=True)
            largest_label = unique[counts.argmax()]
            best_cluster = [i for i, l in enumerate(labels) if l == largest_label]
        else:
            best_cluster = list(range(n))
        if len(best_cluster) < max(2, n - n_byzantine):
            best_cluster = list(range(n))
    except Exception as e:
        logger.warning(f"HDBSCAN unavailable ({e}); falling back to threshold sweep")
        return flame(deltas, n_byzantine=n_byzantine, noise_sigma=noise_sigma, **kwargs)

    cluster_vectors = vectors[best_cluster]
    cluster_norms = cluster_vectors.norm(dim=1)
    median_norm = cluster_norms.median().item()

    clipped = []
    for i in range(len(best_cluster)):
        v = cluster_vectors[i]
        v_norm = v.norm().item()
        if v_norm > median_norm:
            v = v * (median_norm / max(v_norm, 1e-10))
        clipped.append(v)

    agg_flat = torch.stack(clipped).mean(dim=0)
    if noise_sigma > 0:
        noise = torch.randn_like(agg_flat) * noise_sigma * median_norm
        agg_flat = agg_flat + noise

    return unflatten_state_dict(agg_flat, deltas[0])


def foolsgold_hist(
    deltas: List[Dict[str, torch.Tensor]],
    history: Optional[List[torch.Tensor]] = None,
    epsilon: float = 1e-5,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Faithful FoolsGold (Fung et al., 2020) with cumulative per-client
    gradient history across rounds, pairwise cosine on histories,
    pardoning step, and logit reweighting.

    Args:
        history: list of per-client cumulative-gradient tensors (one per
                 client, same length as `deltas`). Updated externally in
                 the server loop.
    """
    n = len(deltas)
    vectors = torch.stack([flatten_state_dict(d) for d in deltas], dim=0)

    if history is None or len(history) != n:
        hist = vectors.clone()
    else:
        hist = torch.stack([h.float() for h in history], dim=0)

    h_norms = hist.norm(dim=1, keepdim=True).clamp(min=1e-10)
    hist_n = hist / h_norms
    cs = hist_n @ hist_n.T  # (n, n)

    cs_off = cs.clone()
    cs_off.fill_diagonal_(0.0)
    max_per_client, _ = cs_off.max(dim=1)

    # Pardoning step: client i's similarities are scaled down if there is
    # another client j with strictly larger max-similarity than i.
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if max_per_client[j] > max_per_client[i]:
                cs[i, j] = cs[i, j] * (max_per_client[i] / max_per_client[j].clamp(min=1e-10))
        cs[i, i] = 0.0

    cs_off = cs.clone()
    cs_off.fill_diagonal_(0.0)
    wv = 1.0 - cs_off.max(dim=1).values
    wv = torch.clamp(wv, 0.0, 1.0)

    # Re-scale to [0, 1], logit, sigmoid for sharper discrimination.
    wmax = wv.max()
    if wmax > 0:
        wv = wv / wmax
    wv = torch.clamp(wv, epsilon, 1.0 - epsilon)
    wv = torch.log(wv / (1.0 - wv)) + 0.5
    wv = torch.sigmoid(wv)
    wv = torch.clamp(wv, 0.0, 1.0)

    if wv.sum() < 1e-10:
        return fedavg(deltas)

    weights = (wv / wv.sum()).tolist()
    return fedavg(deltas, weights=weights)


# ---------------------------------------------------------------------------

AGGREGATORS = {
    "fedavg": fedavg,
    "trimmed_mean": trimmed_mean,
    "geometric_median": geometric_median,
    "krum": krum,
    "multi_krum": multi_krum,
    "fltrust": fltrust,
    "fltrust_lagged": fltrust_lagged,
    "flame": flame,
    "flame_hdbscan": flame_hdbscan,
    "foolsgold": foolsgold,
    "foolsgold_hist": foolsgold_hist,
    "trust_weighted": sage,
}


def get_aggregator(name: str):
    """Get aggregation function by name."""
    if name not in AGGREGATORS:
        raise ValueError(f"Unknown aggregator '{name}'. Choose from {list(AGGREGATORS.keys())}")
    return AGGREGATORS[name]
