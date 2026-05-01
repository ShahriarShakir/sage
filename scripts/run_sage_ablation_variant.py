#!/usr/bin/env python3
"""
scripts/run_sage_ablation_variant.py — Run SAGE with individual signals disabled.

Ablation variants:
  sage_sybil:  Only sybil detection active (align=1.0, norm=1.0 always)
  sage_align:  Only alignment check active (sybil=1.0, norm=1.0 always)
  sage_norm:   Only norm outlier active (sybil=1.0, align=1.0 always)
  sage_full:   All signals active (reference, same as trust_weighted)

This patches frl.agg.sage at runtime to disable selected signals,
then delegates to train_mpe.py logic.
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=".*outside action space.*")
warnings.filterwarnings("ignore", message=".*clipping to space.*")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import frl.agg as agg_module
from frl.agg import flatten_state_dict, geometric_median, weighted_geometric_median
from typing import Dict, List, Optional


def make_sage_ablation(variant: str):
    """Create a SAGE variant with specific signals disabled."""

    def sage_ablation(
        deltas: List[Dict[str, torch.Tensor]],
        trust_scores: Optional[List[float]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        n = len(deltas)
        if n <= 1:
            return deltas[0] if n == 1 else {}
        if n == 2:
            return geometric_median(deltas)

        vectors = torch.stack([flatten_state_dict(d) for d in deltas], dim=0)
        norms = vectors.norm(dim=1).clamp(min=1e-10)
        normalized = vectors / norms.unsqueeze(1)
        S = normalized @ normalized.T
        median_norm = norms.median().item()

        sybil_ws = []
        raw_align_ws = []
        norm_ws = []
        median_sims = []

        for i in range(n):
            mask = torch.ones(n, dtype=torch.bool, device=vectors.device)
            mask[i] = False
            sims_i = S[i, mask]

            # Signal 1: Sybil
            max_sim = sims_i.max().item()
            if variant in ("sage_sybil", "sage_full") and max_sim > 0.9999:
                sybil_w = 0.01
            else:
                sybil_w = 1.0
            sybil_ws.append(sybil_w)

            # Signal 2: Alignment
            median_sim = sims_i.median().item()
            median_sims.append(median_sim)
            if variant in ("sage_align", "sage_full") and median_sim < 0:
                align_w = max(0.1, 1.0 + median_sim)
            else:
                align_w = 1.0
            raw_align_ws.append(align_w)

            # Signal 3: Norm
            norm_ratio = norms[i].item() / max(median_norm, 1e-10)
            if variant in ("sage_norm", "sage_full") and (norm_ratio > 2.0 or norm_ratio < 0.5):
                norm_w = 0.01
            else:
                norm_w = 1.0
            norm_ws.append(norm_w)

        # Alignment disable check (only when alignment is active)
        if variant in ("sage_align", "sage_full"):
            n_negative_median = sum(1 for m in median_sims if m < 0)
            if n_negative_median >= n / 2:
                align_ws = [1.0] * n
            else:
                align_ws = raw_align_ws
        else:
            align_ws = [1.0] * n

        weights = torch.ones(n, device=vectors.device)
        for i in range(n):
            weights[i] = sybil_ws[i] * align_ws[i] * norm_ws[i]

        w_min = weights.min().item()
        w_max = weights.max().item()
        if w_min > 0 and (w_max / w_min) < 1.05:
            return geometric_median(deltas)

        return weighted_geometric_median(deltas, weights)

    return sage_ablation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--attack", type=str, required=True)
    parser.add_argument("--byzantine_fraction", type=float, required=True)
    parser.add_argument("--variant", type=str, required=True,
                        choices=["sage_sybil", "sage_align", "sage_norm", "sage_full"])
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_frequency", type=int, default=5)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--log_suffix", type=str, default="")
    args = parser.parse_args()

    # Monkey-patch the aggregator registry
    ablation_fn = make_sage_ablation(args.variant)
    agg_module.AGGREGATORS["trust_weighted"] = ablation_fn

    # Now delegate to train_mpe
    sys.argv = [
        "train_mpe.py",
        "--scenario", args.scenario,
        "--attack", args.attack,
        "--byzantine_fraction", str(args.byzantine_fraction),
        "--aggregator", "trust_weighted",
        "--rounds", str(args.rounds),
        "--seed", str(args.seed),
        "--eval_frequency", str(args.eval_frequency),
        "--eval_episodes", str(args.eval_episodes),
        "--log_suffix", args.log_suffix,
    ]

    from scripts.train_mpe import main as train_main
    train_main()


if __name__ == "__main__":
    main()
