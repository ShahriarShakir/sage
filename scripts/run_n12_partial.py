#!/usr/bin/env python3
"""
scripts/run_n12_partial.py — Partial sweep at n=12 clients to test scale
robustness of SAGE and key baselines. With n=12 we keep byzantine fraction
at 50% (6 of 12) but the ratio of honest signal to attacker noise improves
modestly; this matches App F's scale claim.

Default partial grid: 3 envs × 3 attacks × 3 seeds × 5 methods = 135 expts
At ~6 min/expt with n=12 (rollout cost grows linearly) → ~13.5 GPU-hours.

Use --full for 5 attacks × 5 seeds × all 11 methods → ~63 GPU-hours.
"""

import argparse
import os
import subprocess
import sys
import time
import json
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parent.parent

ENVIRONMENTS = ["simple_spread_v3", "simple_adversary_v3", "simple_tag_v3"]

PARTIAL_AGGREGATORS = ["trust_weighted", "krum", "trimmed_mean",
                       "fltrust_lagged", "fedavg"]
FULL_AGGREGATORS = ["fedavg", "trimmed_mean", "geometric_median",
                    "krum", "multi_krum", "fltrust", "fltrust_lagged",
                    "flame", "flame_hdbscan",
                    "foolsgold", "foolsgold_hist", "trust_weighted"]

PARTIAL_ATTACKS = [
    ("none",              0.0,   1.0,  "no_attack"),
    ("sign_flip",         0.5,   1.0,  "50pct_signflip"),
    ("adaptive_strategic", 0.5,  1.0,  "50pct_adaptive"),
]
FULL_ATTACKS = [
    ("none",              0.0,   1.0,  "no_attack"),
    ("sign_flip",         0.17,  1.0,  "17pct_signflip"),
    ("sign_flip",         0.5,   1.0,  "50pct_signflip"),
    ("normalized",        0.5,   1.0,  "50pct_normalized"),
    ("adaptive_strategic", 0.5,  1.0,  "50pct_adaptive"),
]

PARTIAL_SEEDS = [42, 123, 456]
FULL_SEEDS = [42, 123, 456, 789, 1024]

ROUNDS = 200
N_CLIENTS = 12


def get_log_dir(env, agg, attack, byz_frac, seed, n_clients=N_CLIENTS):
    n_byz = max(1, int(byz_frac * n_clients)) if byz_frac > 0 else 0
    return ROOT / "logs" / f"{env}_{agg}_{attack}_n{n_clients}_b{n_byz}_r{ROUNDS}_s{seed}_n12"


def is_done(log_dir):
    metrics = log_dir / "metrics.json"
    if not metrics.exists():
        return False
    try:
        with open(metrics) as f:
            data = json.load(f)
        return len(data) >= int(ROUNDS * 0.9)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--envs", nargs="+", default=ENVIRONMENTS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    aggs    = FULL_AGGREGATORS if args.full else PARTIAL_AGGREGATORS
    attacks = FULL_ATTACKS    if args.full else PARTIAL_ATTACKS
    seeds   = args.seeds or (FULL_SEEDS if args.full else PARTIAL_SEEDS)

    experiments = list(product(args.envs, aggs, attacks, seeds))
    total = len(experiments)
    print(f"n=12 sweep: {total} experiments")

    completed = skipped = failed = 0
    t_start = time.time()

    for i, (env, agg, (atk, byz, atk_scale, atk_lbl), seed) in enumerate(experiments, 1):
        log_dir = get_log_dir(env, agg, atk, byz, seed)
        if args.resume and is_done(log_dir):
            skipped += 1; continue
        if args.dry_run:
            print(f"[{i:3d}/{total}] DRY {log_dir.name}"); continue

        cmd = [
            sys.executable, "scripts/train_mpe.py",
            "--scenario", env,
            "--n_clients", str(N_CLIENTS),
            "--byzantine_fraction", str(byz),
            "--attack", atk,
            "--attack_scale", str(atk_scale),
            "--aggregator", agg,
            "--seed", str(seed),
            "--rounds", str(ROUNDS),
            "--log_suffix", "_n12",
            "--use_gpu",
        ]
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "run.log"
        t0 = time.time()
        try:
            with open(log_file, "w") as f:
                proc = subprocess.run(
                    cmd, stdout=f, stderr=subprocess.STDOUT,
                    cwd=str(ROOT), timeout=args.timeout * 60,
                )
            if proc.returncode == 0:
                completed += 1
                print(f"[{i:3d}/{total}] OK   {log_dir.name} ({time.time()-t0:.0f}s)")
            else:
                failed += 1
                print(f"[{i:3d}/{total}] FAIL {log_dir.name}")
        except subprocess.TimeoutExpired:
            failed += 1
            print(f"[{i:3d}/{total}] TMOT {log_dir.name}")

    print(f"\nDone: {completed} OK, {skipped} skipped, {failed} failed in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
