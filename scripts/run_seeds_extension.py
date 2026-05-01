#!/usr/bin/env python3
"""
scripts/run_seeds_extension.py — Extend the main n=6 grid by 5 additional
seeds (2024, 4096, 8192, 16384, 32768) on the top-3 baselines + SAGE
to double statistical power for Wilcoxon/Friedman.

Default: 4 methods × 3 envs × 5 attacks × 5 seeds = 300 expts (~25 GPU-h).
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

DEFAULT_METHODS = ["trust_weighted", "krum", "trimmed_mean", "fltrust_lagged"]

ATTACK_SCENARIOS = [
    ("none",              0.0,   1.0,  "no_attack"),
    ("sign_flip",         0.17,  1.0,  "17pct_signflip"),
    ("sign_flip",         0.5,   1.0,  "50pct_signflip"),
    ("normalized",        0.5,   1.0,  "50pct_normalized"),
    ("adaptive_strategic", 0.5,  1.0,  "50pct_adaptive"),
]

EXTRA_SEEDS = [2024, 4096, 8192, 16384, 32768]
ROUNDS = 200


def get_log_dir(env, agg, attack, byz_frac, seed, n_clients=6):
    n_byz = max(1, int(byz_frac * n_clients)) if byz_frac > 0 else 0
    return ROOT / "logs" / f"{env}_{agg}_{attack}_n{n_clients}_b{n_byz}_r{ROUNDS}_s{seed}"


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
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--envs", nargs="+", default=ENVIRONMENTS)
    p.add_argument("--seeds", nargs="+", type=int, default=EXTRA_SEEDS)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--timeout", type=int, default=15)
    args = p.parse_args()

    experiments = list(product(args.envs, args.methods, ATTACK_SCENARIOS, args.seeds))
    print(f"Seed-extension: {len(experiments)} experiments")
    completed = skipped = failed = 0
    t_start = time.time()

    for i, (env, agg, (atk, byz, atk_scale, atk_lbl), seed) in enumerate(experiments, 1):
        log_dir = get_log_dir(env, agg, atk, byz, seed)
        if args.resume and is_done(log_dir):
            skipped += 1; continue
        if args.dry_run:
            print(f"[{i:3d}] DRY {log_dir.name}"); continue
        cmd = [
            sys.executable, "scripts/train_mpe.py",
            "--scenario", env,
            "--n_clients", "6",
            "--byzantine_fraction", str(byz),
            "--attack", atk,
            "--attack_scale", str(atk_scale),
            "--aggregator", agg,
            "--seed", str(seed),
            "--rounds", str(ROUNDS),
            "--use_gpu",
        ]
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "run.log"
        t0 = time.time()
        try:
            with open(log_file, "w") as f:
                proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                       cwd=str(ROOT), timeout=args.timeout * 60)
            if proc.returncode == 0:
                completed += 1
                print(f"[{i:3d}] OK   {log_dir.name} ({time.time()-t0:.0f}s)")
            else:
                failed += 1
                print(f"[{i:3d}] FAIL {log_dir.name}")
        except subprocess.TimeoutExpired:
            failed += 1
            print(f"[{i:3d}] TMOT {log_dir.name}")

    print(f"\nDone: {completed} OK, {skipped} skipped, {failed} failed in "
          f"{(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
