#!/usr/bin/env python3
"""
scripts/run_faithful_baselines.py — Re-runs only the three faithful baseline
columns (FLTrust-lagged, FLAME-HDBSCAN, FoolsGold-history) against the
existing 5-seed × 5-attack × 3-env grid. Used to populate App E faithful-
baseline comparison.

  experiments = 3 envs × 3 baselines × 5 attacks × 5 seeds = 225 expts
  ~5 min/expt → ~19 GPU-hours on a single RTX 5000 Ada
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

FAITHFUL_AGGREGATORS = ["fltrust_lagged", "flame_hdbscan", "foolsgold_hist"]

ATTACK_SCENARIOS = [
    ("none",              0.0,   1.0,  "no_attack"),
    ("sign_flip",         0.17,  1.0,  "17pct_signflip"),
    ("sign_flip",         0.5,   1.0,  "50pct_signflip"),
    ("normalized",        0.5,   1.0,  "50pct_normalized"),
    ("adaptive_strategic", 0.5,  1.0,  "50pct_adaptive"),
]

SEEDS = [42, 123, 456, 789, 1024]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--envs", nargs="+", default=ENVIRONMENTS)
    parser.add_argument("--aggregators", nargs="+", default=FAITHFUL_AGGREGATORS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()

    experiments = list(product(
        args.envs, args.aggregators, ATTACK_SCENARIOS, args.seeds
    ))
    total = len(experiments)
    print(f"Faithful-baseline relaunch: {total} experiments "
          f"({len(args.envs)} envs × {len(args.aggregators)} aggs × "
          f"{len(ATTACK_SCENARIOS)} attacks × {len(args.seeds)} seeds)")

    completed = skipped = failed = 0
    t_start = time.time()

    for i, (env, agg, (atk, byz, atk_scale, atk_lbl), seed) in enumerate(experiments, 1):
        log_dir = get_log_dir(env, agg, atk, byz, seed)
        if args.resume and is_done(log_dir):
            skipped += 1
            print(f"[{i:3d}/{total}] SKIP {log_dir.name}")
            continue
        if args.dry_run:
            print(f"[{i:3d}/{total}] DRY  {log_dir.name}")
            continue

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
