#!/usr/bin/env python3
"""
scripts/collect_learning_curves.py — Extract per-round eval returns from existing logs.

Produces a CSV with columns: method, environment, attack, seed, round, eval_return_mean, eval_return_std

This reads metrics.json from each experiment log directory.
"""

import json
import os
import sys
import csv
import re
from pathlib import Path

LOG_DIR = Path("logs")
OUTPUT = Path("plots/learning_curves.csv")

# Map aggregator names in directory to display names
AGG_MAP = {
    "fedavg": "FedAvg",
    "trimmed_mean": "TrimmedMean",
    "geometric_median": "GeoMedian",
    "krum": "Krum",
    "multi_krum": "MultiKrum",
    "fltrust": "FLTrust",
    "flame": "FLAME",
    "foolsgold": "FoolsGold",
    "trust_weighted": "SAGE",
}

# Map scenario names
ENV_MAP = {
    "simple_spread_v3": "Spread",
    "simple_adversary_v3": "Adversary",
    "simple_tag_v3": "Tag",
}

ATTACK_MAP = {
    "none": "none",
    "sign_flip": "sign_flip",
    "normalized": "normalized",
    "adaptive_strategic": "adaptive",
}

SEEDS = [42, 123, 456, 789, 1024]
METHODS_OF_INTEREST = list(AGG_MAP.keys())


def parse_log_dir_name(dirname):
    """Parse experiment directory name to extract components."""
    # Pattern: {scenario}_{aggregator}_{attack}_n{N}_b{B}_r{R}_s{seed}{suffix}
    # Examples:
    #   simple_spread_v3_fedavg_none_n6_b0_r200_s42
    #   simple_spread_v3_trust_weighted_sign_flip_n6_b3_r200_s42sage
    #   simple_adversary_v3_trust_weighted_none_n6_b0_r200_s42sage

    for env_key in ["simple_adversary_v3", "simple_spread_v3", "simple_tag_v3"]:
        if dirname.startswith(env_key + "_"):
            rest = dirname[len(env_key) + 1:]
            break
    else:
        return None

    for agg_key in sorted(AGG_MAP.keys(), key=len, reverse=True):
        if rest.startswith(agg_key + "_"):
            rest2 = rest[len(agg_key) + 1:]
            break
    else:
        return None

    # Extract attack and parameters
    # rest2 might be: none_n6_b0_r200_s42sage or sign_flip_n6_b3_r200_s42sage
    for atk_key in sorted(ATTACK_MAP.keys(), key=len, reverse=True):
        if rest2.startswith(atk_key + "_"):
            rest3 = rest2[len(atk_key) + 1:]
            break
    else:
        return None

    # Extract seed: _s{seed}
    m = re.search(r'_s(\d+)', rest3)
    if not m:
        return None
    seed = int(m.group(1))

    # Check for suffix patterns to identify SAGE vs HATT vs baselines
    suffix = rest3[m.end():]

    # Only keep SAGE runs (suffix "sage" or empty for baselines)
    if agg_key == "trust_weighted":
        # Must have "sage" suffix (not "_sage_v1" which is old)
        if "sage" not in suffix or "_v1" in suffix:
            return None
        method = "SAGE"
    else:
        # Baseline methods - no sage suffix
        if "sage" in suffix or "ablation" in suffix:
            return None
        method = AGG_MAP[agg_key]

    # Determine HATT from trust_weighted without sage suffix
    # (we already filtered those out above)

    return {
        "method": method,
        "environment": ENV_MAP[env_key],
        "env_key": env_key,
        "attack": ATTACK_MAP[atk_key],
        "seed": seed,
    }


def main():
    rows = []
    found_dirs = 0
    parsed_dirs = 0

    for entry in sorted(LOG_DIR.iterdir()):
        if not entry.is_dir():
            continue
        metrics_file = entry / "metrics.json"
        if not metrics_file.exists():
            continue

        found_dirs += 1
        info = parse_log_dir_name(entry.name)
        if info is None:
            continue
        if info["seed"] not in SEEDS:
            continue

        parsed_dirs += 1

        with open(metrics_file) as f:
            metrics = json.load(f)

        for m in metrics:
            if "eval_return_mean" in m:
                rows.append({
                    "method": info["method"],
                    "environment": info["environment"],
                    "attack": info["attack"],
                    "seed": info["seed"],
                    "round": m["round"],
                    "eval_return_mean": m["eval_return_mean"],
                    "eval_return_std": m.get("eval_return_std", 0.0),
                })

    # Also look for HATT logs
    for entry in sorted(LOG_DIR.iterdir()):
        if not entry.is_dir():
            continue
        metrics_file = entry / "metrics.json"
        if not metrics_file.exists():
            continue

        dirname = entry.name

        # HATT pattern: trust_weighted without "sage" suffix
        for env_key in ["simple_adversary_v3", "simple_spread_v3", "simple_tag_v3"]:
            if dirname.startswith(env_key + "_trust_weighted_"):
                rest = dirname[len(env_key) + len("_trust_weighted_"):]
                for atk_key in sorted(ATTACK_MAP.keys(), key=len, reverse=True):
                    if rest.startswith(atk_key + "_"):
                        rest3 = rest[len(atk_key) + 1:]
                        m_match = re.search(r'_s(\d+)', rest3)
                        if m_match:
                            seed_val = int(m_match.group(1))
                            suffix = rest3[m_match.end():]
                            # HATT: no suffix or non-sage suffix
                            if "sage" not in suffix and seed_val in SEEDS:
                                with open(metrics_file) as f:
                                    hatt_metrics = json.load(f)
                                for mm in hatt_metrics:
                                    if "eval_return_mean" in mm:
                                        rows.append({
                                            "method": "HATT",
                                            "environment": ENV_MAP[env_key],
                                            "attack": ATTACK_MAP[atk_key],
                                            "seed": seed_val,
                                            "round": mm["round"],
                                            "eval_return_mean": mm["eval_return_mean"],
                                            "eval_return_std": mm.get("eval_return_std", 0.0),
                                        })
                        break
                break

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "environment", "attack", "seed", "round",
            "eval_return_mean", "eval_return_std"
        ])
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    methods = sorted(set(r["method"] for r in rows))
    print(f"Scanned {found_dirs} directories, parsed {parsed_dirs}")
    print(f"Total data points: {len(rows)}")
    print(f"Methods found: {methods}")
    for method in methods:
        n = sum(1 for r in rows if r["method"] == method)
        print(f"  {method}: {n} points")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
