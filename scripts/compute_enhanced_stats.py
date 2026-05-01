#!/usr/bin/env python3
"""
scripts/compute_enhanced_stats.py — Enhanced statistics for the reported experiments.

Produces:
1. Holm-corrected pairwise Wilcoxon signed-rank tests (SAGE vs each baseline)
2. Cliff's delta effect sizes
3. Bootstrap 95% CI on mean ranks
4. Per-environment performance summary

Reads: plots/neurips_results_all.csv
Outputs: plots/enhanced_stats.json, prints tables
"""

import csv
import json
import itertools
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Statistics helpers (pure numpy, no scipy dependency issues)
# ---------------------------------------------------------------------------

def wilcoxon_signed_rank_test(x, y):
    """
    Wilcoxon signed-rank test (two-sided).
    Returns (statistic, p_value).
    """
    from scipy.stats import wilcoxon
    d = np.array(x) - np.array(y)
    d = d[d != 0]
    if len(d) < 5:
        return np.nan, 1.0
    stat, p = wilcoxon(d, alternative='two-sided')
    return stat, p


def cliffs_delta(x, y):
    """
    Cliff's delta effect size.
    Returns (delta, interpretation).
    """
    x, y = np.array(x), np.array(y)
    n_x, n_y = len(x), len(y)
    more = sum(1 for xi in x for yj in y if xi > yj)
    less = sum(1 for xi in x for yj in y if xi < yj)
    delta = (more - less) / (n_x * n_y)

    abs_d = abs(delta)
    if abs_d < 0.147:
        interp = "negligible"
    elif abs_d < 0.33:
        interp = "small"
    elif abs_d < 0.474:
        interp = "medium"
    else:
        interp = "large"

    return delta, interp


def bootstrap_ci(data, stat_func=np.mean, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    data = np.array(data)
    boot_stats = np.array([
        stat_func(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_stats, alpha * 100)
    hi = np.percentile(boot_stats, (1 - alpha) * 100)
    return float(lo), float(hi)


def holm_bonferroni(p_values):
    """
    Holm-Bonferroni correction for multiple comparisons.
    Returns adjusted p-values.
    """
    n = len(p_values)
    indices = list(range(n))
    sorted_indices = sorted(indices, key=lambda i: p_values[i])
    adjusted = [0.0] * n
    for rank, idx in enumerate(sorted_indices):
        adjusted[idx] = min(1.0, p_values[idx] * (n - rank))
    # Enforce monotonicity
    running_max = 0.0
    for idx in sorted_indices:
        adjusted[idx] = max(adjusted[idx], running_max)
        running_max = adjusted[idx]
    return adjusted


def benjamini_hochberg(p_values):
    """
    Benjamini-Hochberg (1995) step-up FDR correction.
    Returns BH-adjusted p-values (q-values).
    """
    n = len(p_values)
    if n == 0:
        return []
    indices = list(range(n))
    sorted_indices = sorted(indices, key=lambda i: p_values[i])
    adjusted = [0.0] * n
    # walk in reverse to enforce monotonicity (running min from largest k)
    running_min = 1.0
    for rank in range(n - 1, -1, -1):
        idx = sorted_indices[rank]
        q = p_values[idx] * n / (rank + 1)
        q = min(q, 1.0)
        running_min = min(running_min, q)
        adjusted[idx] = running_min
    return adjusted


def friedman_test(data_matrix):
    """
    Friedman test on a (scenarios x methods) matrix of performance values.
    Returns (chi2, p_value).
    """
    from scipy.stats import friedmanchisquare
    # data_matrix: list of lists, each inner list = performances of all methods for one scenario
    columns = list(zip(*data_matrix))  # transpose to get per-method arrays
    if len(columns) < 3:
        return np.nan, 1.0
    stat, p = friedmanchisquare(*columns)
    return stat, p


def compute_ranks(scores_matrix):
    """
    Compute ranks per scenario (row). Lower rank = better.
    For eval_return higher is better, so rank in descending order.
    """
    from scipy.stats import rankdata
    ranks = []
    for row in scores_matrix:
        # Higher is better -> negate for ranking
        r = rankdata(-np.array(row))
        ranks.append(r)
    return np.array(ranks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DATA_FILE = Path("plots/neurips_results_all.csv")
    OUTPUT_FILE = Path("plots/enhanced_stats.json")

    # Load data
    rows = []
    with open(DATA_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Build per-scenario-seed performance dict
    # Key: (environment, attack, byz_fraction)
    # Value: {aggregator: {seed: eval_return_mean}}
    data = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        key = (row["environment"], row["attack"], row["byz_fraction"])
        agg = row["aggregator"]
        seed = int(row["seed"])
        val = float(row["eval_return_mean"])
        data[key][agg][seed] = val

    # Get all scenarios and aggregators
    scenarios = sorted(data.keys())
    all_aggs = sorted(set(agg for sc in data.values() for agg in sc.keys()))
    seeds = [42, 123, 456, 789, 1024]

    print(f"Scenarios: {len(scenarios)}")
    print(f"Aggregators: {all_aggs}")
    print(f"Seeds: {seeds}")
    print()

    # ---------------------------------------------------------------
    # 1. Compute per-scenario mean performance (averaged over seeds)
    # ---------------------------------------------------------------
    scenario_means = {}  # {scenario: {agg: mean}}
    for sc in scenarios:
        scenario_means[sc] = {}
        for agg in all_aggs:
            if agg in data[sc]:
                vals = [data[sc][agg].get(s, np.nan) for s in seeds]
                vals = [v for v in vals if not np.isnan(v)]
                scenario_means[sc][agg] = np.mean(vals) if vals else np.nan
            else:
                scenario_means[sc][agg] = np.nan

    # ---------------------------------------------------------------
    # 2. Compute ranks
    # ---------------------------------------------------------------
    # For each scenario, rank methods by mean performance (higher=better, lower rank)
    rank_matrix = []
    valid_scenarios = []
    for sc in scenarios:
        row_vals = [scenario_means[sc].get(agg, np.nan) for agg in all_aggs]
        if any(np.isnan(v) for v in row_vals):
            continue
        valid_scenarios.append(sc)
        rank_matrix.append(row_vals)

    if rank_matrix:
        rank_matrix_np = compute_ranks(rank_matrix)
        mean_ranks = rank_matrix_np.mean(axis=0)

        print("=" * 60)
        print("MEAN RANKS (lower = better)")
        print("=" * 60)
        rank_results = {}
        for i, agg in enumerate(all_aggs):
            lo, hi = bootstrap_ci(rank_matrix_np[:, i])
            rank_results[agg] = {
                "mean_rank": float(mean_ranks[i]),
                "bootstrap_ci_95": [lo, hi],
                "n_scenarios": len(valid_scenarios),
            }
            print(f"  {agg:25s}: {mean_ranks[i]:.2f}  [{lo:.2f}, {hi:.2f}]")

        # Friedman test
        chi2, friedman_p = friedman_test(rank_matrix)
        print(f"\nFriedman χ² = {chi2:.2f}, p = {friedman_p:.6f}")
    else:
        print("No complete scenarios found!")
        rank_results = {}
        friedman_p = 1.0

    # ---------------------------------------------------------------
    # 3. Pairwise Wilcoxon: SAGE vs each baseline
    # ---------------------------------------------------------------
    sage_key = "sage"
    if sage_key not in all_aggs:
        print("WARNING: sage not found in aggregators!")
        sage_key = None

    pairwise_results = {}
    if sage_key:
        baselines = [a for a in all_aggs if a != sage_key]
        raw_pvals = []

        print("\n" + "=" * 60)
        print("PAIRWISE WILCOXON SIGNED-RANK (SAGE vs each baseline)")
        print("=" * 60)

        for baseline in baselines:
            # Collect paired observations: mean over seeds for each scenario
            sage_vals = []
            base_vals = []
            for sc in valid_scenarios:
                sv = scenario_means[sc].get(sage_key, np.nan)
                bv = scenario_means[sc].get(baseline, np.nan)
                if not np.isnan(sv) and not np.isnan(bv):
                    sage_vals.append(sv)
                    base_vals.append(bv)

            if len(sage_vals) < 5:
                raw_pvals.append(1.0)
                pairwise_results[baseline] = {
                    "n_pairs": len(sage_vals),
                    "note": "insufficient pairs"
                }
                continue

            stat, p = wilcoxon_signed_rank_test(sage_vals, base_vals)
            delta, interp = cliffs_delta(sage_vals, base_vals)

            raw_pvals.append(p)
            pairwise_results[baseline] = {
                "n_pairs": len(sage_vals),
                "wilcoxon_stat": float(stat) if not np.isnan(stat) else None,
                "p_raw": float(p),
                "cliffs_delta": float(delta),
                "effect_size": interp,
                "sage_wins": int(sum(1 for s, b in zip(sage_vals, base_vals) if s > b)),
                "ties": int(sum(1 for s, b in zip(sage_vals, base_vals) if s == b)),
                "baseline_wins": int(sum(1 for s, b in zip(sage_vals, base_vals) if s < b)),
            }

        # Holm-Bonferroni correction
        adj_pvals = holm_bonferroni(raw_pvals)
        # Benjamini-Hochberg FDR (added 2026-04-27 for reviewer power study)
        bh_pvals = benjamini_hochberg(raw_pvals)
        for i, baseline in enumerate(baselines):
            if baseline in pairwise_results and "p_raw" in pairwise_results[baseline]:
                pairwise_results[baseline]["p_holm"] = float(adj_pvals[i])
                pairwise_results[baseline]["p_bh"] = float(bh_pvals[i])
                sig = "***" if adj_pvals[i] < 0.001 else "**" if adj_pvals[i] < 0.01 else "*" if adj_pvals[i] < 0.05 else "n.s."
                sig_bh = "***" if bh_pvals[i] < 0.001 else "**" if bh_pvals[i] < 0.01 else "*" if bh_pvals[i] < 0.05 else "n.s."
                pairwise_results[baseline]["significance"] = sig
                pairwise_results[baseline]["significance_bh"] = sig_bh

                pr = pairwise_results[baseline]
                print(f"  vs {baseline:25s}: p_raw={pr['p_raw']:.4f}  p_holm={pr['p_holm']:.4f} {sig}  p_BH={pr['p_bh']:.4f} {sig_bh}  "
                      f"Cliff's δ={pr['cliffs_delta']:+.3f} ({pr['effect_size']})  "
                      f"W/T/L={pr['sage_wins']}/{pr['ties']}/{pr['baseline_wins']}")

    # ---------------------------------------------------------------
    # 4. Per-environment breakdown
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PER-ENVIRONMENT BREAKDOWN")
    print("=" * 60)

    env_results = {}
    envs = sorted(set(sc[0] for sc in valid_scenarios))
    for env in envs:
        env_scenarios = [sc for sc in valid_scenarios if sc[0] == env]
        env_idx = [valid_scenarios.index(sc) for sc in env_scenarios]

        if rank_matrix:
            env_ranks = rank_matrix_np[env_idx, :]
            env_mean_ranks = env_ranks.mean(axis=0)

            env_info = {}
            print(f"\n  {env} ({len(env_scenarios)} scenarios):")
            for j, agg in enumerate(all_aggs):
                env_info[agg] = float(env_mean_ranks[j])
                print(f"    {agg:25s}: {env_mean_ranks[j]:.2f}")
            env_results[env] = env_info

    # ---------------------------------------------------------------
    # 5. Win/Place/Show counts for SAGE
    # ---------------------------------------------------------------
    if sage_key and rank_matrix:
        sage_idx = all_aggs.index(sage_key)
        sage_ranks_all = rank_matrix_np[:, sage_idx]
        n_total = len(sage_ranks_all)
        n_first = int((sage_ranks_all == 1.0).sum())
        n_top3 = int((sage_ranks_all <= 3.0).sum())
        n_bottom3 = int((sage_ranks_all >= len(all_aggs) - 2).sum())

        print(f"\nSAGE Performance Summary ({n_total} scenarios):")
        print(f"  First place: {n_first} ({100*n_first/n_total:.1f}%)")
        print(f"  Top 3:       {n_top3} ({100*n_top3/n_total:.1f}%)")
        print(f"  Bottom 3:    {n_bottom3} ({100*n_bottom3/n_total:.1f}%)")

        sage_summary = {
            "n_scenarios": n_total,
            "n_first_place": n_first,
            "pct_first_place": round(100 * n_first / n_total, 1),
            "n_top3": n_top3,
            "pct_top3": round(100 * n_top3 / n_total, 1),
            "n_bottom3": n_bottom3,
        }
    else:
        sage_summary = {}

    # ---------------------------------------------------------------
    # 6. Attack-specific analysis
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ATTACK-SPECIFIC ANALYSIS")
    print("=" * 60)

    attack_results = {}
    attacks = sorted(set(sc[1] for sc in valid_scenarios))
    for attack in attacks:
        atk_scenarios = [sc for sc in valid_scenarios if sc[1] == attack]
        atk_idx = [valid_scenarios.index(sc) for sc in atk_scenarios]

        if rank_matrix:
            atk_ranks = rank_matrix_np[atk_idx, :]
            atk_mean_ranks = atk_ranks.mean(axis=0)

            print(f"\n  Attack: {attack} ({len(atk_scenarios)} scenarios)")
            atk_info = {}
            for j, agg in enumerate(all_aggs):
                atk_info[agg] = float(atk_mean_ranks[j])
            # Print sorted by rank
            for agg, rank in sorted(atk_info.items(), key=lambda x: x[1]):
                marker = " <-- SAGE" if agg == sage_key else ""
                print(f"    {agg:25s}: {rank:.2f}{marker}")
            attack_results[attack] = atk_info

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    output = {
        "friedman_chi2": float(chi2) if rank_matrix else None,
        "friedman_p": float(friedman_p) if rank_matrix else None,
        "mean_ranks": rank_results,
        "pairwise_vs_sage": pairwise_results,
        "sage_summary": sage_summary,
        "per_environment": env_results,
        "per_attack": attack_results,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
