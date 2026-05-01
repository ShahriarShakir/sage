#!/usr/bin/env python3
"""
scripts/analyze_signal_activation.py — SAGE signal activation analysis.

Re-runs inference (forward pass only, no training) through existing experiment
logs to compute how often each of SAGE's three signals fires on each attack type.

Reads existing metrics.json from logs to identify experiments, then loads
the final model weights to simulate one round of aggregation with signal logging.

Alternative simpler approach: parse the SAGE aggregation by replaying
the deltas from logs if stored, or directly instrument sage() and run a
few evaluation rounds.

For the paper, we compute this from the training logs by adding signal
logging to sage() and running fresh short experiments (10 rounds each).

Outputs: plots/signal_activation.csv, plots/signal_activation_summary.json
"""

import json
import csv
import os
import sys
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_signal_analysis_experiment(scenario, attack, byz_fraction, n_agents, seed, rounds=10):
    """
    Run a short experiment with signal logging enabled.
    Returns per-round signal activation data.
    """
    from frl.agg import flatten_state_dict, sage

    # Monkey-patch sage to log signal activations
    signal_log = []

    original_sage = sage.__wrapped__ if hasattr(sage, '__wrapped__') else None

    def logging_sage(deltas, trust_scores=None, **kwargs):
        n = len(deltas)
        if n <= 2:
            from frl.agg import geometric_median
            signal_log.append({
                "n_clients": n,
                "sybil_fired": False,
                "alignment_fired": False,
                "alignment_disabled": False,
                "norm_fired": False,
                "weights_uniform": True,
            })
            return geometric_median(deltas) if n == 2 else (deltas[0] if n == 1 else {})

        vectors = torch.stack([flatten_state_dict(d) for d in deltas], dim=0)
        norms = vectors.norm(dim=1).clamp(min=1e-10)
        normalized = vectors / norms.unsqueeze(1)
        S = normalized @ normalized.T
        median_norm = norms.median().item()

        sybil_flags = []
        align_flags = []
        norm_flags = []
        median_sims = []

        for i in range(n):
            mask = torch.ones(n, dtype=torch.bool, device=vectors.device)
            mask[i] = False
            sims_i = S[i, mask]

            max_sim = sims_i.max().item()
            sybil_flags.append(max_sim > 0.9999)

            median_sim = sims_i.median().item()
            median_sims.append(median_sim)
            align_flags.append(median_sim < 0)

            norm_ratio = norms[i].item() / max(median_norm, 1e-10)
            norm_flags.append(norm_ratio > 2.0 or norm_ratio < 0.5)

        n_negative = sum(1 for m in median_sims if m < 0)
        alignment_disabled = (n_negative >= n / 2)

        signal_log.append({
            "n_clients": n,
            "sybil_fired": any(sybil_flags),
            "sybil_count": sum(sybil_flags),
            "alignment_fired": any(align_flags) and not alignment_disabled,
            "alignment_disabled": alignment_disabled,
            "alignment_raw_count": sum(align_flags),
            "norm_fired": any(norm_flags),
            "norm_count": sum(norm_flags),
            "weights_uniform": not (any(sybil_flags) or
                                    (any(align_flags) and not alignment_disabled) or
                                    any(norm_flags)),
        })

        # Call the real sage function
        from frl.agg import sage as real_sage
        return real_sage(deltas, trust_scores, **kwargs)

    # Patch the aggregator registry
    import frl.agg
    original_agg = frl.agg.AGGREGATORS["trust_weighted"]
    frl.agg.AGGREGATORS["trust_weighted"] = logging_sage

    try:
        # Run a short experiment
        from scripts.train_mpe import build_parser
        args_list = [
            "--scenario", scenario,
            "--attack", attack,
            "--byzantine_fraction", str(byz_fraction),
            "--aggregator", "trust_weighted",
            "--rounds", str(rounds),
            "--seed", str(seed),
            "--n_agents", str(n_agents),
            "--eval_frequency", "5",
            "--eval_episodes", "3",
            "--log_suffix", "signal_analysis",
        ]
        parser = build_parser()
        args = parser.parse_args(args_list)

        from frl.server import FRLServer

        # Build server from args (simplified)
        import importlib
        mod = importlib.import_module("scripts.train_mpe")
        # We need to call main but capture signal_log
        # Use subprocess instead for isolation
        pass

    finally:
        frl.agg.AGGREGATORS["trust_weighted"] = original_agg

    return signal_log


def run_via_subprocess(scenario, attack, byz_frac, n_agents, seed, rounds=15):
    """
    Run a short experiment via subprocess and parse the signal log from debug output.
    """
    log_suffix = "signal_analysis"
    exp_name = f"{scenario}_trust_weighted_{attack}_n{n_agents}_b{int(byz_frac*n_agents)}_r{rounds}_s{seed}{log_suffix}"

    cmd = [
        sys.executable, "scripts/train_mpe.py",
        "--scenario", scenario,
        "--attack", attack,
        "--byzantine_fraction", str(byz_frac),
        "--aggregator", "trust_weighted",
        "--rounds", str(rounds),
        "--seed", str(seed),
        "--n_agents", str(n_agents),
        "--eval_frequency", str(rounds),  # eval only at end
        "--eval_episodes", "3",
        "--log_suffix", log_suffix,
    ]

    env = os.environ.copy()
    env["SAGE_LOG_SIGNALS"] = "1"  # We'll add this flag to sage()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    return exp_name, result.returncode


def main():
    """
    Approach: Instrument sage() to log signal activations to a JSON file,
    then run short (15-round) experiments for each scenario.
    """
    import frl.agg as agg_module

    # First, add signal logging capability to sage
    OUTPUT_CSV = Path("plots/signal_activation.csv")
    OUTPUT_JSON = Path("plots/signal_activation_summary.json")

    scenarios = [
        ("simple_spread_v3", "none", 0.0, 6),
        ("simple_spread_v3", "sign_flip", 0.33, 6),
        ("simple_spread_v3", "sign_flip", 0.5, 6),
        ("simple_spread_v3", "normalized", 0.33, 6),
        ("simple_spread_v3", "adaptive_strategic", 0.33, 6),
        ("simple_adversary_v3", "none", 0.0, 6),
        ("simple_adversary_v3", "sign_flip", 0.33, 6),
        ("simple_adversary_v3", "sign_flip", 0.5, 6),
        ("simple_adversary_v3", "normalized", 0.33, 6),
        ("simple_adversary_v3", "adaptive_strategic", 0.33, 6),
        ("simple_tag_v3", "none", 0.0, 6),
        ("simple_tag_v3", "sign_flip", 0.33, 6),
        ("simple_tag_v3", "sign_flip", 0.5, 6),
        ("simple_tag_v3", "normalized", 0.33, 6),
        ("simple_tag_v3", "adaptive_strategic", 0.33, 6),
    ]

    SIGNAL_LOG_DIR = Path("logs/signal_analysis")
    SIGNAL_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Instead of running new experiments (expensive), analyze by instrumenting
    # sage() and running 15-round experiments for each scenario with seed 42.

    # Monkey-patch sage to log signals
    original_sage = agg_module.AGGREGATORS["trust_weighted"]

    all_records = []

    for scenario, attack, byz_frac, n_agents in scenarios:
        print(f"\nAnalyzing: {scenario} / {attack} / byz={byz_frac}")

        signal_records = []

        def make_logging_sage(records_list):
            def logging_sage(deltas, trust_scores=None, **kwargs):
                n = len(deltas)
                record = {"n_clients": n}

                if n <= 2:
                    record.update({
                        "sybil_fired": False, "sybil_count": 0,
                        "alignment_fired": False, "alignment_disabled": False,
                        "alignment_raw_count": 0,
                        "norm_fired": False, "norm_count": 0,
                        "weights_uniform": True,
                    })
                    records_list.append(record)
                    return original_sage(deltas, trust_scores, **kwargs)

                vectors = torch.stack([agg_module.flatten_state_dict(d) for d in deltas], dim=0)
                norms = vectors.norm(dim=1).clamp(min=1e-10)
                normalized = vectors / norms.unsqueeze(1)
                S = normalized @ normalized.T
                median_norm = norms.median().item()

                sybil_flags = []
                align_flags = []
                norm_flags = []
                median_sims = []

                for i in range(n):
                    mask2 = torch.ones(n, dtype=torch.bool, device=vectors.device)
                    mask2[i] = False
                    sims_i = S[i, mask2]

                    sybil_flags.append(sims_i.max().item() > 0.9999)
                    med_sim = sims_i.median().item()
                    median_sims.append(med_sim)
                    align_flags.append(med_sim < 0)

                    norm_r = norms[i].item() / max(median_norm, 1e-10)
                    norm_flags.append(norm_r > 2.0 or norm_r < 0.5)

                n_neg = sum(1 for m in median_sims if m < 0)
                alignment_disabled = (n_neg >= n / 2)

                record.update({
                    "sybil_fired": any(sybil_flags),
                    "sybil_count": sum(sybil_flags),
                    "alignment_fired": any(align_flags) and not alignment_disabled,
                    "alignment_disabled": alignment_disabled,
                    "alignment_raw_count": sum(align_flags),
                    "norm_fired": any(norm_flags),
                    "norm_count": sum(norm_flags),
                    "weights_uniform": not (any(sybil_flags) or
                                            (any(align_flags) and not alignment_disabled) or
                                            any(norm_flags)),
                })
                records_list.append(record)

                return original_sage(deltas, trust_scores, **kwargs)
            return logging_sage

        signal_records = []
        agg_module.AGGREGATORS["trust_weighted"] = make_logging_sage(signal_records)

        try:
            from scripts.train_mpe import main as train_main

            train_args = [
                "--scenario", scenario,
                "--attack", attack,
                "--byzantine_fraction", str(byz_frac),
                "--aggregator", "trust_weighted",
                "--rounds", "15",
                "--seed", "42",
                "--n_agents", str(n_agents),
                "--eval_frequency", "15",
                "--eval_episodes", "1",
                "--log_suffix", "signal_analysis",
            ]

            old_argv = sys.argv
            sys.argv = ["train_mpe.py"] + train_args
            try:
                train_main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv

        except Exception as e:
            print(f"  ERROR: {e}")
            signal_records = []

        finally:
            agg_module.AGGREGATORS["trust_weighted"] = original_sage

        # Summarize signals for this scenario
        if signal_records:
            n_rounds = len(signal_records)
            sybil_pct = 100 * sum(r["sybil_fired"] for r in signal_records) / n_rounds
            align_pct = 100 * sum(r["alignment_fired"] for r in signal_records) / n_rounds
            align_disabled_pct = 100 * sum(r.get("alignment_disabled", False) for r in signal_records) / n_rounds
            norm_pct = 100 * sum(r["norm_fired"] for r in signal_records) / n_rounds
            uniform_pct = 100 * sum(r["weights_uniform"] for r in signal_records) / n_rounds

            print(f"  Rounds: {n_rounds}")
            print(f"  Sybil fired:       {sybil_pct:.0f}%")
            print(f"  Alignment fired:   {align_pct:.0f}% (disabled: {align_disabled_pct:.0f}%)")
            print(f"  Norm fired:        {norm_pct:.0f}%")
            print(f"  Weights uniform:   {uniform_pct:.0f}%")

            all_records.append({
                "environment": scenario,
                "attack": attack,
                "byz_fraction": byz_frac,
                "n_rounds": n_rounds,
                "sybil_pct": round(sybil_pct, 1),
                "alignment_pct": round(align_pct, 1),
                "alignment_disabled_pct": round(align_disabled_pct, 1),
                "norm_pct": round(norm_pct, 1),
                "uniform_pct": round(uniform_pct, 1),
            })

    # Save CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "environment", "attack", "byz_fraction", "n_rounds",
            "sybil_pct", "alignment_pct", "alignment_disabled_pct",
            "norm_pct", "uniform_pct"
        ])
        writer.writeheader()
        writer.writerows(all_records)

    # Save JSON summary
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_records, f, indent=2)

    print(f"\nResults saved to {OUTPUT_CSV} and {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
