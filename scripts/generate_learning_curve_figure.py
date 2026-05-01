#!/usr/bin/env python3
"""
scripts/generate_learning_curve_figure.py — Generate learning curve figures for the paper.

Creates a compact multi-panel figure showing training dynamics for SAGE vs baselines
across key scenarios.
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

DATA_FILE = Path("plots/learning_curves.csv")
OUTPUT_DIR = Path("plots")

# Methods to plot (subset for clarity)
METHODS_MAIN = ["SAGE", "GeoMedian", "FLTrust", "FedAvg", "FLAME", "Krum"]
COLORS = {
    "SAGE": "#e31a1c",
    "GeoMedian": "#1f78b4",
    "FLTrust": "#33a02c",
    "FedAvg": "#ff7f00",
    "FLAME": "#6a3d9a",
    "Krum": "#b15928",
    "TrimmedMean": "#a6cee3",
    "MultiKrum": "#fb9a99",
    "FoolsGold": "#cab2d6",
    "HATT": "#999999",
}

# Scenarios for the main figure (3 panels: one strong, one medium, one weak for SAGE)
MAIN_SCENARIOS = [
    ("Tag", "normalized", "Tag — Normalized Attack (50%)"),
    ("Adversary", "sign_flip", "Adversary — Sign-Flip (50%)"),
    ("Spread", "adaptive", "Spread — Adaptive Strategic (50%)"),
]


def load_data():
    """Load learning curve data."""
    data = defaultdict(list)
    with open(DATA_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["method"], row["environment"], row["attack"])
            seed = int(row["seed"])
            rnd = int(row["round"])
            val = float(row["eval_return_mean"])
            data[key].append((seed, rnd, val))
    return data


def get_mean_std_by_round(records):
    """From a list of (seed, round, value), compute mean±std per round."""
    by_round = defaultdict(list)
    for seed, rnd, val in records:
        by_round[rnd].append(val)
    rounds = sorted(by_round.keys())
    means = [np.mean(by_round[r]) for r in rounds]
    stds = [np.std(by_round[r]) for r in rounds]
    return np.array(rounds), np.array(means), np.array(stds)


def smooth(y, window=5):
    """Simple moving average smoothing."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    smoothed = np.convolve(y, kernel, mode='valid')
    # Pad to maintain length
    pad = len(y) - len(smoothed)
    return np.concatenate([y[:pad], smoothed])


def map_attack_name(attack_in_data):
    """Map attack names from CSV to scenario names."""
    mapping = {
        "none": "none",
        "sign_flip": "sign_flip",
        "normalized": "normalized",
        "adaptive": "adaptive",
        "adaptive_strategic": "adaptive",
    }
    return mapping.get(attack_in_data, attack_in_data)


def main():
    data = load_data()

    # Print available keys for debugging
    all_keys = set()
    for key in data:
        method, env, attack = key
        all_keys.add((env, attack))
    print("Available (env, attack) combinations:")
    for k in sorted(all_keys):
        print(f"  {k}")

    # Main figure: 1×3 panels
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))

    for ax_idx, (env, attack, title) in enumerate(MAIN_SCENARIOS):
        ax = axes[ax_idx]

        for method in METHODS_MAIN:
            # Try matching attack names
            records = None
            for attack_variant in [attack, attack.replace("adaptive", "adaptive_strategic")]:
                key = (method, env, attack_variant)
                if key in data and len(data[key]) > 0:
                    records = data[key]
                    break

            if records is None or len(records) == 0:
                continue

            rounds, means, stds = get_mean_std_by_round(records)
            means_sm = smooth(means)
            color = COLORS.get(method, "gray")
            lw = 2.0 if method == "SAGE" else 1.0
            alpha = 1.0 if method == "SAGE" else 0.7
            zorder = 10 if method == "SAGE" else 5

            ax.plot(rounds, means_sm, color=color, linewidth=lw,
                    label=method, alpha=alpha, zorder=zorder)
            ax.fill_between(rounds, means_sm - stds * 0.5, means_sm + stds * 0.5,
                            color=color, alpha=0.1, zorder=zorder - 1)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Round")
        if ax_idx == 0:
            ax.set_ylabel("Eval Return")
        ax.grid(True, alpha=0.3)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(METHODS_MAIN),
               bbox_to_anchor=(0.5, 1.08), frameon=False)

    plt.tight_layout()
    outpath = OUTPUT_DIR / "fig_learning_curves.pdf"
    fig.savefig(outpath, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()

    # Also generate a 3×5 full panel (all scenarios)
    all_scenarios = [
        ("Spread", "none", "Spread — No Attack"),
        ("Spread", "sign_flip", "Spread — Sign-Flip"),
        ("Spread", "normalized", "Spread — Normalized"),
        ("Spread", "adaptive", "Spread — Adaptive"),
        ("Adversary", "none", "Adversary — No Attack"),
        ("Adversary", "sign_flip", "Adversary — Sign-Flip"),
        ("Adversary", "normalized", "Adversary — Normalized"),
        ("Adversary", "adaptive", "Adversary — Adaptive"),
        ("Tag", "none", "Tag — No Attack"),
        ("Tag", "sign_flip", "Tag — Sign-Flip"),
        ("Tag", "normalized", "Tag — Normalized"),
        ("Tag", "adaptive", "Tag — Adaptive"),
    ]

    fig2, axes2 = plt.subplots(3, 4, figsize=(16, 10))
    axes_flat = axes2.flatten()

    for ax_idx, (env, attack, title) in enumerate(all_scenarios):
        ax = axes_flat[ax_idx]

        for method in METHODS_MAIN:
            records = None
            for attack_variant in [attack, attack.replace("adaptive", "adaptive_strategic")]:
                key = (method, env, attack_variant)
                if key in data and len(data[key]) > 0:
                    records = data[key]
                    break

            if records is None or len(records) == 0:
                continue

            rounds, means, stds = get_mean_std_by_round(records)
            means_sm = smooth(means)
            color = COLORS.get(method, "gray")
            lw = 2.0 if method == "SAGE" else 1.0
            alpha = 1.0 if method == "SAGE" else 0.7
            zorder = 10 if method == "SAGE" else 5

            ax.plot(rounds, means_sm, color=color, linewidth=lw,
                    label=method, alpha=alpha, zorder=zorder)
            ax.fill_between(rounds, means_sm - stds * 0.5, means_sm + stds * 0.5,
                            color=color, alpha=0.1, zorder=zorder - 1)

        ax.set_title(title, fontsize=8, fontweight='bold')
        if ax_idx % 4 == 0:
            ax.set_ylabel("Eval Return", fontsize=8)
        if ax_idx >= 8:
            ax.set_xlabel("Round", fontsize=8)
        ax.grid(True, alpha=0.3)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='upper center', ncol=len(METHODS_MAIN),
                bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8)

    plt.tight_layout()
    outpath2 = OUTPUT_DIR / "fig_learning_curves_full.pdf"
    fig2.savefig(outpath2, bbox_inches='tight')
    print(f"Saved: {outpath2}")
    plt.close()


if __name__ == "__main__":
    main()
