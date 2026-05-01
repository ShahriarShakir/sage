#!/usr/bin/env python3
"""
scripts/regen_paper_artifacts.py — One-shot pipeline that regenerates every
reproducibility artifact from the available result CSV files.  Output:

  plots/enhanced_stats.json           (Friedman / Holm / BH FDR / Cliff)
  plots/learning_curves.csv           (per-round traces)
  plots/figures/learning_curves.*     (learning-curve figure)

Run after any sweep finishes:
  python scripts/regen_paper_artifacts.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STEPS = [
    ("Compute enhanced stats (Friedman / Holm / BH FDR / Cliff)",
     ["scripts/compute_enhanced_stats.py"]),
    ("Collect per-round learning curves",
     ["scripts/collect_learning_curves.py"]),
    ("Generate learning-curve figure",
     ["scripts/generate_learning_curve_figure.py"]),
    ("Aggregate appendix sweep results",
     ["scripts/analyze_god_mode_results.py"]),
]


def run(step_name, cmd):
    print(f"\n=== {step_name} ===")
    full = [sys.executable] + cmd
    proc = subprocess.run(full, cwd=str(ROOT))
    if proc.returncode != 0:
        print(f"  WARN: step '{step_name}' returned {proc.returncode}")
    return proc.returncode == 0


def main():
    ok = 0
    for step_name, cmd in STEPS:
        script_path = ROOT / cmd[0]
        if not script_path.exists():
            print(f"\n=== {step_name} === SKIP (missing {cmd[0]})")
            continue
        if run(step_name, cmd):
            ok += 1
    print(f"\nRegen pipeline complete: {ok}/{len(STEPS)} steps ok")


if __name__ == "__main__":
    main()

