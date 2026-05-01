#!/bin/bash
# scripts/run_god_mode_sweep.sh
# ============================================================
# Master launcher that starts every reviewer-risk-closing sweep
# in sequence. Total runtime on a single RTX 5000 Ada: ~5 days.
# All sweeps are --resume safe; running again only fills gaps.
# ============================================================
set -e
cd "$(dirname "$0")/.."
# Activate your environment: conda activate sage

mkdir -p logs
LOGFILE="logs/god_mode_$(date +%Y%m%d_%H%M%S).log"
echo "Starting god-mode sweep, log: $LOGFILE"

{
  echo "=== Phase B1: Faithful baselines (3 baselines × 5 attacks × 5 seeds × 3 envs = 225) ==="
  python scripts/run_faithful_baselines.py --resume

  echo "=== Phase B2: Seed extension (4 methods × 5 attacks × 5 new seeds × 3 envs = 300) ==="
  python scripts/run_seeds_extension.py --resume

  echo "=== Phase B3: n=12 partial sweep (5 methods × 3 attacks × 3 seeds × 3 envs = 135) ==="
  python scripts/run_n12_partial.py --resume

  echo "=== Phase C: Regenerate paper artifacts ==="
  python scripts/regen_paper_artifacts.py

  echo "=== DONE ==="
} 2>&1 | tee "$LOGFILE"
