#!/bin/bash
# Run SAGE component ablation experiments
# 5 variants × 15 scenarios × 3 seeds = 225 experiments
#
# Variants (controlled by env vars passed to train_mpe.py via a wrapper):
#   1. gm_only       - geometric_median aggregator (no SAGE signals)
#   2. sage_sybil     - SAGE with only sybil detection (align+norm disabled)
#   3. sage_align     - SAGE with only alignment check (sybil+norm disabled)
#   4. sage_norm      - SAGE with only norm outlier (sybil+align disabled)
#   5. sage_full      - Full SAGE (all three signals) — reference

set -e

cd .
# Activate your environment: conda activate sage

SEEDS=(42 123 456)

SCENARIOS=(
    "simple_adversary_v3 none 0.0"
    "simple_adversary_v3 sign_flip 0.17"
    "simple_adversary_v3 sign_flip 0.5"
    "simple_adversary_v3 normalized 0.5"
    "simple_adversary_v3 adaptive_strategic 0.5"
    "simple_spread_v3 none 0.0"
    "simple_spread_v3 sign_flip 0.17"
    "simple_spread_v3 sign_flip 0.5"
    "simple_spread_v3 normalized 0.5"
    "simple_spread_v3 adaptive_strategic 0.5"
    "simple_tag_v3 none 0.0"
    "simple_tag_v3 sign_flip 0.17"
    "simple_tag_v3 sign_flip 0.5"
    "simple_tag_v3 normalized 0.5"
    "simple_tag_v3 adaptive_strategic 0.5"
)

# We only need to run variants 1-4; variant 5 (sage_full) already exists as trust_weighted
VARIANTS=("gm_only" "sage_sybil" "sage_align" "sage_norm")

TOTAL=$(( ${#VARIANTS[@]} * ${#SCENARIOS[@]} * ${#SEEDS[@]} ))
COUNT=0

echo "=========================================="
echo "SAGE Ablation Study"
echo "Variants: ${#VARIANTS[@]}, Scenarios: ${#SCENARIOS[@]}, Seeds: ${#SEEDS[@]}"
echo "Total experiments: $TOTAL"
echo "=========================================="

for VARIANT in "${VARIANTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for i in "${!SCENARIOS[@]}"; do
            read -r SCENARIO ATTACK BYZ_FRAC <<< "${SCENARIOS[$i]}"
            COUNT=$((COUNT + 1))

            SUFFIX="_ablation_${VARIANT}"

            # Determine aggregator and ablation flags
            if [[ "$VARIANT" == "gm_only" ]]; then
                AGG="geometric_median"
            else
                AGG="trust_weighted"
            fi

            echo ""
            echo "[${COUNT}/${TOTAL}] ${VARIANT} | ${SCENARIO} / ${ATTACK} / byz=${BYZ_FRAC} | seed=${SEED}"
            echo "Started: $(date '+%H:%M:%S')"

            if [[ "$VARIANT" == "gm_only" ]]; then
                python scripts/train_mpe.py \
                    --scenario "$SCENARIO" \
                    --attack "$ATTACK" \
                    --byzantine_fraction "$BYZ_FRAC" \
                    --aggregator "$AGG" \
                    --rounds 200 \
                    --seed "$SEED" \
                    --eval_frequency 5 \
                    --eval_episodes 10 \
                    --log_suffix "$SUFFIX" \
                    2>&1 | tail -3
            else
                # For SAGE ablation variants, we use the ablation wrapper
                python scripts/run_sage_ablation_variant.py \
                    --scenario "$SCENARIO" \
                    --attack "$ATTACK" \
                    --byzantine_fraction "$BYZ_FRAC" \
                    --variant "$VARIANT" \
                    --rounds 200 \
                    --seed "$SEED" \
                    --eval_frequency 5 \
                    --eval_episodes 10 \
                    --log_suffix "$SUFFIX" \
                    2>&1 | tail -3
            fi

            echo "Finished: $(date '+%H:%M:%S')"
        done
    done
done

echo ""
echo "=========================================="
echo "Ablation study complete! ($COUNT experiments)"
echo "=========================================="
