# SAGE: Similarity-Aware Geometric Median for Byzantine-Robust Federated MARL

Anonymous code and reproducibility release for the experiments reported in
the paper "Simplicity over Complexity under Non-Stationary Policy Gradients:
Byzantine-Robust Federated Multi-Agent Reinforcement Learning via
Similarity-Aware Geometric Median".

This archive mirrors the anonymous code repository referenced in the
abstract at https://anonymous.4open.science/r/sage-marl .
Both distributions contain identical source; the ZIP is provided so
reviewers can inspect the code offline.


## Installation

```bash
conda create -n sage python=3.10
conda activate sage
pip install -r requirements.txt
```

## Reproducing the paper

### Smoke test (single experiment)
```bash
python scripts/train_mpe.py \
    --scenario simple_spread_v3 \
    --attack sign_flip \
    --byzantine_fraction 0.5 \
    --aggregator trust_weighted \
    --rounds 200 --seed 42
```
Runs one 200-round experiment (~5 min on an RTX 5000 Ada).

### Run a specific configuration
```bash
python scripts/run_experiment.py --config configs/default.yaml
```
Runs a single experiment defined in `configs/default.yaml`.
Overrides are supported: `python scripts/run_experiment.py --config configs/default.yaml attack.type=sign_flip aggregation.method=krum`.

### Main 750-experiment benchmark
The pre-computed results are provided in `plots/neurips_results_all.csv`
(750 rows = 3 environments × 5 attack conditions × 5 seeds × 10 aggregators).
To re-run individual experiments, loop over the configurations in
`configs/sweep.yaml` and call `scripts/run_experiment.py` for each
combination. Estimated total: ~63 GPU-hours on a single RTX 5000 Ada.

### 180-experiment component ablation (Appendix G)
```bash
bash scripts/run_sage_ablation.sh
```
4 signal-ablation variants × 15 scenario–attack–fraction combinations × 3 seeds = 180 experiments.

### Statistics and artifact regeneration
```bash
python scripts/compute_enhanced_stats.py
python scripts/collect_learning_curves.py
python scripts/generate_learning_curve_figure.py
```
Or run all steps in sequence:
```bash
python scripts/regen_paper_artifacts.py
```

### Appendix sweeps (Appendix J)
```bash
# Run all appendix phases in sequence (approx. 5 days on a single GPU):
bash scripts/run_god_mode_sweep.sh
# Or run individual phases:
# (a) Reference-grade FLTrust*, FLAME*, FoolsGold* baselines (225 experiments)
python scripts/run_faithful_baselines.py --resume
# (b) 5 additional seeds for the 5 strongest aggregators (300 experiments)
python scripts/run_seeds_extension.py --resume
# (c) n_C = 12 client-scale verification (135 experiments)
python scripts/run_n12_partial.py --resume
# (d) Aggregate appendix results
python scripts/analyze_god_mode_results.py
```

## Aggregators implemented (frl/agg.py, frl/trust.py)

- `fedavg` - FedAvg baseline.
- `trimmed_mean` - Coordinate-wise trimmed mean.
- `geometric_median` - Geometric median via Weiszfeld iteration.
- `krum`, `multi_krum` - Krum and Multi-Krum.
- `fltrust` - FLTrust (see disclosures below).
- `flame` - FLAME (see disclosures below).
- `foolsgold` - FoolsGold (see disclosures below).
- `trust_weighted` - **SAGE (our method)**.
- `hatt` - Heterogeneity-Aware Temporal Trust (deliberately elaborate
  baseline; ten weighted components described in Appendix F of the
  paper).

## Baseline fidelity (disclosed in Section 6 and Appendix E of the paper)

To enable a unified benchmark in the federated MARL training loop,
three baselines use documented approximations of components whose
original formulation targets supervised FL with IID data:

- FLTrust: no server root dataset; the server reference is
  approximated by the trimmed mean of client deltas with trim fraction
  0.2 (frl/agg.py).
- FLAME: threshold-sweep connected-component clustering over
  cosine similarity, in place of HDBSCAN.
- FoolsGold: single-round pairwise cosine similarity with no
  cumulative-history tracking.
- HATT: ten raw component weights sum to 1.45; the aggregation
  normalises per-client trust scores before weighting, so this does
  not cause numerical drift, but every client receives evidence from
  more scoring heads than a prior-weighted mixture would suggest.
- Oracle f: Krum, Multi-Krum, FLAME and HATT receive the true
  Byzantine count from the training harness. SAGE does not.

Reference-grade reimplementations of FLTrust, FLAME, and FoolsGold
(`fltrust_lagged`, `flame_hdbscan`, `foolsgold_hist`) are also
included; see Appendix J of the paper.

## Project structure

```
frl/
  agg.py           - All aggregators including SAGE.
  trust.py         - HATT baseline and shared trust utilities.
  server.py        - Federated training loop.
  client.py        - Per-agent training.
  attacks.py       - sign-flip, normalized, adaptive-strategic attacks.
  eval.py          - Evaluation rollouts.
  models.py        - PPO actor/critic.
  envs/mpe_wrapper.py   - PettingZoo MPE wrapper.
configs/
  default.yaml     - PPO + federation hyperparameters.
  sweep.yaml       - Sweep parameter grid (used to define the 750-experiment
                     benchmark; pass to run_experiment.py manually per row).
scripts/
  train_mpe.py     - Single-experiment trainer (smoke-test entry point).
  run_experiment.py - Single-configuration benchmark runner.
  run_sage_ablation.sh - 180-experiment component ablation.
  compute_enhanced_stats.py - Friedman, Wilcoxon, Holm, Cliff's delta.
  analyze_god_mode_results.py - Aggregates all sweep logs into tables/figures;
                               writes plots/auto_appendix_god_mode.tex.
  regen_paper_artifacts.py  - Regenerate all result artifacts in sequence.
plots/
  neurips_results_all.csv   - Pre-computed returns for the 750 main
                              experiments (3 envs x 5 conditions x 5 seeds
                              x 10 aggregators).
  neurips_results.csv       - Combined results for 1203 completed experiments
                              across the main benchmark and appendix sweeps.
                              Note: the target of 1410 total appendix
                              experiments was not fully reached; 1203
                              experiments completed successfully.
  enhanced_stats.json       - Rank statistics.
  learning_curves.csv       - Per-round learning traces.
  auto_appendix_god_mode.tex - LaTeX table fragments generated by
                              analyze_god_mode_results.py (created on first
                              run).
```

## Hardware used

NVIDIA RTX 5000 Ada (32 GB VRAM), Intel Xeon, 64 GB RAM, Ubuntu Linux.
Each 200-round experiment takes approximately 5 minutes. The full
750-experiment benchmark requires approximately 63 GPU-hours.

## Random seeds

All main experiments use seeds {42, 123, 456, 789, 1024}. All 10 aggregators
share identical seeds per evaluation group, ensuring a matched
comparison. The 180-experiment ablation uses seeds {42, 123, 456}.

## Anonymous code and reproducibility artifacts

Anonymous code and reproducibility artifacts: https://anonymous.4open.science/r/sage-marl

## License

Released for anonymous review only; license will be set on de-anonymisation.
