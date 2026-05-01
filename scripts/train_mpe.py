#!/usr/bin/env python3
"""
scripts/train_mpe.py — Full MPE Federated Training with HATT Defense
=====================================================================
Main training script for the reported experiments.

Runs Byzantine-robust federated MARL on PettingZoo MPE environments
with our HATT trust scoring mechanism.

Usage:
    python scripts/train_mpe.py                          # default (simple_spread, 6 clients, 2 byz)
    python scripts/train_mpe.py --scenario simple_tag_v3 --n_clients 8 --byzantine_fraction 0.3
    python scripts/train_mpe.py --attack normalized --aggregator trust_weighted --rounds 300

Features:
    - Full GPU training on RTX 5000 Ada
    - Real-time progress display with metrics
    - Trust TPR/FPR tracking
    - Checkpointing every 25 rounds
    - Final evaluation and plotting
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
import warnings

# Suppress PettingZoo action clipping warnings
warnings.filterwarnings("ignore", message=".*outside action space.*")
warnings.filterwarnings("ignore", message=".*clipping to space.*")

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frl.models import ActorCritic
from frl.client import FRLClient
from frl.server import FRLServer
from frl.trust import HATTTrustScorer
from frl.attacks import get_attack
from frl.envs.mpe_wrapper import make_mpe_env, make_mpe_env_factory

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MPE environment configuration
# ---------------------------------------------------------------------------
MPE_CONFIGS = {
    "simple_spread_v3": {
        "n_agents": 3,          # 3 cooperative agents (same type)
        "obs_dim": 18,          # all agents obs=18
        "act_dim": 5,           # continuous 5-dim
        "continuous": True,
        "max_cycles": 25,
        "primary_agent_idx": 0, # all agents are same type
        "description": "cooperative navigation",
    },
    "simple_adversary_v3": {
        "n_agents": 3,          # 1 adversary + 2 good agents
        "obs_dim": 8,           # adversary obs=8, good agents obs=10
        "act_dim": 5,
        "continuous": True,
        "max_cycles": 25,
        "primary_agent_idx": 0, # train adversary (obs=8)
        "description": "adversary pursuit",
    },
    "simple_tag_v3": {
        "n_agents": 4,          # 3 adversaries + 1 good agent
        "obs_dim": 16,          # adversaries obs=16, good agent obs=14
        "act_dim": 5,
        "continuous": True,
        "max_cycles": 25,
        "primary_agent_idx": 0, # train adversaries (obs=16)
        "description": "predator-prey tag",
    },
}


def get_env_dims(scenario: str, agent_idx: int = 0):
    """Get actual obs/act dims by creating a temp env."""
    env = make_mpe_env(scenario, agent_idx=agent_idx, continuous_actions=True)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
    continuous = hasattr(env.action_space, 'shape')
    env.close()
    return obs_dim, act_dim, continuous


def build_experiment(args):
    """Build the full experiment: clients, server, trust, attacks."""
    device = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    scenario = args.scenario
    n_clients = args.n_clients

    # Get actual environment dimensions (may differ per agent in some MPE scenarios)
    obs_dim, act_dim, continuous = get_env_dims(scenario, agent_idx=0)
    logger.info(f"Environment: {scenario} | obs_dim={obs_dim}, act_dim={act_dim}, continuous={continuous}")

    # Determine number of agents in scenario
    config = MPE_CONFIGS.get(scenario, {"n_agents": 3, "max_cycles": 25})
    n_agents = config["n_agents"]

    # Create global model
    hidden_dims = [args.hidden_dim] * args.n_layers
    global_model = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dims=hidden_dims,
        continuous=continuous,
        activation=args.activation,
    ).to(device)

    param_count = sum(p.numel() for p in global_model.parameters())
    logger.info(f"Model: {param_count:,} parameters | hidden={hidden_dims}")

    # Create clients (each controls one agent in the MPE, cycling through agents)
    clients = []
    for i in range(n_clients):
        agent_idx = i % n_agents  # Cycle agents across clients

        # Check obs/act dims for this specific agent
        client_obs_dim, client_act_dim, _ = get_env_dims(scenario, agent_idx=agent_idx)

        # If dims differ from agent 0, we need a per-client model
        # For simple_spread_v3, all agents have same dims
        if client_obs_dim != obs_dim or client_act_dim != act_dim:
            logger.warning(
                f"Client {i} (agent {agent_idx}) has different dims: "
                f"obs={client_obs_dim}, act={client_act_dim}. Using agent 0 dims."
            )
            agent_idx = 0
            client_obs_dim, client_act_dim = obs_dim, act_dim

        env = make_mpe_env(
            scenario=scenario,
            agent_idx=agent_idx,
            max_cycles=config["max_cycles"],
            continuous_actions=continuous,
        )

        client = FRLClient(
            client_id=i,
            env=env,
            obs_dim=client_obs_dim,
            act_dim=client_act_dim,
            hidden_dims=hidden_dims,
            continuous=continuous,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_eps=args.clip_eps,
            entropy_coef=args.entropy_coef,
            local_epochs=args.local_epochs,
            rollout_steps=args.rollout_steps,
            minibatch_size=args.minibatch_size,
            device=device,
        )
        clients.append(client)

    # Byzantine setup
    if args.attack == "none":
        n_byzantine = 0
        byzantine_ids = []
    else:
        n_byzantine = max(1, int(args.byzantine_fraction * n_clients))
        byzantine_ids = list(range(n_clients - n_byzantine, n_clients))  # Last N are Byzantine
    logger.info(f"Clients: {n_clients} total, {n_byzantine} Byzantine (IDs: {byzantine_ids})")

    # Assign attacks
    attacks = {}
    for bid in byzantine_ids:
        attack_kwargs = {}
        if args.attack == "sign_flip":
            attack_kwargs = {"scale": args.attack_scale}
        elif args.attack == "scaling":
            attack_kwargs = {"scale_factor": args.attack_scale}
        elif args.attack == "gaussian_noise":
            attack_kwargs = {"noise_std": args.attack_scale}
        elif args.attack == "normalized":
            attack_kwargs = {"perturbation_factor": args.attack_scale}
        elif args.attack == "directional":
            attack_kwargs = {"magnitude": args.attack_scale}
        elif args.attack == "adaptive_strategic":
            attack_kwargs = {
                "warmup_rounds": 20,
                "ramp_rounds": 30,
                "max_perturbation": args.attack_scale,
            }
        attacks[bid] = get_attack(args.attack, **attack_kwargs)
    logger.info(f"Attack: {args.attack} (scale={args.attack_scale})")

    # Trust scorer (HATT)
    eval_env_factory = make_mpe_env_factory(
        scenario=scenario,
        agent_idx=0,
        max_cycles=config["max_cycles"],
        continuous_actions=continuous,
    )

    trust_scorer = HATTTrustScorer(
        n_clients=n_clients,
        ema_beta=args.ema_beta,
        high_threshold=0.7,
        low_threshold=0.3,
        hysteresis_window=3,
        envelope_window=10,
        z_score_threshold=2.5,
        spectral_components=2,
        spectral_threshold=2.0,
        correlation_threshold=0.95,
        audit_env_factory=eval_env_factory,  # Always provide for delta-effect audit
        audit_seeds=[42, 123, 456, 789, 314],  # 5 seeds for stability
        audit_steps=args.audit_steps,
        audit_frequency=args.audit_frequency,
        # v2: Coordinate-wise anomaly params
        cwas_mad_threshold=args.cwas_mad_threshold,
        cwas_subsample_dim=args.cwas_subsample_dim,
        cwas_sign_weight=args.cwas_sign_weight,
        # v2 rebalanced weights (no single component > 30%)
        w_delta_effect=args.w_delta_effect,
        w_coordinate_anomaly=args.w_coordinate_anomaly,
        w_spectral=args.w_spectral,
        w_population_dir=args.w_population,
        w_temporal=args.w_temporal,
        w_heterogeneity=args.w_heterogeneity,
        w_correlation=args.w_correlation,
        # v3: New innovation weights
        w_loo_validation=args.w_loo_validation,
        w_gradient_inversion=args.w_gradient_inversion,
        w_trajectory=args.w_trajectory,
        # v2: Adaptive weighting
        adaptive_weights=args.adaptive_weights,
        adaptive_blend=args.adaptive_blend,
        # v3: Bayesian trust
        use_bayesian_trust=args.use_bayesian_trust,
        bayesian_evidence_scale=args.bayesian_evidence_scale,
        bayesian_decay=args.bayesian_decay,
        # v3: Functional-first initialization
        functional_first_rounds=args.functional_first_rounds,
        trust_ema_beta=args.trust_ema_beta,
        warmup_rounds=args.warmup_rounds,
    )
    logger.info(f"Trust: HATT (ema_beta={args.ema_beta}, trust_ema_beta={args.trust_ema_beta})")
    logger.info(f"Aggregator: {args.aggregator}")

    # Create server
    experiment_name = (
        f"{scenario}_{args.aggregator}_{args.attack}"
        f"_n{n_clients}_b{n_byzantine}_r{args.rounds}_s{args.seed}"
        f"{args.log_suffix}"
    )
    log_dir = os.path.join("logs", experiment_name)

    server = FRLServer(
        global_model=global_model,
        clients=clients,
        byzantine_ids=byzantine_ids,
        attacks=attacks,
        aggregator_name=args.aggregator,
        aggregator_kwargs={
            "trim_fraction": 0.1,
            "trust_threshold": args.trust_threshold,
            "trust_power": args.trust_power,
            "base_aggregator": args.base_aggregator,
            "adi_mode": args.adi_mode,
        },
        trust_scorer=trust_scorer,
        use_trust=False,  # TARA handles robustness internally (no external trust scoring needed)
        eval_env_factory=eval_env_factory,
        eval_frequency=args.eval_frequency,
        eval_episodes=args.eval_episodes,
        log_dir=log_dir,
        experiment_name=experiment_name,
        device=device,
    )

    return server, experiment_name


def print_header(args):
    """Print experiment header."""
    width = 70
    print("=" * width)
    print("  BYZANTINE-ROBUST FEDERATED MARL — SAGE Training")
    print("  Reproducibility Experiment")
    print("=" * width)
    print(f"  Scenario:    {args.scenario}")
    print(f"  Clients:     {args.n_clients} ({int(args.byzantine_fraction * 100)}% Byzantine)")
    print(f"  Attack:      {args.attack} (scale={args.attack_scale})")
    print(f"  Trust:       threshold={args.trust_threshold}, warmup={args.warmup_rounds} rounds")
    print(f"  Aggregator:  {args.aggregator}")
    print(f"  Rounds:      {args.rounds}")
    print(f"  Local PPO:   {args.local_epochs} epochs × {args.rollout_steps} steps")
    print(f"  Hidden:      [{args.hidden_dim}] × {args.n_layers}")
    print(f"  LR:          actor={args.lr_actor}, critic={args.lr_critic}")
    print(f"  GPU:         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * width)
    print()
    print(f"{'Round':>5} | {'Reward':>10} | {'EvalRet':>10} | {'TPR':>6} | {'FPR':>6} | "
          f"{'Trust_H':>8} | {'Trust_B':>8} | {'Time':>6}")
    print("-" * width)


def main():
    parser = argparse.ArgumentParser(description="Train Byzantine-Robust Federated MARL")

    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread_v3",
                        choices=list(MPE_CONFIGS.keys()))
    parser.add_argument("--n_clients", type=int, default=6)
    parser.add_argument("--byzantine_fraction", type=float, default=0.33)

    # Attack
    parser.add_argument("--attack", type=str, default="sign_flip",
                        choices=["sign_flip", "scaling", "gaussian_noise", "directional",
                                 "normalized", "adaptive_strategic", "inner_product_manipulation",
                                 "reward_bias", "stale_update", "none"])
    parser.add_argument("--attack_scale", type=float, default=1.0)

    # Aggregation
    parser.add_argument("--aggregator", type=str, default="trust_weighted",
                        choices=["fedavg", "trimmed_mean", "geometric_median",
                                 "krum", "multi_krum", "fltrust", "fltrust_lagged",
                                 "flame", "flame_hdbscan",
                                 "foolsgold", "foolsgold_hist", "trust_weighted"])

    # Training
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=4)
    parser.add_argument("--rollout_steps", type=int, default=512)
    parser.add_argument("--minibatch_size", type=int, default=128)
    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--activation", type=str, default="relu")

    # Trust
    parser.add_argument("--ema_beta", type=float, default=0.8)
    parser.add_argument("--trust_ema_beta", type=float, default=0.85)
    parser.add_argument("--use_audit", action="store_true", default=True)
    parser.add_argument("--audit_frequency", type=int, default=3)
    parser.add_argument("--audit_steps", type=int, default=100)
    parser.add_argument("--warmup_rounds", type=int, default=10)
    parser.add_argument("--trust_threshold", type=float, default=0.5,
                        help="Trust threshold for filtering clients")
    parser.add_argument("--trust_power", type=float, default=2.0,
                        help="Exponent for trust weight sharpening")

    # Ablation: Trust component weights (v2 rebalanced defaults)
    parser.add_argument("--w_delta_effect", type=float, default=0.35)
    parser.add_argument("--w_coordinate_anomaly", type=float, default=0.15)
    parser.add_argument("--w_spectral", type=float, default=0.10)
    parser.add_argument("--w_population", type=float, default=0.15)
    parser.add_argument("--w_temporal", type=float, default=0.05)
    parser.add_argument("--w_heterogeneity", type=float, default=0.05)
    parser.add_argument("--w_correlation", type=float, default=0.15)

    # v2: Coordinate-wise anomaly and adaptive weighting params
    parser.add_argument("--cwas_mad_threshold", type=float, default=3.0,
                        help="MAD threshold for coordinate-wise anomaly detection")
    parser.add_argument("--cwas_subsample_dim", type=int, default=5000,
                        help="Subsample dimensionality for CWAS speed")
    parser.add_argument("--cwas_sign_weight", type=float, default=0.0,
                        help="Weight of sign-agreement in CWAS scoring")
    parser.add_argument("--adaptive_weights", action="store_true", default=True,
                        help="Enable adaptive component weighting")
    parser.add_argument("--no_adaptive_weights", action="store_true", default=False,
                        help="Disable adaptive component weighting")
    parser.add_argument("--adaptive_blend", type=float, default=0.5,
                        help="Blend ratio of adaptive vs prior weights")

    # v3: New innovation weights
    parser.add_argument("--w_loo_validation", type=float, default=0.20,
                        help="Weight for LOO validation trust component")
    parser.add_argument("--w_gradient_inversion", type=float, default=0.15,
                        help="Weight for gradient inversion score component")
    parser.add_argument("--w_trajectory", type=float, default=0.10,
                        help="Weight for temporal trajectory consistency component")

    # v3: Bayesian trust
    parser.add_argument("--use_bayesian_trust", action="store_true", default=True,
                        help="Use Bayesian Beta posterior for trust update")
    parser.add_argument("--no_bayesian_trust", action="store_true", default=False,
                        help="Disable Bayesian trust (use EMA instead)")
    parser.add_argument("--bayesian_evidence_scale", type=float, default=1.5,
                        help="Evidence scale for Bayesian trust update")
    parser.add_argument("--bayesian_decay", type=float, default=0.995,
                        help="Decay rate for Bayesian trust posterior")
    parser.add_argument("--functional_first_rounds", type=int, default=15,
                        help="Rounds using only functional evaluators (LOO+delta) for trust initialization")

    # Ablation: ADI mode and base aggregator
    parser.add_argument("--adi_mode", type=str, default="full",
                        choices=["full", "disabled", "cv_only", "no_bat"],
                        help="ADI mode for ablation studies")
    parser.add_argument("--base_aggregator", type=str, default="trimmed_mean",
                        choices=["trimmed_mean", "geometric_median"],
                        help="Base aggregator for trust-weighted method")

    # Evaluation
    parser.add_argument("--eval_frequency", type=int, default=5)
    parser.add_argument("--eval_episodes", type=int, default=10)

    # System
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_suffix", type=str, default="",
                        help="Suffix to append to log directory name (e.g., '_v2')")

    args = parser.parse_args()

    # Handle no_adaptive_weights flag
    if args.no_adaptive_weights:
        args.adaptive_weights = False

    # Handle no_bayesian_trust flag
    if args.no_bayesian_trust:
        args.use_bayesian_trust = False

    # Set seeds for full reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print_header(args)

    # Build and train
    server, experiment_name = build_experiment(args)

    t_start = time.time()
    server.train(n_rounds=args.rounds, save_frequency=25)
    total_time = time.time() - t_start

    # Final summary
    print()
    print("=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Rounds: {args.rounds}")

    if server.metrics_history:
        final = server.metrics_history[-1]
        print(f"  Final train reward: {final.get('train_reward_mean', 'N/A'):.4f}")
        if "eval_return_mean" in final:
            print(f"  Final eval return:  {final['eval_return_mean']:.4f}")
        if "trust_tpr" in final:
            print(f"  Final Trust TPR:    {final['trust_tpr']:.4f}")
            print(f"  Final Trust FPR:    {final['trust_fpr']:.4f}")

    # Compute averages over last 20 rounds
    last_n = min(20, len(server.metrics_history))
    if last_n > 0:
        avg_reward = np.mean([
            m["train_reward_mean"] for m in server.metrics_history[-last_n:]
        ])
        tpr_values = [m["trust_tpr"] for m in server.metrics_history[-last_n:] if "trust_tpr" in m]
        fpr_values = [m["trust_fpr"] for m in server.metrics_history[-last_n:] if "trust_fpr" in m]

        print(f"\n  === Last {last_n} rounds average ===")
        print(f"  Avg train reward:   {avg_reward:.4f}")
        if tpr_values:
            print(f"  Avg Trust TPR:      {np.mean(tpr_values):.4f}")
            print(f"  Avg Trust FPR:      {np.mean(fpr_values):.4f}")

    print(f"\n  Logs saved to: logs/{experiment_name}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
