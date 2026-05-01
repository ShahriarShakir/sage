#!/usr/bin/env python3
"""
scripts/run_experiment.py — Run a Single FRL Experiment
Created: 2026-02-26

Usage:
  python scripts/run_experiment.py --config configs/default.yaml
  python scripts/run_experiment.py --config configs/default.yaml \
      attack.type=scaling aggregation.method=fedavg
"""

from __future__ import annotations

import sys
import os
import argparse
import logging
import yaml
import copy
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frl.models import ActorCritic
from frl.client import FRLClient
from frl.server import FRLServer
from frl.trust import HATTTrustScorer
from frl.attacks import get_attack, Attack, SybilAttack


def setup_logging(log_dir: str, level: str = "INFO"):
    """Configure logging to file and console."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, "train.log")),
        ],
    )


def load_config(config_path: str, overrides: list = None) -> dict:
    """Load YAML config and apply CLI overrides."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if overrides:
        for override in overrides:
            if "=" not in override:
                continue
            key, value = override.split("=", 1)
            # Navigate nested keys
            keys = key.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            # Parse value
            try:
                d[keys[-1]] = yaml.safe_load(value)
            except yaml.YAMLError:
                d[keys[-1]] = value

    return cfg


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_env(cfg: dict, agent_idx: int = 0):
    """Create environment based on config."""
    env_type = cfg["env"]["type"]

    if env_type == "mpe":
        from frl.envs.mpe_wrapper import make_mpe_env
        return make_mpe_env(
            scenario=cfg["env"]["mpe_scenario"],
            agent_idx=agent_idx,
            max_cycles=cfg["env"]["mpe_max_cycles"],
            continuous_actions=cfg["env"]["mpe_continuous"],
        )
    elif env_type == "smac":
        from frl.envs.smac_wrapper import make_smac_env
        return make_smac_env(
            map_name=cfg["env"]["smac_map"],
            agent_idx=agent_idx,
            max_steps=cfg["env"]["smac_max_steps"],
        )
    elif env_type == "gym":
        from frl.envs.gym_wrapper import make_gym_env
        return make_gym_env(env_id=cfg["env"]["gym_id"])
    else:
        raise ValueError(f"Unknown env type: {env_type}")


def create_env_factory(cfg: dict, agent_idx: int = 0):
    """Return a callable that creates the env."""
    def factory():
        return create_env(cfg, agent_idx)
    return factory


def create_attack(cfg: dict) -> Attack:
    """Create attack instance from config."""
    attack_type = cfg["attack"]["type"]

    if attack_type == "none":
        return Attack()
    elif attack_type == "sign_flip":
        return get_attack("sign_flip", scale=cfg["attack"]["sign_flip_scale"])
    elif attack_type == "scaling":
        return get_attack("scaling", scale_factor=cfg["attack"]["scaling_factor"])
    elif attack_type == "gaussian_noise":
        return get_attack(
            "gaussian_noise",
            noise_std=cfg["attack"]["noise_std"],
            relative=cfg["attack"]["noise_relative"],
        )
    elif attack_type == "directional":
        return get_attack("directional", magnitude=cfg["attack"]["directional_magnitude"])
    elif attack_type == "reward_bias":
        return get_attack("reward_bias", bias=cfg["attack"]["reward_bias"])
    elif attack_type == "reward_sparse_trigger":
        return get_attack(
            "reward_sparse_trigger",
            trigger_prob=cfg["attack"]["trigger_prob"],
            trigger_reward=cfg["attack"]["trigger_reward"],
        )
    elif attack_type == "stale_update":
        return get_attack(
            "stale_update",
            delay_rounds=cfg["attack"]["delay_rounds"],
            dropout_prob=cfg["attack"]["dropout_prob"],
        )
    elif attack_type == "sybil":
        inner = create_attack(
            {**cfg, "attack": {**cfg["attack"], "type": cfg["attack"]["sybil_inner_attack"]}}
        )
        return SybilAttack(
            inner_attack=inner,
            n_sybils=cfg["attack"]["n_sybils"],
        )
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")


def run_experiment(cfg: dict):
    """Run a complete federated RL experiment."""
    seed = cfg["experiment"]["seed"]
    device = cfg["experiment"]["device"]
    log_dir = cfg["experiment"]["log_dir"]

    set_seed(seed)
    setup_logging(log_dir)
    logger = logging.getLogger(__name__)

    logger.info(f"Config: {yaml.dump(cfg, default_flow_style=False)}")

    # Determine dimensions from a probe env
    probe_env = create_env(cfg, agent_idx=0)
    obs_dim = probe_env.observation_space.shape[0]

    if hasattr(probe_env.action_space, 'n'):
        act_dim = probe_env.action_space.n
        continuous = False
    else:
        act_dim = probe_env.action_space.shape[0]
        continuous = True
    probe_env.close()

    logger.info(f"Env: obs_dim={obs_dim}, act_dim={act_dim}, continuous={continuous}")

    # Hidden dims
    hidden_dims = cfg["client"]["hidden_dims"]

    # Create global model
    global_model = ActorCritic(obs_dim, act_dim, hidden_dims, continuous)

    # Byzantine client assignment
    n_clients = cfg["federation"]["n_clients"]
    n_byzantine = int(cfg["federation"]["byzantine_fraction"] * n_clients)
    byzantine_ids = list(range(n_clients - n_byzantine, n_clients))  # last N are byzantine
    logger.info(f"N={n_clients}, Byzantine={n_byzantine}, IDs={byzantine_ids}")

    # Create attack
    attack = create_attack(cfg)
    attacks = {cid: attack for cid in byzantine_ids}

    # Create clients
    clients = []
    for cid in range(n_clients):
        env = create_env(cfg, agent_idx=cid % 3)  # distribute across agents
        client = FRLClient(
            client_id=cid,
            env=env,
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=hidden_dims,
            continuous=continuous,
            lr_actor=cfg["client"]["lr_actor"],
            lr_critic=cfg["client"]["lr_critic"],
            gamma=cfg["client"]["gamma"],
            gae_lambda=cfg["client"]["gae_lambda"],
            clip_eps=cfg["client"]["clip_eps"],
            entropy_coef=cfg["client"]["entropy_coef"],
            value_coef=cfg["client"]["value_coef"],
            max_grad_norm=cfg["client"]["max_grad_norm"],
            local_epochs=cfg["client"]["local_epochs"],
            rollout_steps=cfg["client"]["rollout_steps"],
            minibatch_size=cfg["client"]["minibatch_size"],
            device=device,
        )
        clients.append(client)

    # Create trust scorer
    trust_scorer = None
    if cfg["trust"]["enabled"]:
        audit_factory = create_env_factory(cfg, agent_idx=0)
        trust_scorer = HATTTrustScorer(
            n_clients=n_clients,
            ema_beta=cfg["trust"]["ema_beta"],
            high_threshold=cfg["trust"]["high_threshold"],
            low_threshold=cfg["trust"]["low_threshold"],
            hysteresis_window=cfg["trust"]["hysteresis_window"],
            envelope_window=cfg["trust"]["envelope_window"],
            z_score_threshold=cfg["trust"]["z_score_threshold"],
            audit_env_factory=audit_factory,
            audit_seeds=cfg["trust"]["audit_seeds"],
            audit_steps=cfg["trust"]["audit_steps"],
            audit_frequency=cfg["trust"]["audit_frequency"],
            w_temporal=cfg["trust"]["w_temporal"],
            w_heterogeneity=cfg["trust"]["w_heterogeneity"],
            w_audit=cfg["trust"]["w_audit"],
            trust_ema_beta=cfg["trust"]["trust_ema_beta"],
        )

    # Create server
    eval_factory = create_env_factory(cfg, agent_idx=0)
    server = FRLServer(
        global_model=global_model,
        clients=clients,
        byzantine_ids=byzantine_ids,
        attacks=attacks,
        aggregator_name=cfg["aggregation"]["method"],
        aggregator_kwargs={
            "trim_fraction": cfg["aggregation"]["trim_fraction"],
            "trust_threshold": cfg["aggregation"]["trust_threshold"],
            "base_aggregator": cfg["aggregation"]["base_aggregator"],
            "filter_anomalous": cfg["aggregation"]["filter_anomalous"],
        },
        trust_scorer=trust_scorer,
        use_trust=cfg["trust"]["enabled"],
        eval_env_factory=eval_factory,
        eval_frequency=cfg["federation"]["eval_frequency"],
        eval_episodes=cfg["federation"]["eval_episodes"],
        log_dir=log_dir,
        experiment_name=cfg["experiment"]["name"],
        device=device,
    )

    # Train
    server.train(
        n_rounds=cfg["federation"]["n_rounds"],
        save_frequency=cfg["federation"]["save_frequency"],
    )

    return server.metrics_history


def main():
    parser = argparse.ArgumentParser(description="Run FRL Experiment")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config file",
    )
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
