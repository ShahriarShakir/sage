"""
frl/server.py — Federated Server: Coordinates Training Rounds
Created: 2026-02-26

The server:
  1. Maintains the global model
  2. Distributes weights to clients
  3. Collects deltas from clients (with attack perturbation for Byzantine ones)
  4. Runs trust scoring (HATT)
  5. Aggregates deltas using selected robust aggregation
  6. Logs per-round metrics
"""

from __future__ import annotations

import copy
import time
import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from frl.models import ActorCritic
from frl.client import FRLClient
from frl.agg import get_aggregator, fedavg
from frl.trust import HATTTrustScorer
from frl.attacks import Attack

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class FRLServer:
    """
    Federated RL Server.

    Orchestrates the federated training loop:
      for each round:
        1. broadcast global model to selected clients
        2. clients run K local updates → return deltas
        3. (Byzantine clients apply attacks to their deltas)
        4. server computes trust scores via HATT
        5. server aggregates deltas using robust aggregation
        6. server updates global model
        7. server evaluates and logs metrics
    """

    def __init__(
        self,
        global_model: ActorCritic,
        clients: List[FRLClient],
        # Byzantine configuration
        byzantine_ids: Optional[List[int]] = None,
        attacks: Optional[Dict[int, Attack]] = None,
        # Aggregation
        aggregator_name: str = "fedavg",
        aggregator_kwargs: Optional[Dict[str, Any]] = None,
        # Trust scoring
        trust_scorer: Optional[HATTTrustScorer] = None,
        use_trust: bool = True,
        # Evaluation
        eval_env_factory=None,
        eval_frequency: int = 5,
        eval_episodes: int = 10,
        eval_seed: int = 9999,
        # Logging
        log_dir: str = "logs",
        experiment_name: str = "frl_experiment",
        # Misc
        device: str = "cpu",
    ):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.n_clients = len(clients)
        self.device = device

        # Byzantine setup
        self.byzantine_ids = set(byzantine_ids or [])
        self.attacks = attacks or {}

        # Aggregation
        self.aggregator_fn = get_aggregator(aggregator_name)
        self.aggregator_name = aggregator_name
        self.aggregator_kwargs = aggregator_kwargs or {}

        # Trust
        self.trust_scorer = trust_scorer
        self.use_trust = use_trust and trust_scorer is not None

        # Evaluation
        self.eval_env_factory = eval_env_factory
        self.eval_frequency = eval_frequency
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed

        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        # Metrics history
        self.metrics_history: List[Dict[str, Any]] = []
        self.trust_history: List[Dict[int, float]] = []
        self.round_times: List[float] = []
        self._current_round = 0

        # Faithful-baseline server-side state (added 2026-04-27)
        # FoolsGold cumulative per-client gradient history (keyed by client id)
        self._fg_history: Dict[int, torch.Tensor] = {}
        # FLTrust rolling reference (last round's robust aggregate, flattened)
        self._fltrust_reference: Optional[torch.Tensor] = None

    # ---- Main training loop -------------------------------------------

    def train(self, n_rounds: int, save_frequency: int = 50):
        """Run the full federated training loop."""
        logger.info(
            f"Starting FRL training: {n_rounds} rounds, "
            f"{self.n_clients} clients, "
            f"{len(self.byzantine_ids)} Byzantine, "
            f"aggregator={self.aggregator_name}"
        )

        for rnd in range(n_rounds):
            self._current_round = rnd
            t0 = time.time()

            round_metrics = self.run_round(rnd)

            elapsed = time.time() - t0
            self.round_times.append(elapsed)
            round_metrics["round"] = rnd
            round_metrics["time_s"] = elapsed
            self.metrics_history.append(round_metrics)

            # Periodic evaluation
            if rnd % self.eval_frequency == 0:
                eval_metrics = self.evaluate(rnd)
                round_metrics.update(eval_metrics)

            # Logging
            self._log_round(rnd, round_metrics)

            # Save checkpoint
            if (rnd + 1) % save_frequency == 0:
                self.save_checkpoint(rnd)

        # Final save
        self.save_checkpoint(n_rounds - 1)
        self.save_metrics()
        logger.info("Training complete.")

    def run_round(self, round_idx: int) -> Dict[str, Any]:
        """Execute a single federated round."""
        global_state = self.global_model.state_dict()

        # 1. Broadcast global model and collect deltas
        # Two-pass: first collect honest deltas, then apply attacks
        # (some attacks like NormalizedAttack need the honest aggregate)
        raw_deltas = {}  # cid -> delta
        client_stats_map = {}
        client_models_map = {}
        participating_ids = []

        for client in self.clients:
            cid = client.client_id

            # Check if client participates (stale/dropout attack may skip)
            if cid in self.attacks:
                if not self.attacks[cid].should_participate(round_idx, cid):
                    continue

            # Send global model
            client.receive_global_model(global_state)

            # Local training
            delta, stats = client.run_round()

            raw_deltas[cid] = delta
            client_stats_map[cid] = stats
            participating_ids.append(cid)
            client_models_map[cid] = copy.deepcopy(client.model)

        if len(raw_deltas) == 0:
            logger.warning(f"Round {round_idx}: no clients participated!")
            return {"n_participating": 0}

        # Compute honest aggregate (mean of non-Byzantine deltas) for attacks
        honest_cids = [c for c in participating_ids if c not in self.byzantine_ids]
        honest_aggregate = None
        if honest_cids:
            first_key = list(raw_deltas[honest_cids[0]].keys())
            honest_aggregate = {}
            for k in first_key:
                honest_aggregate[k] = torch.stack(
                    [raw_deltas[c][k].float() for c in honest_cids]
                ).mean(dim=0)

        # Apply attacks (with honest aggregate context)
        deltas = []
        client_stats = []
        client_models = []
        for cid in participating_ids:
            delta = raw_deltas[cid]
            if cid in self.byzantine_ids and cid in self.attacks:
                delta = self.attacks[cid].perturb_delta(
                    delta, round_idx, cid,
                    honest_aggregate=honest_aggregate,
                )
            deltas.append(delta)
            client_stats.append(client_stats_map[cid])
            client_models.append(client_models_map[cid])

        # 2. Trust scoring
        trust_scores = [1.0] * len(deltas)
        if self.use_trust:
            trust_scores = self.trust_scorer.update_all(
                deltas, participating_ids,
                models=client_models,
                global_model=self.global_model,
                device=self.device,
            )
            self.trust_history.append(
                {cid: ts for cid, ts in zip(participating_ids, trust_scores)}
            )

        # 3. Aggregate
        if self.aggregator_name == "trust_weighted":
            agg_delta = self.aggregator_fn(
                deltas, trust_scores=trust_scores, **self.aggregator_kwargs
            )
        elif self.aggregator_name == "krum":
            n_byz = len([cid for cid in participating_ids if cid in self.byzantine_ids])
            agg_delta = self.aggregator_fn(deltas, n_byzantine=max(n_byz, 1))
        elif self.aggregator_name == "multi_krum":
            n_byz = len([cid for cid in participating_ids if cid in self.byzantine_ids])
            agg_delta = self.aggregator_fn(deltas, n_byzantine=max(n_byz, 1))
        elif self.aggregator_name == "fltrust":
            # Use trimmed mean of honest updates as server reference approximation
            agg_delta = self.aggregator_fn(deltas, server_delta=None)
        elif self.aggregator_name == "fltrust_lagged":
            # Faithful FLTrust variant: rolling reference produced by the
            # previous round's aggregate, computed below.
            agg_delta = self.aggregator_fn(
                deltas, server_reference=self._fltrust_reference
            )
        elif self.aggregator_name == "flame":
            n_byz = len([cid for cid in participating_ids if cid in self.byzantine_ids])
            agg_delta = self.aggregator_fn(deltas, n_byzantine=max(n_byz, 1))
        elif self.aggregator_name == "flame_hdbscan":
            n_byz = len([cid for cid in participating_ids if cid in self.byzantine_ids])
            agg_delta = self.aggregator_fn(deltas, n_byzantine=max(n_byz, 1))
        elif self.aggregator_name == "foolsgold":
            agg_delta = self.aggregator_fn(deltas)
        elif self.aggregator_name == "foolsgold_hist":
            # Update cumulative history then aggregate using histories
            for cid, d in zip(participating_ids, deltas):
                v = torch.cat([t.reshape(-1).float() for t in d.values()])
                if cid in self._fg_history and self._fg_history[cid].numel() == v.numel():
                    self._fg_history[cid] = self._fg_history[cid] + v
                else:
                    self._fg_history[cid] = v.clone()
            history_list = [self._fg_history[cid] for cid in participating_ids]
            agg_delta = self.aggregator_fn(deltas, history=history_list)
        elif self.aggregator_name == "trimmed_mean":
            trim_kwargs = {k: v for k, v in self.aggregator_kwargs.items()
                          if k == "trim_fraction"}
            agg_delta = self.aggregator_fn(deltas, **trim_kwargs)
        elif self.aggregator_name == "geometric_median":
            agg_delta = self.aggregator_fn(deltas)
        elif self.aggregator_name == "fedavg":
            agg_delta = self.aggregator_fn(deltas)
        else:
            agg_delta = self.aggregator_fn(deltas)

        # 4. Update global model
        self.global_model.apply_delta(agg_delta)

        # Update rolling FLTrust reference: keep an EMA-smoothed flattened
        # aggregate from previous rounds. This is the reference the next
        # round will score clients against.
        if self.aggregator_name == "fltrust_lagged":
            agg_flat = torch.cat([t.reshape(-1).float() for t in agg_delta.values()])
            if self._fltrust_reference is None or self._fltrust_reference.numel() != agg_flat.numel():
                self._fltrust_reference = agg_flat.clone()
            else:
                self._fltrust_reference = 0.7 * self._fltrust_reference + 0.3 * agg_flat

        # 5. Compute round metrics
        rewards = [s["rollout_reward_mean"] for s in client_stats]

        metrics = {
            "n_participating": len(deltas),
            "train_reward_mean": float(np.mean(rewards)),
            "train_reward_std": float(np.std(rewards)),
            "train_reward_min": float(np.min(rewards)),
            "train_reward_max": float(np.max(rewards)),
        }

        # Trust metrics (TPR/FPR if we know ground truth)
        if self.use_trust:
            metrics.update(
                self._compute_trust_metrics(participating_ids, trust_scores)
            )

        # Communication cost: total parameter delta norms
        delta_norms = []
        for d in deltas:
            norm = sum(v.float().norm().item() ** 2 for v in d.values()) ** 0.5
            delta_norms.append(norm)
        metrics["comm_cost_mean_norm"] = float(np.mean(delta_norms))

        return metrics

    def _compute_trust_metrics(
        self, client_ids: List[int], trust_scores: List[float]
    ) -> Dict[str, float]:
        """Compute trust TPR / FPR given ground-truth Byzantine IDs."""
        tp = fp = tn = fn = 0
        # Use same threshold as aggregation for consistency
        trust_threshold = self.aggregator_kwargs.get("trust_threshold", 0.5)

        for cid, score in zip(client_ids, trust_scores):
            is_byzantine = cid in self.byzantine_ids
            is_flagged = score < trust_threshold

            if is_byzantine and is_flagged:
                tp += 1
            elif is_byzantine and not is_flagged:
                fn += 1
            elif not is_byzantine and is_flagged:
                fp += 1
            else:
                tn += 1

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)

        return {
            "trust_tpr": tpr,
            "trust_fpr": fpr,
            "trust_tp": tp,
            "trust_fp": fp,
            "trust_tn": tn,
            "trust_fn": fn,
            "trust_mean_honest": float(np.mean([
                s for cid, s in zip(client_ids, trust_scores)
                if cid not in self.byzantine_ids
            ])) if any(cid not in self.byzantine_ids for cid in client_ids) else 0.0,
            "trust_mean_byzantine": float(np.mean([
                s for cid, s in zip(client_ids, trust_scores)
                if cid in self.byzantine_ids
            ])) if any(cid in self.byzantine_ids for cid in client_ids) else 0.0,
        }

    # ---- Evaluation ---------------------------------------------------

    def evaluate(self, round_idx: int) -> Dict[str, float]:
        """Evaluate global model on the evaluation environment."""
        if self.eval_env_factory is None:
            return {}

        env = self.eval_env_factory()
        model = copy.deepcopy(self.global_model).to(self.device)
        model.eval()

        returns = []
        for ep in range(self.eval_episodes):
            obs, _ = env.reset(seed=self.eval_seed + ep)
            obs_t = torch.FloatTensor(obs).to(self.device)
            ep_return = 0.0
            done = False
            steps = 0
            while not done and steps < 1000:
                with torch.no_grad():
                    action, _, _ = model.act(obs_t, deterministic=True)
                action_np = action.cpu().numpy()
                if action_np.ndim == 0 or (hasattr(action_np, 'shape') and action_np.size == 1):
                    action_np = action_np.item()
                # Clip continuous actions to bounds
                if hasattr(env, 'action_space') and hasattr(env.action_space, 'low'):
                    action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action_np)
                obs_t = torch.FloatTensor(obs).to(self.device)
                ep_return += reward
                done = terminated or truncated
                steps += 1
            returns.append(ep_return)
        env.close()

        return {
            "eval_return_mean": float(np.mean(returns)),
            "eval_return_std": float(np.std(returns)),
            "eval_return_min": float(np.min(returns)),
            "eval_return_max": float(np.max(returns)),
        }

    # ---- Logging & checkpointing --------------------------------------

    def _log_round(self, round_idx: int, metrics: Dict[str, Any]):
        """Log round metrics."""
        msg = f"Round {round_idx:4d}"
        for key in ["train_reward_mean", "eval_return_mean", "trust_tpr", "trust_fpr",
                     "trust_mean_honest", "trust_mean_byzantine"]:
            if key in metrics:
                msg += f" | {key}={metrics[key]:.4f}"
        msg += f" | time={metrics.get('time_s', 0):.2f}s"
        logger.info(msg)

    def save_checkpoint(self, round_idx: int):
        """Save model and training state."""
        ckpt_path = self.log_dir / f"checkpoint_round_{round_idx}.pt"
        torch.save({
            "round": round_idx,
            "global_model": self.global_model.state_dict(),
            "metrics_history": self.metrics_history,
            "trust_history": self.trust_history,
        }, ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")

    def save_metrics(self):
        """Save all metrics to JSON."""
        metrics_path = self.log_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            # Convert non-serializable items
            serializable = []
            for m in self.metrics_history:
                s = {}
                for k, v in m.items():
                    if isinstance(v, (int, float, str, bool)):
                        s[k] = v
                    else:
                        s[k] = str(v)
                serializable.append(s)
            json.dump(serializable, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")

    def load_checkpoint(self, path: str):
        """Load a checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.global_model.load_state_dict(ckpt["global_model"])
        self.metrics_history = ckpt.get("metrics_history", [])
        self.trust_history = ckpt.get("trust_history", [])
        self._current_round = ckpt.get("round", 0)
        logger.info(f"Loaded checkpoint from {path}, round {self._current_round}")
