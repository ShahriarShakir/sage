"""
frl/eval.py — Evaluation & Metrics for Federated MARL
Created: 2026-02-26

Computes:
  - Per-round return mean/std
  - Worst-case return in attack window
  - Recovery time after attack
  - Trust TPR / FPR and detection delay
  - Communication cost
  - Generates summary tables and plots
"""

from __future__ import annotations

import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

class FRLEvaluator:
    """Post-hoc evaluation of federated training runs."""

    def __init__(self, metrics_history: List[Dict[str, Any]]):
        self.metrics = metrics_history
        self.n_rounds = len(metrics_history)

    @classmethod
    def from_json(cls, path: str) -> "FRLEvaluator":
        with open(path) as f:
            data = json.load(f)
        return cls(data)

    # ---- Return metrics -----------------------------------------------

    def get_returns(self, key: str = "eval_return_mean") -> np.ndarray:
        vals = [m.get(key, np.nan) for m in self.metrics]
        return np.array(vals, dtype=float)

    def worst_case_return(
        self,
        attack_start: int,
        attack_end: int,
        key: str = "eval_return_mean",
    ) -> float:
        """Worst return within the attack window."""
        returns = self.get_returns(key)
        window = returns[attack_start:attack_end + 1]
        valid = window[~np.isnan(window)]
        return float(np.min(valid)) if len(valid) > 0 else float('nan')

    def recovery_time(
        self,
        attack_end: int,
        baseline_return: float,
        recovery_fraction: float = 0.9,
        key: str = "eval_return_mean",
    ) -> int:
        """
        Number of rounds after attack_end until return recovers to
        recovery_fraction * baseline_return.
        Returns -1 if never recovered.
        """
        returns = self.get_returns(key)
        threshold = recovery_fraction * baseline_return
        for t in range(attack_end + 1, len(returns)):
            if not np.isnan(returns[t]) and returns[t] >= threshold:
                return t - attack_end
        return -1

    # ---- Trust metrics ------------------------------------------------

    def get_trust_series(self, key: str = "trust_tpr") -> np.ndarray:
        vals = [m.get(key, np.nan) for m in self.metrics]
        return np.array(vals, dtype=float)

    def detection_delay(
        self,
        attack_start: int,
        tpr_threshold: float = 0.8,
    ) -> int:
        """
        Rounds after attack_start until TPR exceeds tpr_threshold.
        Returns -1 if never reached.
        """
        tpr = self.get_trust_series("trust_tpr")
        for t in range(attack_start, len(tpr)):
            if not np.isnan(tpr[t]) and tpr[t] >= tpr_threshold:
                return t - attack_start
        return -1

    def avg_trust_fpr(self) -> float:
        fpr = self.get_trust_series("trust_fpr")
        valid = fpr[~np.isnan(fpr)]
        return float(np.mean(valid)) if len(valid) > 0 else float('nan')

    def avg_trust_tpr(self, attack_start: int = 0) -> float:
        tpr = self.get_trust_series("trust_tpr")
        valid = tpr[attack_start:]
        valid = valid[~np.isnan(valid)]
        return float(np.mean(valid)) if len(valid) > 0 else float('nan')

    # ---- Communication cost -------------------------------------------

    def total_comm_cost(self) -> float:
        costs = [m.get("comm_cost_mean_norm", 0.0) for m in self.metrics]
        return float(sum(costs))

    # ---- Summary table ------------------------------------------------

    def summary(
        self,
        attack_start: int = 0,
        attack_end: Optional[int] = None,
        baseline_return: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate a summary dictionary of all key metrics."""
        if attack_end is None:
            attack_end = self.n_rounds - 1

        returns = self.get_returns()
        valid_returns = returns[~np.isnan(returns)]

        summary = {
            "n_rounds": self.n_rounds,
            "final_return": float(valid_returns[-1]) if len(valid_returns) > 0 else np.nan,
            "mean_return": float(np.mean(valid_returns)) if len(valid_returns) > 0 else np.nan,
            "worst_case_return": self.worst_case_return(attack_start, attack_end),
            "avg_trust_tpr": self.avg_trust_tpr(attack_start),
            "avg_trust_fpr": self.avg_trust_fpr(),
            "detection_delay": self.detection_delay(attack_start),
            "total_comm_cost": self.total_comm_cost(),
        }

        if baseline_return is not None:
            summary["recovery_time"] = self.recovery_time(attack_end, baseline_return)

        return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

class FRLPlotter:
    """Generate publication-quality figures for the paper."""

    def __init__(self, save_dir: str = "figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams.update({
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "figure.figsize": (8, 5),
        })

    def plot_returns(
        self,
        evaluators: Dict[str, FRLEvaluator],
        title: str = "Evaluation Return Over Rounds",
        filename: str = "returns.pdf",
        attack_window: Optional[Tuple[int, int]] = None,
    ):
        """Plot return curves for multiple methods."""
        fig, ax = plt.subplots()

        for label, ev in evaluators.items():
            returns = ev.get_returns()
            rounds = np.arange(len(returns))
            valid_mask = ~np.isnan(returns)
            ax.plot(rounds[valid_mask], returns[valid_mask], label=label, linewidth=2)

        if attack_window:
            ax.axvspan(
                attack_window[0], attack_window[1],
                alpha=0.15, color="red", label="Attack Window"
            )

        ax.set_xlabel("Round")
        ax.set_ylabel("Evaluation Return")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {filename}")

    def plot_trust_scores(
        self,
        trust_history: List[Dict[int, float]],
        byzantine_ids: set,
        title: str = "Trust Scores Over Time",
        filename: str = "trust_scores.pdf",
    ):
        """Plot per-client trust scores over rounds."""
        fig, ax = plt.subplots()

        if not trust_history:
            return

        all_ids = set()
        for entry in trust_history:
            all_ids.update(entry.keys())

        for cid in sorted(all_ids):
            scores = [entry.get(cid, np.nan) for entry in trust_history]
            rounds = np.arange(len(scores))
            style = "--" if cid in byzantine_ids else "-"
            color = "red" if cid in byzantine_ids else "blue"
            alpha = 0.8 if cid in byzantine_ids else 0.5
            label = f"Client {cid}" + (" (Byz)" if cid in byzantine_ids else "")
            ax.plot(rounds, scores, style, color=color, alpha=alpha, label=label)

        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Threshold")
        ax.set_xlabel("Round")
        ax.set_ylabel("Trust Score")
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {filename}")

    def plot_trust_tpr_fpr(
        self,
        evaluator: FRLEvaluator,
        title: str = "Trust Detection TPR / FPR",
        filename: str = "trust_tpr_fpr.pdf",
    ):
        """Plot TPR and FPR over rounds."""
        fig, ax = plt.subplots()
        tpr = evaluator.get_trust_series("trust_tpr")
        fpr = evaluator.get_trust_series("trust_fpr")
        rounds = np.arange(len(tpr))

        valid_tpr = ~np.isnan(tpr)
        valid_fpr = ~np.isnan(fpr)

        ax.plot(rounds[valid_tpr], tpr[valid_tpr], "g-", label="TPR", linewidth=2)
        ax.plot(rounds[valid_fpr], fpr[valid_fpr], "r-", label="FPR", linewidth=2)
        ax.set_xlabel("Round")
        ax.set_ylabel("Rate")
        ax.set_title(title)
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {filename}")

    def plot_comparison_bar(
        self,
        results: Dict[str, Dict[str, float]],
        metric_key: str = "final_return",
        title: str = "Method Comparison",
        filename: str = "comparison.pdf",
    ):
        """Bar chart comparing methods on a specific metric."""
        fig, ax = plt.subplots()
        methods = list(results.keys())
        values = [results[m].get(metric_key, 0) for m in methods]

        bars = ax.bar(methods, values, color=sns.color_palette("Set2", len(methods)))
        ax.set_ylabel(metric_key.replace("_", " ").title())
        ax.set_title(title)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=10,
            )

        fig.tight_layout()
        fig.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {filename}")
