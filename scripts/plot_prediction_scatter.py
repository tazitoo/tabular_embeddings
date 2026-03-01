#!/usr/bin/env python3
"""Scatter plot of predicted probabilities from two models on the same dataset.

Produces a scatter of P(class=1) from model A vs model B on the same query rows,
with y=x reference line and event-rate decision boundaries.

Usage:
    python scripts/plot_prediction_scatter.py --dataset kddcup09_appetency \
        --model-a tabpfn --model-b tabicl --device cuda
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.concept_performance_diagnostic import (
    _load_splits, compute_metric, DISPLAY_NAMES,
)

logger = logging.getLogger(__name__)


def get_predictions(model_key: str, dataset: str, task: str, device: str) -> np.ndarray:
    """Get P(class=1) predictions from a model on query rows."""
    from scripts.concept_performance_diagnostic import (
        predict_intervention_model, predict_hyperfast,
    )

    if model_key == "hyperfast":
        result = predict_hyperfast(dataset, task, device)
    else:
        result = predict_intervention_model(model_key, dataset, task, device)

    preds = result["preds"]
    # For binary classification, extract P(class=1)
    if preds.ndim == 2:
        return preds[:, 1]
    return preds


def get_active_feature_count(model_key: str, dataset: str) -> int:
    """Get number of SAE features that fire on a specific dataset.

    Reads from importance JSON (authoritative — counts features that fire on
    at least one query row). Falls back to -1 if not available.
    """
    imp_path = PROJECT_ROOT / "output" / "concept_importance" / f"{model_key}_{dataset}.json"
    if imp_path.exists():
        with open(imp_path) as f:
            imp = json.load(f)
        n = imp.get("n_active_features", -1)
        if n >= 0:
            return n
    return -1


def plot_prediction_scatter(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    y_true: np.ndarray,
    model_a: str,
    model_b: str,
    dataset: str,
    auc_a: float,
    auc_b: float,
    features_a: int,
    features_b: int,
    output_path: Path,
):
    """Create scatter plot of predictions from two models."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    event_rate = y_true.mean()

    # Auto-zoom to data range with padding
    all_preds = np.concatenate([preds_a, preds_b])
    lo = max(0, all_preds.min() - 0.02)
    hi = min(1, all_preds.max() + 0.02)
    # Ensure event rate is visible
    hi = max(hi, event_rate + 0.02)

    # Plot class 0 first (background), then class 1 on top
    mask0 = y_true == 0
    mask1 = y_true == 1
    ax.scatter(preds_a[mask0], preds_b[mask0], c="#1f77b4", s=15, alpha=0.5,
               edgecolors="none", label=f"Class 0 (n={mask0.sum()})", zorder=2)
    ax.scatter(preds_a[mask1], preds_b[mask1], c="#d62728", s=40, alpha=0.9,
               edgecolors="k", linewidths=0.5, label=f"Class 1 (n={mask1.sum()})", zorder=3)

    # y=x reference line
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y = x")

    # Event rate lines
    ax.axhline(event_rate, color="gray", lw=0.7, ls=":", alpha=0.7)
    ax.axvline(event_rate, color="gray", lw=0.7, ls=":", alpha=0.7)
    # Place label in top-left area of the plot
    ax.text(0.97, event_rate, f" event rate = {event_rate:.3f}",
            fontsize=7, color="gray", va="bottom", ha="right",
            transform=ax.get_yaxis_transform())

    disp_a = DISPLAY_NAMES.get(model_a, model_a)
    disp_b = DISPLAY_NAMES.get(model_b, model_b)

    ax.set_xlabel(f"{disp_a}  P(class=1)", fontsize=10)
    ax.set_ylabel(f"{disp_b}  P(class=1)", fontsize=10)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")

    # Title with AUC and feature counts
    feat_a_str = f"{features_a} active" if features_a >= 0 else "? active"
    feat_b_str = f"{features_b} active" if features_b >= 0 else "? active"
    ax.set_title(
        f"{dataset}\n"
        f"{disp_a}: AUC={auc_a:.3f}, {feat_a_str}   |   "
        f"{disp_b}: AUC={auc_b:.3f}, {feat_b_str}",
        fontsize=9,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Prediction scatter plot")
    parser.add_argument("--dataset", type=str, default="kddcup09_appetency")
    parser.add_argument("--model-a", type=str, default="tabpfn")
    parser.add_argument("--model-b", type=str, default="tabicl")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "output" / "figures" / "prediction_scatter.pdf")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    task = "classification"
    _, _, _, y_q = _load_splits(args.dataset, task)

    logger.info("Getting %s predictions...", args.model_a)
    preds_a = get_predictions(args.model_a, args.dataset, task, args.device)

    logger.info("Getting %s predictions...", args.model_b)
    preds_b = get_predictions(args.model_b, args.dataset, task, args.device)

    # Compute AUC
    from sklearn.metrics import roc_auc_score
    auc_a = roc_auc_score(y_q, preds_a)
    auc_b = roc_auc_score(y_q, preds_b)
    logger.info("%s AUC=%.3f, %s AUC=%.3f", args.model_a, auc_a, args.model_b, auc_b)

    # Feature counts (active on this dataset)
    features_a = get_active_feature_count(args.model_a, args.dataset)
    features_b = get_active_feature_count(args.model_b, args.dataset)
    logger.info("%s features=%d, %s features=%d",
                args.model_a, features_a, args.model_b, features_b)

    plot_prediction_scatter(
        preds_a, preds_b, y_q,
        args.model_a, args.model_b, args.dataset,
        auc_a, auc_b, features_a, features_b,
        args.output,
    )


if __name__ == "__main__":
    main()
