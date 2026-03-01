#!/usr/bin/env python3
"""Scatter plot of predicted probabilities from two models on the same dataset.

Produces a scatter of P(class=1) from model A vs model B on the same query rows,
with y=x reference line and event-rate decision boundaries.

Optionally overlays ablated predictions: ablate top-N SAE features from one model
and draw arrows showing how each query row's prediction shifts.

Usage:
    # Basic scatter
    python scripts/plot_prediction_scatter.py --dataset kddcup09_appetency \
        --model-a tabpfn --model-b tabicl --device cuda

    # With ablation overlay (top 5 features from model A)
    python scripts/plot_prediction_scatter.py --dataset kddcup09_appetency \
        --model-a tabpfn --model-b tabicl --ablate-model tabpfn --ablate-top 5 --device cuda

    # With specific features ablated
    python scripts/plot_prediction_scatter.py --dataset kddcup09_appetency \
        --model-a tabpfn --model-b tabicl --ablate-model tabpfn \
        --ablate-features 42,108,305 --device cuda
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


def get_ablated_predictions(
    model_key: str, dataset: str, task: str,
    ablate_features: list, device: str,
) -> np.ndarray:
    """Get P(class=1) after ablating specific SAE features."""
    from scripts.intervene_sae import intervene

    X_ctx, y_ctx, X_q, y_q = _load_splits(dataset, task)

    results = intervene(
        model_key=model_key,
        X_context=X_ctx, y_context=y_ctx,
        X_query=X_q, y_query=y_q,
        ablate_features=ablate_features,
        device=device, task=task,
    )

    preds = results["ablated_preds"]
    if preds.ndim == 2:
        return preds[:, 1]
    return preds


def get_top_features(model_key: str, dataset: str, top_n: int) -> list:
    """Get top N feature indices by importance drop from saved sweep."""
    imp_path = PROJECT_ROOT / "output" / "concept_importance" / f"{model_key}_{dataset}.json"
    if not imp_path.exists():
        raise FileNotFoundError(
            f"No importance data at {imp_path}. "
            f"Run: python scripts/concept_importance.py --model {model_key} --dataset {dataset}"
        )
    with open(imp_path) as f:
        imp = json.load(f)
    features = sorted(imp["features"], key=lambda x: -x["drop"])
    return [f["index"] for f in features[:top_n]]


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
    ablated_preds: np.ndarray = None,
    ablated_model: str = None,
    ablation_label: str = None,
):
    """Create scatter plot of predictions from two models.

    If ablated_preds is provided, draws arrows from original to ablated position
    for each query row. ablated_model is "a" or "b" indicating which axis moves.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    event_rate = y_true.mean()

    # Auto-zoom to data range with padding (include ablated preds in range)
    all_preds = np.concatenate([preds_a, preds_b])
    if ablated_preds is not None:
        all_preds = np.concatenate([all_preds, ablated_preds])
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

    # Ablation overlay: arrows from original to ablated position
    if ablated_preds is not None and ablated_model is not None:
        min_shift = 0.005  # skip negligible arrows
        if ablated_model == "a":
            dx = ablated_preds - preds_a
            dy = np.zeros_like(dx)
            orig_x, orig_y = preds_a, preds_b
        else:
            dx = np.zeros_like(preds_b)
            dy = ablated_preds - preds_b
            orig_x, orig_y = preds_a, preds_b

        shift_mag = np.sqrt(dx**2 + dy**2)
        moved = shift_mag > min_shift

        if moved.any():
            ax.quiver(
                orig_x[moved], orig_y[moved],
                dx[moved], dy[moved],
                angles="xy", scale_units="xy", scale=1,
                color="#ff7f0e", alpha=0.6, width=0.003,
                headwidth=4, headlength=5, zorder=4,
            )

            n_moved = int(moved.sum())
            mean_shift = float(shift_mag[moved].mean())
            label = ablation_label or "ablated"
            ax.scatter([], [], c="#ff7f0e", marker=">", s=30,
                       label=f"{label} ({n_moved} rows, mean shift={mean_shift:.3f})")

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

    # Ablation overlay
    parser.add_argument("--ablate-model", type=str, default=None,
                        help="Which model to ablate (must be model-a or model-b)")
    parser.add_argument("--ablate-top", type=int, default=None,
                        help="Ablate top N features by importance drop")
    parser.add_argument("--ablate-features", type=str, default=None,
                        help="Comma-separated feature indices to ablate")

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

    # Ablation overlay
    ablated_preds = None
    ablated_model = None
    ablation_label = None

    if args.ablate_model:
        if args.ablate_model not in (args.model_a, args.model_b):
            parser.error(f"--ablate-model must be {args.model_a} or {args.model_b}")

        # Determine which features to ablate
        if args.ablate_top:
            ablate_features = get_top_features(
                args.ablate_model, args.dataset, args.ablate_top,
            )
            logger.info("Ablating top %d features from %s: %s",
                        args.ablate_top, args.ablate_model, ablate_features)
        elif args.ablate_features:
            ablate_features = [int(x.strip()) for x in args.ablate_features.split(",")]
            logger.info("Ablating features %s from %s", ablate_features, args.ablate_model)
        else:
            parser.error("--ablate-model requires --ablate-top or --ablate-features")

        logger.info("Running ablation on %s...", args.ablate_model)
        ablated_preds = get_ablated_predictions(
            args.ablate_model, args.dataset, task, ablate_features, args.device,
        )
        ablated_model = "a" if args.ablate_model == args.model_a else "b"
        disp = DISPLAY_NAMES.get(args.ablate_model, args.ablate_model)
        ablation_label = f"ablate {len(ablate_features)}f from {disp}"

    plot_prediction_scatter(
        preds_a, preds_b, y_q,
        args.model_a, args.model_b, args.dataset,
        auc_a, auc_b, features_a, features_b,
        args.output,
        ablated_preds=ablated_preds,
        ablated_model=ablated_model,
        ablation_label=ablation_label,
    )


if __name__ == "__main__":
    main()
