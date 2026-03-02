#!/usr/bin/env python3
"""Scatter plot of predicted probabilities from two models on the same dataset.

Produces a scatter of P(class=1) from model A vs model B on the same query rows,
with y=x reference line and event-rate decision boundaries.

Optionally overlays ablated predictions as hollow markers, showing how removing
model-A-only SAE concepts shifts predictions toward model B.

Usage:
    # Basic scatter
    python scripts/plot_prediction_scatter.py --dataset kddcup09_appetency \
        --model-a tabpfn --model-b tabicl --device cuda

    # Ablate unmatched features (3 levels: top5, top10, all)
    python scripts/plot_prediction_scatter.py --dataset kddcup09_appetency \
        --model-a tabpfn --model-b tabicl --ablate-unmatched --device cuda

    # Ablate specific features
    python scripts/plot_prediction_scatter.py --dataset kddcup09_appetency \
        --model-a tabpfn --model-b tabicl --ablate-features 68,79,144 --device cuda
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


def get_differential_features(
    model_a: str, model_b: str, dataset: str,
    ablate_model: str, top_n: int,
) -> list:
    """Get top N features where ablate_model's importance far exceeds the other's.

    Reads the pairwise comparison JSON and returns features from the ablated
    model that have the largest positive differential (matched concepts where
    the ablated model relies on them much more than the other model).
    """
    # Try both orderings
    for a, b in [(model_a, model_b), (model_b, model_a)]:
        cmp_path = (PROJECT_ROOT / "output" / "concept_importance"
                    / f"compare_{a}_vs_{b}_{dataset}.json")
        if cmp_path.exists():
            with open(cmp_path) as f:
                cmp = json.load(f)
            break
    else:
        raise FileNotFoundError(
            f"No comparison data for {model_a} vs {model_b} on {dataset}. "
            f"Run: python scripts/concept_importance.py --model {model_a} "
            f"--compare {model_b} --dataset {dataset}"
        )

    # Matched features (MNN + correlated) sorted by differential
    all_matched = cmp["matched_features"] + cmp.get("correlated_features", [])

    if ablate_model == cmp["model_a"]:
        # Positive differential = model_a relies more
        ranked = sorted(all_matched, key=lambda m: -m["differential"])
        return [m["feat_a"] for m in ranked[:top_n]]
    else:
        # Negative differential = model_b relies more
        ranked = sorted(all_matched, key=lambda m: m["differential"])
        return [m["feat_b"] for m in ranked[:top_n]]


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


def get_unmatched_features(model_a: str, model_b: str, dataset: str) -> list:
    """Get all unmatched features with positive drop for model_a (not in model_b).

    Returns list of (feature_index, drop) sorted by drop descending.
    """
    for a, b in [(model_a, model_b), (model_b, model_a)]:
        cmp_path = (PROJECT_ROOT / "output" / "concept_importance"
                    / f"compare_{a}_vs_{b}_{dataset}.json")
        if cmp_path.exists():
            with open(cmp_path) as f:
                cmp = json.load(f)
            break
    else:
        raise FileNotFoundError(
            f"No comparison data for {model_a} vs {model_b} on {dataset}."
        )

    if model_a == cmp["model_a"]:
        unmatched = cmp["unmatched_a"]
        feat_key, drop_key = "feat_a", "drop_a"
    else:
        unmatched = cmp["unmatched_b"]
        feat_key, drop_key = "feat_b", "drop_b"

    ranked = sorted(unmatched, key=lambda u: -u[drop_key])
    return [(u[feat_key], u[drop_key]) for u in ranked if u[drop_key] > 0]


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
    ablation_levels: list = None,
):
    """Create scatter plot of predictions from two models.

    ablation_levels: list of (label, color, ablated_preds_array).
    Each level is overlaid as colored markers (filled for class 1, hollow for class 0).
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    event_rate = y_true.mean()

    # Auto-zoom to data range with padding
    all_preds = np.concatenate([preds_a, preds_b])
    if ablation_levels:
        for _, _, abl_p in ablation_levels:
            all_preds = np.concatenate([all_preds, abl_p])
    lo = max(0, all_preds.min() - 0.02)
    hi = min(1, all_preds.max() + 0.02)
    hi = max(hi, event_rate + 0.02)

    # Grid for visual alignment
    ax.grid(True, which="major", color="#cccccc", lw=0.5, alpha=0.7, zorder=0)
    ax.grid(True, which="minor", color="#eeeeee", lw=0.3, alpha=0.5, zorder=0)
    ax.minorticks_on()

    mask0 = y_true == 0
    mask1 = y_true == 1
    orig_color = "#999999"

    # Original predictions: faint, small — just a hint of spread
    ax.scatter(preds_a[mask0], preds_b[mask0], facecolors="none",
               edgecolors=orig_color, s=10, alpha=0.4, linewidths=0.5,
               label=f"Class 0 (n={mask0.sum()})", zorder=2)
    ax.scatter(preds_a[mask1], preds_b[mask1], c=orig_color,
               s=10, alpha=0.4, edgecolors="none",
               label=f"Class 1 (n={mask1.sum()})", zorder=2)

    # Ablation overlays
    if ablation_levels:
        for label, color, abl_p in ablation_levels:
            abl_x = abl_p  # ablating model A → x-axis shifts
            abl_y = preds_b
            mean_shift = float(np.abs(abl_p - preds_a).mean())

            ax.scatter(abl_x[mask0], abl_y[mask0], facecolors="none",
                       edgecolors=color, s=12, alpha=0.6, linewidths=0.7, zorder=4)
            ax.scatter(abl_x[mask1], abl_y[mask1], c=color,
                       s=12, alpha=0.6, edgecolors="none", zorder=4)
            ax.scatter([], [], c=color, s=20,
                       label=f"{label} (shift={mean_shift:.3f})")

    # y=x reference line
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y = x")

    # Event rate lines
    ax.axhline(event_rate, color="gray", lw=0.7, ls=":", alpha=0.7)
    ax.axvline(event_rate, color="gray", lw=0.7, ls=":", alpha=0.7)
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
    ax.legend(fontsize=7, loc="upper left")

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
    logger.info("Saved scatter to %s", output_path)


def plot_shift_distribution(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    ablation_levels: list,
    model_a: str,
    model_b: str,
    dataset: str,
    output_path: Path,
):
    """Plot distribution of per-row prediction shifts for each ablation level.

    ablation_levels: list of (label, color, ablated_preds_array).
    Includes vertical line at mean(preds_a - preds_b) as reference.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    disp_a = DISPLAY_NAMES.get(model_a, model_a)
    disp_b = DISPLAY_NAMES.get(model_b, model_b)

    for label, color, abl_p in ablation_levels:
        shifts = abl_p - preds_a  # per-row shift (positive = increased P)
        ax.hist(shifts, bins=50, alpha=0.4, color=color, edgecolor=color,
                linewidth=0.8, label=label)

    # Reference: mean original prediction difference
    mean_gap = float((preds_a - preds_b).mean())
    ax.axvline(-mean_gap, color="black", lw=1.5, ls="--",
               label=f"mean({disp_b} - {disp_a}) = {-mean_gap:+.3f}")

    ax.set_xlabel(f"Prediction shift (ablated {disp_a} - original {disp_a})", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"{dataset}: effect of ablating {disp_a}-only concepts", fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved distribution to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Prediction scatter plot")
    parser.add_argument("--dataset", type=str, default="kddcup09_appetency")
    parser.add_argument("--model-a", type=str, default="tabpfn")
    parser.add_argument("--model-b", type=str, default="tabicl")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "output" / "figures" / "prediction_scatter.pdf")

    # Ablation modes (mutually exclusive)
    abl_group = parser.add_mutually_exclusive_group()
    abl_group.add_argument("--ablate-unmatched", action="store_true",
                           help="Ablate unmatched features from model-a at 3 levels "
                                "(top5, top10, all positive-drop)")
    abl_group.add_argument("--ablate-features", type=str, default=None,
                           help="Comma-separated feature indices to ablate from model-a")

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

    ablation_levels = None

    if args.ablate_unmatched:
        # Get unmatched features (model_a concepts that model_b lacks)
        unmatched = get_unmatched_features(args.model_a, args.model_b, args.dataset)
        if not unmatched:
            logger.warning("No unmatched features with positive drop found.")
        else:
            logger.info("Found %d unmatched features with positive drop", len(unmatched))
            for feat, drop in unmatched[:10]:
                logger.info("  feature %d: drop=%.4f", feat, drop)

            feats = [f for f, _ in unmatched]
            disp_a = DISPLAY_NAMES.get(args.model_a, args.model_a)
            label = f"ablate {len(feats)} {disp_a}-only"
            logger.info("Running ablation: %s (features: %s)...", label, feats)
            abl_preds = get_ablated_predictions(
                args.model_a, args.dataset, task, feats, args.device,
            )
            ablation_levels = [(label, "#2ca02c", abl_preds)]

    elif args.ablate_features:
        feats = [int(x.strip()) for x in args.ablate_features.split(",")]
        logger.info("Ablating features %s from %s...", feats, args.model_a)
        abl_preds = get_ablated_predictions(
            args.model_a, args.dataset, task, feats, args.device,
        )
        disp = DISPLAY_NAMES.get(args.model_a, args.model_a)
        ablation_levels = [(f"ablate {len(feats)}f from {disp}", "#e377c2", abl_preds)]

    # Scatter plot
    plot_prediction_scatter(
        preds_a, preds_b, y_q,
        args.model_a, args.model_b, args.dataset,
        auc_a, auc_b, features_a, features_b,
        args.output,
        ablation_levels=ablation_levels,
    )

    # Shift distribution (only for multi-level ablation)
    if ablation_levels and len(ablation_levels) > 1:
        dist_path = args.output.with_name(
            args.output.stem + "_shift_distribution" + args.output.suffix
        )
        plot_shift_distribution(
            preds_a, preds_b, ablation_levels,
            args.model_a, args.model_b, args.dataset,
            dist_path,
        )


if __name__ == "__main__":
    main()
