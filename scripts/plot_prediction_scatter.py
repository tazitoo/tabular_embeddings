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


def get_unmatched_features(
    model_a: str, model_b: str, dataset: str,
    positive_only: bool = True,
) -> list:
    """Get unmatched features for model_a (not in model_b).

    Args:
        positive_only: If True, only return features with positive ablation drop.
            Use True for ablation (no point ablating zero-drop features).
            Use False for transfer (weak model may benefit from features the
            strong model doesn't rely on).

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
    if positive_only:
        ranked = [u for u in ranked if u[drop_key] > 0]
    return [(u[feat_key], u[drop_key]) for u in ranked]


def _logloss(y_true: np.ndarray, p1: np.ndarray) -> float:
    """Mean per-row cross-entropy loss for binary classification."""
    eps = 1e-7
    p = np.clip(p1, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def find_optimal_ablation(
    ablate_model: str, dataset: str, task: str, device: str,
    ranked_features: list, target_logloss: float,
) -> dict:
    """Sweep k=1..N unmatched features, find k where ablated logloss best matches target.

    The target is the weaker model's logloss. We ablate the stronger model's unique
    concepts to degrade it, looking for the k that brings its logloss closest to
    the weaker model's. This tells us how many unique concepts explain the gap.

    Args:
        ablate_model: Model to ablate (the stronger one)
        ranked_features: list of (feature_index, drop) sorted by importance
        target_logloss: Logloss of the weaker model (what we're trying to match)

    Returns:
        dict with: optimal_k, optimal_features, logloss_curve, etc.
    """
    from scripts.intervene_sae import sweep_intervene
    from scripts.concept_performance_diagnostic import _load_splits

    X_ctx, y_ctx, X_q, y_q = _load_splits(dataset, task)

    # Build cumulative feature lists: top-1, top-2, ..., top-N
    feat_indices = [f for f, _ in ranked_features]
    feature_lists = [feat_indices[:k] for k in range(1, len(feat_indices) + 1)]

    logger.info("Sweeping k=1..%d ablation levels for %s...", len(feature_lists), ablate_model)
    baseline_preds_raw, ablated_list = sweep_intervene(
        model_key=ablate_model,
        X_context=X_ctx, y_context=y_ctx,
        X_query=X_q, y_query=y_q,
        feature_lists=feature_lists,
        device=device, task=task,
    )

    # Extract P(class=1) from each level
    ablated_p1 = []
    for preds in ablated_list:
        if preds.ndim == 2:
            ablated_p1.append(preds[:, 1])
        else:
            ablated_p1.append(preds)

    baseline_p1 = baseline_preds_raw[:, 1] if baseline_preds_raw.ndim == 2 else baseline_preds_raw
    y = y_q.astype(float)

    # Logloss at each k
    logloss_curve = [_logloss(y, ap) for ap in ablated_p1]
    baseline_logloss = _logloss(y, baseline_p1)

    # Find k where ablated logloss is closest to target (weaker model's logloss)
    gaps = [abs(ll - target_logloss) for ll in logloss_curve]
    optimal_k = int(np.argmin(gaps)) + 1  # 1-indexed

    logger.info("Baseline logloss=%.4f, target (weaker model)=%.4f",
                baseline_logloss, target_logloss)
    logger.info("Optimal k=%d (logloss=%.4f, gap to target=%.4f)",
                optimal_k, logloss_curve[optimal_k - 1],
                logloss_curve[optimal_k - 1] - target_logloss)

    return {
        "optimal_k": optimal_k,
        "optimal_features": feat_indices[:optimal_k],
        "optimal_preds": ablated_p1[optimal_k - 1],
        "logloss_curve": logloss_curve,
        "baseline_logloss": baseline_logloss,
        "target_logloss": target_logloss,
        "all_ablated_preds": ablated_p1,
    }


def plot_logloss_curve(
    logloss_curve: list,
    baseline_logloss: float,
    target_logloss: float,
    optimal_k: int,
    ranked_features: list,
    ablate_model: str,
    other_model: str,
    dataset: str,
    output_path: Path,
    action: str = "ablating",
):
    """Plot logloss vs number of ablated/transferred features, with target line."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    ks = np.arange(1, len(logloss_curve) + 1)
    ax.plot(ks, logloss_curve, color="#0072B2", lw=1.5)

    disp_abl = DISPLAY_NAMES.get(ablate_model, ablate_model)
    disp_oth = DISPLAY_NAMES.get(other_model, other_model)

    # Past participle for axis label: "ablating" → "ablated", "transferring" → "transferred"
    action_pp = action.rstrip("ing") + "ed" if action.endswith("ing") else action

    ax.axhline(baseline_logloss, color="gray", ls="--", lw=1,
               label=f"{disp_abl} logloss={baseline_logloss:.3f}")
    ax.axhline(target_logloss, color="#009E73", ls="--", lw=1,
               label=f"{disp_oth} logloss={target_logloss:.3f}")
    ax.axvline(optimal_k, color="#D55E00", ls=":", lw=1,
               label=f"optimal k={optimal_k}")
    ax.plot(optimal_k, logloss_curve[optimal_k - 1], "o", color="#D55E00", ms=8, zorder=5)

    # Ablation removes ablate_model's concepts; transfer injects other_model's concepts
    concept_owner = disp_oth if action == "transferring" else disp_abl
    ax.set_xlabel(f"Number of {concept_owner}-only concepts {action_pp}", fontsize=10)
    ax.set_ylabel("Logloss", fontsize=10)
    ax.set_title(f"{dataset}: {action} {concept_owner}-only concepts", fontsize=10)
    ax.legend(fontsize=8)

    # Annotate top features near the curve
    for i, (feat, drop) in enumerate(ranked_features[:min(5, optimal_k)]):
        ax.annotate(f"f{feat}", (i + 1, logloss_curve[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7, color="#555555")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved logloss curve to %s", output_path)


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
    ablate_axis: str = "x",
):
    """Create scatter plot of predictions from two models.

    ablation_levels: list of (label, color, ablated_preds_array).
    ablate_axis: "x" if ablating model_a (x-axis shifts), "y" if ablating model_b.
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
            if ablate_axis == "x":
                abl_x, abl_y = abl_p, preds_b
                mean_shift = float(np.abs(abl_p - preds_a).mean())
            else:
                abl_x, abl_y = preds_a, abl_p
                mean_shift = float(np.abs(abl_p - preds_b).mean())

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

    # Inset zoom if data is very skewed (most points in a small region)
    from matplotlib.patches import Rectangle
    p95_a = np.percentile(preds_a, 95)
    p95_b = np.percentile(preds_b, 95)
    zoom_hi = max(p95_a, p95_b) * 1.2
    # Add inset if the dense region is < 40% of the full range
    if zoom_hi < 0.4 * (hi - lo) + lo:
        zoom_hi = min(zoom_hi, hi * 0.5)
        zoom_lo = lo
        axins = ax.inset_axes([0.45, 0.45, 0.52, 0.52])  # upper-right
        axins.grid(True, which="major", color="#cccccc", lw=0.3, alpha=0.5)
        axins.grid(True, which="minor", color="#eeeeee", lw=0.2, alpha=0.3)
        axins.minorticks_on()

        # Replot data in inset
        axins.scatter(preds_a[mask0], preds_b[mask0], facecolors="none",
                      edgecolors=orig_color, s=12, alpha=0.4, linewidths=0.5, zorder=2)
        axins.scatter(preds_a[mask1], preds_b[mask1], c=orig_color,
                      s=15, alpha=0.5, edgecolors="none", zorder=2)
        if ablation_levels:
            for _, color, abl_p in ablation_levels:
                if ablate_axis == "x":
                    abl_x, abl_y = abl_p, preds_b
                else:
                    abl_x, abl_y = preds_a, abl_p
                axins.scatter(abl_x[mask0], abl_y[mask0], facecolors="none",
                              edgecolors=color, s=14, alpha=0.6, linewidths=0.7, zorder=4)
                axins.scatter(abl_x[mask1], abl_y[mask1], c=color,
                              s=18, alpha=0.7, edgecolors="none", zorder=4)
        axins.plot([zoom_lo, zoom_hi], [zoom_lo, zoom_hi], "k--", lw=0.6, alpha=0.4)
        axins.set_xlim(zoom_lo, zoom_hi)
        axins.set_ylim(zoom_lo, zoom_hi)
        axins.set_aspect("equal")
        axins.tick_params(labelsize=7)

        # Draw rectangle on main axes showing inset region
        rect = Rectangle((zoom_lo, zoom_lo), zoom_hi - zoom_lo, zoom_hi - zoom_lo,
                          lw=0.8, edgecolor="#555555", facecolor="none", ls="--", zorder=6)
        ax.add_patch(rect)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scatter to %s", output_path)


def find_per_row_optimal_ablation(
    ablate_model: str, other_model: str, dataset: str, task: str, device: str,
    unmatched_features: list, preds_other: np.ndarray, y_query: np.ndarray,
) -> dict:
    """Per-row sweep: find optimal k for each row individually.

    For each row, ranks unmatched features by that row's per-row importance
    (only features that fire on the row), then sweeps k to find the minimum
    number of concepts that closes the row's logloss gap to the other model.

    Args:
        unmatched_features: list of (feature_index, drop) sorted by aggregate drop
        preds_other: P(class=1) from the weaker model for per-row target logloss
    """
    from scripts.intervene_sae import perrow_sweep_intervene
    from scripts.concept_performance_diagnostic import _load_splits

    X_ctx, y_ctx, X_q, y_q = _load_splits(dataset, task)
    feat_indices = [f for f, _ in unmatched_features]

    result = perrow_sweep_intervene(
        model_key=ablate_model,
        X_context=X_ctx, y_context=y_ctx,
        X_query=X_q, y_query=y_q,
        unmatched_features=feat_indices,
        device=device, task=task,
    )

    # Per-row target and baseline logloss
    eps = 1e-7
    y = y_q.astype(float)
    p_other = np.clip(preds_other, eps, 1 - eps)
    target_row_ll = -(y * np.log(p_other) + (1 - y) * np.log(1 - p_other))

    bp1 = result["baseline_preds"][:, 1] if result["baseline_preds"].ndim == 2 \
        else result["baseline_preds"]
    bp = np.clip(bp1, eps, 1 - eps)
    baseline_row_ll = -(y * np.log(bp) + (1 - y) * np.log(1 - bp))

    n_query = len(y_q)
    optimal_k = np.zeros(n_query, dtype=int)
    row_gap_closed = np.zeros(n_query)  # fraction of gap closed at optimal k

    for row_idx in range(n_query):
        orig_gap = target_row_ll[row_idx] - baseline_row_ll[row_idx]
        if orig_gap <= 0:
            # Stronger model is already worse on this row — no ablation needed
            optimal_k[row_idx] = 0
            row_gap_closed[row_idx] = 1.0
            continue

        max_k_row = result["max_k_per_row"][row_idx]
        best_k = 0
        best_gap_remaining = orig_gap

        for k in range(1, max_k_row + 1):
            if k - 1 >= len(result["sweep_preds"]):
                break
            preds_k = result["sweep_preds"][k - 1]
            p1 = preds_k[row_idx, 1] if preds_k.ndim == 2 else preds_k[row_idx]
            p = np.clip(float(p1), eps, 1 - eps)
            row_ll = -(y[row_idx] * np.log(p) + (1 - y[row_idx]) * np.log(1 - p))
            gap_remaining = abs(row_ll - target_row_ll[row_idx])
            if gap_remaining < best_gap_remaining:
                best_gap_remaining = gap_remaining
                best_k = k

        optimal_k[row_idx] = best_k
        row_gap_closed[row_idx] = 1.0 - best_gap_remaining / orig_gap if orig_gap > 0 else 1.0

    logger.info("Per-row optimal k: mean=%.1f, median=%d, max=%d",
                optimal_k.mean(), np.median(optimal_k), optimal_k.max())
    logger.info("Rows needing 0 concepts: %d (%.1f%%)",
                (optimal_k == 0).sum(), 100 * (optimal_k == 0).mean())

    return {
        "optimal_k": optimal_k,
        "row_gap_closed": row_gap_closed,
        "perrow_rankings": result["perrow_rankings"],
        "perrow_importance": result["perrow_importance"],
        "sweep_preds": result["sweep_preds"],
        "baseline_preds": result["baseline_preds"],
        "max_k_per_row": result["max_k_per_row"],
        "target_row_ll": target_row_ll,
        "baseline_row_ll": baseline_row_ll,
        "unmatched_features": result["unmatched_features"],
    }


def plot_perrow_results(
    optimal_k: np.ndarray,
    row_gap_closed: np.ndarray,
    max_k_per_row: np.ndarray,
    perrow_rankings: list,
    ablate_model: str,
    other_model: str,
    dataset: str,
    output_path: Path,
    action: str = "ablating",
):
    """Plot per-row results: histogram + cumulative coverage curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    verb = "transferred" if action == "transferring" else "ablated"
    verb_ing = "Transferring" if action == "transferring" else "Ablating"
    concept_label = f"Concepts {verb}"

    # For transfer: ablate_model = concept owner (source), other_model = model being modified (target)
    # For ablation: ablate_model = model being modified, other_model = reference
    disp_abl = DISPLAY_NAMES.get(ablate_model, ablate_model)
    disp_oth = DISPLAY_NAMES.get(other_model, other_model)
    n_query = len(optimal_k)

    # --- Left: histogram of per-row optimal k ---
    ax = axes[0]
    max_k = int(optimal_k.max())
    bins = np.arange(-0.5, max_k + 1.5, 1)
    ax.hist(optimal_k, bins=bins, color="#0072B2", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(optimal_k), color="#D55E00", ls="--", lw=1.5,
               label=f"median = {np.median(optimal_k):.0f}")
    ax.axvline(optimal_k.mean(), color="#E69F00", ls=":", lw=1.5,
               label=f"mean = {optimal_k.mean():.1f}")
    ax.set_xlabel(f"{concept_label} (per-row optimal k)", fontsize=10)
    ax.set_ylabel("Number of rows", fontsize=10)
    ax.set_title(f"{dataset}: per-row concept count", fontsize=10)
    ax.legend(fontsize=8)

    # Annotate k=0 bar
    n_zero = int((optimal_k == 0).sum())
    if n_zero > 0:
        ax.annotate(f"k=0: {n_zero} rows\n(no gap or reversed)",
                    xy=(0, n_zero), xytext=(max(2, max_k * 0.3), n_zero * 0.8),
                    fontsize=7, color="#555555",
                    arrowprops=dict(arrowstyle="->", color="#999999"))

    # --- Right: cumulative coverage curve ---
    ax = axes[1]
    ks = np.arange(0, max_k + 1)
    coverage = np.array([(optimal_k <= k).mean() for k in ks])
    ax.plot(ks, coverage * 100, color="#0072B2", lw=2)
    ax.fill_between(ks, 0, coverage * 100, alpha=0.1, color="#0072B2")
    ax.axhline(50, color="#999999", ls=":", lw=0.8, alpha=0.6)
    ax.axhline(90, color="#999999", ls=":", lw=0.8, alpha=0.6)

    # Find k for 50% and 90% coverage
    k50 = int(ks[np.searchsorted(coverage, 0.5)])
    k90 = int(ks[np.searchsorted(coverage, 0.9)]) if coverage[-1] >= 0.9 else max_k
    ax.plot(k50, 50, "o", color="#D55E00", ms=6, zorder=5)
    ax.plot(k90, 90, "o", color="#D55E00", ms=6, zorder=5)
    ax.annotate(f"k={k50}", (k50, 50), textcoords="offset points",
                xytext=(8, -5), fontsize=8, color="#D55E00")
    ax.annotate(f"k={k90}", (k90, 90), textcoords="offset points",
                xytext=(8, -5), fontsize=8, color="#D55E00")

    ax.set_xlabel(f"Max concepts {verb} (k)", fontsize=10)
    ax.set_ylabel(f"% rows where gap to {disp_oth} is closed", fontsize=10)
    ax.set_title(f"{dataset}: cumulative coverage", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.5, max_k + 0.5)

    # Feature frequency: which features appear most often in per-row explanations?
    from collections import Counter
    feat_counts = Counter()
    for row_idx, k in enumerate(optimal_k):
        if k > 0:
            feat_counts.update(perrow_rankings[row_idx][:k])
    if feat_counts:
        top5 = feat_counts.most_common(5)
        text = "Top features (by row frequency):\n"
        text += "\n".join(f"  f{f}: {c}/{n_query} rows ({100*c/n_query:.0f}%)"
                          for f, c in top5)
        ax.text(0.98, 0.05, text, transform=ax.transAxes, fontsize=7,
                va="bottom", ha="right", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle(f"{verb_ing} {disp_abl}-only concepts → matching {disp_oth}",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-row results to %s", output_path)


def extract_perrow_ablated_preds(
    optimal_k: np.ndarray,
    sweep_preds: list,
    baseline_preds: np.ndarray,
) -> np.ndarray:
    """Extract each row's ablated prediction at its own optimal k.

    Returns P(class=1) array with the same shape as baseline_preds.
    Rows with optimal_k=0 keep their baseline prediction.
    """
    bp = baseline_preds[:, 1] if baseline_preds.ndim == 2 else baseline_preds
    ablated = np.copy(bp)
    for i, k in enumerate(optimal_k):
        if k > 0 and k - 1 < len(sweep_preds):
            pk = sweep_preds[k - 1]
            ablated[i] = pk[i, 1] if pk.ndim == 2 else pk[i]
    return ablated


def plot_perrow_scatter(
    preds_strong: np.ndarray,
    preds_weak: np.ndarray,
    ablated_strong: np.ndarray,
    optimal_k: np.ndarray,
    y_true: np.ndarray,
    model_strong: str,
    model_weak: str,
    dataset: str,
    auc_strong: float,
    auc_weak: float,
    output_path: Path,
    ablate_axis: str = "x",
    action: str = "ablating",
):
    """Scatter showing only rows where the stronger model outperforms the weaker,
    with original positions (open circles) and per-row modified positions (filled).

    Filtered to rows where optimal_k > 0 (the stronger model genuinely has a
    logloss advantage on that row, and ablation/transfer can close the gap).
    """
    verb = "Transferred" if action == "transferring" else "Ablated"
    verb_lower = "transfer" if action == "transferring" else "ablation"
    fixable = optimal_k > 0
    n_fixable = fixable.sum()
    n_total = len(optimal_k)
    logger.info("Fixable rows: %d / %d (%.1f%%)",
                n_fixable, n_total, 100 * n_fixable / n_total)

    if n_fixable == 0:
        logger.warning("No fixable rows — skipping per-row scatter.")
        return

    # Filter to fixable rows
    ps = preds_strong[fixable]
    pw = preds_weak[fixable]
    ab = ablated_strong[fixable]
    yt = y_true[fixable]
    ok = optimal_k[fixable]

    disp_s = DISPLAY_NAMES.get(model_strong, model_strong)
    disp_w = DISPLAY_NAMES.get(model_weak, model_weak)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    event_rate = y_true.mean()  # full dataset event rate

    # Axis assignments
    if ablate_axis == "x":
        orig_x, orig_y = ps, pw
        abl_x, abl_y = ab, pw
        xlabel = f"{disp_s}  P(class=1)"
        ylabel = f"{disp_w}  P(class=1)"
    else:
        orig_x, orig_y = pw, ps
        abl_x, abl_y = pw, ab
        xlabel = f"{disp_w}  P(class=1)"
        ylabel = f"{disp_s}  P(class=1)"

    # Auto-zoom
    all_vals = np.concatenate([orig_x, orig_y, abl_x, abl_y])
    lo = max(0, all_vals.min() - 0.02)
    hi = min(1, all_vals.max() + 0.02)
    hi = max(hi, event_rate + 0.02)

    # Grid
    ax.grid(True, which="major", color="#cccccc", lw=0.5, alpha=0.7, zorder=0)
    ax.grid(True, which="minor", color="#eeeeee", lw=0.3, alpha=0.5, zorder=0)
    ax.minorticks_on()

    mask0 = yt == 0
    mask1 = yt == 1

    # Original positions: open circles, faint gray
    orig_color = "#999999"
    ax.scatter(orig_x[mask0], orig_y[mask0], facecolors="none",
               edgecolors=orig_color, s=14, alpha=0.5, linewidths=0.6,
               label=f"Original class 0 (n={mask0.sum()})", zorder=2)
    ax.scatter(orig_x[mask1], orig_y[mask1], facecolors="none",
               edgecolors=orig_color, s=14, alpha=0.5, linewidths=0.6,
               marker="s",
               label=f"Original class 1 (n={mask1.sum()})", zorder=2)

    # Ablated positions: filled, colored by class
    color0 = "#0072B2"  # blue
    color1 = "#D55E00"  # orange
    ax.scatter(abl_x[mask0], abl_y[mask0], c=color0,
               s=14, alpha=0.6, edgecolors="none",
               label=f"{verb} class 0", zorder=4)
    ax.scatter(abl_x[mask1], abl_y[mask1], c=color1,
               s=14, alpha=0.6, edgecolors="none",
               marker="s",
               label=f"{verb} class 1", zorder=4)

    # y=x reference
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y = x")

    # Event rate lines
    ax.axhline(event_rate, color="gray", lw=0.7, ls=":", alpha=0.7)
    ax.axvline(event_rate, color="gray", lw=0.7, ls=":", alpha=0.7)
    ax.text(0.97, event_rate, f" event rate = {event_rate:.3f}",
            fontsize=7, color="gray", va="bottom", ha="right",
            transform=ax.get_yaxis_transform())

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left")

    ax.set_title(
        f"{dataset} — per-row {verb_lower} ({n_fixable}/{n_total} fixable rows)\n"
        f"{disp_s}: AUC={auc_strong:.3f}   |   {disp_w}: AUC={auc_weak:.3f}\n"
        f"median k={np.median(ok):.0f}, mean k={ok.mean():.1f}",
        fontsize=9,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-row scatter to %s", output_path)


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
                           help="Ablate unmatched features from the stronger model "
                                "(auto-detected by AUC)")
    abl_group.add_argument("--perrow", action="store_true",
                           help="Per-row heterogeneous ablation: find minimal concept "
                                "set per row to close the logloss gap")
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
    ablate_axis = "x"  # default: ablating model_a shifts x-axis

    if args.ablate_unmatched:
        # Ablate the stronger model's unique concepts
        if auc_a >= auc_b:
            ablate_model, other_model = args.model_a, args.model_b
            ablate_axis = "x"
        else:
            ablate_model, other_model = args.model_b, args.model_a
            ablate_axis = "y"

        disp = DISPLAY_NAMES.get(ablate_model, ablate_model)
        disp_other = DISPLAY_NAMES.get(other_model, other_model)
        logger.info("Ablating stronger model: %s (AUC=%.3f) vs %s (AUC=%.3f)",
                    disp, max(auc_a, auc_b), disp_other, min(auc_a, auc_b))

        unmatched = get_unmatched_features(ablate_model, other_model, args.dataset)
        if not unmatched:
            logger.warning("No unmatched features with positive drop found.")
        else:
            logger.info("Found %d unmatched features with positive drop", len(unmatched))
            for feat, drop in unmatched[:10]:
                logger.info("  feature %d: drop=%.4f", feat, drop)

            # Compute target logloss (the weaker model's logloss)
            other_preds = preds_b if ablate_axis == "x" else preds_a
            target_logloss = _logloss(y_q.astype(float), other_preds)

            # Sweep to find optimal k
            sweep = find_optimal_ablation(
                ablate_model, args.dataset, task, args.device,
                unmatched, target_logloss,
            )
            k = sweep["optimal_k"]
            label = f"ablate {k}/{len(unmatched)} {disp}-only"
            ablation_levels = [(label, "#0072B2", sweep["optimal_preds"])]

            # Save logloss curve plot
            fig_dir = args.output.parent
            curve_path = fig_dir / f"loss_search{args.output.suffix}"
            plot_logloss_curve(
                sweep["logloss_curve"], sweep["baseline_logloss"],
                sweep["target_logloss"], k,
                unmatched, ablate_model, other_model, args.dataset,
                curve_path,
            )

    elif args.perrow:
        # Per-row heterogeneous ablation
        if auc_a >= auc_b:
            ablate_model, other_model = args.model_a, args.model_b
            ablate_axis = "x"
        else:
            ablate_model, other_model = args.model_b, args.model_a
            ablate_axis = "y"

        disp = DISPLAY_NAMES.get(ablate_model, ablate_model)
        disp_other = DISPLAY_NAMES.get(other_model, other_model)
        logger.info("Per-row ablation of %s (AUC=%.3f) → %s (AUC=%.3f)",
                    disp, max(auc_a, auc_b), disp_other, min(auc_a, auc_b))

        unmatched = get_unmatched_features(ablate_model, other_model, args.dataset)
        if not unmatched:
            logger.warning("No unmatched features with positive drop found.")
        else:
            logger.info("Found %d unmatched features with positive drop", len(unmatched))
            other_preds = preds_b if ablate_axis == "x" else preds_a

            perrow = find_per_row_optimal_ablation(
                ablate_model, other_model, args.dataset, task, args.device,
                unmatched, other_preds, y_q,
            )

            # Save per-row results plot
            fig_dir = args.output.parent
            perrow_path = fig_dir / f"perrow_ablation{args.output.suffix}"
            plot_perrow_results(
                perrow["optimal_k"], perrow["row_gap_closed"],
                perrow["max_k_per_row"], perrow["perrow_rankings"],
                ablate_model, other_model, args.dataset,
                perrow_path,
            )

            # Per-row scatter: filtered to fixable rows
            ablated_preds = extract_perrow_ablated_preds(
                perrow["optimal_k"], perrow["sweep_preds"],
                perrow["baseline_preds"],
            )
            strong_preds = preds_a if ablate_axis == "x" else preds_b
            weak_preds = preds_b if ablate_axis == "x" else preds_a
            auc_strong = max(auc_a, auc_b)
            auc_weak = min(auc_a, auc_b)
            perrow_scatter_path = fig_dir / f"perrow_scatter{args.output.suffix}"
            plot_perrow_scatter(
                strong_preds, weak_preds, ablated_preds,
                perrow["optimal_k"], y_q,
                ablate_model, other_model, args.dataset,
                auc_strong, auc_weak,
                perrow_scatter_path,
                ablate_axis=ablate_axis,
            )

            # Also run the dataset-level sweep for scatter + logloss curve
            target_logloss = _logloss(y_q.astype(float), other_preds)
            sweep = find_optimal_ablation(
                ablate_model, args.dataset, task, args.device,
                unmatched, target_logloss,
            )
            k = sweep["optimal_k"]
            label = f"ablate {k}/{len(unmatched)} {disp}-only"
            ablation_levels = [(label, "#0072B2", sweep["optimal_preds"])]

            curve_path = fig_dir / f"loss_search{args.output.suffix}"
            plot_logloss_curve(
                sweep["logloss_curve"], sweep["baseline_logloss"],
                sweep["target_logloss"], k,
                unmatched, ablate_model, other_model, args.dataset,
                curve_path,
            )

    elif args.ablate_features:
        feats = [int(x.strip()) for x in args.ablate_features.split(",")]
        logger.info("Ablating features %s from %s...", feats, args.model_a)
        abl_preds = get_ablated_predictions(
            args.model_a, args.dataset, task, feats, args.device,
        )
        disp = DISPLAY_NAMES.get(args.model_a, args.model_a)
        ablation_levels = [(f"ablate {len(feats)}f from {disp}", "#0072B2", abl_preds)]

    # Scatter plot
    plot_prediction_scatter(
        preds_a, preds_b, y_q,
        args.model_a, args.model_b, args.dataset,
        auc_a, auc_b, features_a, features_b,
        args.output,
        ablation_levels=ablation_levels,
        ablate_axis=ablate_axis,
    )


if __name__ == "__main__":
    main()
