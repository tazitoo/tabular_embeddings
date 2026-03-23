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
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from scripts._project_root import PROJECT_ROOT

from scripts.intervention.concept_performance_diagnostic import (
    _load_splits, compute_metric, DISPLAY_NAMES,
)
from scripts.intervention.intervene_sae import (
    compute_perrow_logloss, get_improvable_rows,
)

logger = logging.getLogger(__name__)


def get_ablated_predictions(
    model_key: str, dataset: str, task: str,
    ablate_features: list, device: str,
) -> np.ndarray:
    """Get P(class=1) after ablating specific SAE features."""
    from scripts.intervention.intervene_sae import intervene

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
    from scripts.intervention.concept_performance_diagnostic import (
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
    from scripts.intervention.intervene_sae import sweep_intervene
    from scripts.intervention.concept_performance_diagnostic import _load_splits

    X_ctx, y_ctx, X_q, y_q = _load_splits(dataset, task)

    # Build cumulative feature lists: top-1, top-2, ..., top-N
    feat_indices = [f for f, _ in ranked_features]
    feature_lists = [feat_indices[:k] for k in range(1, len(feat_indices) + 1)]

    logger.info("Sweeping k=1..%d ablation levels for %s...", len(feature_lists), ablate_model)
    baseline_preds_raw, ablated_list = sweep_intervene(
        model_key=ablate_model,
        dataset_name=dataset,
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
    from scripts.intervention.intervene_sae import perrow_sweep_intervene
    from scripts.intervention.concept_performance_diagnostic import _load_splits

    X_ctx, y_ctx, X_q, y_q = _load_splits(dataset, task)
    feat_indices = [f for f, _ in unmatched_features]

    result = perrow_sweep_intervene(
        model_key=ablate_model,
        dataset_name=dataset,
        X_context=X_ctx, y_context=y_ctx,
        X_query=X_q, y_query=y_q,
        unmatched_features=feat_indices,
        device=device, task=task,
    )

    # Per-row target and baseline logloss
    target_row_ll = compute_perrow_logloss(preds_other, y_q)
    baseline_row_ll = compute_perrow_logloss(result["baseline_preds"], y_q)

    eps = 1e-7
    y = y_q.astype(float)
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
    preds_strong: np.ndarray = None,
    preds_weak: np.ndarray = None,
    y_query: np.ndarray = None,
):
    """Plot per-row results: concept count histograms + cumulative coverage.

    Left column: histograms of firing concept count (max_k_per_row).
      Top: fixable rows (optimal_k > 0).
      Bottom: unfixable rows (improvable but optimal_k == 0).
    Right: cumulative coverage curve for fixable rows.

    If preds_strong/preds_weak/y_query are provided, uses get_improvable_rows
    to separate unfixable from not-improvable. Otherwise falls back to
    treating all optimal_k==0 rows as unfixable.
    """
    verb = "transferred" if action == "transferring" else "ablated"
    verb_ing = "Transferring" if action == "transferring" else "Ablating"
    concept_label = f"Concepts {verb}"

    disp_abl = DISPLAY_NAMES.get(ablate_model, ablate_model)
    disp_oth = DISPLAY_NAMES.get(other_model, other_model)
    n_total = len(optimal_k)

    fixable = optimal_k > 0
    n_fixable = int(fixable.sum())
    ok_fixable = optimal_k[fixable]

    # Determine improvable vs unfixable
    if preds_strong is not None and preds_weak is not None and y_query is not None:
        improvable = get_improvable_rows(preds_strong, preds_weak, y_query)
    else:
        # Fallback: treat all rows as potentially improvable
        improvable = np.ones(n_total, dtype=bool)
    unfixable = improvable & ~fixable
    n_unfixable = int(unfixable.sum())
    n_improvable = int(improvable.sum())

    if n_fixable == 0:
        logger.warning("No fixable rows — skipping per-row results plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             gridspec_kw={"height_ratios": [1, 1]})

    # --- Top left: histogram of firing concepts for FIXABLE rows ---
    ax = axes[0, 0]
    avail_fixable = max_k_per_row[fixable]
    max_avail = int(max_k_per_row.max()) if len(max_k_per_row) > 0 else 1
    bins = np.arange(-0.5, max_avail + 1.5, 1)
    ax.hist(avail_fixable, bins=bins, color="#0072B2", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(avail_fixable), color="#D55E00", ls="--", lw=1.5,
               label=f"median = {np.median(avail_fixable):.0f}")
    ax.axvline(avail_fixable.mean(), color="#E69F00", ls=":", lw=1.5,
               label=f"mean = {avail_fixable.mean():.1f}")
    ax.set_xlabel("Firing unmatched concepts", fontsize=10)
    ax.set_ylabel("Number of rows", fontsize=10)
    ax.set_title(
        f"Fixable ({n_fixable}/{n_improvable} improvable)",
        fontsize=10,
    )
    ax.legend(fontsize=8)

    # --- Bottom left: histogram of firing concepts for UNFIXABLE rows ---
    ax = axes[1, 0]
    if n_unfixable > 0:
        avail_unfixable = max_k_per_row[unfixable]
        ax.hist(avail_unfixable, bins=bins, color="#999999", edgecolor="white", alpha=0.8)
        ax.axvline(np.median(avail_unfixable), color="#D55E00", ls="--", lw=1.5,
                   label=f"median = {np.median(avail_unfixable):.0f}")
        ax.axvline(avail_unfixable.mean(), color="#E69F00", ls=":", lw=1.5,
                   label=f"mean = {avail_unfixable.mean():.1f}")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No unfixable rows", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#999999")
    ax.set_xlabel("Firing unmatched concepts", fontsize=10)
    ax.set_ylabel("Number of rows", fontsize=10)
    ax.set_title(
        f"Unfixable ({n_unfixable}/{n_improvable} improvable)",
        fontsize=10,
    )

    # Match x-axis range between top and bottom
    xlim = axes[0, 0].get_xlim()
    axes[1, 0].set_xlim(xlim)

    # --- Top right: cumulative coverage curve (fixable rows only) ---
    ax = axes[0, 1]
    max_k = int(ok_fixable.max())
    ks = np.arange(1, max_k + 1)
    coverage = np.array([(ok_fixable <= k).mean() for k in ks])
    ax.plot(ks, coverage * 100, color="#0072B2", lw=2)
    ax.fill_between(ks, 0, coverage * 100, alpha=0.1, color="#0072B2")
    ax.axhline(50, color="#999999", ls=":", lw=0.8, alpha=0.6)
    ax.axhline(90, color="#999999", ls=":", lw=0.8, alpha=0.6)

    k50 = int(ks[np.searchsorted(coverage, 0.5)])
    k90 = int(ks[np.searchsorted(coverage, 0.9)]) if coverage[-1] >= 0.9 else max_k
    ax.plot(k50, 50, "o", color="#D55E00", ms=6, zorder=5)
    ax.plot(k90, 90, "o", color="#D55E00", ms=6, zorder=5)
    ax.annotate(f"k={k50}", (k50, 50), textcoords="offset points",
                xytext=(8, -5), fontsize=8, color="#D55E00")
    ax.annotate(f"k={k90}", (k90, 90), textcoords="offset points",
                xytext=(8, -5), fontsize=8, color="#D55E00")

    ax.set_xlabel(f"Max concepts {verb} (k)", fontsize=10)
    ax.set_ylabel(f"% fixable rows covered", fontsize=10)
    ax.set_title(f"Cumulative coverage ({n_fixable} fixable rows)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_xlim(0.5, max_k + 0.5)

    # Feature frequency: which features appear most often in per-row explanations?
    from collections import Counter
    feat_counts = Counter()
    for row_idx, k in enumerate(optimal_k):
        if k > 0:
            feat_counts.update(perrow_rankings[row_idx][:k])
    if feat_counts:
        top5 = feat_counts.most_common(5)
        text = "Top features (by row frequency):\n"
        text += "\n".join(f"  f{f}: {c}/{n_fixable} rows ({100*c/n_fixable:.0f}%)"
                          for f, c in top5)
        ax.text(0.98, 0.05, text, transform=ax.transAxes, fontsize=7,
                va="bottom", ha="right", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # --- Bottom right: summary stats ---
    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        f"Dataset: {dataset}",
        f"Total rows: {n_total}",
        f"Improvable: {n_improvable} ({100*n_improvable/n_total:.0f}%)",
        f"  Fixable: {n_fixable} ({100*n_fixable/n_improvable:.0f}% of improvable)",
        f"  Unfixable: {n_unfixable} ({100*n_unfixable/n_improvable:.0f}% of improvable)",
        f"Not improvable: {n_total - n_improvable}",
        "",
    ]
    if n_fixable > 0 and n_unfixable > 0:
        lines.append(f"Firing concepts (fixable):   "
                     f"mean={avail_fixable.mean():.1f}, median={np.median(avail_fixable):.0f}")
        lines.append(f"Firing concepts (unfixable): "
                     f"mean={avail_unfixable.mean():.1f}, median={np.median(avail_unfixable):.0f}")

    # Add AUC summary if predictions are available
    if preds_strong is not None and preds_weak is not None and y_query is not None:
        from sklearn.metrics import roc_auc_score
        auc_s = roc_auc_score(y_query, preds_strong)
        auc_w = roc_auc_score(y_query, preds_weak)
        gap = auc_s - auc_w
        lines.append("")
        lines.append(f"{disp_abl} AUC: {auc_s:.4f}")
        lines.append(f"{disp_oth} AUC: {auc_w:.4f}")
        # Compute post-intervention AUC from optimal_k
        # For transfer: weak model changes; for ablation: strong model changes
        is_transfer = action == "transferring"
        # We don't have the full intervened preds here, but we can note the gap
        if abs(gap) > 0.001:
            lines.append(f"AUC gap: {gap:+.4f}")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=9,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", alpha=0.8))

    fig.suptitle(f"{verb_ing} {disp_abl}-only concepts → matching {disp_oth}",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-row results to %s", output_path)


def plot_perrow_diagnostic(
    optimal_k: np.ndarray,
    row_gap_closed: np.ndarray,
    accepted_preds: np.ndarray,
    baseline_preds: np.ndarray,
    source_preds: np.ndarray,
    max_k_per_row: np.ndarray,
    y_query: np.ndarray,
    ablate_model: str,
    other_model: str,
    dataset: str,
    output_path: Path,
    action: str = "ablating",
):
    """Diagnostic figure for per-row transfer/ablation.

    Panel 1: Per-row logloss distributions (strong, weak-before, weak-after)
    Panel 2: Per-row concept budget (available / accepted / rejected)
    Panel 3: Acceptance rate vs logloss reduction
    """
    eps = 1e-7
    verb_ing = "Transfer" if action == "transferring" else "Ablation"
    disp_abl = DISPLAY_NAMES.get(ablate_model, ablate_model)
    disp_oth = DISPLAY_NAMES.get(other_model, other_model)
    y = y_query.astype(float)

    # P(class=1) for each set
    sp1 = np.clip(source_preds[:, 1] if source_preds.ndim == 2 else source_preds, eps, 1 - eps)
    bp1 = np.clip(baseline_preds[:, 1] if baseline_preds.ndim == 2 else baseline_preds, eps, 1 - eps)
    ap1 = np.clip(accepted_preds[:, 1] if accepted_preds.ndim == 2 else accepted_preds, eps, 1 - eps)

    # Per-row logloss
    ll_strong = -(y * np.log(sp1) + (1 - y) * np.log(1 - sp1))
    ll_before = -(y * np.log(bp1) + (1 - y) * np.log(1 - bp1))
    ll_after = -(y * np.log(ap1) + (1 - y) * np.log(1 - ap1))

    # Filter to fixable rows
    fixable = optimal_k > 0
    n_fixable = int(fixable.sum())
    n_total = len(optimal_k)
    if n_fixable == 0:
        logger.warning("No fixable rows — skipping diagnostic plot.")
        return

    ll_s = ll_strong[fixable]
    ll_b = ll_before[fixable]
    ll_a = ll_after[fixable]
    n_available = max_k_per_row[fixable]
    n_accepted = optimal_k[fixable]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- Panel 1: Per-row logloss distributions ---
    ax = axes[0]
    max_ll = max(ll_b.max(), ll_a.max(), ll_s.max())
    bins = np.linspace(0, min(max_ll * 1.05, 5.0), 40)
    ax.hist(ll_b, bins=bins, color="#D55E00", alpha=0.4,
            label=f"{disp_oth} before (mean={ll_b.mean():.3f})")
    ax.hist(ll_a, bins=bins, color="#0072B2", alpha=0.4,
            label=f"{disp_oth} after (mean={ll_a.mean():.3f})")
    ax.hist(ll_s, bins=bins, color="#009E73", alpha=0.4,
            label=f"{disp_abl} target (mean={ll_s.mean():.3f})")
    ax.axvline(ll_b.mean(), color="#D55E00", ls=":", lw=1.5)
    ax.axvline(ll_a.mean(), color="#0072B2", ls=":", lw=1.5)
    ax.axvline(ll_s.mean(), color="#009E73", ls=":", lw=1.5)
    pct_improved = (ll_a < ll_b - 1e-6).mean()
    pct_worse = (ll_a > ll_b + 1e-6).mean()
    summary = (f"Improved: {pct_improved:.0%}\n"
               f"Unchanged: {1 - pct_improved - pct_worse:.0%}\n"
               f"Worsened: {pct_worse:.0%}")
    ax.text(0.97, 0.95, summary, transform=ax.transAxes, fontsize=8,
            va="top", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_xlabel("Per-row logloss", fontsize=10)
    ax.set_ylabel("Number of rows", fontsize=10)
    ax.set_title(f"{dataset}: logloss distributions ({n_fixable}/{n_total} fixable)",
                 fontsize=10)
    ax.legend(fontsize=7)

    # --- Panel 2: Concept budget per row ---
    ax = axes[1]
    max_avail = int(n_available.max())
    bins_k = np.arange(-0.5, max_avail + 1.5, max(1, max_avail // 30))
    ax.hist(n_available, bins=bins_k, color="#999999", alpha=0.4,
            label=f"Available (mean={n_available.mean():.1f})")
    ax.hist(n_accepted, bins=bins_k, color="#009E73", alpha=0.6,
            label=f"Accepted (mean={n_accepted.mean():.1f})")
    accept_rate = n_accepted.sum() / max(n_available.sum(), 1)
    ax.set_xlabel("Concepts per row", fontsize=10)
    ax.set_ylabel("Number of rows", fontsize=10)
    ax.set_title(f"{dataset}: concept budget (accept rate {accept_rate:.0%})", fontsize=10)
    ax.legend(fontsize=8)

    # --- Panel 3: Acceptance rate vs logloss reduction ---
    ax = axes[2]
    ll_reduction = ll_b - ll_a  # positive = logloss decreased = improved
    accept_frac = n_accepted / np.maximum(n_available, 1)
    colors = np.where(ll_reduction > 0, "#009E73", "#D55E00")
    ax.scatter(accept_frac, ll_reduction, c=colors, alpha=0.4, s=15, edgecolors="none")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Acceptance rate (accepted / available)", fontsize=10)
    ax.set_ylabel("Logloss reduction (positive = improved)", fontsize=10)
    ax.set_title(f"{dataset}: accept rate vs improvement", fontsize=10)
    n_helped = int((ll_reduction > 1e-6).sum())
    ax.text(0.97, 0.95, f"{n_helped}/{n_fixable} rows improved",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle(f"{verb_ing} diagnostic: {disp_abl} → {disp_oth}", fontsize=11, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-row diagnostic to %s", output_path)


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
    preds_intervened: np.ndarray,
    optimal_k: np.ndarray,
    y_true: np.ndarray,
    model_strong: str,
    model_weak: str,
    dataset: str,
    auc_strong: float,
    auc_weak: float,
    output_path: Path,
    mode: str = "ablation",
    preds_intervened_rev: Optional[np.ndarray] = None,
    optimal_k_rev: Optional[np.ndarray] = None,
):
    """Scatter showing per-row intervention results.

    Convention: x = strong model, y = weak model, always.
    Marker convention: circle = strong model, square = weak model.
    Open markers = original, filled markers = intervened.

    Modes:
      - "ablation": strong model degraded (x changes). Circle markers.
      - "transfer": weak model improved via strong concepts (y changes). Square markers.
      - "reverse_transfer": strong model improved via weak concepts (x changes). Circle markers.
      - "bidirectional": both directions on one plot. Squares for forward, circles for reverse.
        Requires preds_intervened_rev and optimal_k_rev.

    Shows rows where the source model outperforms the target model.
    Rows with optimal_k > 0 get colored intervened markers; others are gray only.
    """
    from sklearn.metrics import roc_auc_score

    disp_s = DISPLAY_NAMES.get(model_strong, model_strong)
    disp_w = DISPLAY_NAMES.get(model_weak, model_weak)
    event_rate = y_true.mean()

    if mode == "bidirectional":
        return _plot_bidirectional_scatter(
            preds_strong, preds_weak,
            preds_intervened, optimal_k,
            preds_intervened_rev, optimal_k_rev,
            y_true, disp_s, disp_w, dataset,
            auc_strong, auc_weak, event_rate, output_path,
        )

    is_reverse = mode == "reverse_transfer"
    if mode == "transfer":
        verb_lower, marker = "transfer", "s"
    elif mode == "reverse_transfer":
        verb_lower, marker = "reverse transfer", "o"
    else:
        verb_lower, marker = "ablation", "o"

    if is_reverse:
        source_better = get_improvable_rows(preds_weak, preds_strong, y_true)
    else:
        source_better = get_improvable_rows(preds_strong, preds_weak, y_true)

    n_source_better = source_better.sum()
    n_intervened = (optimal_k > 0).sum()
    n_total = len(optimal_k)
    src_label = "Weak outperforms strong" if is_reverse else "Strong outperforms weak"
    logger.info("%s: %d / %d (%.1f%%); intervened: %d",
                src_label, n_source_better, n_total,
                100 * n_source_better / n_total, n_intervened)

    if n_source_better == 0:
        logger.warning("No improvable rows — skipping scatter.")
        return

    ps = preds_strong[source_better]
    pw = preds_weak[source_better]
    pi = preds_intervened[source_better]
    yt = y_true[source_better]
    ok = optimal_k[source_better]

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    orig_x, orig_y = ps, pw
    if mode == "transfer":
        int_x, int_y = ps, pi
    else:
        int_x, int_y = pi, pw

    _draw_scatter_layers(ax, orig_x, orig_y, int_x, int_y, yt, ok, marker,
                         event_rate)

    _draw_scatter_chrome(ax, preds_strong, preds_weak, int_x, int_y,
                         disp_s, disp_w, event_rate)

    ok_mod = ok[ok > 0]
    k_summary = (f"median k={np.median(ok_mod):.0f}, mean k={ok_mod.mean():.1f}"
                 if ok_mod.size > 0 else "no rows modified")

    # Post-intervention metric
    def _safe_auc(y, p):
        """AUC for binary, neg_logloss proxy for multiclass/regression."""
        try:
            return roc_auc_score(y, p)
        except ValueError:
            # Multiclass or single-class: use mean P(correct) as proxy
            return float(np.mean(p))

    if mode == "transfer":
        full_p1 = np.where(optimal_k > 0, preds_intervened, preds_weak)
        auc_post = _safe_auc(y_true, full_p1)
        gap = auc_strong - auc_weak
        pct = (auc_post - auc_weak) / gap * 100 if abs(gap) > 0.001 else float("nan")
        auc_line = (f"{disp_s}: AUC={auc_strong:.3f}   |   {disp_w}: AUC={auc_weak:.3f}"
                    f"   |   {disp_w}+: AUC={auc_post:.3f} ({pct:+.0f}%)")
    elif mode == "reverse_transfer":
        full_p1 = np.where(optimal_k > 0, preds_intervened, preds_strong)
        auc_post = _safe_auc(y_true, full_p1)
        auc_line = (f"{disp_s}: AUC={auc_strong:.3f}   |   {disp_w}: AUC={auc_weak:.3f}"
                    f"   |   {disp_s}+: AUC={auc_post:.3f} "
                    f"(\u0394={auc_post - auc_strong:+.4f})")
    else:
        full_p1 = np.where(optimal_k > 0, preds_intervened, preds_strong)
        auc_post = _safe_auc(y_true, full_p1)
        gap = auc_strong - auc_weak
        pct = (auc_post - auc_weak) / gap * 100 if abs(gap) > 0.001 else float("nan")
        auc_line = (f"{disp_s}: AUC={auc_strong:.3f}   |   {disp_w}: AUC={auc_weak:.3f}"
                    f"   |   {disp_s}+: AUC={auc_post:.3f} ({pct:+.0f}%)")

    ax.set_title(
        f"{dataset} — per-row {verb_lower} "
        f"({n_intervened} intervened / {n_source_better} improvable / {n_total} total)\n"
        f"{auc_line}\n{k_summary}",
        fontsize=9,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-row scatter to %s", output_path)


def _draw_scatter_layers(ax, orig_x, orig_y, int_x, int_y, yt, ok,
                         marker, event_rate, label_prefix=""):
    """Draw the three scatter layers: original, unmoved, intervened."""
    modified = ok > 0
    unmodified = ~modified
    mask0 = yt == 0
    mask1 = yt == 1

    pfx = f"{label_prefix} " if label_prefix else ""

    # Layer 1: open gray originals
    ax.scatter(orig_x, orig_y, facecolors="none",
               edgecolors="#999999", s=14, alpha=0.5, linewidths=0.6,
               marker=marker,
               label=f"{pfx}Original (n={len(orig_x)})", zorder=2)

    # Layer 2: filled gray unmoved
    n_un = int(unmodified.sum())
    if n_un > 0:
        ax.scatter(orig_x[unmodified], orig_y[unmodified], c="#aaaaaa",
                   s=14, alpha=0.5, edgecolors="none", marker=marker,
                   label=f"{pfx}Unmoved (n={n_un})", zorder=3)

    # Layer 3: filled color by class
    m0 = mask0 & modified
    m1 = mask1 & modified
    if m0.any():
        ax.scatter(int_x[m0], int_y[m0], c="#0072B2",
                   s=14, alpha=0.6, edgecolors="none", marker=marker,
                   label=f"{pfx}Intervened cls 0 (n={m0.sum()})", zorder=4)
    if m1.any():
        ax.scatter(int_x[m1], int_y[m1], c="#D55E00",
                   s=14, alpha=0.6, edgecolors="none", marker=marker,
                   label=f"{pfx}Intervened cls 1 (n={m1.sum()})", zorder=4)


def _draw_scatter_chrome(ax, preds_strong, preds_weak, int_x, int_y,
                         disp_s, disp_w, event_rate):
    """Add axes, grid, event-rate lines, y=x."""
    all_vals = np.concatenate([preds_strong, preds_weak,
                               np.asarray(int_x), np.asarray(int_y)])
    lo = max(0, all_vals.min() - 0.02)
    hi = min(1, all_vals.max() + 0.02)
    hi = max(hi, event_rate + 0.02)

    ax.grid(True, which="major", color="#cccccc", lw=0.5, alpha=0.7, zorder=0)
    ax.grid(True, which="minor", color="#eeeeee", lw=0.3, alpha=0.5, zorder=0)
    ax.minorticks_on()

    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y = x")
    ax.axhline(event_rate, color="gray", lw=0.7, ls=":", alpha=0.7)
    ax.axvline(event_rate, color="gray", lw=0.7, ls=":", alpha=0.7)
    ax.text(0.97, event_rate, f" event rate = {event_rate:.3f}",
            fontsize=7, color="gray", va="bottom", ha="right",
            transform=ax.get_yaxis_transform())

    ax.set_xlabel(f"{disp_s}  P(class=1)", fontsize=10)
    ax.set_ylabel(f"{disp_w}  P(class=1)", fontsize=10)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")


def _plot_bidirectional_scatter(
    preds_strong, preds_weak,
    preds_intervened_fwd, optimal_k_fwd,
    preds_intervened_rev, optimal_k_rev,
    y_true, disp_s, disp_w, dataset,
    auc_strong, auc_weak, event_rate, output_path,
):
    """Bidirectional transfer: forward (squares) + reverse (circles) on one plot."""
    from sklearn.metrics import roc_auc_score

    # Forward: strong outperforms weak → weak model improved (squares, y changes)
    fwd_fixable = get_improvable_rows(preds_strong, preds_weak, y_true)
    # Reverse: weak outperforms strong → strong model improved (circles, x changes)
    rev_fixable = get_improvable_rows(preds_weak, preds_strong, y_true)

    n_fwd = fwd_fixable.sum()
    n_rev = rev_fixable.sum()
    n_fwd_int = int((optimal_k_fwd[fwd_fixable] > 0).sum()) if n_fwd > 0 else 0
    n_rev_int = int((optimal_k_rev[rev_fixable] > 0).sum()) if n_rev > 0 else 0
    n_total = len(y_true)

    logger.info("Bidirectional: fwd fixable=%d (intervened=%d), "
                "rev fixable=%d (intervened=%d), total=%d",
                n_fwd, n_fwd_int, n_rev, n_rev_int, n_total)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    # Forward direction: squares (weak model modified, y changes)
    if n_fwd > 0:
        ps_f = preds_strong[fwd_fixable]
        pw_f = preds_weak[fwd_fixable]
        pi_f = preds_intervened_fwd[fwd_fixable]
        yt_f = y_true[fwd_fixable]
        ok_f = optimal_k_fwd[fwd_fixable]
        _draw_scatter_layers(ax, ps_f, pw_f, ps_f, pi_f, yt_f, ok_f,
                             "s", event_rate, label_prefix="\u25a1")  # □

    # Reverse direction: circles (strong model modified, x changes)
    if n_rev > 0:
        ps_r = preds_strong[rev_fixable]
        pw_r = preds_weak[rev_fixable]
        pi_r = preds_intervened_rev[rev_fixable]
        yt_r = y_true[rev_fixable]
        ok_r = optimal_k_rev[rev_fixable]
        _draw_scatter_layers(ax, ps_r, pw_r, pi_r, pw_r, yt_r, ok_r,
                             "o", event_rate, label_prefix="\u25cb")  # ○

    # Collect all intervened coords for auto-zoom
    all_int_x = np.concatenate([
        preds_strong[fwd_fixable] if n_fwd else np.array([]),
        preds_intervened_rev[rev_fixable] if n_rev else np.array([]),
    ])
    all_int_y = np.concatenate([
        preds_intervened_fwd[fwd_fixable] if n_fwd else np.array([]),
        preds_weak[rev_fixable] if n_rev else np.array([]),
    ])
    _draw_scatter_chrome(ax, preds_strong, preds_weak, all_int_x, all_int_y,
                         disp_s, disp_w, event_rate)

    ax.legend(fontsize=6.5, loc="upper left")

    # AUC summaries for both directions
    fwd_p1 = np.where(optimal_k_fwd > 0, preds_intervened_fwd, preds_weak)
    auc_fwd = roc_auc_score(y_true, fwd_p1)
    rev_p1 = np.where(optimal_k_rev > 0, preds_intervened_rev, preds_strong)
    auc_rev = roc_auc_score(y_true, rev_p1)

    gap = auc_strong - auc_weak
    fwd_pct = (auc_fwd - auc_weak) / gap * 100 if abs(gap) > 0.001 else float("nan")

    ax.set_title(
        f"{dataset} — bidirectional transfer "
        f"({n_fwd_int}+{n_rev_int} intervened / {n_fwd}+{n_rev} fixable / {n_total} total)\n"
        f"{disp_s}: {auc_strong:.3f}   {disp_w}: {auc_weak:.3f}   "
        f"\u25a1 {disp_w}+: {auc_fwd:.3f} ({fwd_pct:+.0f}%)   "
        f"\u25cb {disp_s}+: {auc_rev:.3f} (\u0394={auc_rev - auc_strong:+.4f})",
        fontsize=8.5,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved bidirectional scatter to %s", output_path)


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


def _plot_ablation_sweep_results(npz_path: Path, output_path: Path):
    """Load ablation_sweep NPZ and generate scatter + histogram plots.

    The NPZ contains pre-computed strong_wins mask (multiclass-aware),
    so no loss recomputation is needed here.
    """
    data = np.load(npz_path, allow_pickle=False)

    preds_s = data["preds_strong"]
    preds_w = data["preds_weak"]
    preds_i = data["preds_intervened"]
    y = data["y_query"].astype(int)
    optimal_k = data["optimal_k"]
    strong_wins = data["strong_wins"]

    # Extract model names from path: {strong}_vs_{weak}/{dataset}.npz
    pair_name = npz_path.parent.name  # e.g. "tabicl_vs_tabdpt"
    dataset = npz_path.stem
    parts = pair_name.split("_vs_")
    model_strong = parts[0] if len(parts) == 2 else "strong"
    model_weak = parts[1] if len(parts) == 2 else "weak"
    disp_s = DISPLAY_NAMES.get(model_strong, model_strong)
    disp_w = DISPLAY_NAMES.get(model_weak, model_weak)

    # Scalar scores for scatter axes
    n_classes = preds_s.shape[1] if preds_s.ndim == 2 else 1
    if n_classes > 1:
        # Classification: P(correct class)
        p_s = preds_s[np.arange(len(y)), y]
        p_w = preds_w[np.arange(len(y)), y]
        p_i = preds_i[np.arange(len(y)), y]
        axis_label = "P(correct class)"
    else:
        # Regression: raw predictions
        p_s = preds_s.ravel()
        p_w = preds_w.ravel()
        p_i = preds_i.ravel()
        axis_label = "Prediction"

    n_total = len(y)
    n_improvable = int(strong_wins.sum())
    n_intervened = int((optimal_k > 0).sum())
    valid_k = optimal_k[strong_wins & (optimal_k > 0)]
    weak_wins = ~strong_wins

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: scatter ---
    ax = axes[0]
    ax.scatter(p_s[weak_wins], p_w[weak_wins], facecolors="none",
               edgecolors="#999999", s=14, alpha=0.5, linewidths=0.6,
               label=f"{disp_w} wins (n={weak_wins.sum()})", zorder=2)

    ax.scatter(p_s[strong_wins], p_w[strong_wins], facecolors="none",
               edgecolors="steelblue", s=20, alpha=0.7, linewidths=0.8,
               label=f"{disp_s} wins (n={n_improvable})", zorder=3)

    mask = strong_wins & (optimal_k > 0)
    for r in np.where(mask)[0]:
        ax.annotate("", xy=(p_i[r], p_w[r]), xytext=(p_s[r], p_w[r]),
                    arrowprops=dict(arrowstyle="->", color="#e74c3c", alpha=0.35, lw=0.7))

    ax.scatter(p_i[mask], p_w[mask], c="#e74c3c", s=16, alpha=0.7,
               label=f"After ablation (n={n_intervened})", zorder=4)

    lims = [min(p_s.min(), p_w.min()) - 0.02, max(p_s.max(), p_w.max()) + 0.02]
    ax.plot(lims, lims, "k--", alpha=0.3, lw=1)
    ax.set_xlabel(f"{disp_s}  {axis_label}", fontsize=11)
    ax.set_ylabel(f"{disp_w}  {axis_label}", fontsize=11)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower right")

    ax.set_title(
        f"{dataset} — per-row ablation "
        f"({n_intervened} intervened / {n_improvable} improvable / {n_total} total)",
        fontsize=10,
    )

    # --- Right: histogram of optimal_k ---
    ax2 = axes[1]
    if len(valid_k) > 0:
        ax2.hist(valid_k, bins=np.arange(0.5, valid_k.max() + 1.5, 1),
                 color="steelblue", edgecolor="white", alpha=0.8)
        ax2.axvline(valid_k.mean(), color="red", linestyle="--", alpha=0.7,
                    label=f"Mean={valid_k.mean():.1f}")
        ax2.axvline(np.median(valid_k), color="orange", linestyle="--", alpha=0.7,
                    label=f"Median={np.median(valid_k):.0f}")
        ax2.legend(fontsize=9)
    ax2.set_xlabel(f"Concepts removed to match {disp_w}", fontsize=11)
    ax2.set_ylabel("Number of rows", fontsize=11)
    ax2.set_title(f"Concept gap distribution", fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ablation scatter to %s", output_path)


def _plot_transfer_results(npz_path: Path, output_path: Path):
    """Load saved vnode transfer results and generate all plots."""
    data = np.load(npz_path, allow_pickle=False)

    source_model = str(data["source_model"])
    target_model = str(data["target_model"])
    dataset = str(data["dataset"])
    sp1 = data["preds_strong"]
    tp1 = data["preds_weak"]
    transferred_p1 = data["preds_transferred"]
    optimal_k = data["optimal_k"]
    row_gap_closed = data["row_gap_closed"]
    max_k_per_row = data["max_k_per_row"]
    rankings_padded = data["perrow_rankings"]
    y_q = data["y_query"]
    auc_s = float(data["auc_strong"])
    auc_t = float(data["auc_weak"])
    auc_transferred = float(data["auc_transferred"])

    # Unpad rankings (remove -1 padding)
    rankings = []
    for row in rankings_padded:
        rankings.append([int(x) for x in row if x >= 0])

    fig_dir = output_path.parent
    fig_dir.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix

    disp_s = DISPLAY_NAMES.get(source_model, source_model)
    disp_t = DISPLAY_NAMES.get(target_model, target_model)

    logger.info(
        "Transfer: %s (AUC=%.3f) -> %s (AUC=%.3f), transferred AUC=%.3f",
        disp_s, auc_s, disp_t, auc_t, auc_transferred,
    )

    # Per-row histogram + coverage curve
    plot_perrow_results(
        optimal_k, row_gap_closed, max_k_per_row, rankings,
        source_model, target_model, dataset,
        fig_dir / f"vnode_transfer_perrow{suffix}",
        action="transferring",
        preds_strong=sp1,
        preds_weak=tp1,
        y_query=y_q,
    )

    # Per-row scatter
    plot_perrow_scatter(
        sp1, tp1, transferred_p1,
        optimal_k, y_q,
        source_model, target_model, dataset,
        auc_s, auc_t,
        fig_dir / f"vnode_transfer_perrow_scatter{suffix}",
        mode="transfer",
    )

    # Diagnostic
    plot_perrow_diagnostic(
        optimal_k, row_gap_closed,
        data["accepted_preds"], data["baseline_preds"], data["source_preds"],
        max_k_per_row, y_q,
        source_model, target_model, dataset,
        fig_dir / f"vnode_transfer_perrow_diagnostic{suffix}",
        action="transferring",
    )

    gap = auc_s - auc_t
    gap_closed_pct = (auc_transferred - auc_t) / gap * 100 if abs(gap) > 0.001 else float("nan")
    print(f"\n{disp_s} -> {disp_t} on {dataset}")
    print(f"  {disp_t}+vnode AUC = {auc_transferred:.4f} (gap closed {gap_closed_pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Prediction scatter plot")
    parser.add_argument("--dataset", type=str, default="kddcup09_appetency")
    parser.add_argument("--model-a", type=str, default="tabpfn")
    parser.add_argument("--model-b", type=str, default="tabicl")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "output" / "figures" / "prediction_scatter.pdf")

    # Intervention modes (mutually exclusive)
    abl_group = parser.add_mutually_exclusive_group()
    abl_group.add_argument("--ablate-unmatched", action="store_true",
                           help="Ablate unmatched features from the stronger model "
                                "(auto-detected by AUC)")
    abl_group.add_argument("--perrow", action="store_true",
                           help="Per-row heterogeneous ablation: find minimal concept "
                                "set per row to close the logloss gap")
    abl_group.add_argument("--ablate-features", type=str, default=None,
                           help="Comma-separated feature indices to ablate from model-a")
    abl_group.add_argument("--transfer", type=Path, default=None,
                           help="Plot from saved vnode transfer results (.npz). "
                                "Ignores --model-a/--model-b/--dataset/--device.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # ── Transfer mode: load saved results and plot ────────────────────────
    if args.transfer is not None:
        _plot_transfer_results(args.transfer, args.output)
        return

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
            strong_preds = preds_a if ablate_axis == "x" else preds_b
            weak_preds = preds_b if ablate_axis == "x" else preds_a
            perrow_path = fig_dir / f"perrow_ablation{args.output.suffix}"
            plot_perrow_results(
                perrow["optimal_k"], perrow["row_gap_closed"],
                perrow["max_k_per_row"], perrow["perrow_rankings"],
                ablate_model, other_model, args.dataset,
                perrow_path,
                preds_strong=strong_preds,
                preds_weak=weak_preds,
                y_query=y_q,
            )

            # Per-row scatter: filtered to fixable rows
            ablated_preds = extract_perrow_ablated_preds(
                perrow["optimal_k"], perrow["sweep_preds"],
                perrow["baseline_preds"],
            )
            auc_strong = max(auc_a, auc_b)
            auc_weak = min(auc_a, auc_b)
            perrow_scatter_path = fig_dir / f"perrow_scatter{args.output.suffix}"
            plot_perrow_scatter(
                strong_preds, weak_preds, ablated_preds,
                perrow["optimal_k"], y_q,
                ablate_model, other_model, args.dataset,
                auc_strong, auc_weak,
                perrow_scatter_path,
                mode="ablation",
            )

    elif args.ablate_features:
        feats = [int(x.strip()) for x in args.ablate_features.split(",")]
        logger.info("Ablating features %s from %s...", feats, args.model_a)
        abl_preds = get_ablated_predictions(
            args.model_a, args.dataset, task, feats, args.device,
        )
        disp = DISPLAY_NAMES.get(args.model_a, args.model_a)
        ablation_levels = [(f"ablate {len(feats)}f from {disp}", "#0072B2", abl_preds)]

    # Global scatter plot (skip for --perrow which produces its own scatter)
    if not args.perrow:
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
