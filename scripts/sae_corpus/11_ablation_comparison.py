#!/usr/bin/env python3
"""Compare task-aware vs per-dataset SAEs via per-row ablation.

Uses pre-computed per-row importance from 09_evaluate_layer_comparison.py
to skip Phase 1+2. Goes straight to Phase 3: per-row selective ablation
to match TabPFN's predictions to Mitra's predictions.

For each SAE variant:
  1. Load pre-computed per-row importance rankings
  2. Get TabPFN baseline predictions + Mitra baseline predictions
  3. Per-row: ablate features in rank order until TabPFN matches Mitra
  4. Scatter plot: ablated TabPFN vs Mitra (should be on y=x)

Usage:
    python scripts/sae_corpus/11_ablation_comparison.py --device cuda
"""
import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.sparse_autoencoder import SAEConfig, SparseAutoencoder
from data.preprocessing import CACHE_DIR, load_preprocessed
from models.layer_extraction import load_and_fit, predict
from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_sae import (
    build_tail,
    compute_ablation_delta,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_training_round10"

EVAL_DATASETS = {
    "airfoil_self_noise": {"task": "regression"},
    "diabetes": {"task": "classification"},
}

VARIANTS = {
    "task_aware": {
        "sae_path": OUTPUT_DIR / "tabpfn_taskaware_sae.pt",
        "stats_path": OUTPUT_DIR / "tabpfn_taskaware_norm_stats.npz",
        "importance_prefix": "layer_comparison_eval_task_aware",
    },
    "per_dataset": {
        "sae_path": OUTPUT_DIR / "tabpfn_perds_sae.pt",
        "stats_path": OUTPUT_DIR / "tabpfn_perds_norm_stats.npz",
        "importance_prefix": "layer_comparison_eval_per_dataset",
    },
}


def load_sae(sae_path, device):
    ckpt = torch.load(str(sae_path), map_location=device, weights_only=False)
    config = SAEConfig(**ckpt["config"])
    sae = SparseAutoencoder(config)
    state = ckpt["state_dict"]
    if "reference_data" in state and state["reference_data"] is not None:
        sae.register_buffer("reference_data", state["reference_data"])
        if "archetype_logits" in state:
            sae.archetype_logits = torch.nn.Parameter(state["archetype_logits"])
        if "archetype_deviation" in state:
            sae.archetype_deviation = torch.nn.Parameter(state["archetype_deviation"])
    sae.load_state_dict(state, strict=False)
    sae.to(device)
    sae.eval()
    return sae, config


def load_norm_stats(stats_path, dataset, device):
    stats = np.load(str(stats_path), allow_pickle=True)
    datasets = list(stats["datasets"])
    idx = datasets.index(dataset)
    mean = torch.tensor(stats["means"][idx], dtype=torch.float32, device=device)
    std = torch.tensor(stats["stds"][idx], dtype=torch.float32, device=device)
    layer = int(stats["layers"][idx])
    return mean, std, layer


def load_importance(prefix, dataset):
    """Load pre-computed per-row importance from 09."""
    path = OUTPUT_DIR / f"{prefix}_{dataset}.npz"
    data = np.load(str(path), allow_pickle=True)
    return data["row_feature_drops"], data["feature_indices"]


def get_predictions(model_name, X_ctx, y_ctx, X_q, task, device):
    """Get baseline predictions from a model."""
    fit_kwargs = {}
    if model_name == "tabpfn":
        data = load_preprocessed("tabpfn", "diabetes", CACHE_DIR)  # just for cat_indices
        if data.cat_indices:
            fit_kwargs["cat_indices"] = data.cat_indices
    clf = load_and_fit(model_name, X_ctx, y_ctx, task=task, device=device, **fit_kwargs)
    return predict(clf, X_q, task=task)


def pred_scalar(preds):
    """Extract scalar prediction (p(class=1) or regression value)."""
    if isinstance(preds, np.ndarray) and preds.ndim == 2:
        return preds[:, 1] if preds.shape[1] == 2 else preds[:, 0]
    return preds.ravel()


def row_loss(y, p, task):
    eps = 1e-7
    if task == "regression":
        return (y - p) ** 2
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def run_ablation(
    sae, data_mean, data_std, extraction_layer,
    row_drops, feat_indices,
    X_ctx, y_ctx, X_q, y_q,
    strong_preds, weak_preds,
    task, device,
):
    """Phase 3: per-row selective ablation using pre-computed importance."""
    n_query = len(X_q)
    sp1 = pred_scalar(strong_preds)
    wp1 = pred_scalar(weak_preds)

    # Build per-row rankings: sort features by importance (most important first)
    # Only include features that fire (importance != 0 is a rough proxy)
    mean_drops = row_drops.mean(axis=0)
    # Global ranking by mean importance
    global_order = np.argsort(-mean_drops)
    # Filter to features with positive mean importance
    ranked_feat_indices = [int(feat_indices[i]) for i in global_order if mean_drops[i] > 0]
    logger.info("  Ranked features: %d with positive importance", len(ranked_feat_indices))

    # Build tail model
    logger.info("  Building TabPFN tail at L%d...", extraction_layer)
    tail = build_tail("tabpfn", X_ctx, y_ctx, X_q, extraction_layer, task, device)

    # Capture embeddings for delta computation
    captured = {}
    from models.tabpfn_utils import load_tabpfn
    clf = load_tabpfn(task=task, device=device, n_estimators=1)
    clf.fit(X_ctx, y_ctx)
    model = clf.model_ if hasattr(clf, "model_") else clf.transformer_
    layers = model.transformer_encoder.layers

    def capture_hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            captured["hidden"] = out.detach()

    handle = layers[extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            if task == "regression":
                clf.predict(X_q)
            else:
                clf.predict_proba(X_q)
    finally:
        handle.remove()

    all_emb = captured["hidden"]
    if all_emb.ndim == 4:
        all_emb = all_emb[0].mean(dim=1)
    elif all_emb.ndim == 3:
        all_emb = all_emb[0] if all_emb.shape[0] == 1 else all_emb.mean(dim=0)

    # Pre-filter: only ablate rows where strong is better
    y = y_q.astype(float)
    strong_loss = row_loss(y, sp1, task)
    weak_loss = row_loss(y, wp1, task)
    fixable = strong_loss < weak_loss
    n_fixable = int(fixable.sum())
    logger.info("  Fixable rows: %d / %d", n_fixable, n_query)

    # Precompute per-concept deltas
    logger.info("  Precomputing deltas for %d features...", len(ranked_feat_indices))
    concept_deltas = {}
    for feat_idx in ranked_feat_indices:
        delta = compute_ablation_delta(
            sae, all_emb[-n_query:], [feat_idx],
            data_mean=data_mean, data_std=data_std,
        )
        norms = delta.norm(dim=1, keepdim=True).clamp(min=1e-10)
        concept_deltas[feat_idx] = delta / norms

    # Per-row ablation with line search
    d_model = all_emb.shape[1]
    cumulative_deltas = torch.zeros(n_query, d_model, device=device)
    accepted_counts = np.zeros(n_query, dtype=int)
    final_p1 = sp1.copy()
    max_backtrack = 6
    converge_threshold = 1e-3

    logger.info("  Phase 3: per-row ablation...")
    for row_idx in range(n_query):
        if not fixable[row_idx]:
            continue

        current_p1 = float(sp1[row_idx])
        target_p1 = float(wp1[row_idx])
        best_dist = abs(current_p1 - target_p1)

        if best_dist < 1e-8:
            continue

        for feat_idx in ranked_feat_indices:
            if feat_idx not in concept_deltas:
                continue

            concept_delta_row = concept_deltas[feat_idx][row_idx]

            # Probe at scale=1
            probe_delta = cumulative_deltas[row_idx] + concept_delta_row
            probe_preds = tail.predict_row(row_idx, probe_delta)
            probe_p1 = float(pred_scalar(probe_preds)[row_idx])

            grad = probe_p1 - current_p1
            if abs(grad) < 1e-10:
                continue

            s_star = (target_p1 - current_p1) / grad

            # Backtracking
            trial_scale = s_star
            for bt in range(max_backtrack + 1):
                if abs(trial_scale) < 1e-10:
                    break
                if bt == 0 and abs(trial_scale - 1.0) < 0.01:
                    trial_p1 = probe_p1
                    trial_delta = probe_delta
                else:
                    trial_delta = cumulative_deltas[row_idx] + trial_scale * concept_delta_row
                    trial_preds = tail.predict_row(row_idx, trial_delta)
                    trial_p1 = float(pred_scalar(trial_preds)[row_idx])

                trial_dist = abs(trial_p1 - target_p1)
                if trial_dist < best_dist - 1e-8:
                    cumulative_deltas[row_idx] = trial_delta
                    accepted_counts[row_idx] += 1
                    best_dist = trial_dist
                    current_p1 = trial_p1
                    break
                trial_scale *= 0.5

            if best_dist < converge_threshold:
                break

        final_p1[row_idx] = current_p1

        if (row_idx + 1) % 50 == 0 or row_idx == n_query - 1:
            logger.info("    row %d/%d: %d concepts, dist %.4f → %.4f",
                        row_idx + 1, n_query,
                        int(accepted_counts[row_idx]),
                        abs(float(sp1[row_idx]) - float(wp1[row_idx])),
                        best_dist)

    logger.info("  Done: mean %.1f, median %.0f, max %d concepts/row",
                accepted_counts.mean(), np.median(accepted_counts),
                int(accepted_counts.max()))

    return final_p1, accepted_counts, sp1, wp1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    all_results = {}

    for ds_name, ds_info in EVAL_DATASETS.items():
        task = ds_info["task"]
        print(f"\n{'=' * 70}")
        print(f"  {ds_name} ({task})")
        print("=" * 70)

        # Load data
        data = load_preprocessed("tabpfn", ds_name, CACHE_DIR)
        X_ctx, y_ctx = data.X_train[:600], data.y_train[:600]
        X_q, y_q = data.X_test[:500], data.y_test[:500]

        # Get Mitra predictions (weak model)
        logger.info("  Getting Mitra predictions...")
        mitra_data = load_preprocessed("mitra", ds_name, CACHE_DIR)
        mitra_clf = load_and_fit("mitra", mitra_data.X_train[:600], mitra_data.y_train[:600],
                                  task=task, device=args.device)
        weak_preds = predict(mitra_clf, mitra_data.X_test[:500], task=task)
        del mitra_clf
        torch.cuda.empty_cache()

        ds_results = {}

        for var_name, var_info in VARIANTS.items():
            print(f"\n  --- {var_name} ---")

            sae, config = load_sae(var_info["sae_path"], args.device)
            data_mean, data_std, layer = load_norm_stats(
                var_info["stats_path"], ds_name, args.device)

            # Load pre-computed importance
            row_drops, feat_indices = load_importance(
                var_info["importance_prefix"], ds_name)

            logger.info("  Layer: L%d, %d features with importance data",
                        layer, len(feat_indices))

            # Get TabPFN predictions (strong model)
            fit_kwargs = {}
            if data.cat_indices:
                fit_kwargs["cat_indices"] = data.cat_indices
            strong_clf = load_and_fit("tabpfn", X_ctx, y_ctx,
                                       task=task, device=args.device, **fit_kwargs)
            strong_preds = predict(strong_clf, X_q, task=task)
            del strong_clf
            torch.cuda.empty_cache()

            t0 = time.time()
            final_p1, accepted, sp1, wp1 = run_ablation(
                sae, data_mean, data_std, layer,
                row_drops, feat_indices,
                X_ctx, y_ctx, X_q, y_q,
                strong_preds, weak_preds,
                task, args.device,
            )
            elapsed = time.time() - t0

            # Metrics
            gap_before = float(np.sqrt(np.mean((sp1 - wp1) ** 2)))
            gap_after = float(np.sqrt(np.mean((final_p1 - wp1) ** 2)))
            corr = float(np.corrcoef(final_p1, wp1)[0, 1]) if len(np.unique(wp1)) > 1 else 0.0

            print(f"  Gap before: {gap_before:.4f}")
            print(f"  Gap after:  {gap_after:.4f}")
            print(f"  Closed:     {(1 - gap_after/gap_before)*100:.1f}%")
            print(f"  Corr:       {corr:.4f}")
            print(f"  Concepts:   mean={accepted.mean():.1f}, max={accepted.max()}")
            print(f"  Time:       {elapsed:.1f}s")

            ds_results[var_name] = {
                "gap_before": gap_before,
                "gap_after": gap_after,
                "gap_closed_pct": float((1 - gap_after/gap_before) * 100),
                "correlation": corr,
                "concepts_mean": float(accepted.mean()),
                "concepts_max": int(accepted.max()),
                "extraction_layer": layer,
                "strong_p1": sp1.tolist(),
                "weak_p1": wp1.tolist(),
                "ablated_p1": final_p1.tolist(),
            }

        all_results[ds_name] = ds_results

    # Save + plot
    out_path = OUTPUT_DIR / "ablation_comparison_results.json"
    with open(str(out_path), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n→ {out_path}")

    # Scatter plots
    n_ds = len(all_results)
    fig, axes = plt.subplots(n_ds, 2, figsize=(12, 5 * n_ds), squeeze=False)
    colors = {"task_aware": "#4C72B0", "per_dataset": "#DD8452"}

    for row, (ds_name, ds_results) in enumerate(all_results.items()):
        for col, var_name in enumerate(VARIANTS):
            ax = axes[row, col]
            r = ds_results.get(var_name, {})
            if not r:
                continue
            weak = np.array(r["weak_p1"])
            ablated = np.array(r["ablated_p1"])
            ax.scatter(weak, ablated, s=8, alpha=0.5, color=colors[var_name])
            lo, hi = min(weak.min(), ablated.min()), max(weak.max(), ablated.max())
            m = (hi - lo) * 0.05
            ax.plot([lo-m, hi+m], [lo-m, hi+m], "k--", lw=1, alpha=0.5)
            ax.set_xlabel("Mitra prediction")
            ax.set_ylabel("Ablated TabPFN")
            ax.set_title(f"{ds_name} — {var_name} (L{r['extraction_layer']})\n"
                         f"gap closed: {r['gap_closed_pct']:.1f}%, "
                         f"r={r['correlation']:.3f}", fontsize=10)
            ax.set_aspect("equal")

    fig.suptitle("Per-row ablation: TabPFN → Mitra", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "ablation_comparison_scatter.png", dpi=150, bbox_inches="tight")
    print(f"→ {OUTPUT_DIR / 'ablation_comparison_scatter.png'}")


if __name__ == "__main__":
    main()
