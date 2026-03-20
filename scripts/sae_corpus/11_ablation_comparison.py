#!/usr/bin/env python3
"""Compare task-aware vs per-dataset SAEs via per-row cumulative ablation.

Uses pre-computed per-row importance from 09 as both the firing indicator
and the ranking. For each row, cumulatively ablates features in importance
order until TabPFN's prediction matches Mitra's.

Uses the existing Phase 3 sweep from perrow_sweep_intervene_tabpfn:
one forward pass per k value, with per-row feature masks.

Usage:
    python scripts/sae_corpus/11_ablation_comparison.py --device cuda
"""
import argparse
import json
import logging
import sys
import time
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
    compute_ablation_delta,
    compute_ablation_delta_perrow,
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
    """Load per-row importance from 09. Returns (row_drops, feat_indices)."""
    path = OUTPUT_DIR / f"{prefix}_{dataset}.npz"
    data = np.load(str(path), allow_pickle=True)
    return data["row_feature_drops"], data["feature_indices"]


def build_rankings(row_drops, feat_indices):
    """Build per-row rankings from importance matrix.

    For each row, features with importance > 0 are ranked descending.
    Returns list of lists of (column_index_in_feat_indices, feat_idx).
    """
    n_query = row_drops.shape[0]
    rankings = []
    for row in range(n_query):
        drops = row_drops[row]
        # Positive importance = feature fires and helps
        positive = [(i, int(feat_indices[i]), drops[i])
                    for i in range(len(drops)) if drops[i] > 0]
        positive.sort(key=lambda x: -x[2])
        rankings.append([feat_idx for _, feat_idx, _ in positive])
    return rankings


def pred_scalar(preds):
    if isinstance(preds, np.ndarray) and preds.ndim == 2:
        return preds[:, 1] if preds.shape[1] == 2 else preds[:, 0]
    return np.asarray(preds).ravel()


def run_sweep(
    sae, data_mean, data_std, extraction_layer,
    rankings, X_ctx, y_ctx, X_q, y_q,
    strong_preds, weak_preds, task, device,
):
    """Phase 3 sweep: one forward pass per k, per-row feature masks."""
    from models.tabpfn_utils import load_tabpfn

    n_query = len(X_q)
    sp1 = pred_scalar(strong_preds)
    wp1 = pred_scalar(weak_preds)
    sae_hidden = sae.config.hidden_dim

    max_k = max((len(r) for r in rankings), default=0)
    logger.info("  Max features/row: %d", max_k)

    # Fit TabPFN and capture hidden state
    clf = load_tabpfn(task=task, device=device, n_estimators=1)
    fit_kwargs = {}
    tabpfn_data = load_preprocessed("tabpfn", "dummy", CACHE_DIR) if False else None
    clf.fit(X_ctx, y_ctx)

    layers = clf.model_.transformer_encoder.layers
    captured = {}

    def capture_hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            captured["hidden"] = out.detach()

    handle = layers[extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            baseline_check = clf.predict(X_q) if task == "regression" \
                else clf.predict_proba(X_q)
    finally:
        handle.remove()

    all_emb = captured["hidden"]
    if all_emb.ndim == 4:
        all_emb = all_emb[0].mean(dim=1)
    elif all_emb.ndim == 3:
        all_emb = all_emb[0] if all_emb.shape[0] == 1 else all_emb.mean(dim=0)

    query_emb = all_emb[-n_query:]
    ctx_emb = all_emb[:-n_query]

    # Sweep k=1..max_k
    best_p1 = sp1.copy()
    best_k = np.zeros(n_query, dtype=int)
    best_dist = np.abs(sp1 - wp1)

    logger.info("  Sweeping k=1..%d (%d forward passes)...", max_k, max_k)
    t0 = time.time()

    for k in range(1, max_k + 1):
        # Build per-row masks: each row ablates its top-k features
        masks = torch.zeros(n_query, sae_hidden, dtype=torch.bool, device=device)
        any_active = False
        for row_idx in range(n_query):
            feats = rankings[row_idx][:k]
            if feats:
                masks[row_idx, feats] = True
                any_active = True

        if not any_active:
            break

        # Context: ablate union of all features at this k
        all_feats_k = set()
        for r in rankings:
            all_feats_k.update(r[:k])
        ctx_delta = compute_ablation_delta(
            sae, ctx_emb, list(all_feats_k),
            data_mean=data_mean, data_std=data_std,
        )

        # Query: per-row ablation
        query_delta = compute_ablation_delta_perrow(
            sae, query_emb, masks,
            data_mean=data_mean, data_std=data_std,
        )

        combined = torch.cat([ctx_delta, query_delta], dim=0)
        if combined.ndim == 2:
            combined = combined.unsqueeze(1)  # (seq, 1, d) for 4D hook

        def make_hook(d):
            def hook(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                if isinstance(out, torch.Tensor):
                    out = out.clone()
                    if out.ndim == 4:
                        out[0] += d
                    elif out.ndim == 3:
                        out[0] += d.squeeze(1) if d.ndim == 3 else d
                    if isinstance(output, tuple):
                        return (out,) + output[1:]
                    return out
                return output
            return hook

        h = layers[extraction_layer].register_forward_hook(make_hook(combined))
        try:
            with torch.no_grad():
                preds = clf.predict(X_q) if task == "regression" \
                    else clf.predict_proba(X_q)
        finally:
            h.remove()

        preds_p1 = pred_scalar(np.asarray(preds))
        dist = np.abs(preds_p1 - wp1)

        # Update best per row
        improved = dist < best_dist
        best_p1[improved] = preds_p1[improved]
        best_k[improved] = k
        best_dist[improved] = dist[improved]

        if k % 10 == 0 or k == max_k:
            elapsed = time.time() - t0
            mean_dist = best_dist.mean()
            logger.info("    k=%d: mean_dist=%.4f, improved=%d rows (%.1fs)",
                        k, mean_dist, improved.sum(), elapsed)

    logger.info("  Done: mean k=%.1f, median k=%.0f, max k=%d",
                best_k.mean(), np.median(best_k), best_k.max())

    return best_p1, best_k, sp1, wp1


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

        # Load preprocessed data
        tabpfn_data = load_preprocessed("tabpfn", ds_name, CACHE_DIR)
        X_ctx, y_ctx = tabpfn_data.X_train[:600], tabpfn_data.y_train[:600]
        X_q, y_q = tabpfn_data.X_test[:500], tabpfn_data.y_test[:500]

        # Mitra predictions (weak model)
        logger.info("  Getting Mitra predictions...")
        mitra_data = load_preprocessed("mitra", ds_name, CACHE_DIR)
        mitra_clf = load_and_fit("mitra", mitra_data.X_train[:600], mitra_data.y_train[:600],
                                  task=task, device=args.device)
        weak_preds = predict(mitra_clf, mitra_data.X_test[:500], task=task)
        del mitra_clf
        torch.cuda.empty_cache()

        # TabPFN predictions (strong model)
        fit_kwargs = {}
        if tabpfn_data.cat_indices:
            fit_kwargs["cat_indices"] = tabpfn_data.cat_indices
        strong_clf = load_and_fit("tabpfn", X_ctx, y_ctx,
                                   task=task, device=args.device, **fit_kwargs)
        strong_preds = predict(strong_clf, X_q, task=task)
        del strong_clf
        torch.cuda.empty_cache()

        ds_results = {}

        for var_name, var_info in VARIANTS.items():
            print(f"\n  --- {var_name} ---")

            sae, config = load_sae(var_info["sae_path"], args.device)
            data_mean, data_std, layer = load_norm_stats(
                var_info["stats_path"], ds_name, args.device)

            row_drops, feat_indices = load_importance(
                var_info["importance_prefix"], ds_name)
            rankings = build_rankings(row_drops, feat_indices)

            n_firing = sum(1 for r in rankings if len(r) > 0)
            mean_k = np.mean([len(r) for r in rankings if len(r) > 0]) if n_firing else 0
            logger.info("  Layer: L%d, %d/%d rows have firing features, "
                        "mean %.1f features/row",
                        layer, n_firing, len(rankings), mean_k)

            t0 = time.time()
            final_p1, best_k, sp1, wp1 = run_sweep(
                sae, data_mean, data_std, layer, rankings,
                X_ctx, y_ctx, X_q, y_q,
                strong_preds, weak_preds, task, args.device,
            )
            elapsed = time.time() - t0

            gap_before = float(np.sqrt(np.mean((sp1 - wp1) ** 2)))
            gap_after = float(np.sqrt(np.mean((final_p1 - wp1) ** 2)))
            corr = float(np.corrcoef(final_p1, wp1)[0, 1]) if len(np.unique(wp1)) > 1 else 0.0

            print(f"  Gap before: {gap_before:.4f}")
            print(f"  Gap after:  {gap_after:.4f}")
            print(f"  Closed:     {(1 - gap_after/gap_before)*100:.1f}%")
            print(f"  Corr:       {corr:.4f}")
            print(f"  Time:       {elapsed:.1f}s")

            ds_results[var_name] = {
                "gap_before": gap_before, "gap_after": gap_after,
                "gap_closed_pct": float((1 - gap_after/gap_before) * 100),
                "correlation": corr,
                "concepts_mean": float(best_k[best_k > 0].mean()) if (best_k > 0).any() else 0,
                "concepts_max": int(best_k.max()),
                "extraction_layer": layer, "elapsed_s": elapsed,
                "strong_p1": sp1.tolist(), "weak_p1": wp1.tolist(),
                "ablated_p1": final_p1.tolist(),
            }

        all_results[ds_name] = ds_results

    # Save
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

    fig.suptitle("Per-row ablation: TabPFN → Mitra", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "ablation_comparison_scatter.png", dpi=150, bbox_inches="tight")
    print(f"→ {OUTPUT_DIR / 'ablation_comparison_scatter.png'}")


if __name__ == "__main__":
    main()
