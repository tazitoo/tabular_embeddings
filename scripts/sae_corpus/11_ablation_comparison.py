#!/usr/bin/env python3
"""Compare task-aware vs per-dataset SAEs via per-row cumulative ablation.

For each SAE variant:
  1. Load pre-computed per-row importance rankings (from 09)
  2. Get TabPFN (strong) and Mitra (weak) predictions
  3. For each row: cumulatively zero SAE features in rank order,
     decode + inject via tail model, stop when prediction matches weak model
  4. Scatter plot: ablated TabPFN vs Mitra (should be on y=x)

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
from scripts.intervention.intervene_sae import build_tail

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
    """Load pre-computed per-row importance from 09.

    Returns per-row rankings: list of lists, where rankings[row] is a list
    of feature indices sorted by importance for that row (most important first),
    filtered to features with positive importance on that row.
    """
    path = OUTPUT_DIR / f"{prefix}_{dataset}.npz"
    data = np.load(str(path), allow_pickle=True)
    row_drops = data["row_feature_drops"]  # (n_query, n_features)
    feat_indices = data["feature_indices"]  # (n_features,)

    n_query = row_drops.shape[0]
    per_row_rankings = []
    for row_idx in range(n_query):
        row = row_drops[row_idx]
        order = np.argsort(-row)
        ranked = [int(feat_indices[i]) for i in order if row[i] > 0]
        per_row_rankings.append(ranked)

    total_features = sum(len(r) for r in per_row_rankings)
    logger.info("  Per-row rankings: mean %.1f, max %d features/row",
                total_features / max(n_query, 1),
                max(len(r) for r in per_row_rankings))
    return per_row_rankings


def pred_scalar(preds):
    if isinstance(preds, np.ndarray) and preds.ndim == 2:
        return preds[:, 1] if preds.shape[1] == 2 else preds[:, 0]
    return np.asarray(preds).ravel()


def capture_hidden(clf, X_q, extraction_layer, task):
    """Run forward pass and capture hidden state at extraction layer."""
    model = clf.model_ if hasattr(clf, "model_") else clf.transformer_
    layers = model.transformer_encoder.layers
    captured = {}

    def hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            captured["hidden"] = out.detach()

    handle = layers[extraction_layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            if task == "regression":
                clf.predict(X_q)
            else:
                clf.predict_proba(X_q)
    finally:
        handle.remove()

    h = captured["hidden"]
    if h.ndim == 4:
        return h[0].mean(dim=1)  # (seq, H)
    elif h.ndim == 3:
        return h[0] if h.shape[0] == 1 else h.mean(dim=0)
    return h


def run_ablation(
    sae, data_mean, data_std, extraction_layer,
    per_row_rankings, X_ctx, y_ctx, X_q, y_q,
    strong_preds, weak_preds, task, device,
):
    """Per-row cumulative ablation: zero features in per-row rank order until gap closes."""
    n_query = len(X_q)
    sp1 = pred_scalar(strong_preds)
    wp1 = pred_scalar(weak_preds)
    converge_threshold = 1e-3

    # Build tail model
    logger.info("  Building TabPFN tail at L%d...", extraction_layer)
    tail = build_tail("tabpfn", X_ctx, y_ctx, X_q, extraction_layer, task, device)

    # Capture hidden state and encode through SAE
    from models.tabpfn_utils import load_tabpfn
    clf = load_tabpfn(task=task, device=device, n_estimators=1)
    clf.fit(X_ctx, y_ctx)

    all_emb = capture_hidden(clf, X_q, extraction_layer, task)
    del clf
    torch.cuda.empty_cache()

    query_emb = all_emb[-n_query:]

    with torch.no_grad():
        x_norm = (query_emb - data_mean) / data_std
        h_full = sae.encode(x_norm)
        original_recon = sae.decode(h_full)

    # Determine fixable rows (strong is better than weak)
    eps = 1e-7
    if task == "regression":
        strong_loss = (y_q - sp1) ** 2
        weak_loss = (y_q - wp1) ** 2
    else:
        y = y_q.astype(float)
        sp_clip = np.clip(sp1, eps, 1 - eps)
        wp_clip = np.clip(wp1, eps, 1 - eps)
        strong_loss = -(y * np.log(sp_clip) + (1 - y) * np.log(1 - sp_clip))
        weak_loss = -(y * np.log(wp_clip) + (1 - y) * np.log(1 - wp_clip))

    fixable = strong_loss < weak_loss
    n_fixable = int(fixable.sum())
    logger.info("  Fixable rows: %d / %d", n_fixable, n_query)

    # Per-row cumulative ablation
    final_p1 = sp1.copy()
    accepted_counts = np.zeros(n_query, dtype=int)

    t0 = time.time()

    for row_idx in range(n_query):
        if not fixable[row_idx]:
            continue

        ranking = per_row_rankings[row_idx]
        if not ranking:
            continue

        target_p1 = float(wp1[row_idx])
        best_dist = abs(float(sp1[row_idx]) - target_p1)

        if best_dist < converge_threshold:
            continue

        # Start with full encoding, cumulatively zero features in this row's rank order
        h_row = h_full[row_idx].clone()

        for k, feat_idx in enumerate(ranking):
            h_row[feat_idx] = 0.0

            # Decode and compute delta vs original
            with torch.no_grad():
                ablated_recon = sae.decode(h_row.unsqueeze(0))
                delta_norm = ablated_recon - original_recon[row_idx:row_idx+1]
                delta_raw = delta_norm * data_std

            preds = tail.predict_row(row_idx, delta_raw.squeeze(0))
            new_p1 = float(pred_scalar(preds)[row_idx])

            new_dist = abs(new_p1 - target_p1)
            if new_dist < best_dist:
                best_dist = new_dist
                final_p1[row_idx] = new_p1
                accepted_counts[row_idx] = k + 1

            if best_dist < converge_threshold:
                break

        if (row_idx + 1) % 100 == 0 or row_idx == n_query - 1:
            elapsed = time.time() - t0
            logger.info("    row %d/%d: %d/%d concepts, dist %.4f → %.4f (%.1fs)",
                        row_idx + 1, n_query,
                        int(accepted_counts[row_idx]), len(ranking),
                        abs(float(sp1[row_idx]) - float(wp1[row_idx])),
                        best_dist, elapsed)

    logger.info("  Done: mean %.1f, median %.0f, max %d concepts/row",
                accepted_counts[fixable].mean() if n_fixable > 0 else 0,
                np.median(accepted_counts[fixable]) if n_fixable > 0 else 0,
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

        # Load data for both models
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
            per_row_rankings = load_importance(var_info["importance_prefix"], ds_name)

            logger.info("  Layer: L%d", layer)

            t0 = time.time()
            final_p1, accepted, sp1, wp1 = run_ablation(
                sae, data_mean, data_std, layer, per_row_rankings,
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
                "concepts_mean": float(accepted[accepted > 0].mean()) if (accepted > 0).any() else 0,
                "concepts_max": int(accepted.max()),
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
