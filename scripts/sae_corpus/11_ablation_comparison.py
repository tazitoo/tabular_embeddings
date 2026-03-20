#!/usr/bin/env python3
"""Compare task-aware vs per-dataset SAEs via per-row cumulative ablation.

Uses causal per-row importance from 09_perrow_importance.py.
Fits TabPFN once, builds tail once.  For each query row, cumulatively
ablates SAE features (query only, context untouched) until TabPFN's
prediction matches Mitra's.

Output:
    output/sae_training_round10/ablation_comparison_results.json
    output/sae_training_round10/ablation_comparison_scatter.png

Usage:
    python scripts/sae_corpus/11_ablation_comparison.py --device cuda
    python scripts/sae_corpus/11_ablation_comparison.py --device cuda --datasets airfoil_self_noise
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
from scripts.intervention.intervene_sae import TabPFNTail

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
    },
    "per_dataset": {
        "sae_path": OUTPUT_DIR / "tabpfn_perds_sae.pt",
        "stats_path": OUTPUT_DIR / "tabpfn_perds_norm_stats.npz",
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


def pred_scalar(preds):
    preds = np.asarray(preds)
    if preds.ndim == 2:
        return preds[:, 1] if preds.shape[1] == 2 else preds[:, 0]
    return preds.ravel()


def run_ablation(
    sae, data_mean, data_std, extraction_layer,
    importance_path, X_ctx, y_ctx, X_q, y_q,
    weak_p1, task, device,
):
    """Per-row cumulative ablation. One fit, one tail, query-only deltas."""
    n_query = len(X_q)
    converge_threshold = 1e-3

    # Load importance
    imp = np.load(str(importance_path), allow_pickle=True)
    row_drops = imp["row_feature_drops"]
    feat_indices = imp["feature_indices"]
    baseline_preds = imp["baseline_preds"]
    sp1 = pred_scalar(baseline_preds)

    # Build tail once — all query rows
    logger.info("  Building tail at L%d with %d query rows...", extraction_layer, n_query)
    tail = TabPFNTail.from_data(X_ctx, y_ctx, X_q, extraction_layer, task, device)

    # Get mean-pooled hidden state for SAE encoding
    # tail.hidden_state is (1, seq, n_structure, H)
    full_state = tail.hidden_state[0].mean(dim=1)  # (seq, H)
    ctx_len = tail.single_eval_pos

    final_p1 = sp1.copy()
    n_concepts = np.zeros(n_query, dtype=int)

    # Identify fixable rows up front
    fixable = np.abs(sp1 - weak_p1) > converge_threshold
    n_fixable = fixable.sum()
    logger.info("  Fixable rows: %d / %d", n_fixable, n_query)

    t0 = time.time()
    for row_idx in range(n_query):
        if not fixable[row_idx]:
            continue

        target = weak_p1[row_idx]
        best_dist = abs(sp1[row_idx] - target)

        # Per-row ranking: firing features sorted by importance
        drops = row_drops[row_idx]
        candidates = [(int(feat_indices[i]), drops[i])
                      for i in range(len(drops)) if drops[i] > 0]
        candidates.sort(key=lambda x: -x[1])
        if not candidates:
            continue

        ranking = [fi for fi, _ in candidates]
        K = len(ranking)

        # SAE encode this row's query embedding only
        query_emb = full_state[ctx_len + row_idx:ctx_len + row_idx + 1]  # (1, H)
        with torch.no_grad():
            x_norm = (query_emb - data_mean) / data_std
            h_q = sae.encode(x_norm)   # (1, hidden_dim)
            recon_q = sae.decode(h_q)  # (1, H)

            # K cumulative ablations → K deltas
            for k in range(K):
                h_abl = h_q.clone()
                for j in range(k + 1):
                    h_abl[:, ranking[j]] = 0.0
                recon_abl = sae.decode(h_abl)
                delta_row = ((recon_abl - recon_q) * data_std)[0]  # (H,)

                # predict_row modifies only this query position
                preds = tail.predict_row(row_idx, delta_row)
                new_p1 = pred_scalar(preds)[row_idx]
                new_dist = abs(new_p1 - target)

                if new_dist < best_dist:
                    best_dist = new_dist
                    final_p1[row_idx] = new_p1
                    n_concepts[row_idx] = k + 1

                if best_dist < converge_threshold:
                    break

        if (row_idx + 1) % 50 == 0 or row_idx == n_query - 1:
            elapsed = time.time() - t0
            rate = (row_idx + 1) / elapsed
            eta = (n_query - row_idx - 1) / rate
            logger.info("    row %d/%d: %d concepts, dist %.4f → %.4f "
                        "(%.1f rows/s, ETA %.0fs)",
                        row_idx + 1, n_query,
                        n_concepts[row_idx],
                        abs(sp1[row_idx] - target), best_dist,
                        rate, eta)

    elapsed = time.time() - t0
    active = n_concepts > 0
    n_fixed = (active & fixable).sum()
    n_unfixed = (fixable & ~active).sum()
    logger.info("  Done in %.1fs", elapsed)
    logger.info("  Fixable: %d, Fixed: %d (%.0f%%), Unfixed: %d",
                n_fixable, n_fixed, 100 * n_fixed / max(n_fixable, 1), n_unfixed)
    if active.any():
        logger.info("  Concepts: mean %.1f, median %.0f, max %d",
                    n_concepts[active].mean(),
                    np.median(n_concepts[active]),
                    n_concepts.max())

    del tail
    torch.cuda.empty_cache()
    return final_p1, n_concepts, sp1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()

    datasets = {k: v for k, v in EVAL_DATASETS.items()
                if args.datasets is None or k in args.datasets}

    all_results = {}

    for ds_name, ds_info in datasets.items():
        task = ds_info["task"]
        print(f"\n{'=' * 70}")
        print(f"  {ds_name} ({task})")
        print("=" * 70)

        # Mitra predictions (weak model)
        logger.info("  Getting Mitra predictions...")
        mitra_data = load_preprocessed("mitra", ds_name, CACHE_DIR)
        mitra_clf = load_and_fit("mitra", mitra_data.X_train[:600], mitra_data.y_train[:600],
                                  task=task, device=args.device)
        weak_preds = predict(mitra_clf, mitra_data.X_test[:500], task=task)
        wp1 = pred_scalar(weak_preds)
        del mitra_clf
        torch.cuda.empty_cache()

        tabpfn_data = load_preprocessed("tabpfn", ds_name, CACHE_DIR)
        X_ctx, y_ctx = tabpfn_data.X_train[:600], tabpfn_data.y_train[:600]
        X_q, y_q = tabpfn_data.X_test[:500], tabpfn_data.y_test[:500]

        ds_results = {}

        for var_name, var_info in VARIANTS.items():
            print(f"\n  --- {var_name} ---")

            sae, config = load_sae(var_info["sae_path"], args.device)
            data_mean, data_std, layer = load_norm_stats(
                var_info["stats_path"], ds_name, args.device)

            imp_path = OUTPUT_DIR / f"perrow_importance_{var_name}_{ds_name}.npz"
            if not imp_path.exists():
                logger.info("  SKIP: no importance file")
                continue

            logger.info("  Layer: L%d", layer)

            final_p1, n_concepts, sp1 = run_ablation(
                sae, data_mean, data_std, layer, imp_path,
                X_ctx, y_ctx, X_q, y_q, wp1, task, args.device,
            )

            gap_before = float(np.sqrt(np.mean((sp1 - wp1) ** 2)))
            gap_after = float(np.sqrt(np.mean((final_p1 - wp1) ** 2)))
            corr = float(np.corrcoef(final_p1, wp1)[0, 1]) if len(np.unique(wp1)) > 1 else 0.0

            print(f"  Gap before: {gap_before:.4f}")
            print(f"  Gap after:  {gap_after:.4f}")
            print(f"  Closed:     {(1 - gap_after/gap_before)*100:.1f}%")
            print(f"  Corr:       {corr:.4f}")

            active = n_concepts > 0
            ds_results[var_name] = {
                "gap_before": gap_before, "gap_after": gap_after,
                "gap_closed_pct": float((1 - gap_after/gap_before) * 100),
                "correlation": corr,
                "concepts_mean": float(n_concepts[active].mean()) if active.any() else 0,
                "concepts_max": int(n_concepts.max()),
                "extraction_layer": layer,
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
                ax.set_visible(False)
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
                         f"r={r['correlation']:.3f}, "
                         f"concepts: {r['concepts_mean']:.1f}",
                         fontsize=10)

    fig.suptitle("Per-row cumulative ablation: TabPFN → Mitra\n"
                 "(query-only, context untouched)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "ablation_comparison_scatter.png", dpi=150, bbox_inches="tight")
    print(f"→ {OUTPUT_DIR / 'ablation_comparison_scatter.png'}")


if __name__ == "__main__":
    main()
