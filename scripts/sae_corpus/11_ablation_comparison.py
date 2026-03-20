#!/usr/bin/env python3
"""Compare task-aware vs per-dataset SAEs via per-row cumulative ablation.

For each query row individually:
  1. Get TabPFN baseline prediction
  2. Cumulatively ablate SAE features in per-row importance order
  3. Stop when prediction matches Mitra's prediction

One query row per forward pass — no masks, no approximations.

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
from scripts.intervention.intervene_sae import compute_ablation_delta

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
    """Load per-row importance from 09."""
    path = OUTPUT_DIR / f"{prefix}_{dataset}.npz"
    data = np.load(str(path), allow_pickle=True)
    return data["row_feature_drops"], data["feature_indices"]


def build_row_ranking(row_drops, feat_indices, row_idx):
    """Build ranking for one row: features sorted by importance descending.

    Only includes features with positive importance (ablation hurts this row).
    """
    drops = row_drops[row_idx]
    candidates = [(int(feat_indices[i]), drops[i]) for i in range(len(drops)) if drops[i] > 0]
    candidates.sort(key=lambda x: -x[1])
    return [fi for fi, _ in candidates]


def pred_scalar(preds):
    if isinstance(preds, np.ndarray) and preds.ndim == 2:
        return float(preds[0, 1]) if preds.shape[1] == 2 else float(preds[0, 0])
    return float(np.asarray(preds).ravel()[0])


def run_ablation(
    sae, data_mean, data_std, extraction_layer,
    row_drops, feat_indices,
    X_ctx, y_ctx, X_q, y_q,
    weak_p1_all, task, device,
):
    """Per-row ablation: fit once, single query row per forward pass."""
    from models.tabpfn_utils import load_tabpfn

    n_query = len(X_q)
    converge_threshold = 1e-3

    final_p1 = np.zeros(n_query)
    strong_p1 = np.zeros(n_query)
    n_concepts = np.zeros(n_query, dtype=int)

    # Fit TabPFN once
    clf = load_tabpfn(task=task, device=device, n_estimators=1)
    clf.fit(X_ctx, y_ctx)
    layers = clf.model_.transformer_encoder.layers

    t0 = time.time()

    for row_idx in range(n_query):
        x_row = X_q[row_idx:row_idx + 1]
        target = weak_p1_all[row_idx]

        # Capture hidden state + baseline prediction for this single row
        captured = {}

        def capture_hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            if isinstance(out, torch.Tensor):
                captured["hidden"] = out.detach()

        handle = layers[extraction_layer].register_forward_hook(capture_hook)
        try:
            with torch.no_grad():
                if task == "regression":
                    baseline = clf.predict(x_row)
                else:
                    baseline = clf.predict_proba(x_row)
        finally:
            handle.remove()

        all_emb = captured["hidden"]
        if all_emb.ndim == 4:
            all_emb = all_emb[0].mean(dim=1)
        elif all_emb.ndim == 3:
            all_emb = all_emb[0] if all_emb.shape[0] == 1 else all_emb.mean(dim=0)

        baseline_p1 = pred_scalar(baseline)
        strong_p1[row_idx] = baseline_p1
        final_p1[row_idx] = baseline_p1
        best_dist = abs(baseline_p1 - target)

        if best_dist < converge_threshold:
            continue

        # Get ranking for this row
        ranking = build_row_ranking(row_drops, feat_indices, row_idx)
        if not ranking:
            continue

        # Cumulatively ablate features
        ablated_so_far = []
        for feat_idx in ranking:
            ablated_so_far.append(feat_idx)

            delta = compute_ablation_delta(
                sae, all_emb, ablated_so_far,
                data_mean=data_mean, data_std=data_std,
            )
            if delta.ndim == 2:
                delta = delta.unsqueeze(1)

            def make_hook(d):
                def hook(module, input, output):
                    out = output[0] if isinstance(output, tuple) else output
                    if isinstance(out, torch.Tensor):
                        out = out.clone()
                        if out.ndim == 4:
                            out[0] += d
                        elif out.ndim == 3:
                            out[0] += d.squeeze(1)
                        if isinstance(output, tuple):
                            return (out,) + output[1:]
                        return out
                    return output
                return hook

            handle = layers[extraction_layer].register_forward_hook(make_hook(delta))
            try:
                with torch.no_grad():
                    if task == "regression":
                        preds = clf.predict(x_row)
                    else:
                        preds = clf.predict_proba(x_row)
            finally:
                handle.remove()

            new_p1 = pred_scalar(preds)
            new_dist = abs(new_p1 - target)

            if new_dist < best_dist:
                best_dist = new_dist
                final_p1[row_idx] = new_p1
                n_concepts[row_idx] = len(ablated_so_far)

            if best_dist < converge_threshold:
                break

        if (row_idx + 1) % 50 == 0 or row_idx == n_query - 1:
            elapsed = time.time() - t0
            rate = (row_idx + 1) / elapsed
            eta = (n_query - row_idx - 1) / rate
            logger.info("    row %d/%d: %d concepts, dist %.4f → %.4f "
                        "(%.1f rows/s, ETA %.0fs)",
                        row_idx + 1, n_query,
                        n_concepts[row_idx], abs(strong_p1[row_idx] - target),
                        best_dist, rate, eta)

    del clf
    torch.cuda.empty_cache()

    active = n_concepts > 0
    logger.info("  Done: mean %.1f, median %.0f, max %d concepts/row",
                n_concepts[active].mean() if active.any() else 0,
                np.median(n_concepts[active]) if active.any() else 0,
                n_concepts.max())

    return final_p1, n_concepts, strong_p1


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

        # Weak model p1
        if isinstance(weak_preds, np.ndarray) and weak_preds.ndim == 2:
            wp1 = weak_preds[:, 1] if weak_preds.shape[1] == 2 else weak_preds[:, 0]
        else:
            wp1 = np.asarray(weak_preds).ravel()

        ds_results = {}

        for var_name, var_info in VARIANTS.items():
            print(f"\n  --- {var_name} ---")

            sae, config = load_sae(var_info["sae_path"], args.device)
            data_mean, data_std, layer = load_norm_stats(
                var_info["stats_path"], ds_name, args.device)
            row_drops, feat_indices = load_importance(
                var_info["importance_prefix"], ds_name)

            logger.info("  Layer: L%d", layer)

            t0 = time.time()
            final_p1, n_concepts, sp1 = run_ablation(
                sae, data_mean, data_std, layer,
                row_drops, feat_indices,
                X_ctx, y_ctx, X_q, y_q,
                wp1, task, args.device,
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
                "concepts_mean": float(n_concepts[n_concepts > 0].mean()) if (n_concepts > 0).any() else 0,
                "concepts_max": int(n_concepts.max()),
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
