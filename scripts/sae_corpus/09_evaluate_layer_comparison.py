#!/usr/bin/env python3
"""Evaluate task-aware vs per-dataset layer SAEs via per-row importance.

Uses the existing concept_importance.sweep_tabpfn() pipeline to run
single-feature ablation through the full model (not just reconstruction),
comparing the two SAE variants on:

  - airfoil_self_noise (CKA critical layer L6, regression)
  - polish_companies_bankruptcy (CKA critical layer L23, classification)

For each (SAE variant, dataset), runs the full sweep_tabpfn flow:
  1. Fit TabPFN on context
  2. Capture hidden state at extraction layer
  3. Compute baseline predictions
  4. For each alive feature: ablate → inject delta → measure per-row loss change

Output:
    output/sae_training_round10/layer_comparison_eval.json
    output/sae_training_round10/layer_comparison_eval_{variant}_{dataset}.npz

Usage:
    python scripts/sae_corpus/09_evaluate_layer_comparison.py --device cuda
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.sparse_autoencoder import SAEConfig, SparseAutoencoder
from data.preprocessing import CACHE_DIR, load_preprocessed
from scripts._project_root import PROJECT_ROOT
from scripts.intervention.concept_importance import sweep_tabpfn, compute_importance_metric

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_training_round10"
SPLITS_PATH = PROJECT_ROOT / "output" / "sae_training_round9" / "tabarena_splits.json"

# Test datasets: chosen for maximum contrast with the task-aware fixed layers
EVAL_DATASETS = {
    "airfoil_self_noise": {"critical_layer": 6, "task": "regression"},
    "polish_companies_bankruptcy": {"critical_layer": 23, "task": "classification"},
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


def load_sae(path: Path, device: str) -> SparseAutoencoder:
    """Load trained SAE from checkpoint."""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    config = SAEConfig(**ckpt["config"])
    sae = SparseAutoencoder(config)

    state = ckpt["state_dict"]
    if "reference_data" in state and state["reference_data"] is not None:
        sae.reference_data = state["reference_data"].clone()
        if "archetype_logits" in state:
            sae.archetype_logits = torch.nn.Parameter(state["archetype_logits"].clone())
        if "archetype_deviation" in state:
            sae.archetype_deviation = torch.nn.Parameter(state["archetype_deviation"].clone())

    sae.load_state_dict(state, strict=False)
    sae.to(device)
    sae.eval()
    return sae


def load_norm_stats(stats_path: Path, dataset: str, device: str):
    """Load per-dataset norm stats. Returns (mean, std, layer) as tensors."""
    stats = np.load(str(stats_path), allow_pickle=True)
    datasets = list(stats["datasets"])
    if dataset not in datasets:
        return None, None, None
    idx = datasets.index(dataset)
    layer = int(stats["layers"][idx]) if "layers" in stats else None
    mean = torch.tensor(stats["means"][idx], dtype=torch.float32, device=device)
    std = torch.tensor(stats["stds"][idx], dtype=torch.float32, device=device)
    return mean, std, layer


def load_context_query(dataset: str, task: str, max_context: int = 600, max_query: int = 500):
    """Load context/query splits from preprocessed cache."""
    data = load_preprocessed("tabpfn", dataset, CACHE_DIR)
    X_ctx, y_ctx = data.X_train[:max_context], data.y_train[:max_context]
    X_q, y_q = data.X_test[:max_query], data.y_test[:max_query]
    return X_ctx, y_ctx, X_q, y_q


def get_alive_features(sae: SparseAutoencoder, X_ctx, y_ctx, X_q, extraction_layer,
                       data_mean, data_std, device, task):
    """Get alive feature indices by running a forward pass and checking activations."""
    from models.tabpfn_utils import load_tabpfn

    clf = load_tabpfn(task=task, device=device, n_estimators=1)
    clf.fit(X_ctx, y_ctx)

    model = clf.model_ if hasattr(clf, "model_") else clf.transformer_
    layers = model.transformer_encoder.layers

    captured = {}

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

    hidden = captured["hidden"]
    if hidden.ndim == 4:
        all_emb = hidden[0].mean(dim=1)
    elif hidden.ndim == 3:
        all_emb = hidden[0] if hidden.shape[0] == 1 else hidden.mean(dim=0)
    else:
        all_emb = hidden

    with torch.no_grad():
        x_norm = (all_emb - data_mean) / data_std
        h = sae.encode(x_norm)

    alive = (h > 0).any(dim=0).cpu().numpy()
    return sorted(np.where(alive)[0].tolist())


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate task-aware vs per-dataset SAEs via per-row importance"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    all_results = {}

    for ds_name, ds_info in EVAL_DATASETS.items():
        task = ds_info["task"]
        print(f"\n{'=' * 70}")
        print(f"  {ds_name} (critical L{ds_info['critical_layer']}, {task})")
        print("=" * 70)

        X_ctx, y_ctx, X_q, y_q = load_context_query(ds_name, task)
        print(f"  Context: {X_ctx.shape}  Query: {X_q.shape}")

        ds_results = {}

        for var_name, var_info in VARIANTS.items():
            print(f"\n  --- {var_name} ---")

            # Load SAE and norm stats
            sae = load_sae(var_info["sae_path"], args.device)
            data_mean, data_std, layer = load_norm_stats(
                var_info["stats_path"], ds_name, args.device
            )
            if data_mean is None:
                print(f"  SKIP: no norm stats for {ds_name}")
                continue

            print(f"  Extraction layer: L{layer}")

            # Get alive features
            alive = get_alive_features(
                sae, X_ctx, y_ctx, X_q, layer, data_mean, data_std,
                args.device, task,
            )
            print(f"  Alive features: {len(alive)}/{sae.config.hidden_dim}")

            # Run the full sweep
            t0 = time.time()
            result = sweep_tabpfn(
                X_context=X_ctx,
                y_context=y_ctx,
                X_query=X_q,
                y_query=y_q,
                sae=sae,
                alive_features=alive,
                extraction_layer=layer,
                device=args.device,
                task=task,
                data_mean=data_mean,
                data_std=data_std,
            )
            elapsed = time.time() - t0

            # Summarize
            row_drops = result["row_feature_drops"]  # (n_query, n_features)
            mean_drops = row_drops.mean(axis=0)
            order = np.argsort(-mean_drops)

            print(f"\n  Baseline {result['metric_name']}: {result['baseline_metric']:.4f}")
            print(f"  Sweep: {len(alive)} features in {elapsed:.1f}s")
            print(f"\n  Top {args.top} features (mean per-row loss drop):")
            print(f"  {'Rank':>4} {'Feat':>6} {'MeanDrop':>10} {'Fire':>8}")
            print(f"  {'-'*32}")
            for rank, idx in enumerate(order[:args.top]):
                feat_idx = result["feature_indices"][idx]
                print(f"  {rank+1:>4} {feat_idx:>6} {mean_drops[idx]:>+10.4f} "
                      f"{result['feature_n_firing'][idx]:>4}/{len(y_q)}")

            helpful = (mean_drops > 0).sum()
            print(f"\n  Helpful features (drop>0): {helpful}/{len(alive)}")
            print(f"  Mean importance: {mean_drops.mean():+.4f}")
            print(f"  Max importance:  {mean_drops.max():+.4f}")

            # Save per-row results
            npz_path = OUTPUT_DIR / f"layer_comparison_eval_{var_name}_{ds_name}.npz"
            np.savez_compressed(
                str(npz_path),
                row_feature_drops=row_drops,
                feature_indices=result["feature_indices"],
                feature_n_firing=result["feature_n_firing"],
                baseline_metric=np.array(result["baseline_metric"]),
                metric_name=np.array(result["metric_name"]),
                y_query=result["y_query"],
                extraction_layer=np.array(layer),
            )

            ds_results[var_name] = {
                "extraction_layer": layer,
                "baseline_metric": float(result["baseline_metric"]),
                "metric_name": result["metric_name"],
                "n_alive": len(alive),
                "n_helpful": int(helpful),
                "mean_importance": float(mean_drops.mean()),
                "max_importance": float(mean_drops.max()),
                "top_10_features": [int(result["feature_indices"][i]) for i in order[:10]],
                "top_10_importance": [float(mean_drops[i]) for i in order[:10]],
                "elapsed_s": elapsed,
            }

        # Compare top features
        if len(ds_results) == 2:
            ta = set(ds_results["task_aware"]["top_10_features"])
            pd = set(ds_results["per_dataset"]["top_10_features"])
            overlap = ta & pd
            print(f"\n  Top-10 overlap: {len(overlap)}/10  shared={sorted(overlap)}")

        all_results[ds_name] = ds_results

    # Save summary
    out_path = OUTPUT_DIR / "layer_comparison_eval.json"
    json.dump(all_results, open(str(out_path), "w"), indent=2)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
