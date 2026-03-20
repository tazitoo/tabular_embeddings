#!/usr/bin/env python3
"""Compare task-aware vs per-dataset SAEs via per-row ablation.

For each SAE variant, runs sweep_ablation from transfer_concepts.py:
  1. TabPFN (strong) predictions on airfoil + diabetes
  2. Mitra (weak) predictions on same samples
  3. Per-row ablation of TabPFN SAE features to match Mitra's predictions
  4. Scatter plot: ablated TabPFN preds vs Mitra preds (should be on y=x)

Success metric: how well do ablated predictions match the weaker model?

Usage:
    python scripts/sae_corpus/11_ablation_comparison.py --device cuda
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.sparse_autoencoder import SAEConfig, SparseAutoencoder
from scripts._project_root import PROJECT_ROOT

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


def load_custom_sae(sae_path: Path, device: str):
    """Load SAE from our round 10 checkpoint format."""
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


def load_custom_norm_stats(stats_path: Path, dataset: str, device: str):
    """Load norm stats from round 10 format. Returns (mean, std) as tensors."""
    stats = np.load(str(stats_path), allow_pickle=True)
    datasets = list(stats["datasets"])
    if dataset not in datasets:
        raise ValueError(f"{dataset} not in norm stats ({len(datasets)} datasets)")
    idx = datasets.index(dataset)
    mean = torch.tensor(stats["means"][idx], dtype=torch.float32, device=device)
    std = torch.tensor(stats["stds"][idx], dtype=torch.float32, device=device)
    layer = int(stats["layers"][idx]) if "layers" in stats else None
    return mean, std, layer


def main():
    parser = argparse.ArgumentParser(
        description="Per-row ablation comparison: task-aware vs per-dataset SAEs"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Import after path setup
    import scripts.intervention.intervene_sae as _isae
    from scripts.intervention.transfer_concepts import sweep_ablation

    all_results = {}

    for ds_name, ds_info in EVAL_DATASETS.items():
        task = ds_info["task"]
        print(f"\n{'=' * 70}")
        print(f"  {ds_name} ({task})")
        print("=" * 70)

        ds_results = {}

        for var_name, var_info in VARIANTS.items():
            print(f"\n  --- {var_name} ---")

            # Load our custom SAE and norm stats
            sae, config = load_custom_sae(var_info["sae_path"], args.device)
            data_mean, data_std, extraction_layer = load_custom_norm_stats(
                var_info["stats_path"], ds_name, args.device
            )
            print(f"  SAE: {config.input_dim}→{config.hidden_dim}, "
                  f"extraction layer: L{extraction_layer}")

            # Monkey-patch the module functions so sweep_ablation uses our SAE
            def make_load_sae(s, c):
                def _load_sae(model_key, sae_dir=None, device="cuda"):
                    return s, c
                return _load_sae

            def make_load_norm_stats(m, s):
                def _load_norm_stats(model_key, dataset_name, training_dir=None,
                                     layers_path=None, device="cuda"):
                    return m, s
                return _load_norm_stats

            def make_get_extraction_layer(layer):
                def _get_extraction_layer(model_key, layers_path=None):
                    if model_key == "tabpfn":
                        return layer
                    # For weak model (mitra), use default
                    from config import get_optimal_layer
                    return get_optimal_layer(model_key)
                return _get_extraction_layer

            # Get all alive features as ranked list (by index, no importance pre-ranking)
            # sweep_ablation will compute per-row importance internally
            alive_idx = []
            for i in range(config.hidden_dim):
                alive_idx.append((i, 0.0))  # (feature_idx, placeholder_importance)

            t0 = time.time()
            with patch.object(_isae, 'load_sae', make_load_sae(sae, config)), \
                 patch.object(_isae, 'load_norm_stats', make_load_norm_stats(data_mean, data_std)), \
                 patch.object(_isae, 'get_extraction_layer', make_get_extraction_layer(extraction_layer)):

                result = sweep_ablation(
                    strong_model="tabpfn",
                    weak_model="mitra",
                    dataset=ds_name,
                    ranked_features=alive_idx,
                    device=args.device,
                    task=task,
                )
            elapsed = time.time() - t0

            # Extract key metrics
            strong_p = result["strong_preds"]
            weak_p = result["weak_preds"]
            ablated_p = result["accepted_preds"]
            y_q = result["y_query"]
            accepted = result["accepted_counts"]

            if task == "classification":
                # Use probability of class 1
                if strong_p.ndim == 2:
                    strong_p1 = strong_p[:, 1]
                else:
                    strong_p1 = strong_p
                if weak_p.ndim == 2:
                    weak_p1 = weak_p[:, 1]
                else:
                    weak_p1 = weak_p
                if ablated_p.ndim == 2:
                    ablated_p1 = ablated_p[:, 1]
                else:
                    ablated_p1 = ablated_p
            else:
                strong_p1 = strong_p.ravel()
                weak_p1 = weak_p.ravel()
                ablated_p1 = ablated_p.ravel()

            # Correlation and RMSE between ablated and weak
            corr = float(np.corrcoef(ablated_p1, weak_p1)[0, 1])
            rmse = float(np.sqrt(np.mean((ablated_p1 - weak_p1) ** 2)))
            gap_before = float(np.sqrt(np.mean((strong_p1 - weak_p1) ** 2)))

            print(f"  Elapsed: {elapsed:.1f}s")
            print(f"  Gap before ablation (RMSE strong-weak): {gap_before:.4f}")
            print(f"  Gap after ablation (RMSE ablated-weak): {rmse:.4f}")
            print(f"  Gap closed: {(1 - rmse/gap_before)*100:.1f}%")
            print(f"  Correlation (ablated vs weak): {corr:.4f}")
            print(f"  Concepts used: mean={accepted.mean():.1f}, "
                  f"median={np.median(accepted):.0f}, max={accepted.max()}")

            ds_results[var_name] = {
                "gap_before": gap_before,
                "gap_after": rmse,
                "gap_closed_pct": float((1 - rmse/gap_before) * 100),
                "correlation": corr,
                "concepts_mean": float(accepted.mean()),
                "concepts_median": float(np.median(accepted)),
                "concepts_max": int(accepted.max()),
                "extraction_layer": extraction_layer,
                "elapsed_s": elapsed,
                # Save predictions for scatter plot
                "strong_preds": strong_p1.tolist(),
                "weak_preds": weak_p1.tolist(),
                "ablated_preds": ablated_p1.tolist(),
            }

        all_results[ds_name] = ds_results

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<25} {'Variant':<15} {'Layer':>5} {'Gap%':>8} {'Corr':>8} {'Concepts':>10}")
    print("-" * 73)
    for ds_name in EVAL_DATASETS:
        for var_name in VARIANTS:
            r = all_results.get(ds_name, {}).get(var_name, {})
            if r:
                print(f"{ds_name:<25} {var_name:<15} L{r['extraction_layer']:>3} "
                      f"{r['gap_closed_pct']:>7.1f}% {r['correlation']:>8.4f} "
                      f"{r['concepts_mean']:>10.1f}")

    # Save
    out_path = OUTPUT_DIR / "ablation_comparison_results.json"
    with open(str(out_path), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n→ {out_path}")

    # Scatter plots
    plot_scatters(all_results)


def plot_scatters(all_results):
    """Generate scatter plots: ablated vs weak predictions."""
    import matplotlib.pyplot as plt

    n_datasets = len(all_results)
    fig, axes = plt.subplots(n_datasets, 2, figsize=(12, 5 * n_datasets))
    if n_datasets == 1:
        axes = axes[np.newaxis, :]

    var_names = list(VARIANTS.keys())
    colors = {"task_aware": "#4C72B0", "per_dataset": "#DD8452"}

    for row, (ds_name, ds_results) in enumerate(all_results.items()):
        for col, var_name in enumerate(var_names):
            ax = axes[row, col]
            r = ds_results.get(var_name, {})
            if not r:
                ax.set_visible(False)
                continue

            weak = np.array(r["weak_preds"])
            ablated = np.array(r["ablated_preds"])

            ax.scatter(weak, ablated, s=8, alpha=0.5, color=colors[var_name])

            # y=x line
            lo = min(weak.min(), ablated.min())
            hi = max(weak.max(), ablated.max())
            margin = (hi - lo) * 0.05
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    "k--", linewidth=1, alpha=0.5)

            ax.set_xlabel("Mitra prediction")
            ax.set_ylabel("Ablated TabPFN prediction")
            ax.set_title(
                f"{ds_name} — {var_name} (L{r['extraction_layer']})\n"
                f"gap closed: {r['gap_closed_pct']:.1f}%, "
                f"r={r['correlation']:.3f}, "
                f"concepts: {r['concepts_mean']:.1f}",
                fontsize=10,
            )
            ax.set_aspect("equal")

    fig.suptitle(
        "Per-row ablation: TabPFN → Mitra\n"
        "Ablated TabPFN predictions vs Mitra baseline",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    out_path = OUTPUT_DIR / "ablation_comparison_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
