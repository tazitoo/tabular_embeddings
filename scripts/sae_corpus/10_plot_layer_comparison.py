#!/usr/bin/env python3
"""Plot task-aware vs per-dataset layer SAE comparison.

Reads the per-row importance NPZ files from 09_evaluate_layer_comparison.py
and produces a 2×2 figure:
  - Top row: feature importance distributions (sorted bar + histogram)
  - Bottom row: per-row importance heatmaps

Output:
    output/sae_training_round10/layer_comparison_plot.png

Usage:
    python scripts/sae_corpus/10_plot_layer_comparison.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._project_root import PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_training_round10"

DATASETS = ["airfoil_self_noise", "diabetes"]
DATASET_LABELS = {"airfoil_self_noise": "airfoil (reg, L6)", "diabetes": "diabetes (cls, L23)"}
VARIANTS = ["task_aware", "per_dataset"]
VARIANT_LABELS = {"task_aware": "Task-aware fixed", "per_dataset": "Per-dataset"}
VARIANT_COLORS = {"task_aware": "#4C72B0", "per_dataset": "#DD8452"}


def load_results(variant, dataset):
    path = OUTPUT_DIR / f"layer_comparison_eval_{variant}_{dataset}.npz"
    data = np.load(str(path), allow_pickle=True)
    return {
        "row_drops": data["row_feature_drops"],
        "feat_idx": data["feature_indices"],
        "n_firing": data["feature_n_firing"],
        "layer": int(data["extraction_layer"]),
    }


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, ds in enumerate(DATASETS):
        # Load both variants
        results = {}
        for var in VARIANTS:
            results[var] = load_results(var, ds)

        # Top row: sorted mean importance (top 50 features)
        ax = axes[0, col]
        top_n = 50
        for var in VARIANTS:
            r = results[var]
            mean_drops = r["row_drops"].mean(axis=0)
            sorted_drops = np.sort(mean_drops)[::-1][:top_n]
            ax.plot(range(top_n), sorted_drops,
                    color=VARIANT_COLORS[var], label=VARIANT_LABELS[var],
                    linewidth=2, alpha=0.8)
        ax.set_xlabel("Feature rank")
        ax.set_ylabel("Mean per-row loss drop")
        ax.set_title(f"{DATASET_LABELS[ds]}\nTop-{top_n} feature importance", fontweight="bold")
        ax.legend(fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

        # Bottom row: importance distribution (histogram of all features)
        ax = axes[1, col]
        for var in VARIANTS:
            r = results[var]
            mean_drops = r["row_drops"].mean(axis=0)
            # Only show features with non-trivial importance
            positive = mean_drops[mean_drops > 0.001]
            ax.hist(positive, bins=40, alpha=0.5, color=VARIANT_COLORS[var],
                    label=f"{VARIANT_LABELS[var]} ({len(positive)} features)",
                    density=True)
        ax.set_xlabel("Mean per-row loss drop")
        ax.set_ylabel("Density")
        ax.set_title("Importance distribution (features with drop > 0.001)")
        ax.legend(fontsize=9)

    fig.suptitle(
        "SAE Layer Selection: Task-Aware Fixed vs Per-Dataset Critical Layer\n"
        "Per-row single-feature ablation through full TabPFN forward pass",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    out_path = OUTPUT_DIR / "layer_comparison_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
