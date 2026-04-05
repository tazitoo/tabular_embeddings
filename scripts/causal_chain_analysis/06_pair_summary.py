#!/usr/bin/env python3
"""Stage 6: One-page summary per model pair.

Generates a summary figure with:
- Left: concept category bar chart
- Right: concept × dataset heatmap (top 10 concepts × top 10 datasets)
- Bottom: text listing top 3 concepts with labels

Usage:
    python -m scripts.causal_chain_analysis.06_pair_summary
    python -m scripts.causal_chain_analysis.06_pair_summary --pair tabicl_vs_tabicl_v2
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts._project_root import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

EVIDENCE_DIR = PROJECT_ROOT / "output" / "concept_evidence"
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures" / "concept_evidence"

CATEGORY_COLORS = {
    "Outlier detection": "#e41a1c",
    "Magnitude patterns": "#377eb8",
    "Distribution shape": "#4daf4a",
    "Categorical patterns": "#984ea3",
    "Sparsity patterns": "#ff7f00",
    "Boundary/density": "#a65628",
    "PCA alignment": "#f781bf",
    "Minimum/baseline": "#999999",
    "Other": "#666666",
}


def plot_pair_summary(pair_name, model_key, data, categories):
    """Generate one-page summary for one model's role in a pair."""
    disp = data["display_name"]

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.3)

    # Left: category bar chart
    ax1 = fig.add_subplot(gs[0])
    cats = list(categories["categories"].keys())
    fracs = [categories["categories"][c]["fraction_of_selections"] for c in cats]
    counts = [categories["categories"][c]["n_features"] for c in cats]
    colors = [CATEGORY_COLORS.get(c, "#666666") for c in cats]

    y_pos = np.arange(len(cats))
    bars = ax1.barh(y_pos, fracs, color=colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{c}\n({n} feat)" for c, n in zip(cats, counts)],
                         fontsize=7)
    ax1.set_xlabel("Fraction of row selections", fontsize=9)
    ax1.set_title(f"{disp} concept categories", fontsize=10, fontweight="bold")
    ax1.invert_yaxis()

    # Right: concept × dataset heatmap
    ax2 = fig.add_subplot(gs[1])

    matrix = np.array(data["matrix"])
    features = data["feature_indices"][:10]
    datasets = data["datasets"]
    labels = data["feature_labels"]
    bands = data["feature_bands"]
    unmatched = data["feature_unmatched"]

    if len(features) == 0 or len(datasets) == 0:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        mat_sub = matrix[:min(10, len(features)), :]
        # Sort datasets by total importance
        ds_importance = mat_sub.sum(axis=0)
        ds_order = np.argsort(-ds_importance)[:min(15, len(datasets))]
        mat_plot = mat_sub[:, ds_order]

        im = ax2.imshow(mat_plot, aspect="auto", cmap="YlOrRd",
                        interpolation="nearest")
        ax2.set_xticks(np.arange(len(ds_order)))
        ax2.set_xticklabels([datasets[i][:15] for i in ds_order],
                             rotation=45, ha="right", fontsize=6)
        ax2.set_yticks(np.arange(len(features)))
        feat_labels = []
        for fi in features:
            b = bands.get(str(fi), -1)
            u = "U" if unmatched.get(str(fi)) else "M"
            l = labels.get(str(fi), "")[:25]
            feat_labels.append(f"f{fi} [B{b},{u}] {l}")
        ax2.set_yticklabels(feat_labels, fontsize=6)
        ax2.set_title(f"Concept × Dataset importance", fontsize=10,
                      fontweight="bold")
        plt.colorbar(im, ax=ax2, shrink=0.8, label="Selection fraction")

    fig.suptitle(f"{pair_name}: {disp} when strong", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Stage 6: One-page summary per model pair")
    parser.add_argument("--pair", default=None)
    args = parser.parse_args()

    if args.pair:
        pairs = [args.pair]
    else:
        pairs = sorted(d.name for d in EVIDENCE_DIR.iterdir()
                        if d.is_dir() and (d / "concept_dataset_matrix.json").exists())

    for pair in pairs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Pair: {pair}")

        matrix_path = EVIDENCE_DIR / pair / "concept_dataset_matrix.json"
        cat_path = EVIDENCE_DIR / pair / "concept_categories.json"

        if not matrix_path.exists():
            logger.warning(f"  Missing concept_dataset_matrix.json")
            continue

        matrix_data = json.loads(matrix_path.read_text())

        # Load categories if available
        if cat_path.exists():
            cat_data = json.loads(cat_path.read_text())
        else:
            cat_data = None

        for model_key, data in matrix_data.items():
            categories = cat_data.get(model_key) if cat_data else None
            if categories is None:
                logger.info(f"  No categories for {model_key}, skipping summary")
                continue

            fig = plot_pair_summary(pair, model_key, data, categories)

            out_dir = OUTPUT_DIR / pair
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"summary_{model_key}.pdf"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  Saved {out_path.name}")


if __name__ == "__main__":
    main()
