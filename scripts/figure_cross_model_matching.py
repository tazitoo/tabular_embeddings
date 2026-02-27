#!/usr/bin/env python3
"""
Figure: Cross-model SAE feature matching results.

Reads output/cross_model_feature_matching.json and produces a 3-panel figure:
  (A) Consensus concepts by breadth — stacked bars per band
  (B) Per-model consensus coverage — grouped bars per model
  (C) Pairwise Jaccard heatmap — pooled concept overlap across all bands

Usage:
    python scripts/figure_cross_model_matching.py
    python scripts/figure_cross_model_matching.py --input output/cross_model_feature_matching.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_matching_data(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_pairwise_jaccard(data: dict) -> np.ndarray:
    """Compute pairwise Jaccard similarity between models from consensus concepts.

    For each pair of models, Jaccard = |concepts containing both| / |concepts
    containing either| across all bands.
    """
    model_names = data["metadata"]["models"]
    n = len(model_names)
    sim = np.eye(n)

    # Collect all consensus concepts across bands
    all_concepts = []
    for band_label, band_data in data["bands"].items():
        all_concepts.extend(band_data["concepts"])

    for i in range(n):
        for j in range(i + 1, n):
            mi, mj = model_names[i], model_names[j]
            intersection = 0
            union = 0
            for c in all_concepts:
                has_i = mi in c["members"]
                has_j = mj in c["members"]
                if has_i or has_j:
                    union += 1
                if has_i and has_j:
                    intersection += 1
            sim[i, j] = sim[j, i] = intersection / union if union > 0 else 0.0

    return sim


def make_figure(data: dict, output_path: Path):
    model_names = data["metadata"]["models"]
    band_labels = sorted(data["bands"].keys(), key=lambda x: int(x[1:]))
    n_bands = len(band_labels)
    n_models = len(model_names)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4),
                             gridspec_kw={"wspace": 0.35})

    # -----------------------------------------------------------------------
    # (A) Consensus concepts by breadth — stacked bars per band
    # -----------------------------------------------------------------------
    ax = axes[0]
    max_n_models = n_models
    # Colors for 2, 3, 4, 5, 6, 7 models
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, max_n_models - 1))

    bottom = np.zeros(n_bands)
    legend_handles = []
    for nm in range(2, max_n_models + 1):
        counts = []
        for bl in band_labels:
            band = data["bands"][bl]
            n = sum(1 for c in band["concepts"] if c["n_models"] == nm)
            counts.append(n)
        counts = np.array(counts)
        if counts.sum() > 0:
            bars = ax.bar(range(n_bands), counts, bottom=bottom,
                          color=colors[nm - 2], label=f"{nm} models")
            legend_handles.append(bars)
            bottom += counts

    ax.set_xticks(range(n_bands))
    ax.set_xticklabels(band_labels, fontsize=8)
    ax.set_ylabel("Consensus concepts", fontsize=9)
    ax.set_title("(A) Consensus by breadth", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")

    # -----------------------------------------------------------------------
    # (B) Per-model consensus coverage — grouped bars per model
    # -----------------------------------------------------------------------
    ax = axes[1]
    x = np.arange(n_models)
    width = 0.8 / n_bands
    band_colors = plt.cm.Set2(np.linspace(0, 0.8, n_bands))

    for bi, bl in enumerate(band_labels):
        fracs = []
        for m in model_names:
            band = data["bands"][bl]
            # Count features for this model in consensus for this band
            in_cons = sum(
                len(c["members"].get(m, []))
                for c in band["concepts"]
            )
            # Count model-specific
            in_spec = len(band["model_specific"].get(m, []))
            total = in_cons + in_spec
            fracs.append(in_cons / total if total > 0 else 0.0)

        offset = (bi - n_bands / 2 + 0.5) * width
        ax.bar(x + offset, fracs, width, color=band_colors[bi], label=bl)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Fraction in consensus", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title("(B) Per-model coverage", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", ncol=2)

    # -----------------------------------------------------------------------
    # (C) Pairwise Jaccard heatmap
    # -----------------------------------------------------------------------
    ax = axes[2]
    sim = compute_pairwise_jaccard(data)

    # Lower triangle mask
    mask = np.tri(n_models, k=-1, dtype=bool)
    masked = np.where(mask, sim, np.nan)

    cmap = plt.cm.YlOrRd
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=0.7,
                   interpolation="nearest", aspect="equal")

    for i in range(n_models):
        for j in range(i):
            val = sim[i, j]
            color = "white" if val > 0.45 else "black"
            ax.text(j, i, f".{int(val * 100):02d}", ha="center", va="center",
                    fontsize=6.5, color=color)

    ax.set_xlim(-0.5, n_models - 1.5)
    ax.set_ylim(n_models - 0.5, 0.5)
    ax.set_xticks(range(n_models - 1))
    ax.set_yticks(range(1, n_models))
    ax.set_xticklabels(model_names[:-1], fontsize=6.5, rotation=45, ha="right")
    ax.set_yticklabels(model_names[1:], fontsize=6.5)
    ax.set_title("(C) Pairwise Jaccard", fontsize=10, fontweight="bold")
    ax.tick_params(length=0)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    fig.savefig(str(output_path.with_suffix(".png")), dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path.with_suffix('.png')}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Figure: cross-model SAE feature matching"
    )
    parser.add_argument(
        "--input", type=Path,
        default=PROJECT_ROOT / "output" / "cross_model_feature_matching.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "output" / "paper_figures" / "cross_model_matching.pdf",
    )
    args = parser.parse_args()

    data = load_matching_data(args.input)
    print(f"Loaded: {args.input}")
    print(f"  Models: {data['metadata']['models']}")
    print(f"  Datasets: {data['metadata']['n_datasets']}")
    print(f"  Bands: {list(data['bands'].keys())}")

    make_figure(data, args.output)


if __name__ == "__main__":
    main()
