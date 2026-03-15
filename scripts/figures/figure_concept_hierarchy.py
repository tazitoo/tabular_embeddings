#!/usr/bin/env python3
"""
Visualize the concept hierarchy analysis.

Reads output/concept_hierarchy.json (produced by analyze_concept_hierarchy.py)
and generates a 2-panel figure:

  (A) Signal exhaustion heatmap: models × Matryoshka bands
      Shows fraction of SAE features NOT explained by any PyMFE meta-feature.
      Light = well-explained (dataset-level signal), dark = unexplained.

  (B) Category importance by band (cross-model mean)
      Stacked bars showing which PyMFE super-categories drive each band.

Usage:
    python scripts/figure_concept_hierarchy.py
    python scripts/figure_concept_hierarchy.py --input output/concept_hierarchy.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scripts._project_root import PROJECT_ROOT

# Consistent with scripts/section43/universal_concepts.py
MODEL_ORDER = ["TabPFN", "CARTE", "TabICL", "TabDPT", "Mitra", "Tabula-8B"]
MODEL_COLORS = {
    "TabPFN": "#e41a1c",
    "CARTE": "#377eb8",
    "TabICL": "#ff7f00",
    "TabDPT": "#4daf4a",
    "Mitra": "#984ea3",
    "HyperFast": "#a65628",
    "Tabula-8B": "#f781bf",
}

CATEGORY_COLORS = {
    "General": "#1b9e77",
    "Statistical": "#d95f02",
    "Info-Theory": "#7570b3",
    "Model-Based": "#e7298a",
    "Landmarking": "#66a61e",
    "Complexity": "#e6ab02",
}

CATEGORY_ORDER = ["Complexity", "Model-Based", "General",
                  "Info-Theory", "Statistical", "Landmarking"]

BAND_LABELS = ["S1", "S2", "S3", "S4", "S5"]


def _save_fig(fig, output_path: Path):
    """Save figure as PDF + PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    fig.savefig(str(output_path.with_suffix('.png')), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path.with_suffix('.png')}")
    plt.close(fig)


def _band_number(label: str) -> str:
    """Extract 'S1' from 'S1 [0,96)'."""
    return label.split()[0]


def _get_band_data(model_bands: dict, band_num: str) -> dict | None:
    """Find band data by Sn number regardless of index range."""
    for label, data in model_bands.items():
        if _band_number(label) == band_num:
            return data
    return None


def make_hierarchy_figure(data: dict, output_path: Path):
    """Two-panel concept hierarchy figure."""
    models_in_data = [m for m in MODEL_ORDER if m in data["models"]]
    n_models = len(models_in_data)
    n_bands = len(BAND_LABELS)

    # --- Build matrices ---

    # (A) Signal exhaustion: fraction unexplained (models × bands)
    exhaustion = np.full((n_models, n_bands), np.nan)
    for i, m in enumerate(models_in_data):
        for j, bn in enumerate(BAND_LABELS):
            bd = _get_band_data(data["models"][m]["bands"], bn)
            if bd is not None:
                exhaustion[i, j] = bd["signal_exhaustion"]["frac_unexplained"]

    # (B) Variance decomposition by category (cross-model mean)
    # For each alive SAE feature, best-correlate R² is assigned to its category.
    # Remainder is "Unexplained". Averaged across models.
    cat_var = np.zeros((len(CATEGORY_ORDER), n_bands))
    unexplained_var = np.zeros(n_bands)
    for j, bn in enumerate(BAND_LABELS):
        model_decomps = []
        for m in models_in_data:
            bd = _get_band_data(data["models"][m]["bands"], bn)
            if bd is None or "variance_decomposition" not in bd:
                continue
            vd = bd["variance_decomposition"]
            row = [vd.get(cat, 0.0) for cat in CATEGORY_ORDER]
            row.append(vd.get("Unexplained", 1.0))
            model_decomps.append(row)
        if model_decomps:
            mean_decomp = np.mean(model_decomps, axis=0)
            cat_var[:, j] = mean_decomp[:len(CATEGORY_ORDER)]
            unexplained_var[j] = mean_decomp[-1]

    # --- Figure ---
    fig = plt.figure(figsize=(11, 4.5))
    gs = GridSpec(1, 2, width_ratios=[1.3, 1], wspace=0.35)

    # --- Panel A: Exhaustion heatmap ---
    ax_heat = fig.add_subplot(gs[0, 0])

    # Use "explained" fraction (1 - exhaustion) so lighter = less signal
    explained = 1.0 - exhaustion
    im = ax_heat.imshow(
        explained, aspect='auto', cmap='YlGnBu',
        vmin=0, vmax=1, interpolation='nearest',
    )

    # Annotate cells
    for i in range(n_models):
        for j in range(n_bands):
            val = exhaustion[i, j]
            if np.isnan(val):
                continue
            # Show as "% explained"
            pct = (1 - val) * 100
            color = 'white' if explained[i, j] > 0.6 else 'black'
            ax_heat.text(j, i, f'{pct:.0f}%', ha='center', va='center',
                         fontsize=8, color=color, fontweight='bold')

    ax_heat.set_xticks(range(n_bands))
    ax_heat.set_xticklabels(BAND_LABELS, fontsize=9)
    ax_heat.set_yticks(range(n_models))
    ax_heat.set_yticklabels(models_in_data, fontsize=9)
    ax_heat.set_xlabel('Matryoshka band', fontsize=10)
    ax_heat.set_title('(A) % SAE features explained by PyMFE',
                       fontsize=11, fontweight='bold', loc='left', pad=8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.02)
    cbar.set_label('Fraction explained', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # --- Panel B: Variance decomposition stacked bars ---
    ax_bar = fig.add_subplot(gs[0, 1])

    x = np.arange(n_bands)
    bar_width = 0.6
    bottom = np.zeros(n_bands)

    for ci, cat in enumerate(CATEGORY_ORDER):
        vals = cat_var[ci, :] * 100
        ax_bar.bar(x, vals, bar_width, bottom=bottom, label=cat,
                   color=CATEGORY_COLORS[cat], zorder=3)
        bottom += vals

    # Unexplained on top in grey
    ax_bar.bar(x, unexplained_var * 100, bar_width, bottom=bottom,
               label='Unexplained', color='#cccccc', zorder=3)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(BAND_LABELS, fontsize=9)
    ax_bar.set_ylabel('Fraction of feature variance (%)', fontsize=10)
    ax_bar.set_ylim(0, 105)
    ax_bar.set_xlabel('Matryoshka band', fontsize=10)
    ax_bar.set_title('(B) Variance decomposition by category',
                       fontsize=11, fontweight='bold', loc='left', pad=8)
    ax_bar.legend(fontsize=7, loc='upper right', framealpha=0.9)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.tick_params(labelsize=8)

    _save_fig(fig, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize concept hierarchy analysis"
    )
    parser.add_argument("--input", type=str,
                        default="output/concept_hierarchy.json",
                        help="Input JSON from analyze_concept_hierarchy.py")
    parser.add_argument("--output", type=str,
                        default="output/figures/concept_hierarchy.pdf",
                        help="Output figure path")
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    with open(input_path) as f:
        data = json.load(f)

    print(f"Loaded: {input_path}")
    print(f"  Models: {list(data['models'].keys())}")
    print(f"  Datasets: {data['metadata']['n_datasets']}")

    output_path = PROJECT_ROOT / args.output
    make_hierarchy_figure(data, output_path)


if __name__ == "__main__":
    main()
