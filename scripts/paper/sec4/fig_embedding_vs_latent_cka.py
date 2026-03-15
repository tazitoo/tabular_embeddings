#!/usr/bin/env python3
"""
fig:embedding_vs_latent_cka — Embedding CKA vs Latent CKA scatter (two panels).

Section 4 (Results). Each point is a model pair. x-axis = Embedding CKA (geometry),
y-axis = Latent CKA (SAE concept similarity). No free parameters — both axes use CKA.

Left panel: Classification (7 models, 21 pairs)
Right panel: Regression (5 models, 10 pairs)

Data sources:
  - Embedding CKA: output/geometric_sweep_tabarena_7model.csv (per-dataset)
  - Latent CKA: output/paper_data/sec4/latent_cka_{classification,regression}.json

Usage:
    PYTHONPATH=. python scripts/paper/sec4/fig_embedding_vs_latent_cka.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adjustText import adjust_text

from scripts._project_root import PROJECT_ROOT

from scripts.paper.sec4.fig_geometric_vs_concept import (
    compute_cka_for_task,
    load_cka_per_dataset,
    TRANSFORMER_MODELS,
)

# Architecture groups for same/cross classification
ARCH_GROUPS = {
    'TabPFN': 'transformer', 'TabICL': 'transformer',
    'TabDPT': 'transformer', 'Mitra': 'transformer',
    'CARTE': 'gnn', 'HyperFast': 'hypernetwork', 'Tabula-8B': 'llm',
}


def is_same_arch(label: str) -> bool:
    """True if both models in the pair share an architecture group."""
    a, b = label.split('--')
    return ARCH_GROUPS.get(a) == ARCH_GROUPS.get(b)


def load_latent_cka(json_path: Path) -> dict:
    """Load latent CKA from JSON. Returns {pair_key: mean_cka}."""
    with open(json_path) as f:
        data = json.load(f)
    return {k: v['mean'] for k, v in data.items()}


def merge_data(emb_cka: dict, latent_cka: dict) -> dict:
    """Merge embedding CKA and latent CKA, matching pair keys in both orderings."""
    pairs = {}
    for pair_key, lat_cka in latent_cka.items():
        a, b = pair_key.split('--')
        emb_val = emb_cka.get(pair_key) or emb_cka.get(f"{b}--{a}")
        if emb_val is None:
            print(f"  Warning: no embedding CKA for {pair_key}, skipping")
            continue
        pairs[pair_key] = {'emb_cka': emb_val, 'latent_cka': lat_cka}
    return pairs


def plot_panel(ax, pairs: dict, title: str, show_ylabel: bool = True):
    """Plot embedding CKA vs latent CKA scatter on a single axes."""
    if not pairs:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=10, fontweight='bold')
        return

    # Plot points: circle = same architecture, triangle = different
    for label, vals in pairs.items():
        same = is_same_arch(label)
        marker = 'o' if same else '^'
        color = '#e41a1c' if same else '#377eb8'
        ax.scatter(vals['emb_cka'], vals['latent_cka'],
                   c=color, marker=marker, s=60, zorder=3,
                   edgecolors='white', linewidth=0.7)

    # Identity line (y = x)
    all_vals = [v['emb_cka'] for v in pairs.values()] + \
               [v['latent_cka'] for v in pairs.values()]
    lo, hi = min(all_vals) - 0.05, max(all_vals) + 0.05
    ax.plot([0, hi], [0, hi], ':', color='#dddddd', linewidth=0.8,
            zorder=1)

    # Label all pairs with adjustText
    texts = []
    for label, vals in pairs.items():
        same = is_same_arch(label)
        color = '#e41a1c' if same else '#377eb8'
        display = label.replace('--', '\u2013')
        texts.append(ax.text(vals['emb_cka'], vals['latent_cka'], display,
                             fontsize=5, color=color))
    adjust_text(texts, ax=ax, force_text=(1.5, 1.5), force_points=(2.0, 2.0),
                expand=(2.0, 2.0), ensure_inside_axes=True,
                arrowprops=dict(arrowstyle='-', color='#cccccc',
                                linewidth=0.5))

    ax.set_xlabel('Embedding CKA (geometry)', fontsize=9)
    if show_ylabel:
        ax.set_ylabel('Latent CKA (concepts)', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=7)


def main():
    cka_csv = PROJECT_ROOT / 'output' / 'geometric_sweep_tabarena_7model.csv'
    cls_latent = PROJECT_ROOT / 'output' / 'paper_data' / 'sec4' / 'latent_cka_classification.json'
    reg_latent = PROJECT_ROOT / 'output' / 'paper_data' / 'sec4' / 'latent_cka_regression.json'

    if not cka_csv.exists():
        print(f"Error: CKA CSV not found: {cka_csv}")
        return

    cka_rows = load_cka_per_dataset(cka_csv)
    print(f"Loaded {len(cka_rows)} CKA rows from {cka_csv.name}")

    # Classification panel
    cls_pairs = {}
    if cls_latent.exists():
        cls_emb_cka = compute_cka_for_task(cka_rows, 'classification')
        cls_lat_cka = load_latent_cka(cls_latent)
        cls_pairs = merge_data(cls_emb_cka, cls_lat_cka)
        print(f"Classification: {len(cls_pairs)} pairs")
    else:
        print(f"Warning: {cls_latent} not found — run compute_latent_cka.py --task-filter classification")

    # Regression panel
    reg_pairs = {}
    if reg_latent.exists():
        reg_emb_cka = compute_cka_for_task(cka_rows, 'regression')
        reg_lat_cka = load_latent_cka(reg_latent)
        reg_pairs = merge_data(reg_emb_cka, reg_lat_cka)
        print(f"Regression: {len(reg_pairs)} pairs")
    else:
        print(f"Warning: {reg_latent} not found — run compute_latent_cka.py --task-filter regression")

    if not cls_pairs and not reg_pairs:
        print("Error: no data for either panel")
        return

    # Two-panel figure
    fig, (ax_cls, ax_reg) = plt.subplots(1, 2, figsize=(10, 5.5))

    n_cls = len(cls_pairs)
    n_reg = len(reg_pairs)
    plot_panel(ax_cls, cls_pairs, f'Classification ({n_cls} pairs)', show_ylabel=True)
    plot_panel(ax_reg, reg_pairs, f'Regression ({n_reg} pairs)', show_ylabel=False)

    # Axis limits starting at 0
    all_pairs = list(cls_pairs.values()) + list(reg_pairs.values())
    if all_pairs:
        y_max = max(v['latent_cka'] for v in all_pairs) + 0.05
        x_max = max(v['emb_cka'] for v in all_pairs) + 0.05
        for ax in (ax_cls, ax_reg):
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, y_max)

    # Legend on classification panel
    ax_cls.scatter([], [], c='#e41a1c', marker='o', s=40, label='Same architecture')
    ax_cls.scatter([], [], c='#377eb8', marker='^', s=40, label='Different architecture')
    ax_cls.legend(fontsize=7, loc='upper left', framealpha=0.9)

    fig.tight_layout()

    out_dir = PROJECT_ROOT / 'output' / 'paper_figures' / 'sec4'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'fig_embedding_vs_latent_cka.pdf'
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    fig.savefig(str(out_path.with_suffix('.png')), dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path.with_suffix('.png')}")
    plt.close(fig)


if __name__ == '__main__':
    main()
