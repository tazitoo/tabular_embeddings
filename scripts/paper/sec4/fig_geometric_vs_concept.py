#!/usr/bin/env python3
"""
fig:geometric_vs_concept — CKA vs SAE concept overlap scatter (two panels).

Section 4 (Results). Each point is a model pair. x-axis = CKA (geometry),
y-axis = Jaccard (concept overlap). S1 (coarse) and S5 (fine) shown together:
coarse concepts track geometry, fine concepts don't.

Left panel: Classification (7 models, 38 datasets, 21 pairs)
Right panel: Regression (5 models, 13 datasets, 10 pairs)

CKA is now computed per-task by filtering the per-dataset CSV, ensuring
CKA and Jaccard are measured over the same dataset population.

Data sources:
  - CKA: output/geometric_sweep_tabarena_7model.csv (per-dataset)
  - Jaccard: output/paper_data/sec1/pairwise_jaccard_{cls,regression}.json

Usage:
    python scripts/paper/sec4/fig_geometric_vs_concept.py
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts._project_root import PROJECT_ROOT

from scripts.compare_sae_cross_model import get_dataset_tasks

# Model name mapping: CSV uses lowercase, display uses title case
MODEL_DISPLAY = {
    'tabpfn': 'TabPFN',
    'carte': 'CARTE',
    'tabicl': 'TabICL',
    'tabdpt': 'TabDPT',
    'mitra': 'Mitra',
    'hyperfast': 'HyperFast',
    'tabula': 'Tabula-8B',
}

TRANSFORMER_MODELS = {'TabPFN', 'TabICL', 'TabDPT', 'Mitra'}


def load_cka_per_dataset(csv_path: Path) -> list:
    """Load per-dataset CKA rows. Returns list of dicts."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['model_a_display'] = MODEL_DISPLAY.get(row['model_a'], row['model_a'])
            row['model_b_display'] = MODEL_DISPLAY.get(row['model_b'], row['model_b'])
            row['dataset_name'] = row['dataset'].replace('tabarena_', '')
            row['cka_score'] = float(row['cka_score'])
            rows.append(row)
    return rows


def compute_cka_for_task(rows: list, task: str) -> dict:
    """Filter CKA rows by task type and compute per-pair mean.

    Returns {pair_key: cka_mean}.
    """
    dataset_tasks = get_dataset_tasks()

    # Group by model pair
    pair_scores = defaultdict(list)
    for row in rows:
        ds = row['dataset_name']
        if dataset_tasks.get(ds) != task:
            continue
        a, b = row['model_a_display'], row['model_b_display']
        key = f"{a}--{b}"
        pair_scores[key].append(row['cka_score'])

    return {k: np.mean(v) for k, v in pair_scores.items()}


def load_jaccard_data(json_path: Path) -> dict:
    """Load pairwise Jaccard from JSON. Returns {pair_key: {s1: ..., s5: ...}}."""
    with open(json_path) as f:
        data = json.load(f)

    band_keys = list(data['bands'].keys())
    short_keys = ['s1', 's2', 's3', 's4', 's5']

    jaccard = {}
    for pair_key in data['overall']:
        jaccard[pair_key] = {'overall': data['overall'][pair_key]}
        for bk, sk in zip(band_keys, short_keys):
            jaccard[pair_key][sk] = data['bands'][bk].get(pair_key, 0.0)
    return jaccard


def pair_category(label: str) -> str:
    """Classify a model pair by architecture group."""
    models = set(label.split('--'))
    if models <= TRANSFORMER_MODELS:
        return 'transformer'
    if 'CARTE' in models and models - {'CARTE'} <= TRANSFORMER_MODELS:
        return 'carte'
    if 'Tabula-8B' in models:
        return 'tabula8b'
    return 'hyperfast'


CATEGORY_STYLE = {
    'transformer': ('#e41a1c', 'o', 'Transformer'),
    'carte':       ('#377eb8', 's', 'CARTE'),
    'hyperfast':   ('#4daf4a', 'D', 'HyperFast'),
    'tabula8b':    ('#984ea3', '^', 'Tabula-8B'),
}


def merge_cka_jaccard(cka_data: dict, jaccard_data: dict) -> dict:
    """Merge CKA and Jaccard data, matching pair keys in both orderings."""
    pairs = {}
    for pair_key, jac in jaccard_data.items():
        a, b = pair_key.split('--')
        cka_val = cka_data.get(pair_key) or cka_data.get(f"{b}--{a}")
        if cka_val is None:
            print(f"  Warning: no CKA data for {pair_key}, skipping")
            continue
        pairs[pair_key] = {'cka': cka_val, **{k: v for k, v in jac.items() if k != 'overall'}}
    return pairs


def plot_scatter_panel(ax, pairs: dict, title: str, show_ylabel: bool = True):
    """Plot CKA vs Jaccard scatter on a single axes."""
    if not pairs:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=9, fontweight='bold')
        return

    # Plot S5 first (behind, faded)
    for label, vals in pairs.items():
        cat = pair_category(label)
        color, marker, _ = CATEGORY_STYLE[cat]
        ax.scatter(vals['cka'], vals['s5'],
                   c=color, marker=marker, s=35, zorder=2, alpha=0.3,
                   edgecolors='none')

    # Plot S1 (foreground, solid)
    for label, vals in pairs.items():
        cat = pair_category(label)
        color, marker, _ = CATEGORY_STYLE[cat]
        ax.scatter(vals['cka'], vals['s1'],
                   c=color, marker=marker, s=60, zorder=3, edgecolors='white',
                   linewidth=0.7)
        # Connect S1 to S5 with a thin line
        ax.plot([vals['cka'], vals['cka']], [vals['s1'], vals['s5']],
                color=color, alpha=0.25, linewidth=0.8, zorder=1)

    # Trend line for S1
    x = np.array([v['cka'] for v in pairs.values()])
    y_s1 = np.array([v['s1'] for v in pairs.values()])
    if len(x) >= 3:
        coeffs = np.polyfit(x, y_s1, 1)
        x_fit = np.linspace(x.min() - 0.03, x.max() + 0.03, 100)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), '--', color='gray',
                alpha=0.5, linewidth=1, zorder=1)

    # Label informative pairs
    for label, vals in pairs.items():
        cat = pair_category(label)
        color, _, _ = CATEGORY_STYLE[cat]
        models = set(label.split('--'))
        display = label.replace('--', '\u2013')

        # Only label a subset to avoid clutter
        should_label = False
        if models == {'TabDPT', 'Mitra'} or models == {'TabPFN', 'TabDPT'}:
            should_label = True
        elif 'HyperFast' in models and ('Tabula-8B' in models or 'Mitra' in models):
            should_label = True
        elif 'Tabula-8B' in models and 'TabPFN' in models:
            should_label = True

        if not should_label:
            continue

        ha, va, dx, dy = 'left', 'bottom', 0.015, 0.012
        if 'Tabula-8B' in models and 'HyperFast' in models:
            ha, va, dx, dy = 'left', 'top', 0.015, -0.012
        elif 'HyperFast' in models and 'Mitra' in models:
            ha, va, dx, dy = 'right', 'bottom', -0.015, 0.012

        ax.annotate(display, (vals['cka'] + dx, vals['s1'] + dy),
                    fontsize=5.5, ha=ha, va=va, color=color)

    ax.set_xlabel('CKA Similarity', fontsize=9)
    if show_ylabel:
        ax.set_ylabel('Concept Overlap (Jaccard)', fontsize=9)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=7)


def plot_band_strip(ax, pairs: dict, title: str = ''):
    """Plot r-value and mean Jaccard by band on a strip axes."""
    if not pairs:
        return [], []

    bands = ['s1', 's2', 's3', 's4', 's5']
    band_labels = ['S1\n[0,32)', 'S2\n[32,64)', 'S3\n[64,128)',
                    'S4\n[128,256)', 'S5\n[256,N)']
    x_arr = np.array([v['cka'] for v in pairs.values()])

    r_vals, mean_j = [], []
    for b in bands:
        y_b = np.array([v[b] for v in pairs.values()])
        r_vals.append(np.corrcoef(x_arr, y_b)[0, 1] if len(x_arr) >= 3 else 0.0)
        mean_j.append(y_b.mean())

    ax2 = ax.twinx()
    ax2.bar(range(5), mean_j, 0.5, color='#dddddd', alpha=0.5, zorder=1)
    ax.plot(range(5), r_vals, 'o-', color='#c0392b', markersize=5,
            linewidth=1.5, zorder=3)
    ax.axhline(0, color='#999999', linewidth=0.5, linestyle=':', zorder=2)

    for i, rv in enumerate(r_vals):
        ax.annotate(f'{rv:+.2f}', (i, rv),
                    textcoords='offset points', xytext=(0, 7),
                    fontsize=5.5, ha='center', va='bottom', color='#c0392b',
                    fontweight='bold')

    ax.set_xticks(range(5))
    ax.set_xticklabels(band_labels, fontsize=6)
    ax.set_ylim(-0.7, 0.7)
    ax.set_yticks([-0.5, 0, 0.5])
    ax.tick_params(labelsize=6)
    ax2.set_ylim(0, 0.45)
    ax2.tick_params(labelsize=6, colors='#888888')
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    if title:
        ax.set_title(title, fontsize=7, fontstyle='italic', pad=2)

    return r_vals, mean_j


def make_figure(cls_pairs: dict, reg_pairs: dict, output_path: Path):
    """Two-panel figure: classification (left) and regression (right)."""
    fig = plt.figure(figsize=(10, 5.5))

    # 2x2 grid: top row = scatter, bottom row = band strips
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.2], hspace=0.35, wspace=0.25)

    ax_cls = fig.add_subplot(gs[0, 0])
    ax_reg = fig.add_subplot(gs[0, 1])
    ax_strip_cls = fig.add_subplot(gs[1, 0])
    ax_strip_reg = fig.add_subplot(gs[1, 1])

    # Compute shared y-axis range
    all_pairs = list(cls_pairs.values()) + list(reg_pairs.values())
    if all_pairs:
        y_max = max(max(v.get('s1', 0) for v in all_pairs),
                    max(v.get('s5', 0) for v in all_pairs)) + 0.05
        y_max = max(0.58, y_max)
    else:
        y_max = 0.58

    n_cls = len(cls_pairs)
    n_reg = len(reg_pairs)

    plot_scatter_panel(ax_cls, cls_pairs,
                       f'Classification ({n_cls} pairs)', show_ylabel=True)
    plot_scatter_panel(ax_reg, reg_pairs,
                       f'Regression ({n_reg} pairs)', show_ylabel=False)

    # Shared y limits
    ax_cls.set_ylim(-0.02, y_max)
    ax_reg.set_ylim(-0.02, y_max)

    # x limits per panel based on data
    for ax, pairs in [(ax_cls, cls_pairs), (ax_reg, reg_pairs)]:
        if pairs:
            xs = [v['cka'] for v in pairs.values()]
            ax.set_xlim(min(xs) - 0.05, max(xs) + 0.05)

    # Shared legend (on classification panel)
    present_cats = set()
    for label in list(cls_pairs.keys()) + list(reg_pairs.keys()):
        present_cats.add(pair_category(label))
    for cat_key in ['transformer', 'carte', 'hyperfast', 'tabula8b']:
        if cat_key in present_cats:
            color, marker, cat_label = CATEGORY_STYLE[cat_key]
            ax_cls.scatter([], [], c=color, marker=marker, s=40, label=cat_label)
    ax_cls.scatter([], [], c='gray', marker='o', s=25, alpha=0.3, label='S5 (fine)')
    ax_cls.scatter([], [], c='gray', marker='o', s=40, label='S1 (coarse)')
    ax_cls.legend(fontsize=6, loc='upper left', framealpha=0.9)

    # Band strips
    cls_r, cls_mj = plot_band_strip(ax_strip_cls, cls_pairs)
    reg_r, reg_mj = plot_band_strip(ax_strip_reg, reg_pairs)

    ax_strip_cls.set_ylabel('r', fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    fig.savefig(str(output_path.with_suffix('.png')), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path.with_suffix('.png')}")
    plt.close(fig)

    # Print summary
    bands = ['s1', 's2', 's3', 's4', 's5']
    for label, r_vals, mj_vals in [('Classification', cls_r, cls_mj),
                                     ('Regression', reg_r, reg_mj)]:
        print(f"\n{label}:")
        for i, b in enumerate(bands):
            print(f"  {b}: r = {r_vals[i]:.3f}, mean Jaccard = {mj_vals[i]:.3f}")


def main():
    cka_csv_path = PROJECT_ROOT / 'output' / 'geometric_sweep_tabarena_7model.csv'
    cls_jaccard_path = PROJECT_ROOT / 'output' / 'paper_data' / 'sec1' / 'pairwise_jaccard_classification.json'
    reg_jaccard_path = PROJECT_ROOT / 'output' / 'paper_data' / 'sec1' / 'pairwise_jaccard_regression.json'

    if not cka_csv_path.exists():
        print(f"Error: CKA per-dataset CSV not found: {cka_csv_path}")
        return

    # Load per-dataset CKA
    cka_rows = load_cka_per_dataset(cka_csv_path)
    print(f"Loaded {len(cka_rows)} CKA rows from {cka_csv_path.name}")

    # Classification panel
    cls_pairs = {}
    if cls_jaccard_path.exists():
        cls_cka = compute_cka_for_task(cka_rows, 'classification')
        cls_jaccard = load_jaccard_data(cls_jaccard_path)
        cls_pairs = merge_cka_jaccard(cls_cka, cls_jaccard)
        print(f"Classification: {len(cls_pairs)} pairs")
    else:
        print(f"Warning: {cls_jaccard_path} not found — run fig_dictionary_comparison.py --task-filter classification")

    # Regression panel
    reg_pairs = {}
    if reg_jaccard_path.exists():
        reg_cka = compute_cka_for_task(cka_rows, 'regression')
        reg_jaccard = load_jaccard_data(reg_jaccard_path)
        reg_pairs = merge_cka_jaccard(reg_cka, reg_jaccard)
        print(f"Regression: {len(reg_pairs)} pairs")
    else:
        print(f"Warning: {reg_jaccard_path} not found — run fig_dictionary_comparison.py --task-filter regression")

    if not cls_pairs and not reg_pairs:
        print("Error: no data for either panel")
        return

    output = PROJECT_ROOT / 'output' / 'paper_figures' / 'sec4' / 'fig_geometric_vs_concept.pdf'
    make_figure(cls_pairs, reg_pairs, output)


if __name__ == '__main__':
    main()
