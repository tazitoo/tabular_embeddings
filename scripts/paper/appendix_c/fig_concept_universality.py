#!/usr/bin/env python3
"""
Figure: Concept universality across tabular foundation models.

Panel A: Dictionary completeness — # concepts with d>1.0 per model per scale band.
Panel B: Universality — # concepts shared by N models at each scale band.

Usage:
    python scripts/figure_concept_universality.py \
        --output output/figures/concept_universality.pdf
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scripts._project_root import PROJECT_ROOT

from scripts.compare_sae_architectures import (
    META_NAMES,
    compute_activations,
    get_train_test_split,
    meta_features_to_array,
)
from scripts.compare_sae_cross_model import (
    DEFAULT_MODELS,
    SAE_FILENAME,
    collect_meta_for_datasets,
    find_common_datasets,
    pool_embeddings_for_datasets,
    sae_sweep_dir,
)
from scripts.analyze_sae_concepts_deep import load_sae_checkpoint


BAND_LABELS = ['S1\n[0,32)', 'S2\n[32,64)', 'S3\n[64,128)', 'S4\n[128,256)', 'S5\n[256,N)']
BAND_LABELS_SHORT = ['S1', 'S2', 'S3', 'S4', 'S5']
D_THRESHOLD = 1.0


def compute_covered_sets(
    activations: np.ndarray,
    meta_array: np.ndarray,
    config,
    d_threshold: float = D_THRESHOLD,
    top_k_samples: int = 100,
) -> List[Set[str]]:
    """
    For each Matryoshka scale band, return the set of meta-features
    that have at least one feature with |d| > threshold.
    """
    mat_dims = [d for d in config.matryoshka_dims if d <= config.hidden_dim]
    boundaries = [0] + mat_dims
    if boundaries[-1] < config.hidden_dim:
        boundaries.append(config.hidden_dim)

    baseline_means = meta_array.mean(axis=0)
    baseline_stds = meta_array.std(axis=0)

    band_sets = []
    for bi in range(len(boundaries) - 1):
        start, end = boundaries[bi], boundaries[bi + 1]
        band_acts = activations[:, start:end]
        band_max = band_acts.max(axis=0)
        alive_local = np.where(band_max > 0.001)[0]

        covered = set()
        for local_idx in alive_local:
            global_idx = start + local_idx
            feat_acts = activations[:, global_idx]
            top_indices = np.argsort(feat_acts)[-top_k_samples:]

            if feat_acts[top_indices].max() < 0.01:
                continue

            top_means = meta_array[top_indices].mean(axis=0)
            for j, name in enumerate(META_NAMES):
                if baseline_stds[j] > 1e-8:
                    d = abs((top_means[j] - baseline_means[j]) / baseline_stds[j])
                    if d >= d_threshold:
                        covered.add(name)

        band_sets.append(covered)

    return band_sets


def make_figure(
    model_band_sets: Dict[str, List[Set[str]]],
    model_names: List[str],
    output_path: Path,
    d_threshold: float = D_THRESHOLD,
):
    """Two-panel figure: completeness + universality."""
    n_bands = 5
    n_models = len(model_names)

    # --- Data for Panel A: completeness (# covered per model per band) ---
    completeness = np.zeros((n_models, n_bands))
    for i, name in enumerate(model_names):
        for bi in range(min(n_bands, len(model_band_sets[name]))):
            completeness[i, bi] = len(model_band_sets[name][bi])

    # --- Data for Panel B: universality (# shared by N models per band) ---
    # For each band, count how many models cover each meta-feature
    universality = np.zeros((n_models, n_bands))  # rows: shared by N, N-1, ..., 1
    for bi in range(n_bands):
        coverage_count = Counter()
        for name in model_names:
            if bi < len(model_band_sets[name]):
                for meta in model_band_sets[name][bi]:
                    coverage_count[meta] += 1

        for meta, count in coverage_count.items():
            universality[count - 1, bi] += 1

    # Also compute cumulative (across all bands) for an "Overall" column
    overall_coverage = Counter()
    for name in model_names:
        all_covered = set()
        for band_set in model_band_sets[name]:
            all_covered |= band_set
        for meta in all_covered:
            overall_coverage[meta] += 1

    overall_universality = np.zeros(n_models)
    for meta, count in overall_coverage.items():
        overall_universality[count - 1] += 1

    # Total meta-features
    n_total = len(META_NAMES)

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Colors for models (tab10 colormap scales to any count)
    cmap = plt.cm.tab10
    model_colors = [cmap(i) for i in range(n_models)]

    # Panel A: Grouped bar chart — completeness
    bar_width = min(0.18, 0.8 / n_models)
    x = np.arange(n_bands + 1)  # +1 for "Overall" column

    for i, name in enumerate(model_names):
        vals = list(completeness[i]) + [len(set().union(*model_band_sets[name]))]
        ax1.bar(x + i * bar_width - (n_models - 1) * bar_width / 2,
                vals, bar_width, label=name, color=model_colors[i], alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(BAND_LABELS_SHORT + ['Overall'], fontsize=10)
    ax1.set_ylabel(f'Concepts covered (|d| > {d_threshold})', fontsize=11)
    ax1.set_title('A. Dictionary Completeness', fontsize=13, fontweight='bold', loc='left')
    ax1.axhline(y=n_total, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.text(n_bands + 0.5, n_total + 0.3, f'{n_total} total', fontsize=8, color='gray',
             ha='right')
    ax1.legend(fontsize=8, loc='upper left', ncol=2 if n_models > 4 else 1)
    ax1.set_ylim(0, n_total + 3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Stacked bar chart — universality
    # Generate colors from dark (all models) to light (1 model)
    stack_cmap = plt.cm.RdYlBu_r
    stack_colors = [stack_cmap(0.15 + 0.7 * i / max(n_models - 1, 1)) for i in range(n_models)]
    stack_labels = [f'All {n_models} models' if k == n_models else
                    f'{k} model{"s" if k > 1 else ""}' for k in range(n_models, 0, -1)]

    x2 = np.arange(n_bands + 1)
    # Add "Overall" column
    univ_with_overall = np.column_stack([universality, overall_universality])
    # Plot from "All N" (bottom) to "1 model" (top)
    bottom = np.zeros(n_bands + 1)
    for idx, level in enumerate(range(n_models - 1, -1, -1)):
        vals = univ_with_overall[level]
        ax2.bar(x2, vals, 0.6, bottom=bottom,
                label=stack_labels[idx], color=stack_colors[idx], alpha=0.85)
        bottom += vals

    # Add count labels inside bars
    bottom = np.zeros(n_bands + 1)
    for level in range(n_models - 1, -1, -1):
        vals = univ_with_overall[level]
        for xi in range(n_bands + 1):
            if vals[xi] > 0:
                ax2.text(xi, bottom[xi] + vals[xi] / 2, f'{int(vals[xi])}',
                         ha='center', va='center', fontsize=8, fontweight='bold',
                         color='white' if level >= n_models // 2 else 'black')
        bottom += vals

    # Mark uncovered concepts
    n_covered_per_band = bottom
    for xi in range(n_bands + 1):
        n_uncovered = n_total - int(n_covered_per_band[xi])
        if n_uncovered > 0:
            ax2.bar(x2[xi], n_uncovered, 0.6, bottom=n_covered_per_band[xi],
                    color='#f0f0f0', edgecolor='#cccccc', linewidth=0.5)
            ax2.text(xi, n_covered_per_band[xi] + n_uncovered / 2, f'{n_uncovered}',
                     ha='center', va='center', fontsize=8, color='#999999')

    ax2.set_xticks(x2)
    ax2.set_xticklabels(BAND_LABELS_SHORT + ['Overall'], fontsize=10)
    ax2.set_ylabel('Number of meta-features', fontsize=11)
    ax2.set_title('B. Concept Universality', fontsize=13, fontweight='bold', loc='left')
    ax2.axhline(y=n_total, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.set_ylim(0, n_total + 3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Sparse Autoencoder Concept Coverage Across Tabular Foundation Models',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    fig.savefig(str(output_path.with_suffix('.png')), dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path.with_suffix('.png')}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Concept universality figure")
    parser.add_argument("--output", type=str,
                        default="output/figures/concept_universality.pdf")
    parser.add_argument("--max-per-dataset", type=int, default=500)
    parser.add_argument("--d-threshold", type=float, default=D_THRESHOLD)
    args = parser.parse_args()
    d_threshold = args.d_threshold

    base_sae = sae_sweep_dir()
    base_emb = PROJECT_ROOT / "output" / "embeddings" / "tabarena"

    # Resolve models
    model_configs = []
    for display_name, sweep_dir, emb_dir_name in DEFAULT_MODELS:
        sae_path = base_sae / sweep_dir / SAE_FILENAME
        emb_dir = base_emb / emb_dir_name
        if sae_path.exists() and emb_dir.exists():
            model_configs.append((display_name, sae_path, emb_dir))
    print(f"Models: {[m[0] for m in model_configs]}")

    # Common datasets
    emb_dirs = {name: emb_dir for name, _, emb_dir in model_configs}
    common_datasets = find_common_datasets(emb_dirs)

    # Meta-features (once)
    print(f"\nComputing meta-features for {len(common_datasets)} datasets...")
    meta_array, loaded_datasets = collect_meta_for_datasets(
        common_datasets, model_configs[0][2], max_per_dataset=args.max_per_dataset
    )
    print(f"  {meta_array.shape} from {len(loaded_datasets)} datasets")

    # Per-model: compute covered concept sets at each band
    model_band_sets = {}
    model_names = []

    for display_name, sae_path, emb_dir in model_configs:
        print(f"\nProcessing {display_name}...")
        model, config, _ = load_sae_checkpoint(sae_path)

        pooled = pool_embeddings_for_datasets(emb_dir, loaded_datasets, args.max_per_dataset)

        # Normalize using train split
        train_ds, _ = get_train_test_split(loaded_datasets)
        train_embs = []
        for ds in train_ds:
            path = emb_dir / f"tabarena_{ds}.npz"
            if not path.exists():
                continue
            data = np.load(path, allow_pickle=True)
            emb = data['embeddings'].astype(np.float32)
            if len(emb) > args.max_per_dataset:
                np.random.seed(42)
                emb = emb[np.random.choice(len(emb), args.max_per_dataset, replace=False)]
            train_embs.append(emb)
        train_pooled = np.concatenate(train_embs)
        train_std = train_pooled.std(axis=0, keepdims=True)
        train_std[train_std < 1e-8] = 1.0
        train_mean = (train_pooled / train_std).mean(axis=0, keepdims=True)

        acts = compute_activations(model, pooled, train_std, train_mean)
        n_min = min(len(meta_array), len(acts))

        band_sets = compute_covered_sets(
            acts[:n_min], meta_array[:n_min], config, d_threshold=args.d_threshold
        )
        model_band_sets[display_name] = band_sets
        model_names.append(display_name)

        print(f"  Covered per band: {[len(s) for s in band_sets]}")
        print(f"  Total unique: {len(set().union(*band_sets))}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"CONCEPT UNIVERSALITY SUMMARY (d > {args.d_threshold})")
    print(f"{'=' * 60}")
    for name in model_names:
        total = len(set().union(*model_band_sets[name]))
        per_band = [len(s) for s in model_band_sets[name]]
        print(f"  {name:8s}: {total} total | per band: {per_band}")

    # Overall: concepts shared by all
    all_union = set()
    all_intersection = set(META_NAMES)
    for name in model_names:
        model_all = set().union(*model_band_sets[name])
        all_union |= model_all
        all_intersection &= model_all

    print(f"\n  Universal (all {len(model_names)}):  {len(all_intersection)}")
    print(f"  Any model:         {len(all_union)}")
    print(f"  None:              {len(META_NAMES) - len(all_union)}")

    if all_intersection:
        print(f"\n  Universal concepts: {sorted(all_intersection)}")

    uncovered = set(META_NAMES) - all_union
    if uncovered:
        print(f"  Uncovered by all:  {sorted(uncovered)}")

    # Generate figure
    print("\nGenerating figure...")
    make_figure(model_band_sets, model_names, Path(args.output), d_threshold=args.d_threshold)


if __name__ == "__main__":
    main()
