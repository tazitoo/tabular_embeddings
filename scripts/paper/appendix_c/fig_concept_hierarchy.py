#!/usr/bin/env python3
"""
Figure: Cross-model SAE concept hierarchy heatmap.

Shows concept coverage (max |Cohen's d|) at each Matryoshka scale band
for all models side-by-side. Reveals universal vs model-specific concepts
and hierarchical structure.

Usage:
    python scripts/figure_cross_model_heatmap.py \
        --output output/figures/cross_model_concept_hierarchy.pdf
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from scripts._project_root import PROJECT_ROOT

from scripts.compare_sae_architectures import (
    META_NAMES,
    compute_activations,
    compute_basic_metrics,
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
from scripts.analyze_sae_concepts_deep import (
    compute_column_stats,
    compute_concept_coverage,
    compute_row_meta_features,
    load_sae_checkpoint,
)
from data.extended_loader import load_tabarena_dataset

# Meta-feature display names (shorter, for figure)
META_DISPLAY = {
    'missing_rate': 'Missing rate',
    'missing_numeric_rate': 'Missing (numeric)',
    'missing_categorical_rate': 'Missing (categorical)',
    'numeric_mean_zscore': 'Mean z-score',
    'numeric_max_zscore': 'Max z-score',
    'numeric_min_zscore': 'Min z-score',
    'numeric_std': 'Std deviation',
    'numeric_skewness': 'Skewness',
    'numeric_kurtosis': 'Kurtosis',
    'numeric_range': 'Range',
    'numeric_iqr': 'IQR',
    'frac_zeros': 'Frac zeros',
    'frac_negative': 'Frac negative',
    'frac_positive_outliers': 'Outliers (+)',
    'frac_negative_outliers': 'Outliers (-)',
    'categorical_rarity': 'Cat. rarity',
    'categorical_modal_frac': 'Cat. modal frac',
    'n_rare_categories': 'Rare categories',
    'n_unique_categories': 'Unique categories',
    'categorical_entropy': 'Cat. entropy',
    'row_entropy': 'Row entropy',
    'row_uniformity': 'Row uniformity',
    'n_distinct_values': 'Distinct values',
    'centroid_distance': 'Centroid dist.',
    'nearest_neighbor_dist': 'NN distance',
    'local_density': 'Local density',
    'pca_pc1': 'PCA PC1',
    'pca_pc2': 'PCA PC2',
    'pca_residual': 'PCA residual',
    'n_numeric': 'N numeric cols',
    'n_categorical': 'N categorical',
    'n_rows_total': 'N rows',
    'n_cols_total': 'N cols total',
    'dataset_sparsity': 'Dataset sparsity',
    'numeric_correlation_mean': 'Feature corr.',
    # Scale/Magnitude
    'log_magnitude_mean': 'Log magnitude',
    'log_magnitude_std': 'Log mag. std',
    'frac_very_small': 'Frac tiny',
    'frac_very_large': 'Frac huge',
    'frac_integers': 'Frac integers',
    'frac_round_tens': 'Frac round 10s',
    # Supervised Complexity
    'fisher_ratio': 'Fisher ratio',
    'borderline': 'Borderline',
    'knn_class_ratio': 'kNN class ratio',
    'linear_boundary_dist': 'SVM boundary',
    # Graph Topology
    'hub_score': 'Hub score',
    'local_clustering': 'Local clustering',
    'local_intrinsic_dim': 'Local dim.',
    # Information-Theoretic
    'row_surprise': 'Row surprise',
    'mi_contribution': 'MI contribution',
    # Target
    'target_is_minority': 'Minority class',
    'target_zscore': 'Target z-score',
}

# Group meta-features by category for visual organization
META_GROUPS = {
    'Geometric': [
        'centroid_distance', 'nearest_neighbor_dist', 'local_density',
        'pca_pc1', 'pca_pc2', 'pca_residual',
    ],
    'Distribution': [
        'frac_negative', 'numeric_skewness', 'numeric_kurtosis',
        'numeric_range', 'frac_zeros', 'row_entropy', 'row_uniformity',
        'numeric_max_zscore', 'numeric_min_zscore', 'numeric_mean_zscore',
        'numeric_std', 'numeric_iqr',
        'frac_positive_outliers', 'frac_negative_outliers',
        'n_distinct_values',
    ],
    'Scale': [
        'log_magnitude_mean', 'log_magnitude_std',
        'frac_very_small', 'frac_very_large', 'frac_integers', 'frac_round_tens',
    ],
    'Structure': [
        'n_numeric', 'n_cols_total', 'n_rows_total',
        'numeric_correlation_mean', 'dataset_sparsity',
    ],
    'Supervised Complexity': [
        'fisher_ratio', 'borderline', 'knn_class_ratio', 'linear_boundary_dist',
    ],
    'Graph Topology': [
        'hub_score', 'local_clustering', 'local_intrinsic_dim',
    ],
    'Information': [
        'row_surprise', 'mi_contribution',
    ],
    'Target': [
        'target_is_minority', 'target_zscore',
    ],
    'Categorical': [
        'n_categorical', 'categorical_rarity', 'categorical_modal_frac',
        'n_rare_categories', 'n_unique_categories', 'categorical_entropy',
    ],
    'Missing': [
        'missing_rate', 'missing_numeric_rate', 'missing_categorical_rate',
    ],
}


def compute_band_coverage(
    activations: np.ndarray,
    meta_array: np.ndarray,
    config,
    top_k_samples: int = 100,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-meta-feature coverage score at each Matryoshka scale band.

    Returns: {band_label: {meta_name: max_abs_cohen_d}}
    """
    mat_dims = [d for d in config.matryoshka_dims if d <= config.hidden_dim]
    boundaries = [0] + mat_dims
    if boundaries[-1] < config.hidden_dim:
        boundaries.append(config.hidden_dim)

    baseline_means = meta_array.mean(axis=0)
    baseline_stds = meta_array.std(axis=0)

    band_coverage = {}

    for bi in range(len(boundaries) - 1):
        start, end = boundaries[bi], boundaries[bi + 1]
        band_label = f"[{start},{end})"

        band_acts = activations[:, start:end]
        band_max = band_acts.max(axis=0)
        alive_mask = band_max > 0.001
        alive_local = np.where(alive_mask)[0]

        # For each alive feature in this band, compute effect sizes
        meta_max_d = {name: 0.0 for name in META_NAMES}

        for local_idx in alive_local:
            global_idx = start + local_idx
            feat_acts = activations[:, global_idx]
            top_indices = np.argsort(feat_acts)[-top_k_samples:]

            if feat_acts[top_indices].max() < 0.01:
                continue

            top_meta = meta_array[top_indices]
            top_means = top_meta.mean(axis=0)

            for j, name in enumerate(META_NAMES):
                if baseline_stds[j] > 1e-8:
                    d = abs((top_means[j] - baseline_means[j]) / baseline_stds[j])
                    meta_max_d[name] = max(meta_max_d[name], d)

        band_coverage[band_label] = meta_max_d

    return band_coverage


def get_ordered_meta_features() -> List[str]:
    """Return meta-features ordered by group for visual coherence."""
    ordered = []
    for group_name in ['Geometric', 'Distribution', 'Structure', 'Target', 'Categorical', 'Missing']:
        ordered.extend(META_GROUPS[group_name])
    return ordered


def make_figure(
    all_band_coverage: Dict[str, Dict[str, Dict[str, float]]],
    model_names: List[str],
    output_path: Path,
):
    """Create the multi-panel heatmap figure."""
    ordered_meta = get_ordered_meta_features()
    n_meta = len(ordered_meta)

    # Use 5 scale bands (shared Matryoshka dims + remainder)
    band_labels = ['S1\n[0,32)', 'S2\n[32,64)', 'S3\n[64,128)', 'S4\n[128,256)', 'S5\n[256,N)']
    n_bands = 5

    # Build data matrices: one per model
    matrices = {}
    for model_name in model_names:
        mat = np.zeros((n_meta, n_bands))
        bands = list(all_band_coverage[model_name].keys())

        for bi in range(min(n_bands, len(bands))):
            band_key = bands[bi]
            coverage = all_band_coverage[model_name][band_key]
            for mi, meta_name in enumerate(ordered_meta):
                mat[mi, bi] = coverage.get(meta_name, 0.0)
        matrices[model_name] = mat

    # Compute sort order: by mean coverage across all models (descending)
    mean_coverage = np.zeros(n_meta)
    for model_name in model_names:
        mean_coverage += matrices[model_name].max(axis=1)
    mean_coverage /= len(model_names)

    # Group boundaries for visual separators
    group_sizes = [len(META_GROUPS[g]) for g in
                   ['Geometric', 'Distribution', 'Structure', 'Target', 'Categorical', 'Missing']]
    group_boundaries = np.cumsum(group_sizes)[:-1]
    group_labels = ['Geometric', 'Distribution', 'Structure', 'Target', 'Categorical', 'Missing']

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(1, len(model_names) + 1, width_ratios=[1] * len(model_names) + [0.05],
                  wspace=0.08)

    vmin, vmax = 0, 5.0
    cmap = plt.cm.YlOrRd

    axes = []
    for i, model_name in enumerate(model_names):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)

        mat = matrices[model_name]
        im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='nearest')

        # Title
        ax.set_title(model_name, fontsize=13, fontweight='bold', pad=8)

        # X-axis: scale bands
        ax.set_xticks(range(n_bands))
        ax.set_xticklabels(band_labels, fontsize=8)

        # Y-axis: meta-features (only on leftmost panel)
        if i == 0:
            display_names = [META_DISPLAY.get(m, m) for m in ordered_meta]
            ax.set_yticks(range(n_meta))
            ax.set_yticklabels(display_names, fontsize=8)
        else:
            ax.set_yticks([])

        # Group separators
        for boundary in group_boundaries:
            ax.axhline(y=boundary - 0.5, color='white', linewidth=2)

        # Annotate cells with values > 2.0
        for mi in range(n_meta):
            for bi in range(n_bands):
                val = mat[mi, bi]
                if val > 2.0:
                    color = 'white' if val > 3.5 else 'black'
                    ax.text(bi, mi, f'{val:.1f}', ha='center', va='center',
                            fontsize=6, color=color, fontweight='bold')

    # Group labels on the right side
    ax_right = axes[-1]
    cumsum = 0
    for gi, (gname, gsize) in enumerate(zip(group_labels, group_sizes)):
        mid = cumsum + gsize / 2 - 0.5
        cumsum += gsize

    # Colorbar
    cax = fig.add_subplot(gs[0, -1])
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("|Cohen's d|", fontsize=11)

    # Overall title
    fig.suptitle("Concept Coverage by Matryoshka Scale Band Across Models",
                 fontsize=15, fontweight='bold', y=0.98)
    fig.text(0.5, 0.01,
             "Scale bands: S1 = coarsest (32 features), S5 = finest (remainder). "
             "Color = max |Cohen's d| for features in that band.",
             ha='center', fontsize=9, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    # Also save PNG version
    png_path = output_path.with_suffix('.png')
    fig.savefig(str(png_path), dpi=200, bbox_inches='tight')
    print(f"Figure saved to {png_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model SAE concept hierarchy heatmap"
    )
    parser.add_argument("--output", type=str,
                        default="output/figures/cross_model_concept_hierarchy.pdf",
                        help="Output figure path")
    parser.add_argument("--max-per-dataset", type=int, default=500)
    args = parser.parse_args()

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
    print("\nFinding common datasets...")
    emb_dirs = {name: emb_dir for name, _, emb_dir in model_configs}
    common_datasets = find_common_datasets(emb_dirs)

    # Meta-features (once)
    print(f"\nComputing meta-features for {len(common_datasets)} datasets...")
    meta_array, loaded_datasets = collect_meta_for_datasets(
        common_datasets, model_configs[0][2], max_per_dataset=args.max_per_dataset
    )
    print(f"  {meta_array.shape} from {len(loaded_datasets)} datasets")

    # Per-model band coverage
    all_band_coverage = {}
    model_names = []

    for display_name, sae_path, emb_dir in model_configs:
        print(f"\nProcessing {display_name}...")
        model, config, _ = load_sae_checkpoint(sae_path)

        # Pool & normalize
        pooled = pool_embeddings_for_datasets(emb_dir, loaded_datasets, args.max_per_dataset)
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

        band_cov = compute_band_coverage(acts[:n_min], meta_array[:n_min], config)
        all_band_coverage[display_name] = band_cov
        model_names.append(display_name)

        n_bands = len(band_cov)
        print(f"  {n_bands} bands, acts shape {acts.shape}")

    # Generate figure
    print("\nGenerating figure...")
    make_figure(all_band_coverage, model_names, Path(args.output))


if __name__ == "__main__":
    main()
