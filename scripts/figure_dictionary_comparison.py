#!/usr/bin/env python3
"""
Figure: Direct dictionary comparison across models.

Compares SAE dictionaries by clustering features based on their activation
patterns (not meta-features). Features from different models that fire on
the same samples encode the same concept.

One heatmap per Matryoshka scale band showing which concepts each model encodes.

Usage:
    python scripts/figure_dictionary_comparison.py \
        --output output/figures/dictionary_comparison.pdf
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_sae_architectures import (
    compute_activations,
    get_train_test_split,
)
from scripts.compare_sae_cross_model import (
    DEFAULT_MODELS,
    collect_meta_for_datasets,
    find_common_datasets,
    pool_embeddings_for_datasets,
    sae_sweep_dir,
)
from scripts.analyze_sae_concepts_deep import load_sae_checkpoint

BAND_LABELS = ['S1 [0,32)', 'S2 [32,64)', 'S3 [64,128)', 'S4 [128,256)', 'S5 [256,N)']


def get_band_activations(
    activations: np.ndarray,
    config,
    alive_threshold: float = 0.001,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split activations into Matryoshka scale bands.

    Returns list of (band_activations, alive_mask) per band.
    band_activations: (n_samples, n_alive_in_band)
    """
    mat_dims = [d for d in config.matryoshka_dims if d <= config.hidden_dim]
    boundaries = [0] + mat_dims
    if boundaries[-1] < config.hidden_dim:
        boundaries.append(config.hidden_dim)

    bands = []
    for bi in range(len(boundaries) - 1):
        start, end = boundaries[bi], boundaries[bi + 1]
        band_acts = activations[:, start:end]
        alive_mask = band_acts.max(axis=0) > alive_threshold
        bands.append((band_acts[:, alive_mask], alive_mask))

    return bands


def cluster_features_across_models(
    model_band_acts: Dict[str, np.ndarray],
    model_names: List[str],
    corr_threshold: float = 0.5,
    min_activation_rate: float = 0.001,
    max_features_per_model: int = 500,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Cluster features across models by activation correlation.

    Features are subsampled per model to max_features_per_model to prevent
    any single model from dominating clusters and to keep the correlation
    matrix tractable.

    Returns:
        presence: (n_clusters, n_models) binary matrix (1.0 if model present)
        cluster_labels: label for each cluster (e.g. "C1", "C2")
        feature_labels: which model each feature came from
    """
    rng = np.random.RandomState(42)

    # Collect all alive feature activation vectors
    all_vectors = []
    feature_model = []

    for name in model_names:
        acts = model_band_acts[name]
        if acts.shape[1] == 0:
            continue
        # Filter: feature must fire on at least min_activation_rate of samples
        fire_rate = (acts > 0).mean(axis=0)
        active_indices = np.where(fire_rate >= min_activation_rate)[0]

        # Subsample if too many alive features from this model
        if len(active_indices) > max_features_per_model:
            print(f"    {name}: {len(active_indices)} alive, "
                  f"subsampled to {max_features_per_model}")
            active_indices = rng.choice(
                active_indices, max_features_per_model, replace=False
            )

        for fi in active_indices:
            all_vectors.append(acts[:, fi])
            feature_model.append(name)

    if len(all_vectors) < 2:
        return np.zeros((0, len(model_names))), [], feature_model

    # Pearson correlation matrix between all features
    X = np.column_stack(all_vectors)  # (n_samples, n_features)
    X_centered = X - X.mean(axis=0, keepdims=True)
    stds = X_centered.std(axis=0, keepdims=True)
    stds[stds < 1e-10] = 1.0
    X_std = X_centered / stds
    corr = (X_std.T @ X_std) / X.shape[0]
    np.fill_diagonal(corr, 1.0)

    # Distance = 1 - |correlation|
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0, 1)

    # Agglomerative clustering
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')
    labels = fcluster(Z, t=1.0 - corr_threshold, criterion='distance')

    n_clusters = labels.max()

    # Build binary presence matrix: 1.0 if model has any feature in cluster
    presence = np.zeros((n_clusters, len(model_names)))

    for ci in range(1, n_clusters + 1):
        members = np.where(labels == ci)[0]
        for mi, model_name in enumerate(model_names):
            model_members = [m for m in members if feature_model[m] == model_name]
            if model_members:
                presence[ci - 1, mi] = 1.0

    # Sort clusters: by number of models that have them (universal first),
    # then by mean presence strength
    n_models_per_cluster = (presence > 0).sum(axis=1)
    mean_strength = presence.mean(axis=1)
    sort_idx = np.lexsort((-mean_strength, -n_models_per_cluster))
    presence = presence[sort_idx]

    cluster_labels = [f'C{i+1}' for i in range(n_clusters)]

    return presence, cluster_labels, feature_model


def pairwise_jaccard(presence: np.ndarray) -> np.ndarray:
    """Compute pairwise Jaccard similarity between models from binary presence matrix.

    Args:
        presence: (n_clusters, n_models) binary matrix

    Returns:
        (n_models, n_models) symmetric Jaccard similarity matrix
    """
    n_models = presence.shape[1]
    binary = (presence > 0).astype(float)
    sim = np.eye(n_models)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            intersection = (binary[:, i] * binary[:, j]).sum()
            union = ((binary[:, i] + binary[:, j]) > 0).sum()
            sim[i, j] = sim[j, i] = intersection / union if union > 0 else 0.0
    return sim


def make_figure(
    all_presence: List[np.ndarray],
    model_names: List[str],
    output_path: Path,
):
    """Pairwise Jaccard similarity heatmap per Matryoshka scale band.

    Produces a compact, full-width figure suitable for Figure 1 of a paper.
    Each panel is a 4x4 symmetric heatmap showing concept overlap between
    model pairs at one scale band, plus an 'Overall' panel.
    """
    valid_bands = [(i, p) for i, p in enumerate(all_presence) if p.shape[0] > 0]
    if not valid_bands:
        print("No clusters found!")
        return

    n_models = len(model_names)

    # Compute per-band Jaccard matrices
    band_sims = []
    for band_idx, presence in valid_bands:
        band_sims.append((band_idx, pairwise_jaccard(presence)))

    # Overall: pool presence across all bands
    overall_presence = np.vstack([p for _, p in valid_bands])
    overall_sim = pairwise_jaccard(overall_presence)

    # Figure: 2 rows x 3 cols — Overall + 5 bands
    panels = [(-1, overall_sim)] + band_sims
    panel_titles = ['Overall'] + [BAND_LABELS[bi] for bi, _ in band_sims]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(7.0, 4.8),
        gridspec_kw={'wspace': 0.4, 'hspace': 0.5},
    )
    axes = axes.flatten()

    cmap = plt.cm.YlOrRd
    vmin, vmax = 0.0, 0.7  # cap at 0.7 so mid-range values get more color contrast

    for ax_idx, ((_, sim), title) in enumerate(zip(panels, panel_titles)):
        ax = axes[ax_idx]
        # Mask upper triangle + diagonal
        mask = np.tri(n_models, k=-1, dtype=bool)
        masked = np.where(mask, sim, np.nan)
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='nearest', aspect='equal')

        # Annotate lower triangle only
        for i in range(n_models):
            for j in range(i):
                val = sim[i, j]
                color = 'white' if val > 0.45 else 'black'
                ax.text(j, i, f'.{int(val*100):02d}', ha='center', va='center',
                        fontsize=6.5, color=color)

        # Crop to lower triangle: skip top row, rightmost column
        ax.set_xlim(-0.5, n_models - 1.5)
        ax.set_ylim(n_models - 0.5, 0.5)
        ax.set_xticks(range(n_models - 1))
        ax.set_yticks(range(1, n_models))
        ax.set_xticklabels(model_names[:-1], fontsize=6.5, rotation=45, ha='right')

        if ax_idx % 3 == 0:
            ax.set_yticklabels(model_names[1:], fontsize=6.5)
        else:
            ax.set_yticklabels([])

        ax.set_title(title, fontsize=7.5, fontweight='bold', pad=3)
        ax.tick_params(length=0)


    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    fig.savefig(str(output_path.with_suffix('.png')), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path.with_suffix('.png')}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Dictionary comparison figure")
    parser.add_argument("--output", type=str,
                        default="output/figures/dictionary_comparison.pdf")
    parser.add_argument("--max-per-dataset", type=int, default=500)
    parser.add_argument("--corr-threshold", type=float, default=0.15,
                        help="Correlation threshold for clustering (higher = stricter)")
    parser.add_argument("--max-features-per-model", type=int, default=500,
                        help="Max alive features per model per band (default: 500)")
    args = parser.parse_args()

    base_sae = sae_sweep_dir()
    base_emb = PROJECT_ROOT / "output" / "embeddings" / "tabarena"

    # Resolve models
    model_configs = []
    for display_name, sweep_dir, emb_dir_name in DEFAULT_MODELS:
        sae_path = base_sae / sweep_dir / "sae_matryoshka_archetypal_validated.pt"
        emb_dir = base_emb / emb_dir_name
        if sae_path.exists() and emb_dir.exists():
            model_configs.append((display_name, sae_path, emb_dir))
    model_names = [m[0] for m in model_configs]
    print(f"Models: {model_names}")

    # Common datasets
    emb_dirs = {name: emb_dir for name, _, emb_dir in model_configs}
    common_datasets = find_common_datasets(emb_dirs)

    # We don't need meta-features — just the common dataset list for alignment
    # Find which datasets actually load (use first model's emb dir for reference)
    ref_emb_dir = model_configs[0][2]
    loaded_datasets = []
    for ds in common_datasets:
        if (ref_emb_dir / f"tabarena_{ds}.npz").exists():
            loaded_datasets.append(ds)
    print(f"Using {len(loaded_datasets)} datasets")

    # Load all models and compute activations
    model_activations = {}
    model_configs_loaded = {}

    for display_name, sae_path, emb_dir in model_configs:
        print(f"\nLoading {display_name}...")
        model, config, _ = load_sae_checkpoint(sae_path)
        model_configs_loaded[display_name] = config

        pooled = pool_embeddings_for_datasets(emb_dir, loaded_datasets, args.max_per_dataset)

        # Normalize
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
        model_activations[display_name] = acts
        print(f"  Activations: {acts.shape}")

    # Align sample counts
    n_samples = min(v.shape[0] for v in model_activations.values())
    for name in model_names:
        model_activations[name] = model_activations[name][:n_samples]
    print(f"\nAligned to {n_samples} samples")

    # Split into bands and cluster
    model_bands = {}
    for name in model_names:
        bands = get_band_activations(model_activations[name], model_configs_loaded[name])
        model_bands[name] = bands
        print(f"  {name}: {len(bands)} bands, alive per band: "
              f"{[b[0].shape[1] for b in bands]}")

    n_bands = min(len(model_bands[name]) for name in model_names)

    all_presence = []
    for bi in range(n_bands):
        print(f"\nClustering band {bi} ({BAND_LABELS[bi]})...")
        band_acts = {}
        for name in model_names:
            band_acts[name] = model_bands[name][bi][0]  # (n_samples, n_alive)

        presence, cluster_labels, feature_model = cluster_features_across_models(
            band_acts, model_names, corr_threshold=args.corr_threshold,
            max_features_per_model=args.max_features_per_model,
        )

        n_clusters = presence.shape[0]
        n_universal = (presence > 0).all(axis=1).sum() if n_clusters > 0 else 0
        per_model = [(presence[:, mi] > 0).sum() for mi in range(len(model_names))]
        print(f"  {n_clusters} clusters, {n_universal} universal")
        print(f"  Per model: {dict(zip(model_names, per_model))}")

        # Pairwise Jaccard for this band
        sim = pairwise_jaccard(presence)
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                print(f"    {model_names[i]}-{model_names[j]}: {sim[i,j]:.3f}")

        all_presence.append(presence)

    # Generate figure
    print("\nGenerating figure...")
    make_figure(all_presence, model_names, Path(args.output))


if __name__ == "__main__":
    main()
