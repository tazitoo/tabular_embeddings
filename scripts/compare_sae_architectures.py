#!/usr/bin/env python3
"""
Head-to-head comparison of SAE architectures on the same embeddings.

Loads two SAE checkpoints (e.g. Archetypal vs Matryoshka-Archetypal),
runs both on pooled TabArena layer-16 embeddings, and produces a structured
comparison covering: basic metrics, monosemanticity, concept coverage,
dictionary overlap, Mat-Arch scale analysis, and concept redundancy.

Usage:
    python scripts/compare_sae_architectures.py \
        --sae-a output/sae_tabarena_sweep/tabpfn_layer16/sae_archetypal_validated.pt \
        --sae-b output/sae_tabarena_sweep/tabpfn_layer16/sae_matryoshka_archetypal_validated.pt \
        --emb-dir output/embeddings/tabarena/tabpfn_layer16_ctx600 \
        --output output/sae_architecture_comparison.json

    # With concept labeling via LLM
    python scripts/compare_sae_architectures.py \
        --sae-a ... --sae-b ... --emb-dir ... \
        --label-concepts --output output/sae_architecture_comparison_labeled.json
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.sparse_autoencoder import (
    SAEConfig,
    SparseAutoencoder,
    compare_dictionaries,
)
from scripts.analyze_sae_concepts_deep import (
    NumpyEncoder,
    RowMetaFeatures,
    compute_column_stats,
    compute_concept_coverage,
    compute_row_meta_features,
    convert_keys_to_native,
    interpret_pattern,
    load_sae_checkpoint,
    collect_raw_activating_samples,
    generate_concept_labels_with_llm,
)
from data.extended_loader import load_tabarena_dataset


# ── Meta-feature names (must match compute_row_meta_features output order) ──

META_NAMES = [
    'missing_rate', 'missing_numeric_rate', 'missing_categorical_rate',
    'numeric_mean_zscore', 'numeric_max_zscore', 'numeric_min_zscore',
    'numeric_std', 'numeric_skewness', 'numeric_kurtosis',
    'numeric_range', 'numeric_iqr', 'frac_zeros', 'frac_negative',
    'frac_positive_outliers', 'frac_negative_outliers',
    'categorical_rarity', 'categorical_modal_frac', 'n_rare_categories',
    'n_unique_categories', 'categorical_entropy',
    'row_entropy', 'row_uniformity', 'n_distinct_values',
    'centroid_distance', 'nearest_neighbor_dist', 'local_density',
    'pca_pc1', 'pca_pc2', 'pca_residual',
    'n_numeric', 'n_categorical', 'n_rows_total', 'n_cols_total',
    'dataset_sparsity', 'numeric_correlation_mean',
    'target_is_minority', 'target_zscore',
]


def meta_features_to_array(mf: RowMetaFeatures) -> List[float]:
    """Convert a RowMetaFeatures instance to a flat list matching META_NAMES."""
    return [
        mf.missing_rate, mf.missing_numeric_rate, mf.missing_categorical_rate,
        mf.numeric_mean_zscore, mf.numeric_max_zscore, mf.numeric_min_zscore,
        mf.numeric_std, mf.numeric_skewness, mf.numeric_kurtosis,
        mf.numeric_range, mf.numeric_iqr, mf.frac_zeros, mf.frac_negative,
        mf.frac_positive_outliers, mf.frac_negative_outliers,
        mf.categorical_rarity, mf.categorical_modal_frac, mf.n_rare_categories,
        mf.n_unique_categories, mf.categorical_entropy,
        mf.row_entropy, mf.row_uniformity, mf.n_distinct_values,
        mf.centroid_distance, mf.nearest_neighbor_dist, mf.local_density,
        mf.pca_pc1, mf.pca_pc2, mf.pca_residual,
        mf.n_numeric, mf.n_categorical, mf.n_rows_total, mf.n_cols_total,
        mf.dataset_sparsity, mf.numeric_correlation_mean,
        mf.target_is_minority, mf.target_zscore,
    ]


def get_train_test_split(datasets: List[str]) -> Tuple[List[str], List[str]]:
    """Deterministic 70/30 train/test split matching sae_tabarena_sweep.py."""
    train, test = [], []
    for ds in datasets:
        h = int(hashlib.md5(ds.encode()).hexdigest(), 16)
        (train if h % 10 < 7 else test).append(ds)
    return train, test


# ── Section 1: Load both SAEs and pool embeddings ───────────────────────────

def load_embeddings_and_normalize(
    emb_dir: Path,
    max_per_dataset: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Pool embeddings, compute train-split normalization, and return centered data.

    Returns:
        pooled: raw pooled embeddings (n_samples, dim)
        train_std: per-dim std from train split (1, dim)
        train_mean: per-dim mean of std-normalized train split (1, dim)
        all_datasets: list of dataset names
        train_datasets: list of train-split dataset names
    """
    all_datasets = sorted([
        f.stem.replace("tabarena_", "")
        for f in emb_dir.glob("tabarena_*.npz")
    ])
    train_datasets, _ = get_train_test_split(all_datasets)

    # Pool training embeddings for normalization stats
    train_embs = []
    for ds in train_datasets:
        path = emb_dir / f"tabarena_{ds}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        emb = data['embeddings'].astype(np.float32)
        if len(emb) > max_per_dataset:
            np.random.seed(42)
            idx = np.random.choice(len(emb), max_per_dataset, replace=False)
            emb = emb[idx]
        train_embs.append(emb)

    train_pooled = np.concatenate(train_embs)
    train_std = train_pooled.std(axis=0, keepdims=True)
    train_std[train_std < 1e-8] = 1.0
    train_norm = train_pooled / train_std
    train_mean = train_norm.mean(axis=0, keepdims=True)

    # Pool all embeddings (for analysis, not just train)
    all_embs = []
    for ds in all_datasets:
        path = emb_dir / f"tabarena_{ds}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        emb = data['embeddings'].astype(np.float32)
        if len(emb) > max_per_dataset:
            np.random.seed(42)
            idx = np.random.choice(len(emb), max_per_dataset, replace=False)
            emb = emb[idx]
        all_embs.append(emb)

    pooled = np.concatenate(all_embs)
    return pooled, train_std, train_mean, all_datasets, train_datasets


def compute_activations(
    model: SparseAutoencoder,
    embeddings: np.ndarray,
    train_std: np.ndarray = None,  # Kept for backward compat, but unused
    train_mean: np.ndarray = None,  # Kept for backward compat, but unused
) -> np.ndarray:
    """
    Compute SAE activations from raw embeddings.

    The SAE's internal BatchNorm layer applies learned normalization automatically,
    so we pass raw embeddings directly. train_std/train_mean args are kept for
    backward compatibility but are no longer used.
    """
    with torch.no_grad():
        x = torch.tensor(embeddings, dtype=torch.float32)
        h = model.encode(x).numpy()
    return h


# ── Section 2: Basic dictionary metrics ─────────────────────────────────────

def compute_basic_metrics(
    activations: np.ndarray,
    config: SAEConfig,
    alive_threshold: float = 0.001,
) -> Dict:
    """Compute alive features, L0, mean/max activation."""
    feature_max = activations.max(axis=0)
    alive_mask = feature_max > alive_threshold
    alive_count = int(alive_mask.sum())

    l0_per_sample = (activations > 0).sum(axis=1)
    mean_activation = activations[activations > 0].mean() if (activations > 0).any() else 0.0
    max_activation = float(activations.max())

    return {
        'hidden_dim': config.hidden_dim,
        'alive_features': alive_count,
        'alive_frac': alive_count / config.hidden_dim,
        'dead_features': config.hidden_dim - alive_count,
        'l0_mean': float(l0_per_sample.mean()),
        'l0_std': float(l0_per_sample.std()),
        'mean_activation': float(mean_activation),
        'max_activation': max_activation,
        'alive_indices': np.where(alive_mask)[0].tolist(),
    }


# ── Section 3: Feature monosemanticity via effect sizes ─────────────────────

def compute_feature_effects(
    activations: np.ndarray,
    meta_array: np.ndarray,
    alive_indices: List[int],
    top_k_samples: int = 100,
) -> Dict:
    """
    For each alive feature, compute Cohen's d effect sizes against all meta-features.

    Returns per-feature analysis with effect sizes and dominant patterns.
    """
    baseline_means = meta_array.mean(axis=0)
    baseline_stds = meta_array.std(axis=0)

    feature_analysis = {}

    for feat_idx in alive_indices:
        feat_acts = activations[:, feat_idx]
        top_indices = np.argsort(feat_acts)[-top_k_samples:]
        top_acts = feat_acts[top_indices]

        if top_acts.max() < 0.01:
            continue

        top_meta = meta_array[top_indices]
        top_means = top_meta.mean(axis=0)

        effect_sizes = {}
        for i, name in enumerate(META_NAMES):
            if baseline_stds[i] > 1e-8:
                d = (top_means[i] - baseline_means[i]) / baseline_stds[i]
                effect_sizes[name] = float(d)
            else:
                effect_sizes[name] = 0.0

        sorted_effects = sorted(effect_sizes.items(), key=lambda x: -abs(x[1]))
        dominant = [(n, d) for n, d in sorted_effects if abs(d) > 0.3]

        feature_analysis[feat_idx] = {
            'mean_activation': float(top_acts.mean()),
            'max_activation': float(top_acts.max()),
            'effect_sizes': effect_sizes,
            'dominant_patterns': dominant,
            'interpretation': interpret_pattern(dominant),
        }

    return feature_analysis


def compute_monosemanticity(feature_analysis: Dict, threshold: float = 0.5) -> Dict:
    """
    Compute monosemanticity metrics from feature effect sizes.

    Lower mean_strong_effects = more monosemantic = more interpretable.
    """
    strong_counts = []
    max_effects = []

    for fid, analysis in feature_analysis.items():
        effects = analysis['effect_sizes']
        n_strong = sum(1 for d in effects.values() if abs(d) > threshold)
        strong_counts.append(n_strong)
        max_effects.append(max(abs(d) for d in effects.values()))

    return {
        'mean_strong_effects': float(np.mean(strong_counts)) if strong_counts else 0.0,
        'median_strong_effects': float(np.median(strong_counts)) if strong_counts else 0.0,
        'mean_max_effect': float(np.mean(max_effects)) if max_effects else 0.0,
        'frac_monosemantic': float(np.mean([c <= 3 for c in strong_counts])) if strong_counts else 0.0,
    }


# ── Section 4: Concept coverage comparison ──────────────────────────────────
# Reuses compute_concept_coverage from analyze_sae_concepts_deep.py


# ── Section 5: Dictionary overlap ───────────────────────────────────────────
# Reuses compare_dictionaries from analysis/sparse_autoencoder.py


# ── Section 6: Mat-Arch scale analysis ──────────────────────────────────────

def compute_scale_analysis(
    activations: np.ndarray,
    meta_array: np.ndarray,
    config: SAEConfig,
    alive_threshold: float = 0.001,
    top_k_samples: int = 100,
) -> Optional[Dict]:
    """
    Analyze features at each Matryoshka scale band.

    Only applicable to matryoshka_archetypal SAEs. Returns per-band metrics:
    alive count, monosemanticity, coverage, activation magnitude.
    """
    if config.sparsity_type != 'matryoshka_archetypal':
        return None

    mat_dims = [d for d in config.matryoshka_dims if d <= config.hidden_dim]
    # Build scale bands: [0, 32), [32, 64), ..., [last_dim, hidden_dim)
    boundaries = [0] + mat_dims
    if boundaries[-1] < config.hidden_dim:
        boundaries.append(config.hidden_dim)

    baseline_means = meta_array.mean(axis=0)
    baseline_stds = meta_array.std(axis=0)

    scale_results = {}
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        band_label = f"[{start},{end})"
        band_indices = list(range(start, end))

        # Alive features in this band
        band_acts = activations[:, start:end]
        band_max = band_acts.max(axis=0)
        alive_mask = band_max > alive_threshold
        alive_count = int(alive_mask.sum())
        alive_indices_local = np.where(alive_mask)[0]

        # Mean activation magnitude for alive features
        alive_acts = band_acts[:, alive_mask]
        if alive_acts.size > 0:
            mean_act = float(alive_acts[alive_acts > 0].mean()) if (alive_acts > 0).any() else 0.0
        else:
            mean_act = 0.0

        # Per-feature effect sizes in this band
        band_feature_analysis = {}
        for local_idx in alive_indices_local:
            global_idx = start + local_idx
            feat_acts = activations[:, global_idx]
            top_indices = np.argsort(feat_acts)[-top_k_samples:]
            top_acts = feat_acts[top_indices]

            if top_acts.max() < 0.01:
                continue

            top_meta = meta_array[top_indices]
            top_means = top_meta.mean(axis=0)

            effect_sizes = {}
            for j, name in enumerate(META_NAMES):
                if baseline_stds[j] > 1e-8:
                    d = (top_means[j] - baseline_means[j]) / baseline_stds[j]
                    effect_sizes[name] = float(d)
                else:
                    effect_sizes[name] = 0.0

            band_feature_analysis[global_idx] = {'effect_sizes': effect_sizes}

        # Monosemanticity for this band
        strong_counts = []
        for fid, analysis in band_feature_analysis.items():
            n_strong = sum(1 for d in analysis['effect_sizes'].values() if abs(d) > 0.5)
            strong_counts.append(n_strong)
        mean_effects = float(np.mean(strong_counts)) if strong_counts else 0.0

        # Coverage for this band
        band_coverage = compute_concept_coverage(band_feature_analysis, META_NAMES)
        well_covered = [name for name, cov in band_coverage.items() if cov['coverage_score'] > 1.0]

        scale_results[band_label] = {
            'start': start,
            'end': end,
            'n_features': end - start,
            'alive_features': alive_count,
            'alive_frac': alive_count / (end - start) if (end - start) > 0 else 0.0,
            'mean_activation': mean_act,
            'mean_strong_effects': mean_effects,
            'n_well_covered_meta': len(well_covered),
            'well_covered_meta': well_covered,
        }

    return scale_results


# ── Section 7: Concept redundancy ───────────────────────────────────────────

def compute_redundancy(
    model: SparseAutoencoder,
    alive_indices: List[int],
    similarity_threshold: float = 0.8,
) -> Dict:
    """
    Compute intra-dictionary redundancy via cosine similarity of decoder weights.
    """
    dictionary = model.get_dictionary()  # (hidden_dim, input_dim)
    alive_dict = dictionary[alive_indices]

    # Normalize
    norms = np.linalg.norm(alive_dict, axis=1, keepdims=True)
    alive_norm = alive_dict / (norms + 1e-8)

    # Pairwise cosine similarity
    sim_matrix = alive_norm @ alive_norm.T
    np.fill_diagonal(sim_matrix, 0)

    # Count near-duplicate pairs
    n_alive = len(alive_indices)
    upper_tri = sim_matrix[np.triu_indices(n_alive, k=1)]
    n_near_duplicates = int((upper_tri > similarity_threshold).sum())
    n_pairs = len(upper_tri)

    # Count features involved in at least one near-duplicate pair
    dup_mask = sim_matrix > similarity_threshold
    n_features_with_dup = int((dup_mask.any(axis=1)).sum())

    return {
        'n_alive': n_alive,
        'n_near_duplicate_pairs': n_near_duplicates,
        'n_total_pairs': n_pairs,
        'frac_duplicate_pairs': n_near_duplicates / n_pairs if n_pairs > 0 else 0.0,
        'n_features_with_duplicate': n_features_with_dup,
        'frac_features_with_duplicate': n_features_with_dup / n_alive if n_alive > 0 else 0.0,
        'mean_pairwise_similarity': float(upper_tri.mean()) if len(upper_tri) > 0 else 0.0,
        'max_pairwise_similarity': float(upper_tri.max()) if len(upper_tri) > 0 else 0.0,
    }


# ── Meta-feature collection (shared across both SAEs) ──────────────────────

def collect_meta_features(
    emb_dir: Path,
    all_datasets: List[str],
    max_per_dataset: int = 500,
) -> Tuple[np.ndarray, int]:
    """
    Load tabular data and compute row meta-features for all datasets.

    Returns:
        meta_array: (n_samples, n_meta_features)
        n_datasets_loaded: number of datasets successfully loaded
    """
    all_meta = []
    n_loaded = 0

    for ds_name in all_datasets:
        emb_path = emb_dir / f"tabarena_{ds_name}.npz"
        if not emb_path.exists():
            continue

        emb_data = np.load(emb_path, allow_pickle=True)
        n_emb = len(emb_data['embeddings'])

        # Determine sample indices (must match pooling logic)
        if n_emb > max_per_dataset:
            np.random.seed(42)
            sample_indices = np.random.choice(n_emb, max_per_dataset, replace=False)
        else:
            sample_indices = np.arange(n_emb)

        try:
            X, y, dataset_info = load_tabarena_dataset(ds_name)
            df = X if hasattr(X, 'iloc') else __import__('pandas').DataFrame(X)
            df = df.iloc[sample_indices].reset_index(drop=True)
            y_subset = y[sample_indices] if y is not None else None

            numeric_cols, categorical_cols, col_stats, dataset_stats = compute_column_stats(df)
            meta_features = compute_row_meta_features(
                df, y_subset, numeric_cols, categorical_cols, col_stats, dataset_stats
            )
            all_meta.extend([meta_features_to_array(m) for m in meta_features])
            n_loaded += 1
            print(f"  {ds_name}: {len(meta_features)} rows")
        except Exception as e:
            print(f"  Skipping {ds_name}: {e}")

    meta_array = np.array(all_meta)
    return meta_array, n_loaded


# ── Output formatting ───────────────────────────────────────────────────────

def print_comparison(
    name_a: str,
    name_b: str,
    metrics_a: Dict,
    metrics_b: Dict,
    mono_a: Dict,
    mono_b: Dict,
    coverage_a: Dict,
    coverage_b: Dict,
    overlap: Dict,
    redundancy_a: Dict,
    redundancy_b: Dict,
    scale_analysis: Optional[Dict],
):
    """Print formatted comparison table to stdout."""
    w = 18  # Column width

    print()
    print("=" * 70)
    print(f"{'ARCHETYPAL vs MATRYOSHKA-ARCHETYPAL COMPARISON':^70}")
    print("=" * 70)

    # Basic metrics table
    print(f"\n{'':30s} {name_a:>{w}s}    {name_b:>{w}s}")
    print("-" * 70)

    rows = [
        ("Hidden dim", metrics_a['hidden_dim'], metrics_b['hidden_dim'], "d"),
        ("Alive features", f"{metrics_a['alive_features']} ({metrics_a['alive_frac']:.0%})",
         f"{metrics_b['alive_features']} ({metrics_b['alive_frac']:.0%})", "s"),
        ("L0 sparsity", f"{metrics_a['l0_mean']:.1f}", f"{metrics_b['l0_mean']:.1f}", "s"),
        ("Mean activation", f"{metrics_a['mean_activation']:.3f}",
         f"{metrics_b['mean_activation']:.3f}", "s"),
        ("Effects/feature (|d|>0.5)", f"{mono_a['mean_strong_effects']:.1f}",
         f"{mono_b['mean_strong_effects']:.1f}", "s"),
        ("Frac monosemantic (<=3)", f"{mono_a['frac_monosemantic']:.0%}",
         f"{mono_b['frac_monosemantic']:.0%}", "s"),
    ]

    # Coverage counts
    well_a = sum(1 for c in coverage_a.values() if c['coverage_score'] > 1.0)
    well_b = sum(1 for c in coverage_b.values() if c['coverage_score'] > 1.0)
    poor_a = sum(1 for c in coverage_a.values() if c['coverage_score'] < 0.5)
    poor_b = sum(1 for c in coverage_b.values() if c['coverage_score'] < 0.5)
    rows.append(("Well-covered meta (d>1)", well_a, well_b, "d"))
    rows.append(("Poorly-covered meta (d<0.5)", poor_a, poor_b, "d"))

    # Redundancy
    rows.append(("Near-dup pairs (cos>0.8)", f"{redundancy_a['frac_duplicate_pairs']:.2%}",
                 f"{redundancy_b['frac_duplicate_pairs']:.2%}", "s"))
    rows.append(("Features w/ duplicate", f"{redundancy_a['frac_features_with_duplicate']:.0%}",
                 f"{redundancy_b['frac_features_with_duplicate']:.0%}", "s"))
    rows.append(("Mean pairwise sim", f"{redundancy_a['mean_pairwise_similarity']:.3f}",
                 f"{redundancy_b['mean_pairwise_similarity']:.3f}", "s"))

    for label, val_a, val_b, fmt in rows:
        if fmt == "d":
            print(f"  {label:28s} {val_a:>{w}}    {val_b:>{w}}")
        else:
            print(f"  {label:28s} {str(val_a):>{w}s}    {str(val_b):>{w}s}")

    # Dictionary overlap
    print(f"\n{'DICTIONARY OVERLAP':^70}")
    print("-" * 70)
    print(f"  {name_a} → {name_b} coverage@0.7: {overlap['coverage_a_at_0.7']:.0%}"
          f"  ({name_b} subsumes {name_a}?)")
    print(f"  {name_b} → {name_a} coverage@0.7: {overlap['coverage_b_at_0.7']:.0%}"
          f"  ({name_a} has unique concepts?)")
    print(f"  Bidirectional matches: {overlap['bidirectional_matches']}"
          f" ({overlap['bidirectional_rate']:.0%})")

    # Coverage comparison details
    print(f"\n{'COVERAGE COMPARISON':^70}")
    print("-" * 70)
    print(f"  {'Meta-feature':28s} {'A score':>8s}  {'B score':>8s}  {'Winner':>8s}")
    print("  " + "-" * 56)
    for name in META_NAMES:
        score_a = coverage_a[name]['coverage_score']
        score_b = coverage_b[name]['coverage_score']
        winner = name_a if score_a > score_b + 0.1 else (name_b if score_b > score_a + 0.1 else "tie")
        if score_a > 0.5 or score_b > 0.5:
            print(f"  {name:28s} {score_a:8.2f}  {score_b:8.2f}  {winner:>8s}")

    # Scale analysis (Mat-Arch only)
    if scale_analysis:
        print(f"\n{'MAT-ARCH SCALE ANALYSIS':^70}")
        print("-" * 70)
        print(f"  {'Scale':15s} {'Alive':>8s}  {'Effects':>8s}  {'Covered':>8s}  Top concepts")
        print("  " + "-" * 66)
        for band, info in scale_analysis.items():
            concepts = ", ".join(info['well_covered_meta'][:3]) if info['well_covered_meta'] else "-"
            print(f"  {band:15s} {info['alive_features']:>5d}/{info['n_features']:<3d}"
                  f"  {info['mean_strong_effects']:8.1f}"
                  f"  {info['n_well_covered_meta']:8d}  {concepts}")

    print("\n" + "=" * 70)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Head-to-head comparison of two SAE architectures"
    )
    parser.add_argument("--sae-a", type=str, required=True,
                        help="Path to first SAE checkpoint (e.g. archetypal)")
    parser.add_argument("--sae-b", type=str, required=True,
                        help="Path to second SAE checkpoint (e.g. matryoshka_archetypal)")
    parser.add_argument("--emb-dir", type=str, required=True,
                        help="Directory with pooled TabArena embeddings (tabpfn_layer16_ctx600)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for structured comparison")
    parser.add_argument("--max-per-dataset", type=int, default=500,
                        help="Max samples per dataset (default: 500)")
    parser.add_argument("--label-concepts", action="store_true",
                        help="Generate concept labels using LLM (expensive)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM, use statistical labels only")
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    if not emb_dir.exists():
        print(f"Error: embedding directory not found: {emb_dir}")
        sys.exit(1)

    # ── Step 1: Load SAEs ────────────────────────────────────────────────
    print("Loading SAE checkpoints...")
    model_a, config_a, ckpt_a = load_sae_checkpoint(Path(args.sae_a))
    model_b, config_b, ckpt_b = load_sae_checkpoint(Path(args.sae_b))

    name_a = config_a.sparsity_type
    name_b = config_b.sparsity_type
    print(f"  A: {name_a} (hidden={config_a.hidden_dim}, topk={config_a.topk})")
    print(f"  B: {name_b} (hidden={config_b.hidden_dim}, topk={config_b.topk})")

    # ── Step 2: Load and normalize embeddings ────────────────────────────
    print(f"\nLoading embeddings from {emb_dir}...")
    pooled, train_std, train_mean, all_datasets, train_datasets = \
        load_embeddings_and_normalize(emb_dir, max_per_dataset=args.max_per_dataset)
    print(f"  Pooled: {pooled.shape[0]} samples, {pooled.shape[1]} dims")
    print(f"  Datasets: {len(all_datasets)} total, {len(train_datasets)} train")

    # ── Step 3: Compute activations for both SAEs ────────────────────────
    print("\nComputing activations...")
    acts_a = compute_activations(model_a, pooled, train_std, train_mean)
    acts_b = compute_activations(model_b, pooled, train_std, train_mean)
    print(f"  A: {acts_a.shape}")
    print(f"  B: {acts_b.shape}")

    # ── Step 4: Basic metrics ────────────────────────────────────────────
    print("\nComputing basic metrics...")
    metrics_a = compute_basic_metrics(acts_a, config_a)
    metrics_b = compute_basic_metrics(acts_b, config_b)
    print(f"  A alive: {metrics_a['alive_features']}/{config_a.hidden_dim}")
    print(f"  B alive: {metrics_b['alive_features']}/{config_b.hidden_dim}")

    # ── Step 5: Collect meta-features (shared, expensive) ────────────────
    print("\nCollecting row meta-features across all datasets...")
    meta_array, n_loaded = collect_meta_features(
        emb_dir, all_datasets, max_per_dataset=args.max_per_dataset
    )
    print(f"  Meta-feature matrix: {meta_array.shape} from {n_loaded} datasets")

    if len(meta_array) != len(pooled):
        # Some datasets failed to load tabular data — truncate activations to match
        print(f"  Warning: meta_array ({len(meta_array)}) != pooled ({len(pooled)}), "
              f"using min of both")
        n_min = min(len(meta_array), len(acts_a), len(acts_b))
        meta_array = meta_array[:n_min]
        acts_a = acts_a[:n_min]
        acts_b = acts_b[:n_min]

    # ── Step 6: Feature effect sizes + monosemanticity ───────────────────
    print("\nComputing feature effect sizes...")
    feat_analysis_a = compute_feature_effects(
        acts_a, meta_array, metrics_a['alive_indices']
    )
    feat_analysis_b = compute_feature_effects(
        acts_b, meta_array, metrics_b['alive_indices']
    )
    print(f"  A: {len(feat_analysis_a)} features analyzed")
    print(f"  B: {len(feat_analysis_b)} features analyzed")

    mono_a = compute_monosemanticity(feat_analysis_a)
    mono_b = compute_monosemanticity(feat_analysis_b)

    # ── Step 7: Concept coverage ─────────────────────────────────────────
    print("\nComputing concept coverage...")
    coverage_a = compute_concept_coverage(feat_analysis_a, META_NAMES)
    coverage_b = compute_concept_coverage(feat_analysis_b, META_NAMES)

    # ── Step 8: Dictionary overlap ───────────────────────────────────────
    print("\nComputing dictionary overlap...")
    dict_a = model_a.get_dictionary()
    dict_b = model_b.get_dictionary()
    overlap = compare_dictionaries(dict_a, dict_b)
    # Remove large similarity matrix from JSON output
    overlap_json = {k: v for k, v in overlap.items() if k != 'similarity_matrix'}

    # ── Step 9: Scale analysis (Mat-Arch only) ───────────────────────────
    scale_results = None
    if config_b.sparsity_type == 'matryoshka_archetypal':
        print("\nComputing Mat-Arch scale analysis...")
        scale_results = compute_scale_analysis(
            acts_b, meta_array, config_b
        )
    elif config_a.sparsity_type == 'matryoshka_archetypal':
        print("\nComputing Mat-Arch scale analysis...")
        scale_results = compute_scale_analysis(
            acts_a, meta_array, config_a
        )

    # ── Step 10: Redundancy ──────────────────────────────────────────────
    print("\nComputing concept redundancy...")
    redundancy_a = compute_redundancy(model_a, metrics_a['alive_indices'])
    redundancy_b = compute_redundancy(model_b, metrics_b['alive_indices'])

    # ── Print summary ────────────────────────────────────────────────────
    print_comparison(
        name_a, name_b,
        metrics_a, metrics_b,
        mono_a, mono_b,
        coverage_a, coverage_b,
        overlap_json,
        redundancy_a, redundancy_b,
        scale_results,
    )

    # ── Step 11: Optional concept labeling ───────────────────────────────
    labels_a, labels_b = None, None
    if args.label_concepts:
        print("\n" + "=" * 70)
        print("CONCEPT LABELING")
        print("=" * 70)

        for label, model, config, feat_analysis, name in [
            ("A", model_a, config_a, feat_analysis_a, name_a),
            ("B", model_b, config_b, feat_analysis_b, name_b),
        ]:
            print(f"\nCollecting raw samples for {name}...")
            feature_ids = list(feat_analysis.keys())
            raw_sample_data = collect_raw_activating_samples(
                model=model,
                datasets=all_datasets,
                train_std=train_std,
                train_mean=train_mean,
                feature_ids=feature_ids,
                emb_dir=emb_dir,
            )

            concept_labels = generate_concept_labels_with_llm(
                raw_sample_data=raw_sample_data,
                feature_analysis=feat_analysis,
                use_llm=not args.no_llm,
            )

            if label == "A":
                labels_a = concept_labels
            else:
                labels_b = concept_labels

            # Print top labels
            sorted_labels = sorted(
                concept_labels.items(),
                key=lambda x: (-x[1]['confidence'], x[0])
            )
            print(f"\nTop 20 concepts for {name}:")
            for feat_id, info in sorted_labels[:20]:
                method_tag = "LLM" if info['method'] == 'llm' else "STAT"
                print(f"  F{feat_id:4d}: {info['label']:30s} [{method_tag}]")

    # ── Save JSON output ─────────────────────────────────────────────────
    if args.output:
        report = {
            'sae_a': {
                'path': args.sae_a,
                'type': name_a,
                'config': {
                    'hidden_dim': config_a.hidden_dim,
                    'topk': config_a.topk,
                    'sparsity_type': config_a.sparsity_type,
                },
                'basic_metrics': {k: v for k, v in metrics_a.items() if k != 'alive_indices'},
                'monosemanticity': mono_a,
                'coverage': coverage_a,
                'redundancy': redundancy_a,
            },
            'sae_b': {
                'path': args.sae_b,
                'type': name_b,
                'config': {
                    'hidden_dim': config_b.hidden_dim,
                    'topk': config_b.topk,
                    'sparsity_type': config_b.sparsity_type,
                },
                'basic_metrics': {k: v for k, v in metrics_b.items() if k != 'alive_indices'},
                'monosemanticity': mono_b,
                'coverage': coverage_b,
                'redundancy': redundancy_b,
            },
            'dictionary_overlap': overlap_json,
            'scale_analysis': scale_results,
            'n_datasets': len(all_datasets),
            'n_samples': len(meta_array),
        }
        if labels_a is not None:
            report['sae_a']['concept_labels'] = labels_a
        if labels_b is not None:
            report['sae_b']['concept_labels'] = labels_b

        report_clean = convert_keys_to_native(report)
        with open(args.output, 'w') as f:
            json.dump(report_clean, f, indent=2, cls=NumpyEncoder)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
