#!/usr/bin/env python3
"""
Deep analysis of SAE concepts: trace back to original tabular data patterns.

Key question: What properties of a tabular row cause each SAE feature to fire?
Since concepts are universal across datasets, we look for meta-patterns:
- Missing value rates
- Numeric outliers
- Categorical rarity
- Row complexity/entropy
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def convert_keys_to_native(obj):
    """Recursively convert dict keys from numpy types to native Python types."""
    if isinstance(obj, dict):
        return {
            (int(k) if isinstance(k, np.integer) else str(k) if not isinstance(k, (str, int, float, bool, type(None))) else k): convert_keys_to_native(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_keys_to_native(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.sparse_autoencoder import SparseAutoencoder, SAEConfig
from data.extended_loader import load_tabarena_dataset


@dataclass
class RowMetaFeatures:
    """
    Comprehensive meta-features computed for each row (dataset-agnostic).

    These capture universal tabular patterns that should generalize across datasets.
    """
    # === Missing Value Patterns ===
    missing_rate: float              # Fraction of missing values
    missing_numeric_rate: float      # Missing rate in numeric columns only
    missing_categorical_rate: float  # Missing rate in categorical columns only

    # === Numeric Distribution Patterns ===
    numeric_mean_zscore: float       # Mean |z-score| of numeric values
    numeric_max_zscore: float        # Max |z-score| (most extreme value)
    numeric_min_zscore: float        # Min |z-score| (most typical value)
    numeric_std: float               # Std of z-scores (within-row variance)
    numeric_skewness: float          # Skewness of numeric values in row
    numeric_kurtosis: float          # Kurtosis (heavy tails)
    numeric_range: float             # Normalized range (max-min)/std
    numeric_iqr: float               # Interquartile range of z-scores
    frac_zeros: float                # Fraction of zero values (sparsity)
    frac_negative: float             # Fraction of negative values
    frac_positive_outliers: float    # Fraction with z > 2
    frac_negative_outliers: float    # Fraction with z < -2

    # === Categorical Patterns ===
    categorical_rarity: float        # Mean rarity of categorical values
    categorical_modal_frac: float    # Mean: is this the modal value? (0-1)
    n_rare_categories: int           # Count of rare (<5%) categorical values
    n_unique_categories: int         # Total unique categories in this row
    categorical_entropy: float       # Entropy of categorical value frequencies

    # === Row Complexity/Structure ===
    row_entropy: float               # Entropy of all discretized values
    row_uniformity: float            # 1 - normalized entropy (how repetitive)
    n_distinct_values: int           # Count of distinct discretized values

    # === Position in Dataset ===
    centroid_distance: float         # Distance to dataset centroid (typicality)
    nearest_neighbor_dist: float     # Distance to nearest row (isolation)
    local_density: float             # Inverse mean distance to k-nearest
    pca_pc1: float                   # Position on first principal component
    pca_pc2: float                   # Position on second principal component
    pca_residual: float              # Reconstruction error from top PCs

    # === Dataset Characteristics (same for all rows in dataset) ===
    n_numeric: int                   # Number of numeric columns
    n_categorical: int               # Number of categorical columns
    n_rows_total: int                # Total rows in dataset
    n_cols_total: int                # Total columns
    dataset_sparsity: float          # Overall dataset missing rate
    numeric_correlation_mean: float  # Mean absolute correlation between numeric cols

    # === Scale/Magnitude Patterns ===
    log_magnitude_mean: float        # Mean of log10(|x|+1) - captures value scale
    log_magnitude_std: float         # Std of log magnitudes - spans many orders?
    frac_very_small: float           # Fraction with |x| < 0.01
    frac_very_large: float           # Fraction with |x| > 1000
    frac_integers: float             # Fraction that are integers (x == int(x))
    frac_round_tens: float           # Fraction divisible by 10

    # === Target-Related (if available) ===
    target_is_minority: float        # 1 if minority class, 0 otherwise (classification)
    target_zscore: float             # Z-score of target value (regression)


def compute_row_meta_features(
    df: pd.DataFrame,
    y: Optional[np.ndarray],
    numeric_cols: List[str],
    categorical_cols: List[str],
    col_stats: Dict,
    dataset_stats: Dict,
) -> List[RowMetaFeatures]:
    """
    Compute comprehensive dataset-agnostic meta-features for each row.

    These features describe the "shape" of a row without depending on
    specific column semantics, enabling cross-dataset concept analysis.
    """
    from scipy.stats import skew, kurtosis
    from sklearn.neighbors import NearestNeighbors

    n_rows = len(df)
    n_cols = len(df.columns)

    # Precompute dataset-level features
    dataset_sparsity = dataset_stats.get('missing_rate', 0.0)
    numeric_corr_mean = dataset_stats.get('numeric_correlation_mean', 0.0)

    # Build numeric matrix for distance computations
    numeric_matrix = np.zeros((n_rows, len(numeric_cols)))
    for j, col in enumerate(numeric_cols):
        if col in col_stats:
            vals = df[col].fillna(col_stats[col]['mean']).values
            mean, std = col_stats[col]['mean'], col_stats[col]['std']
            if std > 1e-8:
                numeric_matrix[:, j] = (vals - mean) / std
            else:
                numeric_matrix[:, j] = 0.0

    # Compute centroid and distances
    centroid = numeric_matrix.mean(axis=0)
    centroid_distances = np.linalg.norm(numeric_matrix - centroid, axis=1)

    # Nearest neighbor distances (for isolation/density)
    if n_rows > 5 and len(numeric_cols) > 0:
        k = min(5, n_rows - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
        nn.fit(numeric_matrix)
        distances, _ = nn.kneighbors(numeric_matrix)
        nn_distances = distances[:, 1]  # Exclude self
        local_densities = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-8)
    else:
        nn_distances = np.zeros(n_rows)
        local_densities = np.ones(n_rows)

    # PCA for position features
    if len(numeric_cols) >= 2 and n_rows > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(3, len(numeric_cols), n_rows - 1))
        pca_coords = pca.fit_transform(numeric_matrix)
        pca_residual = 1.0 - pca.explained_variance_ratio_.sum()
        pc1 = pca_coords[:, 0] if pca_coords.shape[1] > 0 else np.zeros(n_rows)
        pc2 = pca_coords[:, 1] if pca_coords.shape[1] > 1 else np.zeros(n_rows)
    else:
        pc1 = np.zeros(n_rows)
        pc2 = np.zeros(n_rows)
        pca_residual = 1.0

    # Target statistics
    if y is not None:
        unique_targets = np.unique(y)
        if len(unique_targets) <= 10:  # Classification
            target_counts = pd.Series(y).value_counts(normalize=True)
            minority_threshold = 0.5 / len(unique_targets)
            is_minority = np.array([target_counts.get(yi, 0) < minority_threshold for yi in y])
            target_zscores = np.zeros(n_rows)
        else:  # Regression
            is_minority = np.zeros(n_rows)
            y_mean, y_std = y.mean(), y.std()
            target_zscores = (y - y_mean) / (y_std + 1e-8) if y_std > 1e-8 else np.zeros(n_rows)
    else:
        is_minority = np.zeros(n_rows)
        target_zscores = np.zeros(n_rows)

    meta_features = []

    for idx in range(n_rows):
        row = df.iloc[idx]

        # === Missing Value Patterns ===
        missing_rate = row.isna().sum() / n_cols if n_cols > 0 else 0.0
        missing_numeric = sum(pd.isna(row[c]) for c in numeric_cols)
        missing_numeric_rate = missing_numeric / len(numeric_cols) if numeric_cols else 0.0
        missing_cat = sum(pd.isna(row[c]) for c in categorical_cols)
        missing_cat_rate = missing_cat / len(categorical_cols) if categorical_cols else 0.0

        # === Numeric Distribution Patterns ===
        zscores = []  # Signed z-scores
        abs_zscores = []
        raw_values = []
        for col in numeric_cols:
            val = row[col]
            if pd.notna(val) and col in col_stats:
                mean, std = col_stats[col]['mean'], col_stats[col]['std']
                raw_values.append(val)
                if std > 1e-8:
                    z = (val - mean) / std
                    zscores.append(z)
                    abs_zscores.append(abs(z))

        if abs_zscores:
            numeric_mean_zscore = np.mean(abs_zscores)
            numeric_max_zscore = np.max(abs_zscores)
            numeric_min_zscore = np.min(abs_zscores)
            numeric_std = np.std(abs_zscores)
            numeric_skewness = float(skew(zscores)) if len(zscores) > 2 else 0.0
            numeric_kurtosis = float(kurtosis(zscores)) if len(zscores) > 2 else 0.0
            numeric_range = (np.max(zscores) - np.min(zscores)) if len(zscores) > 1 else 0.0
            q75, q25 = np.percentile(abs_zscores, [75, 25])
            numeric_iqr = q75 - q25
            frac_pos_outliers = np.mean([z > 2 for z in zscores])
            frac_neg_outliers = np.mean([z < -2 for z in zscores])
        else:
            numeric_mean_zscore = numeric_max_zscore = numeric_min_zscore = 0.0
            numeric_std = numeric_skewness = numeric_kurtosis = 0.0
            numeric_range = numeric_iqr = 0.0
            frac_pos_outliers = frac_neg_outliers = 0.0

        # Zero and negative fractions
        if raw_values:
            frac_zeros = np.mean([abs(v) < 1e-10 for v in raw_values])
            frac_negative = np.mean([v < 0 for v in raw_values])
        else:
            frac_zeros = frac_negative = 0.0

        # === Categorical Patterns ===
        rarities = []
        modal_fracs = []
        n_rare = 0
        unique_cats = set()
        cat_value_freqs = []

        for col in categorical_cols:
            val = row[col]
            if pd.notna(val) and col in col_stats:
                freq = col_stats[col].get(val, 0.0)
                rarities.append(1.0 - freq)
                if freq < 0.05:
                    n_rare += 1
                # Is this the mode?
                max_freq = max(col_stats[col].values()) if col_stats[col] else 0
                modal_fracs.append(1.0 if abs(freq - max_freq) < 1e-8 else 0.0)
                unique_cats.add(f"{col}_{val}")
                cat_value_freqs.append(freq)

        categorical_rarity = np.mean(rarities) if rarities else 0.0
        categorical_modal_frac = np.mean(modal_fracs) if modal_fracs else 0.0
        n_unique_categories = len(unique_cats)

        if cat_value_freqs:
            probs = np.array(cat_value_freqs)
            probs = probs / (probs.sum() + 1e-10)
            categorical_entropy = -np.sum(probs * np.log(probs + 1e-10))
        else:
            categorical_entropy = 0.0

        # === Row Complexity/Structure ===
        # Discretize all values for entropy
        discrete_values = []
        for col in numeric_cols:
            val = row[col]
            if pd.notna(val) and col in col_stats:
                pctl = col_stats[col].get('percentiles', [])
                if pctl:
                    bin_idx = np.searchsorted(pctl, val)
                    discrete_values.append(f"n{bin_idx}")
        for col in categorical_cols:
            val = row[col]
            if pd.notna(val):
                discrete_values.append(f"c{val}")

        if discrete_values:
            _, counts = np.unique(discrete_values, return_counts=True)
            probs = counts / counts.sum()
            row_entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(discrete_values) + 1e-10)
            row_uniformity = 1.0 - (row_entropy / (max_entropy + 1e-10))
            n_distinct_values = len(counts)
        else:
            row_entropy = 0.0
            row_uniformity = 1.0
            n_distinct_values = 0

        # === Scale/Magnitude Patterns ===
        if raw_values:
            abs_vals = [abs(v) for v in raw_values]
            log_mags = [np.log10(v + 1) for v in abs_vals]
            log_magnitude_mean = np.mean(log_mags)
            log_magnitude_std = np.std(log_mags) if len(log_mags) > 1 else 0.0
            frac_very_small = np.mean([v < 0.01 for v in abs_vals])
            frac_very_large = np.mean([v > 1000 for v in abs_vals])
            frac_integers = np.mean([v == int(v) for v in raw_values])
            frac_round_tens = np.mean([v != 0 and v % 10 == 0 for v in raw_values])
        else:
            log_magnitude_mean = 0.0
            log_magnitude_std = 0.0
            frac_very_small = 0.0
            frac_very_large = 0.0
            frac_integers = 0.0
            frac_round_tens = 0.0

        meta_features.append(RowMetaFeatures(
            # Missing patterns
            missing_rate=missing_rate,
            missing_numeric_rate=missing_numeric_rate,
            missing_categorical_rate=missing_cat_rate,
            # Numeric distribution
            numeric_mean_zscore=numeric_mean_zscore,
            numeric_max_zscore=numeric_max_zscore,
            numeric_min_zscore=numeric_min_zscore,
            numeric_std=numeric_std,
            numeric_skewness=numeric_skewness,
            numeric_kurtosis=numeric_kurtosis,
            numeric_range=numeric_range,
            numeric_iqr=numeric_iqr,
            frac_zeros=frac_zeros,
            frac_negative=frac_negative,
            frac_positive_outliers=frac_pos_outliers,
            frac_negative_outliers=frac_neg_outliers,
            # Categorical
            categorical_rarity=categorical_rarity,
            categorical_modal_frac=categorical_modal_frac,
            n_rare_categories=n_rare,
            n_unique_categories=n_unique_categories,
            categorical_entropy=categorical_entropy,
            # Row complexity
            row_entropy=row_entropy,
            row_uniformity=row_uniformity,
            n_distinct_values=n_distinct_values,
            # Position in dataset
            centroid_distance=float(centroid_distances[idx]),
            nearest_neighbor_dist=float(nn_distances[idx]),
            local_density=float(local_densities[idx]),
            pca_pc1=float(pc1[idx]),
            pca_pc2=float(pc2[idx]),
            pca_residual=float(pca_residual),
            # Dataset characteristics
            n_numeric=len(numeric_cols),
            n_categorical=len(categorical_cols),
            n_rows_total=n_rows,
            n_cols_total=n_cols,
            dataset_sparsity=dataset_sparsity,
            numeric_correlation_mean=numeric_corr_mean,
            # Scale/Magnitude
            log_magnitude_mean=log_magnitude_mean,
            log_magnitude_std=log_magnitude_std,
            frac_very_small=frac_very_small,
            frac_very_large=frac_very_large,
            frac_integers=frac_integers,
            frac_round_tens=frac_round_tens,
            # Target
            target_is_minority=float(is_minority[idx]),
            target_zscore=float(target_zscores[idx]),
        ))

    return meta_features


def compute_column_stats(df: pd.DataFrame) -> Tuple[List[str], List[str], Dict, Dict]:
    """Compute statistics for each column and dataset-level stats."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    col_stats = {}

    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) > 0:
            col_stats[col] = {
                'mean': vals.mean(),
                'std': vals.std(),
                'percentiles': np.percentile(vals, [20, 40, 60, 80]).tolist(),
            }

    for col in categorical_cols:
        freq = df[col].value_counts(normalize=True)
        col_stats[col] = freq.to_dict()

    # Dataset-level statistics
    dataset_stats = {
        'missing_rate': df.isna().sum().sum() / (len(df) * len(df.columns)) if len(df) > 0 else 0.0,
    }

    # Mean absolute correlation between numeric columns
    if len(numeric_cols) >= 2:
        numeric_df = df[numeric_cols].dropna()
        if len(numeric_df) > 10:
            corr_matrix = numeric_df.corr().abs().values
            # Upper triangle excluding diagonal
            n = len(numeric_cols)
            upper_tri = corr_matrix[np.triu_indices(n, k=1)]
            dataset_stats['numeric_correlation_mean'] = float(np.nanmean(upper_tri))
        else:
            dataset_stats['numeric_correlation_mean'] = 0.0
    else:
        dataset_stats['numeric_correlation_mean'] = 0.0

    return numeric_cols, categorical_cols, col_stats, dataset_stats


def load_sae_checkpoint(path: Path) -> Tuple[SparseAutoencoder, SAEConfig, Dict]:
    """Load SAE model from checkpoint."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    cfg = checkpoint['config']
    config = cfg if isinstance(cfg, SAEConfig) else SAEConfig(**cfg)
    model = SparseAutoencoder(config)

    state_dict = checkpoint['model_state_dict']
    if 'reference_data' in state_dict and state_dict['reference_data'] is not None:
        ref_data = state_dict['reference_data']
        model.register_buffer('reference_data', ref_data)
        if 'archetype_logits' in state_dict:
            model.archetype_logits = torch.nn.Parameter(state_dict['archetype_logits'])
        if 'archetype_deviation' in state_dict:
            model.archetype_deviation = torch.nn.Parameter(state_dict['archetype_deviation'])

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, config, checkpoint


def get_train_test_split(datasets: List[str]) -> Tuple[List[str], List[str]]:
    """Deterministic train/test split matching sae_tabarena_sweep.py."""
    train_datasets = []
    test_datasets = []
    for ds in datasets:
        h = int(hashlib.md5(ds.encode()).hexdigest(), 16)
        if h % 10 < 7:
            train_datasets.append(ds)
        else:
            test_datasets.append(ds)
    return train_datasets, test_datasets


def analyze_feature_triggers(
    model: SparseAutoencoder,
    datasets: List[str],
    train_std: np.ndarray,
    train_mean: np.ndarray,
    top_features: List[int],
    samples_per_feature: int = 100,
    max_samples_per_dataset: int = 200,
    emb_dir: Path = None,
) -> Dict:
    """
    For each feature, find what tabular patterns trigger it.

    Returns per-feature analysis of row meta-features.
    """
    if emb_dir is None:
        emb_dir = PROJECT_ROOT / "output" / "embeddings" / "tabarena" / "tabpfn"

    # Collect all samples with their meta-features and activations
    all_meta = []  # List of RowMetaFeatures
    all_activations = []  # List of activation vectors
    all_dataset_names = []  # Track which dataset each sample came from

    print(f"Loading data and computing meta-features for {len(datasets)} datasets...")

    for ds_name in datasets:
        # Load embeddings
        emb_path = emb_dir / f"tabarena_{ds_name}.npz"
        if not emb_path.exists():
            continue

        emb_data = np.load(emb_path, allow_pickle=True)
        embeddings = emb_data['embeddings'].astype(np.float32)

        # Subsample if needed
        if len(embeddings) > max_samples_per_dataset:
            np.random.seed(42)
            idx = np.random.choice(len(embeddings), max_samples_per_dataset, replace=False)
            embeddings = embeddings[idx]
            sample_indices = idx
        else:
            sample_indices = np.arange(len(embeddings))

        # Load original tabular data
        try:
            X, y, dataset_info = load_tabarena_dataset(ds_name)
            df = pd.DataFrame(X)

            # Subset to same samples
            df = df.iloc[sample_indices].reset_index(drop=True)
            y_subset = y[sample_indices] if y is not None else None

            # Compute column stats and row meta-features
            numeric_cols, categorical_cols, col_stats, dataset_stats = compute_column_stats(df)
            meta_features = compute_row_meta_features(
                df, y_subset, numeric_cols, categorical_cols, col_stats, dataset_stats
            )

        except Exception as e:
            print(f"  Skipping {ds_name}: {e}")
            continue

        # Compute SAE activations
        emb_norm = embeddings / train_std
        emb_centered = emb_norm - train_mean

        with torch.no_grad():
            x = torch.tensor(emb_centered, dtype=torch.float32)
            h = model.encode(x).numpy()

        all_meta.extend(meta_features)
        all_activations.append(h)
        all_dataset_names.extend([ds_name] * len(meta_features))

        print(f"  {ds_name}: {len(meta_features)} samples")

    if not all_activations:
        return {}

    all_activations = np.concatenate(all_activations, axis=0)
    print(f"Total: {len(all_meta)} samples from {len(set(all_dataset_names))} datasets")

    # Convert meta-features to array for analysis (all 37 features)
    meta_array = np.array([
        [
            # Missing patterns (3)
            m.missing_rate, m.missing_numeric_rate, m.missing_categorical_rate,
            # Numeric distribution (13)
            m.numeric_mean_zscore, m.numeric_max_zscore, m.numeric_min_zscore,
            m.numeric_std, m.numeric_skewness, m.numeric_kurtosis,
            m.numeric_range, m.numeric_iqr, m.frac_zeros, m.frac_negative,
            m.frac_positive_outliers, m.frac_negative_outliers,
            # Categorical (5)
            m.categorical_rarity, m.categorical_modal_frac, m.n_rare_categories,
            m.n_unique_categories, m.categorical_entropy,
            # Row complexity (3)
            m.row_entropy, m.row_uniformity, m.n_distinct_values,
            # Position in dataset (6)
            m.centroid_distance, m.nearest_neighbor_dist, m.local_density,
            m.pca_pc1, m.pca_pc2, m.pca_residual,
            # Dataset characteristics (5)
            m.n_numeric, m.n_categorical, m.n_rows_total, m.n_cols_total,
            m.dataset_sparsity, m.numeric_correlation_mean,
            # Target (2)
            m.target_is_minority, m.target_zscore,
        ]
        for m in all_meta
    ])

    meta_names = [
        # Missing patterns
        'missing_rate', 'missing_numeric_rate', 'missing_categorical_rate',
        # Numeric distribution
        'numeric_mean_zscore', 'numeric_max_zscore', 'numeric_min_zscore',
        'numeric_std', 'numeric_skewness', 'numeric_kurtosis',
        'numeric_range', 'numeric_iqr', 'frac_zeros', 'frac_negative',
        'frac_positive_outliers', 'frac_negative_outliers',
        # Categorical
        'categorical_rarity', 'categorical_modal_frac', 'n_rare_categories',
        'n_unique_categories', 'categorical_entropy',
        # Row complexity
        'row_entropy', 'row_uniformity', 'n_distinct_values',
        # Position in dataset
        'centroid_distance', 'nearest_neighbor_dist', 'local_density',
        'pca_pc1', 'pca_pc2', 'pca_residual',
        # Dataset characteristics
        'n_numeric', 'n_categorical', 'n_rows_total', 'n_cols_total',
        'dataset_sparsity', 'numeric_correlation_mean',
        # Target
        'target_is_minority', 'target_zscore',
    ]

    # Compute baseline statistics
    baseline_means = meta_array.mean(axis=0)
    baseline_stds = meta_array.std(axis=0)

    # Analyze each feature
    feature_analysis = {}

    for feat_idx in top_features:
        feat_acts = all_activations[:, feat_idx]

        # Find top activating samples
        top_indices = np.argsort(feat_acts)[-samples_per_feature:]
        top_acts = feat_acts[top_indices]

        # Skip if feature is dead
        if top_acts.max() < 0.01:
            continue

        # Get meta-features for top samples
        top_meta = meta_array[top_indices]
        top_means = top_meta.mean(axis=0)

        # Compute effect sizes (Cohen's d)
        effect_sizes = {}
        for i, name in enumerate(meta_names):
            if baseline_stds[i] > 1e-8:
                d = (top_means[i] - baseline_means[i]) / baseline_stds[i]
                effect_sizes[name] = float(d)
            else:
                effect_sizes[name] = 0.0

        # Find which datasets contribute most to this feature
        top_datasets = [all_dataset_names[i] for i in top_indices]
        dataset_counts = defaultdict(int)
        for ds in top_datasets:
            dataset_counts[ds] += 1
        top_dataset_list = sorted(dataset_counts.items(), key=lambda x: -x[1])[:5]

        # Identify dominant pattern
        sorted_effects = sorted(effect_sizes.items(), key=lambda x: -abs(x[1]))
        dominant_patterns = [(name, d) for name, d in sorted_effects if abs(d) > 0.3]

        feature_analysis[feat_idx] = {
            'mean_activation': float(top_acts.mean()),
            'max_activation': float(top_acts.max()),
            'effect_sizes': effect_sizes,
            'dominant_patterns': dominant_patterns,
            'top_datasets': top_dataset_list,
            'interpretation': interpret_pattern(dominant_patterns),
        }

    return {
        'baseline': {name: {'mean': float(baseline_means[i]), 'std': float(baseline_stds[i])}
                     for i, name in enumerate(meta_names)},
        'features': feature_analysis,
    }


def format_raw_data_for_llm(
    raw_samples: List[Dict],
    baseline_samples: List[Dict],
    feature_id: int,
    activation_stats: Dict,
) -> str:
    """
    Format raw data samples for LLM analysis.

    Returns a prompt that shows:
    1. The top-activating raw data samples
    2. Some baseline samples for comparison
    3. Statistical summary of differences
    """
    # Build prompt
    lines = []
    lines.append(f"=== SAE Feature {feature_id} Analysis ===")
    lines.append(f"Mean activation: {activation_stats['mean']:.3f}")
    lines.append(f"Max activation: {activation_stats['max']:.3f}")
    lines.append("")

    # Show top-activating raw samples
    lines.append("TOP-ACTIVATING SAMPLES (raw values from original data):")
    lines.append("-" * 60)
    for i, sample in enumerate(raw_samples[:10]):
        lines.append(f"Sample {i+1} (activation={sample['activation']:.3f}, dataset={sample['dataset']}):")
        # Show non-missing values
        values = sample['values']
        if len(values) > 15:
            # Show first 15 columns
            value_str = ", ".join([f"{k}={v}" for k, v in list(values.items())[:15]])
            lines.append(f"  {value_str}, ...")
        else:
            value_str = ", ".join([f"{k}={v}" for k, v in values.items()])
            lines.append(f"  {value_str}")
        lines.append("")

    lines.append("\nBASELINE SAMPLES (random samples for comparison):")
    lines.append("-" * 60)
    for i, sample in enumerate(baseline_samples[:5]):
        lines.append(f"Baseline {i+1} (activation={sample['activation']:.3f}, dataset={sample['dataset']}):")
        values = sample['values']
        if len(values) > 15:
            value_str = ", ".join([f"{k}={v}" for k, v in list(values.items())[:15]])
            lines.append(f"  {value_str}, ...")
        else:
            value_str = ", ".join([f"{k}={v}" for k, v in values.items()])
            lines.append(f"  {value_str}")
        lines.append("")

    # Add statistical comparison
    lines.append("\nSTATISTICAL DIFFERENCES (top-activating vs baseline):")
    lines.append("-" * 60)

    # Collect all numeric values across samples for statistical comparison
    top_numeric_stats = {}
    baseline_numeric_stats = {}

    for sample in raw_samples:
        for k, v in sample['values'].items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                if k not in top_numeric_stats:
                    top_numeric_stats[k] = []
                top_numeric_stats[k].append(v)

    for sample in baseline_samples:
        for k, v in sample['values'].items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                if k not in baseline_numeric_stats:
                    baseline_numeric_stats[k] = []
                baseline_numeric_stats[k].append(v)

    # Find significant differences
    significant_diffs = []
    for col in set(top_numeric_stats.keys()) & set(baseline_numeric_stats.keys()):
        top_vals = top_numeric_stats[col]
        base_vals = baseline_numeric_stats[col]
        if len(top_vals) >= 5 and len(base_vals) >= 5:
            top_mean = np.mean(top_vals)
            base_mean = np.mean(base_vals)
            pooled_std = np.sqrt((np.var(top_vals) + np.var(base_vals)) / 2)
            if pooled_std > 1e-8:
                d = (top_mean - base_mean) / pooled_std
                if abs(d) > 0.5:
                    significant_diffs.append((col, d, top_mean, base_mean))

    significant_diffs.sort(key=lambda x: -abs(x[1]))
    for col, d, top_mean, base_mean in significant_diffs[:10]:
        direction = "higher" if d > 0 else "lower"
        lines.append(f"  {col}: {direction} in top samples (d={d:.2f}, top_mean={top_mean:.2f}, base_mean={base_mean:.2f})")

    return "\n".join(lines)


def load_api_key() -> Optional[str]:
    """Load Anthropic API key from .env file or environment."""
    import os

    # Check environment first
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key

    # Try .env file
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            value = value.strip().strip('"').strip("'")
            if name.strip() in ("ANTHROPIC_API_KEY", "ANTHROPIC_KEY"):
                return value

    return None


def query_llm_for_concept_label(
    raw_data_prompt: str,
    meta_feature_summary: str,
    client=None,
) -> Optional[str]:
    """
    Query an LLM to generate a concept label based on raw data patterns.

    Uses the Anthropic API with Haiku for speed/cost at scale.
    """
    if client is None:
        return None

    system_prompt = """You are an expert at analyzing tabular data patterns.
You are helping identify concepts that a Sparse Autoencoder (SAE) has learned from tabular foundation model embeddings.

For each SAE feature, you'll see:
1. Raw data samples that maximally activate this feature
2. Baseline samples for comparison
3. Statistical differences between top-activating and baseline samples

Your task: Identify what tabular pattern causes this feature to activate.
Focus on patterns that are:
- Universal (not specific to one dataset)
- About data properties (not semantics)
- Concise (2-5 words max)

Examples of good concept labels:
- "extreme outliers"
- "sparse rows"
- "negative-skewed values"
- "high feature correlation"
- "isolated points"
- "uniform distribution"
- "large magnitude"
- "integer values"

Respond with ONLY the concept label (2-5 words). Do not explain."""

    user_prompt = f"""{raw_data_prompt}

Meta-feature analysis:
{meta_feature_summary}

What is the concept this feature detects? (2-5 words only)"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"  LLM query failed: {e}")
        return None


def collect_raw_activating_samples(
    model: SparseAutoencoder,
    datasets: List[str],
    train_std: np.ndarray,
    train_mean: np.ndarray,
    feature_ids: List[int],
    n_samples_per_feature: int = 100,
    n_baseline_samples: int = 50,
    max_samples_per_dataset: int = 200,
    emb_dir: Path = None,
) -> Dict:
    """
    Collect raw data samples that maximally activate each feature.

    Returns a dict mapping feature_id -> {
        'top_samples': List of raw data dicts,
        'baseline_samples': List of random baseline dicts,
        'activation_stats': {mean, max, std}
    }
    """
    if emb_dir is None:
        emb_dir = PROJECT_ROOT / "output" / "embeddings" / "tabarena" / "tabpfn"

    # Collect all samples with their raw data, activations, and dataset info
    all_raw_samples = []  # List of dicts with raw values
    all_activations = []  # Corresponding activation vectors

    print(f"\nCollecting raw data for {len(datasets)} datasets...")

    for ds_name in datasets:
        # Load embeddings
        emb_path = emb_dir / f"tabarena_{ds_name}.npz"
        if not emb_path.exists():
            continue

        emb_data = np.load(emb_path, allow_pickle=True)
        embeddings = emb_data['embeddings'].astype(np.float32)

        # Subsample if needed
        if len(embeddings) > max_samples_per_dataset:
            np.random.seed(42)
            idx = np.random.choice(len(embeddings), max_samples_per_dataset, replace=False)
            embeddings = embeddings[idx]
            sample_indices = idx
        else:
            sample_indices = np.arange(len(embeddings))

        # Load original tabular data
        try:
            X, y, dataset_info = load_tabarena_dataset(ds_name)
            if hasattr(X, 'iloc'):
                df = X
            else:
                df = pd.DataFrame(X)

            # Subset to same samples
            df = df.iloc[sample_indices].reset_index(drop=True)

        except Exception as e:
            print(f"  Skipping {ds_name}: {e}")
            continue

        # Compute SAE activations
        emb_norm = embeddings / train_std
        emb_centered = emb_norm - train_mean

        with torch.no_grad():
            x = torch.tensor(emb_centered, dtype=torch.float32)
            h = model.encode(x).numpy()

        # Store raw data with activations
        for i in range(len(df)):
            row = df.iloc[i]
            raw_values = {col: row[col] for col in df.columns if pd.notna(row[col])}

            # Convert numpy types to Python types for JSON serialization
            clean_values = {}
            for k, v in raw_values.items():
                if isinstance(v, (np.integer,)):
                    clean_values[str(k)] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean_values[str(k)] = float(v)
                else:
                    clean_values[str(k)] = str(v)

            all_raw_samples.append({
                'dataset': ds_name,
                'values': clean_values,
                'n_values': len(clean_values),
            })

        all_activations.append(h)
        print(f"  {ds_name}: {len(df)} samples, {len(df.columns)} columns")

    if not all_activations:
        return {}

    all_activations = np.concatenate(all_activations, axis=0)
    print(f"Total: {len(all_raw_samples)} samples")

    # Collect samples for each feature
    results = {}

    for feat_id in feature_ids:
        feat_acts = all_activations[:, feat_id]

        # Skip dead features
        if feat_acts.max() < 0.001:
            continue

        # Get top-activating samples
        top_indices = np.argsort(feat_acts)[-n_samples_per_feature:]
        top_samples = []
        for idx in reversed(top_indices):  # Highest first
            sample = all_raw_samples[idx].copy()
            sample['activation'] = float(feat_acts[idx])
            top_samples.append(sample)

        # Get random baseline samples (with low activation)
        low_mask = feat_acts < np.percentile(feat_acts, 25)
        low_indices = np.where(low_mask)[0]
        if len(low_indices) >= n_baseline_samples:
            np.random.seed(feat_id)  # Reproducible baseline
            baseline_indices = np.random.choice(low_indices, n_baseline_samples, replace=False)
        else:
            baseline_indices = low_indices

        baseline_samples = []
        for idx in baseline_indices:
            sample = all_raw_samples[idx].copy()
            sample['activation'] = float(feat_acts[idx])
            baseline_samples.append(sample)

        results[feat_id] = {
            'top_samples': top_samples,
            'baseline_samples': baseline_samples,
            'activation_stats': {
                'mean': float(feat_acts.mean()),
                'max': float(feat_acts.max()),
                'std': float(feat_acts.std()),
            }
        }

    return results


def generate_concept_labels_with_llm(
    raw_sample_data: Dict,
    feature_analysis: Dict,
    use_llm: bool = True,
) -> Dict:
    """
    Generate concept labels for ALL features using LLM + statistical fallback.

    Returns dict mapping feature_id -> {
        'label': str,
        'method': 'llm' or 'statistical',
        'confidence': float,
    }
    """
    labels = {}

    # Initialize LLM client once
    client = None
    if use_llm:
        api_key = load_api_key()
        if api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                print("  LLM client initialized (Haiku 4.5)")
            except ImportError:
                print("  Warning: anthropic package not installed, using statistical labels only")
        else:
            print("  Warning: No API key found, using statistical labels only")

    # Label ALL features, sorted by activation magnitude
    sorted_features = sorted(
        raw_sample_data.keys(),
        key=lambda x: raw_sample_data[x]['activation_stats']['mean'],
        reverse=True
    )

    n_total = len(sorted_features)
    n_llm = 0
    n_stat = 0
    n_fail = 0

    print(f"\nGenerating labels for all {n_total} features...")

    for i, feat_id in enumerate(sorted_features):
        data = raw_sample_data[feat_id]
        analysis = feature_analysis.get(feat_id, feature_analysis.get(str(feat_id), {}))

        # Get meta-feature summary
        if 'effect_sizes' in analysis:
            sorted_effects = sorted(analysis['effect_sizes'].items(), key=lambda x: -abs(x[1]))
            top_effects = [(n, d) for n, d in sorted_effects if abs(d) > 0.3][:5]
            meta_summary = "\n".join([f"  {n}: d={d:.2f}" for n, d in top_effects])
        else:
            meta_summary = "No meta-feature analysis available"

        # Format raw data for LLM
        raw_prompt = format_raw_data_for_llm(
            data['top_samples'],
            data['baseline_samples'],
            feat_id,
            data['activation_stats'],
        )

        # Try LLM
        llm_label = None
        if client is not None:
            llm_label = query_llm_for_concept_label(raw_prompt, meta_summary, client=client)

        if llm_label:
            labels[feat_id] = {
                'label': llm_label,
                'method': 'llm',
                'confidence': 0.8,
            }
            n_llm += 1
        elif analysis.get('interpretation'):
            interp = analysis['interpretation']
            short_label = " ".join(interp.split()[:5])
            labels[feat_id] = {
                'label': short_label,
                'method': 'statistical',
                'confidence': 0.5,
            }
            n_stat += 1
        else:
            labels[feat_id] = {
                'label': 'unknown',
                'method': 'none',
                'confidence': 0.0,
            }
            n_fail += 1

        # Progress every 50 features
        if (i + 1) % 50 == 0 or (i + 1) == n_total:
            print(f"  [{i+1}/{n_total}] LLM: {n_llm}, statistical: {n_stat}, failed: {n_fail}")

    return labels


def interpret_pattern(patterns: List[Tuple[str, float]]) -> str:
    """Generate human-readable interpretation of dominant patterns."""
    if not patterns:
        return "No strong pattern detected"

    # Mapping from feature name to interpretation template
    interpretations_map = {
        # Missing patterns
        'missing_rate': "{s}{d} missing values overall",
        'missing_numeric_rate': "{s}{d} missing in numeric columns",
        'missing_categorical_rate': "{s}{d} missing in categorical columns",
        # Numeric distribution
        'numeric_mean_zscore': "{s}{d} average outlier-ness",
        'numeric_max_zscore': "{s}{d} extreme numeric value",
        'numeric_min_zscore': "{s}{d} minimum deviation from mean",
        'numeric_std': "{s}{d} within-row numeric variance",
        'numeric_skewness': "{s}{d} numeric skewness",
        'numeric_kurtosis': "{s}{d} numeric heavy-tailedness",
        'numeric_range': "{s}{d} numeric range spread",
        'numeric_iqr': "{s}{d} interquartile spread",
        'frac_zeros': "{s}{d} fraction of zeros (sparsity)",
        'frac_negative': "{s}{d} fraction of negative values",
        'frac_positive_outliers': "{s}{d} positive outliers (z>2)",
        'frac_negative_outliers': "{s}{d} negative outliers (z<-2)",
        # Categorical
        'categorical_rarity': "{s}{d} rare category usage",
        'categorical_modal_frac': "{s}{d} modal category match",
        'n_rare_categories': "{s}{d} count of rare categories",
        'n_unique_categories': "{s}{d} category diversity",
        'categorical_entropy': "{s}{d} categorical entropy",
        # Row complexity
        'row_entropy': "{s}{d} row complexity/entropy",
        'row_uniformity': "{s}{d} row uniformity (repetitiveness)",
        'n_distinct_values': "{s}{d} distinct value count",
        # Position in dataset
        'centroid_distance': "{s}{d} distance from centroid (atypical)",
        'nearest_neighbor_dist': "{s}{d} isolation (far from neighbors)",
        'local_density': "{s}{d} local density (crowded region)",
        'pca_pc1': "{s}{d} on PC1",
        'pca_pc2': "{s}{d} on PC2",
        'pca_residual': "{s}{d} PCA reconstruction error",
        # Dataset characteristics
        'n_numeric': "datasets with {d} numeric columns",
        'n_categorical': "datasets with {d} categorical columns",
        'n_rows_total': "{d} dataset size",
        'n_cols_total': "{d} total columns",
        'dataset_sparsity': "datasets with {s}{d} sparsity",
        'numeric_correlation_mean': "datasets with {s}{d} feature correlation",
        # Target
        'target_is_minority': "minority class samples" if "{d}" == "high" else "majority class samples",
        'target_zscore': "{s}{d} target value (regression)",
    }

    interpretations = []
    for name, d in patterns[:8]:  # Limit to top 8 patterns
        direction = "high" if d > 0 else "low"
        strength = "very " if abs(d) > 1.0 else ""

        if name in interpretations_map:
            template = interpretations_map[name]
            interp = template.format(s=strength, d=direction)
            interpretations.append(interp)
        else:
            interpretations.append(f"{strength}{direction} {name}")

    return "; ".join(interpretations)


def cluster_concepts(
    feature_analysis: Dict,
    meta_names: List[str],
    n_clusters: int = 20,
) -> Dict:
    """
    Cluster SAE features by their meta-feature effect profiles.

    This groups features that trigger on similar tabular patterns.
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler

    # Build effect size matrix: (n_features, n_meta_features)
    feature_ids = list(feature_analysis.keys())
    effect_matrix = np.array([
        [feature_analysis[fid]['effect_sizes'].get(name, 0.0) for name in meta_names]
        for fid in feature_ids
    ])

    # Standardize for clustering
    scaler = StandardScaler()
    effect_scaled = scaler.fit_transform(effect_matrix)

    # Hierarchical clustering for interpretable groups
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clustering.fit_predict(effect_scaled)

    # Analyze each cluster
    clusters = {}
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_features = [fid for fid, m in zip(feature_ids, mask) if m]

        if not cluster_features:
            continue

        # Mean effect sizes for this cluster
        cluster_effects = effect_matrix[mask].mean(axis=0)

        # Find dominant patterns (top effects by magnitude)
        effect_dict = {name: float(cluster_effects[i]) for i, name in enumerate(meta_names)}
        sorted_effects = sorted(effect_dict.items(), key=lambda x: -abs(x[1]))
        dominant = [(name, d) for name, d in sorted_effects if abs(d) > 0.2][:5]

        clusters[cluster_id] = {
            'n_features': len(cluster_features),
            'feature_ids': cluster_features,
            'mean_effects': effect_dict,
            'dominant_patterns': dominant,
            'interpretation': interpret_pattern(dominant),
        }

    # Sort clusters by size
    clusters = dict(sorted(clusters.items(), key=lambda x: -x[1]['n_features']))

    return {
        'n_clusters': n_clusters,
        'clusters': clusters,
        'feature_to_cluster': {fid: int(labels[i]) for i, fid in enumerate(feature_ids)},
    }


def compute_concept_coverage(
    feature_analysis: Dict,
    meta_names: List[str],
    threshold: float = 0.3,
) -> Dict:
    """
    Compute which meta-features are covered by the SAE's concepts.

    A meta-feature is "covered" if some SAE feature has strong effect on it.
    """
    coverage = {name: {'max_positive': 0.0, 'max_negative': 0.0, 'n_strong': 0}
                for name in meta_names}

    for fid, analysis in feature_analysis.items():
        for name in meta_names:
            d = analysis['effect_sizes'].get(name, 0.0)
            if d > coverage[name]['max_positive']:
                coverage[name]['max_positive'] = d
            if d < coverage[name]['max_negative']:
                coverage[name]['max_negative'] = d
            if abs(d) > threshold:
                coverage[name]['n_strong'] += 1

    # Compute coverage score: max(|max_pos|, |max_neg|)
    for name in meta_names:
        coverage[name]['coverage_score'] = max(
            abs(coverage[name]['max_positive']),
            abs(coverage[name]['max_negative'])
        )

    return coverage


def main():
    parser = argparse.ArgumentParser(description="Deep SAE concept analysis")
    parser.add_argument("--model-path", type=str, required=True, help="Path to SAE checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--top-k-features", type=int, default=None, help="Number of features (None=all alive)")
    parser.add_argument("--samples-per-feature", type=int, default=100, help="Top samples per feature")
    parser.add_argument("--n-clusters", type=int, default=25, help="Number of concept clusters")
    parser.add_argument("--label-concepts", action="store_true", help="Generate concept labels using LLM")
    parser.add_argument("--n-labels", type=int, default=50, help="Number of features to label with LLM")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM and use only statistical labels")
    parser.add_argument("--emb-dir", type=str, default=None,
                        help="Embedding directory (default: output/embeddings/tabarena/tabpfn)")
    args = parser.parse_args()

    print("Loading SAE checkpoint...")
    model, config, checkpoint = load_sae_checkpoint(Path(args.model_path))
    print(f"  SAE type: {config.sparsity_type}")
    print(f"  Hidden dim: {config.hidden_dim}")

    # Get datasets and compute training stats
    if args.emb_dir:
        emb_dir = Path(args.emb_dir)
    else:
        emb_dir = PROJECT_ROOT / "output" / "embeddings" / "tabarena" / "tabpfn"
    all_datasets = [f.stem.replace("tabarena_", "") for f in emb_dir.glob("tabarena_*.npz")]
    train_datasets, test_datasets = get_train_test_split(all_datasets)

    print(f"\nComputing training normalization stats from {len(train_datasets)} datasets...")

    # Pool training embeddings
    train_embs = []
    for ds in train_datasets:
        path = emb_dir / f"tabarena_{ds}.npz"
        if path.exists():
            data = np.load(path, allow_pickle=True)
            emb = data['embeddings'].astype(np.float32)
            if len(emb) > 200:
                np.random.seed(42)
                idx = np.random.choice(len(emb), 200, replace=False)
                emb = emb[idx]
            train_embs.append(emb)

    train_pooled = np.concatenate(train_embs)
    train_std = train_pooled.std(axis=0, keepdims=True)
    train_std[train_std < 1e-8] = 1.0
    train_norm = train_pooled / train_std
    train_mean = train_norm.mean(axis=0, keepdims=True)

    # Find all alive features (or top-k if specified)
    print("\nFinding active features...")
    train_centered = train_norm - train_mean
    with torch.no_grad():
        h = model.encode(torch.tensor(train_centered, dtype=torch.float32)).numpy()

    feature_means = h.mean(axis=0)
    feature_max = h.max(axis=0)
    alive_mask = feature_max > 0.001  # Feature fires on at least some samples (lowered from 0.01)
    alive_features = np.where(alive_mask)[0].tolist()

    print(f"  Total features: {config.hidden_dim}")
    print(f"  Alive features: {len(alive_features)} ({100*len(alive_features)/config.hidden_dim:.1f}%)")

    if args.top_k_features is not None:
        # Use top-k by mean activation
        sorted_features = np.argsort(feature_means)[::-1]
        top_features = [f for f in sorted_features if f in alive_features][:args.top_k_features]
        print(f"  Analyzing top {len(top_features)} features")
    else:
        # Analyze ALL alive features
        top_features = alive_features
        print(f"  Analyzing ALL {len(top_features)} alive features")

    # Deep analysis
    print("\n" + "="*60)
    print("DEEP CONCEPT ANALYSIS")
    print("="*60)

    results = analyze_feature_triggers(
        model=model,
        datasets=all_datasets,
        train_std=train_std,
        train_mean=train_mean,
        top_features=top_features,
        samples_per_feature=args.samples_per_feature,
        emb_dir=emb_dir,
    )

    # Meta-feature names for clustering
    meta_names = [
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

    # Cluster features by their effect profiles
    print("\n" + "="*60)
    print("CONCEPT CLUSTERING")
    print("="*60)

    if len(results.get('features', {})) > args.n_clusters:
        cluster_results = cluster_concepts(
            results['features'],
            meta_names,
            n_clusters=args.n_clusters,
        )
        results['clustering'] = cluster_results

        print(f"\nFound {args.n_clusters} concept clusters:")
        print("-"*60)
        for cid, cluster in cluster_results['clusters'].items():
            print(f"\nCluster {cid} ({cluster['n_features']} features):")
            print(f"  Interpretation: {cluster['interpretation']}")
            if cluster['dominant_patterns']:
                patterns = [(n, f"{d:.2f}") for n, d in cluster['dominant_patterns'][:3]]
                print(f"  Key patterns: {patterns}")

    # Compute coverage - which meta-features are well-represented?
    print("\n" + "="*60)
    print("META-FEATURE COVERAGE")
    print("="*60)

    coverage = compute_concept_coverage(results['features'], meta_names)
    results['coverage'] = coverage

    # Sort by coverage score
    sorted_coverage = sorted(coverage.items(), key=lambda x: -x[1]['coverage_score'])

    print("\nWell-covered meta-features (max |d| > 1.0):")
    for name, cov in sorted_coverage:
        if cov['coverage_score'] > 1.0:
            print(f"  {name}: max_d={cov['coverage_score']:.2f}, n_strong={cov['n_strong']}")

    print("\nPoorly-covered meta-features (max |d| < 0.5):")
    gaps = []
    for name, cov in sorted_coverage:
        if cov['coverage_score'] < 0.5:
            gaps.append(name)
            print(f"  {name}: max_d={cov['coverage_score']:.2f}")

    if gaps:
        print(f"\n⚠ GAPS: The SAE has weak coverage for: {gaps}")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Total features analyzed: {len(results['features'])}")
    print(f"  Meta-features with strong coverage: {sum(1 for c in coverage.values() if c['coverage_score'] > 1.0)}")
    print(f"  Meta-features with weak coverage: {sum(1 for c in coverage.values() if c['coverage_score'] < 0.5)}")

    # Concept labeling with LLM
    if args.label_concepts:
        print("\n" + "="*60)
        print("CONCEPT LABELING (Raw Data + LLM)")
        print("="*60)

        # Collect raw data samples for top features
        raw_sample_data = collect_raw_activating_samples(
            model=model,
            datasets=all_datasets,
            train_std=train_std,
            train_mean=train_mean,
            feature_ids=top_features,
            n_samples_per_feature=args.samples_per_feature,
            n_baseline_samples=50,
            emb_dir=emb_dir,
        )

        # Generate labels for ALL features
        concept_labels = generate_concept_labels_with_llm(
            raw_sample_data=raw_sample_data,
            feature_analysis=results['features'],
            use_llm=not args.no_llm,
        )

        results['concept_labels'] = concept_labels

        # Print labeled concepts
        print("\n" + "-"*60)
        print("LABELED CONCEPTS:")
        print("-"*60)

        # Sort by confidence then by feature ID
        sorted_labels = sorted(
            concept_labels.items(),
            key=lambda x: (-x[1]['confidence'], x[0])
        )

        for feat_id, label_info in sorted_labels[:30]:
            method_tag = "🤖" if label_info['method'] == 'llm' else "📊"
            print(f"  F{feat_id:4d}: {label_info['label']:30s} {method_tag}")

        # Summary stats
        n_llm = sum(1 for l in concept_labels.values() if l['method'] == 'llm')
        n_stat = sum(1 for l in concept_labels.values() if l['method'] == 'statistical')
        print(f"\n  LLM labels: {n_llm}, Statistical labels: {n_stat}")

    # Save results
    if args.output:
        # Convert numpy keys to native Python types
        results_clean = convert_keys_to_native(results)
        with open(args.output, 'w') as f:
            json.dump(results_clean, f, indent=2, cls=NumpyEncoder)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
