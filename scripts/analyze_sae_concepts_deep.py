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

    # === Supervised Complexity (require y, classification only) ===
    fisher_ratio: float = 0.0       # F1: row-level class separability
    borderline: float = 0.0         # N1: fraction of k-NN from different class
    knn_class_ratio: float = 0.0    # N2: same-class / different-class NN ratio
    linear_boundary_dist: float = 0.0  # L1: distance to SVM hyperplane

    # === Graph Topology (from existing k-NN graph) ===
    hub_score: float = 0.0          # How often this row appears in others' k-NN
    local_clustering: float = 0.0   # Fraction of neighbor pairs that are mutual neighbors
    local_intrinsic_dim: float = 0.0  # MLE local dimensionality (Levina-Bickel 2004)

    # === Information-Theoretic ===
    row_surprise: float = 0.0       # -log p(row) under per-column Gaussian
    mi_contribution: float = 0.0    # Pointwise MI between features and target

    # === Target-Related (if available) ===
    target_is_minority: float = 0.0  # 1 if minority class, 0 otherwise (classification)
    target_zscore: float = 0.0       # Z-score of target value (regression)


def _compute_borderline(nn_indices: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Fraction of k-NN from different class (N1 complexity measure)."""
    neighbor_labels = y[nn_indices[:, 1:k+1]]  # exclude self
    own_labels = y[:, np.newaxis]
    return (neighbor_labels != own_labels).mean(axis=1).astype(np.float64)


def _compute_knn_class_ratio(nn_indices: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Ratio: same-class NN / (different-class NN + 1). Higher = more separable."""
    neighbor_labels = y[nn_indices[:, 1:k+1]]
    own_labels = y[:, np.newaxis]
    same = (neighbor_labels == own_labels).sum(axis=1)
    diff = (neighbor_labels != own_labels).sum(axis=1)
    return (same / (diff + 1.0)).astype(np.float64)


def _compute_fisher_per_row(numeric_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row Fisher discriminant ratio: sum_d (mu_own - mu_other)^2 / (var_own + var_other).

    Vectorized: precompute per-class Fisher scores, then index by class label.
    """
    classes = np.unique(y)
    n_samples, n_dims = numeric_matrix.shape
    global_mean = numeric_matrix.mean(axis=0)
    global_var = numeric_matrix.var(axis=0) + 1e-8

    # Per-class Fisher score (same for all members of a class)
    class_fisher = np.zeros(len(classes))
    class_map = {}
    for ci, c in enumerate(classes):
        mask = y == c
        own_mean = numeric_matrix[mask].mean(axis=0)
        own_var = numeric_matrix[mask].var(axis=0) + 1e-8
        ratio = ((own_mean - global_mean) ** 2) / (own_var + global_var)
        class_fisher[ci] = ratio.mean()
        class_map[c] = ci

    # Map each sample to its class's Fisher score
    class_indices = np.array([class_map[yi] for yi in y])
    return class_fisher[class_indices]


def _compute_linear_boundary_dist(numeric_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distance to SVM decision boundary. Fit once, evaluate all rows."""
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_matrix)
    try:
        svc = LinearSVC(max_iter=1000, dual='auto')
        svc.fit(X_scaled, y)
        distances = svc.decision_function(X_scaled)
        if distances.ndim > 1:
            distances = np.abs(distances).min(axis=1)  # Multi-class: min distance
        else:
            distances = np.abs(distances)
        return distances
    except Exception:
        return np.zeros(len(numeric_matrix))


def _compute_hub_scores(nn_indices: np.ndarray) -> np.ndarray:
    """How often each point appears in others' k-NN lists (hubness)."""
    n = nn_indices.shape[0]
    neighbor_flat = nn_indices[:, 1:].ravel()  # exclude self column
    return np.bincount(neighbor_flat, minlength=n).astype(np.float64)


def _compute_local_clustering(nn_indices: np.ndarray, k: int) -> np.ndarray:
    """Fraction of neighbor pairs that are mutual neighbors.

    For each point i, check all C(k,2) pairs of i's neighbors (a,b):
    is b in a's neighbor list? Uses a sparse membership set for O(1) lookup.
    """
    n = nn_indices.shape[0]
    neighbors = nn_indices[:, 1:k+1]  # (n, k) excluding self

    if k < 2:
        return np.zeros(n)

    # Build neighbor sets once for O(1) membership lookup
    neighbor_sets = [set(neighbors[i]) for i in range(n)]

    # Precompute all neighbor pairs per point
    n_pairs = k * (k - 1) // 2
    clustering = np.zeros(n)
    for i in range(n):
        connected = 0
        ni = neighbors[i]
        for a in range(k):
            ns_a = neighbor_sets[ni[a]]
            for b in range(a + 1, k):
                if ni[b] in ns_a:
                    connected += 1
        clustering[i] = connected / n_pairs
    return clustering


def _compute_local_id(distances: np.ndarray, k: int) -> np.ndarray:
    """MLE local intrinsic dimensionality (Levina-Bickel 2004)."""
    # distances shape: (n, k+1), column 0 is self (0)
    nn_dists = distances[:, 1:k+1]  # (n, k)
    nn_dists = np.maximum(nn_dists, 1e-10)  # avoid log(0)
    # MLE: d_hat = 1 / mean(log(max_dist / dist_j)) for j=1..k-1
    max_dist = nn_dists[:, -1:]  # (n, 1)
    log_ratios = np.log(max_dist / nn_dists[:, :-1])  # (n, k-1)
    mean_log = log_ratios.mean(axis=1)
    local_id = 1.0 / (mean_log + 1e-10)
    return np.clip(local_id, 0, 100)  # Cap at 100 to avoid numerical instability


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
    import torch
    from scipy.stats import skew, kurtosis

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

    # Nearest neighbor distances via torch (for isolation/density and graph topology)
    has_nn = n_rows > 5 and len(numeric_cols) > 0
    if has_nn:
        k = min(5, n_rows - 1)
        X_t = torch.tensor(numeric_matrix, dtype=torch.float32)
        # Pairwise Euclidean distance via torch.cdist
        dists = torch.cdist(X_t, X_t)  # (n_rows, n_rows)
        # k+1 nearest (including self at distance 0)
        distances_t, nn_indices_t = dists.topk(k + 1, largest=False)
        distances = distances_t.numpy()
        nn_indices = nn_indices_t.numpy()
        nn_distances = distances[:, 1]  # Exclude self
        local_densities = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-8)
    else:
        k = 0
        nn_distances = np.zeros(n_rows)
        local_densities = np.ones(n_rows)
        distances = np.zeros((n_rows, 1))
        nn_indices = np.zeros((n_rows, 1), dtype=int)

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

    # === Vectorized precomputation: supervised complexity + graph topology ===
    is_classification = (y is not None and len(np.unique(y)) <= 10 and len(np.unique(y)) >= 2)

    if is_classification and has_nn and len(numeric_cols) > 0:
        borderline_arr = _compute_borderline(nn_indices, y, k)
        knn_class_ratio_arr = _compute_knn_class_ratio(nn_indices, y, k)
        fisher_arr = _compute_fisher_per_row(numeric_matrix, y)
        linear_boundary_dist_arr = _compute_linear_boundary_dist(numeric_matrix, y)
    else:
        borderline_arr = np.zeros(n_rows)
        knn_class_ratio_arr = np.zeros(n_rows)
        fisher_arr = np.zeros(n_rows)
        linear_boundary_dist_arr = np.zeros(n_rows)

    if has_nn:
        hub_scores_arr = _compute_hub_scores(nn_indices)
        local_clustering_arr = _compute_local_clustering(nn_indices, k)
        local_id_arr = _compute_local_id(distances, k)
    else:
        hub_scores_arr = np.zeros(n_rows)
        local_clustering_arr = np.zeros(n_rows)
        local_id_arr = np.zeros(n_rows)

    # Row surprise: -log p(row) under per-column Gaussian
    # Precompute column means/stds for Gaussian model
    col_means = np.array([col_stats[c]['mean'] for c in numeric_cols if c in col_stats])
    col_stds = np.array([col_stats[c]['std'] for c in numeric_cols if c in col_stats])
    col_stds_safe = np.where(col_stds > 1e-8, col_stds, 1.0)

    # MI contribution: discretize features + target, compute pointwise MI
    mi_contributions = np.zeros(n_rows)
    if y is not None and len(numeric_cols) > 0:
        # Discretize target into 10 bins (or use class labels for classification)
        if is_classification:
            y_disc = y.astype(int)
        else:
            y_disc = np.digitize(y, np.percentile(y, np.linspace(0, 100, 11)[1:-1]))
        # Discretize features into 5 bins each
        n_feat = numeric_matrix.shape[1]
        feat_disc = np.zeros_like(numeric_matrix, dtype=int)
        for j in range(n_feat):
            col_vals = numeric_matrix[:, j]
            try:
                feat_disc[:, j] = np.digitize(col_vals, np.percentile(col_vals, [20, 40, 60, 80]))
            except Exception:
                feat_disc[:, j] = 0
        # Compute joint and marginal counts for pointwise MI
        from collections import Counter
        # Build joint distribution: (feature_bin_tuple, y_bin) -> count
        feat_keys = [tuple(feat_disc[i]) for i in range(n_rows)]
        joint_counts = Counter(zip(feat_keys, y_disc))
        feat_counts = Counter(feat_keys)
        y_counts = Counter(y_disc)
        for i in range(n_rows):
            fk = feat_keys[i]
            yk = y_disc[i]
            p_joint = joint_counts[(fk, yk)] / n_rows
            p_feat = feat_counts[fk] / n_rows
            p_y = y_counts[yk] / n_rows
            if p_joint > 0 and p_feat > 0 and p_y > 0:
                mi_contributions[i] = np.log(p_joint / (p_feat * p_y + 1e-15))

    # =======================================================================
    # Vectorized computation of all per-row meta-features
    # =======================================================================

    # --- Missing patterns (vectorized over DataFrame) ---
    missing_total = df.isna().sum(axis=1).values / n_cols if n_cols > 0 else np.zeros(n_rows)
    if numeric_cols:
        missing_num = df[numeric_cols].isna().sum(axis=1).values / len(numeric_cols)
    else:
        missing_num = np.zeros(n_rows)
    if categorical_cols:
        missing_cat = df[categorical_cols].isna().sum(axis=1).values / len(categorical_cols)
    else:
        missing_cat = np.zeros(n_rows)

    # --- Numeric distribution patterns (from z-score matrix) ---
    # numeric_matrix is already z-scored; build raw matrix + valid mask
    n_num = len(numeric_cols)
    if n_num > 0:
        # Raw numeric values with NaN preserved for masking
        raw_matrix = np.full((n_rows, n_num), np.nan)
        valid_mask = np.zeros((n_rows, n_num), dtype=bool)  # non-NaN and in col_stats
        zscore_valid = np.zeros((n_rows, n_num), dtype=bool)  # also has std > 1e-8
        for j, col in enumerate(numeric_cols):
            if col in col_stats:
                vals = df[col].values
                not_na = ~pd.isna(vals)
                raw_matrix[not_na, j] = vals[not_na].astype(np.float64)
                valid_mask[:, j] = not_na
                if col_stats[col]['std'] > 1e-8:
                    zscore_valid[:, j] = not_na

        # Signed z-scores (already in numeric_matrix, but only where zscore_valid)
        zscores_full = np.where(zscore_valid, numeric_matrix, np.nan)
        abs_zscores_full = np.abs(zscores_full)

        n_valid_z = zscore_valid.sum(axis=1)  # per row
        has_z = n_valid_z > 0

        # Replace NaN with 0 for nanmean etc. — use masked computations
        with np.errstate(all='ignore'):
            mean_abs_z = np.nanmean(abs_zscores_full, axis=1)
            max_abs_z = np.nanmax(abs_zscores_full, axis=1)
            min_abs_z = np.nanmin(abs_zscores_full, axis=1)
            std_abs_z = np.nanstd(abs_zscores_full, axis=1)

        mean_abs_z = np.where(has_z, mean_abs_z, 0.0)
        max_abs_z = np.where(has_z, max_abs_z, 0.0)
        min_abs_z = np.where(has_z, min_abs_z, 0.0)
        std_abs_z = np.where(has_z, std_abs_z, 0.0)

        # Skewness and kurtosis (need > 2 valid z-scores per row)
        has_enough = n_valid_z > 2
        skew_arr = np.zeros(n_rows)
        kurt_arr = np.zeros(n_rows)
        for idx in np.where(has_enough)[0]:
            row_z = zscores_full[idx, zscore_valid[idx]]
            skew_arr[idx] = skew(row_z)
            kurt_arr[idx] = kurtosis(row_z)

        # Range and IQR
        with np.errstate(all='ignore'):
            z_max = np.nanmax(zscores_full, axis=1)
            z_min = np.nanmin(zscores_full, axis=1)
        range_arr = np.where(n_valid_z > 1, z_max - z_min, 0.0)

        iqr_arr = np.zeros(n_rows)
        for idx in np.where(has_z)[0]:
            row_abs = abs_zscores_full[idx, zscore_valid[idx]]
            q75, q25 = np.percentile(row_abs, [75, 25])
            iqr_arr[idx] = q75 - q25

        # Outlier fractions
        frac_pos_out = np.where(has_z,
            np.nansum(zscores_full > 2, axis=1) / np.maximum(n_valid_z, 1), 0.0)
        frac_neg_out = np.where(has_z,
            np.nansum(zscores_full < -2, axis=1) / np.maximum(n_valid_z, 1), 0.0)

        # Zero and negative fractions (from raw values)
        n_valid_raw = valid_mask.sum(axis=1)
        has_raw = n_valid_raw > 0
        raw_for_calc = np.where(valid_mask, raw_matrix, np.nan)
        with np.errstate(all='ignore'):
            frac_zeros_arr = np.nansum(np.abs(raw_for_calc) < 1e-10, axis=1) / np.maximum(n_valid_raw, 1)
            frac_neg_arr = np.nansum(raw_for_calc < 0, axis=1) / np.maximum(n_valid_raw, 1)
        frac_zeros_arr = np.where(has_raw, frac_zeros_arr, 0.0)
        frac_neg_arr = np.where(has_raw, frac_neg_arr, 0.0)

        # --- Scale/Magnitude (from raw values) ---
        abs_raw = np.abs(np.where(valid_mask, raw_matrix, np.nan))
        log_mags = np.log10(abs_raw + 1)
        with np.errstate(all='ignore'):
            log_mag_mean = np.nanmean(log_mags, axis=1)
            log_mag_std = np.nanstd(log_mags, axis=1)
        log_mag_mean = np.where(has_raw, log_mag_mean, 0.0)
        log_mag_std = np.where(n_valid_raw > 1, log_mag_std, 0.0)

        frac_tiny = np.where(has_raw,
            np.nansum(abs_raw < 0.01, axis=1) / np.maximum(n_valid_raw, 1), 0.0)
        frac_huge = np.where(has_raw,
            np.nansum(abs_raw > 1000, axis=1) / np.maximum(n_valid_raw, 1), 0.0)

        # Integer fraction: x == floor(x) for non-NaN raw values
        raw_nonan = np.where(valid_mask, raw_matrix, 0.0)
        is_int = valid_mask & (raw_nonan == np.floor(raw_nonan))
        frac_int = np.where(has_raw, is_int.sum(axis=1) / np.maximum(n_valid_raw, 1), 0.0)

        # Round-tens fraction
        is_round10 = valid_mask & (raw_nonan != 0) & (raw_nonan % 10 == 0)
        frac_r10 = np.where(has_raw, is_round10.sum(axis=1) / np.maximum(n_valid_raw, 1), 0.0)

    else:
        mean_abs_z = max_abs_z = min_abs_z = std_abs_z = np.zeros(n_rows)
        skew_arr = kurt_arr = range_arr = iqr_arr = np.zeros(n_rows)
        frac_pos_out = frac_neg_out = np.zeros(n_rows)
        frac_zeros_arr = frac_neg_arr = np.zeros(n_rows)
        log_mag_mean = log_mag_std = np.zeros(n_rows)
        frac_tiny = frac_huge = frac_int = frac_r10 = np.zeros(n_rows)

    # --- Categorical patterns (vectorized per-column, aggregated per-row) ---
    if categorical_cols:
        # Build arrays: rarity, modal_frac, is_rare per (row, col)
        cat_rarity_sum = np.zeros(n_rows)
        cat_modal_sum = np.zeros(n_rows)
        cat_rare_count = np.zeros(n_rows, dtype=int)
        cat_valid_count = np.zeros(n_rows, dtype=int)
        cat_freq_lists = [[] for _ in range(n_rows)]  # for entropy
        cat_unique_sets = [set() for _ in range(n_rows)]

        for col in categorical_cols:
            vals = df[col].values
            not_na = ~pd.isna(vals)
            if col not in col_stats:
                continue
            freq_map = col_stats[col]
            max_freq = max(freq_map.values()) if freq_map else 0

            for idx in np.where(not_na)[0]:
                v = vals[idx]
                freq = freq_map.get(v, 0.0)
                cat_rarity_sum[idx] += (1.0 - freq)
                cat_modal_sum[idx] += (1.0 if abs(freq - max_freq) < 1e-8 else 0.0)
                if freq < 0.05:
                    cat_rare_count[idx] += 1
                cat_valid_count[idx] += 1
                cat_unique_sets[idx].add(f"{col}_{v}")
                cat_freq_lists[idx].append(freq)

        has_cat = cat_valid_count > 0
        cat_rarity_arr = np.where(has_cat, cat_rarity_sum / cat_valid_count, 0.0)
        cat_modal_arr = np.where(has_cat, cat_modal_sum / cat_valid_count, 0.0)
        n_unique_cat_arr = np.array([len(s) for s in cat_unique_sets])

        cat_entropy_arr = np.zeros(n_rows)
        for idx in np.where(has_cat)[0]:
            probs = np.array(cat_freq_lists[idx])
            probs = probs / (probs.sum() + 1e-10)
            cat_entropy_arr[idx] = -np.sum(probs * np.log(probs + 1e-10))
    else:
        cat_rarity_arr = cat_modal_arr = np.zeros(n_rows)
        cat_rare_count = np.zeros(n_rows, dtype=int)
        n_unique_cat_arr = np.zeros(n_rows, dtype=int)
        cat_entropy_arr = np.zeros(n_rows)

    # --- Row entropy/complexity (discretize all values) ---
    # Build discretized matrix: numeric bins + categorical labels
    # For numeric: bin into percentile bins; for categorical: use value directly
    row_entropy_arr = np.zeros(n_rows)
    row_uniformity_arr = np.ones(n_rows)
    n_distinct_arr = np.zeros(n_rows, dtype=int)

    # Pre-build percentile arrays for each numeric column
    pctl_arrays = {}
    for col in numeric_cols:
        if col in col_stats:
            pctl = col_stats[col].get('percentiles', [])
            if pctl:
                pctl_arrays[col] = np.array(pctl)

    # Vectorized discretization for numeric columns
    num_bins = np.full((n_rows, n_num), -1, dtype=int)  # -1 = missing/invalid
    for j, col in enumerate(numeric_cols):
        if col in pctl_arrays:
            vals = df[col].values
            not_na = ~pd.isna(vals)
            num_bins[not_na, j] = np.searchsorted(pctl_arrays[col], vals[not_na].astype(np.float64))

    # Compute row entropy from discretized values
    for idx in range(n_rows):
        # Collect discrete tokens for this row
        tokens = []
        for j in range(n_num):
            if num_bins[idx, j] >= 0:
                tokens.append(num_bins[idx, j])  # numeric bin index
        # Categorical tokens: offset by max_bin to avoid collision
        cat_offset = 100
        for col in categorical_cols:
            val = df[col].iat[idx]
            if pd.notna(val):
                tokens.append(hash(f"c{val}") % 100000 + cat_offset)

        if tokens:
            _, counts = np.unique(tokens, return_counts=True)
            probs = counts / counts.sum()
            ent = -np.sum(probs * np.log(probs + 1e-10))
            max_ent = np.log(len(tokens) + 1e-10)
            row_entropy_arr[idx] = ent
            row_uniformity_arr[idx] = 1.0 - (ent / (max_ent + 1e-10))
            n_distinct_arr[idx] = len(counts)

    # --- Row surprise (vectorized) ---
    if len(col_means) > 0:
        z_for_surprise = (numeric_matrix[:, :len(col_means)] - col_means) / col_stds_safe
        log_pdf = -0.5 * z_for_surprise**2 - np.log(col_stds_safe * np.sqrt(2 * np.pi) + 1e-10)
        row_surprise_arr = -log_pdf.sum(axis=1)
    else:
        row_surprise_arr = np.zeros(n_rows)

    # --- Assemble RowMetaFeatures list from arrays ---
    # Scalar dataset-level values
    _n_numeric = len(numeric_cols)
    _n_categorical = len(categorical_cols)

    meta_features = []
    for idx in range(n_rows):
        meta_features.append(RowMetaFeatures(
            missing_rate=float(missing_total[idx]),
            missing_numeric_rate=float(missing_num[idx]),
            missing_categorical_rate=float(missing_cat[idx]),
            numeric_mean_zscore=float(mean_abs_z[idx]),
            numeric_max_zscore=float(max_abs_z[idx]),
            numeric_min_zscore=float(min_abs_z[idx]),
            numeric_std=float(std_abs_z[idx]),
            numeric_skewness=float(skew_arr[idx]),
            numeric_kurtosis=float(kurt_arr[idx]),
            numeric_range=float(range_arr[idx]),
            numeric_iqr=float(iqr_arr[idx]),
            frac_zeros=float(frac_zeros_arr[idx]),
            frac_negative=float(frac_neg_arr[idx]),
            frac_positive_outliers=float(frac_pos_out[idx]),
            frac_negative_outliers=float(frac_neg_out[idx]),
            categorical_rarity=float(cat_rarity_arr[idx]),
            categorical_modal_frac=float(cat_modal_arr[idx]),
            n_rare_categories=int(cat_rare_count[idx]),
            n_unique_categories=int(n_unique_cat_arr[idx]),
            categorical_entropy=float(cat_entropy_arr[idx]),
            row_entropy=float(row_entropy_arr[idx]),
            row_uniformity=float(row_uniformity_arr[idx]),
            n_distinct_values=int(n_distinct_arr[idx]),
            centroid_distance=float(centroid_distances[idx]),
            nearest_neighbor_dist=float(nn_distances[idx]),
            local_density=float(local_densities[idx]),
            pca_pc1=float(pc1[idx]),
            pca_pc2=float(pc2[idx]),
            pca_residual=float(pca_residual),
            n_numeric=_n_numeric,
            n_categorical=_n_categorical,
            n_rows_total=n_rows,
            n_cols_total=n_cols,
            dataset_sparsity=dataset_sparsity,
            numeric_correlation_mean=numeric_corr_mean,
            log_magnitude_mean=float(log_mag_mean[idx]),
            log_magnitude_std=float(log_mag_std[idx]),
            frac_very_small=float(frac_tiny[idx]),
            frac_very_large=float(frac_huge[idx]),
            frac_integers=float(frac_int[idx]),
            frac_round_tens=float(frac_r10[idx]),
            fisher_ratio=float(fisher_arr[idx]),
            borderline=float(borderline_arr[idx]),
            knn_class_ratio=float(knn_class_ratio_arr[idx]),
            linear_boundary_dist=float(linear_boundary_dist_arr[idx]),
            hub_score=float(hub_scores_arr[idx]),
            local_clustering=float(local_clustering_arr[idx]),
            local_intrinsic_dim=float(local_id_arr[idx]),
            row_surprise=float(row_surprise_arr[idx]),
            mi_contribution=float(mi_contributions[idx]),
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

    # Convert meta-features to array — use canonical META_NAMES from compare_sae_architectures
    from scripts.compare_sae_architectures import META_NAMES as meta_names, meta_features_to_array
    meta_array = np.array([meta_features_to_array(m) for m in all_meta])

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
