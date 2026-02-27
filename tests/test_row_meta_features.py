"""Tests for expanded row-level meta-features (52 probes)."""

import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_sae_concepts_deep import (
    RowMetaFeatures,
    compute_column_stats,
    compute_row_meta_features,
    _compute_borderline,
    _compute_knn_class_ratio,
    _compute_fisher_per_row,
    _compute_hub_scores,
    _compute_local_clustering,
    _compute_local_id,
    _compute_linear_boundary_dist,
)
from scripts.compare_sae_architectures import META_NAMES, meta_features_to_array


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    """Simple numeric DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'a': np.random.randn(n),
        'b': np.random.randn(n) * 2 + 1,
        'c': np.random.randn(n) * 0.5 - 1,
    })


@pytest.fixture
def classification_y():
    """Binary classification target."""
    np.random.seed(42)
    return np.array([0] * 50 + [1] * 50)


@pytest.fixture
def meta_result(simple_df, classification_y):
    """Compute meta-features for fixture data."""
    numeric_cols, categorical_cols, col_stats, dataset_stats = compute_column_stats(simple_df)
    return compute_row_meta_features(
        simple_df, classification_y, numeric_cols, categorical_cols,
        col_stats, dataset_stats,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFieldCountMatchesMetaNames:
    def test_field_count_equals_meta_names(self):
        """RowMetaFeatures should have exactly as many fields as META_NAMES."""
        n_fields = len(fields(RowMetaFeatures))
        assert n_fields == len(META_NAMES), (
            f"RowMetaFeatures has {n_fields} fields but META_NAMES has {len(META_NAMES)}"
        )

    def test_meta_features_to_array_length(self, meta_result):
        """meta_features_to_array output length must match META_NAMES."""
        arr = meta_features_to_array(meta_result[0])
        assert len(arr) == len(META_NAMES)

    def test_field_count_is_52(self):
        """We expect exactly 52 meta-features after expansion."""
        assert len(META_NAMES) == 52


class TestBackwardCompatibleOrder:
    """First 35 positions (before scale/magnitude) must be unchanged from original 37."""

    ORIGINAL_35 = [
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
    ]

    def test_first_35_unchanged(self):
        assert META_NAMES[:35] == self.ORIGINAL_35

    def test_target_fields_at_end(self):
        assert META_NAMES[-2:] == ['target_is_minority', 'target_zscore']


class TestBorderline:
    def test_pure_neighbors_zero(self):
        """All same-class k-NN → borderline = 0.0."""
        n, k = 50, 5
        nn_indices = np.zeros((n, k + 1), dtype=int)
        for i in range(n):
            nn_indices[i] = [i] + list(range(k))  # Neighbors within same block
        y = np.zeros(n, dtype=int)  # All class 0
        result = _compute_borderline(nn_indices, y, k)
        np.testing.assert_array_almost_equal(result, 0.0)

    def test_mixed_neighbors_one(self):
        """All different-class k-NN → borderline = 1.0."""
        n, k = 20, 5
        y = np.array([0] * 10 + [1] * 10)
        # Each point's neighbors are all from the opposite class
        nn_indices = np.zeros((n, k + 1), dtype=int)
        for i in range(10):
            nn_indices[i] = [i] + list(range(10, 10 + k))  # class 0 → neighbors in class 1
        for i in range(10, 20):
            nn_indices[i] = [i] + list(range(k))  # class 1 → neighbors in class 0
        result = _compute_borderline(nn_indices, y, k)
        np.testing.assert_array_almost_equal(result, 1.0)


class TestFisher:
    def test_separable_higher_than_overlapping(self):
        """Well-separated Gaussians → higher Fisher ratio than overlapping."""
        np.random.seed(42)
        X_sep = np.vstack([
            np.random.randn(50, 3) + 10,
            np.random.randn(50, 3) - 10,
        ])
        X_overlap = np.vstack([
            np.random.randn(50, 3),
            np.random.randn(50, 3) + 0.1,
        ])
        y = np.array([0] * 50 + [1] * 50)
        fisher_sep = _compute_fisher_per_row(X_sep, y)
        fisher_overlap = _compute_fisher_per_row(X_overlap, y)
        assert fisher_sep.mean() > fisher_overlap.mean() * 5, (
            f"Separable Fisher {fisher_sep.mean():.3f} should be much larger "
            f"than overlapping {fisher_overlap.mean():.3f}"
        )


class TestHubScore:
    def test_star_topology(self):
        """Central point in a star → highest hub score."""
        np.random.seed(42)
        # Point 0 is central, others form a ring
        n, k = 20, 3
        nn_indices = np.zeros((n, k + 1), dtype=int)
        # All non-center points have point 0 as nearest neighbor
        for i in range(1, n):
            nn_indices[i] = [i, 0, (i % (n - 1)) + 1, ((i + 1) % (n - 1)) + 1]
        nn_indices[0] = [0, 1, 2, 3]
        hub = _compute_hub_scores(nn_indices)
        # Point 0 should have the highest hub score
        assert hub[0] == hub.max()
        assert hub[0] >= n - 2  # Most non-center points list 0


class TestLocalIntrinsicDim:
    def test_2d_manifold_in_10d(self):
        """2D plane embedded in 10D → local_id ≈ 2."""
        np.random.seed(42)
        n = 500
        plane = np.random.randn(n, 2)
        X = np.zeros((n, 10))
        X[:, :2] = plane  # 2D manifold embedded in 10D
        from sklearn.neighbors import NearestNeighbors
        k = 10
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        local_id = _compute_local_id(distances, k)
        mean_id = local_id.mean()
        assert 1.5 < mean_id < 4.0, f"Expected local ID near 2, got {mean_id:.2f}"


class TestSupervisedZeroWithoutY:
    def test_all_supervised_fields_zero(self):
        """With y=None, all supervised complexity fields should be 0."""
        np.random.seed(42)
        df = pd.DataFrame({'a': np.random.randn(50), 'b': np.random.randn(50)})
        numeric_cols, categorical_cols, col_stats, dataset_stats = compute_column_stats(df)
        mfs = compute_row_meta_features(
            df, None, numeric_cols, categorical_cols, col_stats, dataset_stats,
        )
        for mf in mfs:
            assert mf.fisher_ratio == 0.0
            assert mf.borderline == 0.0
            assert mf.knn_class_ratio == 0.0
            assert mf.linear_boundary_dist == 0.0
            assert mf.mi_contribution == 0.0


class TestLocalClustering:
    def test_fully_connected_neighborhood(self):
        """When all neighbors are mutual neighbors, clustering ≈ 1."""
        # 5 points in tight cluster → all are each other's neighbors
        np.random.seed(42)
        n, k = 5, 4
        nn_indices = np.array([
            [0, 1, 2, 3, 4],
            [1, 0, 2, 3, 4],
            [2, 0, 1, 3, 4],
            [3, 0, 1, 2, 4],
            [4, 0, 1, 2, 3],
        ])
        clustering = _compute_local_clustering(nn_indices, k)
        np.testing.assert_array_almost_equal(clustering, 1.0)


class TestLinearBoundaryDist:
    def test_separable_data_positive_distance(self):
        """Well-separated data → all points have positive boundary distance."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(50, 3) + 5,
            np.random.randn(50, 3) - 5,
        ])
        y = np.array([0] * 50 + [1] * 50)
        dist = _compute_linear_boundary_dist(X, y)
        assert (dist > 0).all()
        assert dist.mean() > 1.0


class TestEndToEnd:
    def test_meta_features_computed(self, meta_result):
        """Verify all 100 rows get meta-features."""
        assert len(meta_result) == 100

    def test_new_fields_populated(self, meta_result):
        """New fields should have some non-zero values across rows."""
        hub_scores = [m.hub_score for m in meta_result]
        assert max(hub_scores) > 0, "hub_score should be non-zero for some rows"

        local_ids = [m.local_intrinsic_dim for m in meta_result]
        assert np.mean(local_ids) > 0.5, "local_intrinsic_dim should be positive"

        surprises = [m.row_surprise for m in meta_result]
        assert np.mean(surprises) > 0, "row_surprise should be positive"

    def test_borderline_populated_with_classification(self, meta_result):
        """Borderline should have non-zero values for classification tasks."""
        borderlines = [m.borderline for m in meta_result]
        assert max(borderlines) > 0, "borderline should be non-zero for classification"

    def test_scale_fields_populated(self, meta_result):
        """Scale/magnitude fields should be non-trivial."""
        log_mags = [m.log_magnitude_mean for m in meta_result]
        assert any(v != 0.0 for v in log_mags), "log_magnitude_mean should be non-zero"
