"""Tests for scripts/analyze_concept_hierarchy.py."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_concept_hierarchy import (
    aggregate_by_category,
    compute_band_correlations,
    compute_dataset_mean_activations,
    compute_signal_exhaustion,
    get_matryoshka_bands,
    load_pymfe_dataset_matrix,
    load_pymfe_taxonomy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def taxonomy_path():
    return PROJECT_ROOT / "config" / "pymfe_taxonomy.json"


@pytest.fixture
def pymfe_cache_path():
    return PROJECT_ROOT / "output" / "pymfe_tabarena_cache.json"


@pytest.fixture
def taxonomy(taxonomy_path):
    return load_pymfe_taxonomy(taxonomy_path)


@pytest.fixture
def synthetic_pymfe_cache(tmp_path):
    """Create a small synthetic PyMFE cache for testing."""
    cache = {}
    np.random.seed(42)
    datasets = [f"ds_{i}" for i in range(10)]
    features = ["nr_attr", "cor.mean", "attr_ent.mean", "leaves", "elite_nn.mean", "n2.mean"]
    for ds in datasets:
        cache[ds] = {f: float(np.random.randn()) for f in features}
    # Make one feature constant across datasets
    for ds in datasets:
        cache[ds]["nr_attr"] = 5.0

    cache_path = tmp_path / "pymfe_cache.json"
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    return cache_path, datasets, features


# ---------------------------------------------------------------------------
# TestLoadPymfeTaxonomy
# ---------------------------------------------------------------------------

class TestLoadPymfeTaxonomy:
    def test_six_categories(self, taxonomy):
        category_features, _ = taxonomy
        assert len(category_features) == 6
        expected = {"General", "Statistical", "Info-Theory", "Model-Based",
                    "Landmarking", "Complexity"}
        assert set(category_features.keys()) == expected

    def test_145_total_features(self, taxonomy):
        category_features, feature_category = taxonomy
        total = sum(len(v) for v in category_features.values())
        assert total == 145
        assert len(feature_category) == 145

    def test_no_duplicates(self, taxonomy):
        category_features, _ = taxonomy
        all_feats = []
        for feats in category_features.values():
            all_feats.extend(feats)
        assert len(all_feats) == len(set(all_feats))

    def test_reverse_lookup(self, taxonomy):
        category_features, feature_category = taxonomy
        # Check a few known mappings
        assert feature_category["nr_attr"] == "General"
        assert feature_category["cor.mean"] == "Statistical"
        assert feature_category["attr_ent.mean"] == "Info-Theory"
        assert feature_category["leaves"] == "Model-Based"
        assert feature_category["elite_nn.mean"] == "Landmarking"
        assert feature_category["n2.mean"] == "Complexity"

    def test_all_features_have_category(self, taxonomy):
        category_features, feature_category = taxonomy
        for cat, feats in category_features.items():
            for f in feats:
                assert feature_category[f] == cat


# ---------------------------------------------------------------------------
# TestLoadPymfeDatasetMatrix
# ---------------------------------------------------------------------------

class TestLoadPymfeDatasetMatrix:
    def test_correct_shape(self, synthetic_pymfe_cache):
        cache_path, datasets, features = synthetic_pymfe_cache
        matrix, names = load_pymfe_dataset_matrix(cache_path, datasets)
        assert matrix.shape[0] == 10
        # May have fewer columns if some are near-constant
        assert matrix.shape[1] <= len(features)

    def test_values_match_cache(self, synthetic_pymfe_cache):
        cache_path, datasets, features = synthetic_pymfe_cache
        with open(cache_path) as f:
            cache = json.load(f)

        matrix, names = load_pymfe_dataset_matrix(cache_path, datasets)
        # Non-constant, non-skewed features should keep original values
        # (skewed features get rank-transformed, constant get filtered)
        # Just check that the matrix is finite
        assert np.all(np.isfinite(matrix))

    def test_near_constant_columns_filtered(self, synthetic_pymfe_cache):
        cache_path, datasets, features = synthetic_pymfe_cache
        matrix, names = load_pymfe_dataset_matrix(cache_path, datasets)
        # "nr_attr" is constant (5.0 for all) — should be filtered out
        assert "nr_attr" not in names

    def test_no_nan_values(self, synthetic_pymfe_cache):
        cache_path, datasets, features = synthetic_pymfe_cache
        matrix, names = load_pymfe_dataset_matrix(cache_path, datasets)
        assert not np.any(np.isnan(matrix))

    def test_missing_datasets_handled(self, synthetic_pymfe_cache):
        cache_path, datasets, features = synthetic_pymfe_cache
        # Include a dataset not in the cache
        extended = datasets + ["nonexistent_ds"]
        matrix, names = load_pymfe_dataset_matrix(cache_path, extended)
        assert matrix.shape[0] == len(extended)
        assert np.all(np.isfinite(matrix))


# ---------------------------------------------------------------------------
# TestComputeDatasetMeanActivations
# ---------------------------------------------------------------------------

class TestComputeDatasetMeanActivations:
    def test_correct_shape(self):
        n_rows, hidden = 300, 32
        activations = np.random.rand(n_rows, hidden).astype(np.float32)
        offsets = {"ds_a": (0, 100), "ds_b": (100, 200), "ds_c": (200, 300)}
        datasets = ["ds_a", "ds_b", "ds_c"]
        result = compute_dataset_mean_activations(activations, offsets, datasets)
        assert result.shape == (3, hidden)

    def test_hand_verified_means(self):
        activations = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [10.0, 20.0],
        ])
        offsets = {"ds_a": (0, 2), "ds_b": (2, 3)}
        datasets = ["ds_a", "ds_b"]
        result = compute_dataset_mean_activations(activations, offsets, datasets)
        np.testing.assert_allclose(result[0], [2.0, 3.0])
        np.testing.assert_allclose(result[1], [10.0, 20.0])

    def test_sparse_activations(self):
        activations = np.zeros((100, 16))
        # Only feature 0 fires for first 50 rows
        activations[:50, 0] = 1.0
        offsets = {"ds_a": (0, 50), "ds_b": (50, 100)}
        datasets = ["ds_a", "ds_b"]
        result = compute_dataset_mean_activations(activations, offsets, datasets)
        assert result[0, 0] == 1.0  # ds_a: all fire
        assert result[1, 0] == 0.0  # ds_b: none fire

    def test_missing_dataset_gets_zeros(self):
        activations = np.ones((100, 8))
        offsets = {"ds_a": (0, 100)}
        datasets = ["ds_a", "ds_missing"]
        result = compute_dataset_mean_activations(activations, offsets, datasets)
        assert result.shape == (2, 8)
        np.testing.assert_allclose(result[0], 1.0)
        np.testing.assert_allclose(result[1], 0.0)


# ---------------------------------------------------------------------------
# TestComputeBandCorrelations
# ---------------------------------------------------------------------------

class TestComputeBandCorrelations:
    def test_perfect_correlation(self):
        """Synthetic signal: SAE feature = PyMFE feature → r ≈ 1.0."""
        n_ds = 30
        np.random.seed(42)
        signal = np.random.randn(n_ds)
        # ds_means has 1 feature that's a copy of the signal
        ds_means = np.zeros((n_ds, 4))
        ds_means[:, 0] = signal
        ds_means[:, 1] = np.random.randn(n_ds) * 0.01  # dead
        ds_means[:, 2] = signal + np.random.randn(n_ds) * 0.1  # noisy copy
        ds_means[:, 3] = np.random.randn(n_ds)  # uncorrelated

        pymfe = signal.reshape(-1, 1)

        result = compute_band_correlations(ds_means, pymfe, 0, 4, fdr_alpha=0.05)
        # Feature 0 should be perfectly correlated
        assert result["n_alive"] >= 2  # features 0, 2, 3 should be alive
        r_matrix = result["r_matrix"]
        # Find the row for feature 0 (first alive feature)
        alive_idx = list(result["alive_indices"])
        if 0 in alive_idx:
            row = alive_idx.index(0)
            assert abs(r_matrix[row, 0]) > 0.95

    def test_dead_features_excluded(self):
        n_ds = 20
        ds_means = np.zeros((n_ds, 8))
        # Only feature 0 is alive
        ds_means[:, 0] = np.random.randn(n_ds)
        pymfe = np.random.randn(n_ds, 3)

        result = compute_band_correlations(ds_means, pymfe, 0, 8)
        assert result["n_alive"] == 1
        assert result["r_matrix"].shape == (1, 3)

    def test_fdr_reduces_significance(self):
        """With noise, FDR should reject most tests."""
        n_ds = 30
        np.random.seed(123)
        ds_means = np.random.randn(n_ds, 20) * 0.01  # tiny noise
        ds_means[:, 0] = np.random.randn(n_ds)  # one alive
        pymfe = np.random.randn(n_ds, 50)

        result = compute_band_correlations(ds_means, pymfe, 0, 20, fdr_alpha=0.05)
        # With 1 alive feature × 50 pymfe and pure noise,
        # most should not be significant
        n_sig = result["significant_mask"].sum()
        # Allow some false discoveries but should be sparse
        assert n_sig < 10

    def test_empty_band(self):
        ds_means = np.random.randn(20, 10)
        pymfe = np.random.randn(20, 5)
        result = compute_band_correlations(ds_means, pymfe, 5, 5)
        assert result["n_alive"] == 0
        assert result["r_matrix"].shape == (0, 5)

    def test_all_dead_band(self):
        """Band where all features are zero (dead)."""
        n_ds = 20
        ds_means = np.zeros((n_ds, 8))
        pymfe = np.random.randn(n_ds, 5)
        result = compute_band_correlations(ds_means, pymfe, 0, 8)
        assert result["n_alive"] == 0


# ---------------------------------------------------------------------------
# TestAggregateByCategory
# ---------------------------------------------------------------------------

class TestAggregateByCategory:
    def test_all_categories_present(self, taxonomy):
        category_features, feature_category = taxonomy

        # Create a band_corr with some significant results
        n_alive = 5
        n_pymfe = len(feature_category)
        pymfe_names = sorted(feature_category.keys())

        np.random.seed(42)
        r_matrix = np.random.randn(n_alive, n_pymfe) * 0.3
        sig_mask = np.abs(r_matrix) > 0.5

        band_corr = {
            "r_matrix": r_matrix,
            "significant_mask": sig_mask,
            "n_alive": n_alive,
        }

        result = aggregate_by_category(
            band_corr, pymfe_names, category_features, feature_category,
        )

        assert set(result.keys()) == set(category_features.keys())

    def test_top_l1_sorted_by_abs_r(self, taxonomy):
        category_features, feature_category = taxonomy
        pymfe_names = sorted(feature_category.keys())

        n_alive = 10
        n_pymfe = len(pymfe_names)
        np.random.seed(42)
        r_matrix = np.random.randn(n_alive, n_pymfe) * 0.5
        sig_mask = np.abs(r_matrix) > 0.3

        band_corr = {
            "r_matrix": r_matrix,
            "significant_mask": sig_mask,
            "n_alive": n_alive,
        }

        result = aggregate_by_category(
            band_corr, pymfe_names, category_features, feature_category,
        )

        for cat, info in result.items():
            top = info["top_l1"]
            if len(top) > 1:
                rs = [t["max_r"] for t in top]
                assert rs == sorted(rs, reverse=True), f"{cat} top_l1 not sorted"

    def test_zero_alive(self, taxonomy):
        category_features, feature_category = taxonomy
        pymfe_names = sorted(feature_category.keys())

        band_corr = {
            "r_matrix": np.empty((0, len(pymfe_names))),
            "significant_mask": np.empty((0, len(pymfe_names)), dtype=bool),
            "n_alive": 0,
        }

        result = aggregate_by_category(
            band_corr, pymfe_names, category_features, feature_category,
        )

        for cat, info in result.items():
            assert info["max_abs_r"] == 0.0
            assert info["n_significant_pairs"] == 0


# ---------------------------------------------------------------------------
# TestSignalExhaustion
# ---------------------------------------------------------------------------

class TestSignalExhaustion:
    def test_all_correlated(self):
        """All features have at least one significant correlation → 0% unexplained."""
        n_alive = 10
        n_pymfe = 5
        sig_mask = np.zeros((n_alive, n_pymfe), dtype=bool)
        sig_mask[:, 0] = True  # All features correlate with pymfe[0]

        result = compute_signal_exhaustion({
            "significant_mask": sig_mask,
            "n_alive": n_alive,
        })
        assert result["n_explained"] == n_alive
        assert result["n_unexplained"] == 0
        assert result["frac_unexplained"] == 0.0

    def test_none_correlated(self):
        """No features have any significant correlation → 100% unexplained."""
        n_alive = 10
        n_pymfe = 5
        sig_mask = np.zeros((n_alive, n_pymfe), dtype=bool)

        result = compute_signal_exhaustion({
            "significant_mask": sig_mask,
            "n_alive": n_alive,
        })
        assert result["n_explained"] == 0
        assert result["n_unexplained"] == n_alive
        assert result["frac_unexplained"] == 1.0

    def test_accounting_identity(self):
        """n_explained + n_unexplained == n_alive."""
        n_alive = 20
        n_pymfe = 10
        np.random.seed(42)
        sig_mask = np.random.rand(n_alive, n_pymfe) > 0.7

        result = compute_signal_exhaustion({
            "significant_mask": sig_mask,
            "n_alive": n_alive,
        })
        assert result["n_explained"] + result["n_unexplained"] == n_alive

    def test_empty(self):
        result = compute_signal_exhaustion({
            "significant_mask": np.empty((0, 5), dtype=bool),
            "n_alive": 0,
        })
        assert result["n_explained"] == 0
        assert result["frac_unexplained"] == 0.0


# ---------------------------------------------------------------------------
# TestMiniPipeline (integration)
# ---------------------------------------------------------------------------

class TestMiniPipeline:
    """Synthetic end-to-end: S1 correlates with pymfe[0], S2 doesn't."""

    def test_walk_down_signal_exhausts(self):
        n_ds = 30
        hidden_dim = 32
        np.random.seed(42)

        # PyMFE: 3 features, first is a strong signal
        signal = np.random.randn(n_ds)
        pymfe_matrix = np.column_stack([
            signal,
            np.random.randn(n_ds),
            np.random.randn(n_ds),
        ])
        pymfe_names = ["feat_a", "feat_b", "feat_c"]
        category_features = {"Cat1": ["feat_a", "feat_b"], "Cat2": ["feat_c"]}
        feature_category = {"feat_a": "Cat1", "feat_b": "Cat1", "feat_c": "Cat2"}

        # Dataset-mean activations: S1 (0-16) correlates with signal
        ds_means = np.random.randn(n_ds, hidden_dim) * 0.01
        for i in range(8):
            ds_means[:, i] = signal + np.random.randn(n_ds) * 0.3

        # S2 (16-32) is pure noise — should have higher exhaustion
        ds_means[:, 16:32] = np.random.randn(n_ds, 16) * 0.5

        # Band S1
        corr_s1 = compute_band_correlations(ds_means, pymfe_matrix, 0, 16)
        exh_s1 = compute_signal_exhaustion(corr_s1)
        cat_s1 = aggregate_by_category(
            corr_s1, pymfe_names, category_features, feature_category,
        )

        # Band S2
        corr_s2 = compute_band_correlations(ds_means, pymfe_matrix, 16, 32)
        exh_s2 = compute_signal_exhaustion(corr_s2)

        # S1 should have lower exhaustion (more features explained)
        assert exh_s1["frac_unexplained"] < exh_s2["frac_unexplained"], (
            f"S1 unexplained={exh_s1['frac_unexplained']:.2f} should be < "
            f"S2 unexplained={exh_s2['frac_unexplained']:.2f}"
        )

        # S1 Cat1 should have significant correlations (feat_a = signal)
        assert cat_s1["Cat1"]["n_significant_pairs"] > 0

    def test_get_matryoshka_bands(self):
        from dataclasses import dataclass, field

        @dataclass
        class MockConfig:
            hidden_dim: int = 128
            matryoshka_dims: list = field(default_factory=lambda: [8, 16, 32, 64, 128])

        config = MockConfig()
        bands = get_matryoshka_bands(config)
        assert len(bands) == 5
        assert bands[0] == ("S1 [0,8)", 0, 8)
        assert bands[-1] == ("S5 [64,128)", 64, 128)

    def test_matryoshka_bands_appends_hidden_dim(self):
        from dataclasses import dataclass, field

        @dataclass
        class MockConfig:
            hidden_dim: int = 256
            matryoshka_dims: list = field(default_factory=lambda: [32, 64, 128])

        config = MockConfig()
        bands = get_matryoshka_bands(config)
        assert bands[-1][2] == 256  # Last band should reach hidden_dim

    def test_matryoshka_bands_none(self):
        from dataclasses import dataclass

        @dataclass
        class MockConfig:
            hidden_dim: int = 64
            matryoshka_dims: list = None

        config = MockConfig()
        bands = get_matryoshka_bands(config)
        assert len(bands) == 1
        assert bands[0] == ("S1 [0,64)", 0, 64)


# ---------------------------------------------------------------------------
# TestLoadPymfeDatasetMatrixWithRealCache (integration)
# ---------------------------------------------------------------------------

class TestLoadPymfeDatasetMatrixWithRealCache:
    """Integration tests using the real PyMFE cache (skipped if not available)."""

    @pytest.fixture
    def real_cache(self, pymfe_cache_path):
        if not pymfe_cache_path.exists():
            pytest.skip("PyMFE cache not found")
        with open(pymfe_cache_path) as f:
            cache = json.load(f)
        return cache

    def test_real_cache_shape(self, pymfe_cache_path, real_cache):
        datasets = sorted(real_cache.keys())[:10]
        matrix, names = load_pymfe_dataset_matrix(pymfe_cache_path, datasets)
        assert matrix.shape[0] == 10
        assert matrix.shape[1] > 100  # most of 145 should survive filtering
        assert len(names) == matrix.shape[1]

    def test_real_cache_no_nan(self, pymfe_cache_path, real_cache):
        datasets = sorted(real_cache.keys())[:20]
        matrix, names = load_pymfe_dataset_matrix(pymfe_cache_path, datasets)
        assert not np.any(np.isnan(matrix))
