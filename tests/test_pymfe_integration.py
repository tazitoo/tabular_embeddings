"""Tests for PyMFE dataset-level meta-feature augmentation."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_concept_regression import (
    augment_meta_with_pymfe,
    regress_features_on_probes,
)
from scripts.compare_sae_architectures import META_NAMES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_PROBES = len(META_NAMES)


@pytest.fixture
def sample_boundaries():
    """3 datasets: 100, 200, 200 rows."""
    return np.array([0, 100, 300, 500])


@pytest.fixture
def sample_datasets():
    return ["dataset_a", "dataset_b", "dataset_c"]


@pytest.fixture
def sample_meta():
    """Random row-level meta array (500 x 52)."""
    np.random.seed(42)
    return np.random.randn(500, N_PROBES)


@pytest.fixture
def pymfe_cache(tmp_path, sample_datasets):
    """PyMFE cache JSON with 3 features per dataset."""
    cache = {
        "dataset_a": {"feat_x": 1.0, "feat_y": 2.0, "feat_z": 3.0},
        "dataset_b": {"feat_x": 4.0, "feat_y": 5.0, "feat_z": 6.0},
        "dataset_c": {"feat_x": 7.0, "feat_y": 8.0, "feat_z": 9.0},
    }
    path = tmp_path / "pymfe_cache.json"
    with open(path, 'w') as f:
        json.dump(cache, f)
    return str(path)


# ---------------------------------------------------------------------------
# Tests: augment_meta_with_pymfe
# ---------------------------------------------------------------------------

class TestAugmentBroadcast:
    def test_output_shape(self, sample_meta, sample_boundaries, sample_datasets, pymfe_cache):
        """Augmented array has n_probes + n_pymfe columns."""
        result, names = augment_meta_with_pymfe(
            sample_meta, sample_boundaries, sample_datasets, pymfe_cache
        )
        assert result.shape == (500, N_PROBES + 3)
        assert len(names) == 3

    def test_pymfe_constant_within_dataset(
        self, sample_meta, sample_boundaries, sample_datasets, pymfe_cache
    ):
        """PyMFE features must be constant within each dataset's row range."""
        result, names = augment_meta_with_pymfe(
            sample_meta, sample_boundaries, sample_datasets, pymfe_cache
        )
        pymfe_cols = result[:, N_PROBES:]

        # Dataset A: rows 0-99
        for col in range(3):
            vals = pymfe_cols[0:100, col]
            assert np.all(vals == vals[0]), (
                f"PyMFE feature {col} not constant in dataset_a"
            )

        # Dataset B: rows 100-299
        for col in range(3):
            vals = pymfe_cols[100:300, col]
            assert np.all(vals == vals[0])

        # Dataset C: rows 300-499
        for col in range(3):
            vals = pymfe_cols[300:500, col]
            assert np.all(vals == vals[0])

    def test_pymfe_values_correct(
        self, sample_meta, sample_boundaries, sample_datasets, pymfe_cache
    ):
        """Broadcast values match the cache."""
        result, names = augment_meta_with_pymfe(
            sample_meta, sample_boundaries, sample_datasets, pymfe_cache
        )
        pymfe_cols = result[:, N_PROBES:]

        # feat_x is first (sorted), dataset_a = 1.0
        x_idx = names.index("feat_x")
        assert pymfe_cols[0, x_idx] == 1.0
        # dataset_b = 4.0
        assert pymfe_cols[100, x_idx] == 4.0
        # dataset_c = 7.0
        assert pymfe_cols[300, x_idx] == 7.0

    def test_original_probes_unchanged(
        self, sample_meta, sample_boundaries, sample_datasets, pymfe_cache
    ):
        """First N_PROBES columns remain identical to input."""
        result, _ = augment_meta_with_pymfe(
            sample_meta, sample_boundaries, sample_datasets, pymfe_cache
        )
        np.testing.assert_array_equal(result[:, :N_PROBES], sample_meta)


class TestAugmentMissingDataset:
    def test_missing_dataset_gets_zeros(self, sample_meta, sample_boundaries, tmp_path):
        """Datasets not in cache get zeros for PyMFE columns."""
        cache = {
            "dataset_a": {"feat_x": 1.0},
            # dataset_b missing
            "dataset_c": {"feat_x": 7.0},
        }
        path = tmp_path / "partial_cache.json"
        with open(path, 'w') as f:
            json.dump(cache, f)

        datasets = ["dataset_a", "dataset_b", "dataset_c"]
        boundaries = np.array([0, 100, 300, 500])

        result, names = augment_meta_with_pymfe(
            sample_meta, boundaries, datasets, str(path)
        )
        pymfe_cols = result[:, N_PROBES:]

        # dataset_b rows should be zero
        assert np.all(pymfe_cols[100:300] == 0.0)
        # dataset_a should have values
        assert pymfe_cols[0, 0] == 1.0

    def test_no_datasets_in_cache(self, sample_meta, sample_boundaries, tmp_path):
        """Empty cache → returns original meta unchanged."""
        path = tmp_path / "empty_cache.json"
        with open(path, 'w') as f:
            json.dump({}, f)

        datasets = ["dataset_a", "dataset_b"]
        boundaries = np.array([0, 100, 300])

        result, names = augment_meta_with_pymfe(
            sample_meta[:300], boundaries, datasets, str(path)
        )
        assert len(names) == 0
        np.testing.assert_array_equal(result, sample_meta[:300])


class TestR2ImprovementWithPyMFE:
    def test_informative_pymfe_increases_r2(self, tmp_path):
        """Dataset-level feature that predicts activation → R² increases."""
        np.random.seed(42)
        n_ds = 5
        n_per_ds = 100
        n_total = n_ds * n_per_ds

        # Row-level probes: random noise
        meta_base = np.random.randn(n_total, N_PROBES)

        # Dataset-level signal: each dataset gets a value that drives activation
        ds_signal = np.array([1.0, 3.0, 0.5, 2.0, 4.0])
        boundaries = np.array([i * n_per_ds for i in range(n_ds + 1)])
        datasets = [f"ds_{i}" for i in range(n_ds)]

        # Activation = dataset_signal + small noise
        activations = np.zeros((n_total, 2))
        for i in range(n_ds):
            s, e = boundaries[i], boundaries[i + 1]
            activations[s:e, 0] = ds_signal[i] + np.random.randn(n_per_ds) * 0.1
            activations[s:e, 1] = np.random.randn(n_per_ds)  # random feature

        # R² without PyMFE (probes are noise → low R²)
        r2_base = regress_features_on_probes(
            activations, meta_base, [0], alpha=1.0
        )

        # Create cache with the signal as a PyMFE feature
        cache = {
            f"ds_{i}": {"ds_signal": float(ds_signal[i])}
            for i in range(n_ds)
        }
        path = tmp_path / "signal_cache.json"
        with open(path, 'w') as f:
            json.dump(cache, f)

        # Augment
        meta_aug, pymfe_names = augment_meta_with_pymfe(
            meta_base, boundaries, datasets, str(path)
        )
        probe_names = list(META_NAMES) + pymfe_names

        r2_aug = regress_features_on_probes(
            activations, meta_aug, [0], alpha=1.0,
            probe_names=probe_names,
        )

        assert r2_aug[0]['r2'] > r2_base[0]['r2'] + 0.3, (
            f"PyMFE should substantially improve R²: "
            f"base={r2_base[0]['r2']:.3f}, augmented={r2_aug[0]['r2']:.3f}"
        )

        # Random feature should NOT improve much
        r2_random_base = regress_features_on_probes(
            activations, meta_base, [1], alpha=1.0
        )
        r2_random_aug = regress_features_on_probes(
            activations, meta_aug, [1], alpha=1.0,
            probe_names=probe_names,
        )
        # Allow small improvement from random correlation
        assert r2_random_aug[1]['r2'] < 0.3, (
            f"Random feature should remain unexplained: {r2_random_aug[1]['r2']:.3f}"
        )


class TestCollinearity:
    def test_high_condition_number_with_constant_features(self):
        """Dataset-constant columns increase condition number."""
        from sklearn.preprocessing import StandardScaler

        np.random.seed(42)
        n = 500

        # Row-varying features
        X_varied = np.random.randn(n, 10)

        # Dataset-constant: same value repeated within blocks
        X_const = np.zeros((n, 5))
        block_size = 100
        for i in range(5):
            s, e = i * block_size, (i + 1) * block_size
            X_const[s:e] = np.random.randn(5)

        X_aug = np.hstack([X_varied, X_const])

        cond_varied = np.linalg.cond(StandardScaler().fit_transform(X_varied))
        cond_aug = np.linalg.cond(StandardScaler().fit_transform(X_aug))

        # Condition number should increase with dataset-constant features
        assert cond_aug > cond_varied


class TestProbeNamesPassthrough:
    def test_extended_probe_names_in_output(self):
        """Regression with extended probe_names reports correct coefficient keys."""
        np.random.seed(42)
        n = 200
        n_extra = 3
        extended_names = list(META_NAMES) + [f"pymfe_{i}" for i in range(n_extra)]
        meta = np.random.randn(n, N_PROBES + n_extra)

        # Feature driven by last PyMFE column
        activations = np.zeros((n, 1))
        activations[:, 0] = meta[:, -1] * 5 + 1.0

        results = regress_features_on_probes(
            activations, meta, [0], alpha=1.0, probe_names=extended_names
        )

        # Coefficients should include PyMFE names
        assert f"pymfe_{n_extra - 1}" in results[0]['coefficients']
        assert len(results[0]['coefficients']) == N_PROBES + n_extra

        # Top probe should be the PyMFE feature
        top_name = results[0]['top_probes'][0][0]
        assert top_name == f"pymfe_{n_extra - 1}", (
            f"Expected pymfe_{n_extra - 1} as top probe, got {top_name}"
        )
