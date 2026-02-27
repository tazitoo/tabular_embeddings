"""Tests for cross-model SAE feature matching via dataset-mean activations."""

import numpy as np
import pytest

from scripts.match_cross_model_features import (
    cluster_consensus_concepts,
    compute_cross_model_correlation,
    compute_matching_summary,
    get_alive_features,
    match_all_bands,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def simple_ds_means():
    """(10 datasets, 64 hidden_dim) with known alive/dead structure."""
    rng = np.random.RandomState(42)
    means = rng.rand(10, 64).astype(np.float32)
    # Kill features 0-9 (set max < threshold)
    means[:, :10] = 0.0001 * rng.rand(10, 10)
    return means


# ---------------------------------------------------------------------------
# TestGetAliveFeatures
# ---------------------------------------------------------------------------

class TestGetAliveFeatures:

    def test_correct_shape(self, simple_ds_means):
        vecs, idxs = get_alive_features(simple_ds_means, 0, 64, threshold=0.001)
        assert vecs.ndim == 2
        assert vecs.shape[1] == 10  # n_datasets
        assert len(idxs) == vecs.shape[0]

    def test_dead_excluded(self, simple_ds_means):
        """Features 0-9 are dead (max < 0.001), should be excluded."""
        vecs, idxs = get_alive_features(simple_ds_means, 0, 64, threshold=0.001)
        # None of the global indices should be < 10
        assert all(idx >= 10 for idx in idxs)

    def test_global_indices_correct(self, simple_ds_means):
        """Global indices should be offset by band_start."""
        vecs, idxs = get_alive_features(simple_ds_means, 20, 40, threshold=0.001)
        assert all(20 <= idx < 40 for idx in idxs)

    def test_all_dead_band(self):
        """Band where all features are dead returns empty arrays."""
        ds_means = np.zeros((10, 32), dtype=np.float32)
        vecs, idxs = get_alive_features(ds_means, 0, 32, threshold=0.001)
        assert vecs.shape == (0, 10)
        assert len(idxs) == 0

    def test_threshold_sensitivity(self, rng):
        """Higher threshold means fewer alive features."""
        ds_means = rng.rand(10, 32).astype(np.float32) * 0.5
        # Set some features to barely alive
        ds_means[:, :5] = 0.01
        ds_means[0, :5] = 0.1  # max = 0.1

        _, idxs_low = get_alive_features(ds_means, 0, 32, threshold=0.05)
        _, idxs_high = get_alive_features(ds_means, 0, 32, threshold=0.2)
        assert len(idxs_low) >= len(idxs_high)


# ---------------------------------------------------------------------------
# TestComputeCrossModelCorrelation
# ---------------------------------------------------------------------------

class TestComputeCrossModelCorrelation:

    def test_identical_vectors_r1(self):
        """Identical feature vectors should have |r| = 1."""
        vec = np.array([[1, 2, 3, 4, 5.0]])
        vectors = np.vstack([vec, vec])
        labels = [("A", 0), ("B", 0)]
        corr = compute_cross_model_correlation(vectors, labels)
        np.testing.assert_allclose(corr[0, 1], 1.0, atol=1e-6)

    def test_negation_r1(self):
        """Negated vector should have |r| = 1 (we use absolute value)."""
        vec = np.array([[1, 2, 3, 4, 5.0]])
        vectors = np.vstack([vec, -vec])
        labels = [("A", 0), ("B", 0)]
        corr = compute_cross_model_correlation(vectors, labels)
        np.testing.assert_allclose(corr[0, 1], 1.0, atol=1e-6)

    def test_uncorrelated_near_zero(self, rng):
        """Uncorrelated random vectors should have |r| ~ 0."""
        n_datasets = 1000  # large n for stable estimate
        v1 = rng.randn(1, n_datasets)
        v2 = rng.randn(1, n_datasets)
        vectors = np.vstack([v1, v2])
        labels = [("A", 0), ("B", 0)]
        corr = compute_cross_model_correlation(vectors, labels)
        assert corr[0, 1] < 0.1

    def test_constant_feature_no_nan(self):
        """Constant feature vectors should give r=0, not NaN."""
        vectors = np.array([[5.0, 5.0, 5.0], [1.0, 2.0, 3.0]])
        labels = [("A", 0), ("B", 0)]
        corr = compute_cross_model_correlation(vectors, labels)
        assert not np.any(np.isnan(corr))
        assert corr[0, 1] == 0.0  # constant vs varying = 0

    def test_symmetric(self, rng):
        """Correlation matrix should be symmetric."""
        vectors = rng.randn(5, 20)
        labels = [("A", i) for i in range(5)]
        corr = compute_cross_model_correlation(vectors, labels)
        np.testing.assert_allclose(corr, corr.T, atol=1e-10)

    def test_empty_input(self):
        """Empty input returns empty matrix."""
        vectors = np.zeros((0, 10))
        labels = []
        corr = compute_cross_model_correlation(vectors, labels)
        assert corr.shape == (0, 0)


# ---------------------------------------------------------------------------
# TestClusterConsensusConcepts
# ---------------------------------------------------------------------------

class TestClusterConsensusConcepts:

    def test_two_model_consensus(self):
        """Two highly correlated features from different models form consensus."""
        # Two identical vectors from different models
        vectors = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.1, 2.1, 3.1, 4.1, 5.1],  # nearly identical
            [9.0, 1.0, 0.0, 8.0, 2.0],  # uncorrelated
        ])
        labels = [("ModelA", 0), ("ModelB", 5), ("ModelC", 10)]
        corr = compute_cross_model_correlation(vectors, labels)
        consensus, model_spec = cluster_consensus_concepts(
            corr, labels, corr_threshold=0.5
        )
        # First two should form a consensus concept
        assert len(consensus) >= 1
        # At least one concept has >= 2 models
        assert any(c["n_models"] >= 2 for c in consensus)

    def test_no_consensus(self):
        """Uncorrelated features from different models → all model-specific."""
        rng = np.random.RandomState(42)
        vectors = rng.randn(4, 50)  # large n_datasets for stable 0 corr
        labels = [("A", 0), ("B", 1), ("C", 2), ("D", 3)]
        corr = compute_cross_model_correlation(vectors, labels)
        consensus, model_spec = cluster_consensus_concepts(
            corr, labels, corr_threshold=0.9
        )
        # All should be model-specific at high threshold
        assert len(consensus) == 0
        total_spec = sum(len(v) for v in model_spec.values())
        assert total_spec == 4

    def test_multi_model_cluster(self):
        """Features from 3 models that are all correlated."""
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        vectors = np.array([
            base,
            base + 0.01 * np.random.randn(8),
            base + 0.02 * np.random.randn(8),
        ])
        labels = [("A", 0), ("B", 10), ("C", 20)]
        corr = compute_cross_model_correlation(vectors, labels)
        consensus, _ = cluster_consensus_concepts(corr, labels, corr_threshold=0.5)
        assert len(consensus) == 1
        assert consensus[0]["n_models"] == 3

    def test_mean_inter_model_r(self):
        """mean_inter_model_r should only count cross-model pairs."""
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        vectors = np.array([
            base,        # A, idx 0
            base * 0.9,  # B, idx 5
        ])
        labels = [("A", 0), ("B", 5)]
        corr = compute_cross_model_correlation(vectors, labels)
        consensus, _ = cluster_consensus_concepts(corr, labels, corr_threshold=0.5)
        assert len(consensus) == 1
        # r should be very high (nearly identical scaled vectors)
        assert consensus[0]["mean_inter_model_r"] > 0.99

    def test_no_feature_duplication(self, rng):
        """Every feature appears in exactly one group (consensus or model-specific)."""
        vectors = rng.randn(10, 20)
        # Make some correlated pairs across models
        vectors[1] = vectors[0] + 0.01 * rng.randn(20)
        labels = [
            ("A", 0), ("B", 1), ("A", 2), ("B", 3), ("C", 4),
            ("A", 5), ("B", 6), ("C", 7), ("A", 8), ("B", 9),
        ]
        corr = compute_cross_model_correlation(vectors, labels)
        consensus, model_spec = cluster_consensus_concepts(
            corr, labels, corr_threshold=0.5
        )

        # Collect all features
        seen = set()
        for c in consensus:
            for model, indices in c["members"].items():
                for idx in indices:
                    key = (model, idx)
                    assert key not in seen, f"Duplicate: {key}"
                    seen.add(key)
        for model, indices in model_spec.items():
            for idx in indices:
                key = (model, idx)
                assert key not in seen, f"Duplicate: {key}"
                seen.add(key)

        # Total should match input
        assert len(seen) == len(labels)

    def test_single_feature(self):
        """Single feature should be model-specific."""
        corr = np.array([[1.0]])
        labels = [("A", 0)]
        consensus, model_spec = cluster_consensus_concepts(corr, labels)
        assert len(consensus) == 0
        assert model_spec == {"A": [0]}


# ---------------------------------------------------------------------------
# TestComputeMatchingSummary
# ---------------------------------------------------------------------------

class TestComputeMatchingSummary:

    def test_by_n_models_sums_to_total(self):
        consensus = [
            {"members": {"A": [0], "B": [1]}, "n_models": 2, "mean_inter_model_r": 0.6},
            {"members": {"A": [2], "B": [3], "C": [4]}, "n_models": 3, "mean_inter_model_r": 0.7},
        ]
        model_spec = {"A": [5, 6], "D": [7]}
        model_names = ["A", "B", "C", "D"]
        total_alive = {"A": 4, "B": 2, "C": 1, "D": 1}

        summary = compute_matching_summary(consensus, model_spec, model_names, total_alive)

        assert summary["total_consensus"] == 2
        n_models_sum = sum(summary["by_n_models"].values())
        assert n_models_sum == 2  # 2 consensus concepts total

    def test_coverage_fractions_valid(self):
        consensus = [
            {"members": {"A": [0, 1], "B": [2]}, "n_models": 2, "mean_inter_model_r": 0.6},
        ]
        model_spec = {"A": [3]}
        model_names = ["A", "B"]
        total_alive = {"A": 3, "B": 1}

        summary = compute_matching_summary(consensus, model_spec, model_names, total_alive)

        for m in model_names:
            frac = summary["per_model_coverage"][m]["frac_consensus"]
            assert 0.0 <= frac <= 1.0

    def test_empty_inputs(self):
        summary = compute_matching_summary([], {}, ["A"], {"A": 0})
        assert summary["total_consensus"] == 0
        assert summary["total_model_specific"] == 0
        assert summary["per_model_coverage"]["A"]["frac_consensus"] == 0.0


# ---------------------------------------------------------------------------
# TestMatchAllBands (integration)
# ---------------------------------------------------------------------------

class TestMatchAllBands:

    def _make_mock_config(self, input_dim, hidden_dim, mat_dims):
        """Create a minimal SAEConfig-like object for testing."""
        from dataclasses import dataclass, field
        from typing import List, Optional

        @dataclass
        class MockConfig:
            input_dim: int = 128
            hidden_dim: int = 256
            matryoshka_dims: Optional[List[int]] = None

        config = MockConfig(input_dim=input_dim, hidden_dim=hidden_dim,
                            matryoshka_dims=mat_dims)
        return config

    def test_synthetic_pipeline(self):
        """3-model pipeline with known correlation structure."""
        rng = np.random.RandomState(42)
        n_datasets = 20
        hidden = 64

        # Model A and B share a common signal, C is independent
        shared_signal = rng.randn(n_datasets)

        ds_means_a = rng.randn(n_datasets, hidden) * 0.1
        ds_means_b = rng.randn(n_datasets, hidden) * 0.1
        ds_means_c = rng.randn(n_datasets, hidden) * 0.1

        # Feature 5 in A and feature 10 in B both encode shared_signal
        ds_means_a[:, 5] = shared_signal + 0.01 * rng.randn(n_datasets)
        ds_means_b[:, 10] = shared_signal + 0.01 * rng.randn(n_datasets)

        mat_dims = [16, 32, 64]
        config = self._make_mock_config(32, hidden, mat_dims)

        model_ds_means = {"A": ds_means_a, "B": ds_means_b, "C": ds_means_c}
        model_configs = {"A": config, "B": config, "C": config}
        model_names = ["A", "B", "C"]

        result = match_all_bands(
            model_ds_means, model_configs, model_names,
            datasets=[f"ds{i}" for i in range(n_datasets)],
            corr_threshold=0.5, alive_threshold=0.001,
        )

        assert "bands" in result
        assert "summary" in result
        assert "S1" in result["bands"]

    def test_output_schema(self):
        """Verify output has correct structure."""
        rng = np.random.RandomState(42)
        n_datasets = 15
        hidden = 32

        config = self._make_mock_config(16, hidden, [16, 32])
        ds_means = rng.rand(n_datasets, hidden).astype(np.float32)

        result = match_all_bands(
            {"A": ds_means, "B": ds_means * 0.9 + 0.01 * rng.randn(n_datasets, hidden)},
            {"A": config, "B": config},
            ["A", "B"],
            datasets=[f"ds{i}" for i in range(n_datasets)],
        )

        # Check band structure
        for band_label, band_data in result["bands"].items():
            assert "n_consensus" in band_data
            assert "n_model_specific" in band_data
            assert "concepts" in band_data
            assert "model_specific" in band_data
            for concept in band_data["concepts"]:
                assert "id" in concept
                assert "members" in concept
                assert "n_models" in concept
                assert "mean_inter_model_r" in concept
                assert "centroid" in concept

        # Check summary
        s = result["summary"]
        assert "total_consensus" in s
        assert "total_model_specific" in s
        assert "by_n_models" in s
        assert "per_model_coverage" in s

    def test_accounting_identity(self):
        """consensus + model_specific = total alive per model per band."""
        rng = np.random.RandomState(42)
        n_datasets = 15
        hidden = 32

        config = self._make_mock_config(16, hidden, [16, 32])

        model_ds_means = {
            "A": rng.rand(n_datasets, hidden).astype(np.float32),
            "B": rng.rand(n_datasets, hidden).astype(np.float32),
        }

        result = match_all_bands(
            model_ds_means,
            {"A": config, "B": config},
            ["A", "B"],
            datasets=[f"ds{i}" for i in range(n_datasets)],
            corr_threshold=0.5,
            alive_threshold=0.001,
        )

        for band_label, band_data in result["bands"].items():
            # Count features per model in consensus
            consensus_count = {"A": 0, "B": 0}
            for concept in band_data["concepts"]:
                for m, indices in concept["members"].items():
                    consensus_count[m] += len(indices)

            # Count model-specific
            specific_count = {
                m: len(band_data["model_specific"].get(m, []))
                for m in ["A", "B"]
            }

            # Get total alive per model for this band
            from scripts.match_cross_model_features import (
                get_alive_features,
                get_matryoshka_bands,
            )
            bands = get_matryoshka_bands(config)
            for m in ["A", "B"]:
                for bl, start, end in bands:
                    if bl == band_label:
                        _, alive_idxs = get_alive_features(
                            model_ds_means[m], start, end, threshold=0.001
                        )
                        total_alive = len(alive_idxs)
                        accounted = consensus_count[m] + specific_count[m]
                        assert accounted == total_alive, (
                            f"Band {band_label}, model {m}: "
                            f"{accounted} accounted != {total_alive} alive"
                        )
