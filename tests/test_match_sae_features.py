"""Tests for scripts/match_sae_features.py — row-level cross-model SAE feature matching."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.match_sae_features import (
    compute_cross_correlation,
    compute_sae_activations,
    get_alive_mask,
    match_hungarian,
    match_many_to_one,
    match_model_pair,
    match_mutual_nearest_neighbors,
)
import importlib as _importlib
_mnn_mod = _importlib.import_module("scripts.matching.01_match_sae_concepts_mnn")
filter_matches_by_noise_floor = _mnn_mod.filter_matches_by_noise_floor


# ── TestGetAliveMask ───────────────────────────────────────────────────────


class TestGetAliveMask:
    def test_correct_shape(self):
        acts = np.array([[0.5, 0.0, 0.3], [0.0, 0.0, 0.1]])
        mask = get_alive_mask(acts)
        assert mask.shape == (3,)

    def test_dead_features_excluded(self):
        acts = np.array([[0.5, 0.0, 0.3], [0.2, 0.0, 0.0]])
        mask = get_alive_mask(acts)
        assert mask[0] is np.True_
        assert mask[1] is np.False_
        assert mask[2] is np.True_

    def test_any_positive_is_alive(self):
        """With TopK, any activation > 0 means the feature fired."""
        acts = np.array([[0.01, 0.001, 0.0005]])
        mask = get_alive_mask(acts)
        assert mask.sum() == 3  # all > 0

    def test_all_dead(self):
        acts = np.zeros((10, 5))
        mask = get_alive_mask(acts)
        assert mask.sum() == 0


# ── TestComputeCrossCorrelation ────────────────────────────────────────────


class TestComputeCrossCorrelation:
    def test_identical_features(self):
        """Identical feature columns → r = 1.0."""
        rng = np.random.RandomState(42)
        x = rng.randn(100, 3)
        corr = compute_cross_correlation(x, x)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)

    def test_negated_features(self):
        """Negated features → |r| = 1.0 (anticorrelation captured)."""
        rng = np.random.RandomState(42)
        x = rng.randn(100, 3)
        corr = compute_cross_correlation(x, -x)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)

    def test_uncorrelated(self):
        """Independent features → |r| ≈ 0."""
        rng = np.random.RandomState(42)
        a = rng.randn(10000, 2)
        b = rng.randn(10000, 2)
        corr = compute_cross_correlation(a, b)
        assert np.all(corr < 0.05)

    def test_constant_column_safe(self):
        """Constant columns should produce r=0, not NaN."""
        a = np.ones((50, 2))  # all constant
        b = np.random.RandomState(42).randn(50, 2)
        corr = compute_cross_correlation(a, b)
        assert not np.any(np.isnan(corr))
        np.testing.assert_allclose(corr, 0.0, atol=1e-10)

    def test_correct_shape(self):
        a = np.random.RandomState(0).randn(100, 5)
        b = np.random.RandomState(1).randn(100, 8)
        corr = compute_cross_correlation(a, b)
        assert corr.shape == (5, 8)

    def test_sample_count_mismatch_raises(self):
        a = np.random.RandomState(0).randn(100, 3)
        b = np.random.RandomState(1).randn(50, 3)
        with pytest.raises(AssertionError):
            compute_cross_correlation(a, b)


# ── TestMatchMutualNearestNeighbors ────────────────────────────────────────


class TestMatchMutualNearestNeighbors:
    def test_perfect_correspondence(self):
        """Diagonal dominance → all features matched to themselves."""
        # Strong diagonal, weak off-diagonal
        n = 5
        corr = np.eye(n) * 0.9 + np.ones((n, n)) * 0.1
        indices = np.arange(n)
        matches = match_mutual_nearest_neighbors(corr, indices, indices)
        assert len(matches) == n
        for idx_a, idx_b, r in matches:
            assert idx_a == idx_b
            assert r == pytest.approx(1.0, abs=0.01)

    def test_no_mutual_agreement(self):
        """When best matches don't agree, returns empty."""
        # Row 0 best → col 0, row 1 best → col 0 (both want col 0)
        # Col 0 best → row 0, col 1 best → row 0 (both want row 0)
        corr = np.array([[0.9, 0.1], [0.8, 0.2]])
        indices = np.arange(2)
        matches = match_mutual_nearest_neighbors(corr, indices, indices)
        # Only (0, 0) is mutual — row 1's best is col 0, but col 0's best is row 0
        assert len(matches) == 1
        assert matches[0][0] == 0 and matches[0][1] == 0

    def test_partial_matches(self):
        """Mix of mutual and non-mutual pairs."""
        corr = np.array([
            [0.9, 0.1, 0.1],
            [0.1, 0.1, 0.8],
            [0.1, 0.7, 0.1],
        ])
        indices = np.arange(3)
        matches = match_mutual_nearest_neighbors(corr, indices, indices)
        matched_pairs = {(m[0], m[1]) for m in matches}
        assert (0, 0) in matched_pairs
        assert (1, 2) in matched_pairs
        assert (2, 1) in matched_pairs

    def test_rectangular_matrix(self):
        """Works with non-square correlation matrix."""
        corr = np.array([[0.9, 0.1], [0.1, 0.8], [0.2, 0.3]])
        idx_a = np.array([10, 20, 30])
        idx_b = np.array([100, 200])
        matches = match_mutual_nearest_neighbors(corr, idx_a, idx_b)
        assert len(matches) == 2
        matched_pairs = {(m[0], m[1]) for m in matches}
        assert (10, 100) in matched_pairs
        assert (20, 200) in matched_pairs

    def test_empty_matrix(self):
        corr = np.array([]).reshape(0, 0)
        matches = match_mutual_nearest_neighbors(corr, np.array([]), np.array([]))
        assert matches == []


# ── TestMatchHungarian ─────────────────────────────────────────────────────


class TestMatchHungarian:
    def test_square_optimal(self):
        """Diagonal is optimal assignment."""
        corr = np.eye(4) * 0.9 + np.ones((4, 4)) * 0.05
        indices = np.arange(4)
        matches = match_hungarian(corr, indices, indices)
        assert len(matches) == 4
        matched_pairs = {(m[0], m[1]) for m in matches}
        for i in range(4):
            assert (i, i) in matched_pairs

    def test_rectangular_assigns_min(self):
        """Rectangular: assigns min(m, n) pairs."""
        corr = np.random.RandomState(42).rand(3, 5)
        idx_a = np.arange(3)
        idx_b = np.arange(5)
        matches = match_hungarian(corr, idx_a, idx_b)
        assert len(matches) == 3  # min(3, 5)

    def test_one_to_one(self):
        """No duplicates in either idx_a or idx_b."""
        rng = np.random.RandomState(42)
        corr = rng.rand(10, 8)
        idx_a = np.arange(10)
        idx_b = np.arange(8)
        matches = match_hungarian(corr, idx_a, idx_b)
        a_vals = [m[0] for m in matches]
        b_vals = [m[1] for m in matches]
        assert len(set(a_vals)) == len(a_vals)
        assert len(set(b_vals)) == len(b_vals)

    def test_empty_matrix(self):
        corr = np.array([]).reshape(0, 0)
        matches = match_hungarian(corr, np.array([]), np.array([]))
        assert matches == []


# ── TestMatchManyToOne ─────────────────────────────────────────────────────


class TestMatchManyToOne:
    def test_allows_duplicate_targets(self):
        """Multiple small-model features can map to the same large-model feature."""
        # A is smaller (3), B is larger (5)
        # All rows have best match at column 0
        corr = np.array([
            [0.9, 0.1, 0.1, 0.1, 0.1],
            [0.8, 0.2, 0.1, 0.1, 0.1],
            [0.7, 0.3, 0.1, 0.1, 0.1],
        ])
        idx_a = np.arange(3)
        idx_b = np.arange(5)
        matches = match_many_to_one(corr, idx_a, idx_b)
        assert len(matches) == 3
        target_indices = [m[1] for m in matches]
        assert all(t == 0 for t in target_indices)  # all map to feature 0

    def test_all_source_features_matched(self):
        """Every feature in the smaller model gets matched."""
        rng = np.random.RandomState(42)
        corr = rng.rand(3, 10)
        idx_a = np.arange(3)
        idx_b = np.arange(10)
        matches = match_many_to_one(corr, idx_a, idx_b)
        assert len(matches) == 3  # smaller model has 3 features
        source_indices = sorted(m[0] for m in matches)
        assert source_indices == [0, 1, 2]

    def test_b_smaller(self):
        """When B is smaller, B features are matched to A."""
        rng = np.random.RandomState(42)
        corr = rng.rand(10, 3)
        idx_a = np.arange(10)
        idx_b = np.arange(3)
        matches = match_many_to_one(corr, idx_a, idx_b)
        assert len(matches) == 3  # B is smaller
        source_indices = sorted(m[0] for m in matches)
        assert source_indices == [0, 1, 2]

    def test_empty_matrix(self):
        corr = np.array([]).reshape(0, 0)
        matches = match_many_to_one(corr, np.array([]), np.array([]))
        assert matches == []


# ── TestMatchModelPair (integration) ───────────────────────────────────────


class TestMatchModelPair:
    """Integration test with synthetic SAEs having known shared features."""

    def _make_mock_sae(self, input_dim: int, hidden_dim: int, W_enc: np.ndarray):
        """Create a mock SAE that applies a fixed linear transform + ReLU."""
        mock = MagicMock()
        mock.eval = MagicMock()
        W = torch.tensor(W_enc, dtype=torch.float32)

        def encode(x):
            return torch.relu(x @ W.T)

        mock.encode = encode
        return mock

    @patch("scripts.match_sae_features.load_embeddings")
    def test_shared_features_recovered(self, mock_load):
        """Two SAEs with correlated encoding directions should have matched features."""
        rng = np.random.RandomState(42)
        input_dim = 10
        n_samples = 200

        # Shared encoding directions — features 0 and 1 in both SAEs use same dir
        shared_dir_0 = rng.randn(input_dim)
        shared_dir_0 /= np.linalg.norm(shared_dir_0)
        shared_dir_1 = rng.randn(input_dim)
        shared_dir_1 /= np.linalg.norm(shared_dir_1)

        # SAE A: 4 features — first 2 shared, last 2 random
        W_a = np.zeros((4, input_dim))
        W_a[0] = shared_dir_0
        W_a[1] = shared_dir_1
        W_a[2] = rng.randn(input_dim)
        W_a[3] = rng.randn(input_dim)

        # SAE B: 5 features — first 2 shared (permuted: 1→0, 0→1), rest random
        W_b = np.zeros((5, input_dim))
        W_b[0] = shared_dir_1  # B's feature 0 = A's feature 1
        W_b[1] = shared_dir_0  # B's feature 1 = A's feature 0
        W_b[2] = rng.randn(input_dim)
        W_b[3] = rng.randn(input_dim)
        W_b[4] = rng.randn(input_dim)

        model_a = self._make_mock_sae(input_dim, 4, W_a)
        model_b = self._make_mock_sae(input_dim, 5, W_b)

        # Generate embeddings with enough variance in shared directions
        embeddings = rng.randn(n_samples, input_dim).astype(np.float32)
        mock_load.return_value = embeddings

        datasets = ["ds1", "ds2", "ds3"]
        result = match_model_pair(
            model_a, model_b,
            Path("/fake/a"), Path("/fake/b"),
            datasets,
            method="mnn",
        )

        # Verify correct pairs recovered
        matched_pairs = {(m["idx_a"], m["idx_b"]) for m in result["matches"]}
        assert (0, 1) in matched_pairs, f"Expected (0,1) in {matched_pairs}"
        assert (1, 0) in matched_pairs, f"Expected (1,0) in {matched_pairs}"

        # Verify accounting: matched + unmatched = alive
        n_matched_a = len({m["idx_a"] for m in result["matches"]})
        n_unmatched_a = len(result["unmatched_a"])
        assert n_matched_a + n_unmatched_a == result["n_alive_a"]

    @patch("scripts.match_sae_features.load_embeddings")
    def test_hungarian_assigns_all(self, mock_load):
        """Hungarian method assigns min(alive_a, alive_b) pairs."""
        rng = np.random.RandomState(42)
        input_dim = 10

        W_a = rng.randn(4, input_dim)
        W_b = rng.randn(6, input_dim)

        model_a = self._make_mock_sae(input_dim, 4, W_a)
        model_b = self._make_mock_sae(input_dim, 6, W_b)

        embeddings = rng.randn(200, input_dim).astype(np.float32)
        mock_load.return_value = embeddings

        result = match_model_pair(
            model_a, model_b,
            Path("/fake/a"), Path("/fake/b"),
            ["ds1"],
            method="hungarian",
        )

        # Hungarian assigns min(alive_a, alive_b) pairs
        expected = min(result["n_alive_a"], result["n_alive_b"])
        assert result["n_matched"] == expected

        # 1-to-1: no duplicates
        a_vals = [m["idx_a"] for m in result["matches"]]
        b_vals = [m["idx_b"] for m in result["matches"]]
        assert len(set(a_vals)) == len(a_vals)
        assert len(set(b_vals)) == len(b_vals)


# ── TestFilterMatchesByNoiseFloor ──────────────────────────────────────────


class TestFilterMatchesByNoiseFloor:
    def test_filters_below_threshold(self):
        matches = [
            {"idx_a": 0, "idx_b": 0, "r": 0.5},
            {"idx_a": 1, "idx_b": 1, "r": 0.1},  # below noise floor
            {"idx_a": 2, "idx_b": 2, "r": 0.3},
        ]
        thresholds = {("A", "B"): 0.2, ("B", "A"): 0.15}
        filtered, n_removed = filter_matches_by_noise_floor(
            matches, "A", "B", thresholds
        )
        assert n_removed == 1
        assert len(filtered) == 2
        assert all(m["r"] >= 0.2 for m in filtered)

    def test_uses_max_of_both_directions(self):
        """Effective threshold is max(A->B, B->A) since both must pass."""
        matches = [
            {"idx_a": 0, "idx_b": 0, "r": 0.25},  # above A->B but below B->A
        ]
        thresholds = {("A", "B"): 0.2, ("B", "A"): 0.3}
        filtered, n_removed = filter_matches_by_noise_floor(
            matches, "A", "B", thresholds
        )
        assert n_removed == 1
        assert len(filtered) == 0

    def test_missing_baseline_keeps_all(self):
        matches = [{"idx_a": 0, "idx_b": 0, "r": 0.05}]
        filtered, n_removed = filter_matches_by_noise_floor(
            matches, "A", "B", {}
        )
        assert n_removed == 0
        assert len(filtered) == 1

    def test_empty_matches(self):
        filtered, n_removed = filter_matches_by_noise_floor([], "A", "B", {})
        assert n_removed == 0
        assert filtered == []
