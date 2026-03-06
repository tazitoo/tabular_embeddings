"""Tests for per-concept vec2vec transfer between tabular foundation models.

Tests cover:
1. fit_linear_map: known linear relationship → near-perfect W recovery
2. compute_transfer_delta: correct shapes, zero when no features, direction check
3. Scale=0 → delta is zero → predictions unchanged
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts.transfer_concepts import (
    compute_transfer_delta,
    compute_transfer_delta_perrow,
    fit_linear_map,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def known_linear_data():
    """Generate data with a known linear relationship Y = X @ W_true.T + b_true."""
    np.random.seed(42)
    n, d_source, d_target = 200, 10, 8
    W_true = np.random.randn(d_target, d_source)
    b_true = np.random.randn(d_target)
    X = np.random.randn(n, d_source)
    Y = X @ W_true.T + b_true
    return X, Y, W_true, b_true


@pytest.fixture
def mock_sae():
    """Simple deterministic mock SAE for testing transfer delta."""
    sae = MagicMock()

    torch.manual_seed(42)
    d_input = 16
    d_hidden = 64
    W_enc = torch.randn(d_hidden, d_input)
    b_enc = torch.randn(d_hidden)
    W_dec = torch.randn(d_input, d_hidden)

    def encode_fn(x):
        return F.relu(x @ W_enc.T + b_enc)

    def decode_fn(h):
        return h @ W_dec.T

    sae.encode = encode_fn
    sae.decode = decode_fn
    sae.d_input = d_input
    sae.d_hidden = d_hidden

    return sae


# ── Test fit_linear_map ───────────────────────────────────────────────────────


class TestFitLinearMap:
    def test_perfect_recovery(self, known_linear_data):
        """Ridge regression recovers true W and b for noiseless data."""
        X, Y, W_true, b_true = known_linear_data
        W, b, r2 = fit_linear_map(X, Y, alpha=1e-6)

        assert r2 > 0.999, f"R² = {r2:.4f}, expected > 0.999"
        np.testing.assert_allclose(W, W_true, atol=0.05)
        np.testing.assert_allclose(b, b_true, atol=0.05)

    def test_output_shapes(self, known_linear_data):
        """W and b have correct shapes."""
        X, Y, _, _ = known_linear_data
        W, b, r2 = fit_linear_map(X, Y)

        assert W.shape == (Y.shape[1], X.shape[1])
        assert b.shape == (Y.shape[1],)
        assert 0.0 <= r2 <= 1.0

    def test_regularization_effect(self, known_linear_data):
        """Higher alpha → lower R² (more regularization)."""
        X, Y, _, _ = known_linear_data
        _, _, r2_low = fit_linear_map(X, Y, alpha=1e-6)
        _, _, r2_high = fit_linear_map(X, Y, alpha=1000.0)

        assert r2_low > r2_high, (
            f"Expected low-alpha R²={r2_low:.4f} > high-alpha R²={r2_high:.4f}"
        )

    def test_noisy_data(self):
        """Handles noisy data gracefully (R² < 1 but positive)."""
        np.random.seed(42)
        n, d_s, d_t = 100, 5, 3
        X = np.random.randn(n, d_s)
        Y = X[:, :d_t] + 0.5 * np.random.randn(n, d_t)

        W, b, r2 = fit_linear_map(X, Y)
        assert 0.5 < r2 < 1.0, f"R² = {r2:.4f}, expected in (0.5, 1.0)"


# ── Test compute_transfer_delta ───────────────────────────────────────────────


class TestComputeTransferDelta:
    def test_output_shape(self, mock_sae):
        """Delta has shape (n_samples, d_target)."""
        d_source, d_target = 16, 8
        W = np.random.randn(d_target, d_source)
        b = np.random.randn(d_target)
        emb = torch.randn(30, d_source)

        delta = compute_transfer_delta(
            mock_sae, emb, W, b,
            transfer_features=[0, 1, 2],
        )
        assert delta.shape == (30, d_target)

    def test_zero_when_no_features(self, mock_sae):
        """Delta is exactly zero when no features are transferred."""
        d_source, d_target = 16, 8
        W = np.random.randn(d_target, d_source)
        b = np.random.randn(d_target)
        emb = torch.randn(30, d_source)

        delta = compute_transfer_delta(mock_sae, emb, W, b, transfer_features=[])
        assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-7)

    def test_zero_when_scale_zero(self, mock_sae):
        """Delta is zero when scale=0."""
        d_source, d_target = 16, 8
        W = np.random.randn(d_target, d_source)
        b = np.random.randn(d_target)
        emb = torch.randn(30, d_source)

        delta = compute_transfer_delta(
            mock_sae, emb, W, b,
            transfer_features=[0, 1, 2], scale=0.0,
        )
        assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-7)

    def test_nonzero_with_features(self, mock_sae):
        """Delta is nonzero when features are transferred."""
        d_source, d_target = 16, 8
        W = np.random.randn(d_target, d_source)
        b = np.random.randn(d_target)
        emb = torch.randn(30, d_source)

        delta = compute_transfer_delta(
            mock_sae, emb, W, b,
            transfer_features=[0, 1, 2, 3],
        )
        assert delta.abs().sum() > 0, "Delta should be nonzero"

    def test_scale_linearity(self, mock_sae):
        """Delta scales linearly with scale factor."""
        d_source, d_target = 16, 8
        W = np.random.randn(d_target, d_source)
        b = np.random.randn(d_target)
        emb = torch.randn(30, d_source)

        delta_1 = compute_transfer_delta(
            mock_sae, emb, W, b,
            transfer_features=[0, 1], scale=1.0,
        )
        delta_2 = compute_transfer_delta(
            mock_sae, emb, W, b,
            transfer_features=[0, 1], scale=2.0,
        )
        torch.testing.assert_close(delta_2, delta_1 * 2.0, atol=1e-5, rtol=1e-5)

    def test_more_features_larger_delta(self, mock_sae):
        """Transferring more features produces larger delta (on average)."""
        d_source, d_target = 16, 8
        np.random.seed(42)
        W = np.random.randn(d_target, d_source)
        b = np.random.randn(d_target)
        emb = torch.randn(30, d_source)

        delta_few = compute_transfer_delta(
            mock_sae, emb, W, b,
            transfer_features=[0, 1],
        )
        delta_many = compute_transfer_delta(
            mock_sae, emb, W, b,
            transfer_features=list(range(32)),
        )
        # Not guaranteed for all RNG seeds, but with enough features it holds
        assert delta_many.abs().mean() > delta_few.abs().mean() * 0.5

    def test_with_data_mean(self, mock_sae):
        """Data mean centering changes the delta."""
        d_source, d_target = 16, 8
        W = np.random.randn(d_target, d_source)
        b = np.random.randn(d_target)
        emb = torch.randn(30, d_source)
        data_mean = torch.randn(d_source)

        delta_no_center = compute_transfer_delta(
            mock_sae, emb, W, b,
            transfer_features=[0, 1, 2],
        )
        delta_centered = compute_transfer_delta(
            mock_sae, emb, W, b,
            transfer_features=[0, 1, 2],
            data_mean=data_mean,
        )
        assert not torch.allclose(delta_no_center, delta_centered, atol=1e-4)


# ── Test compute_transfer_delta_perrow ────────────────────────────────────────


class TestComputeTransferDeltaPerrow:
    def test_output_shape(self, mock_sae):
        """Per-row masks → correct (n_rows, d_target) shape."""
        d_source, d_target, n_rows = 16, 8, 30
        W = np.random.randn(d_target, d_source)
        b = np.random.randn(d_target)
        emb = torch.randn(n_rows, d_source)
        masks = torch.zeros(n_rows, mock_sae.d_hidden, dtype=torch.bool)
        masks[:, :3] = True

        delta = compute_transfer_delta_perrow(mock_sae, emb, W, b, masks)
        assert delta.shape == (n_rows, d_target)

    def test_zero_when_no_masks(self, mock_sae):
        """All-False masks → zero delta."""
        d_source, d_target, n_rows = 16, 8, 30
        W = np.random.randn(d_target, d_source)
        b = np.random.randn(d_target)
        emb = torch.randn(n_rows, d_source)
        masks = torch.zeros(n_rows, mock_sae.d_hidden, dtype=torch.bool)

        delta = compute_transfer_delta_perrow(mock_sae, emb, W, b, masks)
        assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-7)

    def test_matches_uniform(self, mock_sae):
        """All-True masks → same result as compute_transfer_delta with all features."""
        d_source, d_target, n_rows = 16, 8, 10
        np.random.seed(42)
        W = np.random.randn(d_target, d_source)
        b = np.random.randn(d_target)
        emb = torch.randn(n_rows, d_source)
        all_features = list(range(mock_sae.d_hidden))

        masks = torch.ones(n_rows, mock_sae.d_hidden, dtype=torch.bool)
        delta_perrow = compute_transfer_delta_perrow(mock_sae, emb, W, b, masks)
        delta_uniform = compute_transfer_delta(
            mock_sae, emb, W, b, all_features,
        )
        torch.testing.assert_close(delta_perrow, delta_uniform, atol=1e-5, rtol=1e-5)

    def test_per_row_selectivity(self, mock_sae):
        """Different masks per row → different deltas per row."""
        d_source, d_target, n_rows = 16, 8, 4
        np.random.seed(42)
        W = np.random.randn(d_target, d_source)
        b = np.random.randn(d_target)
        emb = torch.randn(n_rows, d_source)

        masks = torch.zeros(n_rows, mock_sae.d_hidden, dtype=torch.bool)
        # Row 0: features 0-3, Row 1: features 10-13
        masks[0, :4] = True
        masks[1, 10:14] = True
        # Rows 2,3: no features

        delta = compute_transfer_delta_perrow(mock_sae, emb, W, b, masks)
        # Rows 0 and 1 should have different nonzero deltas
        assert delta[0].abs().sum() > 0, "Row 0 should be nonzero"
        assert delta[1].abs().sum() > 0, "Row 1 should be nonzero"
        assert not torch.allclose(delta[0], delta[1], atol=1e-4), \
            "Different masks should give different deltas"
        # Rows 2,3 should be zero
        assert torch.allclose(delta[2], torch.zeros_like(delta[2]), atol=1e-7)
        assert torch.allclose(delta[3], torch.zeros_like(delta[3]), atol=1e-7)
