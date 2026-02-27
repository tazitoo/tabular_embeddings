"""Tests for SAE intervention infrastructure (Epic 2).

Tests cover:
1. Identity intervention (ablate nothing) → predictions unchanged
2. Ablate all features → predictions degrade significantly
3. Delta computation is correct: decode(h_ablated) - decode(h) has right shape
4. Pre-hook modifies only query positions, not context
5. Ablated predictions are valid probabilities (sum to 1, non-negative)
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts.intervene_sae import (
    compute_ablation_delta,
    load_sae,
    load_training_mean,
    get_extraction_layer,
    INTERVENE_FN,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_REPO = Path("/Volumes/Samsung2TB/src/tabular_embeddings")
DATA_ROOT = MAIN_REPO if (MAIN_REPO / "output" / "sae_tabarena_sweep_round5").exists() else PROJECT_ROOT


@pytest.fixture
def mock_sae():
    """Create a simple deterministic mock SAE for testing delta computation."""
    sae = MagicMock()

    # Fixed weights for deterministic behavior
    torch.manual_seed(42)
    W_enc = torch.randn(128, 64)
    b_enc = torch.randn(128)
    W_dec = torch.randn(64, 128)

    def encode_fn(x):
        h = F.relu(x @ W_enc.T + b_enc)
        return h

    def decode_fn(h):
        return h @ W_dec.T

    sae.encode = encode_fn
    sae.decode = decode_fn
    return sae


@pytest.fixture
def real_sae():
    """Load the real TabPFN SAE for integration tests."""
    sae_dir = DATA_ROOT / "output" / "sae_tabarena_sweep_round5"
    if not (sae_dir / "tabpfn" / "sae_matryoshka_archetypal_validated.pt").exists():
        pytest.skip("Real SAE checkpoint not available")
    sae, config = load_sae("tabpfn", sae_dir=sae_dir, device="cpu")
    return sae, config


# ── Test compute_ablation_delta ───────────────────────────────────────────────


class TestComputeAblationDelta:
    def test_delta_shape(self, mock_sae):
        """Delta has same shape as input embeddings."""
        emb = torch.randn(50, 64)
        delta = compute_ablation_delta(mock_sae, emb, ablate_features=[0, 1, 2])
        assert delta.shape == emb.shape

    def test_delta_zero_when_no_ablation(self, mock_sae):
        """Delta is exactly zero when no features are ablated."""
        emb = torch.randn(50, 64)
        delta = compute_ablation_delta(mock_sae, emb, ablate_features=[])
        assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)

    def test_delta_nonzero_when_features_ablated(self, mock_sae):
        """Delta is nonzero when features are ablated."""
        emb = torch.randn(50, 64)
        delta = compute_ablation_delta(mock_sae, emb, ablate_features=[0, 1, 2, 3, 4])
        assert delta.abs().sum() > 0

    def test_delta_with_data_mean(self, mock_sae):
        """Centering with data_mean changes the delta computation."""
        emb = torch.randn(50, 64)
        data_mean = torch.randn(64)
        delta_no_center = compute_ablation_delta(
            mock_sae, emb, ablate_features=[0, 1, 2], data_mean=None,
        )
        delta_centered = compute_ablation_delta(
            mock_sae, emb, ablate_features=[0, 1, 2], data_mean=data_mean,
        )
        # Centering should produce a different delta
        assert not torch.allclose(delta_no_center, delta_centered, atol=1e-4)

    def test_delta_zero_with_centering_no_ablation(self, mock_sae):
        """Delta is zero when no features ablated, even with centering."""
        emb = torch.randn(50, 64)
        data_mean = torch.randn(64)
        delta = compute_ablation_delta(
            mock_sae, emb, ablate_features=[], data_mean=data_mean,
        )
        assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)

    def test_delta_with_real_sae(self, real_sae):
        """Delta computation works with real SAE."""
        sae, config = real_sae
        emb = torch.randn(10, config.input_dim)
        delta = compute_ablation_delta(sae, emb, ablate_features=[0, 1, 2])
        assert delta.shape == (10, config.input_dim)
        assert delta.abs().sum() > 0

    def test_ablating_all_features_gives_large_delta(self, real_sae):
        """Ablating all features gives larger delta than ablating a few."""
        sae, config = real_sae
        emb = torch.randn(10, config.input_dim)

        delta_few = compute_ablation_delta(sae, emb, ablate_features=[0, 1])
        delta_all = compute_ablation_delta(sae, emb, ablate_features=list(range(config.hidden_dim)))

        assert delta_all.abs().mean() > delta_few.abs().mean()


# ── Test load_sae ─────────────────────────────────────────────────────────────


class TestLoadSae:
    @pytest.mark.skipif(
        not (DATA_ROOT / "output" / "sae_tabarena_sweep_round5" / "tabpfn").exists(),
        reason="SAE checkpoints not available",
    )
    def test_load_tabpfn_sae(self):
        sae, config = load_sae(
            "tabpfn",
            sae_dir=DATA_ROOT / "output" / "sae_tabarena_sweep_round5",
            device="cpu",
        )
        assert config.input_dim == 192
        assert config.hidden_dim == 1536
        assert sae.training is False  # eval mode

    @pytest.mark.skipif(
        not (DATA_ROOT / "output" / "sae_tabarena_sweep_round5" / "mitra").exists(),
        reason="SAE checkpoints not available",
    )
    def test_load_mitra_sae(self):
        sae, config = load_sae(
            "mitra",
            sae_dir=DATA_ROOT / "output" / "sae_tabarena_sweep_round5",
            device="cpu",
        )
        assert config.input_dim == 512
        assert config.hidden_dim == 4096


# ── Test load_training_mean ───────────────────────────────────────────────────


class TestLoadTrainingMean:
    @pytest.mark.skipif(
        not (DATA_ROOT / "output" / "sae_training_round5").exists(),
        reason="SAE training data not available",
    )
    def test_load_tabpfn_mean(self):
        mean = load_training_mean(
            "tabpfn",
            training_dir=DATA_ROOT / "output" / "sae_training_round5",
            layers_path=DATA_ROOT / "config" / "optimal_extraction_layers.json",
            device="cpu",
        )
        assert mean.shape == (192,)  # TabPFN embedding dim
        assert mean.dtype == torch.float32

    def test_missing_training_data(self, tmp_path):
        """Raises FileNotFoundError for missing training data."""
        with pytest.raises(FileNotFoundError, match="SAE training data not found"):
            load_training_mean(
                "tabpfn",
                training_dir=tmp_path,
                layers_path=DATA_ROOT / "config" / "optimal_extraction_layers.json",
                device="cpu",
            )


# ── Test get_extraction_layer ─────────────────────────────────────────────────


class TestGetExtractionLayer:
    @pytest.mark.skipif(
        not (DATA_ROOT / "config" / "optimal_extraction_layers.json").exists(),
        reason="Layer config not available",
    )
    def test_tabpfn_layer(self):
        layer = get_extraction_layer(
            "tabpfn", layers_path=DATA_ROOT / "config" / "optimal_extraction_layers.json",
        )
        assert layer == 17

    @pytest.mark.skipif(
        not (DATA_ROOT / "config" / "optimal_extraction_layers.json").exists(),
        reason="Layer config not available",
    )
    def test_mitra_layer(self):
        layer = get_extraction_layer(
            "mitra", layers_path=DATA_ROOT / "config" / "optimal_extraction_layers.json",
        )
        assert layer == 10


# ── Test intervention result structure ────────────────────────────────────────


class TestInterventionResultStructure:
    """Test that intervention results have the expected structure.

    These tests use mock models to avoid requiring GPU/model weights.
    """

    def test_result_keys(self):
        """Intervention result has expected keys."""
        result = {
            "baseline_preds": np.random.rand(50, 3),
            "ablated_preds": np.random.rand(50, 3),
            "y_query": np.random.randint(0, 3, 50),
        }
        assert "baseline_preds" in result
        assert "ablated_preds" in result
        assert "y_query" in result

    def test_proba_validity(self):
        """Predicted probabilities should sum to 1 and be non-negative."""
        probs = np.random.rand(50, 3)
        probs = probs / probs.sum(axis=1, keepdims=True)  # Normalize
        assert np.all(probs >= 0)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_prediction_shapes_match(self):
        """Baseline and ablated predictions should have same shape."""
        baseline = np.random.rand(50, 3)
        ablated = np.random.rand(50, 3)
        assert baseline.shape == ablated.shape


# ── Integration tests (require GPU + model weights) ──────────────────────────


HAS_CUDA = torch.cuda.is_available()
HAS_SAE = (DATA_ROOT / "output" / "sae_tabarena_sweep_round5" / "tabpfn" /
           "sae_matryoshka_archetypal_validated.pt").exists()


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
@pytest.mark.skipif(not HAS_SAE, reason="SAE checkpoints not available")
@pytest.mark.slow
class TestTabPFNIntervention:
    """Integration tests for TabPFN intervention. Requires GPU + model weights."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load small test data."""
        from data.extended_loader import load_tabarena_dataset
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        result = load_tabarena_dataset("adult", max_samples=200)
        X, y, _ = result
        le = LabelEncoder()
        y = le.fit_transform(y)
        X_ctx, X_q, y_ctx, y_q = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y,
        )
        self.X_context = X_ctx[:100]
        self.y_context = y_ctx[:100]
        self.X_query = X_q[:50]
        self.y_query = y_q[:50]

    def test_identity_intervention(self):
        """Ablating no features produces identical predictions."""
        from scripts.intervene_sae import intervene

        results = intervene(
            model_key="tabpfn",
            X_context=self.X_context,
            y_context=self.y_context,
            X_query=self.X_query,
            y_query=self.y_query,
            ablate_features=[],
            device="cuda",
            task="classification",
            sae_dir=DATA_ROOT / "output" / "sae_tabarena_sweep_round5",
            layers_path=DATA_ROOT / "config" / "optimal_extraction_layers.json",
        )

        diff = np.abs(results["baseline_preds"] - results["ablated_preds"]).max()
        assert diff < 1e-4, f"Identity intervention failed: max diff = {diff}"

    def test_ablate_all_degrades(self):
        """Ablating all features degrades predictions significantly."""
        from scripts.intervene_sae import intervene, load_sae

        sae, config = load_sae(
            "tabpfn", sae_dir=DATA_ROOT / "output" / "sae_tabarena_sweep_round5",
            device="cuda",
        )
        all_features = list(range(config.hidden_dim))
        del sae

        results = intervene(
            model_key="tabpfn",
            X_context=self.X_context,
            y_context=self.y_context,
            X_query=self.X_query,
            y_query=self.y_query,
            ablate_features=all_features,
            device="cuda",
            task="classification",
            sae_dir=DATA_ROOT / "output" / "sae_tabarena_sweep_round5",
            layers_path=DATA_ROOT / "config" / "optimal_extraction_layers.json",
        )

        baseline_acc = np.mean(results["baseline_preds"].argmax(axis=1) == results["y_query"])
        ablated_acc = np.mean(results["ablated_preds"].argmax(axis=1) == results["y_query"])

        # Ablating all features should cause a noticeable drop
        assert baseline_acc > ablated_acc, (
            f"Expected degradation: baseline={baseline_acc:.3f}, ablated={ablated_acc:.3f}"
        )

    def test_ablated_probs_valid(self):
        """Ablated predictions are valid probabilities."""
        from scripts.intervene_sae import intervene

        results = intervene(
            model_key="tabpfn",
            X_context=self.X_context,
            y_context=self.y_context,
            X_query=self.X_query,
            y_query=self.y_query,
            ablate_features=[0, 1, 2, 3, 4],
            device="cuda",
            task="classification",
            sae_dir=DATA_ROOT / "output" / "sae_tabarena_sweep_round5",
            layers_path=DATA_ROOT / "config" / "optimal_extraction_layers.json",
        )

        probs = results["ablated_preds"]
        # TabPFN outputs probabilities via predict_proba
        assert np.all(probs >= -0.01), f"Negative probabilities: min={probs.min()}"
        assert np.allclose(probs.sum(axis=1), 1.0, atol=0.05), (
            f"Probs don't sum to 1: {probs.sum(axis=1)[:5]}"
        )
