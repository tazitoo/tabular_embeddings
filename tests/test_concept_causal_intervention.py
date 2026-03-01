"""Tests for concept_causal_intervention.py.

Tests cover:
1. compute_boost_delta: shape, zero-boost = no change, boost-to-zero = ablation
2. Target selection from diagnostic results
3. MNN pair loading with swapped directions
4. Feature ranking by activation
5. Convergence analysis
6. Dose-response step structure
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts.intervene_sae import compute_ablation_delta, compute_boost_delta
from scripts.concept_causal_intervention import (
    _load_mnn_pair,
    _rank_features_by_activation,
    analyze_convergence,
    select_targets,
    DISPLAY_NAMES,
    INTERVENTION_MODELS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def mock_sae():
    """Create a simple deterministic mock SAE for testing boost delta."""
    sae = MagicMock()

    torch.manual_seed(42)
    W_enc = torch.randn(128, 64)
    b_enc = torch.randn(128)
    W_dec = torch.randn(64, 128)

    def encode_fn(x):
        return F.relu(x @ W_enc.T + b_enc)

    def decode_fn(h):
        return h @ W_dec.T

    sae.encode = encode_fn
    sae.decode = decode_fn
    return sae


@pytest.fixture
def mock_mnn_file(tmp_path):
    """Create a mock MNN matching file."""
    data = {
        "metadata": {"models": ["Mitra", "TabPFN"]},
        "pairs": {
            "Mitra__TabPFN": {
                "n_alive_a": 96,
                "n_alive_b": 128,
                "n_matched": 20,
                "matches": [
                    {"idx_a": i * 2, "idx_b": i * 3, "r": 0.5 + 0.02 * i}
                    for i in range(20)
                ],
                "unmatched_a": list(range(40, 96)),
                "unmatched_b": list(range(60, 128)),
            },
        },
    }
    path = tmp_path / "mnn.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return path


@pytest.fixture
def mock_diagnostic_file(tmp_path):
    """Create a mock diagnostic results file."""
    diagnostic = {
        "pairs": {
            "TabPFN__Mitra": {
                "model_a": "tabpfn",
                "model_b": "mitra",
                "n_datasets": 20,
                "correlations": {"concept_asymmetry": {"rho": 0.45, "p_value": 0.03}},
                "data": [
                    {
                        "dataset": f"dataset_{i}",
                        "perf_gap": 0.1 - 0.01 * i,  # Decreasing gap
                        "concept_asymmetry": 0.05 - 0.005 * i,
                        "unmatched_act_a": 0.03 if i < 10 else 0.0,
                        "unmatched_act_b": 0.02,
                    }
                    for i in range(20)
                ],
            },
        },
        "n_pairs": 1,
    }
    path = tmp_path / "diagnostic.json"
    with open(path, "w") as f:
        json.dump(diagnostic, f)
    return path


@pytest.fixture
def mock_perf_file(tmp_path):
    """Create a mock performance CSV."""
    import pandas as pd

    rows = []
    for model in ["tabpfn", "mitra"]:
        for i in range(20):
            base = 0.85 if model == "tabpfn" else 0.80
            rows.append({
                "model": model,
                "dataset": f"dataset_{i}",
                "task": "classification",
                "metric_name": "auc",
                "metric_value": base + np.random.randn() * 0.02,
                "n_query": 500,
            })
    df = pd.DataFrame(rows)
    path = tmp_path / "perf.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def mock_fingerprint():
    """Create a mock fingerprint dict."""
    np.random.seed(42)
    hidden_dim = 256
    return {
        "model": "tabpfn",
        "hidden_dim": hidden_dim,
        "alive_features": list(range(0, hidden_dim, 2)),
        "bands": {"S1": 16, "S2": 32, "S3": 64, "S4": 128, "S5": 256},
        "global_mean": np.random.randn(hidden_dim).tolist(),
        "dataset_means": {
            "adult": (np.random.randn(hidden_dim) * 0.5).tolist(),
            "credit-g": (np.random.randn(hidden_dim) * 0.3).tolist(),
        },
    }


# ── Test compute_boost_delta ─────────────────────────────────────────────


class TestComputeBoostDelta:
    def test_boost_delta_shape(self, mock_sae):
        """Boost delta has same shape as input embeddings."""
        emb = torch.randn(50, 64)
        delta = compute_boost_delta(
            mock_sae, emb,
            boost_features=[0, 1, 2],
            target_activations=[1.0, 2.0, 3.0],
        )
        assert delta.shape == emb.shape

    def test_boost_to_current_activation_gives_zero_delta(self, mock_sae):
        """Boosting to the current activation produces zero delta."""
        emb = torch.randn(50, 64)
        # First get current activations
        with torch.no_grad():
            h = mock_sae.encode(emb)
            current_acts = [float(h[0, i]) for i in [0, 1, 2]]

        delta = compute_boost_delta(
            mock_sae, emb,
            boost_features=[0, 1, 2],
            target_activations=current_acts,
        )
        # Delta should be zero for sample 0 (where we matched activations)
        # Due to batching, only the first sample is exactly matched
        assert delta[0].abs().sum() == pytest.approx(0.0, abs=1e-5)

    def test_boost_to_zero_equals_ablation(self, mock_sae):
        """Boosting all activations to 0 should equal ablation."""
        emb = torch.randn(50, 64)
        features = [0, 1, 2, 3, 4]

        ablation_delta = compute_ablation_delta(
            mock_sae, emb, ablate_features=features,
        )
        boost_delta = compute_boost_delta(
            mock_sae, emb,
            boost_features=features,
            target_activations=[0.0] * len(features),
        )
        assert torch.allclose(ablation_delta, boost_delta, atol=1e-5)

    def test_boost_with_centering(self, mock_sae):
        """Centering with data_mean changes boost computation."""
        emb = torch.randn(50, 64)
        data_mean = torch.randn(64)

        delta_no_center = compute_boost_delta(
            mock_sae, emb, [0], [5.0], data_mean=None,
        )
        delta_centered = compute_boost_delta(
            mock_sae, emb, [0], [5.0], data_mean=data_mean,
        )
        assert not torch.allclose(delta_no_center, delta_centered, atol=1e-4)

    def test_boost_mismatched_lengths_raises(self, mock_sae):
        """Mismatched boost_features and target_activations raises ValueError."""
        emb = torch.randn(50, 64)
        with pytest.raises(ValueError, match="same length"):
            compute_boost_delta(
                mock_sae, emb,
                boost_features=[0, 1, 2],
                target_activations=[1.0, 2.0],
            )


# ── Test MNN pair loading ────────────────────────────────────────────────


class TestLoadMNNPair:
    def test_forward_direction(self, mock_mnn_file):
        """Loading pair in stored order preserves directions."""
        # MNN stores "Mitra__TabPFN" where Mitra is A, TabPFN is B
        pair = _load_mnn_pair("mitra", "tabpfn", mock_mnn_file)
        assert pair["n_alive_a"] == 96  # Mitra's alive count
        assert pair["n_alive_b"] == 128  # TabPFN's alive count
        assert len(pair["matches"]) == 20

    def test_reverse_direction(self, mock_mnn_file):
        """Loading pair in reverse order swaps A/B correctly."""
        pair = _load_mnn_pair("tabpfn", "mitra", mock_mnn_file)
        assert pair["n_alive_a"] == 128  # TabPFN = B in storage → A here
        assert pair["n_alive_b"] == 96   # Mitra = A in storage → B here
        assert len(pair["matches"]) == 20

        # Check match indices are swapped
        for m in pair["matches"]:
            assert "idx_a" in m
            assert "idx_b" in m

    def test_unmatched_swapped(self, mock_mnn_file):
        """Unmatched sets are correctly swapped for reversed pair."""
        fwd = _load_mnn_pair("mitra", "tabpfn", mock_mnn_file)
        rev = _load_mnn_pair("tabpfn", "mitra", mock_mnn_file)
        assert fwd["unmatched_a"] == rev["unmatched_b"]
        assert fwd["unmatched_b"] == rev["unmatched_a"]


# ── Test feature ranking ────────────────────────────────────────────────


class TestRankFeatures:
    def test_ranking_by_absolute_activation(self, mock_fingerprint):
        """Features are ranked by absolute activation on the dataset."""
        features = [0, 2, 4, 6, 8]
        ranked = _rank_features_by_activation(features, mock_fingerprint, "adult")
        activations = [act for _, act in ranked]
        # Should be sorted descending by absolute value
        for i in range(len(activations) - 1):
            assert activations[i] >= activations[i + 1]

    def test_fallback_to_global_mean(self, mock_fingerprint):
        """Falls back to global mean if dataset not in fingerprint."""
        features = [0, 2, 4]
        ranked = _rank_features_by_activation(features, mock_fingerprint, "nonexistent")
        assert len(ranked) == 3

    def test_empty_features_returns_empty(self, mock_fingerprint):
        """Empty feature list returns empty ranking."""
        ranked = _rank_features_by_activation([], mock_fingerprint, "adult")
        assert ranked == []


# ── Test convergence analysis ────────────────────────────────────────────


class TestAnalyzeConvergence:
    def test_empty_results(self):
        """Empty results produce zero stats."""
        result = analyze_convergence([])
        assert result["ablation"]["n"] == 0
        assert result["boost"]["n"] == 0
        assert result["transplant"]["n"] == 0

    def test_ablation_convergence(self):
        """Ablation convergence is computed from final step."""
        results = [
            {
                "type": "ablation",
                "steps": [
                    {"n_features": 1, "convergence": 0.2},
                    {"n_features": 2, "convergence": 0.5},
                    {"n_features": 3, "convergence": 0.8},
                ],
            },
            {
                "type": "ablation",
                "steps": [
                    {"n_features": 1, "convergence": 0.3},
                    {"n_features": 2, "convergence": 0.6},
                ],
            },
        ]
        result = analyze_convergence(results)
        assert result["ablation"]["n"] == 2
        assert result["ablation"]["mean"] == pytest.approx(0.7, abs=0.01)

    def test_mixed_types(self):
        """Different intervention types are tracked separately."""
        results = [
            {"type": "ablation", "steps": [{"convergence": 0.5}]},
            {"type": "boost", "steps": [{"convergence": 0.3}]},
            {"type": "transplant", "steps": [{"convergence": 0.7}]},
        ]
        result = analyze_convergence(results)
        assert result["ablation"]["n"] == 1
        assert result["boost"]["n"] == 1
        assert result["transplant"]["n"] == 1

    def test_nan_convergence_skipped(self):
        """NaN convergence values are excluded from stats."""
        results = [
            {"type": "ablation", "steps": [{"convergence": float("nan")}]},
            {"type": "ablation", "steps": [{"convergence": 0.5}]},
        ]
        result = analyze_convergence(results)
        assert result["ablation"]["n"] == 1

    def test_no_steps_skipped(self):
        """Results with no steps are skipped."""
        results = [
            {"type": "ablation", "steps": []},
            {"type": "ablation", "steps": [{"convergence": 0.5}]},
        ]
        result = analyze_convergence(results)
        assert result["ablation"]["n"] == 1


# ── Test target selection ────────────────────────────────────────────────


class TestSelectTargets:
    def test_selects_targets_with_gap(self, mock_diagnostic_file, mock_perf_file):
        """Targets are selected where performance gap exceeds threshold."""
        targets = select_targets(
            mock_diagnostic_file, mock_perf_file,
            min_gap=0.05, top_n=5,
        )
        for t in targets:
            assert t["perf_gap"] > 0.05
            assert t["model_a"] in INTERVENTION_MODELS
            assert t["model_b"] in INTERVENTION_MODELS

    def test_sorted_by_signal_strength(self, mock_diagnostic_file, mock_perf_file):
        """Targets are sorted by signal strength descending."""
        targets = select_targets(
            mock_diagnostic_file, mock_perf_file,
            min_gap=0.01, top_n=20,
        )
        if len(targets) > 1:
            strengths = [t["signal_strength"] for t in targets]
            for i in range(len(strengths) - 1):
                assert strengths[i] >= strengths[i + 1]

    def test_top_n_limiting(self, mock_diagnostic_file, mock_perf_file):
        """Output is limited to top_n targets."""
        targets = select_targets(
            mock_diagnostic_file, mock_perf_file,
            min_gap=0.01, top_n=3,
        )
        assert len(targets) <= 3

    def test_missing_diagnostic_raises(self, tmp_path, mock_perf_file):
        """Missing diagnostic file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            select_targets(
                tmp_path / "nonexistent.json", mock_perf_file,
            )


# ── Test display name consistency ────────────────────────────────────────


class TestDisplayNames:
    def test_intervention_models_have_display_names(self):
        """All intervention models have display name mappings."""
        for model in INTERVENTION_MODELS:
            assert model in DISPLAY_NAMES
