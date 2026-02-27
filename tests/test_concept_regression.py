"""Tests for regression analysis of SAE concepts."""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_concept_regression import (
    compute_band_regression_summary,
    identify_interpolated_concepts,
    regress_features_on_probes,
)
from scripts.compare_sae_architectures import META_NAMES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SAMPLES = 500
N_PROBES = len(META_NAMES)  # 52


@dataclass
class MockConfig:
    """Minimal SAEConfig mock for testing."""
    hidden_dim: int = 64
    matryoshka_dims: list = field(default_factory=lambda: [16, 32, 64])
    topk: int = 8


@pytest.fixture
def random_meta():
    """Random meta-feature matrix."""
    np.random.seed(42)
    return np.random.randn(N_SAMPLES, N_PROBES)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPerfectLinear:
    def test_r2_near_one(self, random_meta):
        """activation = 2*probe[0] + 3*probe[1] → R² ≈ 1.0."""
        activations = np.zeros((N_SAMPLES, 4))
        # Feature 0: perfect linear combination
        activations[:, 0] = 2 * random_meta[:, 0] + 3 * random_meta[:, 1] + 5
        # Feature 1: another combination
        activations[:, 1] = -1 * random_meta[:, 5] + 0.5 * random_meta[:, 10]

        results = regress_features_on_probes(activations, random_meta, [0, 1])
        assert results[0]['r2'] > 0.99, f"R² = {results[0]['r2']:.4f}, expected ≈ 1.0"
        assert results[1]['r2'] > 0.99, f"R² = {results[1]['r2']:.4f}, expected ≈ 1.0"


class TestRandomLowR2:
    def test_random_activations_low_r2(self, random_meta):
        """Random activations → R² ≈ 0."""
        np.random.seed(99)
        activations = np.random.randn(N_SAMPLES, 4) + 1.0  # Ensure non-zero

        results = regress_features_on_probes(activations, random_meta, [0, 1, 2, 3])
        for fid, r in results.items():
            assert r['r2'] < 0.3, f"Feature {fid}: R² = {r['r2']:.3f}, expected < 0.3"


class TestCoefficientsCorrect:
    def test_dominant_probes(self, random_meta):
        """Known combination → correct top probes identified."""
        activations = np.zeros((N_SAMPLES, 2))
        # Feature 0: dominated by probes 0 and 1
        activations[:, 0] = 5 * random_meta[:, 0] - 3 * random_meta[:, 1] + 1.0

        results = regress_features_on_probes(activations, random_meta, [0])
        top_names = [p[0] for p in results[0]['top_probes'][:2]]

        # The top two probes should be META_NAMES[0] and META_NAMES[1]
        assert META_NAMES[0] in top_names, f"Expected {META_NAMES[0]} in top probes, got {top_names}"
        assert META_NAMES[1] in top_names, f"Expected {META_NAMES[1]} in top probes, got {top_names}"


class TestInterpolatedDetection:
    def test_low_d_high_r2_flagged(self):
        """Features with low single-probe d but high regression R² → flagged."""
        reg_results = {
            0: {'r2': 0.8, 'coefficients': {n: 0.1 for n in META_NAMES},
                'top_probes': [(META_NAMES[0], 0.1, 1)]},
            1: {'r2': 0.05, 'coefficients': {n: 0.01 for n in META_NAMES},
                'top_probes': [(META_NAMES[0], 0.01, 1)]},
            2: {'r2': 0.6, 'coefficients': {n: 0.08 for n in META_NAMES},
                'top_probes': [(META_NAMES[0], 0.08, 1)]},
        }
        feat_effects = {
            0: {'effect_sizes': {n: 0.2 for n in META_NAMES}},  # Low d, high R² → interpolated
            1: {'effect_sizes': {n: 0.1 for n in META_NAMES}},  # Low d, low R² → not flagged
            2: {'effect_sizes': {n: 2.0 for n in META_NAMES}},  # High d, high R² → not interpolated
        }

        interpolated = identify_interpolated_concepts(
            reg_results, feat_effects, d_threshold=0.5, r2_threshold=0.3
        )

        flagged_ids = [ic['feat_idx'] for ic in interpolated]
        assert 0 in flagged_ids, "Feature 0 (low d, high R²) should be flagged"
        assert 1 not in flagged_ids, "Feature 1 (low R²) should NOT be flagged"
        assert 2 not in flagged_ids, "Feature 2 (high d) should NOT be flagged"

    def test_empty_when_all_explained(self):
        """No interpolated concepts when all have high d."""
        reg_results = {
            0: {'r2': 0.9, 'coefficients': {n: 0.5 for n in META_NAMES},
                'top_probes': [(META_NAMES[0], 0.5, 1)]},
        }
        feat_effects = {
            0: {'effect_sizes': {n: 2.0 for n in META_NAMES}},
        }
        interpolated = identify_interpolated_concepts(reg_results, feat_effects)
        assert len(interpolated) == 0


class TestBandRegressionSummary:
    def test_matryoshka_bands(self):
        """Band summary splits features by Matryoshka scale."""
        config = MockConfig()
        reg_results = {}
        # S1: features 0-15, S2: 16-31, S3: 32-63
        for i in range(64):
            reg_results[i] = {'r2': 0.5 if i < 16 else (0.3 if i < 32 else 0.1)}

        summary = compute_band_regression_summary(reg_results, config)
        assert len(summary) == 3
        # S1 should have highest R²
        band_labels = list(summary.keys())
        assert summary[band_labels[0]]['mean_r2'] > summary[band_labels[2]]['mean_r2']

    def test_non_matryoshka(self):
        """Non-Matryoshka config → single 'all' band."""
        config = MockConfig()
        config.matryoshka_dims = None
        reg_results = {0: {'r2': 0.5}, 1: {'r2': 0.3}}
        summary = compute_band_regression_summary(reg_results, config)
        assert 'all' in summary
        assert summary['all']['n_features'] == 2


class TestTopProbesOutput:
    def test_top_probes_have_5_entries(self, random_meta):
        """Each feature should report top-5 probes."""
        activations = np.zeros((N_SAMPLES, 2))
        activations[:, 0] = random_meta[:, 0] + 1.0

        results = regress_features_on_probes(activations, random_meta, [0])
        assert len(results[0]['top_probes']) == 5

    def test_top_probes_format(self, random_meta):
        """Each top probe entry should be (name, coeff, rank)."""
        activations = np.zeros((N_SAMPLES, 2))
        activations[:, 0] = random_meta[:, 0] + 1.0

        results = regress_features_on_probes(activations, random_meta, [0])
        for name, coeff, rank in results[0]['top_probes']:
            assert isinstance(name, str)
            assert isinstance(coeff, float)
            assert isinstance(rank, int)
            assert name in META_NAMES
