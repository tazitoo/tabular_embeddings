"""Tests for scripts/section43/universal_concepts.py analysis functions."""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.section43.universal_concepts import (
    DOMAIN_MERGES,
    EXCLUDED_DOMAINS,
    build_domain_row_indices,
    compute_domain_reconstruction_r2,
    compute_domain_taxonomy_agreement,
    compute_feature_selectivity,
    load_domain_taxonomy,
    pool_embeddings_with_offsets,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class MockConfig:
    """Minimal SAEConfig mock for testing."""
    input_dim: int = 8
    hidden_dim: int = 32
    matryoshka_dims: list = field(default_factory=lambda: [4, 8, 16])
    topk: int = 8


@pytest.fixture
def sample_offsets():
    """Dataset offsets for 3 datasets."""
    return {
        "ds_finance1": (0, 100),
        "ds_finance2": (100, 200),
        "ds_health1": (200, 300),
        "ds_science1": (300, 350),
        "ds_excluded": (350, 400),
    }


@pytest.fixture
def sample_domain_map():
    """Dataset → domain mapping."""
    return {
        "ds_finance1": "Finance",
        "ds_finance2": "Finance",
        "ds_health1": "Healthcare",
        "ds_science1": "Science",
        # ds_excluded intentionally absent
    }


@pytest.fixture
def sample_activations():
    """Synthetic activations: (400, 32) with known structure."""
    np.random.seed(42)
    h = np.zeros((400, 32), dtype=np.float32)

    # Features 0-3 (band S1): fire everywhere → universal
    h[:, 0:4] = np.random.rand(400, 4) * 0.5 + 0.1

    # Features 4-7 (band S2): fire only on finance (rows 0-200)
    h[:200, 4:8] = np.random.rand(200, 4) * 0.5 + 0.1

    # Features 8-15 (band S3): fire on finance + health (0-300)
    h[:300, 8:16] = np.random.rand(300, 8) * 0.3 + 0.05

    # Features 16-31 (band S4): mixed, some fire on specific domains
    h[200:300, 16:20] = np.random.rand(100, 4) * 0.4 + 0.1  # health only
    h[:100, 20:24] = np.random.rand(100, 4) * 0.4 + 0.1      # finance1 only
    # Features 24-31: dead (all zeros)

    return h


# ---------------------------------------------------------------------------
# Tests for build_domain_row_indices
# ---------------------------------------------------------------------------

class TestBuildDomainRowIndices:
    def test_basic_mapping(self, sample_offsets, sample_domain_map):
        result = build_domain_row_indices(sample_offsets, sample_domain_map)
        assert set(result.keys()) == {"Finance", "Healthcare", "Science"}

    def test_finance_has_two_datasets(self, sample_offsets, sample_domain_map):
        result = build_domain_row_indices(sample_offsets, sample_domain_map)
        assert len(result["Finance"]) == 200  # 100 + 100

    def test_excluded_dataset_ignored(self, sample_offsets, sample_domain_map):
        result = build_domain_row_indices(sample_offsets, sample_domain_map)
        # ds_excluded is not in domain_map, should not appear anywhere
        all_indices = np.concatenate(list(result.values()))
        assert 375 not in all_indices  # index in ds_excluded range

    def test_contiguous_indices(self, sample_offsets, sample_domain_map):
        result = build_domain_row_indices(sample_offsets, sample_domain_map)
        # Finance should cover [0,200)
        finance_idx = sorted(result["Finance"])
        assert finance_idx[0] == 0
        assert finance_idx[-1] == 199


# ---------------------------------------------------------------------------
# Tests for compute_feature_selectivity
# ---------------------------------------------------------------------------

class TestFeatureSelectivity:
    def test_band_count(self, sample_activations, sample_offsets, sample_domain_map):
        config = MockConfig()
        domain_idx = build_domain_row_indices(sample_offsets, sample_domain_map)
        result = compute_feature_selectivity(
            sample_activations, domain_idx, config
        )
        # Should have 4 bands: [0,4), [4,8), [8,16), [16,32)
        assert len(result) == 4

    def test_universal_features_in_s1(
        self, sample_activations, sample_offsets, sample_domain_map
    ):
        config = MockConfig()
        domain_idx = build_domain_row_indices(sample_offsets, sample_domain_map)
        result = compute_feature_selectivity(
            sample_activations, domain_idx, config
        )
        # S1 [0,4): all 4 features fire everywhere → all universal
        s1 = result["S1 [0,4)"]
        assert s1["universal"] == 4
        assert s1["domain_specific"] == 0

    def test_domain_specific_features(
        self, sample_activations, sample_offsets, sample_domain_map
    ):
        config = MockConfig()
        domain_idx = build_domain_row_indices(sample_offsets, sample_domain_map)
        result = compute_feature_selectivity(
            sample_activations, domain_idx, config
        )
        # S2 [4,8): fire only on finance → domain_specific (active in 1 domain)
        s2 = result["S2 [4,8)"]
        assert s2["domain_specific"] == 4
        assert s2["universal"] == 0

    def test_dead_features_counted(
        self, sample_activations, sample_offsets, sample_domain_map
    ):
        config = MockConfig()
        domain_idx = build_domain_row_indices(sample_offsets, sample_domain_map)
        result = compute_feature_selectivity(
            sample_activations, domain_idx, config
        )
        # S4 [16,32) has features 24-31 that are dead
        s4 = result["S4 [16,32)"]
        assert s4["dead"] == 8  # features 24-31

    def test_selectivity_categories_sum_to_alive(
        self, sample_activations, sample_offsets, sample_domain_map
    ):
        config = MockConfig()
        domain_idx = build_domain_row_indices(sample_offsets, sample_domain_map)
        result = compute_feature_selectivity(
            sample_activations, domain_idx, config
        )
        for band_label, counts in result.items():
            alive = counts["universal"] + counts["domain_cluster"] + counts["domain_specific"]
            # Each band's alive + dead should equal the band size
            band_parts = band_label.split("[")[1].rstrip(")")
            start, end = map(int, band_parts.split(","))
            assert alive + counts["dead"] == end - start


# ---------------------------------------------------------------------------
# Tests for compute_domain_reconstruction_r2
# ---------------------------------------------------------------------------

class TestDomainReconstructionR2:
    def test_perfect_reconstruction(self):
        """When decode gives back the input, R² should be ~1.0."""
        n = 100
        dim = 8
        x_true = np.random.randn(n, dim).astype(np.float32)

        mock_model = MagicMock()
        mock_model.decode.return_value = torch.tensor(x_true)

        domain_idx = {"A": np.arange(n)}
        result = compute_domain_reconstruction_r2(
            mock_model, np.zeros((n, 16)), x_true, domain_idx, [16]
        )
        assert abs(result["A"][16] - 1.0) < 1e-5

    def test_zero_reconstruction(self):
        """When decode returns zeros, R² should be negative."""
        n = 100
        dim = 8
        np.random.seed(0)
        x_true = np.random.randn(n, dim).astype(np.float32)

        mock_model = MagicMock()
        mock_model.decode.return_value = torch.zeros(n, dim)

        domain_idx = {"A": np.arange(n)}
        result = compute_domain_reconstruction_r2(
            mock_model, np.zeros((n, 16)), x_true, domain_idx, [16]
        )
        assert result["A"][16] < 0

    def test_multiple_scales(self):
        """R² should improve with more features (higher scale)."""
        n = 200
        dim = 8
        np.random.seed(42)
        x_true = np.random.randn(n, dim).astype(np.float32)

        # Simulate: small scale = noisy, full scale = exact
        def mock_decode(h, max_dim=None):
            if max_dim == 4:
                return torch.tensor(x_true * 0.5 + 0.5 * np.random.randn(n, dim).astype(np.float32))
            return torch.tensor(x_true)

        mock_model = MagicMock()
        mock_model.decode.side_effect = mock_decode

        domain_idx = {"A": np.arange(n)}
        result = compute_domain_reconstruction_r2(
            mock_model, np.zeros((n, 16)), x_true, domain_idx, [4, 16]
        )
        assert result["A"][16] > result["A"][4]


# ---------------------------------------------------------------------------
# Tests for compute_domain_taxonomy_agreement
# ---------------------------------------------------------------------------

class TestDomainTaxonomyAgreement:
    def test_perfect_clustering(self):
        """Well-separated domain activations should give high ARI/NMI."""
        np.random.seed(42)
        activations = np.zeros((600, 16), dtype=np.float32)
        # Domain A: features 0-7 active
        activations[:200, :8] = np.random.rand(200, 8) + 5.0
        # Domain B: features 8-15 active
        activations[200:400, 8:16] = np.random.rand(200, 8) + 5.0
        # Domain C: uniform low
        activations[400:600, :] = np.random.rand(200, 16) * 0.1

        offsets = {
            "ds1": (0, 100), "ds2": (100, 200),   # A
            "ds3": (200, 300), "ds4": (300, 400),  # B
            "ds5": (400, 500), "ds6": (500, 600),  # C
        }
        domain_map = {
            "ds1": "A", "ds2": "A",
            "ds3": "B", "ds4": "B",
            "ds5": "C", "ds6": "C",
        }

        result = compute_domain_taxonomy_agreement(activations, offsets, domain_map)
        assert result["ari"] > 0.5
        assert result["nmi"] > 0.5

    def test_random_activations_low_agreement(self):
        """Random activations should give ARI near 0."""
        np.random.seed(42)
        activations = np.random.randn(600, 16).astype(np.float32)

        offsets = {f"ds{i}": (i*100, (i+1)*100) for i in range(6)}
        domain_map = {
            "ds0": "A", "ds1": "A",
            "ds2": "B", "ds3": "B",
            "ds4": "C", "ds5": "C",
        }

        result = compute_domain_taxonomy_agreement(activations, offsets, domain_map)
        assert result["n_datasets"] == 6
        # ARI should be near 0 for random data
        assert abs(result["ari"]) < 0.5

    def test_too_few_datasets(self):
        """With < 4 datasets, should return zeros."""
        activations = np.random.randn(200, 8).astype(np.float32)
        offsets = {"ds1": (0, 100), "ds2": (100, 200)}
        domain_map = {"ds1": "A", "ds2": "B"}

        result = compute_domain_taxonomy_agreement(activations, offsets, domain_map)
        assert result["ari"] == 0.0
        assert result["nmi"] == 0.0


# ---------------------------------------------------------------------------
# Tests for load_domain_taxonomy
# ---------------------------------------------------------------------------

class TestLoadDomainTaxonomy:
    def test_merges_applied(self):
        """Natural Sciences and Chemistry & Materials should merge."""
        dataset_domain = load_domain_taxonomy()
        # SDSS17 is Natural Sciences → should become Science & Materials
        assert dataset_domain.get("SDSS17") == "Science & Materials"
        # qsar-biodeg is Chemistry & Materials → should also become Science & Materials
        assert dataset_domain.get("qsar-biodeg") == "Science & Materials"

    def test_excluded_domains_absent(self):
        """Excluded domains should not appear in the output."""
        dataset_domain = load_domain_taxonomy()
        domains_present = set(dataset_domain.values())
        for excluded in EXCLUDED_DOMAINS:
            assert excluded not in domains_present

    def test_main_domains_present(self):
        """Core domains should be present."""
        dataset_domain = load_domain_taxonomy()
        domains = set(dataset_domain.values())
        assert "Business & Marketing" in domains
        assert "Finance & Insurance" in domains
        assert "Science & Materials" in domains


# ---------------------------------------------------------------------------
# Tests for pool_embeddings_with_offsets
# ---------------------------------------------------------------------------

class TestPoolEmbeddingsWithOffsets:
    def test_basic_pooling(self, tmp_path):
        """Pool two datasets and verify offsets."""
        np.savez(
            tmp_path / "tabarena_ds1.npz",
            embeddings=np.random.randn(50, 8).astype(np.float32),
        )
        np.savez(
            tmp_path / "tabarena_ds2.npz",
            embeddings=np.random.randn(30, 8).astype(np.float32),
        )

        pooled, offsets = pool_embeddings_with_offsets(
            tmp_path, ["ds1", "ds2"], max_per_dataset=500
        )
        assert pooled.shape == (80, 8)
        assert offsets["ds1"] == (0, 50)
        assert offsets["ds2"] == (50, 80)

    def test_subsampling(self, tmp_path):
        """Large datasets should be subsampled to max_per_dataset."""
        np.savez(
            tmp_path / "tabarena_big.npz",
            embeddings=np.random.randn(1000, 4).astype(np.float32),
        )

        pooled, offsets = pool_embeddings_with_offsets(
            tmp_path, ["big"], max_per_dataset=100
        )
        assert pooled.shape == (100, 4)
        assert offsets["big"] == (0, 100)

    def test_missing_dataset_skipped(self, tmp_path):
        """Missing dataset files should be silently skipped."""
        np.savez(
            tmp_path / "tabarena_exists.npz",
            embeddings=np.random.randn(20, 4).astype(np.float32),
        )

        pooled, offsets = pool_embeddings_with_offsets(
            tmp_path, ["exists", "missing"], max_per_dataset=500
        )
        assert pooled.shape == (20, 4)
        assert "exists" in offsets
        assert "missing" not in offsets

    def test_offsets_are_contiguous(self, tmp_path):
        """End of one dataset should equal start of next."""
        for i in range(3):
            np.savez(
                tmp_path / f"tabarena_ds{i}.npz",
                embeddings=np.random.randn(40 + i * 10, 4).astype(np.float32),
            )

        _, offsets = pool_embeddings_with_offsets(
            tmp_path, ["ds0", "ds1", "ds2"], max_per_dataset=500
        )
        assert offsets["ds0"][1] == offsets["ds1"][0]
        assert offsets["ds1"][1] == offsets["ds2"][0]
