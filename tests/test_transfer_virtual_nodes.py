"""Tests for virtual-node concept transfer core functions.

Tests cover:
1. extract_decoder_atoms: archetypal vs standard SAE, shape correctness
2. build_mnn_matches: clear matches, ambiguous cases, threshold filtering
3. fit_concept_map: ridge regression direction map
4. compute_virtual_atoms + calibrate_virtual_atom_norms: magnitude correction
5. compute_virtual_delta: shape and known-value checks
6. compute_virtual_delta_perrow: masking behaviour
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from scripts.intervene_sae import compute_perrow_logloss, get_improvable_rows
from scripts.transfer_virtual_nodes import (
    build_concept_bridge,
    build_mnn_matches,
    calibrate_virtual_atom_norms,
    compute_virtual_atoms,
    compute_virtual_delta,
    compute_virtual_delta_perrow,
    extract_decoder_atoms,
    fit_concept_map,
    load_cross_correlations,
)


# -- Fixtures ------------------------------------------------------------------


# -- extract_decoder_atoms -----------------------------------------------------


class TestExtractDecoderAtoms:
    def test_archetypal_path(self):
        """Uses get_archetypal_dictionary() when available."""
        sae = MagicMock()
        hidden_dim, input_dim = 64, 16
        dictionary = torch.randn(hidden_dim, input_dim)
        sae.get_archetypal_dictionary.return_value = dictionary

        atoms = extract_decoder_atoms(sae)

        sae.get_archetypal_dictionary.assert_called_once()
        assert atoms.shape == (hidden_dim, input_dim)
        torch.testing.assert_close(atoms, dictionary)

    def test_standard_path(self):
        """Falls back to W_dec.T for standard SAEs."""
        sae = MagicMock(spec=[])  # no get_archetypal_dictionary
        input_dim, hidden_dim = 16, 64
        sae.W_dec = torch.nn.Parameter(torch.randn(input_dim, hidden_dim))

        atoms = extract_decoder_atoms(sae)

        assert atoms.shape == (hidden_dim, input_dim)
        torch.testing.assert_close(atoms, sae.W_dec.detach().cpu().T)

    def test_returns_cpu_tensor(self):
        """Output is always on CPU regardless of model device."""
        sae = MagicMock()
        sae.get_archetypal_dictionary.return_value = torch.randn(32, 10)
        atoms = extract_decoder_atoms(sae)
        assert atoms.device == torch.device("cpu")


# -- build_mnn_matches ---------------------------------------------------------


class TestBuildMnnMatches:
    def test_identity_correlation(self):
        """Perfect identity correlation matrix yields diagonal matches."""
        n = 5
        corr = np.eye(n)
        matches = build_mnn_matches(corr, min_r=0.5)
        assert matches == [(i, i) for i in range(n)]

    def test_permuted_correlation(self):
        """Permuted identity recovers correct pairings."""
        perm = [2, 0, 4, 1, 3]
        n = 5
        corr = np.zeros((n, n))
        for i, j in enumerate(perm):
            corr[i, j] = 1.0
        matches = build_mnn_matches(corr, min_r=0.5)
        expected = [(i, perm[i]) for i in range(n)]
        assert sorted(matches) == sorted(expected)

    def test_threshold_filtering(self):
        """Pairs below min_r are excluded even if mutually best."""
        corr = np.array([[0.1, 0.05], [0.05, 0.1]])
        matches = build_mnn_matches(corr, min_r=0.2)
        assert matches == []

    def test_non_mutual_excluded(self):
        """When best matches are not mutual, no pair is returned."""
        # Row 0 and Row 1 both prefer column 0; column 0 prefers row 0.
        # So row 1 has no mutual match.
        corr = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
        ])
        matches = build_mnn_matches(corr, min_r=0.2)
        # (0, 0) is mutual. (1, 0) is not mutual (col 0 prefers row 0).
        # Col 1 prefers row 1 (0.2 > 0.1), row 1 prefers col 0 (0.8 > 0.2).
        # So only (0, 0) survives.
        assert matches == [(0, 0)]

    def test_empty_matrix(self):
        """Empty inputs produce empty output."""
        assert build_mnn_matches(np.zeros((0, 5)), min_r=0.1) == []
        assert build_mnn_matches(np.zeros((5, 0)), min_r=0.1) == []

    def test_rectangular_matrix(self):
        """Works with non-square correlation matrices."""
        # 3 source features, 5 target features
        corr = np.zeros((3, 5))
        corr[0, 2] = 0.9
        corr[1, 4] = 0.8
        corr[2, 0] = 0.7
        # Make sure column-side argmax points back
        # Col 2 max is row 0, col 4 max is row 1, col 0 max is row 2
        matches = build_mnn_matches(corr, min_r=0.5)
        assert sorted(matches) == [(0, 2), (1, 4), (2, 0)]

    def test_all_zeros(self):
        """All-zero matrix returns no matches (below any positive threshold)."""
        corr = np.zeros((4, 4))
        matches = build_mnn_matches(corr, min_r=0.0)
        # argmax on all-zeros returns index 0 for every row/col.
        # Only (0, 0) can be mutual, and 0.0 >= 0.0 is True.
        assert (0, 0) in matches
        # With positive threshold, nothing matches.
        matches_strict = build_mnn_matches(corr, min_r=0.01)
        assert matches_strict == []


# -- fit_concept_map -----------------------------------------------------------


class TestFitConceptMap:
    def test_perfect_recovery(self):
        """Ridge with low alpha recovers M_true for noiseless atom pairs."""
        np.random.seed(0)
        n_pairs, d_source, d_target = 50, 10, 8
        M_true = np.random.randn(d_target, d_source)
        atoms_s = np.random.randn(n_pairs, d_source)
        atoms_t = atoms_s @ M_true.T

        M, r2 = fit_concept_map(atoms_s, atoms_t, alpha=1e-8)
        assert M.shape == M_true.shape
        np.testing.assert_allclose(M, M_true, atol=0.05)
        assert r2 > 0.99

    def test_few_pairs_nan_r2(self):
        """Fewer than 10 pairs yields r2 = nan."""
        np.random.seed(2)
        atoms_s = np.random.randn(5, 4)
        atoms_t = np.random.randn(5, 3)
        M, r2 = fit_concept_map(atoms_s, atoms_t)
        assert M.shape == (3, 4)
        assert np.isnan(r2)


# -- compute_virtual_atoms + calibrate_virtual_atom_norms ----------------------


class TestComputeVirtualAtoms:
    def test_shape(self):
        """Output shape is (n_unmatched, d_target)."""
        np.random.seed(4)
        n, d_s, d_t = 7, 10, 8
        M = np.random.randn(d_t, d_s)
        atoms = np.random.randn(n, d_s)
        virtual = compute_virtual_atoms(atoms, M)
        assert virtual.shape == (n, d_t)

    def test_identity_map(self):
        """Identity map returns atoms unchanged."""
        atoms = np.random.randn(10, 8)
        virtual = compute_virtual_atoms(atoms, np.eye(8))
        np.testing.assert_allclose(virtual, atoms)


class TestCalibrateVirtualAtomNorms:
    def test_uniform_scaling(self):
        """When all landmarks have 2x norm ratio, virtual atoms scale by 2x."""
        np.random.seed(10)
        n_unmatched, d = 5, 8
        virtual = np.random.randn(n_unmatched, d)
        src_norms = np.ones(n_unmatched) * 1.0
        matched_src = np.ones(10) * 1.0
        matched_tgt = np.ones(10) * 2.0  # 2x scaling
        result = calibrate_virtual_atom_norms(
            virtual, src_norms, matched_src, matched_tgt,
        )
        result_norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(result_norms, 2.0, atol=0.1)

    def test_norm_dependent_scaling(self):
        """Larger source atoms get smaller target norms (negative correlation)."""
        np.random.seed(20)
        n_unmatched, d = 3, 8
        virtual = np.random.randn(n_unmatched, d)
        src_norms = np.array([0.5, 1.0, 1.5])
        # Landmark relationship: target_norm = -0.5 * source_norm + 2.0
        matched_src = np.linspace(0.3, 1.8, 15)
        matched_tgt = -0.5 * matched_src + 2.0
        result = calibrate_virtual_atom_norms(
            virtual, src_norms, matched_src, matched_tgt,
        )
        result_norms = np.linalg.norm(result, axis=1)
        # Expected: 1.75, 1.50, 1.25
        np.testing.assert_allclose(result_norms, [1.75, 1.50, 1.25], atol=0.05)

    def test_preserves_direction(self):
        """Calibration changes norm but preserves direction."""
        np.random.seed(30)
        virtual = np.array([[1.0, 2.0, 3.0]])
        src_norms = np.array([1.0])
        matched_src = np.ones(10)
        matched_tgt = np.ones(10) * 0.5
        result = calibrate_virtual_atom_norms(
            virtual, src_norms, matched_src, matched_tgt,
        )
        # Direction should be the same
        orig_dir = virtual[0] / np.linalg.norm(virtual[0])
        res_dir = result[0] / np.linalg.norm(result[0])
        np.testing.assert_allclose(orig_dir, res_dir, atol=1e-6)


# -- compute_virtual_delta ----------------------------------------------------


class TestComputeVirtualDelta:
    def test_shape(self):
        """Output shape is (n_rows, d_target)."""
        n_rows, n_feat, d_t = 20, 5, 8
        acts = np.random.randn(n_rows, n_feat)
        vatoms = np.random.randn(n_feat, d_t)
        delta = compute_virtual_delta(acts, vatoms)
        assert delta.shape == (n_rows, d_t)

    def test_zero_activations(self):
        """Zero activations produce zero delta."""
        acts = np.zeros((10, 5))
        vatoms = np.random.randn(5, 8)
        delta = compute_virtual_delta(acts, vatoms)
        np.testing.assert_allclose(delta, np.zeros((10, 8)))

    def test_single_feature(self):
        """Single feature: delta = activation * virtual_atom."""
        acts = np.array([[3.0]])
        vatoms = np.array([[1.0, 2.0, -1.0]])
        delta = compute_virtual_delta(acts, vatoms)
        np.testing.assert_allclose(delta, np.array([[3.0, 6.0, -3.0]]))

    def test_linearity(self):
        """Delta scales linearly with activations."""
        np.random.seed(5)
        acts = np.random.randn(10, 4)
        vatoms = np.random.randn(4, 6)
        delta1 = compute_virtual_delta(acts, vatoms)
        delta2 = compute_virtual_delta(2.0 * acts, vatoms)
        np.testing.assert_allclose(delta2, 2.0 * delta1)


# -- compute_virtual_delta_perrow ----------------------------------------------


class TestComputeVirtualDeltaPerrow:
    def test_full_mask_matches_unmasked(self):
        """All-ones mask gives same result as unmasked delta."""
        np.random.seed(6)
        n_rows, n_feat, d_t = 15, 5, 8
        acts = np.random.randn(n_rows, n_feat)
        vatoms = np.random.randn(n_feat, d_t)
        masks = np.ones((n_rows, n_feat))

        delta_masked = compute_virtual_delta_perrow(acts, vatoms, masks)
        delta_unmasked = compute_virtual_delta(acts, vatoms)
        np.testing.assert_allclose(delta_masked, delta_unmasked)

    def test_zero_mask_gives_zero(self):
        """All-zeros mask produces zero delta."""
        acts = np.random.randn(10, 5)
        vatoms = np.random.randn(5, 8)
        masks = np.zeros((10, 5))

        delta = compute_virtual_delta_perrow(acts, vatoms, masks)
        np.testing.assert_allclose(delta, np.zeros((10, 8)))

    def test_per_row_selectivity(self):
        """Different masks per row select different features."""
        # 2 rows, 3 features, 2 target dims
        acts = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        vatoms = np.eye(3, 2)  # feature 0 -> dim 0, feature 1 -> dim 1
        masks = np.array([
            [1, 0, 0],  # row 0: only feature 0
            [0, 1, 0],  # row 1: only feature 1
        ], dtype=float)

        delta = compute_virtual_delta_perrow(acts, vatoms, masks)
        # Row 0: [1, 0, 0] @ [[1,0],[0,1],[0,0]] = [1, 0]
        # Row 1: [0, 5, 0] @ [[1,0],[0,1],[0,0]] = [0, 5]
        np.testing.assert_allclose(delta, np.array([[1.0, 0.0], [0.0, 5.0]]))

    def test_shape(self):
        """Output shape is (n_rows, d_target)."""
        n_rows, n_feat, d_t = 20, 5, 8
        acts = np.random.randn(n_rows, n_feat)
        vatoms = np.random.randn(n_feat, d_t)
        masks = np.ones((n_rows, n_feat))
        delta = compute_virtual_delta_perrow(acts, vatoms, masks)
        assert delta.shape == (n_rows, d_t)

    def test_fractional_masks(self):
        """Non-binary mask values scale activations proportionally."""
        acts = np.array([[2.0, 4.0]])
        vatoms = np.array([[1.0], [1.0]])
        masks = np.array([[0.5, 0.25]])

        delta = compute_virtual_delta_perrow(acts, vatoms, masks)
        # (2*0.5 + 4*0.25) * 1.0 = 2.0
        np.testing.assert_allclose(delta, np.array([[2.0]]))


# -- load_cross_correlations -------------------------------------------------


class TestLoadCrossCorrelations:
    def test_forward_order(self, tmp_path):
        """Loads file when source__target.npz exists."""
        corr = np.random.rand(10, 15).astype(np.float32)
        idx_a = np.arange(10)
        idx_b = np.arange(15)
        np.savez(
            tmp_path / "TabPFN__TabICL.npz",
            corr_matrix=corr,
            indices_a=idx_a,
            indices_b=idx_b,
            model_a="TabPFN",
            model_b="TabICL",
        )
        c, ia, ib = load_cross_correlations("tabpfn", "tabicl", tmp_path)
        np.testing.assert_array_equal(c, corr)
        np.testing.assert_array_equal(ia, idx_a)
        np.testing.assert_array_equal(ib, idx_b)

    def test_reversed_order(self, tmp_path):
        """Loads and transposes when only target__source.npz exists."""
        corr = np.random.rand(15, 10).astype(np.float32)
        idx_a = np.arange(15)  # stored as model_a = TabICL
        idx_b = np.arange(10)  # stored as model_b = TabPFN
        np.savez(
            tmp_path / "TabICL__TabPFN.npz",
            corr_matrix=corr,
            indices_a=idx_a,
            indices_b=idx_b,
            model_a="TabICL",
            model_b="TabPFN",
        )
        # Request tabpfn (source) -> tabicl (target)
        c, ia, ib = load_cross_correlations("tabpfn", "tabicl", tmp_path)
        assert c.shape == (10, 15)  # transposed
        np.testing.assert_array_equal(c, corr.T)
        np.testing.assert_array_equal(ia, idx_b)  # swapped
        np.testing.assert_array_equal(ib, idx_a)

    def test_file_not_found(self, tmp_path):
        """Raises FileNotFoundError when no file exists for the pair."""
        with pytest.raises(FileNotFoundError, match="No cross-correlation"):
            load_cross_correlations("tabpfn", "mitra", tmp_path)


# -- build_concept_bridge ----------------------------------------------------


class TestBuildConceptBridge:
    @pytest.fixture
    def bridge_inputs(self):
        """Create mock SAEs and a synthetic cross-correlation setup.

        Source SAE: hidden_dim=20, input_dim=8
        Target SAE: hidden_dim=30, input_dim=16
        Alive features: 10 source, 12 target
        """
        np.random.seed(42)

        # Source SAE mock (standard path, no archetypal)
        sae_source = MagicMock(spec=[])
        d_source_hidden, d_source_input = 20, 8
        sae_source.W_dec = torch.nn.Parameter(
            torch.randn(d_source_input, d_source_hidden)
        )

        # Target SAE mock (standard path)
        sae_target = MagicMock(spec=[])
        d_target_hidden, d_target_input = 30, 16
        sae_target.W_dec = torch.nn.Parameter(
            torch.randn(d_target_input, d_target_hidden)
        )

        # Alive features: 10 of 20 source, 12 of 30 target
        indices_a = np.array([0, 2, 4, 5, 7, 9, 11, 13, 15, 17])
        indices_b = np.array([1, 3, 5, 6, 8, 10, 12, 14, 16, 18, 22, 25])

        # Build a correlation matrix where we engineer clear MNN matches
        corr = np.random.rand(10, 12).astype(np.float32) * 0.1
        # Create 5 clear mutual matches
        for k in range(5):
            corr[k, k] = 0.8 + 0.02 * k

        # Unmatched source features: global indices from the alive set
        # Use indices_a[5:8] = [9, 11, 13] as unmatched
        unmatched = [int(indices_a[i]) for i in range(5, 8)]

        return {
            "sae_source": sae_source,
            "sae_target": sae_target,
            "corr_matrix": corr,
            "indices_a": indices_a,
            "indices_b": indices_b,
            "unmatched": unmatched,
            "d_source_input": d_source_input,
            "d_target_input": d_target_input,
        }

    def test_return_keys(self, bridge_inputs):
        """Return dict contains all expected keys."""
        result = build_concept_bridge(
            sae_source=bridge_inputs["sae_source"],
            sae_target=bridge_inputs["sae_target"],
            corr_matrix=bridge_inputs["corr_matrix"],
            indices_a=bridge_inputs["indices_a"],
            indices_b=bridge_inputs["indices_b"],
            unmatched_source_features=bridge_inputs["unmatched"],
        )
        expected_keys = {
            "virtual_atoms",
            "concept_map_r2",
            "n_matched_pairs",
            "matched_pairs",
            "unmatched_indices",
        }
        assert set(result.keys()) == expected_keys

    def test_virtual_atoms_shape(self, bridge_inputs):
        """Virtual atoms have shape (n_unmatched, d_target)."""
        result = build_concept_bridge(
            sae_source=bridge_inputs["sae_source"],
            sae_target=bridge_inputs["sae_target"],
            corr_matrix=bridge_inputs["corr_matrix"],
            indices_a=bridge_inputs["indices_a"],
            indices_b=bridge_inputs["indices_b"],
            unmatched_source_features=bridge_inputs["unmatched"],
        )
        n_unmatched = len(bridge_inputs["unmatched"])
        d_target = bridge_inputs["d_target_input"]
        assert result["virtual_atoms"].shape == (n_unmatched, d_target)

    def test_n_matched_pairs(self, bridge_inputs):
        """Correct number of MNN-matched pairs."""
        result = build_concept_bridge(
            sae_source=bridge_inputs["sae_source"],
            sae_target=bridge_inputs["sae_target"],
            corr_matrix=bridge_inputs["corr_matrix"],
            indices_a=bridge_inputs["indices_a"],
            indices_b=bridge_inputs["indices_b"],
            unmatched_source_features=bridge_inputs["unmatched"],
        )
        # We engineered 5 clear diagonal matches with corr >= 0.8
        assert result["n_matched_pairs"] == 5

    def test_matched_pairs_use_global_indices(self, bridge_inputs):
        """Matched pairs contain global (not local) feature indices."""
        result = build_concept_bridge(
            sae_source=bridge_inputs["sae_source"],
            sae_target=bridge_inputs["sae_target"],
            corr_matrix=bridge_inputs["corr_matrix"],
            indices_a=bridge_inputs["indices_a"],
            indices_b=bridge_inputs["indices_b"],
            unmatched_source_features=bridge_inputs["unmatched"],
        )
        for src_idx, tgt_idx in result["matched_pairs"]:
            assert src_idx in bridge_inputs["indices_a"]
            assert tgt_idx in bridge_inputs["indices_b"]

    def test_unmatched_indices_preserved(self, bridge_inputs):
        """Unmatched indices are returned verbatim."""
        result = build_concept_bridge(
            sae_source=bridge_inputs["sae_source"],
            sae_target=bridge_inputs["sae_target"],
            corr_matrix=bridge_inputs["corr_matrix"],
            indices_a=bridge_inputs["indices_a"],
            indices_b=bridge_inputs["indices_b"],
            unmatched_source_features=bridge_inputs["unmatched"],
        )
        assert result["unmatched_indices"] == bridge_inputs["unmatched"]

    def test_concept_map_r2_is_float(self, bridge_inputs):
        """Concept map R² is a float."""
        result = build_concept_bridge(
            sae_source=bridge_inputs["sae_source"],
            sae_target=bridge_inputs["sae_target"],
            corr_matrix=bridge_inputs["corr_matrix"],
            indices_a=bridge_inputs["indices_a"],
            indices_b=bridge_inputs["indices_b"],
            unmatched_source_features=bridge_inputs["unmatched"],
        )
        assert isinstance(result["concept_map_r2"], float)


# -- Shared utilities: compute_perrow_logloss, get_improvable_rows ----------


class TestComputePerrowLogloss:
    def test_perfect_predictions(self):
        """Logloss near zero for correct confident predictions."""
        y = np.array([0, 1, 0, 1])
        preds = np.array([0.01, 0.99, 0.01, 0.99])
        ll = compute_perrow_logloss(preds, y)
        assert ll.shape == (4,)
        assert np.all(ll < 0.02)

    def test_bad_predictions(self):
        """Logloss high for wrong confident predictions."""
        y = np.array([0, 1, 0, 1])
        preds = np.array([0.99, 0.01, 0.99, 0.01])
        ll = compute_perrow_logloss(preds, y)
        assert np.all(ll > 2.0)

    def test_2d_input(self):
        """Handles (n, 2) probability arrays correctly."""
        y = np.array([0, 1])
        preds_2d = np.array([[0.9, 0.1], [0.1, 0.9]])
        preds_1d = np.array([0.1, 0.9])
        ll_2d = compute_perrow_logloss(preds_2d, y)
        ll_1d = compute_perrow_logloss(preds_1d, y)
        np.testing.assert_allclose(ll_2d, ll_1d)

    def test_clipping(self):
        """Extreme probabilities (0, 1) don't produce inf."""
        y = np.array([0, 1])
        preds = np.array([0.0, 1.0])
        ll = compute_perrow_logloss(preds, y)
        assert np.all(np.isfinite(ll))


class TestGetImprovableRows:
    def test_clear_winner(self):
        """Strong model clearly better on all rows."""
        y = np.array([1, 1, 0, 0])
        preds_strong = np.array([0.95, 0.90, 0.05, 0.10])
        preds_weak = np.array([0.50, 0.50, 0.50, 0.50])
        mask = get_improvable_rows(preds_strong, preds_weak, y)
        assert mask.shape == (4,)
        assert np.all(mask)

    def test_weak_better_excluded(self):
        """Rows where weak model is better are excluded."""
        y = np.array([1, 0])
        preds_strong = np.array([0.5, 0.5])  # mediocre
        preds_weak = np.array([0.95, 0.05])  # perfect
        mask = get_improvable_rows(preds_strong, preds_weak, y)
        assert not np.any(mask)

    def test_mixed(self):
        """Mix of rows where each model wins."""
        y = np.array([1, 0])
        preds_strong = np.array([0.9, 0.5])  # strong wins row 0
        preds_weak = np.array([0.5, 0.1])    # weak wins row 1
        mask = get_improvable_rows(preds_strong, preds_weak, y)
        assert mask[0] == True
        assert mask[1] == False

    def test_2d_inputs(self):
        """Handles (n, 2) probability arrays."""
        y = np.array([1, 0])
        preds_strong = np.array([[0.1, 0.9], [0.9, 0.1]])
        preds_weak = np.array([[0.5, 0.5], [0.5, 0.5]])
        mask = get_improvable_rows(preds_strong, preds_weak, y)
        assert np.all(mask)

    def test_consistent_with_ablation_convention(self):
        """Ablation convention: orig_gap = target_ll - baseline_ll > 0
        is equivalent to get_improvable_rows returning True.

        target_ll = weaker model, baseline_ll = stronger model.
        orig_gap > 0 means stronger has lower logloss.
        """
        np.random.seed(42)
        y = np.random.randint(0, 2, size=100)
        preds_strong = np.clip(np.where(y == 1, 0.8, 0.2) + np.random.randn(100) * 0.1, 0.01, 0.99)
        preds_weak = np.clip(np.where(y == 1, 0.6, 0.4) + np.random.randn(100) * 0.2, 0.01, 0.99)

        # Shared function
        mask = get_improvable_rows(preds_strong, preds_weak, y)

        # Ablation convention (inline)
        ll_strong = compute_perrow_logloss(preds_strong, y)
        ll_weak = compute_perrow_logloss(preds_weak, y)
        orig_gap = ll_weak - ll_strong  # target_ll - baseline_ll
        ablation_fixable = orig_gap > 0

        np.testing.assert_array_equal(mask, ablation_fixable)
