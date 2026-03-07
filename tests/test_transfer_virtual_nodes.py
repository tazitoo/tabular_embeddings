"""Tests for virtual-node concept transfer core functions.

Tests cover:
1. extract_decoder_atoms: archetypal vs standard SAE, shape correctness
2. build_mnn_matches: clear matches, ambiguous cases, threshold filtering
3. fit_concept_map: perfect recovery, noisy recovery, cross-validated R^2
4. compute_virtual_atoms: shape and known-value checks
5. compute_virtual_delta: shape and known-value checks
6. compute_virtual_delta_perrow: masking behaviour
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from scripts.transfer_virtual_nodes import (
    build_mnn_matches,
    compute_virtual_atoms,
    compute_virtual_delta,
    compute_virtual_delta_perrow,
    extract_decoder_atoms,
    fit_concept_map,
)


# -- Fixtures ------------------------------------------------------------------


@pytest.fixture
def identity_map():
    """A trivial identity concept map for d=8."""
    return np.eye(8)


@pytest.fixture
def known_linear_atoms():
    """Matched atom pairs linked by a known linear map M_true."""
    np.random.seed(0)
    n_pairs, d_source, d_target = 50, 10, 8
    M_true = np.random.randn(d_target, d_source)
    atoms_source = np.random.randn(n_pairs, d_source)
    atoms_target = atoms_source @ M_true.T
    return atoms_source, atoms_target, M_true


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
    def test_perfect_recovery(self, known_linear_atoms):
        """Ridge with low alpha recovers M_true for noiseless atom pairs."""
        atoms_s, atoms_t, M_true = known_linear_atoms
        M, r2 = fit_concept_map(atoms_s, atoms_t, alpha=1e-8)

        assert M.shape == M_true.shape
        np.testing.assert_allclose(M, M_true, atol=0.05)
        assert r2 > 0.99, f"Expected R^2 > 0.99, got {r2:.4f}"

    def test_noisy_recovery(self):
        """Noisy atom pairs yield lower but positive R^2."""
        np.random.seed(1)
        n, d_s, d_t = 40, 10, 8
        M_true = np.random.randn(d_t, d_s) * 0.5
        atoms_s = np.random.randn(n, d_s)
        noise = np.random.randn(n, d_t) * 0.3
        atoms_t = atoms_s @ M_true.T + noise

        M, r2 = fit_concept_map(atoms_s, atoms_t, alpha=1.0)
        assert M.shape == (d_t, d_s)
        # R^2 should be positive but not perfect
        assert 0.0 < r2 < 1.0, f"Expected 0 < R^2 < 1, got {r2:.4f}"

    def test_few_pairs_nan_r2(self):
        """Fewer than 10 pairs yields r2 = nan."""
        np.random.seed(2)
        atoms_s = np.random.randn(5, 4)
        atoms_t = np.random.randn(5, 3)

        M, r2 = fit_concept_map(atoms_s, atoms_t)
        assert M.shape == (3, 4)
        assert np.isnan(r2)

    def test_output_shapes(self):
        """M has shape (d_target, d_source)."""
        np.random.seed(3)
        atoms_s = np.random.randn(20, 6)
        atoms_t = np.random.randn(20, 12)

        M, _ = fit_concept_map(atoms_s, atoms_t)
        assert M.shape == (12, 6)


# -- compute_virtual_atoms ----------------------------------------------------


class TestComputeVirtualAtoms:
    def test_identity_map(self, identity_map):
        """Identity map returns atoms unchanged."""
        atoms = np.random.randn(10, 8)
        virtual = compute_virtual_atoms(atoms, identity_map)
        np.testing.assert_allclose(virtual, atoms)

    def test_shape(self):
        """Output shape is (n_unmatched, d_target)."""
        np.random.seed(4)
        n, d_s, d_t = 7, 10, 8
        M = np.random.randn(d_t, d_s)
        atoms = np.random.randn(n, d_s)
        virtual = compute_virtual_atoms(atoms, M)
        assert virtual.shape == (n, d_t)

    def test_known_transform(self):
        """Scaling map doubles each atom."""
        M = 2.0 * np.eye(5)
        atoms = np.ones((3, 5))
        virtual = compute_virtual_atoms(atoms, M)
        np.testing.assert_allclose(virtual, 2.0 * np.ones((3, 5)))

    def test_zero_atoms(self):
        """Zero source atoms produce zero virtual atoms."""
        M = np.random.randn(6, 4)
        atoms = np.zeros((3, 4))
        virtual = compute_virtual_atoms(atoms, M)
        np.testing.assert_allclose(virtual, np.zeros((3, 6)))


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
