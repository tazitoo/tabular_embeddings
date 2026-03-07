#!/usr/bin/env python3
"""Virtual-node concept transfer between tabular foundation models.

Instead of mapping concept contributions through an embedding-level linear map,
builds a concept-level map from matched decoder atom pairs and creates virtual
latent nodes in the target SAE for unmatched concepts.

The concept-level map M is fitted on matched decoder atoms -- directions that
represent the same concept in both models. For unmatched concepts, the virtual
decoder atom d_B_virtual = M @ d_A gives the interpolated direction where that
concept would sit in the target model's representation space.

Transfer delta = activation * virtual_atom, structurally identical to how the
target model's own SAE contributes to its embeddings.

Usage:
    python scripts/transfer_virtual_nodes.py --source tabpfn --target tabicl \\
        --dataset credit-g --device cuda --perrow
"""

import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# -- Decoder atom extraction ---------------------------------------------------


def extract_decoder_atoms(sae: torch.nn.Module) -> torch.Tensor:
    """Extract decoder dictionary atoms from an SAE.

    For Archetypal/Matryoshka-Archetypal SAEs, uses the convex-hull dictionary.
    For standard SAEs, transposes W_dec.

    Returns:
        Tensor of shape (hidden_dim, input_dim) -- one atom per row.
    """
    if hasattr(sae, "get_archetypal_dictionary"):
        return sae.get_archetypal_dictionary().detach().cpu()

    # W_dec is (input_dim, hidden_dim); transpose to (hidden_dim, input_dim)
    return sae.W_dec.detach().cpu().T


# -- Mutual nearest-neighbor matching -----------------------------------------


def build_mnn_matches(
    corr_matrix: np.ndarray,
    min_r: float = 0.2,
) -> List[Tuple[int, int]]:
    """Find mutual nearest-neighbor pairs from an absolute correlation matrix.

    For each row i, finds argmax_j C[i,j]. For each column j, finds argmax_i
    C[i,j]. A pair (i, j) is kept only if both point at each other and
    C[i,j] >= min_r.

    Args:
        corr_matrix: (n_a, n_b) absolute correlation matrix.
        min_r: Minimum correlation to accept a match.

    Returns:
        List of (idx_a, idx_b) tuples for mutually best-matching pairs.
    """
    n_a, n_b = corr_matrix.shape
    if n_a == 0 or n_b == 0:
        return []

    best_col_per_row = corr_matrix.argmax(axis=1)  # (n_a,)
    best_row_per_col = corr_matrix.argmax(axis=0)  # (n_b,)

    matches = []
    for i in range(n_a):
        j = int(best_col_per_row[i])
        if best_row_per_col[j] == i and corr_matrix[i, j] >= min_r:
            matches.append((i, j))

    return matches


# -- Concept-level map ---------------------------------------------------------


def fit_concept_map(
    atoms_source: np.ndarray,
    atoms_target: np.ndarray,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Fit a ridge regression map from source to target decoder atoms.

    Each row of atoms_source / atoms_target is a matched decoder atom pair.
    The map M satisfies: atoms_target ~ atoms_source @ M.T, with
    fit_intercept=False (directions, not points).

    Args:
        atoms_source: (n_pairs, d_source) matched source decoder atoms.
        atoms_target: (n_pairs, d_target) matched target decoder atoms.
        alpha: Ridge regularisation strength.

    Returns:
        M: (d_target, d_source) concept-level linear map.
        r2: Cross-validated R^2 (nan if < 10 pairs).
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(atoms_source, atoms_target)
    M = reg.coef_  # (d_target, d_source)

    n_pairs = atoms_source.shape[0]
    if n_pairs >= 10:
        n_folds = min(5, n_pairs)
        scores = cross_val_score(
            Ridge(alpha=alpha, fit_intercept=False),
            atoms_source,
            atoms_target,
            cv=n_folds,
            scoring="r2",
        )
        r2 = float(scores.mean())
    else:
        r2 = float("nan")

    return M, r2


# -- Virtual atom construction -------------------------------------------------


def compute_virtual_atoms(
    atoms_source_unmatched: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """Project unmatched source decoder atoms into the target space.

    d_B_virtual = atoms_source_unmatched @ M.T

    Args:
        atoms_source_unmatched: (n_unmatched, d_source) unmatched source atoms.
        M: (d_target, d_source) concept-level map.

    Returns:
        (n_unmatched, d_target) virtual decoder atoms in target space.
    """
    return atoms_source_unmatched @ M.T


# -- Transfer delta computation ------------------------------------------------


def compute_virtual_delta(
    activations: np.ndarray,
    virtual_atoms: np.ndarray,
) -> np.ndarray:
    """Compute transfer delta from activations and virtual decoder atoms.

    delta = activations @ virtual_atoms

    Args:
        activations: (n_rows, n_unmatched) SAE activations for unmatched features.
        virtual_atoms: (n_unmatched, d_target) virtual decoder atoms.

    Returns:
        (n_rows, d_target) transfer deltas in target embedding space.
    """
    return activations @ virtual_atoms


def compute_virtual_delta_perrow(
    activations: np.ndarray,
    virtual_atoms: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """Compute per-row masked transfer delta.

    delta = (activations * masks) @ virtual_atoms

    Args:
        activations: (n_rows, n_unmatched) SAE activations for unmatched features.
        virtual_atoms: (n_unmatched, d_target) virtual decoder atoms.
        masks: (n_rows, n_unmatched) binary masks (1 = transfer, 0 = skip).

    Returns:
        (n_rows, d_target) per-row transfer deltas in target embedding space.
    """
    return (activations * masks) @ virtual_atoms
