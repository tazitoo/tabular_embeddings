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

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.intervene_sae import (
    DEFAULT_LAYERS_PATH,
    DEFAULT_SAE_DIR,
    DEFAULT_TRAINING_DIR,
    _perrow_importance,
    _perrow_rankings,
    get_extraction_layer,
    intervene,
    load_sae,
    load_training_mean,
)
from scripts.concept_performance_diagnostic import _load_splits, DISPLAY_NAMES
from scripts.plot_prediction_scatter import (
    _logloss,
    get_active_feature_count,
    get_unmatched_features,
    plot_logloss_curve,
    plot_perrow_diagnostic,
    plot_perrow_results,
    plot_perrow_scatter,
    plot_prediction_scatter,
)
from scripts.transfer_concepts import capture_embeddings

logger = logging.getLogger(__name__)

DEFAULT_CROSS_CORR_DIR = PROJECT_ROOT / "output" / "sae_cross_correlations"

# Display name mapping for cross-correlation filenames
CROSS_CORR_NAMES = {
    "tabpfn": "TabPFN",
    "tabicl": "TabICL",
    "mitra": "Mitra",
    "tabdpt": "TabDPT",
    "hyperfast": "HyperFast",
    "carte": "CARTE",
    "tabula8b": "Tabula-8B",
}


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


# -- Cross-correlation loading ------------------------------------------------


def load_cross_correlations(
    source_model: str,
    target_model: str,
    cross_corr_dir: Path = DEFAULT_CROSS_CORR_DIR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load cross-correlation matrix between two models' SAE features.

    Tries both orderings of the model pair (A__B.npz and B__A.npz).
    If the file is stored in reversed order, transposes the correlation
    matrix and swaps indices so the result is always (source, target).

    Args:
        source_model: Model key (e.g. "tabpfn", "tabicl").
        target_model: Model key (e.g. "mitra", "tabdpt").
        cross_corr_dir: Directory containing cross-correlation NPZ files.

    Returns:
        corr_matrix: (n_source_alive, n_target_alive) correlation matrix.
        indices_source: Alive feature indices for the source SAE.
        indices_target: Alive feature indices for the target SAE.

    Raises:
        FileNotFoundError: If neither ordering of the model pair is found.
    """
    cross_corr_dir = Path(cross_corr_dir)
    name_a = CROSS_CORR_NAMES.get(source_model, source_model)
    name_b = CROSS_CORR_NAMES.get(target_model, target_model)

    path_ab = cross_corr_dir / f"{name_a}__{name_b}.npz"
    path_ba = cross_corr_dir / f"{name_b}__{name_a}.npz"

    if path_ab.exists():
        data = np.load(path_ab, allow_pickle=True)
        corr_matrix = data["corr_matrix"]
        indices_source = data["indices_a"]
        indices_target = data["indices_b"]
        logger.debug("Loaded %s (shape %s)", path_ab.name, corr_matrix.shape)
    elif path_ba.exists():
        data = np.load(path_ba, allow_pickle=True)
        corr_matrix = data["corr_matrix"].T
        indices_source = data["indices_b"]
        indices_target = data["indices_a"]
        logger.debug(
            "Loaded %s (reversed, transposed to %s)",
            path_ba.name,
            corr_matrix.shape,
        )
    else:
        raise FileNotFoundError(
            f"No cross-correlation file found for {name_a} <-> {name_b}. "
            f"Tried {path_ab} and {path_ba}."
        )

    return corr_matrix, indices_source, indices_target


# -- Concept bridge builder ---------------------------------------------------


def build_concept_bridge(
    sae_source: torch.nn.Module,
    sae_target: torch.nn.Module,
    corr_matrix: np.ndarray,
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    unmatched_source_features: List[int],
    min_match_r: float = 0.2,
    ridge_alpha: float = 1.0,
) -> Dict:
    """Build a concept bridge from source SAE to target SAE.

    Orchestrates the full pipeline: extract decoder atoms, find MNN matches,
    fit a concept-level linear map on matched pairs, and project unmatched
    source features into the target's embedding space as virtual decoder atoms.

    Args:
        sae_source: Source SAE model.
        sae_target: Target SAE model.
        corr_matrix: (n_alive_a, n_alive_b) cross-correlation matrix.
        indices_a: Global feature indices for alive source features.
        indices_b: Global feature indices for alive target features.
        unmatched_source_features: Global feature indices of source features
            to project as virtual nodes (must be a subset of indices_a).
        min_match_r: Minimum correlation for MNN matching.
        ridge_alpha: Ridge regularisation strength for the concept map.

    Returns:
        Dict with keys:
            virtual_atoms: (n_unmatched, d_target) virtual decoder atoms.
            concept_map_r2: float, cross-validated R^2 of the concept map.
            n_matched_pairs: int, number of MNN-matched decoder atom pairs.
            matched_pairs: list of (source_global, target_global) tuples.
            unmatched_indices: list of ints (the input unmatched feature indices).
            concept_map_M: (d_target, d_source) concept-level linear map.
    """
    # 1. Extract full decoder atom matrices
    atoms_source = extract_decoder_atoms(sae_source).numpy()  # (H_s, d_s)
    atoms_target = extract_decoder_atoms(sae_target).numpy()  # (H_t, d_t)
    logger.info(
        "Decoder atoms: source %s, target %s", atoms_source.shape, atoms_target.shape
    )

    # 2. MNN matching on the cross-correlation matrix (local indices)
    local_matches = build_mnn_matches(corr_matrix, min_r=min_match_r)
    logger.info(
        "MNN matches: %d pairs (min_r=%.2f)", len(local_matches), min_match_r
    )

    # 3. Convert local MNN indices to global feature indices
    indices_a = np.asarray(indices_a)
    indices_b = np.asarray(indices_b)
    matched_pairs = [
        (int(indices_a[i]), int(indices_b[j])) for i, j in local_matches
    ]

    # 4. Gather matched decoder atoms using global indices
    matched_source_atoms = np.stack(
        [atoms_source[gi] for gi, _ in matched_pairs], axis=0
    )  # (n_matched, d_source)
    matched_target_atoms = np.stack(
        [atoms_target[gj] for _, gj in matched_pairs], axis=0
    )  # (n_matched, d_target)

    # 5. Fit concept-level linear map on matched pairs
    M, r2 = fit_concept_map(matched_source_atoms, matched_target_atoms, alpha=ridge_alpha)
    logger.info("Concept map: R²=%.4f, shape=%s", r2, M.shape)

    # 6. Compute virtual atoms for unmatched source features
    unmatched_atoms = np.stack(
        [atoms_source[gi] for gi in unmatched_source_features], axis=0
    )  # (n_unmatched, d_source)
    virtual = compute_virtual_atoms(unmatched_atoms, M)
    logger.info(
        "Virtual atoms: %d unmatched -> shape %s",
        len(unmatched_source_features),
        virtual.shape,
    )

    return {
        "virtual_atoms": virtual,
        "concept_map_r2": r2,
        "n_matched_pairs": len(matched_pairs),
        "matched_pairs": matched_pairs,
        "unmatched_indices": list(unmatched_source_features),
        "concept_map_M": M,
    }


# -- Encoding & delta helpers -------------------------------------------------


def _encode_unmatched_activations(
    sae_source: torch.nn.Module,
    emb_source: torch.Tensor,
    unmatched_indices: List[int],
    data_mean: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Encode source embeddings and extract activations for unmatched features.

    Returns:
        (n_rows, n_unmatched) activation matrix.
    """
    with torch.no_grad():
        x = emb_source
        if data_mean is not None:
            x = x - data_mean
        h = sae_source.encode(x)
    return h[:, unmatched_indices].cpu().numpy()


def _make_virtual_delta(
    activations: np.ndarray,
    virtual_atoms: np.ndarray,
    feature_mask: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Build delta from activations and virtual atoms, optionally masked.

    Args:
        activations: (n_rows, n_unmatched) source activations.
        virtual_atoms: (n_unmatched, d_target) virtual decoder atoms.
        feature_mask: (n_unmatched,) boolean -- only include these features.

    Returns:
        (n_rows, d_target) delta tensor.
    """
    if feature_mask is not None:
        activations = activations[:, feature_mask]
        virtual_atoms = virtual_atoms[feature_mask]
    delta = compute_virtual_delta(activations, virtual_atoms)
    return torch.tensor(delta, dtype=torch.float32)


def _build_full_delta_from_parts(
    delta_ctx: torch.Tensor,
    delta_query: torch.Tensor,
) -> torch.Tensor:
    """Concatenate context and query deltas into a single full-sequence delta."""
    return torch.cat([delta_ctx, delta_query], dim=0)


# -- Cumulative sweep ---------------------------------------------------------


def sweep_virtual_transfer(
    source_model: str,
    target_model: str,
    dataset: str,
    bridge: Dict,
    device: str,
    task: str = "classification",
    sae_dir: Path = DEFAULT_SAE_DIR,
    layers_path: Path = DEFAULT_LAYERS_PATH,
    training_dir: Path = DEFAULT_TRAINING_DIR,
    emb_source: Optional[torch.Tensor] = None,
    emb_target: Optional[torch.Tensor] = None,
    source_preds: Optional[np.ndarray] = None,
    target_baseline_preds: Optional[np.ndarray] = None,
) -> Dict:
    """Cumulative sweep: transfer top-1, top-2, ..., top-N virtual concepts.

    Accumulates unmatched source concepts via virtual decoder atoms, tracking
    logloss at each k. Finds the optimal k where the target model's transferred
    logloss best matches the source model's logloss.

    Args:
        source_model: Source model key (e.g. "tabpfn").
        target_model: Target model key (e.g. "tabicl").
        dataset: TabArena dataset name.
        bridge: Dict from build_concept_bridge() with keys: virtual_atoms,
            unmatched_indices, concept_map_r2, n_matched_pairs.
        device: Torch device string.
        task: "classification" or "regression".
        sae_dir: Path to SAE checkpoints.
        layers_path: Path to optimal_extraction_layers.json.
        training_dir: Path to SAE training data.
        emb_source: Pre-captured source embeddings (optional).
        emb_target: Pre-captured target embeddings (optional).
        source_preds: Pre-captured source predictions (optional).
        target_baseline_preds: Pre-captured target baseline predictions (optional).

    Returns:
        Dict with: optimal_k, optimal_features, optimal_preds, logloss_curve,
        baseline_logloss, target_logloss, all_transferred_preds,
        source_preds_p1, target_baseline_p1, y_query, concept_map_r2,
        n_matched_pairs.
    """
    # 1. Load data splits
    X_ctx, y_ctx, X_q, y_q = _load_splits(dataset, task)
    n_query = len(X_q)

    # 2. Capture embeddings if not provided
    if emb_source is None or source_preds is None:
        source_layer = get_extraction_layer(source_model, layers_path)
        logger.info("Capturing %s embeddings (layer %d)...", source_model, source_layer)
        emb_source, source_preds = capture_embeddings(
            source_model, X_ctx, y_ctx, X_q, source_layer, device, task,
        )
    if emb_target is None or target_baseline_preds is None:
        target_layer = get_extraction_layer(target_model, layers_path)
        logger.info("Capturing %s embeddings (layer %d)...", target_model, target_layer)
        emb_target, target_baseline_preds = capture_embeddings(
            target_model, X_ctx, y_ctx, X_q, target_layer, device, task,
        )

    # 3. Load source SAE and compute data mean for centering
    sae_source, _ = load_sae(source_model, sae_dir=sae_dir, device=device)
    if source_model == "tabicl":
        data_mean = emb_source.mean(dim=0)
    else:
        data_mean = load_training_mean(
            source_model, training_dir=training_dir,
            layers_path=layers_path, device=device,
        )

    # 4. Encode all source positions and split into context / query
    acts_all = _encode_unmatched_activations(
        sae_source, emb_source, bridge["unmatched_indices"], data_mean,
    )
    acts_ctx = acts_all[:-n_query]
    acts_query = acts_all[-n_query:]

    # 5. Compute baselines
    sp1 = source_preds[:, 1] if source_preds.ndim == 2 else source_preds
    tp1 = target_baseline_preds[:, 1] if target_baseline_preds.ndim == 2 else target_baseline_preds
    target_logloss = _logloss(y_q.astype(float), sp1)
    baseline_logloss = _logloss(y_q.astype(float), tp1)

    logger.info(
        "Target (weaker) baseline logloss=%.4f, source logloss=%.4f",
        baseline_logloss, target_logloss,
    )

    # 6. Cumulative sweep k=1..N
    virtual_atoms = bridge["virtual_atoms"]
    n_unmatched = len(bridge["unmatched_indices"])
    all_transferred_p1 = []

    logger.info("Sweeping k=1..%d virtual concept transfer levels...", n_unmatched)
    for k in range(1, n_unmatched + 1):
        mask = np.zeros(n_unmatched, dtype=bool)
        mask[:k] = True

        delta_ctx = _make_virtual_delta(acts_ctx, virtual_atoms, feature_mask=mask)
        delta_query = _make_virtual_delta(acts_query, virtual_atoms, feature_mask=mask)
        full_delta = _build_full_delta_from_parts(delta_ctx, delta_query)

        result = intervene(
            model_key=target_model,
            X_context=X_ctx,
            y_context=y_ctx,
            X_query=X_q,
            y_query=y_q,
            external_delta=full_delta.to(device),
            device=device,
            task=task,
            layers_path=layers_path,
        )

        preds_k = result["ablated_preds"]
        pk1 = preds_k[:, 1] if preds_k.ndim == 2 else preds_k
        all_transferred_p1.append(pk1)

        ll = _logloss(y_q.astype(float), pk1)
        delta_ll = ll - baseline_logloss
        logger.info(
            "  k=%d (f%d): logloss=%.4f (delta=%+.4f from baseline)",
            k, bridge["unmatched_indices"][k - 1], ll, delta_ll,
        )

    # 7. Find optimal k
    logloss_curve = [_logloss(y_q.astype(float), p) for p in all_transferred_p1]
    gaps = [abs(ll - target_logloss) for ll in logloss_curve]
    optimal_k = int(np.argmin(gaps)) + 1

    logger.info(
        "Optimal k=%d (logloss=%.4f, gap to source=%.4f)",
        optimal_k,
        logloss_curve[optimal_k - 1],
        logloss_curve[optimal_k - 1] - target_logloss,
    )

    # 8. Return results
    return {
        "optimal_k": optimal_k,
        "optimal_features": bridge["unmatched_indices"][:optimal_k],
        "optimal_preds": all_transferred_p1[optimal_k - 1],
        "logloss_curve": logloss_curve,
        "baseline_logloss": baseline_logloss,
        "target_logloss": target_logloss,
        "all_transferred_preds": all_transferred_p1,
        "source_preds_p1": sp1,
        "target_baseline_p1": tp1,
        "y_query": y_q,
        "concept_map_r2": bridge["concept_map_r2"],
        "n_matched_pairs": bridge["n_matched_pairs"],
    }


# -- Per-row selective sweep --------------------------------------------------


def perrow_sweep_virtual_transfer(
    source_model: str,
    target_model: str,
    dataset: str,
    bridge: Dict,
    device: str,
    task: str = "classification",
    sae_dir: Path = DEFAULT_SAE_DIR,
    layers_path: Path = DEFAULT_LAYERS_PATH,
    training_dir: Path = DEFAULT_TRAINING_DIR,
    emb_source: Optional[torch.Tensor] = None,
    emb_target: Optional[torch.Tensor] = None,
    source_preds: Optional[np.ndarray] = None,
    target_baseline_preds: Optional[np.ndarray] = None,
) -> Dict:
    """Per-row selective sweep: heterogeneous virtual concept transfer.

    Three-phase pipeline:
      Phase 1: Per-feature importance (N forward passes) -- one per unmatched
               concept, measuring how each virtual node changes predictions.
      Phase 2: Per-row ranking -- sort features by per-row importance, filtered
               to features that fire in the source SAE for that row.
      Phase 3: Selective accept/reject (max_k forward passes) -- greedily add
               the next-ranked feature per row, keeping it only if the prediction
               moves closer to the source (strong) model.

    Args:
        source_model: Source (strong) model key.
        target_model: Target (weak) model key.
        dataset: TabArena dataset name.
        bridge: Dict from build_concept_bridge().
        device: Torch device string.
        task: "classification" or "regression".
        sae_dir: Path to SAE checkpoints.
        layers_path: Path to optimal_extraction_layers.json.
        training_dir: Path to SAE training data.
        emb_source: Pre-captured source embeddings (optional).
        emb_target: Pre-captured target embeddings (optional).
        source_preds: Pre-captured source predictions (optional).
        target_baseline_preds: Pre-captured target baseline predictions (optional).

    Returns:
        Dict with keys: baseline_preds, perrow_importance, perrow_rankings,
        sweep_preds, accepted_preds, accepted_counts, max_k_per_row,
        accepted_masks_np, query_activations_unmatched, unmatched_features,
        source_preds, y_query, concept_map_r2.
    """
    # 1. Load data splits
    X_ctx, y_ctx, X_q, y_q = _load_splits(dataset, task)
    n_query = len(X_q)

    # 2. Capture embeddings if not provided
    if emb_source is None or source_preds is None:
        source_layer = get_extraction_layer(source_model, layers_path)
        logger.info("Capturing %s embeddings (layer %d)...", source_model, source_layer)
        emb_source, source_preds = capture_embeddings(
            source_model, X_ctx, y_ctx, X_q, source_layer, device, task,
        )
    if emb_target is None or target_baseline_preds is None:
        target_layer = get_extraction_layer(target_model, layers_path)
        logger.info("Capturing %s embeddings (layer %d)...", target_model, target_layer)
        emb_target, target_baseline_preds = capture_embeddings(
            target_model, X_ctx, y_ctx, X_q, target_layer, device, task,
        )

    # 3. Load source SAE and compute data mean for centering
    sae_source, _ = load_sae(source_model, sae_dir=sae_dir, device=device)
    if source_model == "tabicl":
        data_mean = emb_source.mean(dim=0)
    else:
        data_mean = load_training_mean(
            source_model, training_dir=training_dir,
            layers_path=layers_path, device=device,
        )

    # 4. Encode source embeddings → extract unmatched activations
    unmatched_indices = bridge["unmatched_indices"]
    n_unmatched = len(unmatched_indices)
    acts_all = _encode_unmatched_activations(
        sae_source, emb_source, unmatched_indices, data_mean,
    )
    acts_ctx = acts_all[:-n_query]
    acts_query = acts_all[-n_query:]

    virtual_atoms = bridge["virtual_atoms"]
    baseline_np = target_baseline_preds

    # --- Phase 1: per-feature importance (N forward passes) ---
    logger.info("Phase 1: per-feature importance for %d features...", n_unmatched)
    individual_preds = []

    for feat_local in range(n_unmatched):
        mask = np.zeros(n_unmatched, dtype=bool)
        mask[feat_local] = True
        delta_ctx = _make_virtual_delta(acts_ctx, virtual_atoms, feature_mask=mask)
        delta_query = _make_virtual_delta(acts_query, virtual_atoms, feature_mask=mask)
        full_delta = _build_full_delta_from_parts(delta_ctx, delta_query)

        result = intervene(
            model_key=target_model,
            X_context=X_ctx, y_context=y_ctx,
            X_query=X_q, y_query=y_q,
            external_delta=full_delta.to(device),
            device=device, task=task,
            layers_path=layers_path,
        )
        individual_preds.append(result["ablated_preds"])

    # Compute importance and NEGATE: _perrow_importance measures logloss increase
    # from ablation (positive = feature helpful). For transfer, we want positive =
    # transfer helped (logloss decrease), so we negate.
    importance = _perrow_importance(baseline_np, individual_preds, y_q)
    importance = -importance

    # --- Phase 2: per-row ranking ---
    # Use LOCAL indices (0..n_unmatched-1) since acts_query is already sliced
    # to unmatched features. _perrow_rankings checks query_activations[row, feat_idx],
    # so local indices directly index into acts_query columns.
    feature_indices = list(range(n_unmatched))
    rankings = _perrow_rankings(importance, feature_indices, acts_query)
    max_k = max((len(r) for r in rankings), default=0)
    logger.info("Phase 2: rankings built. Max firing features/row: %d", max_k)

    # --- Phase 3: selective accept/reject (max_k forward passes) ---
    logger.info("Phase 3: selective sweep k=1..%d...", max_k)
    sp1 = source_preds[:, 1] if source_preds.ndim == 2 else source_preds
    bp1 = baseline_np[:, 1] if baseline_np.ndim == 2 else baseline_np
    best_dist = np.abs(bp1 - sp1)

    accepted_masks = np.zeros((n_query, n_unmatched), dtype=bool)
    sweep_preds = []

    for k in range(1, max_k + 1):
        # Build tentative masks: accepted + k-th ranked feature per row
        tentative_masks = accepted_masks.copy()
        for row_idx in range(n_query):
            if k - 1 < len(rankings[row_idx]):
                tentative_masks[row_idx, rankings[row_idx][k - 1]] = True

        # Query delta: per-row masked
        delta_query_t = torch.tensor(
            compute_virtual_delta_perrow(acts_query, virtual_atoms, tentative_masks),
            dtype=torch.float32,
        )

        # Context delta: union of all features tentatively active at level k
        ctx_mask = tentative_masks.any(axis=0)  # (n_unmatched,) bool
        delta_ctx = _make_virtual_delta(acts_ctx, virtual_atoms, feature_mask=ctx_mask)

        full_delta = _build_full_delta_from_parts(delta_ctx, delta_query_t)

        result = intervene(
            model_key=target_model,
            X_context=X_ctx, y_context=y_ctx,
            X_query=X_q, y_query=y_q,
            external_delta=full_delta.to(device),
            device=device, task=task,
            layers_path=layers_path,
        )
        preds_k = result["ablated_preds"]

        # Per-row accept/reject: keep feature only if prediction moves
        # closer to the strong model (no labels used in this decision)
        p1_k = preds_k[:, 1] if preds_k.ndim == 2 else preds_k
        dist_k = np.abs(p1_k - sp1)

        for row_idx in range(n_query):
            if dist_k[row_idx] < best_dist[row_idx] - 1e-8:
                accepted_masks[row_idx] = tentative_masks[row_idx]
                best_dist[row_idx] = dist_k[row_idx]

        sweep_preds.append(preds_k)

    n_accepted = accepted_masks.sum(axis=1)
    logger.info(
        "Selective sweep: mean %.1f, median %.0f, max %d concepts accepted/row",
        n_accepted.mean(), np.median(n_accepted), int(n_accepted.max()),
    )

    # Final forward pass with accepted masks
    delta_query_final = torch.tensor(
        compute_virtual_delta_perrow(acts_query, virtual_atoms, accepted_masks),
        dtype=torch.float32,
    )
    ctx_mask_final = accepted_masks.any(axis=0)
    delta_ctx_final = _make_virtual_delta(
        acts_ctx, virtual_atoms, feature_mask=ctx_mask_final,
    )
    full_delta_final = _build_full_delta_from_parts(delta_ctx_final, delta_query_final)

    result_final = intervene(
        model_key=target_model,
        X_context=X_ctx, y_context=y_ctx,
        X_query=X_q, y_query=y_q,
        external_delta=full_delta_final.to(device),
        device=device, task=task,
        layers_path=layers_path,
    )

    return {
        "baseline_preds": baseline_np,
        "perrow_importance": importance,
        "perrow_rankings": rankings,
        "sweep_preds": sweep_preds,
        "accepted_preds": result_final["ablated_preds"],
        "accepted_counts": n_accepted.astype(int),
        "max_k_per_row": np.array([len(r) for r in rankings]),
        "accepted_masks_np": accepted_masks,
        "query_activations_unmatched": acts_query,
        "unmatched_features": list(unmatched_indices),
        "source_preds": source_preds,
        "y_query": y_q,
        "concept_map_r2": bridge["concept_map_r2"],
    }


# -- Per-row optimal analysis -------------------------------------------------


def find_per_row_optimal_virtual(
    perrow_result: Dict,
    source_preds: np.ndarray,
    target_baseline_preds: np.ndarray,
    y_query: np.ndarray,
) -> Dict:
    """Compute per-row gap closed using selective virtual transfer results.

    With selective transfer, Phase 3 already determined the optimal set per row
    (accepted_counts). This function computes the gap-closed metric — how much
    of the logloss gap between the weak and strong models was closed by transfer.

    Args:
        perrow_result: Output dict from perrow_sweep_virtual_transfer().
        source_preds: Source (strong) model predictions.
        target_baseline_preds: Target (weak) model baseline predictions.
        y_query: Ground-truth labels for query rows.

    Returns:
        Dict with keys: optimal_k, row_gap_closed, perrow_rankings,
        perrow_importance, sweep_preds, baseline_preds, accepted_preds,
        max_k_per_row, target_row_ll, baseline_row_ll, unmatched_features.
    """
    eps = 1e-7
    y = y_query.astype(float)

    # Source (strong) model's per-row logloss — the goal
    sp1 = source_preds[:, 1] if source_preds.ndim == 2 else source_preds
    sp = np.clip(sp1, eps, 1 - eps)
    target_row_ll = -(y * np.log(sp) + (1 - y) * np.log(1 - sp))

    # Target (weak) model's per-row logloss — starting point
    bp1 = target_baseline_preds[:, 1] if target_baseline_preds.ndim == 2 else target_baseline_preds
    bp = np.clip(bp1, eps, 1 - eps)
    baseline_row_ll = -(y * np.log(bp) + (1 - y) * np.log(1 - bp))

    # Accepted predictions — final state after selective accept/reject
    ap = perrow_result["accepted_preds"]
    ap1 = ap[:, 1] if ap.ndim == 2 else ap
    ap_clipped = np.clip(ap1, eps, 1 - eps)
    accepted_row_ll = -(y * np.log(ap_clipped) + (1 - y) * np.log(1 - ap_clipped))

    # Optimal k = number of concepts accepted per row
    raw_counts = perrow_result["accepted_counts"].astype(int)
    n_query = len(y_query)
    optimal_k = np.zeros_like(raw_counts)
    row_gap_closed = np.zeros(n_query)

    for row_idx in range(n_query):
        orig_gap = baseline_row_ll[row_idx] - target_row_ll[row_idx]
        if orig_gap <= 0:
            # Weak model already equal or better — no transfer needed
            optimal_k[row_idx] = 0
            row_gap_closed[row_idx] = 1.0
        else:
            optimal_k[row_idx] = raw_counts[row_idx]
            gap_remaining = abs(accepted_row_ll[row_idx] - target_row_ll[row_idx])
            row_gap_closed[row_idx] = 1.0 - gap_remaining / orig_gap

    logger.info(
        "Per-row accepted concepts: mean=%.1f, median=%d, max=%d",
        optimal_k.mean(), int(np.median(optimal_k)), optimal_k.max(),
    )
    logger.info(
        "Rows with 0 concepts accepted: %d (%.1f%%)",
        int((optimal_k == 0).sum()), 100 * (optimal_k == 0).mean(),
    )

    return {
        "optimal_k": optimal_k,
        "row_gap_closed": row_gap_closed,
        "perrow_rankings": perrow_result["perrow_rankings"],
        "perrow_importance": perrow_result["perrow_importance"],
        "sweep_preds": perrow_result["sweep_preds"],
        "baseline_preds": perrow_result["baseline_preds"],
        "accepted_preds": perrow_result["accepted_preds"],
        "max_k_per_row": perrow_result["max_k_per_row"],
        "target_row_ll": target_row_ll,
        "baseline_row_ll": baseline_row_ll,
        "unmatched_features": perrow_result["unmatched_features"],
    }


# -- CLI -----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Virtual-node concept transfer between tabular foundation models",
    )
    parser.add_argument("--source", type=str, required=True,
                        help="Model A key (auto-detects which is stronger)")
    parser.add_argument("--target", type=str, required=True,
                        help="Model B key (auto-detects which is weaker)")
    parser.add_argument("--dataset", type=str, default="credit-g")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--task", type=str, default="classification")
    parser.add_argument("--min-match-r", type=float, default=0.2,
                        help="Minimum |r| for MNN anchor pairs")
    parser.add_argument("--ridge-alpha", type=float, default=1.0,
                        help="Ridge regularization for concept-level map")
    parser.add_argument("--perrow", action="store_true",
                        help="Per-row selective transfer: find minimal concept "
                             "set per row that closes the gap to the strong model")
    parser.add_argument("--sae-dir", type=Path, default=DEFAULT_SAE_DIR)
    parser.add_argument("--layers-config", type=Path, default=DEFAULT_LAYERS_PATH)
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--cross-corr-dir", type=Path, default=DEFAULT_CROSS_CORR_DIR)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "output" / "figures")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    fig_dir = args.output_dir / args.dataset

    # Auto-detect stronger model by running both and comparing AUC.
    # Transfer FROM the stronger model TO the weaker one.
    from sklearn.metrics import roc_auc_score

    model_a, model_b = args.source, args.target

    logger.info("Getting baseline predictions to determine transfer direction...")
    X_ctx, y_ctx, X_q, y_q = _load_splits(args.dataset, args.task)
    layer_a = get_extraction_layer(model_a, args.layers_config)
    layer_b = get_extraction_layer(model_b, args.layers_config)

    emb_a, preds_a = capture_embeddings(
        model_a, X_ctx, y_ctx, X_q, layer_a, args.device, args.task,
    )
    emb_b, preds_b = capture_embeddings(
        model_b, X_ctx, y_ctx, X_q, layer_b, args.device, args.task,
    )

    pa1 = preds_a[:, 1] if preds_a.ndim == 2 else preds_a
    pb1 = preds_b[:, 1] if preds_b.ndim == 2 else preds_b
    auc_a = float(roc_auc_score(y_q, pa1))
    auc_b = float(roc_auc_score(y_q, pb1))

    if auc_a >= auc_b:
        source_model, target_model = model_a, model_b
        emb_source, emb_target = emb_a, emb_b
        source_preds, target_preds = preds_a, preds_b
    else:
        source_model, target_model = model_b, model_a
        emb_source, emb_target = emb_b, emb_a
        source_preds, target_preds = preds_b, preds_a

    disp_s = DISPLAY_NAMES.get(source_model, source_model)
    disp_t = DISPLAY_NAMES.get(target_model, target_model)
    logger.info(
        "Transfer direction: %s (AUC=%.3f) -> %s (AUC=%.3f)",
        disp_s, max(auc_a, auc_b), disp_t, min(auc_a, auc_b),
    )

    # Get unmatched features from the stronger model (ranked by importance)
    try:
        unmatched = get_unmatched_features(
            source_model, target_model, args.dataset, positive_only=False,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No comparison data for {source_model} vs {target_model} on {args.dataset}. "
            f"Run: python scripts/concept_importance.py --model {source_model} "
            f"--compare {target_model} --dataset {args.dataset}"
        )

    if not unmatched:
        logger.warning("No unmatched features found.")
        return

    unmatched_feat_indices = [f for f, _ in unmatched]
    n_pos = sum(1 for _, d in unmatched if d > 0)
    logger.info(
        "Found %d unmatched %s-only features (%d with positive drop)",
        len(unmatched), source_model, n_pos,
    )
    for feat, drop in unmatched[:10]:
        logger.info("  feature %d: drop=%.4f", feat, drop)

    # Build concept bridge
    sae_source, _ = load_sae(source_model, sae_dir=args.sae_dir, device=args.device)
    sae_target, _ = load_sae(target_model, sae_dir=args.sae_dir, device=args.device)

    corr_matrix, indices_a, indices_b = load_cross_correlations(
        source_model, target_model, cross_corr_dir=args.cross_corr_dir,
    )

    bridge = build_concept_bridge(
        sae_source, sae_target, corr_matrix, indices_a, indices_b,
        unmatched_feat_indices,
        min_match_r=args.min_match_r,
        ridge_alpha=args.ridge_alpha,
    )

    logger.info(
        "Concept bridge: %d MNN pairs, map R²=%.4f, %d virtual atoms",
        bridge["n_matched_pairs"],
        bridge["concept_map_r2"],
        len(bridge["unmatched_indices"]),
    )

    # ── Per-row transfer mode ─────────────────────────────────────────────
    if args.perrow:
        perrow_result = perrow_sweep_virtual_transfer(
            source_model=source_model,
            target_model=target_model,
            dataset=args.dataset,
            bridge=bridge,
            device=args.device,
            task=args.task,
            sae_dir=args.sae_dir,
            layers_path=args.layers_config,
            training_dir=args.training_dir,
            emb_source=emb_source,
            emb_target=emb_target,
            source_preds=source_preds,
            target_baseline_preds=target_preds,
        )

        sp1_pr = source_preds[:, 1] if source_preds.ndim == 2 else source_preds
        tp1_pr = target_preds[:, 1] if target_preds.ndim == 2 else target_preds

        optimal = find_per_row_optimal_virtual(
            perrow_result, source_preds, target_preds, y_q,
        )

        # Use accepted predictions for fixable rows, baseline for non-fixable
        ap = optimal["accepted_preds"]
        ap1 = ap[:, 1] if ap.ndim == 2 else ap
        transferred_p1 = np.where(optimal["optimal_k"] > 0, ap1, tp1_pr)

        # Histogram + coverage curve
        plot_perrow_results(
            optimal["optimal_k"],
            optimal["row_gap_closed"],
            optimal["max_k_per_row"],
            optimal["perrow_rankings"],
            source_model,  # concept owner
            target_model,  # model being improved
            args.dataset,
            fig_dir / "vnode_transfer_perrow.pdf",
            action="transferring",
        )

        # Per-row scatter
        auc_s = float(roc_auc_score(y_q, sp1_pr))
        auc_t = float(roc_auc_score(y_q, tp1_pr))
        auc_transferred = float(roc_auc_score(y_q, transferred_p1))
        gap = auc_s - auc_t
        gap_closed_pct = (
            (auc_transferred - auc_t) / gap * 100
            if abs(gap) > 0.001
            else float("nan")
        )
        logger.info(
            "Per-row transfer AUC: %.4f (baseline %.4f -> target %.4f, "
            "gap closed %.1f%%)",
            auc_transferred, auc_t, auc_s, gap_closed_pct,
        )

        plot_perrow_scatter(
            sp1_pr, tp1_pr, transferred_p1,
            optimal["optimal_k"], y_q,
            source_model, target_model,
            args.dataset,
            auc_s, auc_t,
            fig_dir / "vnode_transfer_perrow_scatter.pdf",
            ablate_axis="y",
            action="transferring",
        )

        # Diagnostic: logloss distributions + concept budget + accept rate
        plot_perrow_diagnostic(
            optimal["optimal_k"],
            optimal["row_gap_closed"],
            optimal["accepted_preds"],
            optimal["baseline_preds"],
            source_preds,
            optimal["max_k_per_row"],
            y_q,
            source_model,  # concept owner
            target_model,  # model being improved
            args.dataset,
            fig_dir / "vnode_transfer_perrow_diagnostic.pdf",
            action="transferring",
        )

    # ── Cumulative sweep (always runs) ────────────────────────────────────
    sweep = sweep_virtual_transfer(
        source_model=source_model,
        target_model=target_model,
        dataset=args.dataset,
        bridge=bridge,
        device=args.device,
        task=args.task,
        sae_dir=args.sae_dir,
        layers_path=args.layers_config,
        training_dir=args.training_dir,
        emb_source=emb_source,
        emb_target=emb_target,
        source_preds=source_preds,
        target_baseline_preds=target_preds,
    )

    k = sweep["optimal_k"]
    y = sweep["y_query"]
    sp1 = sweep["source_preds_p1"]
    tp1 = sweep["target_baseline_p1"]
    xp1 = sweep["optimal_preds"]

    from sklearn.metrics import roc_auc_score as _roc_auc_score

    auc_source = float(_roc_auc_score(y, sp1))
    auc_target = float(_roc_auc_score(y, tp1))
    auc_transferred = float(_roc_auc_score(y, xp1))

    gap = auc_source - auc_target
    gap_closed = (auc_transferred - auc_target) / gap if gap > 0.001 else float("nan")

    print(f"\n{'='*60}")
    print(f"Virtual-node concept transfer: {source_model} -> {target_model}")
    print(f"Dataset: {args.dataset}")
    print(f"Concept bridge: {bridge['n_matched_pairs']} MNN pairs, "
          f"map R²={bridge['concept_map_r2']:.4f}")
    print(f"Optimal k = {k}/{len(unmatched)} concepts")
    print(f"Optimal features: {sweep['optimal_features']}")
    print(f"\n  {source_model} AUC = {auc_source:.4f}  "
          f"(logloss={sweep['target_logloss']:.4f})")
    print(f"  {target_model} AUC = {auc_target:.4f}  "
          f"(logloss={sweep['baseline_logloss']:.4f})")
    print(f"  {target_model}+vnode AUC = {auc_transferred:.4f}  "
          f"(logloss={sweep['logloss_curve'][k-1]:.4f})")
    if gap > 0.001:
        print(f"\n  Gap closed: {gap_closed:.1%}")
    print(f"{'='*60}")

    # Logloss curve
    plot_logloss_curve(
        sweep["logloss_curve"],
        sweep["baseline_logloss"],
        sweep["target_logloss"],
        k,
        unmatched,
        target_model,   # model being modified
        source_model,   # reference model (whose concepts we're transferring)
        args.dataset,
        fig_dir / "vnode_transfer_logloss.pdf",
        action="transferring",
    )

    # Scatter plot with optimal transfer overlay
    features_s = get_active_feature_count(source_model, args.dataset)
    features_t = get_active_feature_count(target_model, args.dataset)

    transfer_label = f"vnode {k}/{len(unmatched)} {disp_s}-only"
    ablation_levels = [(transfer_label, "#009E73", xp1)]

    plot_prediction_scatter(
        sp1, tp1, y,
        source_model, target_model, args.dataset,
        auc_source, auc_target, features_s, features_t,
        fig_dir / "vnode_transfer_scatter.pdf",
        ablation_levels=ablation_levels,
        ablate_axis="y",
    )


if __name__ == "__main__":
    main()
