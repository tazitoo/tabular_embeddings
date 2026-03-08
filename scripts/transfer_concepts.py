#!/usr/bin/env python3
"""Per-concept vec2vec transfer between tabular foundation models.

The constructive complement to ablation: instead of removing the strong model's
unique concepts to degrade it, we transfer them to the weak model to improve it.
Accumulates concepts one at a time (like ablation), tracking logloss against
the strong model's logloss as the target.

Approach:
    1. Forward pass both models → capture embeddings at optimal layers
    2. Fit linear map: W = ridge(emb_source → emb_target)
    3. Accumulate source concepts k=1..N, compute delta in target space
    4. Inject translated delta into target model → get transferred predictions
    5. Track logloss curve, find optimal k

Usage:
    python scripts/transfer_concepts.py --source tabpfn --target tabicl \
        --dataset kddcup09_appetency --device cuda
"""

import argparse
import logging
import sys
from collections import defaultdict
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
    build_tail,
    get_extraction_layer,
    intervene,
    load_sae,
    load_training_mean,
)
from scripts.concept_performance_diagnostic import _load_splits, DISPLAY_NAMES
from scripts.plot_prediction_scatter import (
    _logloss,
    get_unmatched_features,
)

logger = logging.getLogger(__name__)


# ── Linear Map ────────────────────────────────────────────────────────────────


def fit_linear_map(
    emb_source: np.ndarray,
    emb_target: np.ndarray,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit ridge regression: emb_target ≈ emb_source @ W.T + b.

    Returns:
        W: (d_target, d_source) weight matrix
        b: (d_target,) bias vector
        r2: Coefficient of determination on training data
    """
    from sklearn.linear_model import Ridge

    reg = Ridge(alpha=alpha, fit_intercept=True)
    reg.fit(emb_source, emb_target)

    r2 = float(reg.score(emb_source, emb_target))
    W = reg.coef_  # (d_target, d_source)
    b = reg.intercept_  # (d_target,)

    return W, b, r2


# ── Transfer Delta ────────────────────────────────────────────────────────────


def _get_d_target(W: Optional[np.ndarray], translator: Optional[torch.nn.Module]) -> int:
    """Get target embedding dimension from linear map or translator."""
    if translator is not None:
        # Walk the translator's layers to find output dim
        net = translator.net
        if isinstance(net, torch.nn.Sequential):
            return net[-1].out_features
        return net.out_features
    return W.shape[0]


def _map_to_target_space(
    contribution_source: torch.Tensor,
    emb_source_raw: torch.Tensor,
    W: Optional[np.ndarray],
    translator: Optional[torch.nn.Module],
    scale: float,
) -> torch.Tensor:
    """Map source-space contribution to target space via linear map or MLP translator.

    For linear map: delta = contribution @ W.T (exact, bias cancels).
    For MLP translator: delta = translator(emb + contribution) - translator(emb)
        (finite difference captures nonlinear structure).
    """
    if translator is not None:
        # MLP: finite difference preserves nonlinear structure
        target_base = translator(emb_source_raw)
        target_perturbed = translator(emb_source_raw + contribution_source)
        return (target_perturbed - target_base) * scale
    else:
        W_t = torch.tensor(W, dtype=contribution_source.dtype,
                           device=contribution_source.device)
        return contribution_source @ W_t.T * scale


def compute_transfer_delta(
    sae_source: torch.nn.Module,
    emb_source: torch.Tensor,
    W: Optional[np.ndarray],
    b: Optional[np.ndarray],
    transfer_features: List[int],
    data_mean: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    translator: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """Compute delta in target space for given source concepts.

    1. SAE encode source embeddings → h_source (sparse activations)
    2. Concept contribution: decode(h_selected) - decode(0)  (bias cancels)
    3. Map to target space: contribution @ W.T  (or MLP translator)

    Args:
        translator: Optional EmbeddingTranslator. If provided, uses MLP-based
            delta mapping instead of linear W @ contribution.

    Returns:
        delta_target: (n_samples, d_target) delta to inject into target model
    """
    if not transfer_features or scale == 0.0:
        d_target = _get_d_target(W, translator)
        return torch.zeros(emb_source.shape[0], d_target, device=emb_source.device)

    with torch.no_grad():
        x = emb_source
        if data_mean is not None:
            x = x - data_mean

        h = sae_source.encode(x)

        # Concept contribution in source space: decode(h_selected) - decode(0).
        # Subtraction cancels decoder bias, analogous to ablation delta.
        h_with = torch.zeros_like(h)
        h_with[:, transfer_features] = h[:, transfer_features]
        contribution_source = sae_source.decode(h_with) - sae_source.decode(
            torch.zeros_like(h)
        )  # (n_samples, d_source)

        delta_target = _map_to_target_space(
            contribution_source, emb_source, W, translator, scale,
        )

    return delta_target


def compute_transfer_delta_perrow(
    sae_source: torch.nn.Module,
    emb_source: torch.Tensor,
    W: Optional[np.ndarray],
    b: Optional[np.ndarray],
    feature_masks: torch.Tensor,
    data_mean: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    translator: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """Per-row transfer delta with different features per row.

    Analogous to compute_ablation_delta_perrow but maps through linear map
    (or MLP translator) to target space.

    Args:
        sae_source: Source SAE in eval mode
        emb_source: (n_rows, d_source_emb) raw source embeddings
        W: (d_target, d_source_emb) linear map weight matrix (or None if using translator)
        b: (d_target,) bias (unused — cancels in delta)
        feature_masks: (n_rows, sae_hidden) boolean, True = TRANSFER this feature
        data_mean: (d_source_emb,) centering mean for SAE
        scale: Multiplicative scaling factor
        translator: Optional EmbeddingTranslator for MLP-based delta mapping

    Returns:
        delta_target: (n_rows, d_target) per-row deltas in target space
    """
    d_target = (translator.net[-1].out_features if translator is not None
                else W.shape[0])
    if scale == 0.0:
        return torch.zeros(emb_source.shape[0], d_target, device=emb_source.device)

    with torch.no_grad():
        x = emb_source
        if data_mean is not None:
            x = x - data_mean

        h = sae_source.encode(x)

        # Per-row masking: keep only features marked True
        h_masked = torch.zeros_like(h)
        h_masked[feature_masks] = h[feature_masks]

        # Concept contribution in source space (bias cancels)
        contribution_source = sae_source.decode(h_masked) - sae_source.decode(
            torch.zeros_like(h)
        )

        delta_target = _map_to_target_space(
            contribution_source, emb_source, W, translator, scale,
        )

    return delta_target


# ── Embedding Capture ─────────────────────────────────────────────────────────


def capture_embeddings(
    model_key: str,
    X_ctx: np.ndarray,
    y_ctx: np.ndarray,
    X_query: np.ndarray,
    extraction_layer: int,
    device: str,
    task: str,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Single forward pass with capture hook → (all_emb, preds).

    Returns embeddings for ALL positions (context + query), mean-pooled
    over structure dimensions as done in intervene_sae.py.

    Returns:
        all_emb: (seq_len, d_model) tensor on device
        preds: (n_query, n_classes) numpy array
    """
    # Deterministic forward passes (TabPFN resamples context internally)
    torch.manual_seed(42)
    np.random.seed(42)

    if model_key == "tabpfn":
        return _capture_tabpfn(X_ctx, y_ctx, X_query, extraction_layer, device, task)
    elif model_key == "tabicl":
        return _capture_tabicl(X_ctx, y_ctx, X_query, extraction_layer, device)
    else:
        raise ValueError(f"capture_embeddings not implemented for {model_key}")


def _capture_tabpfn(X_ctx, y_ctx, X_query, extraction_layer, device, task):
    from models.tabpfn_utils import load_tabpfn

    clf = load_tabpfn(task=task, device=device, n_estimators=1)
    clf.fit(X_ctx, y_ctx)
    layers = clf.model_.transformer_encoder.layers

    captured = {}

    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured["hidden"] = output.detach()

    handle = layers[extraction_layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            if task == "regression":
                preds = clf.predict(X_query)
            else:
                preds = clf.predict_proba(X_query)
    finally:
        handle.remove()

    # (1, seq_len, n_structure, hidden) → mean over structure
    all_emb = captured["hidden"][0].mean(dim=1)  # (seq_len, hidden)
    return all_emb, np.asarray(preds)


def _capture_tabicl(X_ctx, y_ctx, X_query, extraction_layer, device):
    from tabicl import TabICLClassifier

    clf = TabICLClassifier(device=device, n_estimators=1)
    clf.fit(X_ctx, y_ctx)
    blocks = clf.model_.icl_predictor.tf_icl.blocks

    captured = {}

    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured["hidden"] = output.detach()

    handle = blocks[extraction_layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            preds = clf.predict_proba(X_query)
    finally:
        handle.remove()

    # (n_ensemble, seq_len, 512) → mean over ensemble
    all_emb = captured["hidden"].mean(dim=0)  # (seq_len, 512)
    return all_emb, np.asarray(preds)


# ── Delta Construction ────────────────────────────────────────────────────────


def _build_full_delta(
    delta_query: torch.Tensor,
    n_total_target: int,
    n_query: int,
) -> torch.Tensor:
    """Build full delta for all target positions.

    Context: mean delta (uniform concept signal so ICL can learn from it).
    Query: per-row delta (row-specific concept magnitudes for prediction).
    """
    n_ctx_target = n_total_target - n_query
    mean_delta = delta_query.mean(dim=0, keepdim=True)  # (1, d_target)
    ctx_delta = mean_delta.expand(n_ctx_target, -1)  # (n_ctx, d_target)
    return torch.cat([ctx_delta, delta_query], dim=0)  # (n_total, d_target)


# ── Cumulative Sweep ──────────────────────────────────────────────────────────


def sweep_transfer(
    source_model: str,
    target_model: str,
    dataset: str,
    ranked_features: List[Tuple[int, float]],
    device: str,
    task: str = "classification",
    alpha: float = 1.0,
    sae_dir: Path = DEFAULT_SAE_DIR,
    layers_path: Path = DEFAULT_LAYERS_PATH,
    training_dir: Path = DEFAULT_TRAINING_DIR,
    emb_source: Optional[torch.Tensor] = None,
    emb_target: Optional[torch.Tensor] = None,
    source_preds: Optional[np.ndarray] = None,
    target_baseline_preds: Optional[np.ndarray] = None,
    translator: Optional[torch.nn.Module] = None,
) -> Dict:
    """Cumulative sweep: transfer top-1, top-2, ..., top-N concepts.

    Mirrors find_optimal_ablation: accumulates concepts by importance rank,
    tracks logloss at each k, finds optimal k where target model's logloss
    best matches the source model's logloss.

    Args:
        ranked_features: list of (feature_index, drop) sorted by importance
        emb_source, emb_target: Pre-captured embeddings (optional, avoids redundant passes)
        source_preds, target_baseline_preds: Pre-captured predictions (optional)

    Returns dict with:
        optimal_k, optimal_features, optimal_preds, logloss_curve,
        baseline_logloss, target_logloss, all_transferred_preds,
        source_preds, target_baseline_preds, y_query, linear_map_r2
    """
    X_ctx, y_ctx, X_q, y_q = _load_splits(dataset, task)
    n_query = len(X_q)

    # Capture embeddings if not provided
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

    # Fit linear map on query embeddings only (models may pad context differently)
    query_source = emb_source[-n_query:]
    query_target = emb_target[-n_query:]
    n_total_target = emb_target.shape[0]

    W, b, r2 = None, None, None
    if translator is not None:
        logger.info("Using universal translator (skipping per-dataset linear map)")
        r2 = 0.0  # placeholder — translator has its own val R²
    else:
        logger.info("Fitting linear map on query embeddings (n=%d, alpha=%.1f)...",
                    n_query, alpha)
        W, b, r2 = fit_linear_map(
            query_source.cpu().numpy(), query_target.cpu().numpy(), alpha=alpha,
        )
        logger.info("Linear map R² = %.4f (shape: %s)", r2, W.shape)

    # Load source SAE once
    sae_source, _ = load_sae(source_model, sae_dir=sae_dir, device=device)
    if source_model == "tabicl":
        data_mean = emb_source.mean(dim=0)
    else:
        data_mean = load_training_mean(
            source_model, training_dir=training_dir,
            layers_path=layers_path, device=device,
        )

    # Compute baselines
    y = y_q.astype(float)
    sp1 = source_preds[:, 1] if source_preds.ndim == 2 else source_preds
    tp1 = target_baseline_preds[:, 1] if target_baseline_preds.ndim == 2 else target_baseline_preds
    target_logloss = _logloss(y, sp1)  # what we're trying to match (source's logloss)
    baseline_logloss = _logloss(y, tp1)  # target model's starting logloss

    logger.info("Target (weaker) baseline logloss=%.4f, source logloss=%.4f",
                baseline_logloss, target_logloss)

    # Cumulative sweep: top-1, top-2, ..., top-N
    feat_indices = [f for f, _ in ranked_features]
    all_transferred_p1 = []

    logger.info("Sweeping k=1..%d transfer levels...", len(feat_indices))
    for k in range(1, len(feat_indices) + 1):
        features_k = feat_indices[:k]

        delta_query = compute_transfer_delta(
            sae_source, query_source, W, b,
            features_k, data_mean=data_mean, scale=1.0,
            translator=translator,
        )
        full_delta = _build_full_delta(delta_query, n_total_target, n_query)

        result = intervene(
            model_key=target_model,
            X_context=X_ctx, y_context=y_ctx,
            X_query=X_q, y_query=y_q,
            external_delta=full_delta.to(device),
            device=device, task=task,
            layers_path=layers_path,
        )

        preds_k = result["ablated_preds"]
        pk1 = preds_k[:, 1] if preds_k.ndim == 2 else preds_k
        all_transferred_p1.append(pk1)

        ll = _logloss(y, pk1)
        delta_ll = ll - baseline_logloss
        logger.info("  k=%d (f%d): logloss=%.4f (Δ=%+.4f from baseline)",
                     k, feat_indices[k - 1], ll, delta_ll)

    # Logloss at each k
    logloss_curve = [_logloss(y, p) for p in all_transferred_p1]

    # Find k where transferred logloss best matches source (target_logloss)
    gaps = [abs(ll - target_logloss) for ll in logloss_curve]
    optimal_k = int(np.argmin(gaps)) + 1

    logger.info("Optimal k=%d (logloss=%.4f, gap to source=%.4f)",
                optimal_k, logloss_curve[optimal_k - 1],
                logloss_curve[optimal_k - 1] - target_logloss)

    return {
        "optimal_k": optimal_k,
        "optimal_features": feat_indices[:optimal_k],
        "optimal_preds": all_transferred_p1[optimal_k - 1],
        "logloss_curve": logloss_curve,
        "baseline_logloss": baseline_logloss,
        "target_logloss": target_logloss,
        "all_transferred_preds": all_transferred_p1,
        "source_preds_p1": sp1,
        "target_baseline_p1": tp1,
        "y_query": y_q,
        "linear_map_r2": r2,
    }


# ── Per-Row Sweep ─────────────────────────────────────────────────────────────


def perrow_sweep_transfer(
    source_model: str,
    target_model: str,
    dataset: str,
    ranked_features: List[Tuple[int, float]],
    device: str,
    task: str = "classification",
    alpha: float = 1.0,
    sae_dir: Path = DEFAULT_SAE_DIR,
    layers_path: Path = DEFAULT_LAYERS_PATH,
    training_dir: Path = DEFAULT_TRAINING_DIR,
    emb_source: Optional[torch.Tensor] = None,
    emb_target: Optional[torch.Tensor] = None,
    source_preds: Optional[np.ndarray] = None,
    target_baseline_preds: Optional[np.ndarray] = None,
    translator: Optional[torch.nn.Module] = None,
) -> Dict:
    """Per-row transfer with backtracking line search (3-phase pipeline).

    Phase 1: Per-feature importance (N forward passes through target model)
    Phase 2: Per-row ranking (filtered to firing features in source SAE)
    Phase 3: Per-row backtracking line search (analytical scale + halving)
        - Probe at scale=1 to estimate gradient direction
        - Compute analytical optimal scale per row (negative = sign flip)
        - Backtrack by halving until improvement confirmed

    Returns dict with same shape as perrow_sweep_intervene plus linear map info.
    """
    X_ctx, y_ctx, X_q, y_q = _load_splits(dataset, task)
    n_query = len(X_q)

    # Get extraction layers (needed for tail model and possibly embedding capture)
    source_layer = get_extraction_layer(source_model, layers_path)
    target_layer = get_extraction_layer(target_model, layers_path)

    # Capture embeddings if not provided
    if emb_source is None or source_preds is None:
        logger.info("Capturing %s embeddings (layer %d)...", source_model, source_layer)
        emb_source, source_preds = capture_embeddings(
            source_model, X_ctx, y_ctx, X_q, source_layer, device, task,
        )
    if emb_target is None or target_baseline_preds is None:
        logger.info("Capturing %s embeddings (layer %d)...", target_model, target_layer)
        emb_target, target_baseline_preds = capture_embeddings(
            target_model, X_ctx, y_ctx, X_q, target_layer, device, task,
        )

    # Fit linear map on query embeddings
    query_source = emb_source[-n_query:]
    query_target = emb_target[-n_query:]
    n_total_target = emb_target.shape[0]

    W, b, r2 = None, None, None
    if translator is not None:
        logger.info("Using universal translator (skipping per-dataset linear map)")
        r2 = 0.0
    else:
        logger.info("Fitting linear map on query embeddings (n=%d, alpha=%.1f)...",
                    n_query, alpha)
        W, b, r2 = fit_linear_map(
            query_source.cpu().numpy(), query_target.cpu().numpy(), alpha=alpha,
        )
        logger.info("Linear map R² = %.4f", r2)

    # Load source SAE
    sae_source, _ = load_sae(source_model, sae_dir=sae_dir, device=device)
    if source_model == "tabicl":
        data_mean = emb_source.mean(dim=0)
    else:
        data_mean = load_training_mean(
            source_model, training_dir=training_dir,
            layers_path=layers_path, device=device,
        )

    feat_indices = [f for f, _ in ranked_features]

    # Build tail model once (used for Phase 1 importance probing and Phase 3 line search)
    logger.info("Building %s tail model (layer %d)...", target_model, target_layer)
    tail = build_tail(target_model, X_ctx, y_ctx, X_q, target_layer, task, device)
    seq_len = (tail.hidden_state.shape[1] if target_model == "tabpfn"
               else tail.hidden_state.shape[1])

    # --- Phase 1: per-feature importance via tail model ---
    logger.info("Phase 1: per-feature importance for %d features...", len(feat_indices))
    baseline_np = target_baseline_preds
    individual_preds = []

    for feat_idx in feat_indices:
        delta_query = compute_transfer_delta(
            sae_source, query_source, W, b,
            [feat_idx], data_mean=data_mean, scale=1.0,
            translator=translator,
        )
        full_delta = _build_full_delta(delta_query, seq_len, n_query)
        preds = tail.predict(full_delta.to(device))
        individual_preds.append(preds)

    # Compute importance and NEGATE (transfer helping = logloss decrease = negative)
    importance = _perrow_importance(baseline_np, individual_preds, y_q)
    importance = -importance  # Negate: positive importance = transfer helped

    # --- Phase 2: per-row ranking (filtered to source SAE firing features) ---
    with torch.no_grad():
        x_centered = query_source - data_mean if data_mean is not None else query_source
        h_encoded = sae_source.encode(x_centered)
    query_acts = h_encoded.cpu().numpy()

    rankings = _perrow_rankings(importance, feat_indices, query_acts)
    max_k = max((len(r) for r in rankings), default=0)
    logger.info("Phase 2: rankings built. Max firing features/row: %d", max_k)

    # --- Phase 3: true per-row line search using tail model ---
    # For each row independently:
    #   Walk through its ranked concepts, for each:
    #   1. Probe at scale=1 → estimate gradient
    #   2. Analytical optimal scale s* = (target - current) / gradient
    #   3. Backtrack by halving s* until improvement confirmed
    #   4. Accept or reject; move to next concept
    sae_hidden = h_encoded.shape[1]
    d_target = query_target.shape[1]
    max_backtrack = 6  # halve up to 6 times (minimum step = s*/64)

    # Source (strong) P(class=1) — the goal for each row
    sp1 = source_preds[:, 1] if source_preds.ndim == 2 else source_preds
    bp1 = baseline_np[:, 1] if baseline_np.ndim == 2 else baseline_np

    # Pre-filter: skip rows where weak model already outperforms strong
    eps = 1e-7
    y = y_q.astype(float)
    sp_clip = np.clip(sp1, eps, 1 - eps)
    bp_clip = np.clip(bp1, eps, 1 - eps)
    source_row_ll = -(y * np.log(sp_clip) + (1 - y) * np.log(1 - sp_clip))
    baseline_row_ll = -(y * np.log(bp_clip) + (1 - y) * np.log(1 - bp_clip))
    fixable = baseline_row_ll > source_row_ll  # weak is worse → transfer can help
    n_fixable = fixable.sum()
    logger.info("Fixable rows: %d / %d (%.0f%% already equal or better)",
                n_fixable, n_query, 100 * (1 - n_fixable / n_query))

    # Precompute per-concept deltas for all query rows
    # (fast: SAE decode + linear map, no model forward passes)
    logger.info("Precomputing per-concept deltas for %d features...",
                len(feat_indices))
    concept_deltas = {}
    for feat_idx in feat_indices:
        delta = compute_transfer_delta(
            sae_source, query_source, W, b,
            [feat_idx], data_mean=data_mean, scale=1.0,
            translator=translator,
        )
        concept_deltas[feat_idx] = delta  # (n_query, d_target) tensor

    # Per-row line search state
    cumulative_deltas = torch.zeros(n_query, d_target,
                                    dtype=query_source.dtype,
                                    device=query_source.device)
    accepted_counts = np.zeros(n_query, dtype=int)
    final_p1 = bp1.copy()
    converge_threshold = 1e-3  # stop adding concepts once this close to target

    # Per-concept tracking: acceptance history and scales
    concept_stats = defaultdict(lambda: {"tried": 0, "accepted": 0, "scales": []})
    min_tries_before_skip = 10  # need this many trials before skipping dead concepts

    logger.info("Per-row line search: %d fixable rows, max %d concepts/row...",
                n_fixable, max_k)

    for row_idx in range(n_query):
        if not fixable[row_idx]:
            continue  # weak model already equal or better

        current_p1 = float(bp1[row_idx])
        target_p1 = float(sp1[row_idx])
        best_dist = abs(current_p1 - target_p1)

        if best_dist < 1e-8:
            final_p1[row_idx] = current_p1
            continue

        for feat_idx in rankings[row_idx]:
            cs = concept_stats[feat_idx]

            # Skip concepts that have been tried enough and never accepted
            if (cs["tried"] >= min_tries_before_skip
                    and cs["accepted"] == 0):
                continue

            concept_delta_row = concept_deltas[feat_idx][row_idx]  # (d_target,)

            # Probe at scale=1 to estimate gradient direction
            probe_delta = cumulative_deltas[row_idx] + concept_delta_row
            probe_preds = tail.predict_row(row_idx, probe_delta)
            probe_p1 = float(probe_preds[row_idx, 1] if probe_preds.ndim == 2
                             else probe_preds[row_idx])

            # Gradient: change in P(class=1) per unit scale
            grad = probe_p1 - current_p1
            if abs(grad) < 1e-10:
                cs["tried"] += 1
                continue  # concept has no effect on this row

            # Analytical optimal scale (allows negative = sign flip)
            s_star = (target_p1 - current_p1) / grad

            # Warm-start: if historical data suggests a typical scale,
            # cap s_star to within 2x of the historical median to avoid
            # wasting backtracking steps on extreme analytical scales
            if cs["scales"]:
                hist_median = float(np.median(cs["scales"]))
                if abs(s_star) > 2 * abs(hist_median) and abs(hist_median) > 1e-6:
                    s_star = np.sign(s_star) * 2 * abs(hist_median)

            # Backtracking: try s*, then s*/2, ... until improvement
            # Reuse probe prediction when s* ≈ 1 to avoid redundant forward pass
            cs["tried"] += 1
            trial_scale = s_star
            for bt in range(max_backtrack + 1):
                if abs(trial_scale) < 1e-10:
                    break

                if bt == 0 and abs(trial_scale - 1.0) < 0.01:
                    # s* ≈ 1: reuse the probe we already computed
                    trial_p1 = probe_p1
                    trial_delta = probe_delta
                else:
                    trial_delta = cumulative_deltas[row_idx] + trial_scale * concept_delta_row
                    trial_preds = tail.predict_row(row_idx, trial_delta)
                    trial_p1 = float(trial_preds[row_idx, 1] if trial_preds.ndim == 2
                                     else trial_preds[row_idx])
                trial_dist = abs(trial_p1 - target_p1)

                if trial_dist < best_dist - 1e-8:
                    # Accept: update cumulative delta and current prediction
                    cumulative_deltas[row_idx] = trial_delta
                    accepted_counts[row_idx] += 1
                    best_dist = trial_dist
                    current_p1 = trial_p1
                    cs["accepted"] += 1
                    cs["scales"].append(float(trial_scale))
                    break

                trial_scale *= 0.5

            # Early exit: close enough to target
            if best_dist < converge_threshold:
                break

        final_p1[row_idx] = current_p1

        if (row_idx + 1) % 50 == 0 or row_idx == n_query - 1:
            logger.info("  row %d/%d: %d concepts, dist %.4f → %.4f",
                        row_idx + 1, n_query,
                        int(accepted_counts[row_idx]),
                        abs(float(bp1[row_idx]) - float(sp1[row_idx])),
                        best_dist)

    logger.info("Line search complete: mean %.1f, median %.0f, max %d concepts/row",
                accepted_counts.mean(), np.median(accepted_counts),
                int(accepted_counts.max()))

    # Per-concept summary
    all_scales = []
    for cs in concept_stats.values():
        all_scales.extend(cs["scales"])
    if all_scales:
        scales_arr = np.array(all_scales)
        n_neg = (scales_arr < 0).sum()
        logger.info("Scales: median=%.2f, range=[%.2f, %.2f], %d negative (%.0f%%)",
                    np.median(scales_arr), scales_arr.min(), scales_arr.max(),
                    n_neg, 100 * n_neg / len(scales_arr))

    n_tried = sum(1 for cs in concept_stats.values() if cs["tried"] > 0)
    n_ever_accepted = sum(1 for cs in concept_stats.values() if cs["accepted"] > 0)
    n_skipped = sum(1 for cs in concept_stats.values()
                    if cs["tried"] >= min_tries_before_skip and cs["accepted"] == 0)
    logger.info("Concepts: %d tried, %d ever accepted (%.0f%%), %d skipped as dead",
                n_tried, n_ever_accepted,
                100 * n_ever_accepted / max(n_tried, 1), n_skipped)

    # Log top concepts by acceptance rate
    concept_summary = []
    for feat_idx, cs in sorted(concept_stats.items(),
                                key=lambda x: x[1]["accepted"], reverse=True):
        if cs["accepted"] > 0:
            med_scale = float(np.median(cs["scales"]))
            concept_summary.append((feat_idx, cs["accepted"], cs["tried"], med_scale))
    for feat_idx, n_acc, n_try, med_s in concept_summary[:10]:
        logger.info("  feature %d: accepted %d/%d (%.0f%%), median scale=%.2f",
                    feat_idx, n_acc, n_try, 100 * n_acc / max(n_try, 1), med_s)

    # Build accepted_masks
    accepted_masks = torch.zeros(n_query, sae_hidden, dtype=torch.bool,
                                 device=query_source.device)
    for row_idx in range(n_query):
        for ki in range(min(int(accepted_counts[row_idx]), len(rankings[row_idx]))):
            accepted_masks[row_idx, rankings[row_idx][ki]] = True
    n_accepted = accepted_masks.sum(dim=1)

    # Reconstruct full prediction array (binary: p0 = 1 - p1)
    accepted_preds = np.column_stack([1.0 - final_p1, final_p1])

    return {
        "baseline_preds": baseline_np,
        "perrow_importance": importance,
        "perrow_rankings": rankings,
        "sweep_preds": [],
        "accepted_preds": accepted_preds,
        "accepted_counts": n_accepted.cpu().numpy(),
        "max_k_per_row": np.array([len(r) for r in rankings]),
        "accepted_masks": accepted_masks.cpu(),
        "query_activations": query_acts,
        "unmatched_features": list(feat_indices),
        "source_preds": source_preds,
        "y_query": y_q,
        "linear_map_r2": r2,
        "concept_stats": dict(concept_stats),
    }


# ── Per-Row Optimal Transfer ─────────────────────────────────────────────────


def find_per_row_optimal_transfer(
    perrow_result: Dict,
    source_preds: np.ndarray,
    target_baseline_preds: np.ndarray,
    y_query: np.ndarray,
) -> Dict:
    """Compute per-row gap closed using selective transfer results.

    With selective transfer, Phase 3 already determined the optimal set per row
    (accepted_counts). This function computes the gap-closed metric and packages
    results for plotting.
    """
    eps = 1e-7
    y = y_query.astype(float)

    # Source (strong) model's per-row logloss — this is the goal
    sp1 = source_preds[:, 1] if source_preds.ndim == 2 else source_preds
    sp = np.clip(sp1, eps, 1 - eps)
    target_row_ll = -(y * np.log(sp) + (1 - y) * np.log(1 - sp))

    # Target (weak) model's per-row logloss — starting point
    bp1 = target_baseline_preds[:, 1] if target_baseline_preds.ndim == 2 else target_baseline_preds
    bp = np.clip(bp1, eps, 1 - eps)
    baseline_row_ll = -(y * np.log(bp) + (1 - y) * np.log(1 - bp))

    # Accepted predictions — the final state after selective accept/reject
    ap = perrow_result["accepted_preds"]
    ap1 = ap[:, 1] if ap.ndim == 2 else ap
    ap_clipped = np.clip(ap1, eps, 1 - eps)
    accepted_row_ll = -(y * np.log(ap_clipped) + (1 - y) * np.log(1 - ap_clipped))

    # optimal_k = number of concepts accepted per row
    # Zero out for rows where weak model is already better (no transfer needed)
    raw_counts = perrow_result["accepted_counts"].astype(int)
    optimal_k = np.zeros_like(raw_counts)

    n_query = len(y_query)
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

    logger.info("Per-row accepted concepts: mean=%.1f, median=%d, max=%d",
                optimal_k.mean(), np.median(optimal_k), optimal_k.max())
    logger.info("Rows with 0 concepts accepted: %d (%.1f%%)",
                (optimal_k == 0).sum(), 100 * (optimal_k == 0).mean())

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


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Transfer concepts between tabular foundation models",
    )
    parser.add_argument("--source", type=str, required=True,
                        help="Model A key (auto-detects which is stronger)")
    parser.add_argument("--target", type=str, required=True,
                        help="Model B key (auto-detects which is weaker)")
    parser.add_argument("--dataset", type=str, default="kddcup09_appetency")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--task", type=str, default="classification")
    parser.add_argument("--reverse", action="store_true",
                        help="Transfer FROM weaker model TO stronger model "
                             "(reverse: weak model's concepts improve strong model)")
    parser.add_argument("--bidirectional", action="store_true",
                        help="Run both forward and reverse transfer, "
                             "plot both directions on one scatter")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regularization for linear map")
    parser.add_argument("--sae-dir", type=Path, default=DEFAULT_SAE_DIR)
    parser.add_argument("--layers-config", type=Path, default=DEFAULT_LAYERS_PATH)
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--translator", type=Path, default=None,
                        help="Path to universal embedding translator checkpoint. "
                             "If provided, uses MLP translator instead of per-dataset ridge.")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "output" / "figures")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    fig_dir = args.output_dir / args.dataset

    # Load universal translator if provided
    translator_model = None
    if args.translator:
        from scripts.embedding_translator import load_translator
        translator_model, translator_meta = load_translator(
            args.translator, device=args.device,
        )
        logger.info("Loaded universal translator from %s "
                     "(arch=%s, val_R²=%.4f, val_cos=%.4f)",
                     args.translator,
                     translator_meta.get("arch", "?"),
                     translator_meta.get("history", {}).get("best_val_r2", 0),
                     translator_meta.get("history", {}).get("best_val_cosine", 0))

    # Auto-detect stronger model by running both and comparing AUC.
    # Transfer FROM the stronger model TO the weaker one.
    from sklearn.metrics import roc_auc_score

    model_a, model_b = args.source, args.target

    logger.info("Getting baseline predictions to determine transfer direction...")
    X_ctx, y_ctx, X_q, y_q = _load_splits(args.dataset, args.task)
    layer_a = get_extraction_layer(model_a, args.layers_config)
    layer_b = get_extraction_layer(model_b, args.layers_config)

    emb_a, preds_a = capture_embeddings(model_a, X_ctx, y_ctx, X_q, layer_a, args.device, args.task)
    emb_b, preds_b = capture_embeddings(model_b, X_ctx, y_ctx, X_q, layer_b, args.device, args.task)

    pa1 = preds_a[:, 1] if preds_a.ndim == 2 else preds_a
    pb1 = preds_b[:, 1] if preds_b.ndim == 2 else preds_b
    auc_a = float(roc_auc_score(y_q, pa1))
    auc_b = float(roc_auc_score(y_q, pb1))

    # Always identify strong/weak by AUC
    if auc_a >= auc_b:
        strong_model, weak_model = model_a, model_b
        emb_strong, emb_weak = emb_a, emb_b
        strong_preds, weak_preds = preds_a, preds_b
        auc_strong, auc_weak = auc_a, auc_b
    else:
        strong_model, weak_model = model_b, model_a
        emb_strong, emb_weak = emb_b, emb_a
        strong_preds, weak_preds = preds_b, preds_a
        auc_strong, auc_weak = auc_b, auc_a

    # ── Import plot functions ──────────────────────────────────────────────
    from scripts.plot_prediction_scatter import (
        plot_perrow_diagnostic,
        plot_perrow_results,
        plot_perrow_scatter,
    )

    strong_p1 = strong_preds[:, 1] if strong_preds.ndim == 2 else strong_preds
    weak_p1 = weak_preds[:, 1] if weak_preds.ndim == 2 else weak_preds
    disp_strong = DISPLAY_NAMES.get(strong_model, strong_model)
    disp_weak = DISPLAY_NAMES.get(weak_model, weak_model)

    def _run_one_direction(src_model, tgt_model, emb_src, emb_tgt,
                           src_preds, tgt_preds, label):
        """Run transfer in one direction, return (transferred_p1, optimal)."""
        disp_src = DISPLAY_NAMES.get(src_model, src_model)
        disp_tgt = DISPLAY_NAMES.get(tgt_model, tgt_model)

        sp1 = src_preds[:, 1] if src_preds.ndim == 2 else src_preds
        logger.info("%s: %s (AUC=%.3f) → %s (AUC=%.3f)",
                    label, disp_src,
                    float(roc_auc_score(y_q, sp1)),
                    disp_tgt,
                    float(roc_auc_score(y_q, tgt_preds[:, 1] if tgt_preds.ndim == 2 else tgt_preds)))

        try:
            feat = get_unmatched_features(
                src_model, tgt_model, args.dataset, positive_only=False,
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No comparison data for {src_model} vs {tgt_model} on {args.dataset}. "
                f"Run: python scripts/concept_importance.py --model {src_model} "
                f"--compare {tgt_model} --dataset {args.dataset}"
            )
        if not feat:
            logger.warning("No unmatched features found for %s → %s.", disp_src, disp_tgt)
            return None, None

        n_pos = sum(1 for _, d in feat if d > 0)
        logger.info("  %d unmatched %s-only features (%d positive drop)",
                    len(feat), src_model, n_pos)

        result = perrow_sweep_transfer(
            source_model=src_model, target_model=tgt_model,
            dataset=args.dataset, ranked_features=feat,
            device=args.device, task=args.task, alpha=args.alpha,
            sae_dir=args.sae_dir, layers_path=args.layers_config,
            training_dir=args.training_dir,
            emb_source=emb_src, emb_target=emb_tgt,
            source_preds=src_preds, target_baseline_preds=tgt_preds,
            translator=translator_model,
        )
        opt = find_per_row_optimal_transfer(result, src_preds, tgt_preds, y_q)

        tp1_dir = tgt_preds[:, 1] if tgt_preds.ndim == 2 else tgt_preds
        ap = opt["accepted_preds"]
        ap1 = ap[:, 1] if ap.ndim == 2 else ap
        trans_p1 = np.where(opt["optimal_k"] > 0, ap1, tp1_dir)

        return trans_p1, opt

    # ── Run transfer(s) ──────────────────────────────────────────────────
    directions = []
    if args.bidirectional or not args.reverse:
        directions.append("forward")
    if args.bidirectional or args.reverse:
        directions.append("reverse")

    fwd_transferred = fwd_optimal = None
    rev_transferred = rev_optimal = None

    for direction in directions:
        if direction == "forward":
            trans_p1, opt = _run_one_direction(
                strong_model, weak_model, emb_strong, emb_weak,
                strong_preds, weak_preds, "Forward",
            )
            fwd_transferred, fwd_optimal = trans_p1, opt
        else:
            trans_p1, opt = _run_one_direction(
                weak_model, strong_model, emb_weak, emb_strong,
                weak_preds, strong_preds, "Reverse",
            )
            rev_transferred, rev_optimal = trans_p1, opt

    # ── Print results ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"  {disp_strong} AUC = {auc_strong:.4f}")
    print(f"  {disp_weak} AUC = {auc_weak:.4f}")

    if fwd_transferred is not None:
        auc_fwd = float(roc_auc_score(y_q, fwd_transferred))
        gap = auc_strong - auc_weak
        pct = (auc_fwd - auc_weak) / gap * 100 if abs(gap) > 0.001 else float("nan")
        print(f"  Forward ({disp_strong}→{disp_weak}): "
              f"{disp_weak}+ AUC = {auc_fwd:.4f} ({pct:+.1f}%)")

    if rev_transferred is not None:
        auc_rev = float(roc_auc_score(y_q, rev_transferred))
        delta = auc_rev - auc_strong
        print(f"  Reverse ({disp_weak}→{disp_strong}): "
              f"{disp_strong}+ AUC = {auc_rev:.4f} (\u0394={delta:+.4f})")

    print(f"{'='*60}")

    # ── Plots ────────────────────────────────────────────────────────────
    if args.bidirectional:
        # Combined scatter
        plot_perrow_scatter(
            strong_p1, weak_p1, fwd_transferred, fwd_optimal["optimal_k"],
            y_q, strong_model, weak_model, args.dataset,
            auc_strong, auc_weak,
            fig_dir / "transfer_bidirectional_scatter.pdf",
            mode="bidirectional",
            preds_intervened_rev=rev_transferred,
            optimal_k_rev=rev_optimal["optimal_k"],
        )
    elif args.reverse:
        plot_perrow_scatter(
            strong_p1, weak_p1, rev_transferred,
            rev_optimal["optimal_k"], y_q,
            strong_model, weak_model, args.dataset,
            auc_strong, auc_weak,
            fig_dir / "transfer_perrow_scatter.pdf",
            mode="reverse_transfer",
        )
        plot_perrow_results(
            rev_optimal["optimal_k"], rev_optimal["row_gap_closed"],
            rev_optimal["max_k_per_row"], rev_optimal["perrow_rankings"],
            weak_model, strong_model, args.dataset,
            fig_dir / "transfer_perrow.pdf", action="transferring",
        )
        plot_perrow_diagnostic(
            rev_optimal["optimal_k"], rev_optimal["row_gap_closed"],
            rev_optimal["accepted_preds"], rev_optimal["baseline_preds"],
            weak_preds, rev_optimal["max_k_per_row"], y_q,
            weak_model, strong_model, args.dataset,
            fig_dir / "transfer_perrow_diagnostic.pdf", action="transferring",
        )
    else:
        plot_perrow_scatter(
            strong_p1, weak_p1, fwd_transferred,
            fwd_optimal["optimal_k"], y_q,
            strong_model, weak_model, args.dataset,
            auc_strong, auc_weak,
            fig_dir / "transfer_perrow_scatter.pdf",
            mode="transfer",
        )
        plot_perrow_results(
            fwd_optimal["optimal_k"], fwd_optimal["row_gap_closed"],
            fwd_optimal["max_k_per_row"], fwd_optimal["perrow_rankings"],
            strong_model, weak_model, args.dataset,
            fig_dir / "transfer_perrow.pdf", action="transferring",
        )
        plot_perrow_diagnostic(
            fwd_optimal["optimal_k"], fwd_optimal["row_gap_closed"],
            fwd_optimal["accepted_preds"], fwd_optimal["baseline_preds"],
            strong_preds, fwd_optimal["max_k_per_row"], y_q,
            strong_model, weak_model, args.dataset,
            fig_dir / "transfer_perrow_diagnostic.pdf", action="transferring",
        )


if __name__ == "__main__":
    main()
