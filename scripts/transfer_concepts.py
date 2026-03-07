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
    get_unmatched_features,
    plot_prediction_scatter,
    plot_logloss_curve,
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


def compute_transfer_delta(
    sae_source: torch.nn.Module,
    emb_source: torch.Tensor,
    W: np.ndarray,
    b: np.ndarray,
    transfer_features: List[int],
    data_mean: Optional[torch.Tensor] = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Compute delta in target space for given source concepts.

    1. SAE encode source embeddings → h_source (sparse activations)
    2. Concept contribution: decode(h_selected) - decode(0)  (bias cancels)
    3. Map to target space: contribution @ W.T

    Returns:
        delta_target: (n_samples, d_target) delta to inject into target model
    """
    if not transfer_features or scale == 0.0:
        d_target = W.shape[0]
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

        # Map to target space via linear map (bias cancels in delta)
        W_t = torch.tensor(W, dtype=contribution_source.dtype,
                           device=contribution_source.device)
        delta_target = contribution_source @ W_t.T  # (n_samples, d_target)

        delta_target = delta_target * scale

    return delta_target


def compute_transfer_delta_perrow(
    sae_source: torch.nn.Module,
    emb_source: torch.Tensor,
    W: np.ndarray,
    b: np.ndarray,
    feature_masks: torch.Tensor,
    data_mean: Optional[torch.Tensor] = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Per-row transfer delta with different features per row.

    Analogous to compute_ablation_delta_perrow but maps through linear map
    to target space.

    Args:
        sae_source: Source SAE in eval mode
        emb_source: (n_rows, d_source_emb) raw source embeddings
        W: (d_target, d_source_emb) linear map weight matrix
        b: (d_target,) bias (unused — cancels in delta)
        feature_masks: (n_rows, sae_hidden) boolean, True = TRANSFER this feature
        data_mean: (d_source_emb,) centering mean for SAE
        scale: Multiplicative scaling factor

    Returns:
        delta_target: (n_rows, d_target) per-row deltas in target space
    """
    d_target = W.shape[0]
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

        # Map to target space
        W_t = torch.tensor(W, dtype=contribution_source.dtype,
                           device=contribution_source.device)
        delta_target = contribution_source @ W_t.T * scale

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
) -> Dict:
    """Per-row heterogeneous transfer sweep (3-phase pipeline).

    Phase 1: Per-feature importance (N forward passes through target model)
    Phase 2: Per-row ranking (filtered to firing features in source SAE)
    Phase 3: Heterogeneous sweep (max_k forward passes, per-row masks)

    Returns dict with same shape as perrow_sweep_intervene plus linear map info.
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

    # Fit linear map on query embeddings
    query_source = emb_source[-n_query:]
    query_target = emb_target[-n_query:]
    n_total_target = emb_target.shape[0]

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

    # --- Phase 1: per-feature importance (N forward passes through target) ---
    logger.info("Phase 1: per-feature importance for %d features...", len(feat_indices))
    baseline_np = target_baseline_preds
    individual_preds = []

    for feat_idx in feat_indices:
        delta_query = compute_transfer_delta(
            sae_source, query_source, W, b,
            [feat_idx], data_mean=data_mean, scale=1.0,
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
        individual_preds.append(result["ablated_preds"])

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

    # --- Phase 3: selective sweep (max_k forward passes) ---
    # Unlike ablation's cumulative sweep, transfer is selective: at each step k,
    # tentatively add the k-th ranked feature and keep it only if it improves
    # that row's logloss. This avoids accumulating noisy concepts.
    logger.info("Phase 3: selective sweep k=1..%d...", max_k)
    sae_hidden = h_encoded.shape[1]
    sweep_preds = []
    eps = 1e-7
    y = y_q.astype(float)

    # Track accepted masks and current-best logloss per row
    accepted_masks = torch.zeros(n_query, sae_hidden, dtype=torch.bool,
                                 device=query_source.device)
    # Baseline logloss per row
    bp1 = baseline_np[:, 1] if baseline_np.ndim == 2 else baseline_np
    bp_clipped = np.clip(bp1, eps, 1 - eps)
    best_ll = -(y * np.log(bp_clipped) + (1 - y) * np.log(1 - bp_clipped))

    for k in range(1, max_k + 1):
        # Build tentative masks: accepted + k-th ranked feature per row
        tentative_masks = accepted_masks.clone()
        for row_idx in range(n_query):
            if k - 1 < len(rankings[row_idx]):
                tentative_masks[row_idx, rankings[row_idx][k - 1]] = True

        query_delta = compute_transfer_delta_perrow(
            sae_source, query_source, W, b, tentative_masks, data_mean=data_mean,
        )
        full_delta = _build_full_delta(query_delta, n_total_target, n_query)

        result = intervene(
            model_key=target_model,
            X_context=X_ctx, y_context=y_ctx,
            X_query=X_q, y_query=y_q,
            external_delta=full_delta.to(device),
            device=device, task=task,
            layers_path=layers_path,
        )
        preds_k = result["ablated_preds"]

        # Per-row accept/reject: keep feature only if logloss improved
        p1_k = preds_k[:, 1] if preds_k.ndim == 2 else preds_k
        p_clipped = np.clip(p1_k, eps, 1 - eps)
        ll_k = -(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))

        for row_idx in range(n_query):
            if ll_k[row_idx] < best_ll[row_idx] - 1e-8:
                # Accept: update mask and best logloss
                accepted_masks[row_idx] = tentative_masks[row_idx]
                best_ll[row_idx] = ll_k[row_idx]

        # Record tentative preds (for diagnostic trajectory visualization)
        sweep_preds.append(preds_k)

    n_accepted = accepted_masks.sum(dim=1)
    logger.info("Selective sweep: mean %.1f, median %.0f, max %d concepts accepted/row",
                n_accepted.float().mean(), n_accepted.float().median(),
                int(n_accepted.max()))

    # Final accepted predictions: one forward pass with the accepted masks
    query_delta_final = compute_transfer_delta_perrow(
        sae_source, query_source, W, b, accepted_masks, data_mean=data_mean,
    )
    full_delta_final = _build_full_delta(query_delta_final, n_total_target, n_query)
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
        "accepted_counts": n_accepted.cpu().numpy(),
        "max_k_per_row": np.array([len(r) for r in rankings]),
        "accepted_masks": accepted_masks.cpu(),
        "query_activations": query_acts,
        "unmatched_features": list(feat_indices),
        "source_preds": source_preds,
        "y_query": y_q,
        "linear_map_r2": r2,
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
    optimal_k = perrow_result["accepted_counts"].astype(int)

    n_query = len(y_query)
    row_gap_closed = np.zeros(n_query)
    for row_idx in range(n_query):
        orig_gap = baseline_row_ll[row_idx] - target_row_ll[row_idx]
        if orig_gap <= 0:
            row_gap_closed[row_idx] = 1.0
        else:
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
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regularization for linear map")
    parser.add_argument("--sae-dir", type=Path, default=DEFAULT_SAE_DIR)
    parser.add_argument("--layers-config", type=Path, default=DEFAULT_LAYERS_PATH)
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--perrow", action="store_true",
                        help="Per-row heterogeneous transfer: find minimal concept "
                             "set per row that closes the gap to the strong model")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "output" / "figures")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    fig_dir = args.output_dir / args.dataset

    # Auto-detect stronger model by running both and comparing AUC.
    # Transfer FROM the stronger model TO the weaker one.
    from sklearn.metrics import roc_auc_score
    from scripts.plot_prediction_scatter import get_active_feature_count

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
    logger.info("Transfer direction: %s (AUC=%.3f) → %s (AUC=%.3f)",
                disp_s, max(auc_a, auc_b), disp_t, min(auc_a, auc_b))

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

    n_pos = sum(1 for _, d in unmatched if d > 0)
    logger.info("Found %d unmatched %s-only features (%d with positive drop)",
                len(unmatched), source_model, n_pos)
    for feat, drop in unmatched[:10]:
        logger.info("  feature %d: drop=%.4f", feat, drop)

    # ── Per-row transfer mode ───────────────────────────────────────────────
    if args.perrow:
        from scripts.plot_prediction_scatter import (
            plot_perrow_diagnostic,
            plot_perrow_results,
            plot_perrow_scatter,
        )

        perrow_result = perrow_sweep_transfer(
            source_model=source_model,
            target_model=target_model,
            dataset=args.dataset,
            ranked_features=unmatched,
            device=args.device,
            task=args.task,
            alpha=args.alpha,
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

        optimal = find_per_row_optimal_transfer(
            perrow_result, source_preds, target_preds, y_q,
        )

        # Use accepted predictions directly (selective transfer already optimal)
        ap = optimal["accepted_preds"]
        transferred_p1 = ap[:, 1] if ap.ndim == 2 else ap

        # Histogram + coverage curve
        plot_perrow_results(
            optimal["optimal_k"],
            optimal["row_gap_closed"],
            optimal["max_k_per_row"],
            optimal["perrow_rankings"],
            source_model,  # concept owner
            target_model,  # model being improved
            args.dataset,
            fig_dir / "transfer_perrow.pdf",
            action="transferring",
        )

        # Per-row scatter
        auc_s = float(roc_auc_score(y_q, sp1_pr))
        auc_t = float(roc_auc_score(y_q, tp1_pr))
        plot_perrow_scatter(
            sp1_pr, tp1_pr, transferred_p1,
            optimal["optimal_k"], y_q,
            source_model, target_model,
            args.dataset,
            auc_s, auc_t,
            fig_dir / "transfer_perrow_scatter.pdf",
            ablate_axis="y",  # target is y-axis, transfer shifts it toward source
            action="transferring",
        )

        # Diagnostic: gap-closed distribution + logloss trajectory + marginal effect
        plot_perrow_diagnostic(
            optimal["optimal_k"],
            optimal["row_gap_closed"],
            optimal["sweep_preds"],
            optimal["baseline_preds"],
            optimal["target_row_ll"],
            optimal["baseline_row_ll"],
            y_q,
            source_model,  # concept owner
            target_model,  # model being improved
            args.dataset,
            fig_dir / "transfer_perrow_diagnostic.pdf",
            action="transferring",
        )

    # ── Cumulative sweep (always runs) ────────────────────────────────────
    # Pass pre-captured embeddings to avoid redundant forward passes
    sweep = sweep_transfer(
        source_model=source_model,
        target_model=target_model,
        dataset=args.dataset,
        ranked_features=unmatched,
        device=args.device,
        task=args.task,
        alpha=args.alpha,
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

    auc_source = float(roc_auc_score(y, sp1))
    auc_target = float(roc_auc_score(y, tp1))
    auc_transferred = float(roc_auc_score(y, xp1))

    gap = auc_source - auc_target
    gap_closed = (auc_transferred - auc_target) / gap if gap > 0.001 else float("nan")

    print(f"\n{'='*60}")
    print(f"Concept transfer: {source_model} → {target_model}")
    print(f"Dataset: {args.dataset}")
    print(f"Linear map R² = {sweep['linear_map_r2']:.4f}")
    print(f"Optimal k = {k}/{len(unmatched)} concepts")
    print(f"Optimal features: {sweep['optimal_features']}")
    print(f"\n  {source_model} AUC = {auc_source:.4f}  (logloss={sweep['target_logloss']:.4f})")
    print(f"  {target_model} AUC = {auc_target:.4f}  (logloss={sweep['baseline_logloss']:.4f})")
    print(f"  {target_model}+transfer AUC = {auc_transferred:.4f}  "
          f"(logloss={sweep['logloss_curve'][k-1]:.4f})")
    if gap > 0.001:
        print(f"\n  Gap closed: {gap_closed:.1%}")
    print(f"{'='*60}")

    # Logloss curve (same format as ablation)
    # Flip labels: ablation degrades the strong model toward the weak model's logloss,
    # transfer improves the weak model toward the strong model's logloss.
    plot_logloss_curve(
        sweep["logloss_curve"],
        sweep["baseline_logloss"],
        sweep["target_logloss"],  # source's logloss = what we're aiming for
        k,
        unmatched,
        target_model,  # model being modified
        source_model,  # reference model (whose concepts we're transferring)
        args.dataset,
        fig_dir / "transfer_logloss.pdf",
        action="transferring",
    )

    # Scatter plot with optimal transfer overlay
    features_s = get_active_feature_count(source_model, args.dataset)
    features_t = get_active_feature_count(target_model, args.dataset)

    transfer_label = f"transfer {k}/{len(unmatched)} {disp_s}-only"
    ablation_levels = [(transfer_label, "#009E73", xp1)]

    plot_prediction_scatter(
        sp1, tp1, y,
        source_model, target_model, args.dataset,
        auc_source, auc_target, features_s, features_t,
        fig_dir / "transfer_scatter.pdf",
        ablation_levels=ablation_levels,
        ablate_axis="y",  # target is y-axis, transfer shifts it toward source
    )


if __name__ == "__main__":
    main()
