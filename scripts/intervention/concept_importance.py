#!/usr/bin/env python3
"""Per-concept importance via single-feature ablation.

For a given model and dataset, ablate each alive SAE feature one at a time
and measure the accuracy drop. The drop is that feature's "importance" to
the dataset.

Architecture: fits the model ONCE, captures hidden state ONCE, computes
baseline ONCE, then loops over features — each iteration is just a delta
computation + one forward pass with a hook.

Usage:
    python scripts/concept_importance.py --model tabdpt --dataset adult --device cuda
    python scripts/concept_importance.py --model tabdpt --dataset adult --top 20
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT

from scripts.sae.compare_sae_cross_model import DEFAULT_SAE_ROUND, SAE_FILENAME
from scripts.intervention.intervene_sae import (
    load_sae,
    load_norm_stats,
    get_extraction_layer,
    compute_ablation_delta,
)

logger = logging.getLogger(__name__)


def compute_importance_metric(y_true: np.ndarray, preds: np.ndarray, task: str) -> tuple:
    """Compute the appropriate metric for importance measurement.

    Uses TabArena-compatible metrics so importance is meaningful on imbalanced data:
    - Binary classification: AUC (from probabilities)
    - Multiclass classification: neg_logloss (higher = better)
    - Regression: neg_RMSE (higher = better)

    Returns (metric_value, metric_name) where higher is always better,
    so drop = baseline - ablated is positive when ablation hurts.
    """
    from sklearn.metrics import roc_auc_score, log_loss

    if task == "regression":
        rmse = float(np.sqrt(np.mean((preds - y_true) ** 2)))
        return -rmse, "neg_rmse"

    # Classification: check binary vs multiclass
    n_classes = preds.shape[1] if preds.ndim == 2 else len(np.unique(y_true))
    if n_classes == 2:
        proba = preds[:, 1] if preds.ndim == 2 else preds
        try:
            return float(roc_auc_score(y_true, proba)), "auc"
        except ValueError:
            # Fallback if only one class in y_true
            return float(-log_loss(y_true, preds, labels=np.arange(n_classes))), "neg_logloss"
    else:
        return float(-log_loss(y_true, preds, labels=np.arange(n_classes))), "neg_logloss"


def compute_per_row_loss(y_true: np.ndarray, preds: np.ndarray, task: str) -> np.ndarray:
    """Compute per-row loss (lower = better predictions).

    - Binary/multiclass classification: cross-entropy -log(p_correct)
    - Regression: squared error (y - y_hat)^2

    Returns (n_samples,) array. Higher values = worse predictions.
    Importance = ablated_loss - baseline_loss: positive means the feature helped.
    """
    eps = 1e-7
    if task == "regression":
        return (preds.ravel() - y_true.ravel()) ** 2

    y_int = y_true.astype(int)
    if preds.ndim == 2:
        p_correct = preds[np.arange(len(y_int)), y_int]
    else:
        p = preds.ravel()
        p_correct = np.where(y_int == 1, p, 1 - p)

    p_correct = np.clip(p_correct, eps, 1 - eps)
    return -np.log(p_correct)


DEFAULT_CONCEPT_LABELS = PROJECT_ROOT / "output" / "cross_model_concept_labels_round8.json"
DEFAULT_CONCEPT_REGRESSION = PROJECT_ROOT / "output" / f"sae_concept_analysis_round{DEFAULT_SAE_ROUND}.json"
DEFAULT_PYMFE_CACHE = PROJECT_ROOT / "output" / "pymfe_tabarena_cache.json"

# Map our model keys to the concept labels file keys
MODEL_KEY_TO_LABEL_KEY = {
    "tabpfn": "TabPFN",
    "mitra": "Mitra",
    "tabicl": "TabICL",
    "tabicl_v2": "TabICL-v2",
    "tabdpt": "TabDPT",
    "hyperfast": "HyperFast",
    "carte": "CARTE",
    "tabula8b": "Tabula-8B",
}


def get_alive_features(model_key: str, labels_path: Path = DEFAULT_CONCEPT_LABELS) -> List[int]:
    """Get sorted list of alive feature indices for a model."""
    with open(labels_path) as f:
        data = json.load(f)
    label_key = MODEL_KEY_TO_LABEL_KEY.get(model_key, model_key)
    features = data["feature_lookup"][label_key]
    return sorted(int(k) for k in features.keys())


def get_feature_labels(model_key: str, labels_path: Path = DEFAULT_CONCEPT_LABELS) -> Dict[int, str]:
    """Get feature_idx -> label mapping for a model."""
    with open(labels_path) as f:
        data = json.load(f)
    label_key = MODEL_KEY_TO_LABEL_KEY.get(model_key, model_key)
    features = data["feature_lookup"][label_key]
    return {int(k): v.get("label", "unknown") for k, v in features.items()}


# ── TabDPT single-feature sweep ─────────────────────────────────────────────


def sweep_tabdpt(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    sae: torch.nn.Module,
    alive_features: List[int],
    extraction_layer: int,
    device: str = "cuda",
    task: str = "classification",
    data_mean: Optional[torch.Tensor] = None,
    data_std: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Sweep single-feature ablation for TabDPT.

    Fits model once, captures hidden state once, then loops over features.
    Each iteration: compute delta for 1 feature, inject via hook, get predictions.

    Returns:
        Dict with baseline_preds, baseline_metric, feature_indices, row_feature_drops,
        feature_n_firing,
        n_query, y_query, metric_name.
    """
    from tabdpt import TabDPTClassifier, TabDPTRegressor

    if task == "regression":
        clf = TabDPTRegressor(device=device, compile=False)
    else:
        clf = TabDPTClassifier(device=device, compile=False)
    clf.fit(X_context, y_context)

    model = clf.model
    encoder_layers = model.transformer_encoder

    # --- Capture hidden state + baseline predictions (once) ---
    captured = {}

    def capture_hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            captured["hidden"] = out.detach()

    handle = encoder_layers[extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            if task == "regression":
                baseline_preds = clf.predict(X_query)
            else:
                baseline_preds = clf.predict_proba(X_query)
    finally:
        handle.remove()

    hidden_state = captured["hidden"]
    if hidden_state.ndim == 3:
        all_emb = hidden_state.mean(dim=1)  # (n_samples, H)
    elif hidden_state.ndim == 2:
        all_emb = hidden_state
    else:
        raise ValueError(f"Unexpected hidden state shape: {hidden_state.shape}")

    # Pre-compute SAE encoding (once) for delta computation
    with torch.no_grad():
        x_norm = all_emb
        if data_mean is not None:
            x_norm = (x_norm - data_mean) / data_std
        h_full = sae.encode(x_norm)
        recon_full = sae.decode(h_full)

    baseline_preds_np = np.asarray(baseline_preds)
    baseline_metric, metric_name = compute_importance_metric(y_query, baseline_preds_np, task)
    baseline_row_loss = compute_per_row_loss(y_query, baseline_preds_np, task)

    # Query-row SAE activations for firing detection
    n_query = len(y_query)
    h_query = h_full[-n_query:]

    # --- Sweep: ablate one feature at a time ---
    n_features = len(alive_features)
    row_feature_drops = np.zeros((n_query, n_features))
    feature_n_firing = np.zeros(n_features, dtype=int)

    t0 = time.time()
    for i, feat_idx in enumerate(alive_features):
        # Compute single-feature delta using pre-computed encoding
        with torch.no_grad():
            h_ablated = h_full.clone()
            h_ablated[:, feat_idx] = 0.0
            recon_ablated = sae.decode(h_ablated)
            delta = (recon_ablated - recon_full) * data_std  # denormalize to raw space

        # Inject delta via hook
        def make_hook(d):
            def modify_hook(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                if isinstance(out, torch.Tensor):
                    out = out.clone()
                    if out.ndim == 3:
                        out += d.unsqueeze(1)
                    elif out.ndim == 2:
                        out += d
                    if isinstance(output, tuple):
                        return (out,) + output[1:]
                    return out
                return output
            return modify_hook

        handle = encoder_layers[extraction_layer].register_forward_hook(make_hook(delta))
        try:
            with torch.no_grad():
                if task == "regression":
                    preds = clf.predict(X_query)
                else:
                    preds = clf.predict_proba(X_query)
        finally:
            handle.remove()

        preds_np = np.asarray(preds)
        ablated_row_loss = compute_per_row_loss(y_query, preds_np, task)
        row_importance = ablated_row_loss - baseline_row_loss

        fires = (h_query[:, feat_idx] > 0).cpu().numpy()
        n_fire = int(fires.sum())

        row_feature_drops[:, i] = row_importance
        feature_n_firing[i] = n_fire

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_features - i - 1) / rate
            logger.info(
                f"  [{i+1}/{n_features}] feat={feat_idx} "
                f"drop={row_feature_drops[:, i].mean():+.4f} fire={n_fire}/{n_query} "
                f"({rate:.1f} feat/s, ETA {eta:.0f}s)"
            )

    # Count features that fire on at least one row
    n_active = int((h_full[:, alive_features] > 0).any(dim=0).sum().item())

    return {
        "baseline_preds": baseline_preds_np,
        "baseline_metric": baseline_metric,
        "metric_name": metric_name,
        "feature_indices": np.array(alive_features),
        "row_feature_drops": row_feature_drops,
        "feature_n_firing": feature_n_firing,
        "n_query": n_query,
        "n_active_features": n_active,
        "y_query": np.asarray(y_query),
    }


# ── TabPFN single-feature sweep ─────────────────────────────────────────────


def sweep_tabpfn(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    sae: torch.nn.Module,
    alive_features: List[int],
    extraction_layer: int,
    device: str = "cuda",
    task: str = "classification",
    data_mean: Optional[torch.Tensor] = None,
    data_std: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Sweep single-feature ablation for TabPFN."""
    from models.tabpfn_utils import load_tabpfn

    clf = load_tabpfn(task=task, device=device, n_estimators=1)
    clf.fit(X_context, y_context)

    model = clf.model_ if hasattr(clf, "model_") else clf.transformer_
    layers = model.transformer_encoder.layers

    captured = {}

    def capture_hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            captured["hidden"] = out.detach()

    handle = layers[extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            if task == "regression":
                baseline_preds = clf.predict(X_query)
            else:
                baseline_preds = clf.predict_proba(X_query)
    finally:
        handle.remove()

    hidden_state = captured["hidden"]
    # TabPFN: (1, seq, n_features+1, H) — mean over structure dim
    if hidden_state.ndim == 4:
        all_emb = hidden_state[0].mean(dim=1)  # (seq, H)
    elif hidden_state.ndim == 3:
        all_emb = hidden_state[0] if hidden_state.shape[0] == 1 else hidden_state.mean(dim=0)
    else:
        all_emb = hidden_state

    with torch.no_grad():
        x_norm = all_emb
        if data_mean is not None:
            x_norm = (x_norm - data_mean) / data_std
        h_full = sae.encode(x_norm)
        recon_full = sae.decode(h_full)

    baseline_preds_np = np.asarray(baseline_preds)
    baseline_metric, metric_name = compute_importance_metric(y_query, baseline_preds_np, task)
    baseline_row_loss = compute_per_row_loss(y_query, baseline_preds_np, task)

    # Query-row SAE activations for firing detection
    n_query = len(y_query)
    h_query = h_full[-n_query:]

    n_features = len(alive_features)
    row_feature_drops = np.zeros((n_query, n_features))
    feature_n_firing = np.zeros(n_features, dtype=int)

    t0 = time.time()
    for i, feat_idx in enumerate(alive_features):
        with torch.no_grad():
            h_ablated = h_full.clone()
            h_ablated[:, feat_idx] = 0.0
            recon_ablated = sae.decode(h_ablated)
            delta = (recon_ablated - recon_full) * data_std

        def make_hook(d):
            def modify_hook(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                if isinstance(out, torch.Tensor):
                    out = out.clone()
                    if out.ndim == 4:
                        out[0] += d.unsqueeze(1)
                    elif out.ndim == 3:
                        out[0] += d
                    if isinstance(output, tuple):
                        return (out,) + output[1:]
                    return out
                return output
            return modify_hook

        handle = layers[extraction_layer].register_forward_hook(make_hook(delta))
        try:
            with torch.no_grad():
                if task == "regression":
                    preds = clf.predict(X_query)
                else:
                    preds = clf.predict_proba(X_query)
        finally:
            handle.remove()

        preds_np = np.asarray(preds)
        ablated_row_loss = compute_per_row_loss(y_query, preds_np, task)
        row_importance = ablated_row_loss - baseline_row_loss

        fires = (h_query[:, feat_idx] > 0).cpu().numpy()
        n_fire = int(fires.sum())

        row_feature_drops[:, i] = row_importance
        feature_n_firing[i] = n_fire

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_features - i - 1) / rate
            logger.info(
                f"  [{i+1}/{n_features}] feat={feat_idx} "
                f"drop={row_feature_drops[:, i].mean():+.4f} fire={n_fire}/{n_query} "
                f"({rate:.1f} feat/s, ETA {eta:.0f}s)"
            )

    # Count features that fire on at least one row
    n_active = int((h_full[:, alive_features] > 0).any(dim=0).sum().item())

    return {
        "baseline_preds": baseline_preds_np,
        "baseline_metric": baseline_metric,
        "metric_name": metric_name,
        "feature_indices": np.array(alive_features),
        "row_feature_drops": row_feature_drops,
        "feature_n_firing": feature_n_firing,
        "n_query": n_query,
        "n_active_features": n_active,
        "y_query": np.asarray(y_query),
    }


# ── Mitra single-feature sweep ──────────────────────────────────────────────


def sweep_mitra(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    sae: torch.nn.Module,
    alive_features: List[int],
    extraction_layer: int,
    device: str = "cuda",
    task: str = "classification",
    data_mean: Optional[torch.Tensor] = None,
    data_std: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Sweep single-feature ablation for Mitra.

    Mitra layers return (support, query) tuples. Must modify both.
    Must save/restore RNG state for determinism.
    """
    if task == "regression":
        from autogluon.tabular.models.mitra.sklearn_interface import MitraRegressor
        clf = MitraRegressor(device=device, n_estimators=1, fine_tune=False)
    else:
        from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier
        clf = MitraClassifier(device=device, n_estimators=1, fine_tune=False)

    clf.fit(X_context, y_context)
    torch.cuda.empty_cache()

    trainer = clf.trainers[0]
    model = trainer.model
    layers = model.layers

    # Save RNG state for determinism
    rng_state = trainer.rng.get_state()

    # --- Capture both support and query hidden states ---
    captured_support = []
    captured_query = []

    def capture_hook(module, input, output):
        if isinstance(output, tuple) and len(output) >= 2:
            sup, qry = output[0], output[1]
            if isinstance(sup, torch.Tensor):
                captured_support.append(sup.detach())
            if isinstance(qry, torch.Tensor):
                captured_query.append(qry.detach())

    handle = layers[extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            if task == "regression":
                baseline_preds = clf.predict(X_query)
            else:
                baseline_preds = clf.predict_proba(X_query)
    finally:
        handle.remove()

    if not captured_support or not captured_query:
        raise RuntimeError("Mitra hook failed to capture support/query tensors")

    support_hidden = captured_support[0]  # (1, n_ctx, n_feat+1, H)
    query_hidden = captured_query[0]  # (1, n_query, n_feat+1, H)

    # Mean over structure dim
    support_emb = support_hidden[0].mean(dim=1)  # (n_ctx, H)
    query_emb = query_hidden[0].mean(dim=1)  # (n_query, H)
    all_emb = torch.cat([support_emb, query_emb], dim=0)  # (n_ctx+n_query, H)

    with torch.no_grad():
        x_norm = all_emb
        if data_mean is not None:
            x_norm = (x_norm - data_mean) / data_std
        h_full = sae.encode(x_norm)
        recon_full = sae.decode(h_full)

    baseline_preds_np = np.asarray(baseline_preds)
    baseline_metric, metric_name = compute_importance_metric(y_query, baseline_preds_np, task)
    baseline_row_loss = compute_per_row_loss(y_query, baseline_preds_np, task)

    n_sup = support_emb.shape[0]
    n_query = len(y_query)
    h_query = h_full[n_sup:]  # query portion of SAE activations

    n_features = len(alive_features)
    row_feature_drops = np.zeros((n_query, n_features))
    feature_n_firing = np.zeros(n_features, dtype=int)

    t0 = time.time()
    for i, feat_idx in enumerate(alive_features):
        with torch.no_grad():
            h_ablated = h_full.clone()
            h_ablated[:, feat_idx] = 0.0
            recon_ablated = sae.decode(h_ablated)
            delta = (recon_ablated - recon_full) * data_std

        delta_sup = delta[:n_sup]  # (n_ctx, H)
        delta_qry = delta[n_sup:]  # (n_query, H)

        def make_hook(d_sup, d_qry):
            sup_offset = [0]
            qry_offset = [0]

            def modify_hook(module, input, output):
                if not (isinstance(output, tuple) and len(output) >= 2):
                    return output
                sup, qry = output[0], output[1]
                modified = list(output)
                if isinstance(sup, torch.Tensor) and sup.ndim == 4:
                    sup = sup.clone()
                    n_s = sup.shape[1]
                    s = sup_offset[0]
                    sup[0] += d_sup[s:s + n_s].unsqueeze(1)
                    sup_offset[0] = s + n_s
                    modified[0] = sup
                if isinstance(qry, torch.Tensor) and qry.ndim == 4:
                    qry = qry.clone()
                    n_q = qry.shape[1]
                    s = qry_offset[0]
                    qry[0] += d_qry[s:s + n_q].unsqueeze(1)
                    qry_offset[0] = s + n_q
                    modified[1] = qry
                return tuple(modified)
            return modify_hook

        # Restore RNG state for deterministic batching
        trainer.rng.set_state(rng_state)

        handle = layers[extraction_layer].register_forward_hook(make_hook(delta_sup, delta_qry))
        try:
            with torch.no_grad():
                if task == "regression":
                    preds = clf.predict(X_query)
                else:
                    preds = clf.predict_proba(X_query)
        finally:
            handle.remove()

        preds_np = np.asarray(preds)
        ablated_row_loss = compute_per_row_loss(y_query, preds_np, task)
        row_importance = ablated_row_loss - baseline_row_loss

        fires = (h_query[:, feat_idx] > 0).cpu().numpy()
        n_fire = int(fires.sum())

        row_feature_drops[:, i] = row_importance
        feature_n_firing[i] = n_fire

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_features - i - 1) / rate
            logger.info(
                f"  [{i+1}/{n_features}] feat={feat_idx} "
                f"drop={row_feature_drops[:, i].mean():+.4f} fire={n_fire}/{n_query} "
                f"({rate:.1f} feat/s, ETA {eta:.0f}s)"
            )

    # Count features that fire on at least one row
    n_active = int((h_full[:, alive_features] > 0).any(dim=0).sum().item())

    return {
        "baseline_preds": baseline_preds_np,
        "baseline_metric": baseline_metric,
        "metric_name": metric_name,
        "feature_indices": np.array(alive_features),
        "row_feature_drops": row_feature_drops,
        "feature_n_firing": feature_n_firing,
        "n_query": n_query,
        "n_active_features": n_active,
        "y_query": np.asarray(y_query),
    }


# ── TabICL single-feature sweep ─────────────────────────────────────────────


def sweep_tabicl(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    sae: torch.nn.Module,
    alive_features: List[int],
    extraction_layer: int,
    device: str = "cuda",
    task: str = "classification",
    data_mean: Optional[torch.Tensor] = None,
    data_std: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Sweep single-feature ablation for TabICL.

    TabICL has ICL predictor blocks with 3D hidden state (n_ensemble, seq, 512).
    """
    from tabicl import TabICLClassifier

    clf = TabICLClassifier(device=device, n_estimators=1)
    clf.fit(X_context, y_context)

    model = clf.model_
    blocks = model.icl_predictor.tf_icl.blocks

    # --- Pass 1: Capture hidden state + baseline predictions ---
    captured = {}

    def capture_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured["hidden"] = output.detach()

    handle = blocks[extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            baseline_preds = clf.predict_proba(X_query)
    finally:
        handle.remove()

    hidden_state = captured["hidden"]
    # Shape: (n_ensemble, n_ctx+n_query, 512) — mean-pool ensemble dim
    all_emb = hidden_state.mean(dim=0)  # (seq_len, 512)

    with torch.no_grad():
        x_norm = all_emb
        if data_mean is not None:
            x_norm = (x_norm - data_mean) / data_std
        h_full = sae.encode(x_norm)
        recon_full = sae.decode(h_full)

    baseline_preds_np = np.asarray(baseline_preds)
    baseline_metric, metric_name = compute_importance_metric(y_query, baseline_preds_np, task)
    baseline_row_loss = compute_per_row_loss(y_query, baseline_preds_np, task)

    # Query-row SAE activations for firing detection
    n_query = len(y_query)
    h_query = h_full[-n_query:]

    n_features = len(alive_features)
    row_feature_drops = np.zeros((n_query, n_features))
    feature_n_firing = np.zeros(n_features, dtype=int)

    t0 = time.time()
    for i, feat_idx in enumerate(alive_features):
        with torch.no_grad():
            h_ablated = h_full.clone()
            h_ablated[:, feat_idx] = 0.0
            recon_ablated = sae.decode(h_ablated)
            delta = (recon_ablated - recon_full) * data_std

        # Broadcast delta to (1, seq_len, 512) for ensemble dim
        delta_broadcast = delta.unsqueeze(0)

        def make_hook(d):
            def modify_hook(module, input, output):
                if isinstance(output, torch.Tensor) and output.ndim == 3:
                    out = output.clone()
                    out += d
                    return out
                return output
            return modify_hook

        handle = blocks[extraction_layer].register_forward_hook(make_hook(delta_broadcast))
        try:
            with torch.no_grad():
                preds = clf.predict_proba(X_query)
        finally:
            handle.remove()

        preds_np = np.asarray(preds)
        ablated_row_loss = compute_per_row_loss(y_query, preds_np, task)
        row_importance = ablated_row_loss - baseline_row_loss

        fires = (h_query[:, feat_idx] > 0).cpu().numpy()
        n_fire = int(fires.sum())

        row_feature_drops[:, i] = row_importance
        feature_n_firing[i] = n_fire

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_features - i - 1) / rate
            logger.info(
                f"  [{i+1}/{n_features}] feat={feat_idx} "
                f"drop={row_feature_drops[:, i].mean():+.4f} fire={n_fire}/{n_query} "
                f"({rate:.1f} feat/s, ETA {eta:.0f}s)"
            )

    # Count features that fire on at least one row
    n_active = int((h_full[:, alive_features] > 0).any(dim=0).sum().item())

    return {
        "baseline_preds": baseline_preds_np,
        "baseline_metric": baseline_metric,
        "metric_name": metric_name,
        "feature_indices": np.array(alive_features),
        "row_feature_drops": row_feature_drops,
        "feature_n_firing": feature_n_firing,
        "n_query": n_query,
        "n_active_features": n_active,
        "y_query": np.asarray(y_query),
    }


# ── TabICL-v2 single-feature sweep ───────────────────────────────────────────


def sweep_tabicl_v2(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    sae: torch.nn.Module,
    alive_features: List[int],
    extraction_layer: int,
    device: str = "cuda",
    task: str = "classification",
    data_mean: Optional[torch.Tensor] = None,
    data_std: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Sweep single-feature ablation for TabICL v2.

    Clone of sweep_tabicl but supports regression via TabICLRegressor.
    """
    if task == "regression":
        from tabicl import TabICLRegressor
        clf = TabICLRegressor(device=device, n_estimators=1)
    else:
        from tabicl import TabICLClassifier
        clf = TabICLClassifier(device=device, n_estimators=1)

    clf.fit(X_context, y_context)

    model = clf.model_
    blocks = model.icl_predictor.tf_icl.blocks

    # --- Pass 1: Capture hidden state + baseline predictions ---
    captured = {}

    def capture_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured["hidden"] = output.detach()

    handle = blocks[extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            if task == "regression":
                baseline_preds = clf.predict(X_query)
            else:
                baseline_preds = clf.predict_proba(X_query)
    finally:
        handle.remove()

    hidden_state = captured["hidden"]
    # Shape: (n_ensemble, n_ctx+n_query, 512) — mean-pool ensemble dim
    all_emb = hidden_state.mean(dim=0)  # (seq_len, 512)

    with torch.no_grad():
        x_norm = all_emb
        if data_mean is not None:
            x_norm = (x_norm - data_mean) / data_std
        h_full = sae.encode(x_norm)
        recon_full = sae.decode(h_full)

    baseline_preds_np = np.asarray(baseline_preds)
    baseline_metric, metric_name = compute_importance_metric(y_query, baseline_preds_np, task)
    baseline_row_loss = compute_per_row_loss(y_query, baseline_preds_np, task)

    # Query-row SAE activations for firing detection
    n_query = len(y_query)
    h_query = h_full[-n_query:]

    n_features = len(alive_features)
    row_feature_drops = np.zeros((n_query, n_features))
    feature_n_firing = np.zeros(n_features, dtype=int)

    t0 = time.time()
    for i, feat_idx in enumerate(alive_features):
        with torch.no_grad():
            h_ablated = h_full.clone()
            h_ablated[:, feat_idx] = 0.0
            recon_ablated = sae.decode(h_ablated)
            delta = (recon_ablated - recon_full) * data_std

        # Broadcast delta to (1, seq_len, 512) for ensemble dim
        delta_broadcast = delta.unsqueeze(0)

        def make_hook(d):
            def modify_hook(module, input, output):
                if isinstance(output, torch.Tensor) and output.ndim == 3:
                    out = output.clone()
                    out += d
                    return out
                return output
            return modify_hook

        handle = blocks[extraction_layer].register_forward_hook(make_hook(delta_broadcast))
        try:
            with torch.no_grad():
                if task == "regression":
                    preds = clf.predict(X_query)
                else:
                    preds = clf.predict_proba(X_query)
        finally:
            handle.remove()

        preds_np = np.asarray(preds)
        ablated_row_loss = compute_per_row_loss(y_query, preds_np, task)
        row_importance = ablated_row_loss - baseline_row_loss

        fires = (h_query[:, feat_idx] > 0).cpu().numpy()
        n_fire = int(fires.sum())

        row_feature_drops[:, i] = row_importance
        feature_n_firing[i] = n_fire

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_features - i - 1) / rate
            logger.info(
                f"  [{i+1}/{n_features}] feat={feat_idx} "
                f"drop={row_feature_drops[:, i].mean():+.4f} fire={n_fire}/{n_query} "
                f"({rate:.1f} feat/s, ETA {eta:.0f}s)"
            )

    # Count features that fire on at least one row
    n_active = int((h_full[:, alive_features] > 0).any(dim=0).sum().item())

    return {
        "baseline_preds": baseline_preds_np,
        "baseline_metric": baseline_metric,
        "metric_name": metric_name,
        "feature_indices": np.array(alive_features),
        "row_feature_drops": row_feature_drops,
        "feature_n_firing": feature_n_firing,
        "n_query": n_query,
        "n_active_features": n_active,
        "y_query": np.asarray(y_query),
    }


# ── CARTE single-feature sweep ──────────────────────────────────────────────


def sweep_carte(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    sae: torch.nn.Module,
    alive_features: List[int],
    extraction_layer: int,
    device: str = "cuda",
    task: str = "classification",
    data_mean: Optional[torch.Tensor] = None,
    data_std: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Sweep single-feature ablation for CARTE.

    CARTE uses star graphs where central node = row embedding. We hook
    on the appropriate GNN module, extract central node embeddings, and
    inject deltas at central node positions only.
    """
    from models.carte_embeddings import _patch_carte_amp, _find_fasttext_model
    from scripts.intervention.intervene_sae import _carte_prepare_graphs
    _patch_carte_amp()

    from carte_ai import CARTEClassifier, CARTERegressor, Table2GraphTransformer
    from torch_geometric.data import Batch
    from sklearn.preprocessing import RobustScaler

    ft_path = _find_fasttext_model()
    if not ft_path:
        raise ValueError("FastText model not found for CARTE sweep")

    # Robust preprocessing (matches extraction code)
    X_context = np.asarray(X_context, dtype=np.float32)
    X_query = np.asarray(X_query, dtype=np.float32)

    col_std = X_context.std(axis=0)
    nonconstant = col_std > 0
    if not nonconstant.all():
        X_context = X_context[:, nonconstant]
        X_query = X_query[:, nonconstant]

    scaler = RobustScaler()
    X_context = scaler.fit_transform(X_context)
    X_query = scaler.transform(X_query)
    X_context = np.clip(X_context, -10, 10)
    X_query = np.clip(X_query, -10, 10)

    # Prepare targets
    y_context = np.asarray(y_context)
    if y_context.dtype == np.float64:
        y_context = y_context.astype(np.int64)

    feature_names = [f"f{i}" for i in range(X_context.shape[1])]
    t2g = Table2GraphTransformer(lm_model="fasttext", fasttext_model_path=ft_path)

    X_context_graph = _carte_prepare_graphs(X_context, feature_names, t2g, fit=True)
    X_query_graph = _carte_prepare_graphs(X_query, feature_names, t2g, fit=False)

    for i, g in enumerate(X_context_graph):
        g.y = torch.tensor([y_context[i]], dtype=torch.float32)

    if task == "regression":
        clf = CARTERegressor(device=device, num_model=1, max_epoch=50, disable_pbar=True)
    else:
        clf = CARTEClassifier(device=device, num_model=1, max_epoch=50, disable_pbar=True)
    clf.fit(X_context_graph, y_context)
    torch.cuda.empty_cache()

    n_query = len(y_query)
    model = clf.model_list_[0]
    model.eval()
    base = model.ft_base

    # Map extraction_layer to module (same logic as CARTETail)
    if extraction_layer == 0:
        hook_module = base.initial_x
    elif extraction_layer == 1:
        hook_module = base.read_out_block
    else:
        classifier_layers = [m for m in model.ft_classifier
                             if isinstance(m, torch.nn.Linear)]
        cls_idx = extraction_layer - 2
        hook_module = (classifier_layers[cls_idx]
                       if cls_idx < len(classifier_layers)
                       else base.read_out_block)

    # --- Pass 1: Capture hidden state + baseline predictions ---
    captured = {}

    def capture_hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            captured["hidden"] = out.detach()

    batch_q = Batch.from_data_list(X_query_graph).to(device)
    handle = hook_module.register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            model(batch_q)
    finally:
        handle.remove()

    hidden = captured["hidden"]
    # Extract central node indices from batch.ptr
    if hidden.shape[0] > n_query and hasattr(batch_q, 'ptr'):
        central_indices = [int(batch_q.ptr[i]) for i in range(n_query)]
    elif hidden.shape[0] == n_query:
        central_indices = list(range(n_query))
    else:
        raise ValueError(f"Cannot extract central nodes: hidden {hidden.shape}")

    all_emb = hidden[central_indices]  # (n_query, emb_dim)

    # Get baseline predictions via clf (which re-does graph conversion internally)
    with torch.no_grad():
        if task == "regression":
            baseline_preds = clf.predict(X_query_graph)
        else:
            baseline_preds = clf.predict_proba(X_query_graph)

    # Pre-compute SAE encoding with per-dataset normalization
    with torch.no_grad():
        x_norm = all_emb
        if data_mean is not None:
            x_norm = (x_norm - data_mean) / data_std
        h_full = sae.encode(x_norm)
        recon_full = sae.decode(h_full)

    baseline_preds_np = np.asarray(baseline_preds)
    baseline_metric, metric_name = compute_importance_metric(y_query, baseline_preds_np, task)
    baseline_row_loss = compute_per_row_loss(y_query, baseline_preds_np, task)

    # Query-row SAE activations for firing detection (all rows are query rows)
    h_query = h_full

    n_features = len(alive_features)
    row_feature_drops = np.zeros((n_query, n_features))
    feature_n_firing = np.zeros(n_features, dtype=int)

    t0 = time.time()
    for i, feat_idx in enumerate(alive_features):
        with torch.no_grad():
            h_ablated = h_full.clone()
            h_ablated[:, feat_idx] = 0.0
            recon_ablated = sae.decode(h_ablated)
            delta = (recon_ablated - recon_full) * data_std  # denormalize to raw space

        def make_hook(d, c_idx):
            def modify_hook(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                if isinstance(out, torch.Tensor):
                    out = out.clone()
                    for row_i, node_idx in enumerate(c_idx):
                        out[node_idx] += d[row_i]
                    if isinstance(output, tuple):
                        return (out,) + output[1:]
                    return out
                return output
            return modify_hook

        handle = hook_module.register_forward_hook(make_hook(delta, central_indices))
        try:
            with torch.no_grad():
                if task == "regression":
                    preds = clf.predict(X_query_graph)
                else:
                    preds = clf.predict_proba(X_query_graph)
        finally:
            handle.remove()

        preds_np = np.asarray(preds)
        ablated_row_loss = compute_per_row_loss(y_query, preds_np, task)
        row_importance = ablated_row_loss - baseline_row_loss

        fires = (h_query[:, feat_idx] > 0).cpu().numpy()
        n_fire = int(fires.sum())

        row_feature_drops[:, i] = row_importance
        feature_n_firing[i] = n_fire

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_features - i - 1) / rate
            logger.info(
                f"  [{i+1}/{n_features}] feat={feat_idx} "
                f"drop={row_feature_drops[:, i].mean():+.4f} fire={n_fire}/{n_query} "
                f"({rate:.1f} feat/s, ETA {eta:.0f}s)"
            )

    # Count features that fire on at least one row
    n_active = int((h_full[:, alive_features] > 0).any(dim=0).sum().item())

    return {
        "baseline_preds": baseline_preds_np,
        "baseline_metric": baseline_metric,
        "metric_name": metric_name,
        "feature_indices": np.array(alive_features),
        "row_feature_drops": row_feature_drops,
        "feature_n_firing": feature_n_firing,
        "n_query": n_query,
        "n_active_features": n_active,
        "y_query": np.asarray(y_query),
    }


# ── Tabula-8B single-feature sweep ──────────────────────────────────────────


def sweep_tabula8b(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    sae: torch.nn.Module,
    alive_features: List[int],
    extraction_layer: int,
    device: str = "cuda",
    task: str = "classification",
    data_mean: Optional[torch.Tensor] = None,
    data_std: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Sweep single-feature ablation for Tabula-8B (row-first).

    Architecture:
      1. One baseline pass per query row → hidden states + baseline preds
      2. SAE encode all rows → firing map (which concepts fire on which rows)
      3. Row-first LOO: for each row, ablate only its firing concepts
      4. Rows sharing identical firing sets are grouped (same delta)

    Cost: sum(n_firing_per_row) forward passes, not n_features × n_query.
    With topk=128, worst case is 128 × n_query.

    Uses training-mean centering for SAE.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from collections import defaultdict

    import os
    model_path = "/data/models/tabula-8b"
    if not os.path.isdir(model_path):
        model_path = "mlfoundations/tabula-8b"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )
    llm.eval()
    llm_layers = llm.model.layers

    X_context = np.asarray(X_context, dtype=np.float32)
    y_context = np.asarray(y_context)
    X_query = np.asarray(X_query, dtype=np.float32)
    n_classes = len(np.unique(y_context))
    n_query = len(y_query)

    feature_names = [f"f{i}" for i in range(X_context.shape[1])]

    # Serialize context (shared prefix)
    max_ctx = min(32, len(X_context))
    ctx_lines = []
    for row, label in zip(X_context[:max_ctx], y_context[:max_ctx]):
        parts = []
        for name, val in zip(feature_names, row):
            if not (isinstance(val, float) and np.isnan(val)):
                parts.append(f"the {name} is {val}")
        ctx_lines.append(", ".join(parts) + f", the target is {label}")
    ctx_text = "\n".join(ctx_lines)

    # Pre-compute class token IDs for classification
    if task == "classification":
        class_token_ids = [
            tokenizer.encode(str(c), add_special_tokens=False)[0]
            for c in range(n_classes)
        ]

    def _forward_row(row_idx, delta_row=None):
        """Single-row forward pass, optionally injecting delta at last token."""
        row = X_query[row_idx]
        parts = [f"the {name} is {val}" for name, val in zip(feature_names, row)
                 if not (isinstance(val, float) and np.isnan(val))]
        query_text = ", ".join(parts)
        full_text = f"{ctx_text}\n{query_text}, the target is"

        inputs = tokenizer(
            full_text, return_tensors="pt",
            truncation=True, max_length=8000,
        ).to(llm.device)

        if delta_row is not None:
            def modify_hook(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                if isinstance(out, torch.Tensor):
                    out = out.clone()
                    out[0, -1, :] += delta_row.to(out.dtype)
                    if isinstance(output, tuple):
                        return (out,) + output[1:]
                    return out
                return output

            handle = llm_layers[extraction_layer].register_forward_hook(modify_hook)
        else:
            handle = None

        try:
            with torch.no_grad():
                outputs = llm(**inputs)
                logits = outputs.logits[0, -1, :]
        finally:
            if handle is not None:
                handle.remove()

        if task == "classification":
            probs = torch.softmax(logits[class_token_ids].float(), dim=0)
            return probs.cpu().numpy()
        else:
            token = tokenizer.decode(logits.argmax().item()).strip()
            try:
                return float(token)
            except ValueError:
                return 0.0

    # --- Pass 1: Capture hidden states + baseline predictions per row ---
    logger.info(f"Tabula-8B sweep: computing baselines for {n_query} rows...")
    all_emb_list = []
    baseline_preds_list = []

    for row_idx in range(n_query):
        captured = {}

        def capture_hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            if isinstance(out, torch.Tensor):
                captured["hidden"] = out.detach()

        handle = llm_layers[extraction_layer].register_forward_hook(capture_hook)
        try:
            preds = _forward_row(row_idx, delta_row=None)
        finally:
            handle.remove()

        # Last-token hidden state: (4096,)
        all_emb_list.append(captured["hidden"][0, -1, :].float())
        baseline_preds_list.append(preds)

        if (row_idx + 1) % 50 == 0:
            logger.info(f"  Tabula-8B baselines: {row_idx + 1}/{n_query}")

    all_emb = torch.stack(all_emb_list, dim=0)  # (n_query, 4096)
    baseline_preds_np = np.array(baseline_preds_list)

    # --- Step 2: SAE encode → firing map ---
    with torch.no_grad():
        x_norm = all_emb
        if data_mean is not None:
            x_norm = (x_norm - data_mean) / data_std
        h_full = sae.encode(x_norm)
        recon_full = sae.decode(h_full)

    baseline_metric, metric_name = compute_importance_metric(y_query, baseline_preds_np, task)
    baseline_row_loss = compute_per_row_loss(y_query, baseline_preds_np, task)

    # Build alive feature index mapping: position in alive_features list
    alive_set = set(alive_features)
    feat_to_pos = {f: i for i, f in enumerate(alive_features)}

    # Per-row firing map: which alive features fire on each row
    row_firing = []  # row_firing[row_idx] = list of alive feature indices
    total_forward_passes = 0
    for row_idx in range(n_query):
        firing = [f for f in alive_features if h_full[row_idx, f].item() > 0]
        row_firing.append(firing)
        total_forward_passes += len(firing)

    logger.info(
        f"Tabula-8B firing map: {total_forward_passes} forward passes "
        f"({total_forward_passes / n_query:.0f} avg features/row, "
        f"vs {len(alive_features)} alive features)"
    )

    # Group rows by identical firing sets for delta reuse
    firing_groups = defaultdict(list)  # frozenset(firing) → [row_indices]
    for row_idx, firing in enumerate(row_firing):
        key = frozenset(firing)
        firing_groups[key].append(row_idx)

    n_unique = len(firing_groups)
    logger.info(
        f"  {n_unique} unique firing patterns across {n_query} rows "
        f"({n_query / n_unique:.1f}x reuse)"
    )

    # --- Step 3: Row-first LOO ablation ---
    n_features = len(alive_features)
    row_feature_drops = np.zeros((n_query, n_features))
    feature_n_firing = np.zeros(n_features, dtype=int)

    # Count firings across all rows
    for row_idx in range(n_query):
        for f in row_firing[row_idx]:
            feature_n_firing[feat_to_pos[f]] += 1

    t0 = time.time()
    n_done = 0

    for group_idx, (firing_key, group_rows) in enumerate(firing_groups.items()):
        firing_list = sorted(firing_key)
        if not firing_list:
            continue

        # For each feature that fires in this group, compute delta once
        # then apply to all rows in the group
        for feat_idx in firing_list:
            feat_pos = feat_to_pos[feat_idx]

            with torch.no_grad():
                h_ablated = h_full[group_rows].clone()
                h_ablated[:, feat_idx] = 0.0
                recon_ablated = sae.decode(h_ablated)
                recon_orig = recon_full[group_rows]
                delta = (recon_ablated - recon_orig) * data_std

            # Forward pass for each row in the group
            for i, row_idx in enumerate(group_rows):
                delta_row = delta[i]
                preds = _forward_row(row_idx, delta_row=delta_row)
                preds_np = np.atleast_1d(np.asarray(preds))
                ablated_loss = compute_per_row_loss(
                    y_query[row_idx:row_idx+1], preds_np.reshape(1, -1), task
                )
                row_feature_drops[row_idx, feat_pos] = float(
                    ablated_loss[0] - baseline_row_loss[row_idx]
                )

            n_done += len(group_rows)

        if (group_idx + 1) % 10 == 0 or group_idx == 0:
            elapsed = time.time() - t0
            pct = n_done / total_forward_passes * 100 if total_forward_passes > 0 else 0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (total_forward_passes - n_done) / rate if rate > 0 else 0
            logger.info(
                f"  group {group_idx+1}/{n_unique}: "
                f"{n_done}/{total_forward_passes} passes ({pct:.0f}%) "
                f"({rate:.1f} pass/s, ETA {eta:.0f}s)"
            )

    # Final progress
    elapsed = time.time() - t0
    logger.info(
        f"  Tabula-8B sweep complete: {total_forward_passes} passes in {elapsed:.0f}s "
        f"({total_forward_passes / elapsed:.1f} pass/s)"
    )

    # Count features that fire on at least one row
    n_active = int((feature_n_firing > 0).sum())

    return {
        "baseline_preds": baseline_preds_np,
        "baseline_metric": baseline_metric,
        "metric_name": metric_name,
        "feature_indices": np.array(alive_features),
        "row_feature_drops": row_feature_drops,
        "feature_n_firing": feature_n_firing,
        "n_query": n_query,
        "n_active_features": n_active,
        "y_query": np.asarray(y_query),
    }


# ── HyperFast single-feature sweep ──────────────────────────────────────────


def sweep_hyperfast(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    sae: torch.nn.Module,
    alive_features: List[int],
    extraction_layer: int,
    device: str = "cuda",
    task: str = "classification",
    data_mean: Optional[torch.Tensor] = None,
    data_std: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Sweep single-feature ablation for HyperFast.

    Uses forward_main_network() from hyperfast to get penultimate activations
    (same code path as embedding extraction). Tail replay is just the output
    layer. Averages over ensemble members.

    Classification only — raises ValueError if task is regression.
    """
    import torch.nn.functional as F
    from hyperfast.hyperfast import (
        forward_main_network, transform_data_for_main_network,
    )
    from models.hyperfast_embeddings import HyperFastEmbeddingExtractor

    if task == "regression":
        raise ValueError("HyperFast does not support regression")

    extractor = HyperFastEmbeddingExtractor(device=device)
    extractor.load_model()
    X_ctx_arr = np.asarray(X_context, dtype=np.float32)
    y_ctx_clean = np.asarray(y_context, dtype=np.int64)
    extractor._model.fit(X_ctx_arr, y_ctx_clean)
    hf_clf = extractor._model

    n_query = len(y_query)
    X_query_t = torch.tensor(X_query, dtype=torch.float32).to(device)

    # Collect intermediates and baseline outputs across ensemble members
    main_networks = []
    intermediates = []
    baseline_outputs = []

    for jj in range(len(hf_clf._main_networks)):
        main_network = hf_clf._move_to_device(hf_clf._main_networks[jj])
        rf = hf_clf._move_to_device(hf_clf._rfs[jj])
        pca = hf_clf._move_to_device(hf_clf._pcas[jj])

        if hf_clf.feature_bagging:
            X_b = X_query_t[:, hf_clf.selected_features[jj]]
        else:
            X_b = X_query_t

        X_transformed = transform_data_for_main_network(
            X=X_b, cfg=hf_clf._cfg, rf=rf, pca=pca,
        )

        with torch.no_grad():
            outputs, intermediate = forward_main_network(
                X_transformed, main_network,
            )
            baseline_outputs.append(F.softmax(outputs, dim=1).cpu().numpy())

        main_networks.append(main_network)
        intermediates.append(intermediate.detach().clone())

    baseline_preds_np = np.mean(baseline_outputs, axis=0)

    # Mean embedding across ensemble members for SAE
    all_emb = torch.stack(intermediates, dim=0).mean(dim=0)  # (n_query, H)

    # Pre-compute SAE encoding with per-dataset normalization
    with torch.no_grad():
        x_norm = all_emb
        if data_mean is not None:
            x_norm = (x_norm - data_mean) / data_std
        h_full = sae.encode(x_norm)
        recon_full = sae.decode(h_full)

    baseline_metric, metric_name = compute_importance_metric(y_query, baseline_preds_np, task)
    baseline_row_loss = compute_per_row_loss(y_query, baseline_preds_np, task)

    # All rows are query rows
    h_query = h_full

    def _forward_tail(ensemble_idx, x):
        """Forward through the output layer (last layer of generated MLP).

        HyperFast stores weights as (in_features, out_features) and uses
        torch.mm(x, weight) — NOT F.linear which expects (out, in).
        """
        weight, bias = main_networks[ensemble_idx][-1]
        weight = hf_clf._move_to_device(weight)
        bias = hf_clf._move_to_device(bias)
        with torch.no_grad():
            logits = torch.mm(x, weight) + bias
        return F.softmax(logits, dim=1).cpu().numpy()

    # --- Sweep: ablate one feature at a time ---
    n_features = len(alive_features)
    row_feature_drops = np.zeros((n_query, n_features))
    feature_n_firing = np.zeros(n_features, dtype=int)

    t0 = time.time()
    for i, feat_idx in enumerate(alive_features):
        with torch.no_grad():
            h_ablated = h_full.clone()
            h_ablated[:, feat_idx] = 0.0
            recon_ablated = sae.decode(h_ablated)
            delta = (recon_ablated - recon_full) * data_std  # denormalize to raw space

        # Replay tail with delta for each ensemble member, average
        ensemble_preds = []
        for jj in range(len(main_networks)):
            x = intermediates[jj].clone() + delta
            ensemble_preds.append(_forward_tail(jj, x))
        preds_np = np.mean(ensemble_preds, axis=0)

        ablated_row_loss = compute_per_row_loss(y_query, preds_np, task)
        row_importance = ablated_row_loss - baseline_row_loss

        fires = (h_query[:, feat_idx] > 0).cpu().numpy()
        n_fire = int(fires.sum())

        row_feature_drops[:, i] = row_importance
        feature_n_firing[i] = n_fire

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_features - i - 1) / rate
            logger.info(
                f"  [{i+1}/{n_features}] feat={feat_idx} "
                f"drop={row_feature_drops[:, i].mean():+.4f} fire={n_fire}/{n_query} "
                f"({rate:.1f} feat/s, ETA {eta:.0f}s)"
            )

    # Count features that fire on at least one row
    n_active = int((h_full[:, alive_features] > 0).any(dim=0).sum().item())

    return {
        "baseline_preds": baseline_preds_np,
        "baseline_metric": baseline_metric,
        "metric_name": metric_name,
        "feature_indices": np.array(alive_features),
        "row_feature_drops": row_feature_drops,
        "feature_n_firing": feature_n_firing,
        "n_query": n_query,
        "n_active_features": n_active,
        "y_query": np.asarray(y_query),
    }


# ── Dispatcher ───────────────────────────────────────────────────────────────

SWEEP_FN = {
    "tabpfn": sweep_tabpfn,
    "mitra": sweep_mitra,
    "tabdpt": sweep_tabdpt,
    "tabicl": sweep_tabicl,
    "tabicl_v2": sweep_tabicl_v2,
    "carte": sweep_carte,
    "tabula8b": sweep_tabula8b,
    "hyperfast": sweep_hyperfast,
}


def sweep_concept_importance(
    model_key: str,
    dataset: str,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    device: str = "cuda",
    task: str = "classification",
) -> Dict:
    """Run single-feature ablation sweep for all alive features.

    Returns dict with feature_indices, row_feature_drops, feature_labels, baseline_metric.
    """
    if model_key not in SWEEP_FN:
        raise ValueError(f"Unsupported model: {model_key}. Choose from {list(SWEEP_FN.keys())}")

    sae, config = load_sae(model_key, device=device)
    extraction_layer = get_extraction_layer(model_key)
    data_mean, data_std = load_norm_stats(model_key, dataset, device=device)
    alive_features = get_alive_features(model_key)
    feature_labels = get_feature_labels(model_key)

    logger.info(
        f"Sweeping {len(alive_features)} alive features for {model_key} "
        f"(SAE {config.input_dim}->{config.hidden_dim}, extract@L{extraction_layer})"
    )

    result = SWEEP_FN[model_key](
        X_context=X_context,
        y_context=y_context,
        X_query=X_query,
        y_query=y_query,
        sae=sae,
        alive_features=alive_features,
        extraction_layer=extraction_layer,
        device=device,
        task=task,
        data_mean=data_mean,
        data_std=data_std,
    )

    # Attach labels
    result["feature_labels"] = [
        feature_labels.get(idx, "unknown") for idx in alive_features
    ]

    return result


# ── Concept-level analysis ───────────────────────────────────────────────────


def get_matryoshka_bands(model_key: str, sae_dir: Path = None) -> Dict[str, int]:
    """Get Matryoshka band boundaries for a model's SAE.

    Bands are proportional: [h/16, h/8, h/4, h/2, h] where h = hidden_dim.
    Returns dict mapping band name to upper boundary (exclusive).
    """
    import torch as _torch

    if sae_dir is None:
        sae_dir = PROJECT_ROOT / "output" / f"sae_tabarena_sweep_round{DEFAULT_SAE_ROUND}"

    ckpt_path = sae_dir / model_key / SAE_FILENAME
    ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    h = config.hidden_dim if hasattr(config, "hidden_dim") else config["hidden_dim"]

    boundaries = {"S1": h // 16, "S2": h // 8, "S3": h // 4, "S4": h // 2, "S5": h}
    return boundaries


def feature_to_band(feat_idx: int, bands: Dict[str, int]) -> str:
    """Map a feature index to its Matryoshka band."""
    for name in ["S1", "S2", "S3", "S4", "S5"]:
        if feat_idx < bands[name]:
            return name
    return "S5"


def analyze_importance(
    importance_path: Path,
    model_key: str,
    top_n: int = 5,
    labels_path: Path = DEFAULT_CONCEPT_LABELS,
) -> Dict:
    """Analyze concept importance results with cross-model and Matryoshka metadata.

    Loads a saved importance JSON and enriches each feature with:
    - Matryoshka band (S1-S5)
    - Concept group membership and cross-model coverage
    - Top PyMFE probes for the concept group
    - Cumulative group-level importance

    Args:
        importance_path: Path to saved importance JSON (from sweep)
        model_key: Model key (e.g. 'tabdpt')
        top_n: Number of top unique concept groups to return
        labels_path: Path to concept labels JSON (default: MNN-only hierarchy)

    Returns:
        Dict with 'baseline_metric', 'dataset', 'model', 'bands', 'top_groups',
        and 'summary' fields.
    """
    from collections import Counter

    with open(importance_path) as f:
        importance = json.load(f)

    with open(labels_path) as f:
        concept_data = json.load(f)

    bands = get_matryoshka_bands(model_key)
    label_key = MODEL_KEY_TO_LABEL_KEY.get(model_key, model_key)
    feature_lookup = concept_data["feature_lookup"][label_key]
    concept_groups = concept_data["concept_groups"]

    # Build feature list from new format (feature_indices + mean_feature_drops)
    # or old format (features list with per-feature dicts)
    if "feature_indices" in importance and "mean_feature_drops" in importance:
        features = []
        for i, idx in enumerate(importance["feature_indices"]):
            features.append({
                "index": idx,
                "drop": importance["mean_feature_drops"][i],
                "n_firing": importance["feature_n_firing"][i],
                "label": importance["feature_labels"][i],
            })
    else:
        features = importance["features"]

    # Enrich each feature with band and group info
    enriched = []
    for feat in features:
        idx = feat["index"]
        info = feature_lookup.get(str(idx), {})
        enriched.append({
            **feat,
            "band": feature_to_band(idx, bands),
            "group_id": info.get("group_id"),
            "category": info.get("category", "unknown"),
        })

    # Deduplicate to top feature per unique concept group
    seen_groups = set()
    top_by_group = []
    for f in enriched:
        gid = f["group_id"]
        if gid is not None and gid not in seen_groups and f["drop"] > 0:
            seen_groups.add(gid)
            top_by_group.append(f)

    # Build detailed group info for top N
    top_groups = []
    for f in top_by_group[:top_n]:
        gid = str(f["group_id"])
        g = concept_groups.get(gid, {})
        members = g.get("members", [])
        models_in_group = Counter(m[0] for m in members)

        # All features in this group from the importance results
        group_features = [e for e in enriched if e["group_id"] == int(gid)]
        positive_drops = [e["drop"] for e in group_features if e["drop"] > 0]
        cumulative_drop = sum(positive_drops)

        # Band distribution within group
        band_counts = Counter(e["band"] for e in group_features)

        probes = g.get("top_probes", [])

        top_groups.append({
            "rank": len(top_groups) + 1,
            "group_id": int(gid),
            "label": g.get("label", "unknown"),
            "top_feature": f["index"],
            "top_drop": f["drop"],
            "band": f["band"],
            "n_features_in_group": len(group_features),
            "n_positive_drop": len(positive_drops),
            "cumulative_drop": cumulative_drop,
            "n_models": g.get("n_models", len(models_in_group)),
            "models": dict(models_in_group),
            "tier": g.get("tier"),
            "mean_r2": g.get("mean_r2", 0),
            "top_probes": [
                {"name": p[0], "n_models": p[1], "coeff": p[2]}
                for p in probes[:5]
            ],
            "band_distribution": dict(band_counts),
        })

    # Summary statistics
    total_positive = sum(f["drop"] for f in enriched if f["drop"] > 0)
    top_cumulative = sum(g["cumulative_drop"] for g in top_groups)

    # Band distribution of all important features
    important_bands = Counter(
        f["band"] for f in enriched if f["drop"] > 0.001
    )

    # Causal chain analysis (if regression + pymfe data available)
    causal_chain = []
    if DEFAULT_CONCEPT_REGRESSION.exists() and DEFAULT_PYMFE_CACHE.exists():
        causal_chain = analyze_causal_chain(
            importance_path, model_key, importance["dataset"],
            top_n=top_n, n_probes=5,
        )

    return {
        "model": model_key,
        "dataset": importance["dataset"],
        "task": importance["task"],
        "baseline_metric": importance.get("baseline_metric", importance.get("baseline_acc")),
        "baseline_metric_name": importance.get("baseline_metric_name", importance.get("metric_name", "metric")),
        "bands": bands,
        "top_groups": top_groups,
        "causal_chain": causal_chain,
        "summary": {
            "total_positive_drop": total_positive,
            "top_n_cumulative_drop": top_cumulative,
            "top_n_fraction": top_cumulative / total_positive if total_positive > 0 else 0,
            "n_features_positive": sum(1 for f in enriched if f["drop"] > 0),
            "n_features_above_1pp": sum(1 for f in enriched if f["drop"] > 0.01),
            "important_band_distribution": dict(important_bands),
        },
    }


def print_analysis(analysis: Dict) -> None:
    """Pretty-print a concept importance analysis."""
    print(f"\n{'='*100}")
    print(f"Concept Importance Analysis: {analysis['model']} on {analysis['dataset']}")
    print(f"{'='*100}")
    metric = analysis.get("metric_name", analysis.get("baseline_metric_name", "metric"))
    print(f"Baseline {metric}: {analysis['baseline_metric']:.3f}")
    print(f"Matryoshka bands: {analysis['bands']}")

    s = analysis["summary"]
    print(f"\nSummary:")
    print(f"  Features with positive drop: {s['n_features_positive']}")
    print(f"  Features >1pp drop:          {s['n_features_above_1pp']}")
    print(f"  Band distribution (>0.1pp):  {s['important_band_distribution']}")

    print(f"\nTop {len(analysis['top_groups'])} concept groups "
          f"(explain {s['top_n_fraction']*100:.1f}% of total positive drop):")

    for g in analysis["top_groups"]:
        models_str = ", ".join(
            f"{m}({n})" for m, n in sorted(g["models"].items())
        )
        probes_str = ", ".join(
            f"{p['name']}({p['coeff']:+.2f})" for p in g["top_probes"][:3]
        )

        print(f"\n  #{g['rank']} Group {g['group_id']}: {g['label']}")
        print(f"     Band: {g['band']} | Top drop: {g['top_drop']:+.4f} (feat {g['top_feature']})")
        print(f"     {g['n_features_in_group']} features in group, "
              f"{g['n_positive_drop']} with positive drop, "
              f"cumulative: {g['cumulative_drop']:+.4f}")
        print(f"     Cross-model: {g['n_models']} models — {models_str}")
        print(f"     Tier: {g['tier']} | Mean R²: {g['mean_r2']:.3f}")
        print(f"     Top probes: {probes_str}")

    # Print causal chain if available
    chain = analysis.get("causal_chain", [])
    if chain:
        print_causal_chain(chain, analysis["dataset"])

        # Summary: overall alignment rate across all features/probes
        total_aligned = sum(f["n_aligned"] for f in chain)
        total_checked = sum(f["n_probes_checked"] for f in chain)
        if total_checked > 0:
            print(f"\n  Overall probe alignment: {total_aligned}/{total_checked} "
                  f"({total_aligned/total_checked*100:.0f}%)")


# ── Causal chain analysis ────────────────────────────────────────────────────


def load_probe_percentiles(
    pymfe_cache_path: Path = DEFAULT_PYMFE_CACHE,
) -> Dict[str, Dict[str, float]]:
    """Compute percentile rank of each dataset for each PyMFE probe.

    Returns:
        Dict mapping probe_name -> {dataset_name: percentile_0_to_1}.
        NaN values are excluded from ranking.
    """
    import math

    with open(pymfe_cache_path) as f:
        cache = json.load(f)

    # Collect all probe names
    all_probes = set()
    for ds_vals in cache.values():
        all_probes.update(ds_vals.keys())

    percentiles = {}
    for probe in all_probes:
        # Gather (dataset, value) pairs, skip NaN/None
        vals = []
        for ds, ds_vals in cache.items():
            v = ds_vals.get(probe)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                vals.append((ds, v))
        if not vals:
            continue
        # Sort by value and assign percentile ranks
        vals.sort(key=lambda x: x[1])
        n = len(vals)
        pct = {}
        for rank, (ds, _) in enumerate(vals):
            pct[ds] = rank / max(n - 1, 1)
        percentiles[probe] = pct

    return percentiles


def analyze_causal_chain(
    importance_path: Path,
    model_key: str,
    dataset_name: str,
    top_n: int = 5,
    n_probes: int = 5,
    regression_path: Path = DEFAULT_CONCEPT_REGRESSION,
    pymfe_cache_path: Path = DEFAULT_PYMFE_CACHE,
) -> List[Dict]:
    """For top important features, trace the causal chain:

    Dataset property (PyMFE) → SAE activation (feature fires) → Prediction (ablation drops)

    For each feature's top probes, checks whether the dataset's actual value
    on that probe aligns with the probe coefficient direction:
    - Positive coeff + high percentile (>0.6) = ALIGNED
    - Negative coeff + low percentile (<0.4) = ALIGNED
    - Otherwise = OPPOSITE or NEUTRAL

    Returns list of dicts, one per top feature, with probe alignment details.
    """
    import math

    with open(importance_path) as f:
        importance = json.load(f)

    with open(regression_path) as f:
        regression = json.load(f)

    with open(pymfe_cache_path) as f:
        pymfe_cache = json.load(f)

    label_key = MODEL_KEY_TO_LABEL_KEY.get(model_key, model_key)
    per_feature = regression["models"][label_key]["per_feature"]
    ds_pymfe = pymfe_cache.get(dataset_name, {})

    if not ds_pymfe:
        logger.warning(f"Dataset '{dataset_name}' not found in PyMFE cache")
        return []

    # Pre-compute percentile ranks
    percentiles = load_probe_percentiles(pymfe_cache_path)

    # Get top features by ablation drop
    features_sorted = sorted(importance["features"], key=lambda x: -x["drop"])

    results = []
    for feat in features_sorted[:top_n]:
        if feat["drop"] <= 0:
            break

        feat_idx = str(feat["index"])
        feat_reg = per_feature.get(feat_idx, {})
        top_probes = feat_reg.get("top_probes", [])
        feat_r2 = feat_reg.get("r2", 0)

        probe_details = []
        n_aligned = 0
        n_opposite = 0

        for probe_name, coeff, rank in top_probes[:n_probes]:
            ds_value = ds_pymfe.get(probe_name)
            pct_lookup = percentiles.get(probe_name, {})
            ds_pct = pct_lookup.get(dataset_name)

            if ds_value is None or ds_pct is None:
                alignment = "MISSING"
            elif isinstance(ds_value, float) and math.isnan(ds_value):
                alignment = "NaN"
            else:
                if coeff > 0 and ds_pct > 0.6:
                    alignment = "ALIGNED"
                    n_aligned += 1
                elif coeff < 0 and ds_pct < 0.4:
                    alignment = "ALIGNED"
                    n_aligned += 1
                elif coeff > 0 and ds_pct < 0.4:
                    alignment = "OPPOSITE"
                    n_opposite += 1
                elif coeff < 0 and ds_pct > 0.6:
                    alignment = "OPPOSITE"
                    n_opposite += 1
                else:
                    alignment = "NEUTRAL"

            probe_details.append({
                "probe": probe_name,
                "coeff": coeff,
                "rank": rank,
                "ds_value": ds_value,
                "ds_percentile": ds_pct,
                "alignment": alignment,
            })

        results.append({
            "feature_index": feat["index"],
            "ablation_drop": feat["drop"],
            "label": feat.get("label", "unknown"),
            "probe_r2": feat_r2,
            "n_aligned": n_aligned,
            "n_opposite": n_opposite,
            "n_probes_checked": len(probe_details),
            "probes": probe_details,
        })

    return results


def print_causal_chain(chain_results: List[Dict], dataset_name: str) -> None:
    """Pretty-print causal chain analysis."""
    if not chain_results:
        print("  No causal chain data available.")
        return

    print(f"\n{'─'*100}")
    print(f"CAUSAL CHAIN: {dataset_name}")
    print(f"  Dataset property (PyMFE) → SAE feature fires → Ablation drops accuracy")
    print(f"{'─'*100}")

    for feat in chain_results:
        aligned_frac = feat["n_aligned"] / max(feat["n_probes_checked"], 1)
        print(f"\n  Feature {feat['feature_index']}: {feat['label']}")
        print(f"    Ablation drop: {feat['ablation_drop']:+.4f} | "
              f"Probe R²: {feat['probe_r2']:.3f} | "
              f"Alignment: {feat['n_aligned']}/{feat['n_probes_checked']} "
              f"({aligned_frac*100:.0f}%)")

        for p in feat["probes"]:
            pct_str = f"{p['ds_percentile']:.0%}" if p["ds_percentile"] is not None else "N/A"
            val_str = f"{p['ds_value']:.4g}" if isinstance(p.get("ds_value"), (int, float)) else "N/A"
            marker = {"ALIGNED": "+", "OPPOSITE": "x", "NEUTRAL": "~", "MISSING": "?", "NaN": "?"}
            print(f"    [{marker.get(p['alignment'], '?')}] {p['probe']:<25s} "
                  f"coeff={p['coeff']:+6.2f}  value={val_str:>10s}  "
                  f"pctl={pct_str:>4s}  {p['alignment']}")


# ── Pairwise concept bookkeeping ─────────────────────────────────────────────

DEFAULT_MNN_PATH = PROJECT_ROOT / "output" / "sae_feature_matching_mnn_t0.001_n500.json"
DEFAULT_CORR_DIR = PROJECT_ROOT / "output" / "sae_cross_correlations"


def compare_concept_importance(
    model_a: str,
    model_b: str,
    dataset: str,
    importance_dir: Path = None,
    labels_path: Path = DEFAULT_CONCEPT_LABELS,
    mnn_path: Path = DEFAULT_MNN_PATH,
    corr_dir: Path = DEFAULT_CORR_DIR,
) -> Dict:
    """Compare two models' concept importance on a dataset.

    Matching is driven by the cross-correlation matrix (row-level detail).
    Labels come from the concept hierarchy (for display only).
    For each important feature in A, finds its best correlate in B and reports
    the importance differential.

    Features are classified as:
    - MNN-matched: mutual nearest neighbor pair (highest confidence)
    - Correlated: best |r| >= 0.20 but not MNN
    - Unmatched: no correlate above threshold

    Returns structured dict with per-feature comparisons and aggregate tallies.
    """
    import numpy as _np

    if importance_dir is None:
        importance_dir = PROJECT_ROOT / "output" / "interventions" / dataset / "importance"

    # Load importance results
    with open(importance_dir / f"{model_a}.json") as f:
        imp_a = json.load(f)
    with open(importance_dir / f"{model_b}.json") as f:
        imp_b = json.load(f)

    # Build feature_idx -> drop lookup
    drop_a = {f["index"]: f["drop"] for f in imp_a["features"]}
    drop_b = {f["index"]: f["drop"] for f in imp_b["features"]}

    # Load labels (for display only)
    label_key_a = MODEL_KEY_TO_LABEL_KEY.get(model_a, model_a)
    label_key_b = MODEL_KEY_TO_LABEL_KEY.get(model_b, model_b)
    labels_a = get_feature_labels(model_a, labels_path)
    labels_b = get_feature_labels(model_b, labels_path)

    # Bands
    bands_a = get_matryoshka_bands(model_a)
    bands_b = get_matryoshka_bands(model_b)

    # Load MNN matches for this pair
    with open(mnn_path) as f:
        mnn_data = json.load(f)

    pair_key = None
    a_is_first = True
    for k in mnn_data["pairs"]:
        parts = k.split("__")
        if set(parts) == {label_key_a, label_key_b}:
            pair_key = k
            a_is_first = k.startswith(label_key_a)
            break

    mnn_pairs = {}  # feat_a -> (feat_b, r)
    mnn_pairs_rev = {}  # feat_b -> (feat_a, r)
    if pair_key:
        for m in mnn_data["pairs"][pair_key]["matches"]:
            fa = m["idx_a"] if a_is_first else m["idx_b"]
            fb = m["idx_b"] if a_is_first else m["idx_a"]
            mnn_pairs[fa] = (fb, abs(m["r"]))
            mnn_pairs_rev[fb] = (fa, abs(m["r"]))

    # Load cross-correlation matrix
    corr_matrix = None
    idx_to_pos_a = {}
    idx_to_pos_b = {}
    abs_indices_b = []

    for npz_name in [f"{label_key_a}__{label_key_b}", f"{label_key_b}__{label_key_a}"]:
        npz_path = corr_dir / f"{npz_name}.npz"
        if npz_path.exists():
            d = _np.load(npz_path)
            if str(d["model_a"]) == label_key_a:
                corr_matrix = d["corr_matrix"]
                indices_a = d["indices_a"]
                indices_b = d["indices_b"]
            else:
                corr_matrix = d["corr_matrix"].T
                indices_a = d["indices_b"]
                indices_b = d["indices_a"]
            idx_to_pos_a = {int(v): i for i, v in enumerate(indices_a)}
            idx_to_pos_b = {int(v): i for i, v in enumerate(indices_b)}
            abs_indices_b = [int(v) for v in indices_b]
            break

    # --- Build per-feature comparison ---
    # For each feature in A (sorted by drop), find its best correlate in B
    features_a = sorted(imp_a["features"], key=lambda x: -x["drop"])

    matched_features = []   # MNN-matched pairs
    correlated_features = []  # correlated but not MNN
    unmatched_a = []  # no good correlate in B

    matched_b_used = set()  # track which B features are accounted for

    for feat in features_a:
        fa = feat["index"]
        drop_fa = feat["drop"]
        band_a = feature_to_band(fa, bands_a)
        label_a = labels_a.get(fa, "?")

        if fa in mnn_pairs:
            # MNN matched
            fb, r = mnn_pairs[fa]
            drop_fb = drop_b.get(fb, 0.0)
            matched_features.append({
                "feat_a": fa, "feat_b": fb,
                "drop_a": drop_fa, "drop_b": drop_fb,
                "band_a": band_a, "band_b": feature_to_band(fb, bands_b),
                "label_a": label_a, "label_b": labels_b.get(fb, "?"),
                "r": r, "match_type": "mnn",
                "differential": drop_fa - drop_fb,
            })
            matched_b_used.add(fb)
        elif corr_matrix is not None and fa in idx_to_pos_a:
            # Find best available correlate (skip already-claimed B features)
            pos_a = idx_to_pos_a[fa]
            corr_row = corr_matrix[pos_a]
            sorted_j = np.argsort(-corr_row)
            best_r = 0.0
            fb = -1
            for j_cand in sorted_j:
                fb_cand = abs_indices_b[j_cand]
                if fb_cand not in matched_b_used:
                    best_r = float(corr_row[j_cand])
                    fb = fb_cand
                    break
            drop_fb = drop_b.get(fb, 0.0)

            if best_r >= 0.20:
                correlated_features.append({
                    "feat_a": fa, "feat_b": fb,
                    "drop_a": drop_fa, "drop_b": drop_fb,
                    "band_a": band_a, "band_b": feature_to_band(fb, bands_b),
                    "label_a": label_a, "label_b": labels_b.get(fb, "?"),
                    "r": best_r, "match_type": "correlated",
                    "differential": drop_fa - drop_fb,
                })
                matched_b_used.add(fb)
            else:
                unmatched_a.append({
                    "feat_a": fa, "drop_a": drop_fa,
                    "band_a": band_a, "label_a": label_a,
                    "best_r": best_r,
                })
        else:
            unmatched_a.append({
                "feat_a": fa, "drop_a": drop_fa,
                "band_a": band_a, "label_a": label_a,
                "best_r": 0.0,
            })

    # B features not matched to any A feature
    unmatched_b = []
    for feat in sorted(imp_b["features"], key=lambda x: -x["drop"]):
        fb = feat["index"]
        if fb not in matched_b_used:
            unmatched_b.append({
                "feat_b": fb, "drop_b": feat["drop"],
                "band_b": feature_to_band(fb, bands_b),
                "label_b": labels_b.get(fb, "?"),
            })

    # --- Tallies ---
    from collections import defaultdict

    def _band_tally(items, drop_key, band_key):
        t = defaultdict(float)
        for item in items:
            d = item.get(drop_key, 0.0)
            if d > 0:
                t[item[band_key]] += d
        return dict(t)

    # Matched (MNN + correlated): importance that both models share
    all_matched = matched_features + correlated_features
    matched_adv_a = sum(max(0, m["differential"]) for m in all_matched)
    matched_adv_b = sum(max(0, -m["differential"]) for m in all_matched)

    # Unmatched: importance unique to each model
    unique_a_total = sum(max(0, u["drop_a"]) for u in unmatched_a)
    unique_b_total = sum(max(0, u.get("drop_b", 0)) for u in unmatched_b)

    result = {
        "model_a": model_a,
        "model_b": model_b,
        "dataset": dataset,
        "baseline_a": imp_a.get("baseline_metric", imp_a.get("baseline_acc")),
        "baseline_b": imp_b.get("baseline_metric", imp_b.get("baseline_acc")),
        "metric": imp_a.get("metric_name", "auc"),
        "n_active_a": imp_a.get("n_active_features", -1),
        "n_active_b": imp_b.get("n_active_features", -1),
        "n_mnn_matched": len(matched_features),
        "n_correlated": len(correlated_features),
        "n_unmatched_a": len(unmatched_a),
        "n_unmatched_b": len(unmatched_b),
        "matched_features": matched_features,
        "correlated_features": correlated_features,
        "unmatched_a": unmatched_a,
        "unmatched_b": unmatched_b,
        "tally": {
            "matched_advantage_a": matched_adv_a,
            "matched_advantage_b": matched_adv_b,
            "unique_advantage_a": unique_a_total,
            "unique_advantage_b": unique_b_total,
            "total_advantage_a": matched_adv_a + unique_a_total,
            "total_advantage_b": matched_adv_b + unique_b_total,
            "band_matched_a": _band_tally(
                [m for m in all_matched if m["differential"] > 0], "differential", "band_a"),
            "band_matched_b": _band_tally(
                [{"differential": -m["differential"], "band_b": m["band_b"]}
                 for m in all_matched if m["differential"] < 0],
                "differential", "band_b"),
            "band_unique_a": _band_tally(unmatched_a, "drop_a", "band_a"),
            "band_unique_b": _band_tally(unmatched_b, "drop_b", "band_b"),
        },
    }
    return result


def print_concept_comparison(result: Dict) -> None:
    """Pretty-print a pairwise concept importance comparison."""
    a = result["model_a"]
    b = result["model_b"]
    disp_a = MODEL_KEY_TO_LABEL_KEY.get(a, a)
    disp_b = MODEL_KEY_TO_LABEL_KEY.get(b, b)
    t = result["tally"]

    print(f"\n{'='*100}")
    print(f"Concept Bookkeeping: {disp_a} vs {disp_b} on {result['dataset']}")
    print(f"{'='*100}")
    print(f"  {disp_a}: baseline {result['metric']}={result['baseline_a']:.4f}, "
          f"{result['n_active_a']} active features")
    print(f"  {disp_b}: baseline {result['metric']}={result['baseline_b']:.4f}, "
          f"{result['n_active_b']} active features")
    print(f"  Feature matching: {result['n_mnn_matched']} MNN, "
          f"{result['n_correlated']} correlated, "
          f"{result['n_unmatched_a']} {disp_a}-only, "
          f"{result['n_unmatched_b']} {disp_b}-only")

    perf_gap = result["baseline_a"] - result["baseline_b"]
    print(f"\n{'─'*100}")
    print(f"TALLY (sum of single-feature ablation drops; not additive across features)")
    print(f"{'─'*100}")
    print(f"  Actual {result['metric']} gap: {disp_a} - {disp_b} = {perf_gap:+.4f}")
    print(f"  Matched features: {disp_a} advantage = {t['matched_advantage_a']:+.3f}, "
          f"{disp_b} advantage = {t['matched_advantage_b']:+.3f}")
    print(f"  Unmatched:        {disp_a} advantage = {t['unique_advantage_a']:+.3f}, "
          f"{disp_b} advantage = {t['unique_advantage_b']:+.3f}")

    # Band breakdown
    for label, key in [
        (f"{disp_a} advantage (matched)", "band_matched_a"),
        (f"{disp_b} advantage (matched)", "band_matched_b"),
        (f"{disp_a}-only", "band_unique_a"),
        (f"{disp_b}-only", "band_unique_b"),
    ]:
        bands = t.get(key, {})
        if any(v > 0 for v in bands.values()):
            print(f"\n  {label}:")
            for band in ["S1", "S2", "S3", "S4", "S5"]:
                v = bands.get(band, 0)
                if v > 0.001:
                    print(f"    {band}: {v:+.3f}")

    # Top matched features (by |differential|)
    all_matched = result["matched_features"] + result["correlated_features"]
    important = [m for m in all_matched if abs(m["differential"]) > 0.005]
    important.sort(key=lambda x: -abs(x["differential"]))

    if important:
        print(f"\n{'─'*100}")
        print(f"Top matched features (by importance differential):")
        print(f"{'─'*100}")
        print(f"  {'Type':>4} {'|r|':>5} {'Feat_A':>6} {'Drop_A':>7} "
              f"{'Feat_B':>6} {'Drop_B':>7} {'Diff':>7}  {'Band':>4}  Label")
        for m in important[:20]:
            mtype = "MNN" if m["match_type"] == "mnn" else "corr"
            winner = disp_a if m["differential"] > 0 else disp_b
            print(f"  {mtype:>4} {m['r']:>5.2f} {m['feat_a']:>6} {m['drop_a']:>+7.3f} "
                  f"{m['feat_b']:>6} {m['drop_b']:>+7.3f} "
                  f"{m['differential']:>+7.3f}  {m['band_a']:>4}  {m['label_a']}")

    # Top unmatched A features
    imp_unmatched_a = [u for u in result["unmatched_a"] if u["drop_a"] > 0.005]
    if imp_unmatched_a:
        print(f"\n{'─'*100}")
        print(f"{disp_a}-only features (no match in {disp_b}):")
        print(f"{'─'*100}")
        print(f"  {'Feat':>6} {'Drop':>7} {'Band':>4} {'Best_r':>6}  Label")
        for u in imp_unmatched_a[:15]:
            print(f"  {u['feat_a']:>6} {u['drop_a']:>+7.3f} {u['band_a']:>4} "
                  f"{u['best_r']:>6.3f}  {u['label_a']}")

    # Top unmatched B features
    imp_unmatched_b = [u for u in result["unmatched_b"] if u.get("drop_b", 0) > 0.005]
    if imp_unmatched_b:
        print(f"\n  {disp_b}-only features (no match in {disp_a}):")
        print(f"  {'Feat':>6} {'Drop':>7} {'Band':>4}  Label")
        for u in imp_unmatched_b[:15]:
            print(f"  {u['feat_b']:>6} {u.get('drop_b', 0):>+7.3f} {u['band_b']:>4}  "
                  f"{u['label_b']}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def load_tabarena_splits(dataset_name: str, task: str):
    """Load and split a TabArena dataset for intervention experiments."""
    from data.extended_loader import load_tabarena_dataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    result = load_tabarena_dataset(dataset_name, max_samples=1100)
    X, y, _ = result

    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)
        qf = min(500, int(len(y) * 0.45)) / len(y)
        try:
            X_ctx, X_q, y_ctx, y_q = train_test_split(
                X, y, test_size=qf, random_state=42, stratify=y)
        except ValueError:
            X_ctx, X_q, y_ctx, y_q = train_test_split(
                X, y, test_size=qf, random_state=42)
    else:
        n = len(X)
        ctx = min(600, int(n * 0.7))
        X_ctx, y_ctx = X[:ctx], y[:ctx]
        X_q, y_q = X[ctx:ctx+500], y[ctx:ctx+500]

    X_ctx, X_q = X_ctx[:600], X_q[:500]
    y_ctx, y_q = y_ctx[:600], y_q[:500]
    return X_ctx, y_ctx, X_q, y_q


def main():
    parser = argparse.ArgumentParser(description="Per-concept importance via single-feature ablation")
    parser.add_argument("--model", type=str, required=True, choices=list(SWEEP_FN.keys()))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top", type=int, default=20, help="Show top N features/groups")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument(
        "--analyze", action="store_true",
        help="Analyze existing results (skip sweep, load from --output or default path)"
    )
    parser.add_argument(
        "--compare", type=str, default=None, metavar="MODEL_B",
        help="Compare --model vs MODEL_B using existing importance JSONs"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.compare:
        result = compare_concept_importance(args.model, args.compare, args.dataset)
        print_concept_comparison(result)
        # Save comparison
        out_dir = PROJECT_ROOT / "output" / "interventions" / args.dataset / "comparison"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.model}_vs_{args.compare}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {out_path}")
        return

    from scripts.embeddings.extract_layer_embeddings import get_dataset_task
    task = get_dataset_task(args.dataset)

    # Determine output path
    output_path = args.output
    if output_path is None:
        output_dir = PROJECT_ROOT / "output" / "interventions" / args.dataset / "importance"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{args.model}.json")

    if args.analyze:
        # Analysis-only mode: load existing results
        analysis = analyze_importance(
            Path(output_path), args.model, top_n=args.top,
        )
        print_analysis(analysis)
        return

    # Full sweep mode
    X_ctx, y_ctx, X_q, y_q = load_tabarena_splits(args.dataset, task)

    logger.info(f"Dataset: {args.dataset} ({task})")
    logger.info(f"Context: {X_ctx.shape}, Query: {X_q.shape}")
    logger.info(f"Model: {args.model}")
    logger.info("")

    t0 = time.time()
    result = sweep_concept_importance(
        model_key=args.model,
        dataset=args.dataset,
        X_context=X_ctx,
        y_context=y_ctx,
        X_query=X_q,
        y_query=y_q,
        device=args.device,
        task=task,
    )
    elapsed = time.time() - t0

    # Display results
    metric_name = result.get("metric_name", "accuracy")
    logger.info(f"\nBaseline {metric_name}: {result['baseline_metric']:.4f}")
    logger.info(f"Sweep completed in {elapsed:.1f}s ({len(result['feature_indices'])} features)")
    logger.info("")

    # Per-row importance matrix: (n_query, n_features)
    row_drops = result["row_feature_drops"]
    n_query = result.get("n_query", len(result["y_query"]))
    mean_drops = row_drops.mean(axis=0)

    # Sort by mean importance for display (largest drop first)
    order = np.argsort(-mean_drops)

    logger.info(f"Top {args.top} most important features (mean per-row loss):")
    logger.info(f"{'Rank':>4} {'Feat':>6} {'MeanDrop':>10} {'N_fire':>8} {'Label'}")
    logger.info("-" * 70)
    for rank, idx in enumerate(order[:args.top]):
        feat_idx = result["feature_indices"][idx]
        logger.info(
            f"{rank+1:>4} {feat_idx:>6} {mean_drops[idx]:>+10.4f} "
            f"{result['feature_n_firing'][idx]:>4}/{n_query:<3} "
            f"{result['feature_labels'][idx]}"
        )

    # Summary stats
    logger.info(f"\nImportance distribution (mean per-row loss):")
    logger.info(f"  Mean drop:   {mean_drops.mean():+.4f}")
    logger.info(f"  Std drop:    {mean_drops.std():.4f}")
    logger.info(f"  Max drop:    {mean_drops.max():+.4f}")
    logger.info(f"  Min drop:    {mean_drops.min():+.4f}")
    logger.info(f"  >0 (helpful): {(mean_drops > 0).sum()} ({(mean_drops > 0).mean()*100:.1f}%)")

    n_active = result.get("n_active_features", -1)
    logger.info(f"  Active features (fire on >=1 row): {n_active}")

    # Save results: JSON metadata + NPZ for per-row matrix
    save_data = {
        "model": args.model,
        "dataset": args.dataset,
        "task": task,
        "metric_name": "per_row_loss",
        "baseline_metric": float(result["baseline_metric"]),
        "baseline_metric_name": metric_name,
        "n_query": n_query,
        "n_active_features": n_active,
        "n_features": len(result["feature_indices"]),
        "elapsed_seconds": elapsed,
        "feature_indices": [int(x) for x in result["feature_indices"]],
        "feature_labels": result["feature_labels"],
        "feature_n_firing": [int(x) for x in result["feature_n_firing"]],
        "mean_feature_drops": [float(x) for x in mean_drops],
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)

    # Save per-row importance matrix as NPZ (compact)
    npz_path = output_path.replace(".json", ".npz") if isinstance(output_path, str) else str(output_path).replace(".json", ".npz")
    np.savez_compressed(
        npz_path,
        row_feature_drops=row_drops,
        feature_indices=result["feature_indices"],
        y_query=result["y_query"],
    )
    logger.info(f"\nSaved to {output_path} + {npz_path}")

    # Auto-run analysis
    logger.info("")
    analysis = analyze_importance(
        Path(output_path), args.model, top_n=min(args.top, 10),
    )
    print_analysis(analysis)


if __name__ == "__main__":
    main()
