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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.intervene_sae import (
    load_sae,
    load_training_mean,
    get_extraction_layer,
    compute_ablation_delta,
)

logger = logging.getLogger(__name__)

DEFAULT_CONCEPT_LABELS = PROJECT_ROOT / "output" / "cross_model_concept_labels.json"

# Map our model keys to the concept labels file keys
MODEL_KEY_TO_LABEL_KEY = {
    "tabpfn": "TabPFN",
    "mitra": "Mitra",
    "tabicl": "TabICL",
    "tabdpt": "TabDPT",
    "hyperfast": "HyperFast",
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
) -> Dict[str, np.ndarray]:
    """Sweep single-feature ablation for TabDPT.

    Fits model once, captures hidden state once, then loops over features.
    Each iteration: compute delta for 1 feature, inject via hook, get predictions.

    Returns:
        Dict with:
            baseline_preds: (n_query, n_classes) baseline probabilities
            baseline_acc: float
            feature_indices: (n_features,) which features were tested
            feature_accs: (n_features,) accuracy after ablating each feature
            feature_drops: (n_features,) baseline_acc - feature_acc (positive = important)
            y_query: ground truth labels
    """
    from sklearn.metrics import accuracy_score
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
        x_centered = all_emb
        if data_mean is not None:
            x_centered = x_centered - data_mean
        h_full = sae.encode(x_centered)
        recon_full = sae.decode(h_full)

    baseline_preds_np = np.asarray(baseline_preds)
    if task == "classification":
        baseline_acc = accuracy_score(y_query, baseline_preds_np.argmax(axis=1))
    else:
        baseline_acc = float(np.mean((baseline_preds_np - y_query) ** 2))

    # --- Sweep: ablate one feature at a time ---
    n_features = len(alive_features)
    feature_accs = np.zeros(n_features)

    t0 = time.time()
    for i, feat_idx in enumerate(alive_features):
        # Compute single-feature delta using pre-computed encoding
        with torch.no_grad():
            h_ablated = h_full.clone()
            h_ablated[:, feat_idx] = 0.0
            recon_ablated = sae.decode(h_ablated)
            delta = recon_ablated - recon_full  # (n_samples, H)

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
        if task == "classification":
            feature_accs[i] = accuracy_score(y_query, preds_np.argmax(axis=1))
        else:
            feature_accs[i] = float(np.mean((preds_np - y_query) ** 2))

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_features - i - 1) / rate
            logger.info(
                f"  [{i+1}/{n_features}] feat={feat_idx} "
                f"acc={feature_accs[i]:.3f} drop={baseline_acc - feature_accs[i]:+.3f} "
                f"({rate:.1f} feat/s, ETA {eta:.0f}s)"
            )

    feature_drops = baseline_acc - feature_accs

    return {
        "baseline_preds": baseline_preds_np,
        "baseline_acc": baseline_acc,
        "feature_indices": np.array(alive_features),
        "feature_accs": feature_accs,
        "feature_drops": feature_drops,
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
) -> Dict[str, np.ndarray]:
    """Sweep single-feature ablation for TabPFN."""
    from sklearn.metrics import accuracy_score

    if task == "regression":
        from tabpfn import TabPFNRegressor
        clf = TabPFNRegressor(device=device)
    else:
        from tabpfn import TabPFNClassifier
        clf = TabPFNClassifier(device=device)
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
        x_centered = all_emb
        if data_mean is not None:
            x_centered = x_centered - data_mean
        h_full = sae.encode(x_centered)
        recon_full = sae.decode(h_full)

    baseline_preds_np = np.asarray(baseline_preds)
    if task == "classification":
        baseline_acc = accuracy_score(y_query, baseline_preds_np.argmax(axis=1))
    else:
        baseline_acc = float(np.mean((baseline_preds_np - y_query) ** 2))

    n_features = len(alive_features)
    feature_accs = np.zeros(n_features)

    t0 = time.time()
    for i, feat_idx in enumerate(alive_features):
        with torch.no_grad():
            h_ablated = h_full.clone()
            h_ablated[:, feat_idx] = 0.0
            recon_ablated = sae.decode(h_ablated)
            delta = recon_ablated - recon_full

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
        if task == "classification":
            feature_accs[i] = accuracy_score(y_query, preds_np.argmax(axis=1))
        else:
            feature_accs[i] = float(np.mean((preds_np - y_query) ** 2))

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_features - i - 1) / rate
            logger.info(
                f"  [{i+1}/{n_features}] feat={feat_idx} "
                f"acc={feature_accs[i]:.3f} drop={baseline_acc - feature_accs[i]:+.3f} "
                f"({rate:.1f} feat/s, ETA {eta:.0f}s)"
            )

    feature_drops = baseline_acc - feature_accs

    return {
        "baseline_preds": baseline_preds_np,
        "baseline_acc": baseline_acc,
        "feature_indices": np.array(alive_features),
        "feature_accs": feature_accs,
        "feature_drops": feature_drops,
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
) -> Dict[str, np.ndarray]:
    """Sweep single-feature ablation for Mitra.

    Mitra layers return (support, query) tuples. Must modify both.
    Must save/restore RNG state for determinism.
    """
    from sklearn.metrics import accuracy_score
    from autogluon.tabular.models.mitra import MitraModel

    ag_model = MitraModel(path="/tmp/mitra_importance", name="mitra_imp")
    from autogluon.tabular.models.mitra._internal.config.enums import Task as MitraTask
    from autogluon.tabular.models.mitra._internal.core.trainer_finetune import DatasetFinetune

    import pandas as pd
    df_ctx = pd.DataFrame(X_context)
    df_ctx["__target__"] = y_context
    df_q = pd.DataFrame(X_query)
    df_q["__target__"] = y_query

    if task == "regression":
        mitra_task = MitraTask.REGRESSION
    else:
        mitra_task = MitraTask.CLASSIFICATION

    trainer = ag_model._build_trainer(
        df_ctx, df_q, target="__target__", task=mitra_task
    )

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
            baseline_preds = trainer.predict(return_proba=(task == "classification"))
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
        x_centered = all_emb
        if data_mean is not None:
            x_centered = x_centered - data_mean
        h_full = sae.encode(x_centered)
        recon_full = sae.decode(h_full)

    baseline_preds_np = np.asarray(baseline_preds)
    if task == "classification":
        baseline_acc = accuracy_score(y_query, baseline_preds_np.argmax(axis=1))
    else:
        baseline_acc = float(np.mean((baseline_preds_np - y_query) ** 2))

    n_sup = support_emb.shape[0]
    n_features = len(alive_features)
    feature_accs = np.zeros(n_features)

    t0 = time.time()
    for i, feat_idx in enumerate(alive_features):
        with torch.no_grad():
            h_ablated = h_full.clone()
            h_ablated[:, feat_idx] = 0.0
            recon_ablated = sae.decode(h_ablated)
            delta = recon_ablated - recon_full

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
                preds = trainer.predict(return_proba=(task == "classification"))
        finally:
            handle.remove()

        preds_np = np.asarray(preds)
        if task == "classification":
            feature_accs[i] = accuracy_score(y_query, preds_np.argmax(axis=1))
        else:
            feature_accs[i] = float(np.mean((preds_np - y_query) ** 2))

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_features - i - 1) / rate
            logger.info(
                f"  [{i+1}/{n_features}] feat={feat_idx} "
                f"acc={feature_accs[i]:.3f} drop={baseline_acc - feature_accs[i]:+.3f} "
                f"({rate:.1f} feat/s, ETA {eta:.0f}s)"
            )

    feature_drops = baseline_acc - feature_accs

    return {
        "baseline_preds": baseline_preds_np,
        "baseline_acc": baseline_acc,
        "feature_indices": np.array(alive_features),
        "feature_accs": feature_accs,
        "feature_drops": feature_drops,
        "y_query": np.asarray(y_query),
    }


# ── Dispatcher ───────────────────────────────────────────────────────────────

SWEEP_FN = {
    "tabpfn": sweep_tabpfn,
    "mitra": sweep_mitra,
    "tabdpt": sweep_tabdpt,
}


def sweep_concept_importance(
    model_key: str,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    device: str = "cuda",
    task: str = "classification",
) -> Dict:
    """Run single-feature ablation sweep for all alive features.

    Returns dict with feature_indices, feature_drops, feature_labels, baseline_acc.
    """
    if model_key not in SWEEP_FN:
        raise ValueError(f"Unsupported model: {model_key}. Choose from {list(SWEEP_FN.keys())}")

    sae, config = load_sae(model_key, device=device)
    extraction_layer = get_extraction_layer(model_key)
    data_mean = load_training_mean(model_key, device=device)
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
        sae_dir = PROJECT_ROOT / "output" / "sae_tabarena_sweep_round5"

    ckpt_path = sae_dir / model_key / "sae_matryoshka_archetypal_validated.pt"
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
        labels_path: Path to cross_model_concept_labels.json

    Returns:
        Dict with 'baseline_acc', 'dataset', 'model', 'bands', 'top_groups',
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

    # Enrich each feature with band and group info
    enriched = []
    for feat in importance["features"]:
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

    return {
        "model": model_key,
        "dataset": importance["dataset"],
        "task": importance["task"],
        "baseline_acc": importance["baseline_acc"],
        "bands": bands,
        "top_groups": top_groups,
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
    print(f"Baseline accuracy: {analysis['baseline_acc']:.3f}")
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from scripts.extract_layer_embeddings import get_dataset_task
    task = get_dataset_task(args.dataset)

    # Determine output path
    output_path = args.output
    if output_path is None:
        output_dir = PROJECT_ROOT / "output" / "concept_importance"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{args.model}_{args.dataset}.json")

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
        X_context=X_ctx,
        y_context=y_ctx,
        X_query=X_q,
        y_query=y_q,
        device=args.device,
        task=task,
    )
    elapsed = time.time() - t0

    # Display results
    logger.info(f"\nBaseline accuracy: {result['baseline_acc']:.3f}")
    logger.info(f"Sweep completed in {elapsed:.1f}s ({len(result['feature_indices'])} features)")
    logger.info("")

    # Sort by importance (largest drop first)
    order = np.argsort(-result["feature_drops"])

    logger.info(f"Top {args.top} most important features:")
    logger.info(f"{'Rank':>4} {'Feat':>6} {'Drop':>8} {'Acc':>8} {'Label'}")
    logger.info("-" * 70)
    for rank, idx in enumerate(order[:args.top]):
        feat_idx = result["feature_indices"][idx]
        drop = result["feature_drops"][idx]
        acc = result["feature_accs"][idx]
        label = result["feature_labels"][idx]
        logger.info(f"{rank+1:>4} {feat_idx:>6} {drop:>+8.4f} {acc:>8.3f} {label}")

    # Also show bottom features (least important / helpful when ablated)
    logger.info(f"\nBottom {min(10, args.top)} features (least important / improve when ablated):")
    logger.info(f"{'Rank':>4} {'Feat':>6} {'Drop':>8} {'Acc':>8} {'Label'}")
    logger.info("-" * 70)
    for rank, idx in enumerate(order[-min(10, args.top):]):
        feat_idx = result["feature_indices"][idx]
        drop = result["feature_drops"][idx]
        acc = result["feature_accs"][idx]
        label = result["feature_labels"][idx]
        logger.info(f"{'':>4} {feat_idx:>6} {drop:>+8.4f} {acc:>8.3f} {label}")

    # Summary stats
    drops = result["feature_drops"]
    logger.info(f"\nImportance distribution:")
    logger.info(f"  Mean drop:   {drops.mean():+.4f}")
    logger.info(f"  Std drop:    {drops.std():.4f}")
    logger.info(f"  Max drop:    {drops.max():+.4f}")
    logger.info(f"  Min drop:    {drops.min():+.4f}")
    logger.info(f"  >0 (helpful): {(drops > 0).sum()} ({(drops > 0).mean()*100:.1f}%)")
    logger.info(f"  >1pp:         {(drops > 0.01).sum()}")
    logger.info(f"  >5pp:         {(drops > 0.05).sum()}")

    # Save results
    save_data = {
        "model": args.model,
        "dataset": args.dataset,
        "task": task,
        "baseline_acc": float(result["baseline_acc"]),
        "n_features": len(result["feature_indices"]),
        "elapsed_seconds": elapsed,
        "features": [],
    }
    for i in range(len(result["feature_indices"])):
        save_data["features"].append({
            "index": int(result["feature_indices"][i]),
            "drop": float(result["feature_drops"][i]),
            "acc": float(result["feature_accs"][i]),
            "label": result["feature_labels"][i],
        })
    # Sort by drop descending
    save_data["features"].sort(key=lambda x: -x["drop"])

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info(f"\nSaved to {output_path}")

    # Auto-run analysis
    logger.info("")
    analysis = analyze_importance(
        Path(output_path), args.model, top_n=min(args.top, 10),
    )
    print_analysis(analysis)


if __name__ == "__main__":
    main()
