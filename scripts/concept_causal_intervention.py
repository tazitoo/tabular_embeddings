#!/usr/bin/env python3
"""Causal concept interventions: ablation, boost, and transplant.

Proves causality in both directions:
- Ablation: Remove A's unique concepts → A's performance drops toward B.
- Boost: Strengthen B's weak matched concepts → B improves toward A.
- Transplant: Transfer A's unmatched concepts into B → B improves.

Uses MNN-based feature matching for concept identification and SAE fingerprints
for dataset-specific feature ranking.

Usage:
    # Run on top-10 highest-signal targets (auto-selected from diagnostic)
    python scripts/concept_causal_intervention.py --device cuda --top-targets 10

    # Specific pair + dataset
    python scripts/concept_causal_intervention.py --model-a tabpfn --model-b mitra \\
        --datasets adult credit-g --device cuda

    # Only ablation (fastest)
    python scripts/concept_causal_intervention.py --ablation-only --top-targets 10 --device cuda

    # Only boost (matched feature strengthening)
    python scripts/concept_causal_intervention.py --boost-only --top-targets 10 --device cuda

    # All pairs
    python scripts/concept_causal_intervention.py --all-pairs --device cuda
"""

import argparse
import json
import logging
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PERF_CSV = PROJECT_ROOT / "output" / "model_performance.csv"
DEFAULT_MNN_PATH = PROJECT_ROOT / "output" / "sae_feature_matching_mnn_t0.001_n500.json"
DEFAULT_FINGERPRINT_DIR = PROJECT_ROOT / "output" / "concept_fingerprints"
DEFAULT_DIAGNOSTIC_PATH = PROJECT_ROOT / "output" / "concept_performance_diagnostic.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "concept_causal_results.json"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "output" / "figures"
DEFAULT_TRAINING_DIR = PROJECT_ROOT / "output" / "sae_training_round5"

# Models with intervention support
INTERVENTION_MODELS = ["tabpfn", "mitra", "tabicl", "tabdpt", "hyperfast"]

DISPLAY_NAMES = {
    "tabpfn": "TabPFN",
    "mitra": "Mitra",
    "tabicl": "TabICL",
    "tabdpt": "TabDPT",
    "hyperfast": "HyperFast",
    "carte": "CARTE",
    "tabula8b": "Tabula-8B",
}


# ── Target Selection ───────────────────────────────────────────────────────


def select_targets(
    diagnostic_path: Path = DEFAULT_DIAGNOSTIC_PATH,
    perf_csv_path: Path = DEFAULT_PERF_CSV,
    min_gap: float = 0.05,
    top_n: int = 10,
) -> List[Dict]:
    """Select (model_a, model_b, dataset) triples with strongest causal signal.

    Selects cases where:
    - A significantly outperforms B (perf gap > min_gap)
    - Both models have intervention support
    - A has unique concepts active on this dataset

    Returns sorted by expected signal strength (perf_gap * concept_gap).
    """
    import pandas as pd

    if not diagnostic_path.exists():
        raise FileNotFoundError(
            f"Diagnostic results not found: {diagnostic_path}. "
            f"Run 'concept_performance_diagnostic.py analyze' first."
        )

    with open(diagnostic_path) as f:
        diagnostic = json.load(f)

    performance = pd.read_csv(perf_csv_path) if perf_csv_path.exists() else None

    targets = []
    for pair_label, pr in diagnostic["pairs"].items():
        model_a = pr["model_a"]
        model_b = pr["model_b"]

        # Both must support intervention
        if model_a not in INTERVENTION_MODELS or model_b not in INTERVENTION_MODELS:
            continue

        for row in pr["data"]:
            ds = row["dataset"]
            perf_gap = row.get("perf_gap", 0)
            concept_asym = row.get("concept_asymmetry", 0)
            unmatched_a = row.get("unmatched_act_a", 0)

            # A outperforms B significantly
            if perf_gap > min_gap and unmatched_a > 0:
                targets.append({
                    "model_a": model_a,
                    "model_b": model_b,
                    "dataset": ds,
                    "perf_gap": perf_gap,
                    "concept_asymmetry": concept_asym,
                    "unmatched_act_a": unmatched_a,
                    "signal_strength": abs(perf_gap) * abs(concept_asym),
                })

            # Also check reverse direction (B outperforms A)
            if perf_gap < -min_gap and row.get("unmatched_act_b", 0) > 0:
                targets.append({
                    "model_a": model_b,
                    "model_b": model_a,
                    "dataset": ds,
                    "perf_gap": -perf_gap,
                    "concept_asymmetry": -concept_asym,
                    "unmatched_act_a": row.get("unmatched_act_b", 0),
                    "signal_strength": abs(perf_gap) * abs(concept_asym),
                })

    targets.sort(key=lambda x: -x["signal_strength"])
    return targets[:top_n]


# ── Shared Helpers ─────────────────────────────────────────────────────────


def _load_splits(dataset: str, task: str):
    """Load and split a TabArena dataset for evaluation."""
    from data.extended_loader import load_tabarena_dataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    result = load_tabarena_dataset(dataset, max_samples=1100)
    if result is None:
        raise ValueError(f"Failed to load dataset: {dataset}")

    X, y, _ = result
    n = len(X)
    ctx_size = min(600, int(n * 0.7))
    q_size = min(500, n - ctx_size)

    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)
        query_frac = q_size / (ctx_size + q_size)
        try:
            X_ctx, X_q, y_ctx, y_q = train_test_split(
                X, y, test_size=query_frac, random_state=42, stratify=y,
            )
        except ValueError:
            X_ctx, X_q, y_ctx, y_q = train_test_split(
                X, y, test_size=query_frac, random_state=42,
            )
    else:
        X_ctx = X[:ctx_size]
        y_ctx = y[:ctx_size]
        X_q = X[ctx_size:ctx_size + q_size]
        y_q = y[ctx_size:ctx_size + q_size]

    return X_ctx[:ctx_size], y_ctx[:ctx_size], X_q[:q_size], y_q[:q_size]


def _compute_metric(preds, y_true, task):
    """Compute performance metric. Returns (metric_name, metric_value)."""
    from scripts.concept_performance_diagnostic import compute_metric
    return compute_metric(preds, y_true, task)


def _load_mnn_pair(
    model_a: str,
    model_b: str,
    mnn_path: Path = DEFAULT_MNN_PATH,
) -> Dict:
    """Load MNN matching data for a specific pair.

    Returns dict with 'matches', 'unmatched_a', 'unmatched_b' oriented
    so that A/B match the requested model order.
    """
    with open(mnn_path) as f:
        mnn_data = json.load(f)

    display_a = DISPLAY_NAMES[model_a]
    display_b = DISPLAY_NAMES[model_b]

    sorted_displays = sorted([display_a, display_b])
    pk = f"{sorted_displays[0]}__{sorted_displays[1]}"
    pair_data = mnn_data["pairs"][pk]
    swapped = sorted_displays[0] != display_a

    if swapped:
        return {
            "matches": [
                {"idx_a": m["idx_b"], "idx_b": m["idx_a"], "r": m["r"]}
                for m in pair_data["matches"]
            ],
            "unmatched_a": pair_data["unmatched_b"],
            "unmatched_b": pair_data["unmatched_a"],
            "n_alive_a": pair_data["n_alive_b"],
            "n_alive_b": pair_data["n_alive_a"],
        }
    else:
        return {
            "matches": pair_data["matches"],
            "unmatched_a": pair_data["unmatched_a"],
            "unmatched_b": pair_data["unmatched_b"],
            "n_alive_a": pair_data["n_alive_a"],
            "n_alive_b": pair_data["n_alive_b"],
        }


def _load_fingerprint(model_key: str, fp_dir: Path = DEFAULT_FINGERPRINT_DIR) -> Dict:
    """Load pre-computed fingerprint for a model."""
    fp_path = fp_dir / f"{model_key}_fingerprints.json"
    if not fp_path.exists():
        raise FileNotFoundError(f"Fingerprint not found: {fp_path}")
    with open(fp_path) as f:
        return json.load(f)


def _rank_features_by_activation(
    features: List[int],
    fingerprint: Dict,
    dataset: str,
) -> List[Tuple[int, float]]:
    """Rank features by mean activation on a specific dataset.

    Returns [(feature_idx, activation)] sorted descending by |activation|.
    """
    if dataset not in fingerprint["dataset_means"]:
        # Fall back to global mean
        means = fingerprint["global_mean"]
    else:
        means = fingerprint["dataset_means"][dataset]

    ranked = [(f, abs(means[f])) for f in features]
    ranked.sort(key=lambda x: -x[1])
    return ranked


# ── Intervention Type 1: Ablation ──────────────────────────────────────────


def dose_response_ablation(
    model_a: str,
    model_b: str,
    dataset: str,
    task: str,
    device: str = "cuda",
    max_steps: int = 20,
    mnn_path: Path = DEFAULT_MNN_PATH,
    fp_dir: Path = DEFAULT_FINGERPRINT_DIR,
) -> Dict:
    """Progressively ablate A-only concepts from A.

    Tests: does A's performance drop toward B's level?

    Returns dict with dose-response curve data.
    """
    from scripts.intervene_sae import intervene

    X_ctx, y_ctx, X_q, y_q = _load_splits(dataset, task)

    # Get A-only features ranked by activation on this dataset
    mnn_pair = _load_mnn_pair(model_a, model_b, mnn_path)
    fp_a = _load_fingerprint(model_a, fp_dir)
    ranked = _rank_features_by_activation(
        mnn_pair["unmatched_a"], fp_a, dataset,
    )

    if not ranked:
        return {"error": "No unmatched features for A", "steps": []}

    # Limit steps
    n_features = min(len(ranked), max_steps)

    # Get baselines (A and B with no ablation)
    result_a = intervene(
        model_key=model_a, X_context=X_ctx, y_context=y_ctx,
        X_query=X_q, y_query=y_q, ablate_features=[], device=device, task=task,
    )
    _, baseline_a = _compute_metric(result_a["baseline_preds"], y_q, task)

    result_b = intervene(
        model_key=model_b, X_context=X_ctx, y_context=y_ctx,
        X_query=X_q, y_query=y_q, ablate_features=[], device=device, task=task,
    )
    _, baseline_b = _compute_metric(result_b["baseline_preds"], y_q, task)

    # Progressive ablation
    steps = []
    for step in range(1, n_features + 1):
        features_to_ablate = [f for f, _ in ranked[:step]]
        result = intervene(
            model_key=model_a, X_context=X_ctx, y_context=y_ctx,
            X_query=X_q, y_query=y_q, ablate_features=features_to_ablate,
            device=device, task=task,
        )
        _, ablated_metric = _compute_metric(result["ablated_preds"], y_q, task)

        drop = baseline_a - ablated_metric
        gap = baseline_a - baseline_b
        convergence = drop / gap if abs(gap) > 1e-6 else float("nan")

        steps.append({
            "n_features": step,
            "features_ablated": features_to_ablate,
            "ablated_metric": float(ablated_metric),
            "drop": float(drop),
            "convergence": float(convergence),
        })

        logger.info(
            "  Ablation step %d/%d: ablated=%.4f drop=%.4f conv=%.3f",
            step, n_features, ablated_metric, drop, convergence,
        )

    return {
        "type": "ablation",
        "model_a": model_a,
        "model_b": model_b,
        "dataset": dataset,
        "task": task,
        "baseline_a": float(baseline_a),
        "baseline_b": float(baseline_b),
        "gap": float(baseline_a - baseline_b),
        "n_unmatched_a": len(mnn_pair["unmatched_a"]),
        "steps": steps,
    }


# ── Intervention Type 2: Boost ─────────────────────────────────────────────


def dose_response_boost(
    model_a: str,
    model_b: str,
    dataset: str,
    task: str,
    device: str = "cuda",
    max_steps: int = 20,
    mnn_path: Path = DEFAULT_MNN_PATH,
    fp_dir: Path = DEFAULT_FINGERPRINT_DIR,
) -> Dict:
    """Progressively boost B's weak matched features toward A's activation levels.

    For MNN-matched pairs (i in A, j in B), ranks by A_act - B_act and boosts
    B's features to A's activation levels.

    Tests: does B's performance improve toward A's level?
    """
    import torch
    from scripts.intervene_sae import (
        load_sae, load_training_mean, get_extraction_layer,
        compute_boost_delta, INTERVENE_FN,
    )

    X_ctx, y_ctx, X_q, y_q = _load_splits(dataset, task)

    # Get matched features with activation differentials
    mnn_pair = _load_mnn_pair(model_a, model_b, mnn_path)
    fp_a = _load_fingerprint(model_a, fp_dir)
    fp_b = _load_fingerprint(model_b, fp_dir)

    if not mnn_pair["matches"]:
        return {"error": "No matched features", "steps": []}

    means_a = fp_a["dataset_means"].get(dataset, fp_a["global_mean"])
    means_b = fp_b["dataset_means"].get(dataset, fp_b["global_mean"])

    # Rank matched pairs by activation differential (A_act - B_act)
    differentials = []
    for m in mnn_pair["matches"]:
        act_a = means_a[m["idx_a"]]
        act_b = means_b[m["idx_b"]]
        diff = act_a - act_b
        if diff > 0:  # Only boost where A is stronger
            differentials.append({
                "idx_a": m["idx_a"],
                "idx_b": m["idx_b"],
                "r": m["r"],
                "act_a": act_a,
                "act_b": act_b,
                "diff": diff,
            })

    differentials.sort(key=lambda x: -x["diff"])

    if not differentials:
        return {"error": "No matched features where A > B", "steps": []}

    n_features = min(len(differentials), max_steps)

    # Get baselines
    from scripts.intervene_sae import intervene

    result_a = intervene(
        model_key=model_a, X_context=X_ctx, y_context=y_ctx,
        X_query=X_q, y_query=y_q, ablate_features=[], device=device, task=task,
    )
    _, baseline_a = _compute_metric(result_a["baseline_preds"], y_q, task)

    result_b = intervene(
        model_key=model_b, X_context=X_ctx, y_context=y_ctx,
        X_query=X_q, y_query=y_q, ablate_features=[], device=device, task=task,
    )
    _, baseline_b = _compute_metric(result_b["baseline_preds"], y_q, task)

    # Load B's SAE for boost delta computation
    sae_b, _ = load_sae(model_b, device=device)
    extraction_layer_b = get_extraction_layer(model_b)
    data_mean_b = load_training_mean(model_b, device=device)

    # We need to capture B's hidden state for boost computation.
    # Run B's intervention function with custom delta injection.
    intervene_fn_b = INTERVENE_FN[model_b]

    # Progressive boost: use increasing number of matched features
    steps = []
    for step in range(1, n_features + 1):
        top_diffs = differentials[:step]
        boost_features = [d["idx_b"] for d in top_diffs]
        target_activations = [d["act_a"] for d in top_diffs]

        # Run B with boost — use compute_boost_delta through intervene machinery.
        # We override the delta computation by passing boost features as ablation
        # features and swapping the delta function. Since intervene() calls
        # compute_ablation_delta internally, we need to use the lower-level approach.

        # Approach: run B's baseline to capture hidden state, compute boost delta,
        # then inject it. This mirrors how ablation works but with boost_delta.
        result = _run_boost_intervention(
            model_key=model_b,
            X_ctx=X_ctx, y_ctx=y_ctx, X_q=X_q, y_q=y_q,
            sae=sae_b, boost_features=boost_features,
            target_activations=target_activations,
            extraction_layer=extraction_layer_b,
            data_mean=data_mean_b,
            device=device, task=task,
        )

        _, boosted_metric = _compute_metric(result["boosted_preds"], y_q, task)

        improvement = boosted_metric - baseline_b
        gap = baseline_a - baseline_b
        convergence = improvement / gap if abs(gap) > 1e-6 else float("nan")

        steps.append({
            "n_features": step,
            "features_boosted": boost_features,
            "boosted_metric": float(boosted_metric),
            "improvement": float(improvement),
            "convergence": float(convergence),
        })

        logger.info(
            "  Boost step %d/%d: boosted=%.4f imp=%.4f conv=%.3f",
            step, n_features, boosted_metric, improvement, convergence,
        )

    return {
        "type": "boost",
        "model_a": model_a,
        "model_b": model_b,
        "dataset": dataset,
        "task": task,
        "baseline_a": float(baseline_a),
        "baseline_b": float(baseline_b),
        "gap": float(baseline_a - baseline_b),
        "n_matched": len(mnn_pair["matches"]),
        "n_with_differential": len(differentials),
        "steps": steps,
    }


def _run_boost_intervention(
    model_key: str,
    X_ctx: np.ndarray,
    y_ctx: np.ndarray,
    X_q: np.ndarray,
    y_q: np.ndarray,
    sae,
    boost_features: List[int],
    target_activations: List[float],
    extraction_layer: int,
    data_mean,
    device: str = "cuda",
    task: str = "classification",
) -> Dict:
    """Run a model with SAE feature boosting at extraction layer.

    Like intervene() but uses compute_boost_delta instead of ablation.
    Dispatches to model-specific implementations.
    """
    import torch
    from scripts.intervene_sae import compute_boost_delta

    # Use model-specific prediction code from intervene_sae.
    # The pattern is: capture hidden → compute boost delta → inject delta.
    # This reuses the exact same hook machinery as ablation, just different delta.

    if model_key == "tabpfn":
        return _boost_tabpfn(
            X_ctx, y_ctx, X_q, y_q, sae, boost_features,
            target_activations, extraction_layer, device, task, data_mean,
        )
    elif model_key == "tabdpt":
        return _boost_tabdpt(
            X_ctx, y_ctx, X_q, y_q, sae, boost_features,
            target_activations, extraction_layer, device, task, data_mean,
        )
    elif model_key == "mitra":
        return _boost_mitra(
            X_ctx, y_ctx, X_q, y_q, sae, boost_features,
            target_activations, extraction_layer, device, task, data_mean,
        )
    elif model_key == "tabicl":
        return _boost_tabicl(
            X_ctx, y_ctx, X_q, y_q, sae, boost_features,
            target_activations, extraction_layer, device, task, data_mean,
        )
    elif model_key == "hyperfast":
        return _boost_hyperfast(
            X_ctx, y_ctx, X_q, y_q, sae, boost_features,
            target_activations, extraction_layer, device, task, data_mean,
        )
    else:
        raise ValueError(f"Boost not supported for {model_key}")


def _boost_tabpfn(X_ctx, y_ctx, X_q, y_q, sae, boost_features,
                   target_activations, extraction_layer, device, task, data_mean):
    """Boost intervention for TabPFN."""
    import torch
    from scripts.intervene_sae import compute_boost_delta
    from models.tabpfn_utils import load_tabpfn

    clf = load_tabpfn(task=task, device=device, n_estimators=1)
    clf.fit(X_ctx, y_ctx)
    model = clf.model_
    layers = model.transformer_encoder.layers

    captured = {}

    def capture_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured["hidden"] = output.detach()

    handle = layers[extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            if task == "regression":
                baseline = clf.predict(X_q)
            else:
                baseline = clf.predict_proba(X_q)
    finally:
        handle.remove()

    hidden = captured["hidden"]
    all_emb = hidden[0].mean(dim=1)

    delta = compute_boost_delta(sae, all_emb, boost_features, target_activations, data_mean)
    delta_broadcast = delta.unsqueeze(1)

    def modify_hook(module, input, output):
        if isinstance(output, torch.Tensor) and output.ndim == 4:
            output = output.clone()
            output[0] += delta_broadcast
            return output
        return output

    handle = layers[extraction_layer].register_forward_hook(modify_hook)
    try:
        with torch.no_grad():
            if task == "regression":
                boosted = clf.predict(X_q)
            else:
                boosted = clf.predict_proba(X_q)
    finally:
        handle.remove()

    return {
        "baseline_preds": np.asarray(baseline),
        "boosted_preds": np.asarray(boosted),
        "y_query": np.asarray(y_q),
    }


def _boost_tabdpt(X_ctx, y_ctx, X_q, y_q, sae, boost_features,
                   target_activations, extraction_layer, device, task, data_mean):
    """Boost intervention for TabDPT."""
    import torch
    from scripts.intervene_sae import compute_boost_delta
    from tabdpt import TabDPTClassifier, TabDPTRegressor

    if task == "regression":
        clf = TabDPTRegressor(device=device, compile=False)
    else:
        clf = TabDPTClassifier(device=device, compile=False)
    clf.fit(X_ctx, y_ctx)

    encoder_layers = clf.model.transformer_encoder
    captured = {}

    def capture_hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            captured["hidden"] = out.detach()

    handle = encoder_layers[extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            if task == "regression":
                baseline = clf.predict(X_q)
            else:
                baseline = clf.predict_proba(X_q)
    finally:
        handle.remove()

    hidden = captured["hidden"]
    all_emb = hidden.mean(dim=1) if hidden.ndim == 3 else hidden

    delta = compute_boost_delta(sae, all_emb, boost_features, target_activations, data_mean)

    def modify_hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            out = out.clone()
            if out.ndim == 3:
                out += delta.unsqueeze(1)
            elif out.ndim == 2:
                out += delta
            if isinstance(output, tuple):
                return (out,) + output[1:]
            return out
        return output

    handle = encoder_layers[extraction_layer].register_forward_hook(modify_hook)
    try:
        with torch.no_grad():
            if task == "regression":
                boosted = clf.predict(X_q)
            else:
                boosted = clf.predict_proba(X_q)
    finally:
        handle.remove()

    return {
        "baseline_preds": np.asarray(baseline),
        "boosted_preds": np.asarray(boosted),
        "y_query": np.asarray(y_q),
    }


def _boost_mitra(X_ctx, y_ctx, X_q, y_q, sae, boost_features,
                  target_activations, extraction_layer, device, task, data_mean):
    """Boost intervention for Mitra."""
    import torch
    from scripts.intervene_sae import compute_boost_delta

    n_features = X_q.shape[1]
    max_context = max(100, 200_000 // max(n_features, 1))
    if len(X_ctx) > max_context:
        X_ctx = X_ctx[:max_context]
        y_ctx = y_ctx[:max_context]

    if task == "regression":
        from autogluon.tabular.models.mitra.sklearn_interface import MitraRegressor
        clf = MitraRegressor(device=device, n_estimators=1, fine_tune=False)
    else:
        from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier
        clf = MitraClassifier(device=device, n_estimators=1, fine_tune=False)
    clf.fit(X_ctx, y_ctx)
    torch.cuda.empty_cache()

    trainer = clf.trainers[0]
    layers = trainer.model.layers
    rng_state = trainer.rng.get_state()

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
                baseline = clf.predict(X_q)
            else:
                baseline = clf.predict_proba(X_q)
    finally:
        handle.remove()

    def extract_y_tokens(tensor_list):
        embs = []
        for h in tensor_list:
            if h.ndim == 4:
                embs.append(h[0, :, 0, :])
            elif h.ndim == 3:
                embs.append(h.squeeze(0))
            elif h.ndim == 2:
                embs.append(h)
        return torch.cat(embs, dim=0) if embs else None

    support_emb = extract_y_tokens(captured_support)
    query_emb = extract_y_tokens(captured_query)

    delta_sup = compute_boost_delta(sae, support_emb, boost_features, target_activations, data_mean)
    delta_qry = compute_boost_delta(sae, query_emb, boost_features, target_activations, data_mean)

    trainer.rng.set_state(rng_state)
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
            sup[0] += delta_sup[s:s + n_s].unsqueeze(1)
            sup_offset[0] = s + n_s
            modified[0] = sup
        if isinstance(qry, torch.Tensor) and qry.ndim == 4:
            qry = qry.clone()
            n_q = qry.shape[1]
            s = qry_offset[0]
            qry[0] += delta_qry[s:s + n_q].unsqueeze(1)
            qry_offset[0] = s + n_q
            modified[1] = qry
        return tuple(modified)

    handle = layers[extraction_layer].register_forward_hook(modify_hook)
    try:
        with torch.no_grad():
            sup_offset[0] = 0
            qry_offset[0] = 0
            if task == "regression":
                boosted = clf.predict(X_q)
            else:
                boosted = clf.predict_proba(X_q)
    finally:
        handle.remove()

    return {
        "baseline_preds": np.asarray(baseline),
        "boosted_preds": np.asarray(boosted),
        "y_query": np.asarray(y_q),
    }


def _boost_tabicl(X_ctx, y_ctx, X_q, y_q, sae, boost_features,
                   target_activations, extraction_layer, device, task, data_mean):
    """Boost intervention for TabICL."""
    import torch
    from scripts.intervene_sae import compute_boost_delta
    from tabicl import TabICLClassifier

    clf = TabICLClassifier(device=device, n_estimators=1)
    clf.fit(X_ctx, y_ctx)

    blocks = clf.model_.icl_predictor.tf_icl.blocks
    captured = {}

    def capture_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured["hidden"] = output.detach()

    handle = blocks[extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            baseline = clf.predict_proba(X_q)
    finally:
        handle.remove()

    hidden = captured["hidden"]
    all_emb = hidden.mean(dim=0)
    batch_mean = all_emb.mean(dim=0)

    delta = compute_boost_delta(sae, all_emb, boost_features, target_activations, batch_mean)
    delta_broadcast = delta.unsqueeze(0)

    def modify_hook(module, input, output):
        if isinstance(output, torch.Tensor) and output.ndim == 3:
            output = output.clone()
            output += delta_broadcast
            return output
        return output

    handle = blocks[extraction_layer].register_forward_hook(modify_hook)
    try:
        with torch.no_grad():
            boosted = clf.predict_proba(X_q)
    finally:
        handle.remove()

    return {
        "baseline_preds": np.asarray(baseline),
        "boosted_preds": np.asarray(boosted),
        "y_query": np.asarray(y_q),
    }


def _boost_hyperfast(X_ctx, y_ctx, X_q, y_q, sae, boost_features,
                      target_activations, extraction_layer, device, task, data_mean):
    """Boost intervention for HyperFast."""
    import torch
    import torch.nn.functional as F
    from scripts.intervene_sae import compute_boost_delta
    from hyperfast.hyperfast import forward_main_network, transform_data_for_main_network
    from models.hyperfast_embeddings import HyperFastEmbeddingExtractor

    extractor = HyperFastEmbeddingExtractor(device=device)
    extractor.fit(X_ctx, y_ctx)
    clf = extractor._model

    X_q_t = torch.tensor(X_q, dtype=torch.float32).to(device)

    baseline_outputs = []
    boosted_outputs = []

    for jj in range(len(clf._main_networks)):
        main_network = clf._move_to_device(clf._main_networks[jj])
        rf = clf._move_to_device(clf._rfs[jj])
        pca = clf._move_to_device(clf._pcas[jj])

        if clf.feature_bagging:
            X_b = X_q_t[:, clf.selected_features[jj]]
        else:
            X_b = X_q_t

        X_transformed = transform_data_for_main_network(
            X=X_b, cfg=clf._cfg, rf=rf, pca=pca,
        )

        with torch.no_grad():
            outputs_base, _ = forward_main_network(X_transformed, main_network)
        baseline_outputs.append(F.softmax(outputs_base, dim=1).cpu().numpy())

        with torch.no_grad():
            x = X_transformed
            for layer_idx, (weight, bias) in enumerate(main_network):
                weight = clf._move_to_device(weight)
                bias = clf._move_to_device(bias)
                x_new = F.linear(x, weight, bias)
                if layer_idx < len(main_network) - 1:
                    x_new = F.relu(x_new)
                    if x_new.shape[-1] == x.shape[-1]:
                        x = x + x_new
                    else:
                        x = x_new
                else:
                    x = x_new

                if layer_idx == extraction_layer:
                    delta = compute_boost_delta(
                        sae, x, boost_features, target_activations, data_mean,
                    )
                    x = x + delta

            boosted_outputs.append(F.softmax(x, dim=1).cpu().numpy())

    return {
        "baseline_preds": np.mean(baseline_outputs, axis=0),
        "boosted_preds": np.mean(boosted_outputs, axis=0),
        "y_query": np.asarray(y_q),
    }


# ── Intervention Type 3: Transplant ───────────────────────────────────────


def fit_concept_projection(
    model_a: str,
    model_b: str,
    feature_a: int,
    training_dir: Path = DEFAULT_TRAINING_DIR,
    device: str = "cpu",
) -> Tuple[Optional[np.ndarray], float]:
    """Learn how A's concept looks in B's embedding space.

    Uses paired training rows (same original data, different embedding spaces)
    to fit a ridge regression: feature_a_activation ~ B_embeddings.

    Returns:
        (projection_vector, r2) — (emb_dim_b,) projection vector and fit quality.
        Returns (None, 0.0) if the concept is not linearly decodable from B's space.
    """
    import torch
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_score

    from scripts.intervene_sae import load_sae, load_training_mean, get_extraction_layer
    from scripts.concept_fingerprint import load_per_dataset_embeddings

    # Load embeddings for both models
    emb_a_dict = load_per_dataset_embeddings(model_a, training_dir)
    emb_b_dict = load_per_dataset_embeddings(model_b, training_dir)

    # Find common datasets with matching row counts
    common = sorted(set(emb_a_dict.keys()) & set(emb_b_dict.keys()))
    if not common:
        return None, 0.0

    # Pool paired rows
    embs_a = []
    embs_b = []
    for ds in common:
        ea = emb_a_dict[ds]
        eb = emb_b_dict[ds]
        n = min(len(ea), len(eb))
        embs_a.append(ea[:n])
        embs_b.append(eb[:n])

    X_a = np.concatenate(embs_a, axis=0)
    X_b = np.concatenate(embs_b, axis=0)

    # Encode A's embeddings through A's SAE → get activations for feature_a
    sae_a, _ = load_sae(model_a, device=device)
    data_mean_a = load_training_mean(model_a, device=device)

    with torch.no_grad():
        x = torch.tensor(X_a, dtype=torch.float32, device=device)
        x_centered = x - data_mean_a
        h = sae_a.encode(x_centered)
        target = h[:, feature_a].cpu().numpy()

    # Skip if feature is dead in training data
    if np.std(target) < 1e-8:
        return None, 0.0

    # Center B's embeddings
    X_b_centered = X_b - X_b.mean(axis=0)

    # Fit ridge regression: target ~ X_b_centered
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    ridge.fit(X_b_centered, target)

    # Cross-validated R²
    cv_scores = cross_val_score(ridge, X_b_centered, target, cv=3, scoring="r2")
    r2 = float(np.mean(cv_scores))

    if r2 < 0.1:
        return None, r2

    # The regression coefficients = projection vector in B's space
    projection = ridge.coef_.astype(np.float32)

    return projection, r2


def dose_response_transplant(
    model_a: str,
    model_b: str,
    dataset: str,
    task: str,
    device: str = "cuda",
    max_steps: int = 10,
    mnn_path: Path = DEFAULT_MNN_PATH,
    fp_dir: Path = DEFAULT_FINGERPRINT_DIR,
    training_dir: Path = DEFAULT_TRAINING_DIR,
) -> Dict:
    """Progressively transplant A-only concepts into B.

    For each A-only feature:
    1. Fit concept projection (A → B space) via ridge regression.
    2. Filter by R² > 0.1.
    3. Inject projection * target_activation into B's hidden state.

    Tests: does B's performance improve toward A's level?
    """
    import torch
    from scripts.intervene_sae import intervene, get_extraction_layer, INTERVENE_FN

    X_ctx, y_ctx, X_q, y_q = _load_splits(dataset, task)

    # Get A-only features ranked by activation
    mnn_pair = _load_mnn_pair(model_a, model_b, mnn_path)
    fp_a = _load_fingerprint(model_a, fp_dir)
    ranked = _rank_features_by_activation(
        mnn_pair["unmatched_a"], fp_a, dataset,
    )

    if not ranked:
        return {"error": "No unmatched features for A", "steps": []}

    # Get baselines
    result_a = intervene(
        model_key=model_a, X_context=X_ctx, y_context=y_ctx,
        X_query=X_q, y_query=y_q, ablate_features=[], device=device, task=task,
    )
    _, baseline_a = _compute_metric(result_a["baseline_preds"], y_q, task)

    result_b = intervene(
        model_key=model_b, X_context=X_ctx, y_context=y_ctx,
        X_query=X_q, y_query=y_q, ablate_features=[], device=device, task=task,
    )
    _, baseline_b = _compute_metric(result_b["baseline_preds"], y_q, task)

    # Fit projections for top features (CPU-bound, ~30s each)
    projections = []
    for feat_a, act in ranked[:max_steps * 2]:  # Try 2x to find enough with R² > 0.1
        if len(projections) >= max_steps:
            break

        logger.info("  Fitting projection for feature %d (act=%.3f)...", feat_a, act)
        proj, r2 = fit_concept_projection(
            model_a, model_b, feat_a,
            training_dir=training_dir, device=device,
        )
        if proj is not None:
            projections.append({
                "feature_a": feat_a,
                "projection": proj,
                "r2": r2,
                "target_activation": float(act),
            })
            logger.info("    R² = %.3f (accepted)", r2)
        else:
            logger.info("    R² = %.3f (rejected)", r2)

    if not projections:
        return {"error": "No concepts with R² > 0.1", "steps": []}

    extraction_layer_b = get_extraction_layer(model_b)

    # Progressive transplant
    steps = []
    for step in range(1, len(projections) + 1):
        active_projs = projections[:step]

        # Compute combined transplant delta
        result = _run_transplant_intervention(
            model_key=model_b,
            X_ctx=X_ctx, y_ctx=y_ctx, X_q=X_q, y_q=y_q,
            projections=active_projs,
            extraction_layer=extraction_layer_b,
            device=device, task=task,
        )

        _, transplanted_metric = _compute_metric(result["transplanted_preds"], y_q, task)

        improvement = transplanted_metric - baseline_b
        gap = baseline_a - baseline_b
        convergence = improvement / gap if abs(gap) > 1e-6 else float("nan")

        steps.append({
            "n_features": step,
            "features_transplanted": [p["feature_a"] for p in active_projs],
            "projection_r2s": [p["r2"] for p in active_projs],
            "transplanted_metric": float(transplanted_metric),
            "improvement": float(improvement),
            "convergence": float(convergence),
        })

        logger.info(
            "  Transplant step %d/%d: metric=%.4f imp=%.4f conv=%.3f",
            step, len(projections), transplanted_metric, improvement, convergence,
        )

    return {
        "type": "transplant",
        "model_a": model_a,
        "model_b": model_b,
        "dataset": dataset,
        "task": task,
        "baseline_a": float(baseline_a),
        "baseline_b": float(baseline_b),
        "gap": float(baseline_a - baseline_b),
        "n_unmatched_a": len(mnn_pair["unmatched_a"]),
        "n_projectable": len(projections),
        "steps": steps,
    }


def _run_transplant_intervention(
    model_key: str,
    X_ctx: np.ndarray,
    y_ctx: np.ndarray,
    X_q: np.ndarray,
    y_q: np.ndarray,
    projections: List[Dict],
    extraction_layer: int,
    device: str = "cuda",
    task: str = "classification",
) -> Dict:
    """Run B with transplanted concept directions injected into hidden state.

    For each projection, adds (target_activation * projection_vector) along
    the concept direction in B's embedding space.
    """
    import torch
    from scripts.intervene_sae import INTERVENE_FN

    # We need to add delta = sum(target * projection) to B's hidden state.
    # To reuse existing hook machinery, we compute a fixed delta vector
    # and inject it at the extraction layer output.

    # Compute fixed delta per query sample (same for all since projection is constant)
    total_delta = np.zeros(projections[0]["projection"].shape, dtype=np.float32)
    for p in projections:
        total_delta += p["target_activation"] * p["projection"]

    delta_tensor = torch.tensor(total_delta, dtype=torch.float32, device=device)

    # Run model with delta injection (model-specific)
    if model_key == "tabpfn":
        return _transplant_tabpfn(
            X_ctx, y_ctx, X_q, y_q, delta_tensor, extraction_layer, device, task,
        )
    elif model_key == "tabdpt":
        return _transplant_tabdpt(
            X_ctx, y_ctx, X_q, y_q, delta_tensor, extraction_layer, device, task,
        )
    elif model_key == "mitra":
        return _transplant_mitra(
            X_ctx, y_ctx, X_q, y_q, delta_tensor, extraction_layer, device, task,
        )
    elif model_key == "tabicl":
        return _transplant_tabicl(
            X_ctx, y_ctx, X_q, y_q, delta_tensor, extraction_layer, device, task,
        )
    elif model_key == "hyperfast":
        return _transplant_hyperfast(
            X_ctx, y_ctx, X_q, y_q, delta_tensor, extraction_layer, device, task,
        )
    else:
        raise ValueError(f"Transplant not supported for {model_key}")


def _transplant_tabpfn(X_ctx, y_ctx, X_q, y_q, delta, extraction_layer, device, task):
    """Transplant intervention for TabPFN."""
    import torch
    from models.tabpfn_utils import load_tabpfn

    clf = load_tabpfn(task=task, device=device, n_estimators=1)
    clf.fit(X_ctx, y_ctx)
    layers = clf.model_.transformer_encoder.layers

    # Get baseline
    with torch.no_grad():
        if task == "regression":
            baseline = clf.predict(X_q)
        else:
            baseline = clf.predict_proba(X_q)

    # Inject delta at extraction layer
    def modify_hook(module, input, output):
        if isinstance(output, torch.Tensor) and output.ndim == 4:
            output = output.clone()
            output[0] += delta.unsqueeze(0).unsqueeze(1)
            return output
        return output

    handle = layers[extraction_layer].register_forward_hook(modify_hook)
    try:
        with torch.no_grad():
            if task == "regression":
                transplanted = clf.predict(X_q)
            else:
                transplanted = clf.predict_proba(X_q)
    finally:
        handle.remove()

    return {
        "baseline_preds": np.asarray(baseline),
        "transplanted_preds": np.asarray(transplanted),
        "y_query": np.asarray(y_q),
    }


def _transplant_tabdpt(X_ctx, y_ctx, X_q, y_q, delta, extraction_layer, device, task):
    """Transplant intervention for TabDPT."""
    import torch
    from tabdpt import TabDPTClassifier, TabDPTRegressor

    if task == "regression":
        clf = TabDPTRegressor(device=device, compile=False)
    else:
        clf = TabDPTClassifier(device=device, compile=False)
    clf.fit(X_ctx, y_ctx)
    encoder_layers = clf.model.transformer_encoder

    with torch.no_grad():
        if task == "regression":
            baseline = clf.predict(X_q)
        else:
            baseline = clf.predict_proba(X_q)

    def modify_hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            out = out.clone()
            if out.ndim == 3:
                out += delta.unsqueeze(0).unsqueeze(1)
            elif out.ndim == 2:
                out += delta.unsqueeze(0)
            if isinstance(output, tuple):
                return (out,) + output[1:]
            return out
        return output

    handle = encoder_layers[extraction_layer].register_forward_hook(modify_hook)
    try:
        with torch.no_grad():
            if task == "regression":
                transplanted = clf.predict(X_q)
            else:
                transplanted = clf.predict_proba(X_q)
    finally:
        handle.remove()

    return {
        "baseline_preds": np.asarray(baseline),
        "transplanted_preds": np.asarray(transplanted),
        "y_query": np.asarray(y_q),
    }


def _transplant_mitra(X_ctx, y_ctx, X_q, y_q, delta, extraction_layer, device, task):
    """Transplant intervention for Mitra."""
    import torch

    n_features = X_q.shape[1]
    max_context = max(100, 200_000 // max(n_features, 1))
    if len(X_ctx) > max_context:
        X_ctx = X_ctx[:max_context]
        y_ctx = y_ctx[:max_context]

    if task == "regression":
        from autogluon.tabular.models.mitra.sklearn_interface import MitraRegressor
        clf = MitraRegressor(device=device, n_estimators=1, fine_tune=False)
    else:
        from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier
        clf = MitraClassifier(device=device, n_estimators=1, fine_tune=False)
    clf.fit(X_ctx, y_ctx)
    torch.cuda.empty_cache()

    trainer = clf.trainers[0]
    layers = trainer.model.layers
    rng_state = trainer.rng.get_state()

    with torch.no_grad():
        if task == "regression":
            baseline = clf.predict(X_q)
        else:
            baseline = clf.predict_proba(X_q)

    trainer.rng.set_state(rng_state)

    def modify_hook(module, input, output):
        if not (isinstance(output, tuple) and len(output) >= 2):
            return output
        sup, qry = output[0], output[1]
        modified = list(output)
        if isinstance(sup, torch.Tensor) and sup.ndim == 4:
            sup = sup.clone()
            sup[0, :, 0, :] += delta.unsqueeze(0)
            modified[0] = sup
        if isinstance(qry, torch.Tensor) and qry.ndim == 4:
            qry = qry.clone()
            qry[0, :, 0, :] += delta.unsqueeze(0)
            modified[1] = qry
        return tuple(modified)

    handle = layers[extraction_layer].register_forward_hook(modify_hook)
    try:
        with torch.no_grad():
            if task == "regression":
                transplanted = clf.predict(X_q)
            else:
                transplanted = clf.predict_proba(X_q)
    finally:
        handle.remove()

    return {
        "baseline_preds": np.asarray(baseline),
        "transplanted_preds": np.asarray(transplanted),
        "y_query": np.asarray(y_q),
    }


def _transplant_tabicl(X_ctx, y_ctx, X_q, y_q, delta, extraction_layer, device, task):
    """Transplant intervention for TabICL."""
    import torch
    from tabicl import TabICLClassifier

    clf = TabICLClassifier(device=device, n_estimators=1)
    clf.fit(X_ctx, y_ctx)
    blocks = clf.model_.icl_predictor.tf_icl.blocks

    with torch.no_grad():
        baseline = clf.predict_proba(X_q)

    def modify_hook(module, input, output):
        if isinstance(output, torch.Tensor) and output.ndim == 3:
            output = output.clone()
            output += delta.unsqueeze(0).unsqueeze(0)
            return output
        return output

    handle = blocks[extraction_layer].register_forward_hook(modify_hook)
    try:
        with torch.no_grad():
            transplanted = clf.predict_proba(X_q)
    finally:
        handle.remove()

    return {
        "baseline_preds": np.asarray(baseline),
        "transplanted_preds": np.asarray(transplanted),
        "y_query": np.asarray(y_q),
    }


def _transplant_hyperfast(X_ctx, y_ctx, X_q, y_q, delta, extraction_layer, device, task):
    """Transplant intervention for HyperFast."""
    import torch
    import torch.nn.functional as F
    from hyperfast.hyperfast import forward_main_network, transform_data_for_main_network
    from models.hyperfast_embeddings import HyperFastEmbeddingExtractor

    extractor = HyperFastEmbeddingExtractor(device=device)
    extractor.fit(X_ctx, y_ctx)
    clf = extractor._model

    X_q_t = torch.tensor(X_q, dtype=torch.float32).to(device)

    baseline_outputs = []
    transplanted_outputs = []

    for jj in range(len(clf._main_networks)):
        main_network = clf._move_to_device(clf._main_networks[jj])
        rf = clf._move_to_device(clf._rfs[jj])
        pca = clf._move_to_device(clf._pcas[jj])

        if clf.feature_bagging:
            X_b = X_q_t[:, clf.selected_features[jj]]
        else:
            X_b = X_q_t

        X_transformed = transform_data_for_main_network(
            X=X_b, cfg=clf._cfg, rf=rf, pca=pca,
        )

        with torch.no_grad():
            outputs_base, _ = forward_main_network(X_transformed, main_network)
        baseline_outputs.append(F.softmax(outputs_base, dim=1).cpu().numpy())

        with torch.no_grad():
            x = X_transformed
            for layer_idx, (weight, bias) in enumerate(main_network):
                weight = clf._move_to_device(weight)
                bias = clf._move_to_device(bias)
                x_new = F.linear(x, weight, bias)
                if layer_idx < len(main_network) - 1:
                    x_new = F.relu(x_new)
                    if x_new.shape[-1] == x.shape[-1]:
                        x = x + x_new
                    else:
                        x = x_new
                else:
                    x = x_new

                if layer_idx == extraction_layer:
                    x = x + delta.unsqueeze(0)

            transplanted_outputs.append(F.softmax(x, dim=1).cpu().numpy())

    return {
        "baseline_preds": np.mean(baseline_outputs, axis=0),
        "transplanted_preds": np.mean(transplanted_outputs, axis=0),
        "y_query": np.asarray(y_q),
    }


# ── Convergence Analysis ──────────────────────────────────────────────────


def analyze_convergence(results: List[Dict]) -> Dict:
    """Compute convergence metrics across all intervention results.

    For each intervention type, the convergence ratio measures what fraction
    of the performance gap is closed by the intervention.
    """
    ablation_ratios = []
    boost_ratios = []
    transplant_ratios = []

    for r in results:
        if not r.get("steps"):
            continue

        final_step = r["steps"][-1]
        conv = final_step.get("convergence", float("nan"))
        if np.isnan(conv):
            continue

        itype = r.get("type", "unknown")
        if itype == "ablation":
            ablation_ratios.append(conv)
        elif itype == "boost":
            boost_ratios.append(conv)
        elif itype == "transplant":
            transplant_ratios.append(conv)

    def stats(values):
        if not values:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "n": 0}
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "n": len(values),
        }

    return {
        "ablation": stats(ablation_ratios),
        "boost": stats(boost_ratios),
        "transplant": stats(transplant_ratios),
    }


# ── Plotting ──────────────────────────────────────────────────────────────


def plot_dose_response(
    results: List[Dict],
    intervention_type: str,
    output_dir: Path = DEFAULT_FIGURES_DIR,
):
    """Plot dose-response curves for a given intervention type."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    type_results = [r for r in results if r.get("type") == intervention_type and r.get("steps")]
    if not type_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: convergence curves
    ax = axes[0]
    for r in type_results:
        label = f"{DISPLAY_NAMES.get(r['model_a'], r['model_a'])}→" \
                f"{DISPLAY_NAMES.get(r['model_b'], r['model_b'])}: {r['dataset']}"
        steps = r["steps"]
        x = [s["n_features"] for s in steps]
        y = [s["convergence"] for s in steps]
        ax.plot(x, y, marker=".", markersize=4, label=label, alpha=0.7)

    ax.axhline(1.0, color="red", ls="--", lw=0.8, label="Full gap closed")
    ax.axhline(0.0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Convergence ratio")
    ax.set_title(f"{intervention_type.title()} Dose-Response")
    ax.legend(fontsize=6, loc="best")

    # Right: metric curves
    ax = axes[1]
    for r in type_results:
        steps = r["steps"]
        x = [s["n_features"] for s in steps]
        metric_key = {
            "ablation": "ablated_metric",
            "boost": "boosted_metric",
            "transplant": "transplanted_metric",
        }[intervention_type]
        y = [s[metric_key] for s in steps]
        label = f"{r['dataset']}"

        ax.plot(x, y, marker=".", markersize=4, label=label, alpha=0.7)
        ax.axhline(r["baseline_a"], color="green", ls=":", lw=0.5, alpha=0.3)
        ax.axhline(r["baseline_b"], color="blue", ls=":", lw=0.5, alpha=0.3)

    ax.set_xlabel("Number of features")
    ax.set_ylabel("Metric value")
    ax.set_title(f"{intervention_type.title()} Performance Trajectories")
    ax.legend(fontsize=6, loc="best")

    fig.tight_layout()
    path = output_dir / f"dose_response_{intervention_type}.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s dose-response to %s", intervention_type, path)


def plot_convergence_summary(
    convergence: Dict,
    output_dir: Path = DEFAULT_FIGURES_DIR,
):
    """Plot convergence comparison across intervention types."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    types = ["ablation", "boost", "transplant"]
    means = [convergence[t]["mean"] for t in types]
    stds = [convergence[t]["std"] for t in types]
    ns = [convergence[t]["n"] for t in types]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    bars = ax.bar(types, means, yerr=stds, color=colors, edgecolor="k",
                  linewidth=0.5, capsize=5)

    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={n}", ha="center", va="bottom", fontsize=9)

    ax.axhline(1.0, color="red", ls="--", lw=0.8, alpha=0.5, label="Full gap closed")
    ax.set_ylabel("Mean convergence ratio")
    ax.set_title("Causal Intervention Convergence")
    ax.legend()
    fig.tight_layout()

    path = output_dir / "convergence_summary.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved convergence summary to %s", path)


# ── CLI ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Causal concept interventions: ablation, boost, and transplant"
    )
    parser.add_argument("--model-a", type=str, choices=INTERVENTION_MODELS)
    parser.add_argument("--model-b", type=str, choices=INTERVENTION_MODELS)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--top-targets", type=int, default=10,
                        help="Auto-select top N targets from diagnostic")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--min-gap", type=float, default=0.05)
    parser.add_argument("--ablation-only", action="store_true")
    parser.add_argument("--boost-only", action="store_true")
    parser.add_argument("--transplant-only", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC_PATH)
    parser.add_argument("--perf-csv", type=Path, default=DEFAULT_PERF_CSV)
    parser.add_argument("--mnn-path", type=Path, default=DEFAULT_MNN_PATH)
    parser.add_argument("--fingerprint-dir", type=Path, default=DEFAULT_FINGERPRINT_DIR)
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Determine which intervention types to run
    run_ablation = not args.boost_only and not args.transplant_only
    run_boost = not args.ablation_only and not args.transplant_only
    run_transplant = not args.ablation_only and not args.boost_only

    # Determine targets
    if args.model_a and args.model_b:
        from scripts.extract_layer_embeddings import get_dataset_task
        from data.extended_loader import TABARENA_DATASETS

        datasets = args.datasets or sorted(TABARENA_DATASETS.keys())
        targets = []
        for ds in datasets:
            task = get_dataset_task(ds)
            targets.append({
                "model_a": args.model_a,
                "model_b": args.model_b,
                "dataset": ds,
                "task": task,
            })
    elif args.all_pairs:
        from scripts.extract_layer_embeddings import get_dataset_task
        from data.extended_loader import TABARENA_DATASETS

        datasets = args.datasets or sorted(TABARENA_DATASETS.keys())
        targets = []
        for a, b in combinations(INTERVENTION_MODELS, 2):
            for ds in datasets:
                task = get_dataset_task(ds)
                targets.append({
                    "model_a": a,
                    "model_b": b,
                    "dataset": ds,
                    "task": task,
                })
    else:
        # Auto-select from diagnostic
        raw_targets = select_targets(
            args.diagnostic, args.perf_csv,
            min_gap=args.min_gap, top_n=args.top_targets,
        )
        targets = []
        from scripts.extract_layer_embeddings import get_dataset_task
        for t in raw_targets:
            t["task"] = get_dataset_task(t["dataset"])
            targets.append(t)

    logger.info("Running %d target cases", len(targets))

    all_results = []
    for i, target in enumerate(targets):
        model_a = target["model_a"]
        model_b = target["model_b"]
        dataset = target["dataset"]
        task = target.get("task", "classification")

        display_a = DISPLAY_NAMES.get(model_a, model_a)
        display_b = DISPLAY_NAMES.get(model_b, model_b)
        logger.info(
            "\n[%d/%d] %s → %s on %s (%s)",
            i + 1, len(targets), display_a, display_b, dataset, task,
        )

        # Run ablation
        if run_ablation:
            logger.info("  Running ablation...")
            try:
                result = dose_response_ablation(
                    model_a, model_b, dataset, task,
                    device=args.device, max_steps=args.max_steps,
                    mnn_path=args.mnn_path, fp_dir=args.fingerprint_dir,
                )
                all_results.append(result)
            except Exception as e:
                logger.error("  Ablation failed: %s", e)

        # Run boost
        if run_boost:
            logger.info("  Running boost...")
            try:
                result = dose_response_boost(
                    model_a, model_b, dataset, task,
                    device=args.device, max_steps=args.max_steps,
                    mnn_path=args.mnn_path, fp_dir=args.fingerprint_dir,
                )
                all_results.append(result)
            except Exception as e:
                logger.error("  Boost failed: %s", e)

        # Run transplant
        if run_transplant:
            logger.info("  Running transplant...")
            try:
                result = dose_response_transplant(
                    model_a, model_b, dataset, task,
                    device=args.device, max_steps=min(args.max_steps, 10),
                    mnn_path=args.mnn_path, fp_dir=args.fingerprint_dir,
                    training_dir=args.training_dir,
                )
                all_results.append(result)
            except Exception as e:
                logger.error("  Transplant failed: %s", e)

    # Convergence analysis
    convergence = analyze_convergence(all_results)

    # Save results
    output = {
        "results": all_results,
        "convergence": convergence,
        "n_targets": len(targets),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("\nSaved results to %s", args.output)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Causal Intervention Summary")
    print(f"{'='*80}")
    for itype in ["ablation", "boost", "transplant"]:
        c = convergence[itype]
        if c["n"] > 0:
            print(f"  {itype:12s}: mean_conv={c['mean']:.3f} ± {c['std']:.3f} (n={c['n']})")
        else:
            print(f"  {itype:12s}: no results")

    # Plots
    if not args.no_plots:
        try:
            for itype in ["ablation", "boost", "transplant"]:
                plot_dose_response(all_results, itype)
            if convergence["ablation"]["n"] > 0:
                plot_convergence_summary(convergence)
        except ImportError:
            logger.warning("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
