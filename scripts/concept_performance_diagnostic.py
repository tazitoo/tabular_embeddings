#!/usr/bin/env python3
"""Concept gap ↔ performance gap diagnostic.

Correlates concept differences between model pairs (via MNN matching + SAE
fingerprints) with performance differences across 51 TabArena datasets.

Three phases:
1. Collect: Get per-model, per-dataset predictions and compute metrics.
2. Fingerprint: Compute per-dataset SAE activation profiles for each model.
3. Analyze: Correlate concept gaps with performance gaps.

Usage:
    # Collect performance for one model (GPU required)
    python scripts/concept_performance_diagnostic.py collect --model tabpfn --device cuda

    # Collect all models
    python scripts/concept_performance_diagnostic.py collect --all-models --device cuda

    # Compute fingerprints for all models
    python scripts/concept_performance_diagnostic.py fingerprint

    # Full diagnostic analysis (requires performance CSV + fingerprints)
    python scripts/concept_performance_diagnostic.py analyze

    # Quick check on one pair
    python scripts/concept_performance_diagnostic.py analyze --pair tabpfn mitra
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Output paths
DEFAULT_PERF_CSV = PROJECT_ROOT / "output" / "model_performance.csv"
DEFAULT_FINGERPRINT_DIR = PROJECT_ROOT / "output" / "concept_fingerprints"
DEFAULT_MNN_PATH = PROJECT_ROOT / "output" / "sae_feature_matching_mnn_t0.001_n500.json"
DEFAULT_DIAGNOSTIC_PATH = PROJECT_ROOT / "output" / "concept_performance_diagnostic.json"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "output" / "figures"

# Models with intervention support (can get predictions via intervene_sae)
INTERVENTION_MODELS = ["tabpfn", "mitra", "tabicl", "tabdpt", "hyperfast"]

# All models with SAE checkpoints (for fingerprints)
ALL_MODELS = ["tabpfn", "mitra", "tabicl", "tabdpt", "hyperfast", "carte", "tabula8b"]

# Models that can produce predictions (intervention-ready or with predict API)
PREDICTABLE_MODELS = ["tabpfn", "mitra", "tabicl", "tabdpt", "hyperfast", "carte"]

# Display name mapping (matching MNN file convention)
DISPLAY_NAMES = {
    "tabpfn": "TabPFN",
    "mitra": "Mitra",
    "tabicl": "TabICL",
    "tabdpt": "TabDPT",
    "hyperfast": "HyperFast",
    "carte": "CARTE",
    "tabula8b": "Tabula-8B",
}

# Reverse mapping
KEY_FROM_DISPLAY = {v: k for k, v in DISPLAY_NAMES.items()}


# ── Performance Collection ─────────────────────────────────────────────────


def _load_splits(dataset: str, task: str):
    """Load and split a TabArena dataset for evaluation.

    Returns (X_context, y_context, X_query, y_query).
    """
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


def compute_metric(
    preds: np.ndarray,
    y_true: np.ndarray,
    task: str,
) -> Tuple[str, float]:
    """Compute the appropriate TabArena metric.

    Returns (metric_name, metric_value) tuple.
    Binary classification: AUC (higher is better).
    Multiclass classification: log_loss (lower is better, negated for consistency).
    Regression: RMSE (lower is better, negated for consistency).

    All returned values follow "higher is better" convention.
    """
    if task == "regression":
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        return "neg_rmse", -rmse
    else:
        n_classes = len(np.unique(y_true))
        if n_classes == 2 and preds.ndim == 2:
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(y_true, preds[:, 1])
                return "auc", auc
            except ValueError:
                # All one class in y_true
                return "auc", float("nan")
        else:
            from sklearn.metrics import log_loss
            try:
                ll = log_loss(y_true, preds, labels=np.arange(preds.shape[1]))
                return "neg_logloss", -ll
            except ValueError:
                return "neg_logloss", float("nan")


def predict_intervention_model(
    model_key: str,
    dataset: str,
    task: str,
    device: str = "cuda",
) -> Dict:
    """Get predictions from an intervention-ready model (baseline, no ablation)."""
    from scripts.intervene_sae import intervene

    X_ctx, y_ctx, X_q, y_q = _load_splits(dataset, task)

    results = intervene(
        model_key=model_key,
        X_context=X_ctx,
        y_context=y_ctx,
        X_query=X_q,
        y_query=y_q,
        ablate_features=[],  # No ablation — baseline only
        device=device,
        task=task,
    )

    return {
        "preds": results["baseline_preds"],
        "y_true": results["y_query"],
        "task": task,
        "n_query": len(y_q),
    }


def predict_carte(
    dataset: str,
    task: str,
    device: str = "cuda",
) -> Dict:
    """Get predictions from CARTE model."""
    X_ctx, y_ctx, X_q, y_q = _load_splits(dataset, task)

    if task == "regression":
        from carte_ai import CARTERegressor
        clf = CARTERegressor(device=device, num_model=1)
        clf.fit(X_ctx, y_ctx)
        preds = clf.predict(X_q)
    else:
        from carte_ai import CARTEClassifier
        clf = CARTEClassifier(device=device, num_model=1)
        clf.fit(X_ctx, y_ctx)
        preds = clf.predict_proba(X_q)

    return {
        "preds": np.asarray(preds),
        "y_true": np.asarray(y_q),
        "task": task,
        "n_query": len(y_q),
    }


def predict_tabula8b(
    dataset: str,
    task: str,
    device: str = "cuda",
) -> Dict:
    """Get predictions from Tabula-8B model."""
    X_ctx, y_ctx, X_q, y_q = _load_splits(dataset, task)

    from models.tabula_embeddings import TabulaExtractor
    extractor = TabulaExtractor(device=device)

    if task == "regression":
        preds = extractor.predict(X_ctx, y_ctx, X_q, task="regression")
    else:
        preds = extractor.predict_proba(X_ctx, y_ctx, X_q)

    return {
        "preds": np.asarray(preds),
        "y_true": np.asarray(y_q),
        "task": task,
        "n_query": len(y_q),
    }


def collect_performance(
    model_key: str,
    device: str = "cuda",
    output_path: Path = DEFAULT_PERF_CSV,
    datasets: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Collect performance for one model across all TabArena datasets.

    Appends results to existing CSV if present (idempotent — skips existing entries).
    """
    from scripts.extract_layer_embeddings import get_dataset_task
    from data.extended_loader import TABARENA_DATASETS

    if datasets is None:
        datasets = sorted(TABARENA_DATASETS.keys())

    # Load existing results if any
    existing = pd.DataFrame()
    if output_path.exists():
        existing = pd.read_csv(output_path)

    rows = []
    for ds in datasets:
        # Skip if already collected successfully (not error rows)
        if not existing.empty:
            mask = (existing["model"] == model_key) & (existing["dataset"] == ds)
            successful = mask & (existing["metric_name"] != "error")
            if successful.any():
                logger.info("  %s/%s: already collected, skipping", model_key, ds)
                continue

        task = get_dataset_task(ds)

        # Classification-only models: skip regression datasets
        CLASSIFICATION_ONLY = {"tabicl", "hyperfast"}
        if model_key in CLASSIFICATION_ONLY and task == "regression":
            logger.info("  %s/%s: skipping (regression, model is classification-only)", model_key, ds)
            continue

        logger.info("  %s/%s (%s)...", model_key, ds, task)

        try:
            if model_key in INTERVENTION_MODELS:
                result = predict_intervention_model(model_key, ds, task, device)
            elif model_key == "carte":
                result = predict_carte(ds, task, device)
            else:
                raise ValueError(f"Unknown model: {model_key}")

            metric_name, metric_value = compute_metric(
                result["preds"], result["y_true"], task,
            )

            rows.append({
                "model": model_key,
                "dataset": ds,
                "task": task,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "n_query": result["n_query"],
            })
            logger.info("    %s = %.4f", metric_name, metric_value)

        except Exception as e:
            logger.error("    FAILED: %s", e)
            rows.append({
                "model": model_key,
                "dataset": ds,
                "task": task,
                "metric_name": "error",
                "metric_value": float("nan"),
                "n_query": 0,
            })

    # Append new results
    new_df = pd.DataFrame(rows)
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(combined), output_path)

    return combined


# ── Concept Fingerprints ───────────────────────────────────────────────────


def compute_all_fingerprints(
    output_dir: Path = DEFAULT_FINGERPRINT_DIR,
    models: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Compute and save concept fingerprints for all models.

    Uses existing concept_fingerprint.py machinery. CPU only, ~30s per model.

    Returns dict mapping model_key -> fingerprint data.
    """
    from scripts.concept_fingerprint import compute_fingerprints

    if models is None:
        models = ALL_MODELS

    output_dir.mkdir(parents=True, exist_ok=True)
    all_fp = {}

    for model_key in models:
        output_path = output_dir / f"{model_key}_fingerprints.json"

        # Load cached if exists
        if output_path.exists():
            logger.info("Loading cached fingerprints for %s", model_key)
            with open(output_path) as f:
                all_fp[model_key] = json.load(f)
            continue

        logger.info("Computing fingerprints for %s...", model_key)
        try:
            fp = compute_fingerprints(model_key, device="cpu")
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", model_key, e)
            continue

        # Convert numpy arrays for JSON serialization
        save_data = {
            "model": model_key,
            "hidden_dim": fp["hidden_dim"],
            "n_datasets": fp["n_datasets"],
            "alive_features": fp["alive_features"],
            "bands": fp["bands"],
            "global_mean": fp["global_mean"].tolist(),
            "dataset_means": {
                ds: m.tolist() for ds, m in fp["dataset_means"].items()
            },
            "dataset_deviations": {
                ds: d.tolist() for ds, d in fp["dataset_deviations"].items()
            },
        }

        with open(output_path, "w") as f:
            json.dump(save_data, f)
        logger.info("  Saved to %s", output_path)

        all_fp[model_key] = save_data

    return all_fp


def load_fingerprints(
    fingerprint_dir: Path = DEFAULT_FINGERPRINT_DIR,
    models: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Load pre-computed fingerprints from disk."""
    if models is None:
        models = ALL_MODELS

    result = {}
    for model_key in models:
        fp_path = fingerprint_dir / f"{model_key}_fingerprints.json"
        if fp_path.exists():
            with open(fp_path) as f:
                result[model_key] = json.load(f)
        else:
            logger.warning("No fingerprints found for %s at %s", model_key, fp_path)

    return result


# ── Concept Gap Computation ────────────────────────────────────────────────


def load_mnn_matching(path: Path = DEFAULT_MNN_PATH) -> Dict:
    """Load MNN feature matching results."""
    with open(path) as f:
        return json.load(f)


def _pair_key(model_a_display: str, model_b_display: str) -> str:
    """Construct pair key in alphabetical order (matching MNN convention)."""
    a, b = sorted([model_a_display, model_b_display])
    return f"{a}__{b}"


def compute_concept_gaps(
    fingerprints: Dict[str, Dict],
    mnn_data: Dict,
    model_a: str,
    model_b: str,
) -> pd.DataFrame:
    """Compute concept gap metrics between a model pair across all datasets.

    For each dataset, measures how much the two models' SAE activations differ
    on features that are unique to each model (MNN-unmatched) and on features
    that are shared (MNN-matched).

    Returns DataFrame with columns:
        dataset, unmatched_act_a, unmatched_act_b, concept_asymmetry,
        matched_differential, cosine_sim, and per-band (S1-S5) versions.
    """
    fp_a = fingerprints.get(model_a)
    fp_b = fingerprints.get(model_b)
    if fp_a is None or fp_b is None:
        raise ValueError(f"Missing fingerprints for {model_a} or {model_b}")

    display_a = DISPLAY_NAMES[model_a]
    display_b = DISPLAY_NAMES[model_b]

    # Get MNN pair data (keys are alphabetically ordered)
    pk = _pair_key(display_a, display_b)
    pair_data = mnn_data["pairs"].get(pk)
    if pair_data is None:
        raise ValueError(f"No MNN data for pair {pk}")

    # Determine which direction the pair is stored
    sorted_displays = sorted([display_a, display_b])
    swapped = sorted_displays[0] != display_a

    if swapped:
        unmatched_a_indices = pair_data["unmatched_b"]
        unmatched_b_indices = pair_data["unmatched_a"]
        matches = [
            {"idx_a": m["idx_b"], "idx_b": m["idx_a"], "r": m["r"]}
            for m in pair_data["matches"]
        ]
    else:
        unmatched_a_indices = pair_data["unmatched_a"]
        unmatched_b_indices = pair_data["unmatched_b"]
        matches = pair_data["matches"]

    n_alive_a = pair_data["n_alive_b"] if swapped else pair_data["n_alive_a"]
    n_alive_b = pair_data["n_alive_a"] if swapped else pair_data["n_alive_b"]

    bands_a = fp_a["bands"]
    bands_b = fp_b["bands"]
    hidden_a = fp_a["hidden_dim"]
    hidden_b = fp_b["hidden_dim"]

    # Find common datasets
    ds_a = set(fp_a["dataset_means"].keys())
    ds_b = set(fp_b["dataset_means"].keys())
    common_datasets = sorted(ds_a & ds_b)

    rows = []
    for ds in common_datasets:
        mean_a = np.array(fp_a["dataset_means"][ds])
        mean_b = np.array(fp_b["dataset_means"][ds])

        # Overall concept gap metrics
        row = {"dataset": ds}

        # Unmatched activation: L1 of A-only features, normalized by n_alive
        if unmatched_a_indices:
            ua = np.array([mean_a[i] for i in unmatched_a_indices])
            row["unmatched_act_a"] = float(np.sum(np.abs(ua))) / max(n_alive_a, 1)
        else:
            row["unmatched_act_a"] = 0.0

        if unmatched_b_indices:
            ub = np.array([mean_b[i] for i in unmatched_b_indices])
            row["unmatched_act_b"] = float(np.sum(np.abs(ub))) / max(n_alive_b, 1)
        else:
            row["unmatched_act_b"] = 0.0

        row["concept_asymmetry"] = row["unmatched_act_a"] - row["unmatched_act_b"]

        # Matched differential: mean |act_A[i] - act_B[j]| for matched pairs
        if matches:
            diffs = [abs(mean_a[m["idx_a"]] - mean_b[m["idx_b"]]) for m in matches]
            row["matched_differential"] = float(np.mean(diffs))
        else:
            row["matched_differential"] = 0.0

        # Cosine similarity of full alive-feature fingerprint vectors
        alive_a = fp_a["alive_features"]
        alive_b = fp_b["alive_features"]
        vec_a = np.array([mean_a[i] for i in alive_a])
        vec_b = np.array([mean_b[i] for i in alive_b])

        # For cosine sim, project via matching: build aligned vectors
        if matches:
            matched_a = np.array([mean_a[m["idx_a"]] for m in matches])
            matched_b = np.array([mean_b[m["idx_b"]] for m in matches])
            norm_a = np.linalg.norm(matched_a)
            norm_b = np.linalg.norm(matched_b)
            if norm_a > 1e-12 and norm_b > 1e-12:
                row["cosine_sim"] = float(
                    np.dot(matched_a, matched_b) / (norm_a * norm_b)
                )
            else:
                row["cosine_sim"] = 0.0
        else:
            row["cosine_sim"] = 0.0

        # Per-Matryoshka-band breakdown
        from scripts.concept_importance import feature_to_band

        for band_name in ["S1", "S2", "S3", "S4", "S5"]:
            # Unmatched in this band
            ua_band = [
                i for i in unmatched_a_indices
                if feature_to_band(i, bands_a) == band_name
            ]
            ub_band = [
                i for i in unmatched_b_indices
                if feature_to_band(i, bands_b) == band_name
            ]

            if ua_band:
                row[f"unmatched_act_a_{band_name}"] = float(
                    np.sum(np.abs(np.array([mean_a[i] for i in ua_band])))
                ) / max(len(ua_band), 1)
            else:
                row[f"unmatched_act_a_{band_name}"] = 0.0

            if ub_band:
                row[f"unmatched_act_b_{band_name}"] = float(
                    np.sum(np.abs(np.array([mean_b[i] for i in ub_band])))
                ) / max(len(ub_band), 1)
            else:
                row[f"unmatched_act_b_{band_name}"] = 0.0

            row[f"concept_asymmetry_{band_name}"] = (
                row[f"unmatched_act_a_{band_name}"]
                - row[f"unmatched_act_b_{band_name}"]
            )

            # Matched differential in this band
            band_matches = [
                m for m in matches
                if feature_to_band(m["idx_a"], bands_a) == band_name
            ]
            if band_matches:
                diffs = [
                    abs(mean_a[m["idx_a"]] - mean_b[m["idx_b"]])
                    for m in band_matches
                ]
                row[f"matched_differential_{band_name}"] = float(np.mean(diffs))
            else:
                row[f"matched_differential_{band_name}"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


# ── Statistical Analysis ──────────────────────────────────────────────────


def analyze_concept_performance(
    performance: pd.DataFrame,
    fingerprints: Dict[str, Dict],
    mnn_data: Dict,
    pairs: Optional[List[Tuple[str, str]]] = None,
) -> Dict:
    """Correlate concept gaps with performance gaps for all model pairs.

    Returns comprehensive diagnostic results.
    """
    from scipy.stats import spearmanr

    if pairs is None:
        # Generate all pairs from models present in both performance and fingerprints
        perf_models = set(performance["model"].unique())
        fp_models = set(fingerprints.keys())
        available = sorted(perf_models & fp_models)
        pairs = []
        for i, a in enumerate(available):
            for b in available[i + 1:]:
                pairs.append((a, b))

    pair_results = {}
    correlation_matrix = {}

    for model_a, model_b in pairs:
        display_a = DISPLAY_NAMES.get(model_a, model_a)
        display_b = DISPLAY_NAMES.get(model_b, model_b)
        pair_label = f"{display_a}__{display_b}"

        logger.info("Analyzing %s vs %s", display_a, display_b)

        # Compute concept gaps
        try:
            gaps = compute_concept_gaps(fingerprints, mnn_data, model_a, model_b)
        except (ValueError, KeyError) as e:
            logger.warning("  Skipping: %s", e)
            continue

        # Get performance gap
        perf_a = performance[performance["model"] == model_a].set_index("dataset")
        perf_b = performance[performance["model"] == model_b].set_index("dataset")

        # Find common datasets with valid performance
        common = sorted(
            set(perf_a.index) & set(perf_b.index) & set(gaps["dataset"])
        )
        if len(common) < 5:
            logger.warning("  Only %d common datasets, skipping", len(common))
            continue

        # Merge performance gap into concept gaps
        perf_gaps = []
        for ds in common:
            if ds in perf_a.index and ds in perf_b.index:
                va = perf_a.loc[ds, "metric_value"]
                vb = perf_b.loc[ds, "metric_value"]
                if isinstance(va, pd.Series):
                    va = va.iloc[0]
                if isinstance(vb, pd.Series):
                    vb = vb.iloc[0]
                perf_gaps.append({"dataset": ds, "perf_gap": float(va) - float(vb)})

        perf_df = pd.DataFrame(perf_gaps)
        merged = gaps.merge(perf_df, on="dataset", how="inner")

        if len(merged) < 5:
            logger.warning("  Only %d merged rows, skipping", len(merged))
            continue

        # Compute correlations
        gap_metrics = [
            "concept_asymmetry", "unmatched_act_a", "unmatched_act_b",
            "matched_differential", "cosine_sim",
        ]
        band_metrics = []
        for band in ["S1", "S2", "S3", "S4", "S5"]:
            band_metrics.append(f"concept_asymmetry_{band}")
            band_metrics.append(f"matched_differential_{band}")

        correlations = {}
        for metric in gap_metrics + band_metrics:
            if metric not in merged.columns:
                continue
            x = merged[metric].values
            y = merged["perf_gap"].values
            # Skip if all values are identical
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                correlations[metric] = {"rho": 0.0, "p_value": 1.0, "n": len(x)}
                continue
            rho, p = spearmanr(x, y)
            correlations[metric] = {"rho": float(rho), "p_value": float(p), "n": int(len(x))}

        # Identify high-leverage datasets
        if "concept_asymmetry" in merged.columns:
            merged["abs_concept"] = merged["concept_asymmetry"].abs()
            merged["abs_perf"] = merged["perf_gap"].abs()
            merged["leverage"] = merged["abs_concept"] * merged["abs_perf"]
            high_leverage = (
                merged.nlargest(5, "leverage")[
                    ["dataset", "perf_gap", "concept_asymmetry", "leverage"]
                ]
                .to_dict(orient="records")
            )
        else:
            high_leverage = []

        pair_results[pair_label] = {
            "model_a": model_a,
            "model_b": model_b,
            "n_datasets": len(merged),
            "correlations": correlations,
            "high_leverage_datasets": high_leverage,
            "data": merged.to_dict(orient="records"),
        }

        # Store top-level correlation for heatmap
        main_corr = correlations.get("concept_asymmetry", {})
        correlation_matrix[pair_label] = main_corr.get("rho", 0.0)

    # Band importance: average |rho| across all pairs for each band metric
    band_importance = {}
    for band in ["S1", "S2", "S3", "S4", "S5"]:
        metric_key = f"concept_asymmetry_{band}"
        rhos = []
        for pr in pair_results.values():
            c = pr["correlations"].get(metric_key, {})
            if c.get("rho") is not None and not np.isnan(c["rho"]):
                rhos.append(abs(c["rho"]))
        band_importance[band] = float(np.mean(rhos)) if rhos else 0.0

    return {
        "pairs": pair_results,
        "correlation_matrix": correlation_matrix,
        "band_importance": band_importance,
        "n_pairs": len(pair_results),
    }


# ── Plotting ───────────────────────────────────────────────────────────────


def plot_scatter(diagnostic: Dict, output_dir: Path = DEFAULT_FIGURES_DIR):
    """Plot concept_asymmetry vs perf_gap scatter for each pair."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    n_pairs = diagnostic["n_pairs"]
    if n_pairs == 0:
        logger.warning("No pairs to plot")
        return

    ncols = min(3, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, (pair_label, pr) in enumerate(sorted(diagnostic["pairs"].items())):
        ax = axes[idx // ncols][idx % ncols]
        data = pd.DataFrame(pr["data"])
        if "concept_asymmetry" not in data.columns or "perf_gap" not in data.columns:
            ax.set_visible(False)
            continue

        ax.scatter(
            data["concept_asymmetry"], data["perf_gap"],
            alpha=0.6, s=20, edgecolors="k", linewidths=0.3,
        )

        corr = pr["correlations"].get("concept_asymmetry", {})
        rho = corr.get("rho", 0)
        p = corr.get("p_value", 1)
        ax.set_title(f"{pair_label}\nρ={rho:.2f}, p={p:.3f}", fontsize=9)
        ax.set_xlabel("Concept asymmetry", fontsize=8)
        ax.set_ylabel("Performance gap (A − B)", fontsize=8)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(0, color="gray", lw=0.5, ls="--")

    # Hide unused axes
    for idx in range(n_pairs, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    path = output_dir / "concept_vs_performance_scatter.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scatter plot to %s", path)


def plot_heatmap(diagnostic: Dict, output_dir: Path = DEFAULT_FIGURES_DIR):
    """Plot model × model correlation heatmap."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build matrix
    models_seen = set()
    for pair_label in diagnostic["pairs"]:
        a, b = pair_label.split("__")
        models_seen.add(a)
        models_seen.add(b)

    models = sorted(models_seen)
    n = len(models)
    if n < 2:
        return

    matrix = np.zeros((n, n))
    for i, ma in enumerate(models):
        for j, mb in enumerate(models):
            if i == j:
                matrix[i, j] = 1.0
                continue
            pk = f"{ma}__{mb}" if ma < mb else f"{mb}__{ma}"
            rho = diagnostic["correlation_matrix"].get(pk, 0.0)
            matrix[i, j] = rho

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(models, fontsize=9)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, label="Spearman ρ (concept_asymmetry vs perf_gap)")
    ax.set_title("Concept Gap ↔ Performance Gap Correlation")
    fig.tight_layout()

    path = output_dir / "concept_performance_heatmap.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap to %s", path)


def plot_band_importance(diagnostic: Dict, output_dir: Path = DEFAULT_FIGURES_DIR):
    """Plot bar chart of correlation strength by Matryoshka scale."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    bands = list(diagnostic["band_importance"].keys())
    values = [diagnostic["band_importance"][b] for b in bands]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(bands)))
    ax.bar(bands, values, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_xlabel("Matryoshka Band")
    ax.set_ylabel("Mean |ρ| across model pairs")
    ax.set_title("Which Concept Scale Correlates Most with Performance?")
    fig.tight_layout()

    path = output_dir / "band_importance.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved band importance plot to %s", path)


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Concept gap ↔ performance gap diagnostic"
    )
    subparsers = parser.add_subparsers(dest="command")

    # collect subcommand
    collect_p = subparsers.add_parser("collect", help="Collect model performance")
    collect_p.add_argument("--model", type=str, choices=ALL_MODELS)
    collect_p.add_argument("--all-models", action="store_true")
    collect_p.add_argument("--device", type=str, default="cuda")
    collect_p.add_argument("--output", type=Path, default=DEFAULT_PERF_CSV)
    collect_p.add_argument("--datasets", nargs="+", default=None)

    # fingerprint subcommand
    fp_p = subparsers.add_parser("fingerprint", help="Compute concept fingerprints")
    fp_p.add_argument("--models", nargs="+", default=None)
    fp_p.add_argument("--output-dir", type=Path, default=DEFAULT_FINGERPRINT_DIR)

    # analyze subcommand
    analyze_p = subparsers.add_parser("analyze", help="Run diagnostic analysis")
    analyze_p.add_argument("--pair", nargs=2, metavar=("MODEL_A", "MODEL_B"))
    analyze_p.add_argument("--perf-csv", type=Path, default=DEFAULT_PERF_CSV)
    analyze_p.add_argument("--fingerprint-dir", type=Path, default=DEFAULT_FINGERPRINT_DIR)
    analyze_p.add_argument("--mnn-path", type=Path, default=DEFAULT_MNN_PATH)
    analyze_p.add_argument("--output", type=Path, default=DEFAULT_DIAGNOSTIC_PATH)
    analyze_p.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "collect":
        if args.all_models:
            for model in ALL_MODELS:
                logger.info("Collecting %s...", model)
                collect_performance(
                    model, device=args.device, output_path=args.output,
                    datasets=args.datasets,
                )
        elif args.model:
            collect_performance(
                args.model, device=args.device, output_path=args.output,
                datasets=args.datasets,
            )
        else:
            parser.error("Specify --model or --all-models")

    elif args.command == "fingerprint":
        compute_all_fingerprints(
            output_dir=args.output_dir, models=args.models,
        )

    elif args.command == "analyze":
        # Load data
        if not args.perf_csv.exists():
            parser.error(f"Performance CSV not found: {args.perf_csv}")
        performance = pd.read_csv(args.perf_csv)

        fingerprints = load_fingerprints(args.fingerprint_dir)
        if not fingerprints:
            parser.error(f"No fingerprints found in {args.fingerprint_dir}")

        mnn_data = load_mnn_matching(args.mnn_path)

        pairs = None
        if args.pair:
            pairs = [(args.pair[0], args.pair[1])]

        diagnostic = analyze_concept_performance(
            performance, fingerprints, mnn_data, pairs=pairs,
        )

        # Save results
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(diagnostic, f, indent=2, default=str)
        logger.info("Saved diagnostic to %s", args.output)

        # Print summary
        print(f"\n{'='*80}")
        print(f"Concept-Performance Diagnostic Summary")
        print(f"{'='*80}")
        print(f"Pairs analyzed: {diagnostic['n_pairs']}")
        print(f"\nBand importance (mean |ρ| across pairs):")
        for band, imp in diagnostic["band_importance"].items():
            print(f"  {band}: {imp:.3f}")

        print(f"\nCorrelation matrix (concept_asymmetry vs perf_gap):")
        for pair_label, rho in sorted(diagnostic["correlation_matrix"].items()):
            pr = diagnostic["pairs"][pair_label]
            corr = pr["correlations"].get("concept_asymmetry", {})
            p = corr.get("p_value", 1.0)
            sig = "*" if p < 0.05 else ""
            print(f"  {pair_label:40s} ρ={rho:+.3f} (p={p:.3f}){sig}")

        # Plots
        if not args.no_plots:
            try:
                plot_scatter(diagnostic)
                plot_heatmap(diagnostic)
                plot_band_importance(diagnostic)
            except ImportError:
                logger.warning("matplotlib not available, skipping plots")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
