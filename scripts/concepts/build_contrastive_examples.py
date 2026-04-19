#!/usr/bin/env python3
"""Build contrastive example CSVs for SAE feature labeling.

For each (model, feature, dataset): find the top-K activating rows and
K nearest non-activating rows in embedding space. Save as CSV with the
raw data values, activation, and per-row stats (PMI, surprise, compression).

Also saves a preprocessing_context.json describing how each model
transforms raw data before embedding, so labeling agents understand
what the SAE actually sees.

Output:
    output/contrastive_examples/{model}/f{feat}_{dataset}.csv
    output/contrastive_examples/{model}/preprocessing_context.json

Usage:
    python -m scripts.concepts.build_contrastive_examples \
        --model mitra --features 6 11 36 86 92

    # All features for a model (slow)
    python -m scripts.concepts.build_contrastive_examples --model mitra --all
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

from scripts.concepts.dataset_quality_cache import (
    DEFAULT_CACHE_PATH,
    DEFAULT_SCORE_CONFIG,
    cache_entry_for_feature,
    load_quality_cache,
    select_top_datasets,
)
from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import (
    load_sae, load_test_embeddings, SPLITS_PATH,
)

SAE_DATA_DIR = PROJECT_ROOT / "output" / "sae_training_round10"
PMI_DIR = PROJECT_ROOT / "output" / "pmi_cache"
SURPRISE_DIR = PROJECT_ROOT / "output" / "surprise_cache"
COMPRESSION_DIR = PROJECT_ROOT / "output" / "compression_cache"
PYMFE_PATH = PROJECT_ROOT / "output" / "pymfe_tabarena_cache.json"
BASELINE_PRED_DIR = PROJECT_ROOT / "output" / "baseline_predictions"
OUTPUT_DIR = PROJECT_ROOT / "output" / "contrastive_examples"

PREPROCESSING_CONTEXT = {
    "mitra": {
        "model": "Mitra",
        "architecture": "2D attention transformer (72M params), in-context learning with 1-epoch finetune",
        "preprocessing": [
            "AutoGluon AutoMLPipelineFeatureGenerator: auto-detects column types, "
            "ordinal-encodes categoricals (codes as integers), passes numerics through",
            "Median imputation for all NaN values (not NaN-safe)",
            "Category codes converted to float32 (ordinal, not one-hot)",
            "SAE is trained on the output of layer 12 (final_layer_norm)",
        ],
        "implication": (
            "The SAE sees ordinal-encoded categoricals and median-imputed numerics. "
            "A feature that fires on 'high values in column X' in the raw data may "
            "actually be responding to the ordinal code or the imputed median pattern. "
            "Missing values are replaced by the training-set median before the model sees them."
        ),
    },
    "tabpfn": {
        "model": "TabPFN",
        "architecture": "Transformer (12M params), pure in-context learning (no finetuning)",
        "preprocessing": [
            "AutoGluon AutoMLPipelineFeatureGenerator (same as Mitra)",
            "NaN-safe: missing values preserved as NaN (TabPFN handles them natively)",
            "Category codes preserved as integers with NaN for missing",
            "SAE is trained on the output of layer 16 (of 24 transformer layers)",
        ],
        "implication": (
            "The SAE sees the same ordinal encoding as Mitra but with NaN preserved. "
            "Features responding to missingness patterns are possible."
        ),
    },
    "tabicl": {
        "model": "TabICL",
        "architecture": "Column-then-row transformer, in-context learning",
        "preprocessing": [
            "AutoGluon AutoMLPipelineFeatureGenerator",
            "Median imputation for NaN (not NaN-safe)",
            "SAE is trained on the output of layer 10 (of 12 blocks)",
        ],
        "implication": (
            "Same preprocessing as Mitra. Column-level attention means the model "
            "can learn per-feature patterns before combining across features."
        ),
    },
    "tabicl_v2": {
        "model": "TabICL-v2",
        "architecture": "Column-then-row transformer v2, in-context learning",
        "preprocessing": [
            "AutoGluon AutoMLPipelineFeatureGenerator",
            "Median imputation for NaN",
            "SAE is trained on the output of layer 12 (of 12 blocks)",
        ],
        "implication": "Same preprocessing as TabICL/Mitra.",
    },
    "tabdpt": {
        "model": "TabDPT",
        "architecture": "Transformer with retrieval augmentation",
        "preprocessing": [
            "AutoGluon AutoMLPipelineFeatureGenerator",
            "NaN-safe: missing values preserved",
            "SAE is trained on the output of layer 14 (of 18 encoder layers)",
        ],
        "implication": (
            "NaN-safe like TabPFN. Retrieval augmentation means the model's "
            "representation incorporates nearest-neighbor context from training data."
        ),
    },
    "carte": {
        "model": "CARTE",
        "architecture": "Graph transformer (star graph GNN), trains per-dataset",
        "preprocessing": [
            "CARTE handles its own preprocessing internally",
            "Builds a star graph per row: central node + one node per feature",
            "Uses pretrained fastText embeddings for column names",
            "RobustScaler applied to numeric features",
            "SAE is trained on the output of layer 1 (of GNN layers)",
        ],
        "implication": (
            "CARTE's representation is fundamentally different: it embeds column "
            "names via language model, so the same numeric value in different columns "
            "gets different representations. The SAE sees graph-level embeddings, "
            "not tabular features directly."
        ),
    },
    "hyperfast": {
        "model": "HyperFast",
        "architecture": "Hypernetwork that generates MLP weights",
        "preprocessing": [
            "Ordinal encoding for categoricals",
            "Mean imputation for missing numerics, mode for missing categoricals",
            "StandardScaler: zero mean, unit variance per feature",
            "SAE is trained on the output of layer 2",
        ],
        "implication": (
            "Standardized features mean the SAE sees z-scores, not raw values. "
            "A feature firing on 'high values' means high z-score (>2σ from mean)."
        ),
    },
}


def load_row_stats():
    """Load per-row PMI, surprise, compression caches."""
    caches = {}
    for name, dirp, key in [
        ("pmi", PMI_DIR, "row_pmi"),
        ("surprise", SURPRISE_DIR, "row_surprise"),
        ("compression", COMPRESSION_DIR, "row_compress_delta"),
    ]:
        summary_path = dirp / f"{name}_summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        for ds in summary.get("datasets", summary.keys()):
            ds_name = ds if isinstance(ds, str) else ds
            npz_path = dirp / f"{ds_name}.npz"
            if npz_path.exists():
                vals = np.load(npz_path, allow_pickle=True)
                if key in vals:
                    caches.setdefault(ds_name, {})[name] = vals[key]
    return caches


_SAE_TEST_ROW_INDICES_CACHE: dict = {}


def _sae_test_row_indices(model: str, dataset: str) -> Optional[np.ndarray]:
    """Absolute row indices (into full X) that the SAE test split used.

    Reads from `output/sae_training_round10/{model}_taskaware_sae_test.npz`
    once per model and caches the per-dataset arrays.
    """
    if model not in _SAE_TEST_ROW_INDICES_CACHE:
        candidates = sorted(SAE_DATA_DIR.glob(f"{model}_taskaware_sae_test.npz"))
        if not candidates:
            candidates = sorted(SAE_DATA_DIR.glob(f"{model}_*_sae_test.npz"))
        if not candidates:
            return None
        d = np.load(candidates[0], allow_pickle=True)
        spd = d["samples_per_dataset"]
        row_indices = d["row_indices"]
        per_ds = {}
        offset = 0
        for ds_name, count in spd:
            count = int(count)
            per_ds[str(ds_name)] = row_indices[offset:offset + count]
            offset += count
        _SAE_TEST_ROW_INDICES_CACHE[model] = per_ds
    return _SAE_TEST_ROW_INDICES_CACHE[model].get(dataset)


def load_baseline_predictions(model: str, dataset: str) -> Optional[dict]:
    """Load cached baseline predictions for one (model, dataset).

    Returns None if the cache file is missing. The cache is positionally
    aligned with `load_test_embeddings(model)[dataset]` — row `ri` in the
    SAE test embedding corresponds to row `ri` in the prediction arrays.
    """
    path = BASELINE_PRED_DIR / model / f"{dataset}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return {
        "pred_probs": d["pred_probs"],
        "pred_class": d["pred_class"],
        "y_true": d["y_true"],
        "task_type": str(d["task_type"]),
    }


def _stratified_activating(active_idx, active_acts, per_band=2, rng=None):
    """Sample `per_band` rows from each of 3 activation strata: top, p90, p80.

    Returns (indices, bands) where indices are into the embedding matrix and
    bands is a parallel list of stratum names ('top'|'p90'|'p80'). Falls back
    to top-(3*per_band) labelled 'top' if too few activating rows to stratify.
    """
    if rng is None:
        rng = np.random.default_rng(13)
    n_target = 3 * per_band
    if len(active_idx) <= n_target:
        order = np.argsort(-active_acts)
        idx = active_idx[order]
        return idx, ["top"] * len(idx)

    order = np.argsort(-active_acts)  # descending by activation
    sorted_idx = active_idx[order]
    sorted_acts = active_acts[order]

    top = sorted_idx[:per_band]

    p90_val = np.percentile(active_acts, 90)
    p80_val = np.percentile(active_acts, 80)

    used = set(int(i) for i in top)
    p90 = _pick_near(sorted_idx, sorted_acts, p90_val, per_band, used, rng)
    used.update(int(i) for i in p90)
    p80 = _pick_near(sorted_idx, sorted_acts, p80_val, per_band, used, rng)

    indices = np.concatenate([top, p90, p80])
    bands = ["top"] * per_band + ["p90"] * per_band + ["p80"] * per_band
    return indices, bands


def _pick_near(sorted_idx, sorted_acts, target_val, k, used, rng):
    """Pick `k` distinct indices whose activation is closest to `target_val`."""
    dist = np.abs(sorted_acts - target_val)
    order = np.argsort(dist)
    picks = []
    for j in order:
        ix = int(sorted_idx[j])
        if ix in used:
            continue
        picks.append(ix)
        if len(picks) == k:
            break
    return np.array(picks, dtype=sorted_idx.dtype)


def _classify_column(series: pd.Series) -> str:
    """Classify column as 'numeric', 'binary', or 'categorical'.

    String-typed columns (including pandas `string[pyarrow]` / `ArrowStringArray`
    used by newer TabArena cache builds) must bucket as categorical; they are
    neither `object` dtype nor `CategoricalDtype`, so the older check
    `is_categorical_dtype(series) or series.dtype == object` silently missed
    them and they were then annotated with `(pXX)` percentiles from an
    alphabetical sort — garbage.
    """
    if (isinstance(series.dtype, pd.CategoricalDtype)
            or series.dtype == object
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_bool_dtype(series)):
        return "categorical"
    try:
        n_unique = int(series.nunique(dropna=True))
    except TypeError:
        return "categorical"
    if n_unique <= 2:
        return "binary"
    if n_unique <= 20 and pd.api.types.is_integer_dtype(series):
        return "categorical"  # small-cardinality integers: likely ordinal-encoded
    return "numeric"


def _compute_marginals(X_train: pd.DataFrame) -> dict:
    """Per-column marginal stats from the SAE training split."""
    marginals = {}
    for col in X_train.columns:
        col_type = _classify_column(X_train[col])
        if col_type == "numeric":
            vals = X_train[col].dropna().to_numpy()
            marginals[col] = {
                "type": "numeric",
                "sorted": np.sort(vals),
                "n": int(len(vals)),
            }
        else:
            freq = X_train[col].value_counts(dropna=False, normalize=True).to_dict()
            marginals[col] = {"type": col_type, "freq": freq}
    return marginals


def _annotate_value(col: str, val, marginals: dict) -> str:
    """Attach marginal-distribution position to a raw cell value."""
    m = marginals.get(col)
    if pd.isna(val):
        return "NaN"
    if m is None:
        return str(val)
    if m["type"] == "numeric":
        sorted_vals = m["sorted"]
        if len(sorted_vals) == 0:
            return _fmt_num(val)
        pct = int(round(100 * np.searchsorted(sorted_vals, val, side="right") / len(sorted_vals)))
        return f"{_fmt_num(val)} (p{pct})"
    freq = m["freq"].get(val, 0.0)
    return f"{val} (freq {freq:.2f})"


def _fmt_num(val) -> str:
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    try:
        return f"{float(val):.4g}"
    except (ValueError, TypeError):
        return str(val)


def _target_summary(y_train, task_type: str) -> dict:
    """Target distribution from the SAE training split."""
    y = np.asarray(y_train).ravel()
    if task_type == "classification":
        unique, counts = np.unique(y, return_counts=True)
        return {
            "task": "classification",
            "n": int(y.size),
            "class_freq": {str(u): round(float(c / y.size), 4) for u, c in zip(unique, counts)},
        }
    return {
        "task": "regression",
        "n": int(y.size),
        "mean": round(float(np.mean(y)), 4),
        "std": round(float(np.std(y)), 4),
        "p25": round(float(np.percentile(y, 25)), 4),
        "p50": round(float(np.percentile(y, 50)), 4),
        "p75": round(float(np.percentile(y, 75)), 4),
    }


def build_contrastive(model, feat_idx, dataset, sae, test_embs, splits,
                      row_stats, top_k=5, device="cpu", per_band=2):
    """Build contrastive examples for one (model, feature, dataset).

    Activating rows: stratified sample — `per_band` each from top, p90, p80
    activation bands (6 rows by default). Falls back to top-K if fewer
    activating rows exist.

    Contrast rows: nearest non-activating rows in embedding space; size
    matched to activating set.

    Returns a list of dicts, each with:
        label: "activating" or "contrast"
        activation: float
        row_idx: int
        + all raw data columns
        + pmi, surprise, compression (if available)
    """
    from data.extended_loader import load_tabarena_dataset

    if dataset not in test_embs:
        return []

    emb = test_embs[dataset]
    n_emb = len(emb)
    with torch.no_grad():
        emb_t = torch.tensor(emb, dtype=torch.float32, device=device)
        acts = sae.encode(emb_t)
    feat_acts = acts[:, feat_idx].cpu().numpy()

    # Load raw data. The SAE test NPZ stores absolute row indices under
    # `row_indices` — use those directly so X_test, y_test, and per-row stats
    # align with the embeddings the SAE actually saw (test_indices[:n_emb]
    # is WRONG: the SAE training split shuffles test_indices before picking).
    ds_splits = splits.get(dataset)
    if not ds_splits:
        return []
    result = load_tabarena_dataset(dataset, max_samples=999999)
    if result is None:
        return []
    X, y = result[0], result[1]
    sae_row_indices = _sae_test_row_indices(model, dataset)
    if sae_row_indices is None or len(sae_row_indices) != n_emb:
        return []
    train_idx = ds_splits.get("train_indices", ds_splits.get("train"))
    try:
        X_test = X.iloc[sae_row_indices].reset_index(drop=True)
        y_test = np.asarray(y)[sae_row_indices]
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = np.asarray(y)[train_idx]
    except (IndexError, KeyError):
        return []

    # Marginal stats + target distribution from SAE training split
    marginals = _compute_marginals(X_train)
    task_type = ds_splits.get("task_type", "classification")
    target_summary = _target_summary(y_train, task_type)

    # Stratified activating rows: 2 top + 2 near-p90 + 2 near-p80 (6 total)
    active_mask = feat_acts > 0
    if active_mask.sum() < 1:
        return []
    active_idx = np.where(active_mask)[0]
    active_acts = feat_acts[active_idx]
    top_idx, top_bands = _stratified_activating(active_idx, active_acts, per_band=per_band)

    # Contrast: nearest non-activating rows in embedding space
    inactive_idx = np.where(~active_mask)[0]
    if len(inactive_idx) == 0:
        return []

    # For each activating row, find nearest inactive neighbor; union, matched to activating count
    top_emb = emb[top_idx]
    inactive_emb = emb[inactive_idx]
    dists = cdist(top_emb, inactive_emb, metric="cosine")
    contrast_set = set()
    n_contrast = len(top_idx)
    for i in range(len(top_idx)):
        nearest = inactive_idx[np.argsort(dists[i])[:n_contrast]]
        contrast_set.update(nearest.tolist())
    contrast_idx = sorted(contrast_set)[:n_contrast]

    # Build rows (raw values annotated with marginal position from training split)
    ds_stats = row_stats.get(dataset, {})
    rows = []
    groups = [
        (top_idx, ["activating"] * len(top_idx), top_bands),
        (contrast_idx, ["contrast"] * len(contrast_idx), ["contrast"] * len(contrast_idx)),
    ]
    for idx, labels, bands in groups:
        for ri, lab, band in zip(idx, labels, bands):
            row = {
                "label": lab,
                "band": band,
                "dataset": dataset,
                "row_idx": int(ri),
                "activation": float(feat_acts[ri]),
                "target": float(y_test[ri]) if ri < len(y_test) else None,
            }
            # Raw data values, annotated with marginal (training-split) position
            if ri < len(X_test):
                for col in X_test.columns:
                    row[col] = _annotate_value(col, X_test.iloc[ri][col], marginals)
            # Per-row stats
            for stat_name in ["pmi", "surprise", "compression"]:
                vals = ds_stats.get(stat_name)
                if vals is not None and ri < len(vals):
                    row[stat_name] = float(vals[ri])
            rows.append(row)

    return {"rows": rows, "target_summary": target_summary}


def build_validator_examples(
    model: str,
    feat_idx: int,
    n_act: int = 5,
    n_con: int = 5,
    device: str = "cpu",
) -> dict:
    """Build held-out validator CSVs + truth file for one (model, feat).

    Samples n_act activating + n_con non-activating rows per dataset from the
    SAE test set, *excluding* row_idx values already used in the contrastive
    CSVs. Writes per-dataset CSVs (stripped of label/band/activation/row_idx
    giveaways) and a truth JSON with {row_id -> fires: bool} for grading.
    """
    from scripts.intervention.intervene_lib import load_sae as _load_sae
    from data.extended_loader import load_tabarena_dataset

    out_dir = OUTPUT_DIR / model
    out_dir.mkdir(parents=True, exist_ok=True)

    sae, _ = _load_sae(model, device=device)
    test_embs = load_test_embeddings(model)
    splits = json.loads(SPLITS_PATH.read_text())

    truth: dict = {}
    written: list = []

    # Discover which datasets this feature has contrastive CSVs for.
    model_dir = OUTPUT_DIR / model
    csv_glob = sorted(model_dir.glob(f"f{feat_idx}_*.csv"))
    ds_to_csv = {}
    for p in csv_glob:
        stem = p.stem  # f{feat}_{dataset}
        if stem.startswith(f"f{feat_idx}_validator_"):
            continue  # skip any prior validator CSV
        ds = stem[len(f"f{feat_idx}_"):]
        if ds in test_embs:
            ds_to_csv[ds] = p

    if not ds_to_csv:
        raise FileNotFoundError(
            f"No contrastive CSVs for f_{feat_idx} in {model_dir}. "
            "Run the contrastive builder first."
        )

    for ds, contrastive_csv in ds_to_csv.items():
        emb = test_embs[ds]
        used_row_idx = set(int(r) for r in pd.read_csv(contrastive_csv).row_idx.tolist())

        with torch.no_grad():
            emb_t = torch.tensor(emb, dtype=torch.float32, device=device)
            acts = sae.encode(emb_t)
        feat_acts = acts[:, feat_idx].cpu().numpy()

        active_pos = np.where(feat_acts > 0)[0]
        inactive_pos = np.where(feat_acts == 0)[0]
        held_act = np.array([i for i in active_pos if int(i) not in used_row_idx])
        held_con = np.array([i for i in inactive_pos if int(i) not in used_row_idx])

        rng = np.random.default_rng(seed=(42 + feat_idx) * 1000 + hash(ds) % (2**31))
        n_act_take = min(n_act, len(held_act))
        n_con_take = min(n_con, len(held_con))
        picked_act = rng.choice(held_act, size=n_act_take, replace=False) if n_act_take else np.array([], dtype=int)
        picked_con = rng.choice(held_con, size=n_con_take, replace=False) if n_con_take else np.array([], dtype=int)

        ds_splits = splits.get(ds)
        if not ds_splits:
            continue
        train_idx = ds_splits.get("train_indices", ds_splits.get("train"))
        loaded = load_tabarena_dataset(ds, max_samples=999999)
        if loaded is None:
            continue
        X, y = loaded[0], loaded[1]
        sae_row_indices = _sae_test_row_indices(model, ds)
        if sae_row_indices is None or len(sae_row_indices) != len(emb):
            continue
        X_test = X.iloc[sae_row_indices].reset_index(drop=True)
        y_test = np.asarray(y)[sae_row_indices]
        X_train = X.iloc[train_idx].reset_index(drop=True)
        marginals = _compute_marginals(X_train)

        rows = []
        ds_truth: dict = {}
        for positions, is_active in [(picked_act, True), (picked_con, False)]:
            for ri in positions:
                ri = int(ri)
                row_id = f"r{len(rows):03d}"
                row = {"row_id": row_id}
                row["target"] = float(y_test[ri])
                for col in X_test.columns:
                    row[col] = _annotate_value(col, X_test.iloc[ri][col], marginals)
                rows.append(row)
                ds_truth[row_id] = bool(is_active)

        # Shuffle so the validator can't infer class from row order
        shuffle_rng = np.random.default_rng(seed=777 + feat_idx)
        order = list(range(len(rows)))
        shuffle_rng.shuffle(order)
        rows = [rows[i] for i in order]
        # Re-assign opaque row_ids after shuffle so they run 0..N-1 in display order
        new_truth = {}
        for new_i, r in enumerate(rows):
            old_id = r["row_id"]
            new_id = f"r{new_i:03d}"
            r["row_id"] = new_id
            new_truth[new_id] = ds_truth[old_id]
        ds_truth = new_truth

        out_path = out_dir / f"f{feat_idx}_validator_{ds}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        truth[ds] = ds_truth
        written.append(out_path)
        print(f"  {ds}: {len(picked_act)} act + {len(picked_con)} con -> {out_path.name}")

    truth_path = out_dir / f"f{feat_idx}_validator_truth.json"
    truth_path.write_text(json.dumps(truth, indent=2))
    print(f"Wrote {len(written)} validator CSVs + {truth_path.name}")
    return truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--features", nargs="+", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-datasets", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dataset-selection",
        choices=["auto", "n_active", "quality"],
        default="auto",
        help="How to choose datasets per feature. 'auto' prefers the quality cache when available.",
    )
    parser.add_argument(
        "--quality-cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Path to the global dataset-quality cache.",
    )
    parser.add_argument(
        "--require-quality-cache",
        action="store_true",
        help="Fail if quality-based dataset selection is requested but the cache is missing.",
    )
    parser.add_argument("--validator", action="store_true",
                        help="Build validator (held-out) CSVs instead of contrastive examples")
    parser.add_argument("--n-act", type=int, default=5,
                        help="Number of held-out activating rows per dataset (--validator mode)")
    parser.add_argument("--n-con", type=int, default=5,
                        help="Number of held-out non-activating rows per dataset (--validator mode)")
    args = parser.parse_args()

    if args.validator:
        if not args.features:
            parser.error("--validator requires --features")
        for feat in args.features:
            build_validator_examples(args.model, feat, args.n_act, args.n_con, args.device)
        return

    splits = json.loads(SPLITS_PATH.read_text())
    sae, _ = load_sae(args.model, device=args.device)
    test_embs = load_test_embeddings(args.model)
    row_stats = load_row_stats()

    # Load dataset-level metafeatures
    dataset_meta = {}
    if PYMFE_PATH.exists():
        with open(PYMFE_PATH) as f:
            dataset_meta = json.load(f)

    print(f"Loaded SAE, {len(test_embs)} datasets, row stats for {len(row_stats)} datasets")

    if args.all:
        # All alive features
        with torch.no_grad():
            sample_emb = next(iter(test_embs.values()))
            sample_acts = sae.encode(torch.tensor(sample_emb[:1], dtype=torch.float32,
                                                   device=args.device))
            n_features = sample_acts.shape[1]
        features = list(range(n_features))
    elif args.features:
        features = args.features
    else:
        parser.error("Specify --features or --all")

    out_dir = OUTPUT_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save preprocessing context
    ctx = PREPROCESSING_CONTEXT.get(args.model, {})
    if ctx:
        ctx_path = out_dir / "preprocessing_context.json"
        with open(ctx_path, "w") as f:
            json.dump(ctx, f, indent=2)
        print(f"Saved preprocessing context to {ctx_path}")

    # Rank datasets by activation count for each feature
    datasets = sorted(test_embs.keys())
    quality_cache = load_quality_cache(args.quality_cache_path)
    if args.dataset_selection == "quality" and not quality_cache:
        if args.require_quality_cache or args.quality_cache_path != DEFAULT_CACHE_PATH:
            parser.error(f"Quality cache not found at {args.quality_cache_path}")
    if args.dataset_selection == "auto" and args.require_quality_cache and not quality_cache:
        parser.error(f"Required quality cache not found at {args.quality_cache_path}")

    for feat in features:
        # Find datasets where this feature fires most
        ds_counts = []
        ds_quality_entries = None
        selected_via = "n_active"
        for ds in datasets:
            emb = test_embs[ds]
            with torch.no_grad():
                acts = sae.encode(torch.tensor(emb, dtype=torch.float32,
                                               device=args.device))
            n_active = int((acts[:, feat] > 0).sum())
            if n_active > 0:
                ds_counts.append((ds, n_active))
        ds_counts.sort(key=lambda x: -x[1])

        if not ds_counts:
            print(f"f_{feat}: no activations across any dataset")
            continue

        if quality_cache and args.dataset_selection in {"auto", "quality"}:
            feature_block = cache_entry_for_feature(quality_cache, args.model, feat)
            if feature_block:
                ds_quality_entries = feature_block.get("datasets", {})
                selected_ds = select_top_datasets(
                    ds_quality_entries,
                    args.max_datasets,
                    diversity_tie_margin=quality_cache.get("metadata", {})
                    .get("score_config", {})
                    .get("diversity_tie_margin", DEFAULT_SCORE_CONFIG["diversity_tie_margin"]),
                )
                if selected_ds:
                    selected_via = "quality_cache"
                elif args.dataset_selection == "quality" and args.require_quality_cache:
                    parser.error(
                        f"Quality cache has no selectable datasets for model={args.model} f_{feat}"
                    )
            elif args.dataset_selection == "quality" and args.require_quality_cache:
                parser.error(
                    f"Quality cache missing model={args.model} f_{feat} at {args.quality_cache_path}"
                )
            else:
                selected_ds = []
        else:
            selected_ds = []

        if not selected_ds:
            selected_ds = [ds for ds, _ in ds_counts[:args.max_datasets]]
            selected_via = "n_active"

        print(
            f"f_{feat}: {len(ds_counts)} datasets fire, using top {len(selected_ds)} via {selected_via}"
        )

        ds_contexts = {}
        for ds in selected_ds:
            result = build_contrastive(args.model, feat, ds, sae, test_embs,
                                       splits, row_stats, args.top_k, args.device)
            if not result or not result.get("rows"):
                continue
            rows = result["rows"]
            target_summary = result.get("target_summary", {})

            import pandas as pd
            df = pd.DataFrame(rows)
            out_path = out_dir / f"f{feat}_{ds}.csv"
            df.to_csv(out_path, index=False)

            # Collect dataset metafeatures
            meta = dataset_meta.get(ds, {})
            ds_info = splits.get(ds, {})
            ds_contexts[ds] = {
                "task_type": ds_info.get("task_type", "unknown"),
                "n_train": len(ds_info.get("train_indices", [])),
                "n_test": len(ds_info.get("test_indices", [])),
                "n_activating": int((pd.Series([r["label"] for r in rows]) == "activating").sum()),
                "csv_file": f"f{feat}_{ds}.csv",
                "target_summary": target_summary,
                "dataset_selection_method": selected_via,
            }
            if ds_quality_entries and ds in ds_quality_entries:
                ds_contexts[ds]["dataset_quality"] = ds_quality_entries[ds]
            for key in ["nr_inst", "nr_attr", "nr_class", "nr_num", "nr_cat",
                         "inst_to_attr", "cat_to_num", "nr_bin"]:
                if key in meta:
                    ds_contexts[ds][key] = meta[key]

        # Save per-feature context
        feat_context = {
            "model": args.model,
            "feature_idx": feat,
            "n_datasets_firing": len(ds_counts),
            "datasets_used": list(ds_contexts.keys()),
            "dataset_selection_method": selected_via,
            "preprocessing": ctx,
            "dataset_stats": ds_contexts,
        }
        ctx_path = out_dir / f"f{feat}_context.json"
        with open(ctx_path, "w") as f:
            json.dump(feat_context, f, indent=2)

        print(f"  -> {len(ds_contexts)} CSVs + context saved")


if __name__ == "__main__":
    main()
