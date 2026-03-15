#!/usr/bin/env python3
"""Row-level probes for SAE auto-interpretability.

For each SAE feature, collects the rows that maximally activate it vs rows
that don't, computes what distinguishes them, and saves actual tabular data
samples for LLM auto-interpretability.

Usage:
    python scripts/row_level_probes.py --model tabdpt
    python scripts/row_level_probes.py --model tabdpt --top-k 30 --fast
    python scripts/row_level_probes.py --model tabdpt --verify
    python scripts/row_level_probes.py --model tabdpt --features 120,62,56
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from scripts._project_root import PROJECT_ROOT

from data.extended_loader import TABARENA_DATASETS, load_tabarena_dataset
from scripts.sae.analyze_sae_concepts_deep import (
    compute_column_stats,
    compute_row_meta_features,
)
from scripts.sae.compare_sae_architectures import META_NAMES, meta_features_to_array
from scripts.concepts.concept_fingerprint import load_per_dataset_embeddings
from scripts.intervention.concept_importance import (
    get_alive_features,
    get_feature_labels,
    get_matryoshka_bands,
    feature_to_band,
)
from scripts.embeddings.extract_layer_embeddings import SPLIT_SEED, get_dataset_task
from scripts.intervention.intervene_sae import (
    DEFAULT_SAE_DIR,
    DEFAULT_TRAINING_DIR,
    get_extraction_layer,
    load_sae,
)

# Constants matching build_sae_training_data.py
TRAIN_FRACTION = 0.7
OUTPUT_DIR = PROJECT_ROOT / "output" / "row_level_probes"


# ── Row alignment ────────────────────────────────────────────────────────────


def reconstruct_query_data(
    ds_name: str,
    context_size: int = 600,
    query_size: int = 500,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Replay extraction pipeline to get exact X_query, y_query rows.

    Mirrors load_context_query() logic identically, but also returns y_query.
    """
    result = load_tabarena_dataset(ds_name, max_samples=context_size + query_size)
    if result is None:
        raise ValueError(f"Failed to load dataset: {ds_name}")

    X, y, _ = result
    n = len(X)
    if n < context_size + query_size:
        context_size = int(n * 0.7)
        query_size = n - context_size

    task = get_dataset_task(ds_name)
    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)

        query_frac = query_size / (context_size + query_size)
        try:
            _, X_q, _, y_q = train_test_split(
                X, y, test_size=query_frac,
                random_state=SPLIT_SEED, stratify=y,
            )
        except ValueError:
            _, X_q, _, y_q = train_test_split(
                X, y, test_size=query_frac,
                random_state=SPLIT_SEED,
            )
        return X_q[:query_size], y_q[:query_size], task
    else:
        X_q = X[context_size:context_size + query_size]
        y_q = y[context_size:context_size + query_size]
        return X_q, y_q, task


def get_train_indices(n_query: int, max_per_dataset: int = 500) -> np.ndarray:
    """Replay split_rows() to get indices of training rows.

    Returns array of indices into X_query that became SAE training data.
    """
    rng = np.random.RandomState(SPLIT_SEED)

    if n_query > max_per_dataset:
        idx = rng.choice(n_query, max_per_dataset, replace=False)
    else:
        idx = rng.permutation(n_query)

    n_train = int(len(idx) * TRAIN_FRACTION)
    return idx[:n_train]


def get_column_names(ds_name: str) -> Optional[List[str]]:
    """Try to get original column names for a TabArena dataset via OpenML."""
    try:
        import openml

        ds_info = TABARENA_DATASETS.get(ds_name, {})
        openml_id = ds_info.get("openml_id")
        if not openml_id:
            return None

        dataset = openml.datasets.get_dataset(
            openml_id, download_data=False, download_qualities=False,
        )
        # Get feature names excluding the target
        target = dataset.default_target_attribute
        names = [
            f.name for f in dataset.features.values()
            if f.name != target
        ]
        return names if names else None
    except Exception:
        return None


# ── SAE encoding ─────────────────────────────────────────────────────────────


def encode_all_rows(
    model_key: str,
    device: str = "cpu",
) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """Load SAE training embeddings, encode through SAE.

    Returns:
        activations: (n_total, hidden_dim) SAE activations
        samples_per_dataset: list of (name, count) tuples
    """
    per_ds = load_per_dataset_embeddings(model_key)

    ds_order = list(per_ds.keys())
    all_emb = np.concatenate([per_ds[ds] for ds in ds_order], axis=0)
    samples_per_dataset = [(ds, len(per_ds[ds])) for ds in ds_order]

    sae, config = load_sae(model_key, device=device)
    sae.eval()

    with torch.no_grad():
        x = torch.tensor(all_emb, dtype=torch.float32, device=device)
        h = sae.encode(x)
        activations = h.cpu().numpy()

    return activations, samples_per_dataset


# ── Meta-features ────────────────────────────────────────────────────────────


def compute_row_properties(
    X_rows: np.ndarray,
    y_rows: np.ndarray,
    fast: bool = True,
) -> np.ndarray:
    """Compute 52 row-level meta-features.

    Args:
        X_rows: (n_rows, n_features) array
        y_rows: (n_rows,) target array
        fast: If True, skip supervised metrics (SVM boundary distance etc.)

    Returns:
        (n_rows, 52) array matching META_NAMES order.
    """
    col_names = [f"col_{i}" for i in range(X_rows.shape[1])]
    df = pd.DataFrame(X_rows, columns=col_names)

    numeric_cols, categorical_cols, col_stats, dataset_stats = compute_column_stats(df)

    y_arg = None if fast else y_rows
    mf_list = compute_row_meta_features(
        df, y_arg, numeric_cols, categorical_cols, col_stats, dataset_stats,
    )

    return np.array([meta_features_to_array(mf) for mf in mf_list])


# ── Build matched dataset ────────────────────────────────────────────────────


def build_row_dataset(
    model_key: str,
    device: str = "cpu",
    fast: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Build matched dataset: SAE activations + original data + meta-features.

    Returns:
        activations: (n_total, hidden_dim) SAE activations
        meta_features: (n_total, 52) row-level meta-features
        row_metadata: list of dicts with dataset, y, columns, col_names
    """
    print("Encoding all training rows through SAE...")
    activations, samples_per_dataset = encode_all_rows(model_key, device=device)
    n_total = activations.shape[0]
    print(f"  Total rows: {n_total}, Hidden dim: {activations.shape[1]}")

    all_meta = []
    all_metadata = []
    offset = 0

    # Cache column names across datasets
    col_name_cache: Dict[str, List[str]] = {}

    print(f"\nReconstructing original data for {len(samples_per_dataset)} datasets...")
    for ds_name, count in samples_per_dataset:
        print(f"  {ds_name}: {count} rows...", end=" ", flush=True)
        t0 = time.time()

        try:
            X_query, y_query, task = reconstruct_query_data(ds_name)
        except Exception as e:
            print(f"SKIP ({e})")
            all_meta.append(np.zeros((count, len(META_NAMES))))
            for i in range(count):
                all_metadata.append({
                    "dataset": ds_name, "y": None,
                    "columns": {}, "col_names": [],
                })
            offset += count
            continue

        train_idx = get_train_indices(len(X_query))

        if len(train_idx) != count:
            print(f"WARN: expected {count} rows, got {len(train_idx)}. ", end="")
            # Align to the actual count in the SAE training data
            if len(train_idx) > count:
                train_idx = train_idx[:count]
            else:
                # Pad with repeated last index (shouldn't happen in practice)
                pad = np.full(count - len(train_idx), train_idx[-1])
                train_idx = np.concatenate([train_idx, pad])

        X_train = X_query[train_idx]
        y_train = y_query[train_idx]

        # Compute meta-features for this dataset's rows
        mf = compute_row_properties(X_train, y_train, fast=fast)
        all_meta.append(mf)

        # Get real column names (cached)
        if ds_name not in col_name_cache:
            real_names = get_column_names(ds_name)
            if real_names and len(real_names) == X_train.shape[1]:
                col_name_cache[ds_name] = real_names
            else:
                col_name_cache[ds_name] = [f"col_{i}" for i in range(X_train.shape[1])]
        col_names = col_name_cache[ds_name]

        # Store metadata per row
        for i in range(count):
            row_dict = {
                "dataset": ds_name,
                "y": float(y_train[i]),
                "columns": {
                    col_names[j]: float(X_train[i, j])
                    for j in range(X_train.shape[1])
                },
                "col_names": col_names,
            }
            all_metadata.append(row_dict)

        offset += count
        print(f"{time.time() - t0:.1f}s")

    meta_features = np.concatenate(all_meta, axis=0)
    return activations, meta_features, all_metadata


# ── Per-feature sample extraction ────────────────────────────────────────────


def compute_cohens_d(
    high: np.ndarray,
    low: np.ndarray,
) -> List[Tuple[str, float, str]]:
    """Compute Cohen's d for each meta-feature between high and low groups."""
    results = []
    for j, name in enumerate(META_NAMES):
        h = high[:, j]
        l = low[:, j]
        h_mean, l_mean = h.mean(), l.mean()
        pooled_std = np.sqrt((h.var() + l.var()) / 2)
        if pooled_std < 1e-10:
            d = 0.0
        else:
            d = (h_mean - l_mean) / pooled_std
        direction = "high > low" if d > 0 else ("high < low" if d < 0 else "equal")
        results.append((name, d, direction))
    return results


def extract_feature_samples(
    feat_idx: int,
    activations: np.ndarray,
    row_metadata: List[Dict],
    meta_features: np.ndarray,
    top_k: int = 20,
    bands: Optional[Dict[str, int]] = None,
    feature_labels: Optional[Dict[int, str]] = None,
) -> Dict:
    """For one feature, collect high/low activation samples.

    High: top_k rows with highest activation.
    Low: top_k rows with zero/near-zero activation from SAME datasets
    (controlled contrast).
    """
    feat_acts = activations[:, feat_idx]

    # High activation samples: top_k by activation value
    high_indices = np.argsort(feat_acts)[-top_k:][::-1]

    # Get datasets present in high samples
    high_datasets = set(row_metadata[i]["dataset"] for i in high_indices)

    # Low activation samples: zero/near-zero from SAME datasets
    low_candidates = [
        i for i, act in enumerate(feat_acts)
        if act < 0.001 and row_metadata[i]["dataset"] in high_datasets
    ]

    if len(low_candidates) >= top_k:
        rng = np.random.RandomState(feat_idx)
        low_indices = rng.choice(low_candidates, top_k, replace=False)
    else:
        low_indices = np.array(low_candidates, dtype=int)

    # Cohen's d effect sizes
    if len(high_indices) > 1 and len(low_indices) > 1:
        discriminators = compute_cohens_d(
            meta_features[high_indices], meta_features[low_indices],
        )
    else:
        discriminators = [(name, 0.0, "n/a") for name in META_NAMES]

    # Sort discriminators by absolute effect size
    sorted_disc = sorted(discriminators, key=lambda x: abs(x[1]), reverse=True)

    # Top datasets by max activation
    ds_acts: Dict[str, float] = {}
    for i in high_indices:
        ds = row_metadata[i]["dataset"]
        ds_acts[ds] = max(ds_acts.get(ds, 0.0), feat_acts[i])
    top_datasets = sorted(ds_acts, key=ds_acts.get, reverse=True)[:5]

    # Activation stats
    active_mask = feat_acts > 0
    act_stats = {
        "mean": float(feat_acts[active_mask].mean()) if active_mask.any() else 0.0,
        "max": float(feat_acts.max()),
        "frac_active": float(active_mask.mean()),
    }

    # Band and label info
    band = feature_to_band(feat_idx, bands) if bands else "unknown"
    label = feature_labels.get(feat_idx, "unlabeled") if feature_labels else "unlabeled"

    # Build sample dicts (limit columns for readability)
    def _build_sample(i: int) -> Dict:
        m = row_metadata[i]
        return {
            "dataset": m["dataset"],
            "y": m["y"],
            "activation": float(feat_acts[i]),
            "columns": m["columns"],
            "properties": {
                META_NAMES[j]: float(meta_features[i, j])
                for j in range(len(META_NAMES))
            },
        }

    high_samples = [_build_sample(i) for i in high_indices]
    low_samples = [_build_sample(int(i)) for i in low_indices]

    return {
        "feature_idx": feat_idx,
        "band": band,
        "current_label": label,
        "activation_stats": act_stats,
        "top_datasets": top_datasets,
        "n_high_samples": len(high_samples),
        "n_low_samples": len(low_samples),
        "discriminative_properties": [
            {"name": name, "cohens_d": round(float(d), 3), "direction": direction}
            for name, d, direction in sorted_disc
        ],
        "high_activation_samples": high_samples,
        "low_activation_samples": low_samples,
    }


# ── Verification ─────────────────────────────────────────────────────────────


def verify_row_alignment(model_key: str, n_datasets: int = 3) -> bool:
    """Verify row alignment by checking index counts match SAE training data.

    For each dataset, confirms that get_train_indices produces the same count
    as stored in the SAE training .npz samples_per_dataset.
    """
    per_ds = load_per_dataset_embeddings(model_key)
    ds_names = list(per_ds.keys())[:n_datasets]

    all_ok = True
    for ds_name in ds_names:
        stored_count = len(per_ds[ds_name])
        emb_dim = per_ds[ds_name].shape[1]

        try:
            X_query, y_query, task = reconstruct_query_data(ds_name)
            train_idx = get_train_indices(len(X_query))
        except Exception as e:
            print(f"  FAIL {ds_name}: {e}")
            all_ok = False
            continue

        if len(train_idx) != stored_count:
            print(
                f"  FAIL {ds_name}: expected {stored_count} rows, "
                f"got {len(train_idx)} from n_query={len(X_query)}"
            )
            all_ok = False
        else:
            print(
                f"  OK   {ds_name}: {stored_count} rows, "
                f"n_query={len(X_query)}, emb_dim={emb_dim}"
            )

    # Also verify layerwise embeddings match SAE training data (if available)
    layerwise_dir = (
        PROJECT_ROOT / "output" / "embeddings" / "tabarena_layerwise_round5" / model_key
    )
    if layerwise_dir.exists():
        print("\nVerifying embedding alignment with layerwise extractions...")
        layer = get_extraction_layer(model_key)

        for ds_name in ds_names:
            npz_path = layerwise_dir / f"tabarena_{ds_name}.npz"
            if not npz_path.exists():
                continue

            from scripts.embeddings.extract_layer_embeddings import sort_layer_names

            data = np.load(npz_path, allow_pickle=True)
            layer_names = list(data["layer_names"])
            sorted_names = sort_layer_names(layer_names)

            if layer >= len(sorted_names):
                continue

            layer_emb = data[sorted_names[layer]].astype(np.float32)

            # Apply split_rows to get train portion
            rng = np.random.RandomState(SPLIT_SEED)
            if len(layer_emb) > 500:
                idx = rng.choice(len(layer_emb), 500, replace=False)
                layer_emb_shuffled = layer_emb[idx]
            else:
                idx = rng.permutation(len(layer_emb))
                layer_emb_shuffled = layer_emb[idx]

            n_train = int(len(layer_emb_shuffled) * TRAIN_FRACTION)
            train_emb = layer_emb_shuffled[:n_train]

            stored_emb = per_ds[ds_name]
            if train_emb.shape != stored_emb.shape:
                print(
                    f"  FAIL {ds_name}: shape mismatch "
                    f"{train_emb.shape} vs {stored_emb.shape}"
                )
                all_ok = False
                continue

            max_diff = np.abs(train_emb - stored_emb).max()
            if max_diff < 1e-6:
                print(f"  OK   {ds_name}: max diff = {max_diff:.2e}")
            else:
                print(f"  FAIL {ds_name}: max diff = {max_diff:.2e} (> 1e-6)")
                all_ok = False

    return all_ok


# ── Save outputs ─────────────────────────────────────────────────────────────


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_outputs(
    model_key: str,
    activations: np.ndarray,
    meta_features: np.ndarray,
    row_metadata: List[Dict],
    feature_results: Dict[int, Dict],
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Save all outputs to disk."""
    model_dir = output_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save activations + meta-features
    print(f"\nSaving activations and meta-features...")
    ds_names = [m["dataset"] for m in row_metadata]
    y_values = [m["y"] for m in row_metadata]
    np.savez_compressed(
        str(model_dir / "activations.npz"),
        activations=activations,
        meta_features=meta_features,
        dataset_names=np.array(ds_names),
        y_values=np.array(y_values, dtype=np.float32),
        meta_names=np.array(META_NAMES),
    )

    # Save per-feature JSON files
    feat_dir = model_dir / "feature_samples"
    feat_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(feature_results)} feature sample files...")
    for feat_idx, result in feature_results.items():
        feat_path = feat_dir / f"feature_{feat_idx:04d}.json"
        with open(feat_path, "w") as f:
            json.dump(result, f, indent=2, cls=NumpyEncoder)

    # Save summary
    alive_count = len(feature_results)
    total_features = activations.shape[1]
    active_fracs = [r["activation_stats"]["frac_active"] for r in feature_results.values()]

    summary = {
        "model": model_key,
        "n_total_rows": activations.shape[0],
        "hidden_dim": total_features,
        "n_alive_features": alive_count,
        "n_datasets": len(set(ds_names)),
        "datasets": sorted(set(ds_names)),
        "meta_feature_names": META_NAMES,
        "mean_frac_active": float(np.mean(active_fracs)) if active_fracs else 0.0,
        "median_frac_active": float(np.median(active_fracs)) if active_fracs else 0.0,
    }
    with open(model_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    print(f"Saved to {model_dir}")
    return model_dir


# ── Main pipeline ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Row-level probes for SAE auto-interpretability",
    )
    parser.add_argument(
        "--model", required=True,
        help="Model key (e.g. tabdpt, tabpfn, mitra)",
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Number of high/low activation samples per feature (default: 20)",
    )
    parser.add_argument(
        "--fast", action="store_true", default=True,
        help="Skip supervised meta-features (default: True)",
    )
    parser.add_argument(
        "--no-fast", dest="fast", action="store_false",
        help="Include supervised meta-features (slow)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify row alignment only, don't build full dataset",
    )
    parser.add_argument(
        "--features", type=str, default=None,
        help="Comma-separated feature indices to extract (default: all alive)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for SAE encoding (default: cpu)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: output/row_level_probes)",
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUT_DIR

    print(f"Row-level probes for {args.model}")
    print(f"  Device: {args.device}")
    print(f"  Fast mode: {args.fast}")
    print(f"  Top-k: {args.top_k}")

    # Verify mode
    if args.verify:
        print("\n=== Verification ===")
        ok = verify_row_alignment(args.model)
        sys.exit(0 if ok else 1)

    # Full pipeline
    t_start = time.time()

    # Step 1-4: Build matched dataset with meta-features
    activations, meta_features, row_metadata = build_row_dataset(
        args.model, device=args.device, fast=args.fast,
    )

    # Step 5: Per-feature sample extraction
    print("\nExtracting per-feature samples...")

    # Load feature metadata
    try:
        alive_features = get_alive_features(args.model)
        feature_labels = get_feature_labels(args.model)
        bands = get_matryoshka_bands(args.model)
    except Exception as e:
        print(f"  Warning: Could not load feature metadata: {e}")
        # Fall back to using activation-based alive detection
        alive_features = [
            i for i in range(activations.shape[1])
            if (activations[:, i] > 0).mean() > 1e-3
        ]
        feature_labels = {}
        bands = None

    # Filter to requested features
    if args.features:
        requested = [int(x) for x in args.features.split(",")]
        feature_list = [f for f in requested if f < activations.shape[1]]
    else:
        feature_list = alive_features

    # Skip features with truly zero activations (never fire on any training row)
    max_act = activations.max(axis=0)
    pre_filter = len(feature_list)
    feature_list = [f for f in feature_list if max_act[f] > 0]
    n_skipped = pre_filter - len(feature_list)
    if n_skipped:
        print(f"  Skipped {n_skipped} features (zero activation on all training rows)")

    print(f"  Processing {len(feature_list)} features...")

    feature_results = {}
    for i, feat_idx in enumerate(feature_list):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"    Feature {feat_idx} ({i + 1}/{len(feature_list)})")
        result = extract_feature_samples(
            feat_idx, activations, row_metadata, meta_features,
            top_k=args.top_k, bands=bands, feature_labels=feature_labels,
        )
        feature_results[feat_idx] = result

    # Step 6: Save
    save_outputs(
        args.model, activations, meta_features,
        row_metadata, feature_results, output_dir,
    )

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # Quick sanity check
    dead_features = [
        i for i in range(activations.shape[1])
        if activations[:, i].max() < 0.001
    ]
    print(f"  Alive features (from labels): {len(alive_features)}")
    print(f"  Dead features (max act < 0.001): {len(dead_features)}")
    print(f"  Features extracted: {len(feature_results)}")


if __name__ == "__main__":
    main()
