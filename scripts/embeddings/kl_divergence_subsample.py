#!/usr/bin/env python3
"""
KL divergence test: do SAE training embeddings represent the full dataset?

For each model × dataset, computes KL(full_dataset || train_350) per embedding
dimension. High KL indicates the 350-row subsample misses modes present in the
full dataset, meaning SAEs trained on that subsample may be incomplete.

Approach:
  1. Load the full TabArena dataset (no row cap)
  2. Split context/query using the SAME seed as the original extraction
  3. Extract embeddings at the model's optimal layer for ALL query rows
  4. Load the 350-row SAE training subset (already extracted)
  5. Apply per-dataset normalization (same stats used for SAE training)
  6. Compute per-dimension KL divergence using histogram binning

Output:
  output/sae_kl_divergence/{model}_kl_results.json

Usage:
  python scripts/embeddings/kl_divergence_subsample.py --model tabpfn --device cuda
  python scripts/embeddings/kl_divergence_subsample.py --model all --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from scripts._project_root import PROJECT_ROOT
from data.extended_loader import load_tabarena_dataset, TABARENA_DATASETS
from config import load_optimal_layers, get_optimal_layer
from scripts.embeddings.extract_layer_embeddings import (
    EXTRACT_FN,
    sort_layer_names,
    get_dataset_task,
)

import inspect

SPLIT_SEED = 42
CONTEXT_SIZE = 600  # Same as original extraction
ORIGINAL_QUERY_SIZE = 500  # Original cap
SAE_TRAINING_DIR = PROJECT_ROOT / "output" / "sae_training_round6"
OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_kl_divergence"

# Maximum query rows to embed (practical GPU memory limit)
MAX_QUERY_ROWS = 10000


def load_full_context_query(
    dataset_name: str,
    context_size: int = CONTEXT_SIZE,
    max_query: int = MAX_QUERY_ROWS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load full dataset, split context identically to original extraction.

    The original extraction called load_tabarena_dataset(max_samples=1100),
    which subsamples with RandomState(42) BEFORE splitting context/query.
    We must reproduce that exact subsample to get the same 600 context rows,
    then use all non-context rows (from the full dataset) as query.

    Returns (X_context, y_context, X_query, y_query).
    """
    # Load the FULL dataset (no cap)
    result = load_tabarena_dataset(dataset_name, max_samples=999999)
    if result is None:
        raise ValueError(f"Failed to load dataset: {dataset_name}")

    X_full, y_full, _ = result
    n_full = len(X_full)

    # Reproduce the original 1100-row subsample to identify context rows.
    # load_tabarena_dataset uses RandomState(42).choice(n, 1100, replace=False)
    original_cap = context_size + ORIGINAL_QUERY_SIZE  # 1100
    if n_full > original_cap:
        rng_orig = np.random.RandomState(42)
        original_idx = rng_orig.choice(n_full, original_cap, replace=False)
    else:
        original_idx = np.arange(n_full)

    X_orig = X_full[original_idx]
    y_orig = y_full[original_idx]

    # Now split the original 1100 rows the same way as load_context_query
    task = get_dataset_task(dataset_name)
    n_orig = len(X_orig)
    query_size_orig = n_orig - context_size
    if query_size_orig < 1:
        query_size_orig = n_orig - int(n_orig * 0.7)
        context_size = n_orig - query_size_orig

    if task == "classification":
        le = LabelEncoder()
        y_enc = le.fit_transform(y_orig)
        query_frac = query_size_orig / n_orig
        try:
            X_ctx, _, y_ctx_enc, _ = train_test_split(
                X_orig, y_enc, test_size=query_frac,
                random_state=SPLIT_SEED, stratify=y_enc,
            )
        except ValueError:
            X_ctx, _, y_ctx_enc, _ = train_test_split(
                X_orig, y_enc, test_size=query_frac,
                random_state=SPLIT_SEED,
            )
        y_ctx = y_ctx_enc  # Already encoded
    else:
        rng = np.random.RandomState(SPLIT_SEED)
        idx = rng.permutation(n_orig)
        X_ctx = X_orig[idx[:context_size]]
        y_ctx = y_orig[idx[:context_size]]

    # Query set = ALL rows NOT used as context.
    # Identify context rows in the full dataset by their original indices.
    original_idx_set = set(original_idx.tolist()) if n_full > original_cap else set(range(n_full))

    # Find which original indices ended up in context (by matching rows)
    # Simpler: use all non-original-subsample rows + original query rows
    if n_full > original_cap:
        non_original = np.setdiff1d(np.arange(n_full), original_idx)
    else:
        non_original = np.array([], dtype=int)

    # The original query rows (from the 1100 subsample, minus context)
    # We need them too — they're part of the "full" distribution
    # But for KL test, we want rows the SAE NEVER saw, so use non_original only
    # Plus the original test rows (150 from the 500 not used for SAE training)
    # Actually: use ALL non-context rows for maximum coverage
    if task == "classification":
        le = LabelEncoder()
        y_full_enc = le.fit_transform(y_full)
    else:
        y_full_enc = y_full

    # Use non-original rows as extended query (these rows were never in the pipeline)
    if len(non_original) > 0:
        X_query = X_full[non_original]
        y_query = y_full_enc[non_original] if task == "classification" else y_full[non_original]
    else:
        # Dataset was smaller than 1100 — use original query rows
        if task == "classification":
            _, X_query, _, y_query = train_test_split(
                X_orig, le.fit_transform(y_orig),
                test_size=query_size_orig / n_orig,
                random_state=SPLIT_SEED, stratify=le.fit_transform(y_orig),
            )
        else:
            rng = np.random.RandomState(SPLIT_SEED)
            idx = rng.permutation(n_orig)
            X_query = X_orig[idx[context_size:]]
            y_query = y_orig[idx[context_size:]]

    # Cap query size for GPU memory
    if len(X_query) > max_query:
        rng = np.random.RandomState(SPLIT_SEED + 1)
        sel = rng.choice(len(X_query), max_query, replace=False)
        X_query = X_query[sel]
        y_query = y_query[sel]

    return X_ctx, y_ctx, X_query, y_query


def extract_embeddings(
    model: str,
    layer: int,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    dataset_task: str = "classification",
    device: str = "cuda",
    batch_size: int = 2000,
) -> np.ndarray:
    """Extract embeddings at a specific layer, batching query rows if needed.

    Returns (n_query, hidden_dim) array.
    """
    extract_fn = EXTRACT_FN[model]
    sig = inspect.signature(extract_fn)

    # For large query sets, batch to avoid OOM
    n_query = len(X_query)
    if n_query <= batch_size:
        kwargs = dict(device=device)
        if "task" in sig.parameters:
            kwargs["task"] = dataset_task
        layer_embeddings = extract_fn(X_context, y_context, X_query, **kwargs)
        available = sort_layer_names(list(layer_embeddings.keys()))
        return layer_embeddings[available[layer]]

    # Batch extraction
    all_embs = []
    for start in range(0, n_query, batch_size):
        end = min(start + batch_size, n_query)
        X_batch = X_query[start:end]
        kwargs = dict(device=device)
        if "task" in sig.parameters:
            kwargs["task"] = dataset_task
        layer_embeddings = extract_fn(X_context, y_context, X_batch, **kwargs)
        available = sort_layer_names(list(layer_embeddings.keys()))
        all_embs.append(layer_embeddings[available[layer]])

    return np.concatenate(all_embs, axis=0)


def kl_divergence_per_dim(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    n_bins: int = 50,
) -> np.ndarray:
    """Compute KL(P || Q) per dimension using histogram estimation.

    P = full dataset distribution (what we want to approximate)
    Q = training subsample distribution (what the SAE learned from)

    KL(P || Q) measures information lost when Q is used to approximate P.
    High values indicate the subsample misses modes in the full data.

    Args:
        p_samples: (n_p, d) full dataset embeddings
        q_samples: (n_q, d) training subsample embeddings
        n_bins: Number of histogram bins

    Returns:
        (d,) array of per-dimension KL divergences
    """
    d = p_samples.shape[1]
    kl_values = np.zeros(d)

    for dim in range(d):
        p = p_samples[:, dim]
        q = q_samples[:, dim]

        # Shared bin edges covering both distributions
        lo = min(p.min(), q.min())
        hi = max(p.max(), q.max())
        edges = np.linspace(lo, hi, n_bins + 1)

        # Histograms with Laplace smoothing (avoid log(0))
        p_hist, _ = np.histogram(p, bins=edges, density=False)
        q_hist, _ = np.histogram(q, bins=edges, density=False)

        p_prob = (p_hist + 1) / (p_hist.sum() + n_bins)
        q_prob = (q_hist + 1) / (q_hist.sum() + n_bins)

        # KL(P || Q) = sum P * log(P/Q)
        kl_values[dim] = np.sum(p_prob * np.log(p_prob / q_prob))

    return kl_values


def load_sae_training_subset(
    model: str,
    dataset: str,
    optimal_layer: int,
) -> np.ndarray | None:
    """Load the 350-row training subset for one dataset from pooled SAE data.

    The SAE training file pools all datasets. We need to extract rows
    belonging to a specific dataset.
    """
    train_path = SAE_TRAINING_DIR / f"{model}_layer{optimal_layer}_sae_training.npz"
    if not train_path.exists():
        return None

    data = np.load(train_path, allow_pickle=True)
    embeddings = data["embeddings"]
    samples = data["samples_per_dataset"]

    # Find offset for this dataset
    offset = 0
    count = 0
    for ds_name, ds_count in samples:
        if ds_name == dataset:
            count = int(ds_count)
            break
        offset += int(ds_count)

    if count == 0:
        return None

    return embeddings[offset:offset + count]


def load_norm_stats(model: str, optimal_layer: int, dataset: str):
    """Load per-dataset normalization stats (mean, std) from SAE training."""
    stats_path = SAE_TRAINING_DIR / f"{model}_layer{optimal_layer}_norm_stats.npz"
    if not stats_path.exists():
        return None, None

    data = np.load(stats_path)
    datasets = list(data["datasets"])
    if dataset not in datasets:
        return None, None

    idx = datasets.index(dataset)
    return data["means"][idx], data["stds"][idx]


def analyze_model(model: str, device: str = "cuda") -> dict:
    """Run KL divergence analysis for all datasets of one model.

    Returns dict with per-dataset and aggregate results.
    """
    optimal_layer = get_optimal_layer(model)
    datasets = sorted(TABARENA_DATASETS.keys())

    results = {
        "model": model,
        "optimal_layer": optimal_layer,
        "datasets": {},
        "summary": {},
    }

    all_mean_kl = []
    all_max_kl = []

    for ds_name in datasets:
        task = get_dataset_task(ds_name)

        # Skip regression datasets for classification-only models
        if model in ("hyperfast", "tabicl") and task == "regression":
            print(f"  {ds_name} ({task})... SKIP (cls-only model)")
            continue

        print(f"  {ds_name} ({task})...", end=" ", flush=True)

        # Load normalization stats
        mean, std = load_norm_stats(model, optimal_layer, ds_name)
        if mean is None:
            print("SKIP (no norm stats)")
            continue

        # Load SAE training subset (already normalized)
        train_subset = load_sae_training_subset(model, ds_name, optimal_layer)
        if train_subset is None:
            print("SKIP (no training data)")
            continue

        # Load full dataset and extract embeddings
        try:
            X_ctx, y_ctx, X_query, y_query = load_full_context_query(ds_name)
        except (ValueError, Exception) as e:
            print(f"SKIP (load: {e})")
            continue

        n_full = len(X_query)
        n_train = len(train_subset)

        if n_full <= n_train * 1.5:
            # Not enough extra data to make the comparison meaningful
            print(f"SKIP (only {n_full} query rows, need >>{n_train})")
            continue

        try:
            full_emb = extract_embeddings(
                model, optimal_layer, X_ctx, y_ctx, X_query,
                dataset_task=task, device=device,
            )
        except Exception as e:
            print(f"SKIP (extract: {e})")
            continue

        # Normalize full embeddings with the SAME stats used for SAE training
        full_emb_norm = (full_emb - mean) / std

        # Compute KL divergence per dimension
        kl_per_dim = kl_divergence_per_dim(full_emb_norm, train_subset)

        mean_kl = float(np.mean(kl_per_dim))
        max_kl = float(np.max(kl_per_dim))
        median_kl = float(np.median(kl_per_dim))
        p95_kl = float(np.percentile(kl_per_dim, 95))

        results["datasets"][ds_name] = {
            "n_full": n_full,
            "n_train": n_train,
            "subsample_ratio": round(n_train / n_full, 4),
            "mean_kl": round(mean_kl, 6),
            "median_kl": round(median_kl, 6),
            "p95_kl": round(p95_kl, 6),
            "max_kl": round(max_kl, 6),
            "task": task,
        }

        all_mean_kl.append(mean_kl)
        all_max_kl.append(max_kl)

        print(f"n={n_full}, KL mean={mean_kl:.4f} max={max_kl:.4f}")

    if all_mean_kl:
        results["summary"] = {
            "n_datasets": len(all_mean_kl),
            "mean_kl_across_datasets": round(float(np.mean(all_mean_kl)), 6),
            "median_kl_across_datasets": round(float(np.median(all_mean_kl)), 6),
            "max_kl_across_datasets": round(float(np.max(all_mean_kl)), 6),
            "mean_max_kl": round(float(np.mean(all_max_kl)), 6),
            "datasets_with_high_kl": [
                ds for ds, r in results["datasets"].items()
                if r["mean_kl"] > 0.1
            ],
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="KL divergence test: SAE training subsample vs full dataset"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or 'all'")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets (default: all TabArena)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_optimal_layers()
    if args.model == "all":
        # Skip tabula8b by default — too slow for batch extraction
        models = sorted(k for k in EXTRACT_FN.keys() if k != "tabula8b")
    else:
        models = [args.model]

    for model in models:
        print(f"\n{'='*60}")
        print(f"KL divergence analysis: {model}")
        print(f"{'='*60}")

        results = analyze_model(model, device=args.device)

        out_path = OUTPUT_DIR / f"{model}_kl_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved: {out_path}")
        if results["summary"]:
            s = results["summary"]
            print(f"  Mean KL across datasets: {s['mean_kl_across_datasets']:.4f}")
            print(f"  Median KL: {s['median_kl_across_datasets']:.4f}")
            print(f"  High-KL datasets (>0.1): {s['datasets_with_high_kl']}")


if __name__ == "__main__":
    main()
