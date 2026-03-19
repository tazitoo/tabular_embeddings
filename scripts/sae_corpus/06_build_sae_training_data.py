#!/usr/bin/env python3
"""Build SAE training data with full-fold normalization and stratified sampling.

For each model, loads all-layer embeddings from 04_extract_all_layers.py output,
selects the optimal layer, and builds pooled train/test files.

Key improvements over build_sae_training_data.py (round 6):
  1. Normalization stats computed from the FULL holdout fold, not the subsample.
  2. Doubly stratified sampling (target class × difficulty tercile) instead of
     random subsampling — ensures representation across easy/medium/hard rows
     and all target classes.
  3. 700-row cap (500 train + 200 test) per dataset; small datasets contribute
     all available rows with proportional train/test split.

Difficulty is measured by per-row loss from TabPFN predictions on the holdout
set.  This uses the preprocessed cache and OOF predictions already computed in
steps 01-03.

Output structure:
    output/sae_training_round10/{model}_layer{N}_sae_training.npz
    output/sae_training_round10/{model}_layer{N}_sae_test.npz
    output/sae_training_round10/{model}_layer{N}_norm_stats.npz

Each train/test file contains:
    embeddings:          (n_total, hidden_dim) float32, per-dataset normalized
    optimal_layer:       int
    layer_name:          string
    source_datasets:     list of dataset names
    samples_per_dataset: structured array (dataset, count)
    split:               "train" or "test"

The norm_stats file contains:
    datasets:  (n_datasets,) sorted dataset names
    means:     (n_datasets, hidden_dim) per-dataset means from full fold
    stds:      (n_datasets, hidden_dim) per-dataset stds from full fold

Usage:
    python scripts/sae_corpus/06_build_sae_training_data.py --model tabpfn
    python scripts/sae_corpus/06_build_sae_training_data.py --model all
    python scripts/sae_corpus/06_build_sae_training_data.py --model mitra --device cuda
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import get_optimal_layer, load_optimal_layers
from data.preprocessing import CACHE_DIR, load_preprocessed
from models.layer_extraction import load_and_fit, predict, sort_layer_names
from scripts._project_root import PROJECT_ROOT

SPLITS_PATH = PROJECT_ROOT / "output" / "sae_training_round9" / "tabarena_splits.json"
EMBEDDINGS_DIR = PROJECT_ROOT / "output" / "sae_training_round9" / "embeddings"
OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_training_round10"

SAMPLE_CAP = 700
TRAIN_FRAC = 500 / 700  # ≈ 71.4%
MIN_STD = 1e-8

# Models that have preprocessed cache for TabPFN difficulty scoring
DIFFICULTY_MODELS = {"tabpfn"}


# ---------------------------------------------------------------------------
# Difficulty scoring (from 05_sampling_analysis.py)
# ---------------------------------------------------------------------------

def per_row_loss(y_true: np.ndarray, proba: np.ndarray, task_type: str) -> np.ndarray:
    """Compute per-row loss: log-loss for classification, squared error for regression."""
    eps = 1e-15
    if task_type == "classification":
        if proba.ndim == 1:
            p = np.clip(proba, eps, 1 - eps)
            return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
        n = len(y_true)
        p = np.clip(proba[np.arange(n), y_true], eps, 1 - eps)
        return -np.log(p)
    else:
        return (y_true - proba) ** 2


def tercile_labels(losses: np.ndarray) -> np.ndarray:
    """Assign each row to easy (0), medium (1), or hard (2) tercile."""
    t1 = np.percentile(losses, 33.3)
    t2 = np.percentile(losses, 66.7)
    labels = np.zeros(len(losses), dtype=int)
    labels[losses > t1] = 1
    labels[losses > t2] = 2
    return labels


def compute_difficulty(
    dataset_name: str, task_type: str, y_test: np.ndarray, device: str,
    max_context: int = 1024,
) -> np.ndarray | None:
    """Compute per-row difficulty scores using TabPFN on the holdout set.

    Returns losses array or None if TabPFN data unavailable.
    """
    try:
        data = load_preprocessed("tabpfn", dataset_name, CACHE_DIR)
    except FileNotFoundError:
        return None

    X_ctx, y_ctx = data.X_train, data.y_train
    if len(X_ctx) > max_context:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_ctx), max_context, replace=False)
        X_ctx, y_ctx = X_ctx[idx], y_ctx[idx]

    fit_kwargs = {}
    if data.cat_indices:
        fit_kwargs["cat_indices"] = data.cat_indices

    clf = load_and_fit("tabpfn", X_ctx, y_ctx, task=task_type, device=device, **fit_kwargs)
    preds = predict(clf, data.X_test, task=task_type)
    return per_row_loss(data.y_test, preds, task_type)


# ---------------------------------------------------------------------------
# Stratified sampling (from 05_sampling_analysis.py)
# ---------------------------------------------------------------------------

def sample_target_x_difficulty(
    y: np.ndarray, losses: np.ndarray, n_sample: int, seed: int = 42,
) -> np.ndarray:
    """Doubly stratified sampling: target × difficulty with redistribution.

    Equal allocation across strata (class × tercile). When a stratum is smaller
    than its share, take all of it and redistribute the surplus to remaining
    strata. Repeats until budget is spent — no wasted budget, no oversampling.
    """
    rng = np.random.RandomState(seed)
    terciles = tercile_labels(losses)
    classes = np.unique(y)

    strata = {}
    for cls in classes:
        for t in range(3):
            strata[(cls, t)] = np.where((y == cls) & (terciles == t))[0]

    allocation = {k: 0 for k in strata}
    remaining_budget = n_sample
    active_keys = [k for k in strata if len(strata[k]) > 0]

    while remaining_budget > 0 and active_keys:
        per_stratum = max(1, remaining_budget // len(active_keys))
        exhausted = []
        for k in active_keys:
            available = len(strata[k]) - allocation[k]
            take = min(per_stratum, available, remaining_budget)
            allocation[k] += take
            remaining_budget -= take
            if allocation[k] >= len(strata[k]):
                exhausted.append(k)
            if remaining_budget <= 0:
                break
        for k in exhausted:
            active_keys.remove(k)

    indices = []
    for k, n_take in allocation.items():
        if n_take > 0:
            indices.append(rng.choice(strata[k], size=n_take, replace=False))
    return np.concatenate(indices) if indices else np.array([], dtype=int)


def sample_target_stratified(
    y: np.ndarray, n_sample: int, seed: int = 42,
) -> np.ndarray:
    """Sample proportional to class frequency (fallback when no difficulty scores)."""
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    indices = []
    for cls, count in zip(classes, counts):
        cls_idx = np.where(y == cls)[0]
        n_take = max(1, int(n_sample * count / len(y)))
        indices.append(rng.choice(cls_idx, size=min(n_take, len(cls_idx)), replace=False))
    indices = np.concatenate(indices)
    if len(indices) > n_sample:
        indices = rng.choice(indices, size=n_sample, replace=False)
    return indices


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def load_embeddings_at_layer(
    model: str, dataset: str, optimal_layer: int,
) -> np.ndarray | None:
    """Load embeddings for a single dataset at the optimal layer."""
    npz_path = EMBEDDINGS_DIR / model / f"{dataset}.npz"
    if not npz_path.exists():
        return None

    data = np.load(npz_path, allow_pickle=True)
    layer_names = sort_layer_names(list(data["layer_names"]))

    if optimal_layer >= len(layer_names):
        return None

    layer_key = layer_names[optimal_layer]
    return data[layer_key].astype(np.float32)


def select_sample(
    n_holdout: int, y_test: np.ndarray, losses: np.ndarray | None, task_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Select train/test row indices using stratified sampling.

    Returns (train_indices, test_indices) into the holdout set.
    """
    if n_holdout <= SAMPLE_CAP:
        # Small dataset: take all rows, split proportionally
        n_train = round(n_holdout * TRAIN_FRAC)
        rng = np.random.RandomState(42)
        perm = rng.permutation(n_holdout)
        return perm[:n_train], perm[n_train:]

    # Large dataset: stratified sample of SAMPLE_CAP rows
    n_train = 500
    n_test = 200

    if losses is not None and task_type == "classification":
        # Doubly stratified: target × difficulty
        train_idx = sample_target_x_difficulty(y_test, losses, n_train)
        # For test: sample from rows NOT in train
        remaining = np.setdiff1d(np.arange(n_holdout), train_idx)
        remaining_losses = losses[remaining]
        remaining_y = y_test[remaining]
        test_idx = sample_target_x_difficulty(remaining_y, remaining_losses, n_test)
        test_idx = remaining[test_idx]  # Map back to original indices
    elif losses is not None:
        # Regression: difficulty-stratified (no class axis)
        rng = np.random.RandomState(42)
        terciles = tercile_labels(losses)
        per_tercile_train = n_train // 3
        per_tercile_test = n_test // 3
        train_idx, test_idx = [], []
        for t in range(3):
            t_idx = np.where(terciles == t)[0]
            rng.shuffle(t_idx)
            train_idx.append(t_idx[:min(per_tercile_train, len(t_idx))])
            remaining_t = t_idx[min(per_tercile_train, len(t_idx)):]
            test_idx.append(remaining_t[:min(per_tercile_test, len(remaining_t))])
        train_idx = np.concatenate(train_idx)
        test_idx = np.concatenate(test_idx)
    else:
        # Fallback: target-stratified for classification, random for regression
        if task_type == "classification":
            train_idx = sample_target_stratified(y_test, n_train)
            remaining = np.setdiff1d(np.arange(n_holdout), train_idx)
            rng = np.random.RandomState(43)
            test_idx = rng.choice(remaining, size=min(n_test, len(remaining)), replace=False)
        else:
            rng = np.random.RandomState(42)
            perm = rng.permutation(n_holdout)
            train_idx = perm[:n_train]
            test_idx = perm[n_train:n_train + n_test]

    return train_idx, test_idx


def build_training_data(
    model: str, device: str = "cuda",
) -> dict:
    """Build pooled SAE train+test data for a model.

    Pipeline per dataset:
      1. Load full holdout embeddings at optimal layer
      2. Compute norm stats (mean, std) from FULL fold
      3. Normalize all rows with full-fold stats
      4. Stratified subsample → 500 train + 200 test (or all if < 700)
      5. Pool across datasets
    """
    optimal_layer = get_optimal_layer(model)
    splits = json.loads(SPLITS_PATH.read_text())
    dataset_names = sorted(splits.keys())

    # Filter by task compatibility
    model_key = model.lower()
    if model_key in ("hyperfast", "tabicl"):
        dataset_names = [d for d in dataset_names if splits[d]["task_type"] == "classification"]

    print(f"  Optimal layer: {optimal_layer}")
    print(f"  Datasets: {len(dataset_names)}")
    print(f"  Sample cap: {SAMPLE_CAP} (train={int(SAMPLE_CAP * TRAIN_FRAC)}, "
          f"test={SAMPLE_CAP - int(SAMPLE_CAP * TRAIN_FRAC)})")

    # Pre-compute difficulty scores via TabPFN (once, reused across models)
    difficulty_cache = {}

    train_embeddings = []
    test_embeddings = []
    train_samples = {}
    test_samples = {}
    norm_stats = {}
    skipped = 0

    for i, ds_name in enumerate(dataset_names):
        split_info = splits[ds_name]
        task_type = split_info["task_type"]
        test_indices = np.array(split_info["test_indices"], dtype=np.int32)

        # Load embeddings at optimal layer
        emb = load_embeddings_at_layer(model, ds_name, optimal_layer)
        if emb is None:
            print(f"  [{i+1}/{len(dataset_names)}] {ds_name}: SKIP (no embeddings)")
            skipped += 1
            continue

        n_holdout = len(emb)

        # 1. Compute norm stats from FULL fold
        ds_mean = emb.mean(axis=0)
        ds_std = emb.std(axis=0)
        ds_std[ds_std < MIN_STD] = 1.0
        norm_stats[ds_name] = {"mean": ds_mean, "std": ds_std}

        # 2. Normalize all rows with full-fold stats
        emb_norm = (emb - ds_mean) / ds_std

        # 3. Compute difficulty scores (lazy, cached)
        if ds_name not in difficulty_cache:
            y_test = np.array(split_info.get("test_labels", []))
            # Load y_test from preprocessed cache if not in splits
            if len(y_test) == 0:
                try:
                    data = load_preprocessed("tabpfn", ds_name, CACHE_DIR)
                    y_test = data.y_test
                except FileNotFoundError:
                    # Try any available model's cache
                    for try_model in ("tabpfn", "mitra", "tabdpt", "tabicl_v2"):
                        try:
                            data = load_preprocessed(try_model, ds_name, CACHE_DIR)
                            y_test = data.y_test
                            break
                        except FileNotFoundError:
                            continue

            if len(y_test) == n_holdout:
                losses = compute_difficulty(ds_name, task_type, y_test, device)
                difficulty_cache[ds_name] = (y_test, losses)
            else:
                difficulty_cache[ds_name] = (None, None)

        y_test, losses = difficulty_cache[ds_name]

        # 4. Stratified subsample
        if y_test is not None and len(y_test) == n_holdout:
            train_idx, test_idx = select_sample(n_holdout, y_test, losses, task_type)
        else:
            # No labels available: random split
            rng = np.random.RandomState(42)
            if n_holdout <= SAMPLE_CAP:
                perm = rng.permutation(n_holdout)
                n_train = round(n_holdout * TRAIN_FRAC)
                train_idx, test_idx = perm[:n_train], perm[n_train:]
            else:
                perm = rng.permutation(n_holdout)
                train_idx, test_idx = perm[:500], perm[500:700]

        train_emb = emb_norm[train_idx]
        test_emb = emb_norm[test_idx]

        train_embeddings.append(train_emb)
        test_embeddings.append(test_emb)
        train_samples[ds_name] = len(train_emb)
        test_samples[ds_name] = len(test_emb)

        sampling = "all" if n_holdout <= SAMPLE_CAP else "stratified"
        diff_str = "T×D" if losses is not None and task_type == "classification" else (
            "diff" if losses is not None else "rnd")
        print(f"  [{i+1}/{len(dataset_names)}] {ds_name}: "
              f"{len(train_emb)}+{len(test_emb)} ({sampling}/{diff_str}) "
              f"dim={emb.shape[1]}")

    if not train_embeddings:
        raise ValueError(f"No embeddings loaded for {model}")

    # Pool across datasets
    train_pooled = np.concatenate(train_embeddings, axis=0)
    test_pooled = np.concatenate(test_embeddings, axis=0)
    layer_name = f"layer_{optimal_layer}"

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / f"{model}_layer{optimal_layer}_sae_training.npz"
    test_path = OUTPUT_DIR / f"{model}_layer{optimal_layer}_sae_test.npz"
    stats_path = OUTPUT_DIR / f"{model}_layer{optimal_layer}_norm_stats.npz"

    config = load_optimal_layers()[model]

    def _save(path, embeddings, samples_dict, split_name):
        np.savez_compressed(
            str(path),
            embeddings=embeddings,
            optimal_layer=np.array(optimal_layer),
            layer_name=np.array(layer_name),
            source_datasets=np.array(list(samples_dict.keys())),
            samples_per_dataset=np.array(
                [(k, v) for k, v in samples_dict.items()],
                dtype=[("dataset", "U100"), ("count", "i4")],
            ),
            split=np.array(split_name),
            config=np.array(json.dumps(config)),
        )

    _save(train_path, train_pooled, train_samples, "train")
    _save(test_path, test_pooled, test_samples, "test")

    # Save per-dataset normalization stats (from full fold)
    ds_names = sorted(norm_stats.keys())
    np.savez_compressed(
        str(stats_path),
        datasets=np.array(ds_names),
        means=np.stack([norm_stats[d]["mean"] for d in ds_names]),
        stds=np.stack([norm_stats[d]["std"] for d in ds_names]),
    )

    n_datasets = len(train_samples)
    print(f"\n  Train: {train_pooled.shape} from {n_datasets} datasets → {train_path.name}")
    print(f"  Test:  {test_pooled.shape} from {n_datasets} datasets → {test_path.name}")
    print(f"  Norm stats: {len(norm_stats)} datasets (full-fold) → {stats_path.name}")
    if skipped:
        print(f"  Skipped: {skipped} datasets (no embeddings)")

    return {
        "model": model,
        "optimal_layer": optimal_layer,
        "layer_name": layer_name,
        "train_shape": train_pooled.shape,
        "test_shape": test_pooled.shape,
        "n_datasets": n_datasets,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "stats_path": str(stats_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build SAE training data with full-fold normalization"
    )
    parser.add_argument("--model", required=True,
                        help="Model name or 'all' for all available models")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_optimal_layers()

    if args.model == "all":
        # Only models that have embeddings extracted
        models = []
        for m in config.keys():
            model_dir = EMBEDDINGS_DIR / m
            if model_dir.exists() and any(model_dir.glob("*.npz")):
                models.append(m)
        models.sort()
    else:
        models = [args.model]

    if not models:
        print(f"No models with embeddings found in {EMBEDDINGS_DIR}")
        print("Run 04_extract_all_layers.py first.")
        return

    print(f"Building SAE training data (round 10)")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Models: {models}")

    results = []
    for model in models:
        print(f"\n{'=' * 60}")
        print(f"  Model: {model}")
        print("=" * 60)

        try:
            result = build_training_data(model, device=args.device)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if results:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Model':<12} {'Layer':>6} {'Train':>20} {'Test':>20} {'Datasets':>10}")
        print("-" * 70)
        for r in results:
            print(
                f"{r['model']:<12} {r['optimal_layer']:>6} "
                f"{str(r['train_shape']):>20} {str(r['test_shape']):>20} "
                f"{r['n_datasets']:>10}"
            )


if __name__ == "__main__":
    main()
