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


def default_output_dir() -> Path:
    """Corpus directory for the default SAE round (what the sweep reads)."""
    from scripts.sae.compare_sae_cross_model import DEFAULT_SAE_ROUND
    return PROJECT_ROOT / "output" / f"sae_training_round{DEFAULT_SAE_ROUND}"


OUTPUT_DIR = default_output_dir()
CKA_DIR = PROJECT_ROOT / "output" / "layerwise_cka_v2"
DEPTH_ANALYSIS_DIR = PROJECT_ROOT / "output"

SAMPLE_CAP = 700
TRAIN_FRAC = 500 / 700  # ≈ 71.4%
MIN_STD = 1e-8

# Models that have preprocessed cache for TabPFN difficulty scoring
DIFFICULTY_MODELS = {"tabpfn"}


def load_per_dataset_layers(model: str) -> dict[str, int]:
    """Load per-dataset critical layers from CKA depth analysis.

    Returns {dataset_name: critical_layer_index}.
    """
    analysis_path = DEPTH_ANALYSIS_DIR / f"layerwise_depth_analysis_{model}.json"
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"No depth analysis for {model}: {analysis_path}\n"
            f"Run 07_compute_layerwise_cka.py + analyze_layerwise_cka.py first."
        )
    data = json.loads(analysis_path.read_text())
    return {info["dataset"]: info["critical_layer"] for info in data.values()}


def load_task_aware_fixed_layers(model: str) -> dict[str, int]:
    """Compute per-variant fixed layers from CKA depth analysis.

    Groups datasets by (n_layers, task_type) — models with the same
    architecture but different training objectives (e.g. CARTE classifier
    vs regressor) may dedicate layers differently. Falls back to n_layers
    only when splits are unavailable.
    """
    analysis_path = DEPTH_ANALYSIS_DIR / f"layerwise_depth_analysis_{model}.json"
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"No depth analysis for {model}: {analysis_path}\n"
            f"Run 07_compute_layerwise_cka.py + analyze_layerwise_cka.py first."
        )
    data = json.loads(analysis_path.read_text())

    # Load splits to get task type per dataset
    task_types = {}
    if SPLITS_PATH.exists():
        splits = json.loads(SPLITS_PATH.read_text())
        task_types = {ds: s["task_type"] for ds, s in splits.items()}

    # Group by (n_layers, task_type)
    groups: dict[tuple, list] = {}
    for info in data.values():
        nl = info["n_layers"]
        ds = info["dataset"]
        tt = task_types.get(ds, "unknown")
        groups.setdefault((nl, tt), []).append(info)

    # Compute optimal fixed layer per group
    lookup = {}
    for (nl, tt), infos in sorted(groups.items()):
        crit_layers = [i["critical_layer"] for i in infos]
        optimal = int(round(np.mean(crit_layers)))
        for info in infos:
            lookup[info["dataset"]] = optimal
        print(f"    {nl}-layer {tt}: {len(infos)} datasets → L{optimal} "
              f"(mean={np.mean(crit_layers):.1f}±{np.std(crit_layers):.1f})")

    return lookup


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


def load_reference_rows(train_npz: Path, test_npz: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-dataset (train_rows, test_rows) global row ids from a reference corpus pair.

    Used to pin a regenerated model's corpus to the rows every other model's corpus
    already uses: the stratified sampler below depends on a live TabPFN difficulty
    pass, which is environment-sensitive, so resampling in a new environment picks
    different rows and breaks cross-model row alignment.

    Training corpora written before row bookkeeping was added carry no `row_indices`;
    their train entry is None and the caller samples train rows from the remainder.
    """
    out: dict[str, list] = {}
    for split, path in (("train", train_npz), ("test", test_npz)):
        z = np.load(path, allow_pickle=True)
        if "row_indices" not in z.files:
            for name, _ in z["samples_per_dataset"]:
                out.setdefault(str(name), [None, None])
            continue
        rows, offset = z["row_indices"], 0
        for name, n in z["samples_per_dataset"]:
            out.setdefault(str(name), [None, None])[0 if split == "train" else 1] = \
                np.asarray(rows[offset:offset + int(n)], dtype=np.int32)
            offset += int(n)
    return {d: (tr, te) for d, (tr, te) in out.items()}


def select_sample_from_reference(
    holdout_indices: np.ndarray, ref_train: np.ndarray, ref_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Positions (into the holdout) of the reference rows, in reference order."""
    pos = {int(r): i for i, r in enumerate(np.asarray(holdout_indices))}
    missing = [int(r) for r in np.concatenate([ref_train, ref_test]) if int(r) not in pos]
    if missing:
        raise ValueError(f"{len(missing)} reference rows are not in this holdout, e.g. {missing[:5]}")
    return (np.array([pos[int(r)] for r in ref_train], dtype=np.int64),
            np.array([pos[int(r)] for r in ref_test], dtype=np.int64))


def select_sample(
    n_holdout: int, y_test: np.ndarray, losses: np.ndarray | None, task_type: str,
    pinned_test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select train/test row indices using stratified sampling.

    Returns (train_indices, test_indices) into the holdout set. With `pinned_test`
    the test rows are taken as given (cross-model alignment) and only the train rows
    are sampled, from the remainder, with the same stratification.
    """
    if pinned_test is not None:
        test_idx = np.asarray(pinned_test, dtype=np.int64)
        remaining = np.setdiff1d(np.arange(n_holdout), test_idx)
        if n_holdout <= SAMPLE_CAP:
            return remaining, test_idx
        n_train = 500
        if losses is None:
            raise ValueError("Difficulty scores are required to sample train rows around pinned test rows.")
        if task_type == "classification":
            pick = sample_target_x_difficulty(y_test[remaining], losses[remaining], n_train)
        else:
            rng = np.random.RandomState(42)
            terciles = tercile_labels(losses[remaining])
            pick = []
            for t in range(3):
                t_idx = np.where(terciles == t)[0]
                rng.shuffle(t_idx)
                pick.append(t_idx[:min(n_train // 3, len(t_idx))])
            pick = np.concatenate(pick)
        return remaining[pick], test_idx

    if n_holdout <= SAMPLE_CAP:
        # Small dataset: take all rows, split proportionally
        n_train = round(n_holdout * TRAIN_FRAC)
        rng = np.random.RandomState(42)
        perm = rng.permutation(n_holdout)
        return perm[:n_train], perm[n_train:]

    # Large dataset: stratified sample of SAMPLE_CAP rows
    n_train = 500
    n_test = 200

    if losses is None:
        raise ValueError(
            f"Difficulty scores are required for stratified sampling but were None. "
            f"Ensure TabPFN preprocessing cache exists for this dataset."
        )

    if task_type == "classification":
        # Doubly stratified: target × difficulty
        train_idx = sample_target_x_difficulty(y_test, losses, n_train)
        # For test: sample from rows NOT in train
        remaining = np.setdiff1d(np.arange(n_holdout), train_idx)
        remaining_losses = losses[remaining]
        remaining_y = y_test[remaining]
        test_idx = sample_target_x_difficulty(remaining_y, remaining_losses, n_test)
        test_idx = remaining[test_idx]  # Map back to original indices
    else:
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

    return train_idx, test_idx


def build_training_data(
    model: str, device: str = "cuda", layer_mode: str = "fixed_task_aware",
    reference_rows: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    """Build pooled SAE train+test data for a model.

    Pipeline per dataset:
      1. Load full holdout embeddings at the chosen layer
      2. Compute norm stats (mean, std) from FULL fold
      3. Normalize all rows with full-fold stats
      4. Stratified subsample → 500 train + 200 test (or all if < 700)
      5. Pool across datasets

    Args:
        layer_mode: "fixed" uses config/optimal_extraction_layers.json (one
            layer for all datasets). "per_dataset" uses the per-dataset
            critical layer from CKA depth analysis.
    """
    global_optimal = get_optimal_layer(model)
    splits = json.loads(SPLITS_PATH.read_text())
    dataset_names = sorted(splits.keys())

    # Filter by task compatibility
    model_key = model.lower()
    if model_key in ("hyperfast", "tabicl"):
        dataset_names = [d for d in dataset_names if splits[d]["task_type"] == "classification"]

    # Resolve layer selection strategy
    if layer_mode == "per_dataset":
        per_ds_layers = load_per_dataset_layers(model)
        print(f"  Layer mode: per-dataset (from CKA depth analysis)")
        print(f"  Layer range: [{min(per_ds_layers.values())}, {max(per_ds_layers.values())}]")
    elif layer_mode == "fixed_task_aware":
        per_ds_layers = load_task_aware_fixed_layers(model)
        print(f"  Layer mode: fixed-task-aware")
    else:
        per_ds_layers = None
        print(f"  Layer mode: fixed (L{global_optimal})")

    print(f"  Datasets: {len(dataset_names)}")
    print(f"  Sample cap: {SAMPLE_CAP} (train={int(SAMPLE_CAP * TRAIN_FRAC)}, "
          f"test={SAMPLE_CAP - int(SAMPLE_CAP * TRAIN_FRAC)})")

    # Pre-compute difficulty scores via TabPFN (once, reused across models)
    difficulty_cache = {}

    train_embeddings = []
    test_embeddings = []
    train_samples = {}
    test_samples = {}
    train_row_indices = {}
    test_row_indices = {}
    norm_stats = {}
    skipped = 0

    for i, ds_name in enumerate(dataset_names):
        split_info = splits[ds_name]
        task_type = split_info["task_type"]
        test_indices = np.array(split_info["test_indices"], dtype=np.int32)

        # Resolve layer for this dataset
        if per_ds_layers is not None and ds_name in per_ds_layers:
            ds_layer = per_ds_layers[ds_name]
        else:
            ds_layer = global_optimal

        # Load embeddings at chosen layer
        emb = load_embeddings_at_layer(model, ds_name, ds_layer)
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

        holdout_indices = np.array(split_info["test_indices"], dtype=np.int32)
        pinned_test = None
        if reference_rows is not None:
            ref_train, ref_test = reference_rows[ds_name]
            if ref_train is not None:
                # Both splits recorded: no difficulty pass, no resampling.
                train_idx, test_idx = select_sample_from_reference(holdout_indices, ref_train, ref_test)
            else:
                # Only the test rows are recorded: pin them, sample train from the rest.
                _, pinned_test = select_sample_from_reference(holdout_indices, np.array([], dtype=np.int32), ref_test)
        need_losses = reference_rows is None or (pinned_test is not None and n_holdout > SAMPLE_CAP)
        # 3. Compute difficulty scores (lazy, cached)
        if need_losses and ds_name not in difficulty_cache:
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

            if len(y_test) != n_holdout:
                raise ValueError(
                    f"{ds_name}: y_test length ({len(y_test)}) != n_holdout ({n_holdout}). "
                    f"Preprocessing cache may be out of sync with embeddings."
                )
            losses = compute_difficulty(ds_name, task_type, y_test, device)
            difficulty_cache[ds_name] = (y_test, losses)

        losses = None
        if need_losses:
            y_test, losses = difficulty_cache[ds_name]
        if reference_rows is None or pinned_test is not None:
            # 4. Stratified subsample (test rows pinned when a reference gave them)
            y_for_sample = y_test if need_losses else np.zeros(n_holdout)
            train_idx, test_idx = select_sample(n_holdout, y_for_sample, losses, task_type,
                                                pinned_test=pinned_test)

        train_emb = emb_norm[train_idx]
        test_emb = emb_norm[test_idx]

        train_embeddings.append(train_emb)
        test_embeddings.append(test_emb)
        train_samples[ds_name] = len(train_emb)
        test_samples[ds_name] = len(test_emb)

        # Map subsample indices back to original dataset row indices
        train_row_indices[ds_name] = holdout_indices[train_idx]
        test_row_indices[ds_name] = holdout_indices[test_idx]

        sampling = ("reference" if reference_rows is not None and pinned_test is None else
                    "pinned-test" if pinned_test is not None else
                    "all" if n_holdout <= SAMPLE_CAP else "stratified")
        diff_str = "T×D" if losses is not None and task_type == "classification" else (
            "diff" if losses is not None else "rnd")
        layer_str = f"L{ds_layer}" if per_ds_layers else ""
        print(f"  [{i+1}/{len(dataset_names)}] {ds_name}: "
              f"{len(train_emb)}+{len(test_emb)} ({sampling}/{diff_str}) "
              f"dim={emb.shape[1]} {layer_str}")

    if not train_embeddings:
        raise ValueError(f"No embeddings loaded for {model}")

    # Pool across datasets
    train_pooled = np.concatenate(train_embeddings, axis=0)
    test_pooled = np.concatenate(test_embeddings, axis=0)

    # Output naming
    if layer_mode == "per_dataset":
        tag = f"{model}_perds"
        layer_name = "per_dataset"
        reported_layer = -1  # sentinel
    elif layer_mode == "fixed_task_aware":
        tag = f"{model}_taskaware"
        layer_name = "fixed_task_aware"
        reported_layer = -1
    else:
        tag = f"{model}_layer{global_optimal}"
        layer_name = f"layer_{global_optimal}"
        reported_layer = global_optimal

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / f"{tag}_sae_training.npz"
    test_path = OUTPUT_DIR / f"{tag}_sae_test.npz"
    stats_path = OUTPUT_DIR / f"{tag}_norm_stats.npz"

    config = load_optimal_layers()[model]

    def _save(path, embeddings, samples_dict, row_indices_dict, split_name):
        # Concatenate per-dataset row indices in dataset order
        ds_order = list(samples_dict.keys())
        all_indices = np.concatenate([row_indices_dict[ds] for ds in ds_order])
        np.savez_compressed(
            str(path),
            embeddings=embeddings,
            optimal_layer=np.array(reported_layer),
            layer_name=np.array(layer_name),
            source_datasets=np.array(ds_order),
            samples_per_dataset=np.array(
                [(k, v) for k, v in samples_dict.items()],
                dtype=[("dataset", "U100"), ("count", "i4")],
            ),
            row_indices=all_indices,
            split=np.array(split_name),
            config=np.array(json.dumps(config)),
            layer_mode=np.array(layer_mode),
        )

    _save(train_path, train_pooled, train_samples, train_row_indices, "train")
    _save(test_path, test_pooled, test_samples, test_row_indices, "test")

    # Save per-dataset normalization stats (from full fold)
    # Include per-dataset extraction layers for downstream consumers
    ds_names = sorted(norm_stats.keys())
    stats_save = {
        "datasets": np.array(ds_names),
        "means": np.stack([norm_stats[d]["mean"] for d in ds_names]),
        "stds": np.stack([norm_stats[d]["std"] for d in ds_names]),
        "layer_mode": np.array(layer_mode),
    }
    if per_ds_layers is not None:
        stats_save["layers"] = np.array([per_ds_layers.get(d, global_optimal) for d in ds_names])
    else:
        stats_save["layers"] = np.array([global_optimal] * len(ds_names))

    np.savez_compressed(str(stats_path), **stats_save)

    n_datasets = len(train_samples)
    print(f"\n  Train: {train_pooled.shape} from {n_datasets} datasets → {train_path.name}")
    print(f"  Test:  {test_pooled.shape} from {n_datasets} datasets → {test_path.name}")
    print(f"  Norm stats: {len(norm_stats)} datasets (full-fold) → {stats_path.name}")
    if skipped:
        print(f"  Skipped: {skipped} datasets (no embeddings)")

    return {
        "model": model,
        "layer_mode": layer_mode,
        "train_shape": train_pooled.shape,
        "test_shape": test_pooled.shape,
        "n_datasets": n_datasets,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "stats_path": str(stats_path),
    }


def main():
    global EMBEDDINGS_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(
        description="Build SAE training data with full-fold normalization"
    )
    parser.add_argument("--model", required=True,
                        help="Model name or 'all' for all available models")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layer-mode", choices=["fixed", "fixed_task_aware", "per_dataset"],
                        default="fixed_task_aware",
                        help="Layer selection: 'fixed' uses one global layer, "
                             "'fixed_task_aware' uses one layer per architecture variant "
                             "(e.g. TabPFN classifier L19 + regressor L11), "
                             "'per_dataset' uses CKA critical layer per dataset")
    parser.add_argument("--embeddings-dir", type=Path, default=EMBEDDINGS_DIR,
                        help="Root of {model}/{dataset}.npz all-layer embeddings "
                             "(04_extract_all_layers --output-root)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Corpus directory; defaults to the DEFAULT_SAE_ROUND one")
    parser.add_argument("--rows-from-round", type=int, default=None,
                        help="Pin train/test rows to this round's corpus of the same model "
                             "(sae_training_round{R}/{model}_taskaware_sae_{training,test}.npz) "
                             "instead of resampling by TabPFN difficulty")
    args = parser.parse_args()

    EMBEDDINGS_DIR = args.embeddings_dir
    OUTPUT_DIR = args.output_dir

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

    print("Building SAE training data")
    print(f"  Embeddings: {EMBEDDINGS_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Layer mode: {args.layer_mode}")
    print(f"  Models: {models}")

    results = []
    for model in models:
        print(f"\n{'=' * 60}")
        print(f"  Model: {model}")
        print("=" * 60)

        try:
            reference_rows = None
            if args.rows_from_round is not None:
                ref_dir = PROJECT_ROOT / "output" / f"sae_training_round{args.rows_from_round}"
                reference_rows = load_reference_rows(
                    ref_dir / f"{model}_taskaware_sae_training.npz",
                    ref_dir / f"{model}_taskaware_sae_test.npz")
                print(f"  Rows pinned to round {args.rows_from_round} ({len(reference_rows)} datasets)")
            result = build_training_data(model, device=args.device, layer_mode=args.layer_mode,
                                         reference_rows=reference_rows)
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
        print(f"{'Model':<12} {'Mode':>12} {'Train':>20} {'Test':>20} {'Datasets':>10}")
        print("-" * 76)
        for r in results:
            print(
                f"{r['model']:<12} {r['layer_mode']:>12} "
                f"{str(r['train_shape']):>20} {str(r['test_shape']):>20} "
                f"{r['n_datasets']:>10}"
            )


if __name__ == "__main__":
    main()
