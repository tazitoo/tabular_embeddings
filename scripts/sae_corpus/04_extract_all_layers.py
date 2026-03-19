#!/usr/bin/env python3
"""Extract all-layer embeddings for holdout rows using the preprocessed cache.

For each (model, dataset), loads preprocessed data from cache, samples up to
--max-context train rows as ICL context, and extracts embeddings from ALL
layers for the holdout (test) rows.

Output:
    output/sae_training_round9/embeddings/{model}/{dataset}.npz
        layer_0, layer_1, ...:  (n_query, hidden_dim) float32
        layer_names:            sorted list of layer names
        row_indices:            (n_query,) int32 — positions in full dataset
        n_context:              int
        task_type:              str

Usage:
    python scripts/sae_corpus/04_extract_all_layers.py --model tabpfn --device cuda
    python scripts/sae_corpus/04_extract_all_layers.py --model mitra --datasets diabetes anneal
    python scripts/sae_corpus/04_extract_all_layers.py --model tabpfn --max-context 512
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.extended_loader import _load_tabarena_cached_v2
from data.preprocessing import CACHE_DIR, load_preprocessed
from models.layer_extraction import extract_all_layers, load_and_fit, sort_layer_names
from scripts._project_root import PROJECT_ROOT

SPLITS_PATH = PROJECT_ROOT / "output" / "sae_training_round9" / "tabarena_splits.json"
OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_training_round9" / "embeddings"

SUPPORTED_MODELS = ["tabpfn", "tabicl", "tabicl_v2", "tabdpt", "mitra", "hyperfast", "tabula8b", "carte"]

# Models that need raw DataFrames (not preprocessed numpy cache)
DATAFRAME_MODELS = {"tabula8b", "carte"}


def sample_context(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_context: int,
    task_type: str,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample context rows if train set exceeds max_context.

    Uses stratified sampling for classification to preserve class ratios.
    """
    if len(X_train) <= max_context:
        return X_train, y_train

    rng = np.random.RandomState(seed)

    if task_type == "classification":
        # Stratified sample: proportional representation of each class
        classes, counts = np.unique(y_train, return_counts=True)
        indices = []
        for cls, count in zip(classes, counts):
            cls_idx = np.where(y_train == cls)[0]
            n_take = max(1, int(max_context * count / len(y_train)))
            indices.append(rng.choice(cls_idx, size=min(n_take, len(cls_idx)), replace=False))
        indices = np.concatenate(indices)
        # Trim or pad to exactly max_context
        if len(indices) > max_context:
            indices = rng.choice(indices, size=max_context, replace=False)
        elif len(indices) < max_context:
            remaining = np.setdiff1d(np.arange(len(y_train)), indices)
            extra = rng.choice(remaining, size=max_context - len(indices), replace=False)
            indices = np.concatenate([indices, extra])
    else:
        indices = rng.choice(len(X_train), size=max_context, replace=False)

    return X_train[indices], y_train[indices]


def main():
    parser = argparse.ArgumentParser(description="Extract all-layer embeddings")
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-context", type=int, default=1024,
                        help="Max context rows for ICL (default: 1024)")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Max query rows per forward pass (default: 1024)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets (default: all in splits)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing embeddings")
    args = parser.parse_args()

    splits = json.loads(SPLITS_PATH.read_text())
    dataset_names = args.datasets or sorted(splits.keys())

    # Filter datasets by task compatibility
    model_key = args.model.lower()
    if model_key == "hyperfast":
        dataset_names = [d for d in dataset_names if splits[d]["task_type"] == "classification"]
        print(f"HyperFast: filtered to {len(dataset_names)} classification datasets")
    elif model_key == "tabicl":
        dataset_names = [d for d in dataset_names if splits[d]["task_type"] == "classification"]
        print(f"TabICL v1: filtered to {len(dataset_names)} classification datasets")

    out_dir = OUTPUT_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting all layers: {args.model}")
    print(f"  Output:      {out_dir}")
    print(f"  Max context: {args.max_context}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Datasets:    {len(dataset_names)}")
    print(f"  Device:      {args.device}")
    print()

    success = skipped = errors = 0

    for i, ds_name in enumerate(dataset_names):
        out_path = out_dir / f"{ds_name}.npz"
        if out_path.exists() and not args.force:
            print(f"[{i+1}/{len(dataset_names)}] {ds_name}: exists, skipping")
            skipped += 1
            continue

        t0 = time.time()
        try:
            split_info = splits[ds_name]
            task_type = split_info["task_type"]
            train_idx = np.array(split_info["train_indices"])
            test_indices = np.array(split_info["test_indices"], dtype=np.int32)

            if model_key in DATAFRAME_MODELS:
                # DataFrame models: load raw data, model handles serialization
                cached = _load_tabarena_cached_v2(ds_name)
                if cached is None:
                    print(f"[{i+1}/{len(dataset_names)}] {ds_name}: SKIP — no raw cache")
                    skipped += 1
                    continue
                X_df, y = cached
                X_ctx_df = X_df.iloc[train_idx].reset_index(drop=True)
                X_test_df = X_df.iloc[test_indices].reset_index(drop=True)
                y_ctx = y[train_idx]

                # Subsample context rows
                if len(X_ctx_df) > args.max_context:
                    rng = np.random.RandomState(42)
                    idx = rng.choice(len(X_ctx_df), args.max_context, replace=False)
                    X_ctx_df = X_ctx_df.iloc[idx].reset_index(drop=True)
                    y_ctx = y_ctx[idx]

                fit_kwargs = {}
                if model_key == "tabula8b":
                    fit_kwargs["target_name"] = split_info.get("target", "target")

                clf = load_and_fit(
                    args.model, X_ctx_df, y_ctx,
                    task=task_type, device=args.device, **fit_kwargs,
                )
                layer_embs = extract_all_layers(
                    args.model, clf, X_test_df, task=task_type
                )
                n_ctx_used = len(X_ctx_df)
                n_train_total = len(train_idx)
            else:
                # Standard path: load from preprocessing cache
                data = load_preprocessed(args.model, ds_name, CACHE_DIR)

                X_ctx, y_ctx = sample_context(
                    data.X_train, data.y_train, args.max_context, task_type
                )

                fit_kwargs = {}
                if model_key == "tabpfn" and data.cat_indices:
                    fit_kwargs["cat_indices"] = data.cat_indices

                clf = load_and_fit(
                    args.model, X_ctx, y_ctx,
                    task=task_type, device=args.device, **fit_kwargs
                )
                layer_embs = extract_all_layers(
                    args.model, clf, data.X_test,
                    task=task_type, batch_size=args.batch_size
                )
                n_ctx_used = len(X_ctx)
                n_train_total = len(data.X_train)

            # Save
            layer_names = sort_layer_names(list(layer_embs.keys()))
            save_dict = {
                "layer_names": np.array(layer_names, dtype=str),
                "row_indices": test_indices,
                "n_context": np.array(n_ctx_used),
                "task_type": np.array(task_type),
            }
            for name, emb in layer_embs.items():
                save_dict[name] = emb.astype(np.float32)

            np.savez_compressed(str(out_path), **save_dict)

            dt = time.time() - t0
            n_total = n_train_total + len(test_indices)
            sample_layer = layer_names[0]
            dim = layer_embs[sample_layer].shape[1] if sample_layer in layer_embs else "?"
            print(f"[{i+1}/{len(dataset_names)}] {ds_name}: "
                  f"{len(layer_names)} layers, "
                  f"holdout={len(test_indices)}/{n_total} rows, "
                  f"dim={dim}, ctx={n_ctx_used}/{n_train_total} ({dt:.1f}s)")
            success += 1

        except Exception as e:
            dt = time.time() - t0
            print(f"[{i+1}/{len(dataset_names)}] {ds_name}: FAILED ({dt:.1f}s) — {e}")
            import traceback
            traceback.print_exc()
            errors += 1
        finally:
            # Free GPU memory between datasets to prevent OOM cascade
            import torch
            torch.cuda.empty_cache()

    print(f"\nDone: {success} extracted, {skipped} skipped, {errors} failed")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
