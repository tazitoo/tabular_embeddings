#!/usr/bin/env python3
"""
Compute latent CKA — CKA on SAE activation matrices (latents) for all model pairs.

For each dataset, encode embeddings through each model's SAE, then compute
linear CKA between activation matrices. This gives a parameter-free measure
of concept similarity that handles different hidden dims natively.

Outputs per-dataset and mean latent CKA per pair to JSON.

Usage:
    PYTHONPATH=. python scripts/paper/sec4/compute_latent_cka.py --task-filter classification
    PYTHONPATH=. python scripts/paper/sec4/compute_latent_cka.py --task-filter regression
"""

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT

from analysis.similarity import linear_cka
from scripts.analyze_sae_concepts_deep import load_sae_checkpoint
from scripts.compare_sae_cross_model import (
    SAE_FILENAME,
    find_common_datasets,
    get_dataset_tasks,
    get_models_for_task,
    sae_sweep_dir,
)


def load_embeddings(emb_path: Path, max_samples: int = 500) -> np.ndarray:
    """Load and subsample embeddings from .npz file."""
    data = np.load(emb_path, allow_pickle=True)
    emb = data['embeddings'].astype(np.float32)
    if len(emb) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(emb), max_samples, replace=False)
        emb = emb[idx]
    return emb


def encode_through_sae(model, embeddings: np.ndarray) -> np.ndarray:
    """Encode embeddings through SAE, returning activation matrix."""
    with torch.no_grad():
        x = torch.tensor(embeddings, dtype=torch.float32)
        h = model.encode(x).numpy()
    return h


def main():
    parser = argparse.ArgumentParser(description="Compute latent CKA across model pairs")
    parser.add_argument('--task-filter', choices=['classification', 'regression'],
                        default='classification')
    parser.add_argument('--max-samples', type=int, default=500,
                        help='Max samples per dataset')
    parser.add_argument('--round', type=int, default=None,
                        help='SAE sweep round')
    args = parser.parse_args()

    task = args.task_filter
    models, task_filters = get_models_for_task(task)
    dataset_tasks = get_dataset_tasks()

    base_emb = PROJECT_ROOT / 'output' / 'embeddings' / 'tabarena'
    base_sae = sae_sweep_dir(args.round)

    # Build embedding dirs and load SAE checkpoints
    emb_dirs = {}
    sae_models = {}
    for display_name, sweep_dir, emb_dir_name in models:
        emb_dir = base_emb / emb_dir_name
        sae_path = base_sae / sweep_dir / SAE_FILENAME
        if not emb_dir.exists():
            print(f"  Warning: embedding dir missing for {display_name}: {emb_dir}")
            continue
        if not sae_path.exists():
            print(f"  Warning: SAE checkpoint missing for {display_name}: {sae_path}")
            continue
        emb_dirs[display_name] = emb_dir
        model, config, _ = load_sae_checkpoint(sae_path)
        sae_models[display_name] = model
        print(f"  Loaded {display_name}: input_dim={config.input_dim}, "
              f"hidden_dim={config.hidden_dim}")

    if len(sae_models) < 2:
        print("Error: need at least 2 models with SAE checkpoints")
        return

    # Find common datasets, filter by task
    common = find_common_datasets(emb_dirs)
    task_datasets = [ds for ds in common if dataset_tasks.get(ds) == task]
    print(f"\n{task.title()} datasets: {len(task_datasets)}")

    # Filter datasets per model based on task filters
    model_names = list(sae_models.keys())
    pairs = list(combinations(model_names, 2))
    print(f"Model pairs: {len(pairs)}")

    # Compute per-dataset latent CKA for each pair
    pair_scores = defaultdict(dict)  # {pair_key: {dataset: cka}}

    for ds_name in task_datasets:
        # Check task filter for each model
        activations = {}
        for name in model_names:
            model_task_filter = task_filters.get(name)
            if model_task_filter and dataset_tasks.get(ds_name) != model_task_filter:
                continue

            emb_path = emb_dirs[name] / f"tabarena_{ds_name}.npz"
            if not emb_path.exists():
                continue

            emb = load_embeddings(emb_path, args.max_samples)
            activations[name] = encode_through_sae(sae_models[name], emb)

        # Compute CKA for each pair that has data
        for a, b in pairs:
            if a not in activations or b not in activations:
                continue
            # Subsample to common size (should already match via seed=42)
            n = min(len(activations[a]), len(activations[b]))
            cka = linear_cka(activations[a][:n], activations[b][:n])
            pair_key = f"{a}--{b}"
            pair_scores[pair_key][ds_name] = float(cka)

        print(f"  {ds_name}: {len(activations)} models encoded")

    # Compute means
    results = {}
    for pair_key, ds_scores in pair_scores.items():
        values = list(ds_scores.values())
        results[pair_key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'n_datasets': len(values),
            'per_dataset': ds_scores,
        }

    # Save
    out_dir = PROJECT_ROOT / 'output' / 'paper_data' / 'sec4'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'latent_cka_{task}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Summary
    print(f"\n{'Pair':<30s} {'Mean CKA':>8s} {'Std':>6s} {'N':>3s}")
    print('-' * 50)
    for pair_key in sorted(results, key=lambda k: -results[k]['mean']):
        r = results[pair_key]
        print(f"{pair_key:<30s} {r['mean']:8.3f} {r['std']:6.3f} {r['n_datasets']:3d}")


if __name__ == '__main__':
    main()
