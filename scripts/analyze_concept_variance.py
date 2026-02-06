#!/usr/bin/env python3
"""
Between-dataset vs within-dataset concept variance analysis.

For each SAE feature, decompose activation variance into:
- Between-dataset variance (σ²_between): do different datasets activate differently?
- Within-dataset variance (σ²_within): do rows within a dataset vary?
- ICC = σ²_between / (σ²_between + σ²_within)

ICC → 1.0 means the feature is a dataset-level descriptor (same for all rows).
ICC → 0.0 means the feature captures row-level variation.

Tests whether Matryoshka hierarchy maps to abstraction levels:
  L0 (features 0-31): global/dataset identity
  L4 (features 256-1535): row-specific patterns
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.sparse_autoencoder import SAEConfig, SparseAutoencoder


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_sae_checkpoint(path: Path) -> Tuple[SparseAutoencoder, SAEConfig, dict]:
    """Load SAE model from checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    config = SAEConfig(**checkpoint["config"])
    model = SparseAutoencoder(config)

    state_dict = checkpoint["model_state_dict"]
    if "reference_data" in state_dict and state_dict["reference_data"] is not None:
        ref_data = state_dict["reference_data"]
        model.register_buffer("reference_data", ref_data)
        if "archetype_logits" in state_dict:
            model.archetype_logits = torch.nn.Parameter(
                state_dict["archetype_logits"]
            )
        if "archetype_deviation" in state_dict:
            model.archetype_deviation = torch.nn.Parameter(
                state_dict["archetype_deviation"]
            )

    model.load_state_dict(state_dict)
    model.eval()
    return model, config, checkpoint


def get_train_test_split(datasets: List[str]) -> Tuple[List[str], List[str]]:
    """Deterministic train/test split matching sae_tabarena_sweep.py."""
    import hashlib

    train_datasets, test_datasets = [], []
    for ds in datasets:
        h = int(hashlib.md5(ds.encode()).hexdigest(), 16)
        if h % 10 < 7:
            train_datasets.append(ds)
        else:
            test_datasets.append(ds)
    return train_datasets, test_datasets


def compute_normalization_stats(
    emb_dir: Path, train_datasets: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute train_std and train_mean for two-stage normalization."""
    train_embs = []
    for ds in train_datasets:
        path = emb_dir / f"tabarena_{ds}.npz"
        if path.exists():
            data = np.load(path, allow_pickle=True)
            emb = data["embeddings"].astype(np.float32)
            if len(emb) > 200:
                np.random.seed(42)
                idx = np.random.choice(len(emb), 200, replace=False)
                emb = emb[idx]
            train_embs.append(emb)

    train_pooled = np.concatenate(train_embs)
    train_std = train_pooled.std(axis=0, keepdims=True)
    train_std[train_std < 1e-8] = 1.0
    train_norm = train_pooled / train_std
    train_mean = train_norm.mean(axis=0, keepdims=True)
    return train_std, train_mean


def compute_activations(
    model: SparseAutoencoder,
    emb_dir: Path,
    datasets: List[str],
    train_std: np.ndarray,
    train_mean: np.ndarray,
) -> List[Tuple[str, np.ndarray]]:
    """Compute SAE activations for all datasets.

    Returns list of (dataset_name, activations) where activations is (n_samples, hidden_dim).
    """
    results = []
    for ds in datasets:
        path = emb_dir / f"tabarena_{ds}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        emb = data["embeddings"].astype(np.float32)

        # Two-stage normalization: divide by std, then center
        emb_norm = emb / train_std
        emb_centered = emb_norm - train_mean

        with torch.no_grad():
            h = model.encode(torch.tensor(emb_centered, dtype=torch.float32))
            activations = h.numpy()

        results.append((ds, activations))

    return results


def compute_icc(dataset_activations: List[Tuple[str, np.ndarray]]) -> Dict:
    """Compute ICC (intraclass correlation) for each SAE feature.

    ICC = σ²_between / (σ²_between + σ²_within)

    Returns dict with per-feature stats and per-level summaries.
    """
    n_features = dataset_activations[0][1].shape[1]

    # Compute per-dataset means and variances for each feature
    dataset_means = []  # (n_datasets, n_features)
    dataset_vars = []  # (n_datasets, n_features)
    dataset_sizes = []  # (n_datasets,)

    for ds_name, acts in dataset_activations:
        n = acts.shape[0]
        dataset_sizes.append(n)
        dataset_means.append(acts.mean(axis=0))
        dataset_vars.append(acts.var(axis=0, ddof=0))

    dataset_means = np.array(dataset_means)  # (K, D)
    dataset_vars = np.array(dataset_vars)  # (K, D)
    dataset_sizes = np.array(dataset_sizes)  # (K,)
    n_total = dataset_sizes.sum()

    # Grand mean (weighted by dataset size)
    grand_mean = np.average(dataset_means, axis=0, weights=dataset_sizes)

    # Between-dataset variance: weighted variance of dataset means
    deviations = dataset_means - grand_mean[np.newaxis, :]  # (K, D)
    var_between = np.average(deviations**2, axis=0, weights=dataset_sizes)  # (D,)

    # Within-dataset variance: weighted mean of per-dataset variances
    var_within = np.average(dataset_vars, axis=0, weights=dataset_sizes)  # (D,)

    # ICC per feature
    var_total = var_between + var_within
    safe_total = np.where(var_total > 1e-12, var_total, 1.0)
    icc = np.where(var_total > 1e-12, var_between / safe_total, 0.0)

    # Firing rate: fraction of datasets where feature is active
    active_per_dataset = (dataset_means > 1e-6).astype(float)  # (K, D)
    firing_rate = active_per_dataset.mean(axis=0)  # (D,)

    # Alive features (based on max activation > 0.001)
    all_acts = np.concatenate([a for _, a in dataset_activations], axis=0)
    feature_max = all_acts.max(axis=0)
    alive_mask = feature_max > 0.001

    # Matryoshka levels
    matryoshka_dims = [32, 64, 128, 256]
    level_boundaries = [(0, 32), (32, 64), (64, 128), (128, 256), (256, n_features)]
    level_names = ["L0 (0-31)", "L1 (32-63)", "L2 (64-127)", "L3 (128-255)", "L4 (256-1535)"]

    # Per-feature results
    features = {}
    for i in range(n_features):
        level = 4  # default to last level
        for lev, (lo, hi) in enumerate(level_boundaries):
            if lo <= i < hi:
                level = lev
                break
        features[i] = {
            "icc": float(icc[i]),
            "var_between": float(var_between[i]),
            "var_within": float(var_within[i]),
            "var_total": float(var_total[i]),
            "firing_rate": float(firing_rate[i]),
            "alive": bool(alive_mask[i]),
            "matryoshka_level": level,
        }

    # Per-level summaries (alive features only)
    level_summaries = {}
    for lev, (lo, hi) in enumerate(level_boundaries):
        level_icc = icc[lo:hi]
        level_alive = alive_mask[lo:hi]
        alive_icc = level_icc[level_alive]
        n_alive = int(level_alive.sum())

        level_summaries[level_names[lev]] = {
            "n_features": hi - lo,
            "n_alive": n_alive,
            "mean_icc": float(alive_icc.mean()) if n_alive > 0 else 0.0,
            "median_icc": float(np.median(alive_icc)) if n_alive > 0 else 0.0,
            "std_icc": float(alive_icc.std()) if n_alive > 0 else 0.0,
            "min_icc": float(alive_icc.min()) if n_alive > 0 else 0.0,
            "max_icc": float(alive_icc.max()) if n_alive > 0 else 0.0,
            "pct_high_icc": float((alive_icc > 0.5).mean()) if n_alive > 0 else 0.0,
            "mean_firing_rate": float(firing_rate[lo:hi][level_alive].mean()) if n_alive > 0 else 0.0,
        }

    return {
        "features": features,
        "level_summaries": level_summaries,
        "global": {
            "n_features": n_features,
            "n_alive": int(alive_mask.sum()),
            "n_datasets": len(dataset_activations),
            "n_total_samples": int(n_total),
            "overall_mean_icc": float(icc[alive_mask].mean()) if alive_mask.any() else 0.0,
            "overall_median_icc": float(np.median(icc[alive_mask])) if alive_mask.any() else 0.0,
        },
    }


def print_summary(results: Dict) -> None:
    """Print a formatted summary table."""
    g = results["global"]
    print(f"\n{'='*70}")
    print("Between-Dataset vs Within-Dataset Concept Variance (ICC Analysis)")
    print(f"{'='*70}")
    print(f"  Datasets: {g['n_datasets']}")
    print(f"  Total samples: {g['n_total_samples']}")
    print(f"  Alive features: {g['n_alive']} / {g['n_features']}")
    print(f"  Overall mean ICC: {g['overall_mean_icc']:.3f}")
    print(f"  Overall median ICC: {g['overall_median_icc']:.3f}")

    print(f"\n{'Matryoshka Level':<20} {'Alive':>6} {'Mean ICC':>10} {'Med ICC':>10} "
          f"{'Std':>8} {'%>0.5':>8} {'Fire Rate':>10}")
    print("-" * 72)

    for level_name, s in results["level_summaries"].items():
        print(f"{level_name:<20} {s['n_alive']:>6} {s['mean_icc']:>10.3f} "
              f"{s['median_icc']:>10.3f} {s['std_icc']:>8.3f} "
              f"{s['pct_high_icc']:>7.1%} {s['mean_firing_rate']:>10.3f}")

    print()

    # Hypothesis test
    levels = list(results["level_summaries"].values())
    if len(levels) >= 2 and levels[0]["n_alive"] > 0 and levels[-1]["n_alive"] > 0:
        l0_icc = levels[0]["mean_icc"]
        l4_icc = levels[-1]["mean_icc"]
        diff = l0_icc - l4_icc
        direction = "SUPPORTS" if diff > 0.05 else ("NEUTRAL" if abs(diff) <= 0.05 else "CONTRADICTS")
        print(f"Hierarchy hypothesis (L0=dataset, L4=row):")
        print(f"  L0 mean ICC: {l0_icc:.3f}, L4 mean ICC: {l4_icc:.3f}, "
              f"diff: {diff:+.3f} → {direction}")


def main():
    checkpoint_path = (
        PROJECT_ROOT
        / "output"
        / "sae_tabarena_sweep"
        / "tabpfn"
        / "sae_matryoshka_archetypal_validated.pt"
    )
    output_path = PROJECT_ROOT / "output" / "concept_variance_decomposition.json"
    emb_dir = PROJECT_ROOT / "output" / "embeddings" / "tabarena" / "tabpfn"

    # Load SAE
    print(f"Loading SAE from {checkpoint_path}...")
    model, config, checkpoint = load_sae_checkpoint(checkpoint_path)
    print(f"  Type: {config.sparsity_type}, hidden_dim: {config.hidden_dim}")
    print(f"  Matryoshka dims: {config.matryoshka_dims}")

    # Get all datasets and compute normalization stats
    all_datasets = sorted(
        f.stem.replace("tabarena_", "") for f in emb_dir.glob("tabarena_*.npz")
    )
    train_datasets, test_datasets = get_train_test_split(all_datasets)
    print(f"\nDatasets: {len(all_datasets)} total ({len(train_datasets)} train, {len(test_datasets)} test)")

    print("Computing training normalization stats...")
    train_std, train_mean = compute_normalization_stats(emb_dir, train_datasets)

    # Compute activations for ALL datasets
    print(f"Computing SAE activations for {len(all_datasets)} datasets...")
    dataset_activations = compute_activations(
        model, emb_dir, all_datasets, train_std, train_mean
    )
    total_samples = sum(a.shape[0] for _, a in dataset_activations)
    print(f"  Total samples: {total_samples}")

    # Variance decomposition
    print("Computing ICC variance decomposition...")
    results = compute_icc(dataset_activations)

    # Print summary
    print_summary(results)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
