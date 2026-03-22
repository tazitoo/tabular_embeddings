#!/usr/bin/env python3
"""
Cross-model SAE feature matching via dataset-mean activations.

For each Matryoshka band, computes Pearson correlation between dataset-mean
activation vectors across all models. Features from different models that
correlate above threshold form "consensus concepts." This is much cleaner
than row-level Jaccard (which yields 0.03-0.08) because dataset-means
remove row-level noise.

Usage:
    python scripts/match_cross_model_features.py \
        --output output/cross_model_feature_matching.json \
        --corr-threshold 0.5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from scripts._project_root import PROJECT_ROOT

from analysis.sparse_autoencoder import SparseAutoencoder, SAEConfig
from scripts.sae.compare_sae_cross_model import (
    DEFAULT_MODELS,
    DEFAULT_SAE_ROUND,
    SAE_FILENAME,
    find_common_datasets,
    sae_sweep_dir,
)
from scripts.sae.analyze_sae_concepts_deep import (
    NumpyEncoder,
    convert_keys_to_native,
    load_sae_checkpoint,
)

EMB_BASE = PROJECT_ROOT / "output" / "embeddings" / "tabarena"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def pool_embeddings_with_offsets(
    emb_dir: Path,
    datasets: List[str],
    max_per_dataset: int = 500,
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """Pool embeddings and track per-dataset row offsets.

    Returns:
        pooled: (n_total, dim) concatenated embeddings
        offsets: {dataset_name: (start_row, end_row)}
    """
    all_embs = []
    offsets = {}
    cursor = 0
    for ds in datasets:
        path = emb_dir / f"tabarena_{ds}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        emb = data["embeddings"].astype(np.float32)
        if len(emb) > max_per_dataset:
            np.random.seed(42)
            idx = np.random.choice(len(emb), max_per_dataset, replace=False)
            emb = emb[idx]
        offsets[ds] = (cursor, cursor + len(emb))
        cursor += len(emb)
        all_embs.append(emb)
    return np.concatenate(all_embs), offsets


def compute_activations(
    model: SparseAutoencoder,
    embeddings: np.ndarray,
) -> np.ndarray:
    """Compute SAE activations from raw embeddings.

    The SAE's internal BatchNorm handles normalization automatically.
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(embeddings, dtype=torch.float32)
        h = model.encode(x).numpy()
    return h


def compute_dataset_mean_activations(
    activations: np.ndarray,
    offsets: Dict[str, Tuple[int, int]],
    datasets: List[str],
) -> np.ndarray:
    """Average SAE activations per dataset -> (n_datasets, hidden_dim)."""
    means = []
    for ds in datasets:
        if ds not in offsets:
            means.append(np.zeros(activations.shape[1]))
            continue
        start, end = offsets[ds]
        means.append(activations[start:end].mean(axis=0))
    return np.array(means)


def get_matryoshka_bands(config: SAEConfig) -> List[Tuple[str, int, int]]:
    """Extract Matryoshka band boundaries from SAE config.

    Returns list of (label, start, end) tuples.
    """
    mat_dims = config.matryoshka_dims
    if mat_dims is None:
        mat_dims = [config.hidden_dim]
    if mat_dims[-1] < config.hidden_dim:
        mat_dims = list(mat_dims) + [config.hidden_dim]

    boundaries = [0] + list(mat_dims)
    bands = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        label = f"S{i + 1}"
        bands.append((label, start, end))
    return bands


# ---------------------------------------------------------------------------
# Core matching functions
# ---------------------------------------------------------------------------

def get_alive_features(
    ds_means: np.ndarray,
    band_start: int,
    band_end: int,
    threshold: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract alive feature vectors from a Matryoshka band.

    Args:
        ds_means: (n_datasets, hidden_dim) dataset-mean activations
        band_start: start index of band (inclusive)
        band_end: end index of band (exclusive)
        threshold: minimum max activation to be considered alive

    Returns:
        alive_vectors: (n_alive, n_datasets) — each row is a feature's
            activation profile across datasets
        global_indices: (n_alive,) — global indices into hidden_dim
    """
    band = ds_means[:, band_start:band_end]  # (n_datasets, band_size)
    max_acts = band.max(axis=0)  # (band_size,)
    alive_mask = max_acts > threshold
    alive_local = np.where(alive_mask)[0]
    global_indices = alive_local + band_start
    alive_vectors = band[:, alive_mask].T  # (n_alive, n_datasets)
    return alive_vectors, global_indices


def compute_cross_model_correlation(
    feature_vectors: np.ndarray,
    feature_labels: List[Tuple[str, int]],
) -> np.ndarray:
    """Compute Pearson |r| correlation matrix across all features.

    Args:
        feature_vectors: (N_total, n_datasets) stacked across models
        feature_labels: [(model_name, global_index), ...] for each row

    Returns:
        corr_matrix: (N_total, N_total) absolute Pearson correlations.
            Constant features get r=0 (no NaN).
    """
    n = feature_vectors.shape[0]
    if n == 0:
        return np.zeros((0, 0))

    # Z-score each feature vector
    means = feature_vectors.mean(axis=1, keepdims=True)
    stds = feature_vectors.std(axis=1, keepdims=True, ddof=0)

    # Avoid division by zero for constant features
    stds[stds == 0] = 1.0
    z = (feature_vectors - means) / stds

    # Pearson r = Z @ Z.T / n_datasets
    n_datasets = feature_vectors.shape[1]
    corr = z @ z.T / n_datasets

    # Constant features should have r=0, not NaN
    const_mask = (feature_vectors.std(axis=1) == 0)
    corr[const_mask, :] = 0.0
    corr[:, const_mask] = 0.0

    return np.abs(corr)


def cluster_consensus_concepts(
    corr_matrix: np.ndarray,
    feature_labels: List[Tuple[str, int]],
    corr_threshold: float = 0.5,
) -> Tuple[List[dict], Dict[str, List[int]]]:
    """Cluster features into consensus concepts and model-specific groups.

    Uses agglomerative clustering with average linkage on distance = 1 - |r|.

    Args:
        corr_matrix: (N, N) absolute Pearson correlation matrix
        feature_labels: [(model_name, global_index), ...] for each feature
        corr_threshold: minimum |r| for features to be in the same cluster

    Returns:
        consensus: list of concept dicts with keys:
            id, members, n_models, mean_inter_model_r, centroid
        model_specific: {model_name: [global_indices]} for single-model clusters
    """
    n = corr_matrix.shape[0]
    if n == 0:
        return [], {}

    if n == 1:
        model, idx = feature_labels[0]
        return [], {model: [idx]}

    # Distance = 1 - |r|, clipped to [0, 1]
    dist_matrix = np.clip(1.0 - corr_matrix, 0, 1)
    np.fill_diagonal(dist_matrix, 0)

    # Condensed distance for scipy
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=1 - corr_threshold, criterion="distance")

    # Group features by cluster
    clusters: Dict[int, List[int]] = {}
    for i, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(i)

    consensus = []
    model_specific: Dict[str, List[int]] = {}

    for cluster_id, members in sorted(clusters.items()):
        models_in_cluster = set(feature_labels[i][0] for i in members)

        if len(models_in_cluster) >= 2:
            # Consensus concept: features from >= 2 models
            member_dict: Dict[str, List[int]] = {}
            for i in members:
                model, idx = feature_labels[i]
                member_dict.setdefault(model, []).append(int(idx))

            # Mean inter-model correlation
            inter_model_rs = []
            for i_idx, i in enumerate(members):
                for j in members[i_idx + 1:]:
                    if feature_labels[i][0] != feature_labels[j][0]:
                        inter_model_rs.append(float(corr_matrix[i, j]))
            mean_r = float(np.mean(inter_model_rs)) if inter_model_rs else 0.0

            consensus.append({
                "members": member_dict,
                "n_models": len(models_in_cluster),
                "mean_inter_model_r": round(mean_r, 4),
            })
        else:
            # Model-specific: single model cluster
            model = feature_labels[members[0]][0]
            for i in members:
                _, idx = feature_labels[i]
                model_specific.setdefault(model, []).append(int(idx))

    return consensus, model_specific


def compute_matching_summary(
    consensus: List[dict],
    model_specific: Dict[str, List[int]],
    model_names: List[str],
    total_alive_per_model: Dict[str, int],
) -> dict:
    """Compute summary statistics for the matching results.

    Args:
        consensus: list of consensus concept dicts
        model_specific: {model: [indices]} for single-model features
        model_names: all model names
        total_alive_per_model: {model: n_alive} for coverage computation

    Returns:
        Summary dict with by_n_models histogram and per_model coverage.
    """
    # Count features per model in consensus
    consensus_per_model: Dict[str, int] = {m: 0 for m in model_names}
    by_n_models: Dict[int, int] = {}

    for concept in consensus:
        n = concept["n_models"]
        by_n_models[n] = by_n_models.get(n, 0) + 1
        for model, indices in concept["members"].items():
            consensus_per_model[model] += len(indices)

    specific_per_model = {
        m: len(model_specific.get(m, [])) for m in model_names
    }

    per_model_coverage = {}
    for m in model_names:
        total = total_alive_per_model.get(m, 0)
        in_c = consensus_per_model[m]
        in_s = specific_per_model[m]
        frac = in_c / total if total > 0 else 0.0
        per_model_coverage[m] = {
            "in_consensus": in_c,
            "model_specific": in_s,
            "total_alive": total,
            "frac_consensus": round(frac, 4),
        }

    return {
        "total_consensus": len(consensus),
        "total_model_specific": sum(
            len(v) for v in model_specific.values()
        ),
        "by_n_models": {str(k): v for k, v in sorted(by_n_models.items())},
        "per_model_coverage": per_model_coverage,
    }


def match_all_bands(
    model_ds_means: Dict[str, np.ndarray],
    model_configs: Dict[str, SAEConfig],
    model_names: List[str],
    datasets: List[str],
    corr_threshold: float = 0.5,
    alive_threshold: float = 0.001,
) -> dict:
    """Orchestrate per-band feature matching across all models.

    Bands matched by ordinal (S1<->S1), not by index range.

    Args:
        model_ds_means: {model_name: (n_datasets, hidden_dim)} dataset-mean activations
        model_configs: {model_name: SAEConfig}
        model_names: ordered model names
        datasets: list of dataset names
        corr_threshold: Pearson |r| threshold for consensus
        alive_threshold: minimum max activation to consider feature alive

    Returns:
        dict with 'bands' and 'summary' keys
    """
    # Get bands per model
    model_bands = {m: get_matryoshka_bands(model_configs[m]) for m in model_names}

    # Find max number of bands across models
    n_bands = max(len(bands) for bands in model_bands.values())

    bands_result = {}
    global_summary = {
        "total_consensus": 0,
        "total_model_specific": 0,
        "by_n_models": {},
        "per_model_coverage": {m: {"in_consensus": 0, "model_specific": 0,
                                    "total_alive": 0} for m in model_names},
    }

    for band_idx in range(n_bands):
        band_label = f"S{band_idx + 1}"
        all_vectors = []
        all_labels = []
        total_alive = {m: 0 for m in model_names}

        for m in model_names:
            bands = model_bands[m]
            if band_idx >= len(bands):
                continue
            _, start, end = bands[band_idx]
            alive_vecs, alive_idxs = get_alive_features(
                model_ds_means[m], start, end, threshold=alive_threshold
            )
            total_alive[m] = len(alive_idxs)
            for i, gidx in enumerate(alive_idxs):
                all_vectors.append(alive_vecs[i])
                all_labels.append((m, int(gidx)))

        if not all_vectors:
            bands_result[band_label] = {
                "n_consensus": 0,
                "n_model_specific": 0,
                "concepts": [],
                "model_specific": {},
            }
            continue

        feature_matrix = np.stack(all_vectors, axis=0)
        corr = compute_cross_model_correlation(feature_matrix, all_labels)
        consensus, model_spec = cluster_consensus_concepts(
            corr, all_labels, corr_threshold=corr_threshold
        )

        # Assign IDs
        for i, concept in enumerate(consensus):
            concept["id"] = f"{band_label}_C{i + 1:03d}"

        # Compute centroids from dataset-mean vectors
        for concept in consensus:
            member_vecs = []
            for m, indices in concept["members"].items():
                ds_means = model_ds_means[m]
                for idx in indices:
                    member_vecs.append(ds_means[:, idx])
            centroid = np.mean(member_vecs, axis=0)
            concept["centroid"] = centroid.tolist()

        summary = compute_matching_summary(
            consensus, model_spec, model_names, total_alive
        )

        bands_result[band_label] = {
            "n_consensus": len(consensus),
            "n_model_specific": summary["total_model_specific"],
            "concepts": consensus,
            "model_specific": model_spec,
        }

        # Accumulate global summary
        global_summary["total_consensus"] += len(consensus)
        global_summary["total_model_specific"] += summary["total_model_specific"]
        for k, v in summary["by_n_models"].items():
            global_summary["by_n_models"][k] = (
                global_summary["by_n_models"].get(k, 0) + v
            )
        for m in model_names:
            mc = summary["per_model_coverage"].get(m, {})
            global_summary["per_model_coverage"][m]["in_consensus"] += mc.get("in_consensus", 0)
            global_summary["per_model_coverage"][m]["model_specific"] += mc.get("model_specific", 0)
            global_summary["per_model_coverage"][m]["total_alive"] += mc.get("total_alive", 0)

    # Compute global coverage fractions
    for m in model_names:
        mc = global_summary["per_model_coverage"][m]
        total = mc["total_alive"]
        mc["frac_consensus"] = round(mc["in_consensus"] / total, 4) if total > 0 else 0.0

    return {"bands": bands_result, "summary": global_summary}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-model SAE feature matching via dataset-mean activations"
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "output" / "cross_model_feature_matching.json",
    )
    parser.add_argument("--corr-threshold", type=float, default=0.5)
    parser.add_argument("--alive-threshold", type=float, default=0.001)
    parser.add_argument("--max-per-dataset", type=int, default=500)
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Subset of model names to include (default: all)",
    )
    parser.add_argument("--round", type=int, default=None)
    args = parser.parse_args()

    sweep_base = sae_sweep_dir(args.round)
    print(f"SAE sweep dir: {sweep_base}")

    # Filter models
    models = DEFAULT_MODELS
    if args.models:
        models = [m for m in models if m[0] in args.models or m[1] in args.models]
    print(f"Models: {[m[0] for m in models]}")

    # Find common datasets
    emb_dirs = {}
    for display, sae_dir, emb_dir in models:
        emb_dirs[display] = EMB_BASE / emb_dir
    datasets = find_common_datasets(emb_dirs)
    model_names = [m[0] for m in models]

    # Load SAE checkpoints and compute dataset-mean activations
    model_ds_means = {}
    model_configs = {}

    for display, sae_dir, emb_dir in models:
        print(f"\n--- {display} ---")
        ckpt_path = sweep_base / sae_dir / SAE_FILENAME
        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {ckpt_path}")
            continue

        model, config, _ = load_sae_checkpoint(ckpt_path)
        model_configs[display] = config
        print(f"  SAE: input={config.input_dim}, hidden={config.hidden_dim}")

        emb_path = EMB_BASE / emb_dir
        pooled, offsets = pool_embeddings_with_offsets(
            emb_path, datasets, max_per_dataset=args.max_per_dataset
        )
        print(f"  Pooled: {pooled.shape[0]} rows, {pooled.shape[1]} dims")

        acts = compute_activations(model, pooled)
        ds_means = compute_dataset_mean_activations(acts, offsets, datasets)
        model_ds_means[display] = ds_means
        print(f"  Dataset means: {ds_means.shape}")

    # Filter to models that loaded successfully
    model_names = [m for m in model_names if m in model_ds_means]
    print(f"\nLoaded {len(model_names)} models: {model_names}")

    # Run matching
    print(f"\nMatching features (corr_threshold={args.corr_threshold})...")
    result = match_all_bands(
        model_ds_means, model_configs, model_names, datasets,
        corr_threshold=args.corr_threshold,
        alive_threshold=args.alive_threshold,
    )

    # Build output
    output = {
        "metadata": {
            "n_models": len(model_names),
            "n_datasets": len(datasets),
            "corr_threshold": args.corr_threshold,
            "alive_threshold": args.alive_threshold,
            "max_per_dataset": args.max_per_dataset,
            "sae_round": args.round or DEFAULT_SAE_ROUND,
            "models": model_names,
            "datasets": datasets,
        },
        "bands": result["bands"],
        "summary": result["summary"],
    }

    # Print summary
    s = output["summary"]
    print(f"\n=== Summary ===")
    print(f"Total consensus concepts: {s['total_consensus']}")
    print(f"Total model-specific features: {s['total_model_specific']}")
    print(f"By n_models: {s['by_n_models']}")
    for m in model_names:
        mc = s["per_model_coverage"][m]
        print(f"  {m}: {mc['in_consensus']}/{mc['total_alive']} in consensus "
              f"({mc['frac_consensus']:.1%})")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(convert_keys_to_native(output), f, cls=NumpyEncoder, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
