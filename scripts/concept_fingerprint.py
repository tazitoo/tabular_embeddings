#!/usr/bin/env python3
"""Dataset concept fingerprints via SAE activation profiles.

For each TabArena dataset, encodes its embeddings through the trained SAE and
computes mean activation per feature. Comparing per-dataset means against the
global mean reveals which concepts are distinctive for each dataset.

The global mean is near-zero (SAE trained on centered pooled data), so
per-dataset mean activations are already deviations from the global baseline.

Usage:
    # Compute fingerprints for a model (CPU-only, uses pre-extracted embeddings)
    python scripts/concept_fingerprint.py --model tabdpt

    # Show top distinctive concepts for a specific dataset
    python scripts/concept_fingerprint.py --model tabdpt --dataset credit-g --top 20

    # Compare two datasets
    python scripts/concept_fingerprint.py --model tabdpt --compare credit-g diabetes
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.intervene_sae import load_sae, load_training_mean, get_extraction_layer
from scripts.concept_importance import (
    get_alive_features,
    get_feature_labels,
    get_matryoshka_bands,
    feature_to_band,
    MODEL_KEY_TO_LABEL_KEY,
    DEFAULT_CONCEPT_LABELS,
)

logger = logging.getLogger(__name__)

DEFAULT_TRAINING_DIR = PROJECT_ROOT / "output" / "sae_training_round5"


def load_per_dataset_embeddings(
    model_key: str,
    training_dir: Path = DEFAULT_TRAINING_DIR,
) -> Dict[str, np.ndarray]:
    """Load pre-extracted embeddings split by source dataset.

    The SAE training .npz files contain pooled embeddings with metadata
    tracking which rows belong to which dataset.

    Returns:
        Dict mapping dataset_name -> (n_samples, emb_dim) array
    """
    layer = get_extraction_layer(model_key)
    training_path = training_dir / f"{model_key}_layer{layer}_sae_training.npz"
    if not training_path.exists():
        raise FileNotFoundError(f"Training data not found: {training_path}")

    data = np.load(training_path, allow_pickle=True)
    embeddings = data["embeddings"]
    samples_per_dataset = data["samples_per_dataset"]

    result = {}
    offset = 0
    for ds_name, count in samples_per_dataset:
        ds_name = str(ds_name)
        count = int(count)
        result[ds_name] = embeddings[offset : offset + count]
        offset += count

    assert offset == len(embeddings), f"Mismatch: {offset} vs {len(embeddings)}"
    return result


def compute_fingerprints(
    model_key: str,
    device: str = "cpu",
    training_dir: Path = DEFAULT_TRAINING_DIR,
) -> Dict:
    """Compute SAE activation fingerprint for each dataset.

    Returns dict with:
        global_mean: (hidden_dim,) mean activation across all datasets
        dataset_means: {dataset_name: (hidden_dim,)} per-dataset mean activations
        dataset_deviations: {dataset_name: (hidden_dim,)} deviation from global
        alive_features: list of alive feature indices
        feature_labels: {feat_idx: label}
        bands: Matryoshka band boundaries
    """
    sae, config = load_sae(model_key, device=device)
    data_mean = load_training_mean(model_key, device=device)
    per_dataset = load_per_dataset_embeddings(model_key, training_dir)
    alive = get_alive_features(model_key)
    labels = get_feature_labels(model_key)
    bands = get_matryoshka_bands(model_key)

    hidden_dim = config.hidden_dim

    # Encode each dataset through SAE
    dataset_means = {}
    all_activations = []

    with torch.no_grad():
        for ds_name, emb in sorted(per_dataset.items()):
            x = torch.tensor(emb, dtype=torch.float32, device=device)
            x_centered = x - data_mean
            h = sae.encode(x_centered)  # (n_samples, hidden_dim)
            mean_act = h.mean(dim=0).cpu().numpy()  # (hidden_dim,)
            dataset_means[ds_name] = mean_act
            all_activations.append(h.cpu().numpy())

    # Global mean across all datasets (should be near-zero for centered data)
    all_acts = np.concatenate(all_activations, axis=0)
    global_mean = all_acts.mean(axis=0)

    # Deviations from global mean
    dataset_deviations = {
        ds: mean - global_mean for ds, mean in dataset_means.items()
    }

    return {
        "global_mean": global_mean,
        "dataset_means": dataset_means,
        "dataset_deviations": dataset_deviations,
        "alive_features": alive,
        "feature_labels": labels,
        "bands": bands,
        "hidden_dim": hidden_dim,
        "n_datasets": len(per_dataset),
        "model_key": model_key,
    }


def dataset_top_concepts(
    fingerprints: Dict,
    dataset_name: str,
    top_n: int = 20,
) -> List[Dict]:
    """Get top distinctive concepts for a dataset (by deviation from global mean).

    Returns list of dicts with feature info, sorted by absolute deviation.
    """
    if dataset_name not in fingerprints["dataset_deviations"]:
        available = sorted(fingerprints["dataset_deviations"].keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available: {available}"
        )

    deviation = fingerprints["dataset_deviations"][dataset_name]
    mean_act = fingerprints["dataset_means"][dataset_name]
    alive = fingerprints["alive_features"]
    labels = fingerprints["feature_labels"]
    bands = fingerprints["bands"]

    # Rank alive features by absolute deviation
    alive_devs = [(idx, deviation[idx], mean_act[idx]) for idx in alive]
    alive_devs.sort(key=lambda x: -abs(x[1]))

    results = []
    for idx, dev, act in alive_devs[:top_n]:
        results.append({
            "index": idx,
            "deviation": float(dev),
            "mean_activation": float(act),
            "global_mean": float(fingerprints["global_mean"][idx]),
            "band": feature_to_band(idx, bands),
            "label": labels.get(idx, "unknown"),
        })
    return results


def compare_datasets(
    fingerprints: Dict,
    dataset_a: str,
    dataset_b: str,
    top_n: int = 20,
) -> Dict:
    """Compare concept fingerprints between two datasets.

    Returns features where the two datasets differ most, plus shared concepts.
    """
    dev_a = fingerprints["dataset_deviations"][dataset_a]
    dev_b = fingerprints["dataset_deviations"][dataset_b]
    mean_a = fingerprints["dataset_means"][dataset_a]
    mean_b = fingerprints["dataset_means"][dataset_b]
    alive = fingerprints["alive_features"]
    labels = fingerprints["feature_labels"]
    bands = fingerprints["bands"]

    # Difference between datasets
    diff = dev_a - dev_b  # positive = more active in A

    alive_diffs = [(idx, diff[idx], mean_a[idx], mean_b[idx]) for idx in alive]
    alive_diffs.sort(key=lambda x: -abs(x[1]))

    # Cosine similarity between fingerprints (alive features only)
    a_vec = np.array([dev_a[i] for i in alive])
    b_vec = np.array([dev_b[i] for i in alive])
    cos_sim = np.dot(a_vec, b_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(b_vec) + 1e-12)

    # Top features where A > B
    a_dominant = []
    b_dominant = []
    for idx, d, ma, mb in alive_diffs:
        entry = {
            "index": idx,
            "diff": float(d),
            "mean_a": float(ma),
            "mean_b": float(mb),
            "band": feature_to_band(idx, bands),
            "label": labels.get(idx, "unknown"),
        }
        if d > 0 and len(a_dominant) < top_n:
            a_dominant.append(entry)
        elif d < 0 and len(b_dominant) < top_n:
            b_dominant.append(entry)

    return {
        "dataset_a": dataset_a,
        "dataset_b": dataset_b,
        "cosine_similarity": float(cos_sim),
        "a_dominant": a_dominant,
        "b_dominant": b_dominant,
    }


def dataset_similarity_matrix(fingerprints: Dict) -> Tuple[np.ndarray, List[str]]:
    """Compute pairwise cosine similarity between all dataset fingerprints.

    Uses alive features only.
    """
    alive = fingerprints["alive_features"]
    datasets = sorted(fingerprints["dataset_deviations"].keys())
    n = len(datasets)

    # Build matrix of deviation vectors (alive features only)
    vecs = np.zeros((n, len(alive)))
    for i, ds in enumerate(datasets):
        dev = fingerprints["dataset_deviations"][ds]
        vecs[i] = [dev[idx] for idx in alive]

    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    vecs_normed = vecs / norms

    sim_matrix = vecs_normed @ vecs_normed.T
    return sim_matrix, datasets


# ── CLI ──────────────────────────────────────────────────────────────────────


def print_dataset_fingerprint(fingerprints: Dict, dataset_name: str, top_n: int):
    """Pretty-print a dataset's concept fingerprint."""
    top = dataset_top_concepts(fingerprints, dataset_name, top_n)
    dev = fingerprints["dataset_deviations"][dataset_name]
    alive = fingerprints["alive_features"]
    bands = fingerprints["bands"]

    # Summary stats
    alive_devs = [dev[i] for i in alive]
    n_above = sum(1 for d in alive_devs if d > 0.01)
    n_below = sum(1 for d in alive_devs if d < -0.01)

    print(f"\n{'='*100}")
    print(f"Concept Fingerprint: {dataset_name} ({fingerprints['model_key']})")
    print(f"{'='*100}")
    print(f"Matryoshka bands: {bands}")
    print(f"Alive features: {len(alive)}")
    print(f"Features above global mean (>0.01): {n_above}")
    print(f"Features below global mean (<-0.01): {n_below}")

    # Band distribution of distinctive features
    band_dist = Counter(
        feature_to_band(i, bands) for i in alive if abs(dev[i]) > 0.01
    )
    print(f"Band distribution of distinctive features: {dict(band_dist)}")

    print(f"\nTop {top_n} most distinctive concepts (by |deviation| from global mean):")
    print(f"{'Rank':>4} {'Feat':>6} {'Dev':>8} {'Act':>8} {'Glb':>8} {'Band':>4} {'Label'}")
    print("-" * 90)
    for rank, f in enumerate(top):
        direction = "+" if f["deviation"] > 0 else "-"
        print(
            f"{rank+1:>4} {f['index']:>6} {f['deviation']:>+8.3f} "
            f"{f['mean_activation']:>8.3f} {f['global_mean']:>8.3f} "
            f"{f['band']:>4} {f['label']}"
        )


def print_comparison(comparison: Dict, top_n: int):
    """Pretty-print a dataset comparison."""
    a, b = comparison["dataset_a"], comparison["dataset_b"]
    print(f"\n{'='*100}")
    print(f"Concept Comparison: {a} vs {b}")
    print(f"{'='*100}")
    print(f"Cosine similarity: {comparison['cosine_similarity']:.3f}")

    print(f"\nTop concepts more active in {a}:")
    print(f"{'Rank':>4} {'Feat':>6} {'Diff':>8} {'Act_A':>8} {'Act_B':>8} {'Band':>4} {'Label'}")
    print("-" * 90)
    for rank, f in enumerate(comparison["a_dominant"][:top_n]):
        print(
            f"{rank+1:>4} {f['index']:>6} {f['diff']:>+8.3f} "
            f"{f['mean_a']:>8.3f} {f['mean_b']:>8.3f} "
            f"{f['band']:>4} {f['label']}"
        )

    print(f"\nTop concepts more active in {b}:")
    print(f"{'Rank':>4} {'Feat':>6} {'Diff':>8} {'Act_A':>8} {'Act_B':>8} {'Band':>4} {'Label'}")
    print("-" * 90)
    for rank, f in enumerate(comparison["b_dominant"][:top_n]):
        print(
            f"{rank+1:>4} {f['index']:>6} {abs(f['diff']):>+8.3f} "
            f"{f['mean_a']:>8.3f} {f['mean_b']:>8.3f} "
            f"{f['band']:>4} {f['label']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Dataset concept fingerprints via SAE activation profiles"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None, help="Show fingerprint for one dataset")
    parser.add_argument("--compare", nargs=2, metavar=("DS_A", "DS_B"), help="Compare two datasets")
    parser.add_argument("--top", type=int, default=20, help="Show top N concepts")
    parser.add_argument("--similarity-matrix", action="store_true", help="Print pairwise similarity")
    parser.add_argument("--output", type=str, default=None, help="Save fingerprints to JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info(f"Computing fingerprints for {args.model}...")
    fp = compute_fingerprints(args.model, device="cpu")
    logger.info(f"Computed fingerprints for {fp['n_datasets']} datasets, "
                f"{len(fp['alive_features'])} alive features")

    # Global stats
    gm = fp["global_mean"]
    alive = fp["alive_features"]
    alive_gm = [gm[i] for i in alive]
    logger.info(f"Global mean activation: mean={np.mean(alive_gm):.4f}, "
                f"std={np.std(alive_gm):.4f}, "
                f"max={np.max(alive_gm):.4f}")

    if args.dataset:
        print_dataset_fingerprint(fp, args.dataset, args.top)

    elif args.compare:
        comparison = compare_datasets(fp, args.compare[0], args.compare[1], args.top)
        print_comparison(comparison, args.top)

    elif args.similarity_matrix:
        sim, datasets = dataset_similarity_matrix(fp)
        print(f"\nPairwise cosine similarity ({len(datasets)} datasets):")
        # Show most similar and most different pairs
        pairs = []
        for i in range(len(datasets)):
            for j in range(i + 1, len(datasets)):
                pairs.append((sim[i, j], datasets[i], datasets[j]))
        pairs.sort(reverse=True)
        print(f"\nMost similar pairs:")
        for s, a, b in pairs[:10]:
            print(f"  {s:.3f}  {a} — {b}")
        print(f"\nMost different pairs:")
        for s, a, b in pairs[-10:]:
            print(f"  {s:.3f}  {a} — {b}")

    else:
        # Default: show overview of all datasets ranked by distinctiveness
        print(f"\n{'='*100}")
        print(f"Dataset Distinctiveness Ranking ({args.model})")
        print(f"{'='*100}")

        # Rank by L2 norm of deviation (how far from global mean)
        rankings = []
        for ds_name, dev in fp["dataset_deviations"].items():
            alive_dev = np.array([dev[i] for i in alive])
            l2 = np.linalg.norm(alive_dev)
            n_distinctive = sum(1 for d in alive_dev if abs(d) > 0.01)
            # Top concept for this dataset
            top_idx = alive[np.argmax(np.abs(alive_dev))]
            top_dev = dev[top_idx]
            top_label = fp["feature_labels"].get(top_idx, "unknown")
            rankings.append((l2, ds_name, n_distinctive, top_idx, top_dev, top_label))

        rankings.sort(reverse=True)

        print(f"\n{'Rank':>4} {'L2':>7} {'#Dist':>5} {'Dataset':<35} "
              f"{'TopFeat':>7} {'TopDev':>8} {'Label'}")
        print("-" * 110)
        for rank, (l2, ds, n_dist, top_idx, top_dev, label) in enumerate(rankings):
            print(f"{rank+1:>4} {l2:>7.2f} {n_dist:>5} {ds:<35} "
                  f"{top_idx:>7} {top_dev:>+8.3f} {label[:40]}")

    # Save if requested
    if args.output:
        save_data = {
            "model": args.model,
            "n_datasets": fp["n_datasets"],
            "alive_features": fp["alive_features"],
            "bands": fp["bands"],
            "global_mean": fp["global_mean"].tolist(),
            "datasets": {},
        }
        for ds_name in sorted(fp["dataset_means"].keys()):
            save_data["datasets"][ds_name] = {
                "mean_activation": fp["dataset_means"][ds_name].tolist(),
                "deviation": fp["dataset_deviations"][ds_name].tolist(),
            }
        with open(args.output, "w") as f:
            json.dump(save_data, f)
        logger.info(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
