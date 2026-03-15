#!/usr/bin/env python3
"""Embed concept descriptions and compute validation metrics.

Uses nomic-embed-text-v1.5 via sentence-transformers for local embedding.
Matryoshka support enables multi-resolution comparison (768d, 256d, 128d, 64d).

Usage:
    python scripts/embed_concept_descriptions.py
    python scripts/embed_concept_descriptions.py --dim 256
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cosine as cosine_dist

from scripts._project_root import PROJECT_ROOT

INPUT_DIR = PROJECT_ROOT / "output" / "concept_descriptions"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"

_model_cache = None


def _get_model():
    """Lazy-load sentence-transformers model."""
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer
        _model_cache = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    return _model_cache


def embed_descriptions(descriptions: List[str], dim: int = 768) -> np.ndarray:
    """Embed text descriptions using nomic-embed-text-v1.5.

    Args:
        descriptions: List of text strings to embed.
        dim: Embedding dimension (768, 256, 128, or 64 for Matryoshka).

    Returns:
        (n, dim) array of L2-normalized embeddings.
    """
    model = _get_model()
    # nomic requires "search_document: " prefix for documents
    prefixed = [f"search_document: {d}" for d in descriptions]
    embeddings = model.encode(prefixed, normalize_embeddings=True)

    if dim < embeddings.shape[1]:
        embeddings = embeddings[:, :dim]
        # Re-normalize after truncation (Matryoshka)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)

    return embeddings.astype(np.float32)


def compute_within_group_coherence(
    embeddings: np.ndarray,
    group_ids: List[int],
) -> Dict:
    """Mean pairwise cosine similarity within each group.

    Args:
        embeddings: (n, dim) array.
        group_ids: Group ID per feature (-1 for unmatched).

    Returns:
        Dict with mean, std, n_groups, per_group stats.
    """
    unique_groups = sorted(set(g for g in group_ids if g >= 0))
    per_group = {}

    for gid in unique_groups:
        indices = [i for i, g in enumerate(group_ids) if g == gid]
        if len(indices) < 2:
            continue
        sims = []
        for i, j in combinations(indices, 2):
            sim = 1.0 - cosine_dist(embeddings[i], embeddings[j])
            sims.append(sim)
        per_group[str(gid)] = float(np.mean(sims))

    vals = list(per_group.values())
    return {
        "mean": float(np.mean(vals)) if vals else 0.0,
        "std": float(np.std(vals)) if vals else 0.0,
        "n_groups": len(per_group),
        "per_group": per_group,
    }


def compute_matched_pair_agreement(
    embeddings: np.ndarray,
    matched_pairs: List[Tuple[int, int]],
    n_random: int = 1000,
    seed: int = 42,
) -> Dict:
    """Cosine similarity of matched pairs vs random baseline.

    Args:
        embeddings: (n, dim) array.
        matched_pairs: List of (idx_a, idx_b) pairs to compare.
        n_random: Number of random pairs for baseline.

    Returns:
        Dict with mean, std, random_baseline, n_pairs.
    """
    if not matched_pairs:
        return {"mean": 0.0, "std": 0.0, "random_baseline": 0.0, "n_pairs": 0}

    pair_sims = []
    for i, j in matched_pairs:
        sim = 1.0 - cosine_dist(embeddings[i], embeddings[j])
        pair_sims.append(sim)

    rng = np.random.RandomState(seed)
    n = len(embeddings)
    random_sims = []
    for _ in range(n_random):
        i, j = rng.choice(n, 2, replace=False)
        sim = 1.0 - cosine_dist(embeddings[i], embeddings[j])
        random_sims.append(sim)

    return {
        "mean": float(np.mean(pair_sims)),
        "std": float(np.std(pair_sims)),
        "random_baseline": float(np.mean(random_sims)),
        "n_pairs": len(matched_pairs),
    }


def compute_interpolation_coherence(
    embeddings: np.ndarray,
    feature_ids: List[str],
    unmatched_landmarks: Dict[str, List[Tuple[str, float]]],
) -> Dict:
    """Cosine sim between unexplained features and their landmarks.

    Args:
        embeddings: (n, dim) array.
        feature_ids: Feature ID per row in embeddings.
        unmatched_landmarks: {feat_id: [(landmark_feat_id, corr), ...]}.

    Returns:
        Dict with mean, std, n_features.
    """
    id_to_idx = {fid: i for i, fid in enumerate(feature_ids)}
    sims = []

    for feat_id, landmarks in unmatched_landmarks.items():
        if feat_id not in id_to_idx:
            continue
        feat_emb = embeddings[id_to_idx[feat_id]]
        for landmark_id, _ in landmarks:
            if landmark_id in id_to_idx:
                lm_emb = embeddings[id_to_idx[landmark_id]]
                sim = 1.0 - cosine_dist(feat_emb, lm_emb)
                sims.append(sim)

    return {
        "mean": float(np.mean(sims)) if sims else 0.0,
        "std": float(np.std(sims)) if sims else 0.0,
        "n_features": len(unmatched_landmarks),
    }


def main():
    parser = argparse.ArgumentParser(description="Embed concept descriptions")
    parser.add_argument("--dim", type=int, default=768,
                        choices=[64, 128, 256, 768],
                        help="Embedding dimension (Matryoshka)")
    parser.add_argument("--input", type=str,
                        default=str(INPUT_DIR / "concept_descriptions.json"))
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    # Collect all descriptions with feature IDs and group membership
    feature_ids = []
    descriptions = []
    group_ids = []

    for gid, group in data.get("groups", {}).items():
        features = group.get("features", {})
        if features:
            for feat_key, feat in features.items():
                feature_ids.append(feat_key)
                descriptions.append(feat.get("description", feat.get("brief_label", "")))
                group_ids.append(int(gid))
        else:
            # Fallback: use group-level summary/brief_label
            desc = group.get("summary", group.get("brief_label", ""))
            if desc:
                feature_ids.append(f"group:{gid}")
                descriptions.append(desc)
                group_ids.append(int(gid))

    for feat_key, feat in data.get("unmatched", {}).items():
        feature_ids.append(feat_key)
        descriptions.append(feat.get("description", feat.get("brief_label", "")))
        group_ids.append(-1)

    print(f"Embedding {len(descriptions)} descriptions at dim={args.dim}...")
    embeddings = embed_descriptions(descriptions, dim=args.dim)

    # Save embeddings
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = INPUT_DIR / "concept_embeddings.npz"
    np.savez_compressed(
        str(emb_path),
        embeddings=embeddings,
        feature_ids=np.array(feature_ids),
        group_ids=np.array(group_ids),
    )
    print(f"Saved: {emb_path}")

    # Compute metrics
    print("Computing metrics...")
    metrics = {
        "model": EMBEDDING_MODEL,
        "embedding_dim": args.dim,
        "n_features": len(descriptions),
    }

    metrics["within_group_cosine"] = compute_within_group_coherence(
        embeddings, group_ids,
    )
    print(f"  Within-group coherence: {metrics['within_group_cosine']['mean']:.3f}")

    # Build matched pairs from group membership (all in-group pairs)
    matched_pairs = []
    group_indices = {}
    for i, gid in enumerate(group_ids):
        if gid >= 0:
            group_indices.setdefault(gid, []).append(i)
    for indices in group_indices.values():
        for i, j in combinations(indices, 2):
            matched_pairs.append((i, j))

    metrics["matched_pair_cosine"] = compute_matched_pair_agreement(
        embeddings, matched_pairs,
    )
    print(f"  Matched-pair agreement: {metrics['matched_pair_cosine']['mean']:.3f}"
          f" (random: {metrics['matched_pair_cosine']['random_baseline']:.3f})")

    # Interpolation coherence: unmatched features vs their landmarks
    unmatched_landmarks = {}
    for feat_key, feat in data.get("unmatched", {}).items():
        landmarks = feat.get("landmarks", [])
        if landmarks:
            unmatched_landmarks[feat_key] = landmarks

    if unmatched_landmarks:
        metrics["interpolation_cosine"] = compute_interpolation_coherence(
            embeddings, feature_ids, unmatched_landmarks,
        )
        print(f"  Interpolation coherence: {metrics['interpolation_cosine']['mean']:.3f}")

    metrics_path = INPUT_DIR / "concept_embedding_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
