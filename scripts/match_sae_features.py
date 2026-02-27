#!/usr/bin/env python3
"""
Row-level cross-model SAE feature matching.

For each model pair, computes row-level cross-correlation between SAE
activations on shared datasets, then finds 1-to-1 feature correspondences
via mutual nearest neighbors, Hungarian algorithm, or many-to-one matching.
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.sparse_autoencoder import SAEConfig, SparseAutoencoder
from scripts.analyze_sae_concepts_deep import (
    NumpyEncoder,
    convert_keys_to_native,
    load_sae_checkpoint,
)
from scripts.compare_sae_cross_model import (
    DEFAULT_MODELS,
    DEFAULT_SAE_ROUND,
    find_common_datasets,
    sae_sweep_dir,
)

EMB_DIR = PROJECT_ROOT / "output" / "embeddings" / "tabarena"
SAE_FILENAME = "sae_matryoshka_archetypal_validated.pt"


# ── Embedding & activation helpers ─────────────────────────────────────────


def load_embeddings(
    emb_dir: Path, dataset: str, max_per_dataset: int = 500
) -> np.ndarray:
    """Load embeddings for a dataset, subsampled to max_per_dataset rows."""
    path = emb_dir / f"tabarena_{dataset}.npz"
    data = np.load(path, allow_pickle=True)
    emb = data["embeddings"].astype(np.float32)
    if len(emb) > max_per_dataset:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(emb), max_per_dataset, replace=False)
        emb = emb[idx]
    return emb


def compute_sae_activations(
    model: SparseAutoencoder, embeddings: np.ndarray
) -> np.ndarray:
    """Encode raw embeddings through SAE, return activations (n_samples, hidden_dim)."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(embeddings, dtype=torch.float32)
        h = model.encode(x).numpy()
    return h


def get_alive_mask(activations: np.ndarray, threshold: float = 0.001) -> np.ndarray:
    """Boolean mask of features whose max activation exceeds threshold."""
    return activations.max(axis=0) > threshold


# ── Cross-correlation ──────────────────────────────────────────────────────


def compute_cross_correlation(
    acts_A: np.ndarray, acts_B: np.ndarray
) -> np.ndarray:
    """
    Compute |Pearson r| between all feature pairs across two activation matrices.

    Z-scores each column, computes Z_A.T @ Z_B / n, takes absolute value.
    Constant columns get r=0.

    Args:
        acts_A: (n_samples, d_A)
        acts_B: (n_samples, d_B)

    Returns:
        (d_A, d_B) matrix of |r| values.
    """
    n = acts_A.shape[0]
    assert acts_A.shape[0] == acts_B.shape[0], "Sample count mismatch"

    def zscore(x):
        mu = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std[std < 1e-12] = 1.0  # constant columns → zero after centering
        return (x - mu) / std

    z_a = zscore(acts_A)
    z_b = zscore(acts_B)
    corr = np.abs(z_a.T @ z_b) / n
    return corr


# ── Matching algorithms ────────────────────────────────────────────────────


def match_mutual_nearest_neighbors(
    mean_corr: np.ndarray, indices_A: np.ndarray, indices_B: np.ndarray
) -> List[Tuple[int, int, float]]:
    """
    Mutual nearest neighbor matching: keep (i, k) where i is k's best match
    AND k is i's best match.

    Args:
        mean_corr: (n_alive_A, n_alive_B) correlation matrix
        indices_A: original feature indices for rows
        indices_B: original feature indices for columns

    Returns:
        List of (idx_A, idx_B, r) tuples, sorted by r descending.
    """
    if mean_corr.size == 0:
        return []
    best_for_a = mean_corr.argmax(axis=1)  # for each row, best column
    best_for_b = mean_corr.argmax(axis=0)  # for each col, best row
    matches = []
    for i in range(len(indices_A)):
        k = best_for_a[i]
        if best_for_b[k] == i:
            r = float(mean_corr[i, k])
            matches.append((int(indices_A[i]), int(indices_B[k]), r))
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches


def match_hungarian(
    mean_corr: np.ndarray, indices_A: np.ndarray, indices_B: np.ndarray
) -> List[Tuple[int, int, float]]:
    """
    Optimal 1-to-1 matching via Hungarian algorithm.

    Assigns min(n_alive_A, n_alive_B) pairs to maximize total correlation.

    Returns:
        List of (idx_A, idx_B, r) tuples, sorted by r descending.
    """
    if mean_corr.size == 0:
        return []
    cost = 1.0 - mean_corr
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for i, k in zip(row_ind, col_ind):
        r = float(mean_corr[i, k])
        matches.append((int(indices_A[i]), int(indices_B[k]), r))
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches


def match_many_to_one(
    mean_corr: np.ndarray, indices_A: np.ndarray, indices_B: np.ndarray
) -> List[Tuple[int, int, float]]:
    """
    Many-to-one matching: for each feature in the smaller model, find best
    match in the larger model. Allows duplicate targets.

    Returns:
        List of (idx_small, idx_large, r) tuples, sorted by r descending.
    """
    if mean_corr.size == 0:
        return []
    # Determine which axis is smaller
    if len(indices_A) <= len(indices_B):
        # A is smaller: for each row in A, find best column in B
        best = mean_corr.argmax(axis=1)
        matches = [
            (int(indices_A[i]), int(indices_B[best[i]]), float(mean_corr[i, best[i]]))
            for i in range(len(indices_A))
        ]
    else:
        # B is smaller: for each column in B, find best row in A
        best = mean_corr.argmax(axis=0)
        matches = [
            (int(indices_B[k]), int(indices_A[best[k]]), float(mean_corr[best[k], k]))
            for k in range(len(indices_B))
        ]
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches


def match_tiered(
    mean_corr: np.ndarray, indices_A: np.ndarray, indices_B: np.ndarray
) -> List[Tuple[int, int, float]]:
    """
    Tiered matching: MNN → Hungarian → many-to-one on residuals.

    Each match gets a tier tag via 4th tuple element:
      1 = MNN (mutual nearest neighbor, highest confidence)
      2 = Hungarian (optimal 1-to-1 on MNN residuals)
      3 = many-to-one (remaining unmatched, allows duplicates)

    Returns:
        List of (idx_A, idx_B, r, tier) tuples, sorted by tier then r desc.
    """
    if mean_corr.size == 0:
        return []

    # ── Tier 1: MNN ──
    t1 = match_mutual_nearest_neighbors(mean_corr, indices_A, indices_B)
    matched_rows = {
        np.searchsorted(indices_A, m[0]) for m in t1
        if m[0] in indices_A
    }
    matched_cols = {
        np.searchsorted(indices_B, m[1]) for m in t1
        if m[1] in indices_B
    }
    # Use set lookup on actual index values for correctness
    matched_A_vals = {m[0] for m in t1}
    matched_B_vals = {m[1] for m in t1}

    remaining_rows = [i for i in range(len(indices_A)) if indices_A[i] not in matched_A_vals]
    remaining_cols = [j for j in range(len(indices_B)) if indices_B[j] not in matched_B_vals]

    results = [(a, b, r, 1) for a, b, r in t1]

    # ── Tier 2: Hungarian on residual ──
    if remaining_rows and remaining_cols:
        sub_corr = mean_corr[np.ix_(remaining_rows, remaining_cols)]
        sub_idx_A = indices_A[remaining_rows]
        sub_idx_B = indices_B[remaining_cols]
        t2 = match_hungarian(sub_corr, sub_idx_A, sub_idx_B)

        matched_A_vals.update(m[0] for m in t2)
        matched_B_vals.update(m[1] for m in t2)
        results.extend((a, b, r, 2) for a, b, r in t2)

        remaining_rows = [i for i in range(len(indices_A)) if indices_A[i] not in matched_A_vals]
        remaining_cols = [j for j in range(len(indices_B)) if indices_B[j] not in matched_B_vals]

    # ── Tier 3: many-to-one on remainder ──
    if remaining_rows and remaining_cols:
        sub_corr = mean_corr[np.ix_(remaining_rows, remaining_cols)]
        sub_idx_A = indices_A[remaining_rows]
        sub_idx_B = indices_B[remaining_cols]
        # For each remaining A feature, find best B (allows duplicates in B)
        best = sub_corr.argmax(axis=1)
        for i in range(len(sub_idx_A)):
            r = float(sub_corr[i, best[i]])
            results.append((int(sub_idx_A[i]), int(sub_idx_B[best[i]]), r, 3))
    elif remaining_rows:
        # B is exhausted — remaining A features get no match
        pass
    elif remaining_cols:
        # A is exhausted — remaining B features unmatched (tracked via unmatched_b)
        pass

    # Sort by tier, then r descending within tier
    results.sort(key=lambda x: (x[3], -x[2]))
    return results


def match_tiered_m2o(
    mean_corr: np.ndarray, indices_A: np.ndarray, indices_B: np.ndarray
) -> List[Tuple[int, int, float]]:
    """
    Tiered matching: MNN → many-to-one on residuals (concept splitting detector).

    Tier 1: MNN (mutual nearest neighbor, 1-to-1)
    Tier 2: For each remaining A feature, find best match across ALL B features
            (including those already matched by MNN). Allows duplicates in B,
            revealing concept splitting — multiple A features encoding the same
            B concept.

    Returns:
        List of (idx_A, idx_B, r, tier) tuples, sorted by tier then r desc.
    """
    if mean_corr.size == 0:
        return []

    # ── Tier 1: MNN ──
    t1 = match_mutual_nearest_neighbors(mean_corr, indices_A, indices_B)
    matched_A_vals = {m[0] for m in t1}

    results = [(a, b, r, 1) for a, b, r in t1]

    # ── Tier 2: many-to-one for remaining A, searching ALL of B ──
    remaining_rows = [i for i in range(len(indices_A)) if indices_A[i] not in matched_A_vals]
    if remaining_rows:
        # Search full B (all columns), not just unmatched
        sub_corr = mean_corr[remaining_rows, :]  # (n_remaining, all_B)
        best = sub_corr.argmax(axis=1)
        for i, row_idx in enumerate(remaining_rows):
            k = best[i]
            r = float(sub_corr[i, k])
            results.append((int(indices_A[row_idx]), int(indices_B[k]), r, 2))

    # Sort by tier, then r descending within tier
    results.sort(key=lambda x: (x[3], -x[2]))
    return results


# ── Orchestration ──────────────────────────────────────────────────────────


def match_model_pair(
    model_A: SparseAutoencoder,
    model_B: SparseAutoencoder,
    emb_dir_A: Path,
    emb_dir_B: Path,
    datasets: List[str],
    method: str = "mnn",
    alive_threshold: float = 0.001,
    max_per_dataset: int = 500,
) -> dict:
    """
    Match SAE features between two models across shared datasets.

    1. For each dataset: load embeddings, encode through both SAEs
    2. Pool activations across all datasets
    3. Determine alive features from pooled activations
    4. Compute cross-correlation on pooled alive activations (~19k rows)
    5. Match using selected method

    Returns:
        Dict with matches, unmatched features, and statistics.
    """
    # Step 1-2: Collect activations across all datasets
    all_acts_A = []
    all_acts_B = []

    for ds in datasets:
        emb_A = load_embeddings(emb_dir_A, ds, max_per_dataset)
        emb_B = load_embeddings(emb_dir_B, ds, max_per_dataset)

        acts_A = compute_sae_activations(model_A, emb_A)
        acts_B = compute_sae_activations(model_B, emb_B)

        # Align sample count (embeddings may differ per model)
        n = min(len(acts_A), len(acts_B))
        all_acts_A.append(acts_A[:n])
        all_acts_B.append(acts_B[:n])

    pooled_A = np.concatenate(all_acts_A, axis=0)
    pooled_B = np.concatenate(all_acts_B, axis=0)

    # Step 3: Alive mask from pooled activations
    alive_A = get_alive_mask(pooled_A, alive_threshold)
    alive_B = get_alive_mask(pooled_B, alive_threshold)

    indices_A = np.where(alive_A)[0]
    indices_B = np.where(alive_B)[0]

    if len(indices_A) == 0 or len(indices_B) == 0:
        return {
            "n_alive_a": int(alive_A.sum()),
            "n_alive_b": int(alive_B.sum()),
            "n_matched": 0,
            "mean_match_r": 0.0,
            "matches": [],
            "unmatched_a": indices_A.tolist(),
            "unmatched_b": indices_B.tolist(),
            "n_samples": 0,
        }

    # Step 4: Cross-correlation on pooled alive activations
    mean_corr = compute_cross_correlation(
        pooled_A[:, alive_A], pooled_B[:, alive_B]
    )

    # Step 5: Match
    match_fn = {
        "mnn": match_mutual_nearest_neighbors,
        "hungarian": match_hungarian,
        "many_to_one": match_many_to_one,
        "tiered": match_tiered,
        "tiered_m2o": match_tiered_m2o,
    }[method]
    matches = match_fn(mean_corr, indices_A, indices_B)

    # Tiered returns 4-tuples (idx_a, idx_b, r, tier), others return 3-tuples
    is_tiered = method in ("tiered", "tiered_m2o")

    # Compute unmatched
    matched_A = {m[0] for m in matches}
    matched_B = {m[1] for m in matches}
    unmatched_A = sorted(int(i) for i in indices_A if int(i) not in matched_A)
    unmatched_B = sorted(int(i) for i in indices_B if int(i) not in matched_B)

    mean_r = float(np.mean([m[2] for m in matches])) if matches else 0.0

    if is_tiered:
        match_list = [
            {"idx_a": m[0], "idx_b": m[1], "r": round(m[2], 4), "tier": m[3]}
            for m in matches
        ]
        tier_counts = {}
        tier_mean_r = {}
        for t in (1, 2, 3):
            tier_matches = [m for m in matches if m[3] == t]
            if tier_matches:
                tier_counts[t] = len(tier_matches)
                tier_mean_r[t] = round(float(np.mean([m[2] for m in tier_matches])), 4)
    else:
        match_list = [
            {"idx_a": m[0], "idx_b": m[1], "r": round(m[2], 4)} for m in matches
        ]
        tier_counts = None
        tier_mean_r = None

    result = {
        "n_alive_a": int(alive_A.sum()),
        "n_alive_b": int(alive_B.sum()),
        "n_matched": len(matches),
        "mean_match_r": mean_r,
        "n_samples": len(pooled_A),
        "matches": match_list,
        "unmatched_a": unmatched_A,
        "unmatched_b": unmatched_B,
    }
    if tier_counts:
        result["tier_counts"] = tier_counts
        result["tier_mean_r"] = tier_mean_r

    return result


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Row-level cross-model SAE feature matching"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: auto-generated from method/params)",
    )
    parser.add_argument(
        "--method",
        choices=["mnn", "hungarian", "many_to_one", "tiered", "tiered_m2o"],
        default="mnn",
        help="Matching method (default: mnn)",
    )
    parser.add_argument(
        "--alive-threshold",
        type=float,
        default=0.001,
        help="Min max-activation to consider a feature alive (default: 0.001)",
    )
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=500,
        help="Max samples per dataset (default: 500)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model keys to include (default: all with checkpoints)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help=f"SAE sweep round (default: {DEFAULT_SAE_ROUND})",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = (
            f"output/sae_feature_matching"
            f"_{args.method}"
            f"_t{args.alive_threshold}"
            f"_n{args.max_per_dataset}"
            f".json"
        )

    sweep_dir = sae_sweep_dir(args.round)

    # Discover models with matryoshka_archetypal checkpoints
    models = {}
    for display_name, model_key, emb_key in DEFAULT_MODELS:
        if args.models and model_key not in args.models:
            continue
        ckpt = sweep_dir / model_key / SAE_FILENAME
        if not ckpt.exists():
            print(f"  Skipping {display_name}: no checkpoint at {ckpt}")
            continue
        emb_dir = EMB_DIR / emb_key
        if not emb_dir.exists():
            print(f"  Skipping {display_name}: no embeddings at {emb_dir}")
            continue
        models[display_name] = {
            "model_key": model_key,
            "emb_key": emb_key,
            "ckpt": ckpt,
            "emb_dir": emb_dir,
        }

    model_names = sorted(models.keys())
    print(f"Models with checkpoints: {model_names}")

    # Find common datasets across all models
    emb_dirs = {name: info["emb_dir"] for name, info in models.items()}
    datasets = find_common_datasets(emb_dirs)
    print(f"Common datasets: {len(datasets)}")

    # Load SAE checkpoints
    saes = {}
    for name, info in models.items():
        print(f"Loading SAE: {name} from {info['ckpt']}")
        model, config, _ = load_sae_checkpoint(info["ckpt"])
        saes[name] = model
        print(f"  hidden_dim={config.hidden_dim}, topk={config.topk}")

    # Match all pairs
    pairs = {}
    summary = {}
    for name_a, name_b in combinations(model_names, 2):
        pair_key = f"{name_a}__{name_b}"
        print(f"\nMatching: {pair_key}")
        result = match_model_pair(
            model_A=saes[name_a],
            model_B=saes[name_b],
            emb_dir_A=models[name_a]["emb_dir"],
            emb_dir_B=models[name_b]["emb_dir"],
            datasets=datasets,
            method=args.method,
            alive_threshold=args.alive_threshold,
            max_per_dataset=args.max_per_dataset,
        )
        pairs[pair_key] = result

        frac_a = result["n_matched"] / max(result["n_alive_a"], 1)
        frac_b = result["n_matched"] / max(result["n_alive_b"], 1)
        summary[pair_key] = {
            "n_matched": result["n_matched"],
            "mean_r": round(result["mean_match_r"], 4),
            "frac_a": round(frac_a, 4),
            "frac_b": round(frac_b, 4),
        }
        tier_str = ""
        if "tier_counts" in result and result["tier_counts"]:
            parts = []
            for t in (1, 2, 3):
                if t in result["tier_counts"]:
                    label = {1: "MNN", 2: "Hungarian", 3: "many-to-1"}[t]
                    parts.append(
                        f"{label}={result['tier_counts'][t]}"
                        f"(r={result['tier_mean_r'][t]:.3f})"
                    )
            tier_str = "  " + ", ".join(parts)
        print(
            f"  matched={result['n_matched']}, "
            f"mean_r={result['mean_match_r']:.3f}, "
            f"alive_a={result['n_alive_a']}, alive_b={result['n_alive_b']}"
            f"{tier_str}"
        )

    # Write output
    output = {
        "metadata": {
            "n_models": len(models),
            "n_datasets": len(datasets),
            "method": args.method,
            "alive_threshold": args.alive_threshold,
            "max_per_dataset": args.max_per_dataset,
            "models": model_names,
        },
        "pairs": pairs,
        "summary": {"per_pair": summary},
    }

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(convert_keys_to_native(output), f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved to {out_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Pair':<35} {'Matched':>8} {'Mean r':>8} {'Frac A':>8} {'Frac B':>8}")
    print("-" * 70)
    for pair_key in sorted(summary.keys(), key=lambda k: -summary[k]["mean_r"]):
        s = summary[pair_key]
        print(
            f"{pair_key:<35} {s['n_matched']:>8} {s['mean_r']:>8.3f} "
            f"{s['frac_a']:>8.3f} {s['frac_b']:>8.3f}"
        )


if __name__ == "__main__":
    main()
