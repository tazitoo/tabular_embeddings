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

from scripts._project_root import PROJECT_ROOT

from analysis.sparse_autoencoder import SAEConfig, SparseAutoencoder
from scripts.sae.analyze_sae_concepts_deep import (
    NumpyEncoder,
    convert_keys_to_native,
    load_sae_checkpoint,
)
from scripts.sae.compare_sae_cross_model import (
    DEFAULT_MODELS,
    DEFAULT_SAE_ROUND,
    SAE_FILENAME,
    sae_sweep_dir,
)

RANDOM_BASELINE_FILENAME = "sae_matryoshka_archetypal_random_baseline.pt"

# Re-export shared utilities for backward compatibility
from scripts.matching.utils import (  # noqa: E402, F401
    EMB_DIR, SAE_DATA_DIR,
    load_norm_stats, load_test_embeddings,
    load_embeddings, compute_sae_activations, get_alive_mask,
    compute_alive_mask,
)


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
    datasets: List[str],
    test_embs_A: Dict[str, np.ndarray],
    test_embs_B: Dict[str, np.ndarray],
    alive_mask_A: np.ndarray = None,
    alive_mask_B: np.ndarray = None,
    method: str = "mnn",
    return_corr_matrix: bool = False,
) -> dict:
    """
    Match SAE features between two models across shared datasets.

    Args:
        alive_mask_A/B: Pre-computed boolean masks (features that fire on
            at least one row). If None, computed from pooled test activations.

    Returns:
        Dict with matches, unmatched features, and statistics.
    """
    # Step 1-2: Collect test activations across shared datasets
    all_acts_A = []
    all_acts_B = []

    for ds in datasets:
        emb_A = test_embs_A[ds]
        emb_B = test_embs_B[ds]

        acts_A = compute_sae_activations(model_A, emb_A)
        acts_B = compute_sae_activations(model_B, emb_B)

        # Align sample count (test splits may differ per model)
        n = min(len(acts_A), len(acts_B))
        all_acts_A.append(acts_A[:n])
        all_acts_B.append(acts_B[:n])

    pooled_A = np.concatenate(all_acts_A, axis=0)
    pooled_B = np.concatenate(all_acts_B, axis=0)

    # Step 3: Alive mask — features that fire on at least one test row
    alive_A = alive_mask_A if alive_mask_A is not None else get_alive_mask(pooled_A)
    alive_B = alive_mask_B if alive_mask_B is not None else get_alive_mask(pooled_B)

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
        "indices_a": indices_A,
        "indices_b": indices_B,
        "matches": match_list,
        "unmatched_a": unmatched_A,
        "unmatched_b": unmatched_B,
    }
    if tier_counts:
        result["tier_counts"] = tier_counts
        result["tier_mean_r"] = tier_mean_r

    if return_corr_matrix:
        result["corr_matrix"] = mean_corr

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
    parser.add_argument(
        "--save-correlations",
        action="store_true",
        help="Save full cross-correlation matrices to output/sae_cross_correlations/",
    )
    parser.add_argument(
        "--random-baseline",
        action="store_true",
        help="Match each model's trained SAE vs its random baseline (control).",
    )
    parser.add_argument(
        "--cross-model-baseline",
        action="store_true",
        help="Compute cross-model null: trained-A vs random-B for all pairs.",
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        default=None,
        help="Override SAE checkpoint directory (default: sae_tabarena_sweep_round{N})",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = (
            f"output/sae_feature_matching"
            f"_{args.method}"
            f"_t{args.alive_threshold}"
            f".json"
        )

    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else sae_sweep_dir(args.round)

    # Discover models with checkpoints and load train+test embeddings
    models = {}
    for display_name, model_key, emb_key in DEFAULT_MODELS:
        if args.models and model_key not in args.models:
            continue
        ckpt = sweep_dir / model_key / SAE_FILENAME
        if not ckpt.exists():
            print(f"  Skipping {display_name}: no checkpoint at {ckpt}")
            continue
        try:
            test_embs = load_test_embeddings(model_key)
        except FileNotFoundError as e:
            print(f"  Skipping {display_name}: {e}")
            continue
        models[display_name] = {
            "model_key": model_key,
            "emb_key": emb_key,
            "ckpt": ckpt,
            "test_embs": test_embs,
        }
        n_test = sum(len(v) for v in test_embs.values())
        print(f"  {display_name}: {len(test_embs)} datasets, {n_test} test rows")

    model_names = sorted(models.keys())
    print(f"Models with checkpoints: {model_names}")

    # Find common datasets from test splits
    all_ds_sets = [set(info["test_embs"].keys()) for info in models.values()]
    datasets = sorted(set.intersection(*all_ds_sets)) if all_ds_sets else []
    print(f"Common datasets: {len(datasets)}")

    # Load SAE checkpoints and compute alive masks from test data.
    # Using test data (which is also used for matching and regression)
    # ensures consistency: every alive feature has signal in the data
    # we actually analyze downstream.
    saes = {}
    alive_masks = {}
    for name, info in models.items():
        print(f"Loading SAE: {name} from {info['ckpt']}")
        model, config, _ = load_sae_checkpoint(info["ckpt"])
        saes[name] = model
        mask = compute_alive_mask(model, info["test_embs"])
        alive_masks[name] = mask
        print(f"  hidden_dim={config.hidden_dim}, topk={config.topk}, "
              f"alive={mask.sum()}/{len(mask)} ({mask.sum()/len(mask)*100:.1f}%)")

    # Optionally prepare correlation output dir
    corr_dir = None
    if args.save_correlations:
        corr_dir = PROJECT_ROOT / "output" / "sae_cross_correlations"
        corr_dir.mkdir(parents=True, exist_ok=True)

    # Random baseline: trained vs random for each model
    if args.random_baseline:
        pairs = {}
        summary = {}
        for name in model_names:
            info = models[name]
            rand_ckpt = info["ckpt"].parent / RANDOM_BASELINE_FILENAME
            if not rand_ckpt.exists():
                print(f"  Skipping {name}: no random baseline at {rand_ckpt}")
                continue
            print(f"\nBaseline: {name} trained vs random")
            rand_model, rand_config, _ = load_sae_checkpoint(rand_ckpt)
            print(f"  random hidden_dim={rand_config.hidden_dim}, topk={rand_config.topk}")

            # Use all datasets in this model's test split
            model_datasets = sorted(info["test_embs"].keys())
            rand_alive = compute_alive_mask(
                rand_model, info["test_embs"]
            )
            result = match_model_pair(
                model_A=saes[name],
                model_B=rand_model,
                datasets=model_datasets,
                test_embs_A=info["test_embs"],
                test_embs_B=info["test_embs"],
                alive_mask_A=alive_masks[name],
                alive_mask_B=rand_alive,
                method=args.method,
            )
            pair_key = f"{name}__random"
            pairs[pair_key] = result
            frac_a = result["n_matched"] / max(result["n_alive_a"], 1)
            summary[pair_key] = {
                "n_matched": result["n_matched"],
                "mean_r": round(result["mean_match_r"], 4),
                "frac_trained": round(frac_a, 4),
                "alive_trained": result["n_alive_a"],
                "alive_random": result["n_alive_b"],
            }
            print(
                f"  matched={result['n_matched']}, "
                f"mean_r={result['mean_match_r']:.3f}, "
                f"alive_trained={result['n_alive_a']}, "
                f"alive_random={result['n_alive_b']}"
            )

        output = {
            "metadata": {
                "mode": "random_baseline",
                "n_models": len(pairs),
                "method": args.method,
                "alive_threshold": args.alive_threshold,
                "split": "test",
            },
            "pairs": pairs,
            "summary": {"per_pair": summary},
        }
        if args.output == (
            f"output/sae_feature_matching"
            f"_{args.method}"
            f"_t{args.alive_threshold}"
            f".json"
        ):
            args.output = args.output.replace(".json", "_random_baseline.json")

        out_path = PROJECT_ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(convert_keys_to_native(output), f, indent=2, cls=NumpyEncoder)
        print(f"\nSaved to {out_path}")

        print("\n" + "=" * 70)
        print(f"{'Model':<20} {'Matched':>8} {'Mean r':>8} {'Alive(T)':>9} {'Alive(R)':>9}")
        print("-" * 70)
        for pk in sorted(summary.keys()):
            s = summary[pk]
            print(
                f"{pk:<20} {s['n_matched']:>8} {s['mean_r']:>8.3f} "
                f"{s['alive_trained']:>9} {s['alive_random']:>9}"
            )
        return

    # Cross-model random baseline: trained-A vs random-B for all directed pairs
    if args.cross_model_baseline:
        # Load random SAEs and compute their alive masks from test data
        random_saes = {}
        random_alive = {}
        for name in model_names:
            info = models[name]
            rand_ckpt = info["ckpt"].parent / RANDOM_BASELINE_FILENAME
            if not rand_ckpt.exists():
                print(f"  Skipping {name}: no random baseline at {rand_ckpt}")
                continue
            rand_model, _, _ = load_sae_checkpoint(rand_ckpt)
            random_saes[name] = rand_model
            random_alive[name] = compute_alive_mask(
                rand_model, info["test_embs"]
            )
        print(f"Random SAEs loaded: {sorted(random_saes.keys())}")

        pairs = {}
        for name_a in model_names:
            for name_b in model_names:
                if name_a == name_b:
                    continue
                if name_b not in random_saes:
                    continue
                pair_key = f"{name_a}__trained_vs_{name_b}__random"
                print(f"\n  {pair_key}")

                # Trained-A features vs Random-B features on shared datasets
                shared_ds = sorted(
                    set(models[name_a]["test_embs"].keys())
                    & set(models[name_b]["test_embs"].keys())
                )
                if not shared_ds:
                    print(f"    No shared datasets, skipping")
                    continue

                # Pool test activations
                all_acts_A = []
                all_acts_B = []
                for ds in shared_ds:
                    emb_A = models[name_a]["test_embs"][ds]
                    emb_B = models[name_b]["test_embs"][ds]
                    acts_A = compute_sae_activations(saes[name_a], emb_A)
                    acts_B = compute_sae_activations(random_saes[name_b], emb_B)
                    n = min(len(acts_A), len(acts_B))
                    all_acts_A.append(acts_A[:n])
                    all_acts_B.append(acts_B[:n])

                pooled_A = np.concatenate(all_acts_A, axis=0)
                pooled_B = np.concatenate(all_acts_B, axis=0)

                # Use training-derived alive masks
                alive_A = alive_masks[name_a]
                alive_B = random_alive[name_b]

                if alive_A.sum() == 0 or alive_B.sum() == 0:
                    print(f"    No alive features, skipping")
                    continue

                corr = compute_cross_correlation(
                    pooled_A[:, alive_A], pooled_B[:, alive_B]
                )

                # For each trained feature, its max |r| across random features
                max_r = corr.max(axis=1)
                pairs[pair_key] = {
                    "n": int(len(max_r)),
                    "mean": float(np.mean(max_r)),
                    "median": float(np.median(max_r)),
                    "p90": float(np.percentile(max_r, 90)),
                    "p95": float(np.percentile(max_r, 95)),
                }
                print(f"    n={len(max_r)}, mean={np.mean(max_r):.3f}, "
                      f"p90={np.percentile(max_r, 90):.3f}, "
                      f"p95={np.percentile(max_r, 95):.3f}")

        output = {
            "n_datasets": len(datasets),
            "split": "test",
            "pairs": pairs,
        }
        if args.output == (
            f"output/sae_feature_matching"
            f"_{args.method}"
            f"_t{args.alive_threshold}"
            f".json"
        ):
            args.output = "output/sae_cross_model_random_baseline.json"

        out_path = PROJECT_ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(convert_keys_to_native(output), f, indent=2, cls=NumpyEncoder)
        print(f"\nSaved to {out_path}")
        return

    # Match all pairs
    pairs = {}
    summary = {}
    for name_a, name_b in combinations(model_names, 2):
        pair_key = f"{name_a}__{name_b}"
        print(f"\nMatching: {pair_key}")
        result = match_model_pair(
            model_A=saes[name_a],
            model_B=saes[name_b],
            datasets=datasets,
            test_embs_A=models[name_a]["test_embs"],
            test_embs_B=models[name_b]["test_embs"],
            alive_mask_A=alive_masks[name_a],
            alive_mask_B=alive_masks[name_b],
            method=args.method,
            return_corr_matrix=args.save_correlations,
        )

        # Save correlation matrix if requested
        if corr_dir is not None and "corr_matrix" in result:
            npz_path = corr_dir / f"{pair_key}.npz"
            np.savez_compressed(
                npz_path,
                corr_matrix=result["corr_matrix"],
                indices_a=result["indices_a"],
                indices_b=result["indices_b"],
                model_a=name_a,
                model_b=name_b,
            )
            print(f"  Saved correlation matrix: {npz_path} "
                  f"({result['corr_matrix'].shape[0]}×{result['corr_matrix'].shape[1]})")
            del result["corr_matrix"]

        # Strip numpy arrays before JSON serialization
        result.pop("indices_a", None)
        result.pop("indices_b", None)

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
            "split": "test",
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
