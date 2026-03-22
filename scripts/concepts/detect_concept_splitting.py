#!/usr/bin/env python3
"""
Concept splitting detection for many-to-one SAE feature matches.

For each M2O fan-out group (multiple A features → one B target), fits Ridge
regression to determine whether the group genuinely reconstructs the target
(split), only one member matters (single_match), or there is no real
correspondence (noise).

Usage:
    python scripts/detect_concept_splitting.py \
        --matching output/sae_feature_matching_tiered_m2o_t0.001_n500.json \
        --output output/concept_splitting_results.json
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import cross_val_score

from scripts._project_root import PROJECT_ROOT

from scripts.sae.analyze_sae_concepts_deep import (
    NumpyEncoder,
    convert_keys_to_native,
    load_sae_checkpoint,
)
from scripts.sae.compare_sae_cross_model import (
    DEFAULT_MODELS,
    DEFAULT_SAE_ROUND,
    SAE_FILENAME,
    find_common_datasets,
    sae_sweep_dir,
)
from scripts.matching.01_match_sae_concepts_mnn import (
    EMB_DIR,
    compute_sae_activations,
    get_alive_mask,
    load_embeddings,
)


# ── Activation pooling ────────────────────────────────────────────────────


def pool_model_activations(
    sae_path: Path,
    emb_dir: Path,
    datasets: List[str],
    max_per_dataset: int = 500,
    alive_threshold: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load SAE and compute pooled activations across all datasets.

    Returns:
        pooled_acts: (n_samples, hidden_dim) float32 activations
        alive_mask: (hidden_dim,) boolean mask of alive features
    """
    model, config, _ = load_sae_checkpoint(sae_path)

    all_acts = []
    for ds in datasets:
        try:
            emb = load_embeddings(emb_dir, ds, max_per_dataset)
        except FileNotFoundError:
            continue
        acts = compute_sae_activations(model, emb)
        all_acts.append(acts)

    pooled = np.concatenate(all_acts, axis=0)
    alive = get_alive_mask(pooled, alive_threshold)
    return pooled, alive


# ── Grouping ──────────────────────────────────────────────────────────────


def group_matches_by_target(matches: List[dict]) -> Dict[int, List[dict]]:
    """
    Group A-features by their B target from M2O match results.

    Returns dict mapping idx_b → list of match dicts, only for groups with
    2+ members (singletons are excluded).
    """
    groups = defaultdict(list)
    for m in matches:
        groups[m["idx_b"]].append(m)

    return {k: v for k, v in groups.items() if len(v) >= 2}


# ── Core splitting test ──────────────────────────────────────────────────


def evaluate_single_group(
    pooled_A: np.ndarray,
    pooled_B: np.ndarray,
    group_indices_a: List[int],
    target_idx_b: int,
    match_r_by_idx: Optional[Dict[int, float]] = None,
    n_folds: int = 5,
    alphas: Optional[np.ndarray] = None,
    max_group_size: int = 200,
    min_group_r2: float = 0.3,
    min_delta_r2: float = 0.05,
    regression: str = "ridge",
) -> dict:
    """
    Test whether a fan-out group represents genuine concept splitting.

    Args:
        pooled_A: (n_samples, dim_A) full activation matrix for model A
        pooled_B: (n_samples, dim_B) full activation matrix for model B
        group_indices_a: Column indices in pooled_A for the group members
        target_idx_b: Column index in pooled_B for the target feature
        match_r_by_idx: Optional {idx_a: match_r} for ordering when capping
        n_folds: CV folds for regression
        alphas: Regularization grid (Ridge) or ignored (LassoCV auto-selects)
        max_group_size: Cap group to this many members (top by match_r)
        min_group_r2: Threshold for meaningful reconstruction
        min_delta_r2: Threshold for split vs single_match
        regression: "ridge" (L2, uses all features) or "lasso" (L1, sparse selection)

    Returns:
        Dict with group_r2, best_individual_r2, best_individual_idx,
        delta_r2, n_members, classification, members.
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 10)

    y = pooled_B[:, target_idx_b]

    # LassoCV needs float64 — sklearn's precomputed Gram matrix validation
    # fails with float32 precision
    if regression == "lasso":
        y = y.astype(np.float64)

    # Guard: constant target → noise
    if np.std(y) < 1e-10:
        return {
            "group_r2": 0.0,
            "best_individual_r2": 0.0,
            "best_individual_idx": group_indices_a[0] if group_indices_a else -1,
            "delta_r2": 0.0,
            "n_members": len(group_indices_a),
            "classification": "noise",
            "members": [],
        }

    # Cap large groups: keep top members by match_r
    indices = list(group_indices_a)
    if len(indices) > max_group_size:
        if match_r_by_idx:
            indices.sort(key=lambda idx: -match_r_by_idx.get(idx, 0.0))
        indices = indices[:max_group_size]

    def _make_model(n_features: int):
        if regression == "lasso":
            return LassoCV(cv=n_folds, max_iter=5000, n_jobs=1)
        return RidgeCV(alphas=alphas)

    # Individual R² per member (cross-validated)
    individual_r2s = {}
    for idx in indices:
        X_single = pooled_A[:, idx : idx + 1]
        if regression == "lasso":
            X_single = X_single.astype(np.float64)
        scores = cross_val_score(
            _make_model(1), X_single, y, cv=n_folds, scoring="r2"
        )
        individual_r2s[idx] = max(float(np.mean(scores)), 0.0)

    best_idx = max(individual_r2s, key=individual_r2s.get)
    best_individual_r2 = individual_r2s[best_idx]

    # Group R² (cross-validated)
    X_group = pooled_A[:, indices]
    if regression == "lasso":
        X_group = X_group.astype(np.float64)
    group_scores = cross_val_score(
        _make_model(len(indices)), X_group, y, cv=n_folds, scoring="r2"
    )
    group_r2 = max(float(np.mean(group_scores)), 0.0)

    delta_r2 = group_r2 - best_individual_r2

    # Classify
    if group_r2 >= min_group_r2 and delta_r2 >= min_delta_r2:
        classification = "split"
    elif group_r2 >= min_group_r2:
        classification = "single_match"
    else:
        classification = "noise"

    members = []
    for idx in indices:
        members.append({
            "idx_a": int(idx),
            "individual_r2": round(individual_r2s[idx], 4),
            "match_r": round(match_r_by_idx.get(idx, 0.0), 4) if match_r_by_idx else 0.0,
        })

    return {
        "group_r2": round(group_r2, 4),
        "best_individual_r2": round(best_individual_r2, 4),
        "best_individual_idx": int(best_idx),
        "delta_r2": round(delta_r2, 4),
        "n_members": len(indices),
        "classification": classification,
        "members": members,
    }


# ── Pair-level test ───────────────────────────────────────────────────────


def evaluate_pair_splitting(
    pooled_A: np.ndarray,
    pooled_B: np.ndarray,
    matches: List[dict],
    min_group_r2: float = 0.3,
    min_delta_r2: float = 0.05,
    max_group_size: int = 200,
    n_folds: int = 5,
    regression: str = "ridge",
) -> dict:
    """
    Test all fan-out groups for one model pair.

    Args:
        pooled_A: (n_samples, dim_A) activations — full hidden dim
        pooled_B: (n_samples, dim_B) activations — full hidden dim
        matches: List of match dicts with idx_a, idx_b, r, tier fields

    Returns:
        Dict with n_groups_tested, n_split, n_single_match, n_noise, groups.
    """
    groups = group_matches_by_target(matches)

    if not groups:
        return {
            "n_groups_tested": 0,
            "n_split": 0,
            "n_single_match": 0,
            "n_noise": 0,
            "groups": [],
        }

    results = []
    for target_b, group_matches in groups.items():
        group_indices_a = [m["idx_a"] for m in group_matches]
        match_r_by_idx = {m["idx_a"]: m["r"] for m in group_matches}

        result = evaluate_single_group(
            pooled_A,
            pooled_B,
            group_indices_a,
            target_b,
            match_r_by_idx=match_r_by_idx,
            n_folds=n_folds,
            max_group_size=max_group_size,
            min_group_r2=min_group_r2,
            min_delta_r2=min_delta_r2,
            regression=regression,
        )
        result["target_idx_b"] = int(target_b)
        results.append(result)

    n_split = sum(1 for r in results if r["classification"] == "split")
    n_single = sum(1 for r in results if r["classification"] == "single_match")
    n_noise = sum(1 for r in results if r["classification"] == "noise")

    return {
        "n_groups_tested": len(results),
        "n_split": n_split,
        "n_single_match": n_single,
        "n_noise": n_noise,
        "groups": results,
    }


# ── CLI ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Detect concept splitting in M2O feature matches"
    )
    parser.add_argument(
        "--matching",
        type=str,
        default="output/sae_feature_matching_tiered_m2o_t0.001_n500.json",
        help="M2O matching results JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/concept_splitting_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--min-group-r2",
        type=float,
        default=0.3,
        help="Min group R² for real correspondence (default: 0.3)",
    )
    parser.add_argument(
        "--min-delta-r2",
        type=float,
        default=0.05,
        help="Min R² improvement for split vs single_match (default: 0.05)",
    )
    parser.add_argument(
        "--max-group-size",
        type=int,
        default=200,
        help="Cap group members for regression (default: 200)",
    )
    parser.add_argument(
        "--regression",
        choices=["ridge", "lasso"],
        default="ridge",
        help="Regression method: ridge (L2, all features) or lasso (L1, sparse)",
    )
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=500,
        help="Max samples per dataset (default: 500)",
    )
    parser.add_argument(
        "--alive-threshold",
        type=float,
        default=0.001,
        help="Alive feature threshold (default: 0.001)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help=f"SAE sweep round (default: {DEFAULT_SAE_ROUND})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model keys",
    )
    args = parser.parse_args()

    t_start = time.time()

    # Load M2O matching results
    matching_path = PROJECT_ROOT / args.matching
    with open(matching_path) as f:
        matching = json.load(f)
    print(f"Matching: {len(matching['pairs'])} pairs, "
          f"method={matching['metadata']['method']}")

    sweep_dir = sae_sweep_dir(args.round)

    # Discover models (same pattern as match_sae_features.py)
    model_info = {}
    for display_name, model_key, emb_key in DEFAULT_MODELS:
        if args.models and model_key not in args.models:
            continue
        ckpt = sweep_dir / model_key / SAE_FILENAME
        if not ckpt.exists():
            continue
        emb_dir = EMB_DIR / emb_key
        if not emb_dir.exists():
            continue
        model_info[display_name] = {
            "ckpt": ckpt,
            "emb_dir": emb_dir,
        }

    # Determine which models appear in matching pairs
    needed_models = set()
    for pair_key in matching["pairs"]:
        a, b = pair_key.split("__")
        needed_models.add(a)
        needed_models.add(b)
    needed_models &= set(model_info.keys())
    print(f"Models needed: {sorted(needed_models)}")

    # Find common datasets
    emb_dirs = {name: info["emb_dir"] for name, info in model_info.items()
                if name in needed_models}
    datasets = find_common_datasets(emb_dirs)
    print(f"Common datasets: {len(datasets)}")

    # Phase 1: Cache pooled activations per model
    print("\n── Phase 1: Caching pooled activations ──")
    activation_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for i, name in enumerate(sorted(needed_models), 1):
        print(f"  [{i}/{len(needed_models)}] {name}...", end=" ", flush=True)
        pooled, alive = pool_model_activations(
            sae_path=model_info[name]["ckpt"],
            emb_dir=model_info[name]["emb_dir"],
            datasets=datasets,
            max_per_dataset=args.max_per_dataset,
            alive_threshold=args.alive_threshold,
        )
        activation_cache[name] = (pooled, alive)
        print(f"({pooled.shape[0]} samples, {alive.sum()} alive of {pooled.shape[1]})")

    # Phase 2: Splitting tests per pair
    print("\n── Phase 2: Splitting tests ──")
    pair_results = {}
    for pair_key, pair_data in sorted(matching["pairs"].items()):
        model_a, model_b = pair_key.split("__")
        if model_a not in activation_cache or model_b not in activation_cache:
            print(f"  Skipping {pair_key}: missing activations")
            continue

        pooled_A, alive_A = activation_cache[model_a]
        pooled_B, alive_B = activation_cache[model_b]

        # Align sample count (same as match_model_pair)
        n = min(len(pooled_A), len(pooled_B))
        pooled_A_aligned = pooled_A[:n]
        pooled_B_aligned = pooled_B[:n]

        matches = pair_data["matches"]
        n_groups = len(group_matches_by_target(matches))
        print(f"  {pair_key}: {n_groups} groups to test...", end=" ", flush=True)

        result = evaluate_pair_splitting(
            pooled_A_aligned,
            pooled_B_aligned,
            matches,
            min_group_r2=args.min_group_r2,
            min_delta_r2=args.min_delta_r2,
            max_group_size=args.max_group_size,
            regression=args.regression,
        )
        pair_results[pair_key] = result
        print(f"split={result['n_split']}, single={result['n_single_match']}, "
              f"noise={result['n_noise']}")

    # Global summary
    total_groups = sum(r["n_groups_tested"] for r in pair_results.values())
    total_split = sum(r["n_split"] for r in pair_results.values())
    total_single = sum(r["n_single_match"] for r in pair_results.values())
    total_noise = sum(r["n_noise"] for r in pair_results.values())

    runtime = time.time() - t_start

    output = {
        "metadata": {
            "matching_file": args.matching,
            "regression": args.regression,
            "min_group_r2": args.min_group_r2,
            "min_delta_r2": args.min_delta_r2,
            "max_group_size": args.max_group_size,
            "n_folds": 5,
            "n_models": len(needed_models),
            "n_pairs": len(pair_results),
            "n_datasets": len(datasets),
            "runtime_seconds": round(runtime, 1),
        },
        "pairs": pair_results,
        "global_summary": {
            "total_groups_tested": total_groups,
            "total_split": total_split,
            "total_single_match": total_single,
            "total_noise": total_noise,
        },
    }

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(convert_keys_to_native(output), f, indent=2, cls=NumpyEncoder)

    print(f"\n{'='*60}")
    print(f"Concept splitting detection complete ({runtime:.0f}s)")
    print(f"  Groups tested: {total_groups}")
    print(f"  Split: {total_split} ({100*total_split/max(total_groups,1):.1f}%)")
    print(f"  Single match: {total_single} ({100*total_single/max(total_groups,1):.1f}%)")
    print(f"  Noise: {total_noise} ({100*total_noise/max(total_groups,1):.1f}%)")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
