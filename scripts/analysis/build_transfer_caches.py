"""Build virtual-atom caches for transfer experiments.

For each of the 15 model pairs × 2 directions (source→target and target→source)
× 2 SAE conditions (trained, random) × 3 map variants (global ridge, per-pair
CV top-K, per-query CV min-based with K_min=10), compute virtual atoms for every
unmatched source feature and save to disk with full provenance metadata.

Output layout:
  output/transfer_caches/{variant}_{sae_condition}/{source}_to_{target}.npz

The caches are consumed downstream by transfer_sweep_v2.py via a
--virtual-atoms-cache-dir flag that skips the landmark filter + map fit and
jumps straight to the per-row intervention loop.

Landmark source: matches whatever get_matched_pairs / get_unmatched_features
default to (currently sae_feature_matching_mnn_floor_p90.json).

Run: python -m scripts.analysis.build_transfer_caches [--workers 6]
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from multiprocessing import get_context
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.linear_model import Ridge

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import load_sae
from scripts.intervention.transfer_sweep_v2 import (
    DEFAULT_MATCHING_FILE,
    get_matched_pairs,
    get_unmatched_features,
)
from scripts.intervention.transfer_virtual_nodes import (
    extract_decoder_atoms,
    filter_landmarks,
    fit_concept_map,
)
from scripts.analysis.compare_transfer_maps import (
    build_virtual_atoms_per_query_cv,
    build_virtual_atoms_topk,
    sha256_file,
    tune_k_for_pair,
)

MODELS = ["carte", "mitra", "tabdpt", "tabicl", "tabicl_v2", "tabpfn"]
PAIRS = [tuple(sorted(p)) for p in combinations(MODELS, 2)]

SAE_DIRS = {
    "trained": PROJECT_ROOT / "output" / "sae_tabarena_sweep_round10",
    "random": PROJECT_ROOT / "output" / "sae_random_baseline",
}

OUT_ROOT = PROJECT_ROOT / "output" / "transfer_caches"

TOPK_GRID = [3, 5, 10, 15, 20, 30, 50]  # K grid for per-pair CV tuning
PQCV_GRID = [10, 15, 20, 30, 50, 100]   # K grid for per-query CV


def build_global_ridge_virtual_atoms(
    atoms_src_full: np.ndarray,
    filt_src: np.ndarray,
    filt_tgt: np.ndarray,
    unmatched_src_ids: list,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match production semantics (transfer_sweep_v2.py:367-397):
    fit ridge on raw (non-unit) landmarks, project unit query through M,
    renormalize direction, apply magnitude recipe.
    """
    # fit_concept_map fits on the raw (non-unit) atoms
    M_global, _ = fit_concept_map(filt_src, filt_tgt, alpha=alpha)

    src_norms_matched = np.linalg.norm(filt_src, axis=1)
    tgt_norms_matched = np.linalg.norm(filt_tgt, axis=1)
    valid = (src_norms_matched > 1e-8) & (tgt_norms_matched > 1e-8)
    if valid.sum() > 0:
        ratios = tgt_norms_matched[valid] / src_norms_matched[valid]
        median_norm_ratio = float(np.median(ratios))
    else:
        median_norm_ratio = 1.0

    n_unm = len(unmatched_src_ids)
    d_tgt = filt_tgt.shape[1]
    vatoms = np.zeros((n_unm, d_tgt), dtype=np.float32)
    computed = np.zeros(n_unm, dtype=bool)

    for out_idx, fi in enumerate(unmatched_src_ids):
        atom_s = atoms_src_full[fi]
        atom_norm = float(np.linalg.norm(atom_s))
        if atom_norm < 1e-8:
            continue
        unit_atom_s = atom_s / atom_norm
        virtual_dir = unit_atom_s @ M_global.T
        vdir_norm = float(np.linalg.norm(virtual_dir))
        if vdir_norm < 1e-8:
            continue
        vatoms[out_idx] = ((virtual_dir / vdir_norm) * atom_norm * median_norm_ratio).astype(np.float32)
        computed[out_idx] = True
    return vatoms, computed


def save_cache(
    variant: str,
    sae_condition: str,
    source: str,
    target: str,
    vatoms: np.ndarray,
    computed: np.ndarray,
    unmatched_ids: list,
    n_landmarks: int,
    d_target: int,
    matching_file: Path,
    matching_sha: str,
    map_params: dict,
) -> Path:
    out_dir = OUT_ROOT / f"{variant}_{sae_condition}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{source}_to_{target}.npz"
    np.savez_compressed(
        path,
        virtual_atoms=vatoms,
        feature_ids=np.array(unmatched_ids, dtype=np.int64),
        computed_mask=computed,
        variant=np.array(variant),
        sae_condition=np.array(sae_condition),
        source_model=np.array(source),
        target_model=np.array(target),
        n_landmarks=np.int64(n_landmarks),
        n_unmatched=np.int64(len(unmatched_ids)),
        n_computed=np.int64(int(computed.sum())),
        d_target=np.int64(d_target),
        ridge_alpha=np.float32(1.0),
        min_cosine=np.float32(0.0),
        matching_file=np.array(str(matching_file)),
        matching_file_sha256=np.array(matching_sha),
        map_params_json=np.array(json.dumps(map_params)),
    )
    return path


def process_pair_direction(task: tuple) -> tuple:
    """Worker function: build all 3 map variants for one (pair, direction, sae_condition)."""
    pair_models, source, target, sae_condition, matching_file = task
    matching_file = Path(matching_file)
    sae_dir = SAE_DIRS[sae_condition]

    try:
        sae_s, _ = load_sae(source, sae_dir=sae_dir, device="cpu")
        sae_t, _ = load_sae(target, sae_dir=sae_dir, device="cpu")
    except Exception as e:
        return (source, target, sae_condition, f"SAE load failed: {e}")

    atoms_s = extract_decoder_atoms(sae_s).numpy()
    atoms_t = extract_decoder_atoms(sae_t).numpy()

    try:
        m_pairs = get_matched_pairs(source, target, matching_file=matching_file)
    except Exception as e:
        return (source, target, sae_condition, f"get_matched_pairs failed: {e}")

    if len(m_pairs) < 10:
        return (source, target, sae_condition, f"too few matched pairs: {len(m_pairs)}")

    src_idx = [p[0] for p in m_pairs]
    tgt_idx = [p[1] for p in m_pairs]
    matched_src = atoms_s[src_idx]
    matched_tgt = atoms_t[tgt_idx]

    filt_src, filt_tgt, _, _ = filter_landmarks(
        matched_src, matched_tgt, m_pairs, min_cosine=0.0, alpha=1.0
    )
    n = len(filt_src)
    if n < 12:
        return (source, target, sae_condition, f"after filter only {n} landmarks")

    unmatched_src_ids = get_unmatched_features(
        source, target, matching_file=matching_file
    )
    matching_sha = sha256_file(matching_file)

    # Variant 1: Global ridge (matches production semantics)
    vatoms_g, computed_g = build_global_ridge_virtual_atoms(
        atoms_s, filt_src, filt_tgt, unmatched_src_ids
    )
    save_cache(
        "global", sae_condition, source, target, vatoms_g, computed_g,
        unmatched_src_ids, n, filt_tgt.shape[1], matching_file, matching_sha,
        map_params={"method": "global_ridge", "K": "all", "alpha": 1.0},
    )

    # Variant 2: Per-pair CV-tuned top-K
    tune_result = tune_k_for_pair(filt_src, filt_tgt, TOPK_GRID)
    k_star = int(tune_result["best_k"])
    vatoms_t, computed_t = build_virtual_atoms_topk(
        atoms_s, filt_src, filt_tgt, unmatched_src_ids, k=k_star
    )
    save_cache(
        "topk", sae_condition, source, target, vatoms_t, computed_t,
        unmatched_src_ids, n, filt_tgt.shape[1], matching_file, matching_sha,
        map_params={"method": "topk_per_pair", "K": k_star, "k_grid": TOPK_GRID,
                    "alpha": 1.0, "best_mean_cosine": tune_result["best_mean_cosine"]},
    )

    # Variant 3: Per-query CV min-based with K_min=10
    vatoms_p, computed_p, k_chosen = build_virtual_atoms_per_query_cv(
        atoms_s, filt_src, filt_tgt, unmatched_src_ids, PQCV_GRID
    )
    save_cache(
        "pqcv_k10", sae_condition, source, target, vatoms_p, computed_p,
        unmatched_src_ids, n, filt_tgt.shape[1], matching_file, matching_sha,
        map_params={"method": "per_query_cv_min_k10", "k_grid": PQCV_GRID,
                    "alpha": 1.0,
                    "k_chosen_per_concept_mean": float(k_chosen.mean()) if len(k_chosen) else 0.0},
    )

    return (source, target, sae_condition,
            f"ok (n_landmarks={n}, n_unmatched={len(unmatched_src_ids)}, "
            f"global_computed={int(computed_g.sum())}, topk_K*={k_star}, "
            f"topk_computed={int(computed_t.sum())}, pqcv_computed={int(computed_p.sum())})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pair", nargs=2, metavar=("A", "B"), default=None,
                        help="Build only this pair (sorted alphabetically).")
    parser.add_argument("--sae-condition", default=None,
                        choices=["trained", "random"],
                        help="Build only one SAE condition.")
    args = parser.parse_args()

    pairs_to_run = [tuple(sorted(args.pair))] if args.pair else PAIRS
    conditions = [args.sae_condition] if args.sae_condition else ["trained", "random"]

    tasks = []
    for pair in pairs_to_run:
        a, b = pair
        for src, tgt in [(a, b), (b, a)]:
            for cond in conditions:
                tasks.append((pair, src, tgt, cond, DEFAULT_MATCHING_FILE))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Dispatching {len(tasks)} build tasks to {args.workers} workers...")
    print(f"Output root: {OUT_ROOT}")

    results = []
    if args.workers > 1 and len(tasks) > 1:
        ctx = get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for result in pool.imap_unordered(process_pair_direction, tasks):
                src, tgt, cond, status = result
                print(f"  [{cond:<8}] {src}_to_{tgt}: {status}", flush=True)
                results.append(result)
    else:
        for task in tasks:
            result = process_pair_direction(task)
            src, tgt, cond, status = result
            print(f"  [{cond:<8}] {src}_to_{tgt}: {status}", flush=True)
            results.append(result)

    n_ok = sum(1 for r in results if r[3].startswith("ok"))
    print(f"\nDone: {n_ok}/{len(results)} tasks succeeded.")

    # Write a manifest
    manifest = {
        "n_tasks": len(results),
        "n_ok": n_ok,
        "results": [
            {"source": r[0], "target": r[1], "sae_condition": r[2], "status": r[3]}
            for r in results
        ],
    }
    manifest_path = OUT_ROOT / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
