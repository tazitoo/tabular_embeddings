"""Compare transfer map quality: global ridge vs adaptive local ridge vs MLP.

Fits each map family on the filtered landmarks of every model pair and
evaluates on a held-out split by cosine similarity between predicted target
atoms and ground-truth target atoms.

Rationale: R² of a concept map factors into direction (cosine) + magnitude.
Cosine isolates the "did the map learn the right direction" question, which
is the operative question for transfer (we renormalize to per-atom magnitude
downstream anyway).

Output:
  output/figures/appendix/transfer_map_comparison.{json,pdf}

Run: python -m scripts.analysis.compare_transfer_maps
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import load_sae
from scripts.intervention.transfer_sweep_v2 import get_matched_pairs
from scripts.intervention.transfer_virtual_nodes import (
    extract_decoder_atoms,
    filter_landmarks,
    fit_concept_map,
    fit_concept_map_mlp,
)

MODELS = ["carte", "mitra", "tabdpt", "tabicl", "tabicl_v2", "tabpfn"]
PAIRS = [tuple(sorted(p)) for p in combinations(MODELS, 2)]

MODEL_TO_MATCHING_KEY = {
    "carte": "CARTE", "mitra": "Mitra", "tabdpt": "TabDPT",
    "tabicl": "TabICL", "tabicl_v2": "TabICL-v2", "tabpfn": "TabPFN",
}

MATCHING_FILE_DEFAULT = PROJECT_ROOT / "output" / "sae_feature_matching_mnn_floor_p90.json"

OUT_DIR = PROJECT_ROOT / "output" / "figures" / "appendix"
OUT_JSON = OUT_DIR / "transfer_map_comparison.json"
OUT_PDF = OUT_DIR / "transfer_map_comparison.pdf"

VIRTUAL_ATOMS_CACHE_DIR = PROJECT_ROOT / "output" / "virtual_atoms_adaptive_k_tuned"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

VAL_FRAC = 0.2
SEED = 42
COSINE_THRESHOLD = 0.40  # matches adaptive_local 2*noise_floor in transfer_sweep_v2


_MNN_CACHE: dict = {}


def load_mnn_pairs(source: str, target: str, matching_path: Path) -> list:
    """Return pairwise MNN matches as a list of (src_idx, tgt_idx) tuples.

    The matching JSON stores pairs under keys like 'CARTE__Mitra', with
    idx_a/idx_b corresponding to the order in the key name. We preserve
    the source→target orientation requested by the caller.
    """
    if matching_path not in _MNN_CACHE:
        with open(matching_path) as f:
            _MNN_CACHE[matching_path] = json.load(f)
    d = _MNN_CACHE[matching_path]
    pairs_dict = d["pairs"]

    key_s = MODEL_TO_MATCHING_KEY[source]
    key_t = MODEL_TO_MATCHING_KEY[target]
    fwd_key = f"{key_s}__{key_t}"
    rev_key = f"{key_t}__{key_s}"

    if fwd_key in pairs_dict:
        entry = pairs_dict[fwd_key]
        return [(m["idx_a"], m["idx_b"]) for m in entry["matches"]]
    elif rev_key in pairs_dict:
        entry = pairs_dict[rev_key]
        # File has target first; swap so caller gets (source_idx, target_idx).
        return [(m["idx_b"], m["idx_a"]) for m in entry["matches"]]
    else:
        return []


def cosine_row(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def adaptive_local_predict(
    train_src: np.ndarray,
    train_tgt: np.ndarray,
    val_src: np.ndarray,
    cosine_threshold: float = COSINE_THRESHOLD,
    alpha: float = 1.0,
) -> np.ndarray:
    """Predict val targets by fitting a local ridge on train neighbors.

    For each val source atom, pick train landmarks whose normalized source
    has |cosine| >= threshold to the query, fit ridge on that neighborhood,
    and predict. Falls back to global ridge when neighborhood < 3.
    """
    # Normalize
    src_norms = np.linalg.norm(train_src, axis=1, keepdims=True)
    tgt_norms = np.linalg.norm(train_tgt, axis=1, keepdims=True)
    src_norms[src_norms < 1e-8] = 1.0
    tgt_norms[tgt_norms < 1e-8] = 1.0
    train_src_unit = train_src / src_norms
    train_tgt_unit = train_tgt / tgt_norms

    # Global fallback fit (used when the local neighborhood is too small).
    fallback = Ridge(alpha=alpha, fit_intercept=False)
    fallback.fit(train_src_unit, train_tgt_unit)

    preds = np.zeros((len(val_src), train_tgt.shape[1]))
    for i, q in enumerate(val_src):
        qn = np.linalg.norm(q)
        if qn < 1e-8:
            preds[i] = 0.0
            continue
        query_unit = (q / qn).reshape(1, -1)
        sims = cosine_similarity(query_unit, train_src_unit)[0]
        mask = np.abs(sims) >= cosine_threshold
        K = int(mask.sum())
        if K < 3:
            preds[i] = fallback.predict(query_unit)[0]
            continue
        idx = np.where(mask)[0]
        reg = Ridge(alpha=alpha, fit_intercept=False)
        reg.fit(train_src_unit[idx], train_tgt_unit[idx])
        preds[i] = reg.predict(query_unit)[0]
    return preds


def loo_predict_ridge(filt_src: np.ndarray, filt_tgt: np.ndarray) -> np.ndarray:
    """LOO: for each i, fit global ridge on others and predict i."""
    n = len(filt_src)
    preds = np.zeros_like(filt_tgt)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        M, _ = fit_concept_map(filt_src[mask], filt_tgt[mask], alpha=1.0)
        preds[i] = filt_src[i] @ M.T
    return preds


def loo_predict_adaptive(
    filt_src: np.ndarray,
    filt_tgt: np.ndarray,
    cosine_threshold: float = COSINE_THRESHOLD,
) -> np.ndarray:
    """LOO: for each i, fit adaptive local ridge on others and predict i."""
    n = len(filt_src)
    preds = np.zeros_like(filt_tgt)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        others_src = filt_src[mask]
        others_tgt = filt_tgt[mask]
        preds[i:i+1] = adaptive_local_predict(
            others_src, others_tgt, filt_src[i:i+1], cosine_threshold=cosine_threshold
        )
    return preds


def topk_local_predict(
    train_src: np.ndarray,
    train_tgt: np.ndarray,
    val_src: np.ndarray,
    k: int,
    alpha: float = 1.0,
) -> np.ndarray:
    """Predict val targets using top-K nearest-by-|cosine| train landmarks.

    For each val source vector, rank train sources by |cos| to the query,
    take the top K, fit a ridge on those K unit-normalized pairs, predict.
    """
    src_norms = np.linalg.norm(train_src, axis=1, keepdims=True)
    tgt_norms = np.linalg.norm(train_tgt, axis=1, keepdims=True)
    src_norms[src_norms < 1e-8] = 1.0
    tgt_norms[tgt_norms < 1e-8] = 1.0
    train_src_unit = train_src / src_norms
    train_tgt_unit = train_tgt / tgt_norms

    preds = np.zeros((len(val_src), train_tgt.shape[1]))
    for i, q in enumerate(val_src):
        qn = np.linalg.norm(q)
        if qn < 1e-8:
            continue
        query_unit = (q / qn).reshape(1, -1)
        sims = cosine_similarity(query_unit, train_src_unit)[0]
        order = np.argsort(-np.abs(sims))[:k]
        reg = Ridge(alpha=alpha, fit_intercept=False)
        reg.fit(train_src_unit[order], train_tgt_unit[order])
        preds[i] = reg.predict(query_unit)[0]
    return preds


def loo_predict_topk(
    filt_src: np.ndarray,
    filt_tgt: np.ndarray,
    k: int,
) -> np.ndarray:
    """LOO: for each i, fit top-K local ridge on the other N-1 and predict i."""
    n = len(filt_src)
    preds = np.zeros_like(filt_tgt)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        preds[i:i+1] = topk_local_predict(
            filt_src[mask], filt_tgt[mask], filt_src[i:i+1], k=k
        )
    return preds


def _pool_worker_unpack(task):
    """Top-level worker for multiprocessing.Pool (spawn-safe). Returns (key, result)."""
    (s, t), loo, include_mlp, landmark_source, matching_path, tune_k, tune_k_per_query = task
    try:
        r = run_pair(s, t, loo=loo, include_mlp=include_mlp,
                     landmark_source=landmark_source,
                     matching_path=matching_path,
                     tune_k=tune_k,
                     tune_k_per_query=tune_k_per_query)
    except Exception as e:
        r = {"skipped": True, "reason": f"exception: {e}"}
    return f"{s}_vs_{t}", r


def per_query_cv_predict(
    filt_src: np.ndarray,
    filt_tgt: np.ndarray,
    query: np.ndarray,
    k_grid: list,
    alpha: float = 1.0,
    return_info: bool = False,
):
    """Per-query CV with min-LOO selection and one-drop refit.

    For a single query vector:
      1. Rank landmarks by |cos| to query.
      2. For each K in grid, compute per-landmark LOO cosines in top-K.
      3. score(K) = min(LOO cosines) — is there a "really bad fit" in this K?
      4. K* = argmax_K score(K) — the most internally-consistent neighborhood.
      5. At K*, identify j_worst = argmin(LOO cosines).
      6. Refit ridge on K*-1 landmarks (dropping j_worst), predict query target.

    The min-as-selection-score prefers small K (fewer chances for an outlier),
    so callers should floor the grid at K_min big enough to keep ridge
    non-degenerate after the one-drop step.
    """
    qn = float(np.linalg.norm(query))
    if qn < 1e-8:
        out = np.zeros(filt_tgt.shape[1], dtype=np.float32)
        return (out, {"best_k": None, "scores": {}}) if return_info else out
    query_unit = (query / qn).reshape(1, -1)

    src_norms = np.linalg.norm(filt_src, axis=1, keepdims=True)
    tgt_norms = np.linalg.norm(filt_tgt, axis=1, keepdims=True)
    src_norms[src_norms < 1e-8] = 1.0
    tgt_norms[tgt_norms < 1e-8] = 1.0
    filt_src_unit = filt_src / src_norms
    filt_tgt_unit = filt_tgt / tgt_norms

    sims = cosine_similarity(query_unit, filt_src_unit)[0]
    order = np.argsort(-np.abs(sims))
    n = len(order)

    valid_k = sorted({max(5, min(k, n)) for k in k_grid})

    best_k = None
    best_score = -np.inf
    per_k_scores = {}
    per_k_loo = {}  # Store per-landmark LOO cosines at each K for the refit drop
    per_k_raw_norm_mean = {}  # Mean raw prediction norm in LOO sweep at each K

    for k in valid_k:
        top = order[:k]
        pool_src = filt_src_unit[top]
        pool_tgt = filt_tgt_unit[top]
        loo_cos = np.zeros(k)
        loo_raw_norms = np.zeros(k)
        for j in range(k):
            mask = np.ones(k, dtype=bool); mask[j] = False
            reg = Ridge(alpha=alpha, fit_intercept=False)
            reg.fit(pool_src[mask], pool_tgt[mask])
            pred = reg.predict(pool_src[j:j+1])[0]
            actual = pool_tgt[j]
            pn = float(np.linalg.norm(pred))
            loo_raw_norms[j] = pn
            an = float(np.linalg.norm(actual))
            if pn > 0 and an > 0:
                loo_cos[j] = float(np.dot(pred, actual) / (pn * an))
            else:
                loo_cos[j] = 0.0
        min_loo = float(loo_cos.min())
        per_k_scores[k] = min_loo
        per_k_loo[k] = loo_cos
        per_k_raw_norm_mean[k] = float(loo_raw_norms.mean())
        if min_loo > best_score:
            best_score = min_loo
            best_k = k

    # One-drop refit: drop the landmark with the worst LOO cosine at K*.
    top = order[:best_k]
    loo_at_best = per_k_loo[best_k]
    j_worst_in_top = int(loo_at_best.argmin())
    keep_mask = np.ones(best_k, dtype=bool); keep_mask[j_worst_in_top] = False
    refit_idx = top[keep_mask]

    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(filt_src_unit[refit_idx], filt_tgt_unit[refit_idx])
    direction = reg.predict(query_unit)[0]
    dir_norm = float(np.linalg.norm(direction))
    if dir_norm < 1e-8:
        out = np.zeros(filt_tgt.shape[1], dtype=np.float32)
    else:
        local_ratios = tgt_norms[refit_idx].ravel() / src_norms[refit_idx].ravel()
        scale = float(np.median(local_ratios)) * qn
        out = ((direction / dir_norm) * scale).astype(np.float32)

    if return_info:
        return out, {
            "best_k": int(best_k),
            "scores": per_k_scores,
            "best_score": best_score,
            "dropped_worst_loo": float(loo_at_best.min()),
            "final_raw_norm": dir_norm,
            "per_k_raw_norm_mean": per_k_raw_norm_mean,
        }
    return out


def loo_per_query_cv(filt_src: np.ndarray, filt_tgt: np.ndarray,
                     k_grid: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """LOO evaluation: for each landmark i, exclude it, run per_query_cv_predict
    against the other N-1. Returns predictions, chosen K per landmark,
    final raw-direction norms per landmark, and a per-K raw-norm aggregate."""
    n = len(filt_src)
    preds = np.zeros_like(filt_tgt, dtype=np.float32)
    k_chosen = np.zeros(n, dtype=np.int64)
    final_raw_norm = np.zeros(n, dtype=np.float32)
    per_k_raw_norms_agg: dict = {}  # k -> list of mean raw norms (one per query)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        out, info = per_query_cv_predict(
            filt_src[mask], filt_tgt[mask], filt_src[i], k_grid, return_info=True
        )
        preds[i] = out
        k_chosen[i] = info["best_k"] if info["best_k"] is not None else 0
        final_raw_norm[i] = info.get("final_raw_norm", 0.0)
        for k, v in info.get("per_k_raw_norm_mean", {}).items():
            per_k_raw_norms_agg.setdefault(int(k), []).append(v)
    per_k_raw_norm_summary = {
        int(k): {
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)),
            "n": len(v),
        }
        for k, v in per_k_raw_norms_agg.items()
    }
    return preds, k_chosen, final_raw_norm, per_k_raw_norm_summary


def build_virtual_atoms_per_query_cv(
    atoms_src_full: np.ndarray,
    filt_src: np.ndarray,
    filt_tgt: np.ndarray,
    unmatched_src_ids: list,
    k_grid: list,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-query-CV virtual atoms for real unmatched concepts.

    Returns (virtual_atoms, computed_mask, k_chosen_per_concept)."""
    d_tgt = filt_tgt.shape[1]
    n_unm = len(unmatched_src_ids)
    vatoms = np.zeros((n_unm, d_tgt), dtype=np.float32)
    computed = np.zeros(n_unm, dtype=bool)
    k_chosen = np.zeros(n_unm, dtype=np.int64)
    for out_idx, fi in enumerate(unmatched_src_ids):
        query = atoms_src_full[fi]
        if np.linalg.norm(query) < 1e-8:
            continue
        out, info = per_query_cv_predict(
            filt_src, filt_tgt, query, k_grid, alpha=alpha, return_info=True
        )
        if np.linalg.norm(out) > 1e-8:
            vatoms[out_idx] = out
            computed[out_idx] = True
            k_chosen[out_idx] = info["best_k"] if info["best_k"] is not None else 0
    return vatoms, computed, k_chosen


def build_virtual_atoms_topk(
    atoms_src_full: np.ndarray,
    filt_src: np.ndarray,
    filt_tgt: np.ndarray,
    unmatched_src_ids: list,
    k: int,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a virtual atom in target-model space for each unmatched source feature.

    Uses the top-K nearest (by |cos|) filtered landmarks as the local neighborhood,
    fits a ridge on their unit-normalized pairs, predicts a unit direction, and
    scales by median(target_norm / source_norm) * ||source_atom||.

    Returns:
        virtual_atoms: (n_unmatched, d_target) — zeros where computation was skipped
        computed_mask: (n_unmatched,) bool — True for rows that have a valid virtual atom
    """
    src_norms = np.linalg.norm(filt_src, axis=1, keepdims=True)
    tgt_norms = np.linalg.norm(filt_tgt, axis=1, keepdims=True)
    src_norms[src_norms < 1e-8] = 1.0
    tgt_norms[tgt_norms < 1e-8] = 1.0
    filt_src_unit = filt_src / src_norms
    filt_tgt_unit = filt_tgt / tgt_norms

    n_unm = len(unmatched_src_ids)
    d_tgt = filt_tgt.shape[1]
    vatoms = np.zeros((n_unm, d_tgt), dtype=np.float32)
    computed = np.zeros(n_unm, dtype=bool)

    for out_idx, fi in enumerate(unmatched_src_ids):
        atom_s = atoms_src_full[fi]
        norm_s = float(np.linalg.norm(atom_s))
        if norm_s < 1e-8:
            continue
        query_unit = (atom_s / norm_s).reshape(1, -1)
        sims = cosine_similarity(query_unit, filt_src_unit)[0]
        order = np.argsort(-np.abs(sims))[:k]
        reg = Ridge(alpha=alpha, fit_intercept=False)
        reg.fit(filt_src_unit[order], filt_tgt_unit[order])
        direction = reg.predict(query_unit)[0]
        dir_norm = float(np.linalg.norm(direction))
        if dir_norm < 1e-8:
            continue
        local_ratios = tgt_norms[order].ravel() / src_norms[order].ravel()
        scale = float(np.median(local_ratios)) * norm_s
        vatoms[out_idx] = (direction / dir_norm) * scale
        computed[out_idx] = True
    return vatoms, computed


def tune_k_for_pair(filt_src: np.ndarray, filt_tgt: np.ndarray,
                    k_grid: list) -> dict:
    """Sweep K over a grid, run LOO for each K, return best_k and per-K scores."""
    n = len(filt_src)
    valid_k = sorted({max(3, min(k, n - 1)) for k in k_grid})
    scores = {}
    for k in valid_k:
        preds = loo_predict_topk(filt_src, filt_tgt, k)
        cos_vals = [cosine_row(preds[i], filt_tgt[i]) for i in range(n)]
        scores[k] = {
            "mean": float(np.mean(cos_vals)),
            "median": float(np.median(cos_vals)),
            "n": n,
        }
    best_k = max(scores, key=lambda k: scores[k]["mean"])
    return {
        "k_grid": valid_k,
        "per_k": scores,
        "best_k": int(best_k),
        "best_mean_cosine": scores[best_k]["mean"],
    }


def run_pair(source: str, target: str, loo: bool, include_mlp: bool,
             landmark_source: str = "concept_groups",
             matching_path: Path = MATCHING_FILE_DEFAULT,
             tune_k: bool = False,
             tune_k_per_query: bool = False) -> dict:
    sae_s, _ = load_sae(source, device="cpu")
    sae_t, _ = load_sae(target, device="cpu")
    atoms_s = extract_decoder_atoms(sae_s).numpy()
    atoms_t = extract_decoder_atoms(sae_t).numpy()

    if landmark_source == "concept_groups":
        m_pairs = get_matched_pairs(source, target)
    elif landmark_source == "mnn_pairwise":
        m_pairs = load_mnn_pairs(source, target, matching_path)
    else:
        return {"skipped": True, "reason": f"unknown landmark_source={landmark_source}"}

    if len(m_pairs) < 10:
        return {"skipped": True, "reason": "too few matched pairs", "n_matched": len(m_pairs)}

    src_idx = [p[0] for p in m_pairs]
    tgt_idx = [p[1] for p in m_pairs]
    matched_src = atoms_s[src_idx]
    matched_tgt = atoms_t[tgt_idx]

    filt_src, filt_tgt, _, quality = filter_landmarks(
        matched_src, matched_tgt, m_pairs, min_cosine=0.0, alpha=1.0
    )
    n = len(filt_src)
    if n < 12:
        return {"skipped": True, "reason": f"after filter only {n} landmarks"}

    out = {
        "source": source, "target": target, "n_filtered": n,
        "landmark_mean_loo_cosine": float(quality.get("mean_cosine", 0.0)),
        "cosine": {},
        "eval_mode": {"ridge": "LOO" if loo else "80/20", "adaptive": "LOO" if loo else "80/20",
                      "mlp": "80/20"},
    }

    def cosines(preds: np.ndarray, tgts: np.ndarray) -> list[float]:
        return [cosine_row(preds[i], tgts[i]) for i in range(len(tgts))]

    if loo:
        global_pred = loo_predict_ridge(filt_src, filt_tgt)
        local_pred = loo_predict_adaptive(filt_src, filt_tgt)
        cos_global = cosines(global_pred, filt_tgt)
        cos_local = cosines(local_pred, filt_tgt)
        out["n_eval_ridge"] = n
        out["n_eval_adaptive"] = n
        out["cosine"]["global_ridge"] = {
            "mean": float(np.mean(cos_global)), "median": float(np.median(cos_global)),
            "raw": cos_global,
        }
        out["cosine"]["adaptive_local"] = {
            "mean": float(np.mean(cos_local)), "median": float(np.median(cos_local)),
            "raw": cos_local,
        }
    else:
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(n)
        n_val = max(2, int(n * VAL_FRAC))
        val_i, train_i = perm[:n_val], perm[n_val:]
        train_src, train_tgt = filt_src[train_i], filt_tgt[train_i]
        val_src, val_tgt = filt_src[val_i], filt_tgt[val_i]
        global_M, _ = fit_concept_map(train_src, train_tgt, alpha=1.0)
        global_pred = val_src @ global_M.T
        local_pred = adaptive_local_predict(train_src, train_tgt, val_src)
        cos_global = cosines(global_pred, val_tgt)
        cos_local = cosines(local_pred, val_tgt)
        out["n_eval_ridge"] = n_val
        out["n_eval_adaptive"] = n_val
        out["cosine"]["global_ridge"] = {
            "mean": float(np.mean(cos_global)), "median": float(np.median(cos_global)),
            "raw": cos_global,
        }
        out["cosine"]["adaptive_local"] = {
            "mean": float(np.mean(cos_local)), "median": float(np.median(cos_local)),
            "raw": cos_local,
        }

    if tune_k_per_query:
        # Min-LOO selection prefers small K. Floor at 5 (lowered from 10) so
        # we can probe whether queries that want very small neighborhoods
        # are running into ridge rank-deficiency. The per-K raw-norm
        # diagnostic below tells us if the small-K fits are healthy or
        # dominated by the L2 regularizer.
        k_grid_default = [5, 10, 15, 20, 30, 50, 100]
        preds_pq, k_chosen_pq, final_raw_norm, per_k_raw_norm_summary = \
            loo_per_query_cv(filt_src, filt_tgt, k_grid_default)
        cos_pq = [cosine_row(preds_pq[i], filt_tgt[i]) for i in range(n)]
        out["tune_k_per_query"] = {
            "k_grid": sorted({max(5, min(k, n)) for k in k_grid_default}),
            "k_chosen_per_landmark": k_chosen_pq.tolist(),
            "k_chosen_mean": float(k_chosen_pq.mean()),
            "k_chosen_median": float(np.median(k_chosen_pq)),
            "k_chosen_std": float(k_chosen_pq.std()),
            "loo_cosine_mean": float(np.mean(cos_pq)),
            "loo_cosine_median": float(np.median(cos_pq)),
            "final_raw_norm_mean": float(final_raw_norm.mean()),
            "final_raw_norm_median": float(np.median(final_raw_norm)),
            "per_k_raw_norm_summary": per_k_raw_norm_summary,
        }

        # Build virtual atoms for real unmatched concepts using per-query CV
        from scripts.intervention.ablation_sweep import get_unmatched_features
        unmatched_src_ids = get_unmatched_features(
            source, target, matching_file=matching_path
        )
        vatoms, computed, k_chosen_unm = build_virtual_atoms_per_query_cv(
            atoms_s, filt_src, filt_tgt, unmatched_src_ids, k_grid_default
        )
        VIRTUAL_ATOMS_CACHE_DIR_PQ = PROJECT_ROOT / "output" / "virtual_atoms_adaptive_per_query_cv"
        VIRTUAL_ATOMS_CACHE_DIR_PQ.mkdir(parents=True, exist_ok=True)
        cache_path_pq = VIRTUAL_ATOMS_CACHE_DIR_PQ / f"{source}_to_{target}.npz"
        np.savez_compressed(
            cache_path_pq,
            virtual_atoms=vatoms,
            feature_ids=np.array(unmatched_src_ids, dtype=np.int64),
            computed_mask=computed,
            k_chosen_per_concept=k_chosen_unm,
            source_model=np.array(source),
            target_model=np.array(target),
            n_landmarks=np.int64(len(filt_src)),
            n_unmatched=np.int64(len(unmatched_src_ids)),
            n_computed=np.int64(int(computed.sum())),
            d_target=np.int64(filt_tgt.shape[1]),
            ridge_alpha=np.float32(1.0),
            min_cosine=np.float32(0.0),
            neighborhood_mode=np.array("per_query_cv"),
            matching_file=np.array(str(matching_path)),
            matching_file_sha256=np.array(sha256_file(matching_path)),
            landmark_source=np.array(landmark_source),
        )
        out["virtual_atoms_cache_pq"] = {
            "path": str(cache_path_pq),
            "n_unmatched": int(len(unmatched_src_ids)),
            "n_computed": int(computed.sum()),
            "k_chosen_mean": float(k_chosen_unm.mean()) if len(k_chosen_unm) else 0.0,
        }

    if tune_k:
        k_grid_default = [3, 5, 10, 15, 20, 30, 50, n // 2, n - 1]
        tune_result = tune_k_for_pair(filt_src, filt_tgt, k_grid_default)
        out["tune_k"] = tune_result

        # Build virtual atoms at the tuned K* and save them to disk so
        # downstream transfer becomes a lookup.
        from scripts.intervention.ablation_sweep import get_unmatched_features
        unmatched_src_ids = get_unmatched_features(
            source, target, matching_file=matching_path
        )
        k_star = int(tune_result["best_k"])
        vatoms, computed = build_virtual_atoms_topk(
            atoms_s, filt_src, filt_tgt, unmatched_src_ids, k=k_star
        )

        VIRTUAL_ATOMS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = VIRTUAL_ATOMS_CACHE_DIR / f"{source}_to_{target}.npz"
        np.savez_compressed(
            cache_path,
            virtual_atoms=vatoms,
            feature_ids=np.array(unmatched_src_ids, dtype=np.int64),
            computed_mask=computed,
            k_star=np.int64(k_star),
            source_model=np.array(source),
            target_model=np.array(target),
            n_landmarks=np.int64(len(filt_src)),
            n_unmatched=np.int64(len(unmatched_src_ids)),
            n_computed=np.int64(int(computed.sum())),
            d_target=np.int64(filt_tgt.shape[1]),
            ridge_alpha=np.float32(1.0),
            min_cosine=np.float32(0.0),
            neighborhood_mode=np.array("topk"),
            matching_file=np.array(str(matching_path)),
            matching_file_sha256=np.array(sha256_file(matching_path)),
            landmark_source=np.array(landmark_source),
        )
        out["virtual_atoms_cache"] = {
            "path": str(cache_path),
            "k_star": k_star,
            "n_unmatched": int(len(unmatched_src_ids)),
            "n_computed": int(computed.sum()),
        }

    if include_mlp:
        # MLP always evaluated on an 80/20 split (LOO is N fits per pair — prohibitive).
        rng = np.random.RandomState(SEED + 1)
        perm = rng.permutation(n)
        n_val = max(2, int(n * VAL_FRAC))
        val_i, train_i = perm[:n_val], perm[n_val:]
        train_src, train_tgt = filt_src[train_i], filt_tgt[train_i]
        val_src, val_tgt = filt_src[val_i], filt_tgt[val_i]
        try:
            mlp_model, _ = fit_concept_map_mlp(train_src, train_tgt, val_frac=0.25)
            with torch.no_grad():
                mlp_pred = mlp_model(torch.tensor(val_src, dtype=torch.float32)).numpy()
            cos_mlp = cosines(mlp_pred, val_tgt)
            out["n_eval_mlp"] = n_val
            out["cosine"]["mlp"] = {
                "mean": float(np.mean(cos_mlp)), "median": float(np.median(cos_mlp)),
                "raw": cos_mlp,
            }
        except Exception as e:
            out["cosine"]["mlp"] = {"skipped": True, "reason": str(e)}

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loo", action="store_true",
                        help="Evaluate ridge and adaptive_local via LOO over all landmarks "
                             "instead of an 80/20 held-out split. MLP always uses 80/20.")
    parser.add_argument("--no-mlp", action="store_true",
                        help="Skip MLP evaluation.")
    parser.add_argument("--landmark-source", default="mnn_pairwise",
                        choices=["concept_groups", "mnn_pairwise"],
                        help="Source of landmark pairs. Default mnn_pairwise reads raw "
                             "noise-floor-verified MNN edges from --matching-file; "
                             "concept_groups uses the legacy first-src × first-tgt per "
                             "concept group.")
    parser.add_argument("--matching-file", type=Path, default=MATCHING_FILE_DEFAULT,
                        help="Raw MNN matching JSON (used when landmark-source=mnn_pairwise). "
                             "Default is sae_feature_matching_mnn_floor_p90.json.")
    parser.add_argument("--tune-k", action="store_true",
                        help="For each pair, sweep K over a grid and LOO-score top-K local "
                             "ridge. Adds per-pair best_k and best_mean_cosine to the output "
                             "so we can compare CV-tuned K to the |cos|>=0.40 threshold.")
    parser.add_argument("--tune-k-per-query", action="store_true",
                        help="For each unmatched concept (and each held-out landmark), "
                             "pick K by CV-LOO within the query's top-K neighborhood. "
                             "Outputs per-pair loo_cosine and saves per-query-CV virtual atoms.")
    parser.add_argument("--pair", nargs=2, metavar=("SRC", "TGT"), default=None,
                        help="If set, run only this single pair. Enables external parallelism "
                             "via shell loops or dispatch to worker hosts.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (multiprocessing). "
                             "Only applies when running multiple pairs in one invocation.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {"pairs": {}, "config": {
        "val_frac": VAL_FRAC,
        "seed": SEED,
        "cosine_threshold": COSINE_THRESHOLD,
        "ridge_alpha": 1.0,
        "loo": args.loo,
        "include_mlp": not args.no_mlp,
        "landmark_source": args.landmark_source,
        "matching_file": str(args.matching_file) if args.landmark_source == "mnn_pairwise" else None,
        "tune_k": args.tune_k,
        "tune_k_per_query": args.tune_k_per_query,
        "workers": args.workers,
    }}

    pairs_to_run = [tuple(args.pair)] if args.pair else PAIRS

    def run_one(st):
        s, t = st
        print(f"\n=== {s} vs {t} ===", flush=True)
        try:
            r = run_pair(s, t, loo=args.loo, include_mlp=not args.no_mlp,
                         landmark_source=args.landmark_source,
                         matching_path=args.matching_file,
                         tune_k=args.tune_k,
                         tune_k_per_query=args.tune_k_per_query)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = {"skipped": True, "reason": f"exception: {e}"}
        return f"{s}_vs_{t}", r

    if args.workers > 1 and len(pairs_to_run) > 1:
        from multiprocessing import get_context
        ctx = get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for key, r in pool.imap_unordered(_pool_worker_unpack, [
                (st, args.loo, not args.no_mlp, args.landmark_source,
                 args.matching_file, args.tune_k, args.tune_k_per_query)
                for st in pairs_to_run
            ]):
                results["pairs"][key] = r
    else:
        for st in pairs_to_run:
            key, r = run_one(st)
            results["pairs"][key] = r

    # Print per-pair summaries after the pool/serial loop
    for key, r in results["pairs"].items():
        print(f"\n=== {key} ===")
        if r.get("skipped"):
            print(f"  SKIPPED: {r['reason']}")
            continue
        c = r["cosine"]
        mode = "LOO" if args.loo else "80/20"
        print(f"  n_filt={r['n_filtered']}  eval_mode={mode}")
        print(f"  global_ridge   mean_cos={c['global_ridge']['mean']:.3f}  median={c['global_ridge']['median']:.3f}")
        print(f"  adaptive_local mean_cos={c['adaptive_local']['mean']:.3f}  median={c['adaptive_local']['median']:.3f}")
        if "mlp" in c and not c["mlp"].get("skipped"):
            print(f"  mlp [80/20]    mean_cos={c['mlp']['mean']:.3f}  median={c['mlp']['median']:.3f}")
        if "tune_k" in r:
            tk = r["tune_k"]
            tuned_mean = tk["best_mean_cosine"]
            threshold_mean = c["adaptive_local"]["mean"]
            print(f"  tuned top-K    best_K={tk['best_k']}  mean_cos={tuned_mean:.3f}  "
                  f"Δ vs |cos|>=0.4: {tuned_mean - threshold_mean:+.3f}")
        if "tune_k_per_query" in r:
            pq = r["tune_k_per_query"]
            delta_vs_global = pq["loo_cosine_mean"] - c["global_ridge"]["mean"]
            delta_vs_adapt = pq["loo_cosine_mean"] - c["adaptive_local"]["mean"]
            print(f"  per-query CV   mean_cos={pq['loo_cosine_mean']:.3f}  "
                  f"K_chosen mean={pq['k_chosen_mean']:.1f} std={pq['k_chosen_std']:.1f}  "
                  f"Δ vs global: {delta_vs_global:+.3f}  Δ vs |cos|>=0.4: {delta_vs_adapt:+.3f}")

    suffix_parts = []
    if args.loo: suffix_parts.append("loo")
    if args.landmark_source != "mnn_pairwise":
        suffix_parts.append(args.landmark_source)
    if args.tune_k: suffix_parts.append("tunek")
    if args.tune_k_per_query: suffix_parts.append("pqcv")
    if args.pair: suffix_parts.append(f"{args.pair[0]}_{args.pair[1]}")
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    out_json = OUT_DIR / f"transfer_map_comparison{suffix}.json"
    out_pdf = OUT_DIR / f"transfer_map_comparison{suffix}.pdf"

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_json}")

    methods = ["global_ridge", "adaptive_local"]
    if not args.no_mlp: methods.append("mlp")
    agg = {m: [] for m in methods}
    for r in results["pairs"].values():
        if r.get("skipped"): continue
        for m in methods:
            entry = r["cosine"].get(m)
            if entry and not entry.get("skipped"):
                agg[m].extend(entry["raw"])

    print("\nAGGREGATE mean cosine:")
    for m in methods:
        xs = np.array(agg[m])
        if len(xs):
            print(f"  {m:<16} n={len(xs):<5} mean={xs.mean():+.3f}  median={np.median(xs):+.3f}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        labels = {"global_ridge": "global ridge", "adaptive_local": "adaptive local",
                  "mlp": "2-layer MLP [80/20]"}
        present = [m for m in methods if len(agg[m])]
        positions = range(len(present))
        data = [agg[m] for m in present]
        ax.boxplot(data, positions=positions, widths=0.6, showfliers=False)
        ax.set_xticks(positions)
        ax.set_xticklabels([labels[m] for m in present])
        ax.set_ylabel("cosine(pred target, true target)")
        ax.axhline(0.0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.grid(axis="y", alpha=0.2)
        eval_label = "LOO" if args.loo else "80/20 held-out"
        ax.set_title(f"Transfer map quality by family ({eval_label} on landmarks)")
        plt.tight_layout()
        plt.savefig(out_pdf, bbox_inches="tight")
        print(f"Wrote {out_pdf}")
    except Exception as e:
        print(f"plot failed: {e}")


if __name__ == "__main__":
    main()
