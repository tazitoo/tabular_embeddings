#!/usr/bin/env python3
"""REBUTTAL control: NON-greedy top-K unmatched transfer.

The greedy transfer sweep (transfer_sweep_symmetric.py) closes ~50% of the
per-row gap even with a RANDOM SAE. The question this script answers: is that
gap-closure a property of the transferred *concepts*, or of the *greedy search
over them*? To isolate that, this removes every selection step the greedy does:

  greedy               -> this control
  --------------------------------------------------------------------
  rank by |importance| -> rank by |importance|            (KEPT)
  per-row combinatorial  accept/reject                    (REMOVED)
  try +delta AND -delta, reject overshoot past strong     (REMOVED)
  early-stop at gc tol                                     (REMOVED)

Instead: take the top-K unmatched concepts that FIRE on the row (ranked by the
donor's ablation importance, exactly the greedy's candidate ranking), inject ALL
K of them at once (+direction, summed), and measure the same gap-closure. K
defaults to 100 -- deliberately ~an order of magnitude more than the greedy's
mean accepted count (mean_k ~4-30) -- so the control cannot lose for lack of
concepts. If gap-closure collapses relative to greedy, the 50% figure is an
artifact of the search, not the random concepts.

Everything upstream of the per-row loop (strong/weak assignment, recipient tail,
decoder atoms, global concept map, virtual atoms) is byte-identical to the
default path of transfer_sweep_symmetric.run_dataset so the only variable is
greedy-select vs top-K-inject. Concepts must be UNMATCHED (same unmatched set
the greedy transfers), never the MNN-matched landmarks.

Output: <output-dir>/<min_a>_vs_<max_b>/<dataset>.npz  (--resume skips done)

Usage:
    python -m scripts.rebuttal.transfer_topk_unmatched --models tabicl tabpfn --forward \
        --sae-dir output/sae_random_baseline \
        --importance-dir output/perrow_importance_random \
        --matching-file output/sae_feature_matching_mnn_t0.001_random.json \
        --output-dir output/rebuttal/forward_topk_random --top-k 100
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import (
    SPLITS_PATH,
    load_sae, get_extraction_layer_taskaware, build_tail,
    load_dataset_context, load_test_embeddings,
    compute_per_row_loss, compute_importance_metric,
    batched_intervention, batched_intervention_sequential,
    MitraTail, SEQUENTIAL_MODELS,
)
from scripts.matching.utils import load_norm_stats as load_norm_stats_matching
from scripts.intervention.transfer_virtual_nodes import (
    extract_decoder_atoms, fit_concept_map, filter_landmarks,
)
from scripts.rebuttal.transfer_sweep_symmetric import (
    get_matched_pairs, get_unmatched_features, DEFAULT_MATCHING_FILE,
    SUPPORTED_MODELS,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

IMPORTANCE_DIR = PROJECT_ROOT / "output" / "perrow_importance"
EPS = 1e-7


def _widen_binary(p):
    if p.ndim == 2 and p.shape[1] == 1:
        p1 = p.ravel().astype(np.float32)
        return np.column_stack([1.0 - p1, p1]).astype(np.float32)
    return p


def _gap_closed(baseline_p, inter_p, strong_p, y_r):
    """Same gap-closure metric as the greedy sweep (loss-space for cls,
    squared-distance for reg). Inputs are single-row predictions."""
    baseline_p = np.asarray(baseline_p)
    if baseline_p.ndim >= 1 and baseline_p.size > 1:      # classification prob vector
        orig = -np.log(np.clip(baseline_p[y_r], EPS, 1 - EPS))
        tgt = -np.log(np.clip(np.asarray(strong_p)[y_r], EPS, 1 - EPS))
        best = -np.log(np.clip(np.asarray(inter_p)[y_r], EPS, 1 - EPS))
        gap, moved = orig - tgt, orig - best
        return float(min(1.0, max(0.0, moved / gap))) if gap > 1e-8 else 1.0
    orig_sq = float((float(baseline_p) - float(strong_p)) ** 2)   # regression scalar
    best_sq = float((float(inter_p) - float(strong_p)) ** 2)
    return float(min(1.0, max(0.0, 1.0 - best_sq / orig_sq))) if orig_sq > 1e-12 else 1.0


def run_dataset(model_a, model_b, dataset, saes, splits, norm_stats,
                test_embeddings, matched_pairs, unmatched_features, device,
                top_k, min_cosine=0.0, importance_dir=None, min_gap=0.01,
                reverse=True):
    """Top-K unmatched injection for one dataset. Setup mirrors the default path
    of transfer_sweep_symmetric.run_dataset exactly; only the per-row loop differs."""
    imp_dir = importance_dir if importance_dir else IMPORTANCE_DIR
    imp_a = np.load(imp_dir / model_a / f"{dataset}.npz", allow_pickle=True)
    imp_b = np.load(imp_dir / model_b / f"{dataset}.npz", allow_pickle=True)
    preds_a = _widen_binary(imp_a["baseline_preds"])
    preds_b = _widen_binary(imp_b["baseline_preds"])
    assert np.array_equal(imp_a["row_indices"], imp_b["row_indices"])
    y_query = imp_a["y_query"]
    n_query = len(y_query)
    task = "classification" if preds_a.ndim == 2 else "regression"

    preds = {model_a: preds_a, model_b: preds_b}
    losses = {model_a: compute_per_row_loss(y_query, preds_a, task),
              model_b: compute_per_row_loss(y_query, preds_b, task)}
    metric_a, metric_name = compute_importance_metric(y_query, preds[model_a], task)
    metric_b, _ = compute_importance_metric(y_query, preds[model_b], task)
    if metric_name == "degenerate" or metric_a == float("-inf") or metric_b == float("-inf"):
        logger.info("  SKIP (degenerate predictions)")
        return None

    if metric_a >= metric_b:
        strong, weak = model_a, model_b
        metric_strong, metric_weak = metric_a, metric_b
    else:
        strong, weak = model_b, model_a
        metric_strong, metric_weak = metric_b, metric_a
    if reverse:
        strong, weak = weak, strong
        metric_strong, metric_weak = metric_weak, metric_strong

    logger.info(f"  {'[REVERSE] ' if reverse else ''}donor={strong} "
                f"({metric_name}={metric_strong:.4f}), recipient={weak} "
                f"({metric_name}={metric_weak:.4f})")

    if dataset not in norm_stats[weak]:
        logger.info(f"  SKIP (recipient {weak} has no norm stats)")
        return None

    weak_loss, strong_loss = losses[weak], losses[strong]
    baseline_preds_w = preds[weak]
    strong_preds = preds[strong]
    strong_wins = strong_loss < weak_loss
    n_strong_wins = int(strong_wins.sum())
    logger.info(f"  Strong wins on {n_strong_wins}/{n_query} rows")
    if n_strong_wins == 0:
        return None

    # Recipient (weak) tail — identical construction to the greedy sweep.
    X_train_w, y_train_w, X_query_w, _, _, task_w = load_dataset_context(weak, dataset, splits)
    if y_train_w.dtype == np.int32:
        y_train_w = y_train_w.astype(np.int64)
    layer_w = get_extraction_layer_taskaware(weak, dataset=dataset)
    cat_indices = None
    if weak in ("hyperfast", "tabpfn"):
        from data.preprocessing import load_preprocessed, CACHE_DIR
        try:
            pre = load_preprocessed(weak, dataset, CACHE_DIR)
            cat_indices = pre.cat_indices if pre.cat_indices else None
        except Exception:
            pass
    target_name = splits.get(dataset, {}).get("target", "target")
    torch.manual_seed(13); np.random.seed(13)
    tail_w = build_tail(weak, X_train_w, y_train_w, X_query_w, layer_w, task_w,
                        device, cat_indices=cat_indices, target_name=target_name)

    atoms_strong = extract_decoder_atoms(saes[strong]).numpy()
    atoms_weak = extract_decoder_atoms(saes[weak]).numpy()
    with torch.no_grad():
        emb_s = torch.tensor(test_embeddings[strong][dataset], dtype=torch.float32, device=device)
        h_strong = saes[strong].encode(emb_s).cpu().numpy()
    ds_mean_w, ds_std_w = norm_stats[weak][dataset]

    pair_key = f"{strong}_to_{weak}"
    m_pairs = matched_pairs.get(pair_key, [])
    unmatched = unmatched_features.get(pair_key, [])
    if not unmatched:
        logger.info(f"  SKIP (no unmatched features from {strong})")
        return None

    use_sequential = isinstance(tail_w, SEQUENTIAL_MODELS)
    use_mitra = isinstance(tail_w, MitraTail)

    imp_strong = np.load(imp_dir / strong / f"{dataset}.npz", allow_pickle=True)
    row_feature_drops = imp_strong["row_feature_drops"]
    feature_indices = imp_strong["feature_indices"]
    unmatched_set = set(unmatched)

    # Global concept map + virtual atoms — identical to the greedy default path.
    matched_src_atoms = atoms_strong[[si for si, _ in m_pairs]]
    matched_tgt_atoms = atoms_weak[[ti for _, ti in m_pairs]]
    filt_src, filt_tgt, filt_pairs, quality = filter_landmarks(
        matched_src_atoms, matched_tgt_atoms, m_pairs, min_cosine=min_cosine, alpha=1.0)
    if len(filt_pairs) < 5:
        logger.info(f"  SKIP (too few landmarks: {len(filt_pairs)})")
        return None
    d_target = atoms_weak.shape[1]
    M_global, r2_global = fit_concept_map(filt_src, filt_tgt, alpha=1.0)
    src_norms_m = np.linalg.norm(filt_src, axis=1)
    tgt_norms_m = np.linalg.norm(filt_tgt, axis=1)
    valid = (src_norms_m > 1e-8) & (tgt_norms_m > 1e-8)
    median_norm_ratio = float(np.median(tgt_norms_m[valid] / src_norms_m[valid])) if valid.sum() else 1.0
    virtual_atoms_cache = {}
    for fi in unmatched:
        atom_s = atoms_strong[fi]
        an = np.linalg.norm(atom_s)
        if an < 1e-8:
            continue
        vdir = (atom_s / an) @ M_global.T
        vn = np.linalg.norm(vdir)
        if vn < 1e-8:
            continue
        virtual_atoms_cache[fi] = (vdir / vn) * an * median_norm_ratio
    logger.info(f"  Global map R²={r2_global:.3f}; {len(virtual_atoms_cache)}/{len(unmatched)} virtual atoms")

    # ---- Per-row TOP-K injection (no greedy selection) ----
    gap_closed = np.full(n_query, np.nan, dtype=np.float32)
    k_used = np.zeros(n_query, dtype=np.int32)
    n_firing = np.zeros(n_query, dtype=np.int32)
    preds_intervened = baseline_preds_w.copy()
    deployed_delta = np.zeros((n_query, d_target), dtype=np.float32)
    data_std_w = np.asarray(ds_std_w, dtype=np.float32)

    t0 = time.time()
    for r in range(n_query):
        if not strong_wins[r]:
            gap_closed[r] = 1.0
            continue
        # same "models agree" skip as greedy
        if baseline_preds_w.ndim == 2:
            y_r = int(y_query[r])
            pred_gap = abs(float(strong_preds[r, y_r] - baseline_preds_w[r, y_r]))
        else:
            denom = abs(float(strong_preds[r]))
            pred_gap = abs(float(strong_preds[r] - baseline_preds_w[r])) / denom if denom > 1e-8 else 0.0
        if pred_gap < min_gap:
            gap_closed[r] = 1.0
            continue

        row_drops = row_feature_drops[r]
        firing = [(fi, abs(row_drops[i]))
                  for i, fi in enumerate(feature_indices)
                  if fi in unmatched_set and h_strong[r, fi] > 0 and fi in virtual_atoms_cache]
        n_firing[r] = len(firing)
        if not firing:
            gap_closed[r] = 0.0
            continue
        firing.sort(key=lambda x: -x[1])
        chosen = firing[:top_k]

        # Sum the +delta of every chosen concept (no sign search, no accept/reject).
        delta = np.zeros(d_target, dtype=np.float32)
        for fi, _ in chosen:
            delta += float(h_strong[r, fi]) * virtual_atoms_cache[fi].astype(np.float32) * data_std_w
        deployed_delta[r] = delta
        k_used[r] = len(chosen)

        dt = torch.tensor(delta[None], dtype=torch.float32, device=device)
        X_row = X_query_w[r:r + 1]
        if use_sequential:
            p = batched_intervention_sequential(tail_w, X_row, dt, query_idx=r)
        else:
            p = batched_intervention(tail_w, X_row, dt, inject_context=False)
        p = np.asarray(p)[0]
        preds_intervened[r] = p
        gap_closed[r] = _gap_closed(baseline_preds_w[r], p, strong_preds[r], int(y_query[r]))

        if (r + 1) % 50 == 0 or r == n_query - 1:
            el = time.time() - t0
            rate = (r + 1) / el if el > 0 else 0
            vg = gap_closed[:r + 1][strong_wins[:r + 1]]
            vg = vg[~np.isnan(vg)]
            logger.info(f"    row {r+1}/{n_query}: mean_k={k_used[:r+1][strong_wins[:r+1]].mean():.1f} "
                        f"mean_gc={vg.mean() if len(vg) else 0:.3f} ({rate:.1f} rows/s)")

    valid_gc = gap_closed[strong_wins]
    valid_gc = valid_gc[~np.isnan(valid_gc)]
    logger.info(f"  Done in {time.time()-t0:.1f}s  mean_gc={float(valid_gc.mean()) if len(valid_gc) else 0:.3f} "
                f"(top_k={top_k}, mean_k_used={k_used[strong_wins].mean():.1f})")

    return {
        "strong_model": strong, "weak_model": weak, "reverse": bool(reverse),
        "donor_model": strong, "recipient_model": weak,
        "top_k": int(top_k),
        "gap_closed": gap_closed, "k_used": k_used, "n_firing": n_firing,
        "strong_wins": strong_wins,
        "preds_strong": strong_preds, "preds_weak": baseline_preds_w,
        "preds_intervened": preds_intervened, "deployed_delta": deployed_delta,
        "baseline_loss_strong": strong_loss, "baseline_loss_weak": weak_loss,
        "n_query": n_query, "n_strong_wins": n_strong_wins,
        "mean_gap_closed": float(valid_gc.mean()) if len(valid_gc) else 0.0,
        "mean_k_used": float(k_used[strong_wins].mean()),
        "metric_strong": float(metric_strong), "metric_weak": float(metric_weak),
        "metric_name": metric_name, "y_query": y_query.astype(np.float32),
        "row_indices": imp_a["row_indices"].astype(np.int32),
        "concept_map_r2": float(r2_global), "n_landmarks": int(len(filt_pairs)),
        "n_virtual_atoms": int(len(virtual_atoms_cache)),
    }


def main():
    ap = argparse.ArgumentParser(description="Non-greedy top-K unmatched transfer control")
    ap.add_argument("--models", nargs=2, required=True, metavar="MODEL")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-k", type=int, default=100,
                    help="Inject the top-K unmatched firing concepts by |importance| (default 100).")
    ap.add_argument("--min-gap", type=float, default=0.01)
    ap.add_argument("--min-cosine", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sae-dir", type=Path, default=None)
    ap.add_argument("--importance-dir", type=Path, default=None)
    ap.add_argument("--matching-file", type=Path, default=DEFAULT_MATCHING_FILE)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--forward", action="store_true",
                    help="Forward (below-diagonal) direction, matching the paper / the random forward run.")
    args = ap.parse_args()
    reverse = not args.forward

    model_a, model_b = sorted(args.models)
    for m in (model_a, model_b):
        assert m in SUPPORTED_MODELS, f"unknown model {m}"
    out_dir = args.output_dir / f"{model_a}_vs_{model_b}"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = json.loads(SPLITS_PATH.read_text())
    sae_dir = args.sae_dir if args.sae_dir else None
    saes, test_embeddings, norm_stats = {}, {}, {}
    for m in (model_a, model_b):
        s, _ = load_sae(m, device=args.device, **({"sae_dir": sae_dir} if sae_dir else {}))
        s.eval(); saes[m] = s
        test_embeddings[m] = load_test_embeddings(m)
        norm_stats[m] = load_norm_stats_matching(m)

    # Matched + unmatched sets for both directions.
    matched_pairs, unmatched_features = {}, {}
    for src, tgt in ((model_a, model_b), (model_b, model_a)):
        key = f"{src}_to_{tgt}"
        matched_pairs[key] = get_matched_pairs(src, tgt, matching_file=args.matching_file)
        unmatched_features[key] = get_unmatched_features(src, tgt, matching_file=args.matching_file)

    if args.datasets:
        datasets = args.datasets
    else:
        a_ds = {p.stem for p in (args.importance_dir or IMPORTANCE_DIR).joinpath(model_a).glob("*.npz")}
        b_ds = {p.stem for p in (args.importance_dir or IMPORTANCE_DIR).joinpath(model_b).glob("*.npz")}
        datasets = sorted(a_ds & b_ds)
    logger.info(f"Top-K unmatched transfer [{'FORWARD' if args.forward else 'REVERSE'}], "
                f"top_k={args.top_k}: {model_a} vs {model_b} ({len(datasets)} datasets)")

    for i, ds in enumerate(datasets):
        out_path = out_dir / f"{ds}.npz"
        if args.resume and out_path.exists():
            logger.info(f"[{i+1}/{len(datasets)}] {ds}: SKIP (exists)")
            continue
        logger.info(f"\n[{i+1}/{len(datasets)}] {ds}")
        try:
            res = run_dataset(model_a, model_b, ds, saes, splits, norm_stats,
                              test_embeddings, matched_pairs, unmatched_features,
                              args.device, args.top_k, min_cosine=args.min_cosine,
                              importance_dir=args.importance_dir, min_gap=args.min_gap,
                              reverse=reverse)
        except Exception as e:
            logger.info(f"  FAIL: {type(e).__name__}: {e}")
            continue
        if res is not None:
            np.savez_compressed(out_path, **res)


if __name__ == "__main__":
    main()
