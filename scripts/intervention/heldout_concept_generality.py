#!/usr/bin/env python3
"""Held-out concept generality via cross-fitting: does a *train-selected* concept
set close the gap on *held-out* rows?

Rebuttal experiment (ICLR 2027) answering the circularity objection raised by all
three reviewers: the paper's per-row greedy (ablation_sweep.py) selects concepts
using each row's own strong/weak target and then scores that same row, so
gap-closure is near-guaranteed by construction and "2-5 concepts" is a statement
about the search, not the models. This script breaks the selection<->evaluation
coupling with stratified K-fold cross-fitting:

  1. Keep only the active rows (strong wins per-row AND the models disagree past
     min_gap) -- the only rows with a gap to close in the S->W direction.
  2. Stratified K-fold over the active rows (LOO when very few). For each fold:
     select ONE shared set per arm on the K-1 TRAIN folds by greedily maximising
     MEAN gap-closed (a joint, set-level objective), then evaluate the frozen set
     UNCHANGED on the held-out fold.
  3. Pool the held-out (out-of-fold) gc across folds -> every active row is scored
     exactly once under a set that never saw it. Report the pooled mean, the
     across-fold std (variance on small N), and the CONSENSUS set (features chosen
     in >= a fraction of folds) as the interpretable "the gap is these concepts".
  4. One arm only: UNMATCHED trained-SAE concepts (the claim). The comparison is the
     paper's own per-row circular gap-closed, read from the cached forward-ablation
     results and joined by row_indices on the SAME active rows -- a paired,
     per-row contrast of circular (selection saw the eval row) vs honest (it did not).

Success criterion, fixed in advance: a large fraction of the reported per-row
circular closure survives cross-fitting. The honest OOF gc is expected to sit below
the circular per-row gc, and the size of that drop is the quantity of interest.

Output:
    output/round{N}/heldout_generality/{model_a}_vs_{model_b}/{dataset}.npz

Usage:
    python -m scripts.intervention.heldout_concept_generality --models tabpfn tabicl --device cuda
    python -m scripts.intervention.heldout_concept_generality --models tabpfn tabicl --datasets credit-g
"""
import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, KFold

from scripts.round_paths import RESULTS_DIR, FORWARD_ABLATION_DIR, IMPORTANCE_DIR
from scripts.intervention.intervene_lib import (
    SPLITS_PATH,
    load_sae, get_extraction_layer_taskaware, build_tail,
    load_dataset_context, load_test_embeddings,
    compute_per_row_loss, compute_importance_metric,
    batched_ablation, batched_ablation_sequential,
    MitraTail, SEQUENTIAL_MODELS,
)
from scripts.intervention.ablation_sweep import get_unmatched_features, DEFAULT_MATCHING_FILE
from scripts.matching.utils import load_norm_stats as load_norm_stats_matching

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = RESULTS_DIR / "heldout_generality"
SUPPORTED_MODELS = ["tabpfn", "tabicl", "tabicl_v2", "mitra", "tabdpt", "carte"]


# ── gap-closed, mirroring ablation_sweep.py exactly ──────────────────────────

def _gc_from_pred(pred_r, strong_r, weak_r, y_r, task):
    """Gap-closed of an ablated prediction toward the weak model, for one row.

    Classification: fraction of the strong->weak log-loss gap (on the true class)
    recovered. Regression: fraction of the squared strong->weak prediction gap.
    Clipped to [0, 1]. Identical formula to ablation_sweep.run_dataset.
    """
    eps = 1e-7
    if task != "regression":
        orig = -np.log(np.clip(strong_r[y_r], eps, 1 - eps))
        target = -np.log(np.clip(weak_r[y_r], eps, 1 - eps))
        cur = -np.log(np.clip(pred_r[y_r], eps, 1 - eps))
        gap = target - orig
        moved = cur - orig
        return float(min(1.0, max(0.0, moved / gap))) if gap > 1e-8 else 1.0
    orig = (float(strong_r) - float(weak_r)) ** 2
    gap = abs(orig)
    cur = (float(pred_r) - float(weak_r)) ** 2
    moved = abs(cur - orig)
    return float(min(1.0, moved / gap)) if gap > 1e-8 else 1.0


def _predict_sets(tail, sae, h_row, X_row, feat_sets, data_std, row_pos, device,
                  use_mitra, use_seq, max_K):
    """Predict once per feature set: zero each set's features in SAE latent space,
    decode to a delta, inject, forward. Returns (len(feat_sets), ...) predictions.
    """
    with torch.no_grad():
        recon_full = sae.decode(h_row.unsqueeze(0))
        K = len(feat_sets)
        h_batch = h_row.unsqueeze(0).expand(K, -1).clone()
        for k, fset in enumerate(feat_sets):
            for fi in fset:
                h_batch[k, fi] = 0.0
        deltas = (sae.decode(h_batch) - recon_full) * data_std.unsqueeze(0)
    if use_seq:
        return batched_ablation_sequential(tail, X_row, deltas, query_idx=row_pos)
    return batched_ablation(tail, X_row, deltas, max_K=max_K)


# ── selection on the training folds ──────────────────────────────────────────

def _greedy_select(pool, fit_rows, sae, tail, activations, X_query, strong_preds,
                   weak_preds, y_query, task, data_std, firing_mask, m_max,
                   device, use_mitra, use_seq, max_K):
    """Greedily grow ONE shared feature set that maximises mean gap-closed over the
    fit rows. At each step, evaluate every remaining candidate (combined with the
    already-accepted set) across all fit rows and accept the best if it improves
    mean fit gc. Non-firing rows contribute 0 movement (averaged over all fit rows,
    so rarely-firing concepts are penalised). Returns (accepted, train_gc_path).
    """
    accepted, train_gc_path = [], []
    best_mean = 0.0
    remaining = list(pool)
    for _ in range(min(m_max, len(pool))):
        cand_gc_sum = {c: 0.0 for c in remaining}
        for r in fit_rows:
            firing = [c for c in remaining if firing_mask[r, c]]
            if not firing:
                continue
            feat_sets = [accepted + [c] for c in firing]
            preds = _predict_sets(tail, sae, activations[r], X_query[r:r + 1],
                                  feat_sets, data_std, r, device, use_mitra,
                                  use_seq, max_K)
            y_r = int(y_query[r])
            for j, c in enumerate(firing):
                cand_gc_sum[c] += _gc_from_pred(preds[j], strong_preds[r],
                                                weak_preds[r], y_r, task)
        best_c, best_c_mean = None, best_mean
        for c in remaining:
            mean_c = cand_gc_sum[c] / len(fit_rows)
            if mean_c > best_c_mean:
                best_c, best_c_mean = c, mean_c
        if best_c is None:
            break
        accepted.append(best_c)
        remaining.remove(best_c)
        best_mean = best_c_mean
        train_gc_path.append(best_c_mean)
    return accepted, train_gc_path


def _eval_perrow(features, rows, sae, tail, activations, X_query, strong_preds,
                 weak_preds, y_query, task, data_std, firing_mask, device,
                 use_mitra, use_seq, max_K):
    """Per-row gap-closed under the frozen set's size-k prefixes. Returns
    (len(rows), len(features)); column -1 is the full set. Empty if no features."""
    if not features or not rows:
        return np.zeros((len(rows), len(features)), dtype=np.float32)
    mat = np.zeros((len(rows), len(features)), dtype=np.float32)
    for ri, r in enumerate(rows):
        prefixes = [features[:k + 1] for k in range(len(features))]
        preds = _predict_sets(tail, sae, activations[r], X_query[r:r + 1],
                              prefixes, data_std, r, device, use_mitra, use_seq, max_K)
        y_r = int(y_query[r])
        for k in range(len(features)):
            mat[ri, k] = _gc_from_pred(preds[k], strong_preds[r], weak_preds[r], y_r, task)
    return mat


def _make_folds(active_idx, y_active, task, n_folds, loo_threshold, seed):
    """Yield (train_rows, heldout_rows) as absolute row indices. Stratified K-fold
    on the label for classification (falls back to plain K-fold), LOO when the
    active set is below loo_threshold."""
    n = len(active_idx)
    if n < loo_threshold:
        k = n                                   # leave-one-out
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
        folds = splitter.split(active_idx)
    else:
        k = n_folds
        if task != "regression":
            try:
                splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
                folds = list(splitter.split(active_idx, y_active.astype(int)))
            except ValueError:
                folds = list(KFold(n_splits=k, shuffle=True, random_state=seed).split(active_idx))
        else:
            folds = list(KFold(n_splits=k, shuffle=True, random_state=seed).split(active_idx))
    for tr_local, te_local in folds:
        yield (sorted(active_idx[tr_local].tolist()),
               sorted(active_idx[te_local].tolist()))


def run_dataset(model_a, model_b, dataset, saes, splits, norm_stats,
                test_embeddings, unmatched, device, args):
    """Cross-fitted held-out generality for one dataset."""
    imp_dir = Path(args.importance_dir) if args.importance_dir else IMPORTANCE_DIR
    imp_a = np.load(imp_dir / model_a / f"{dataset}.npz", allow_pickle=True)
    imp_b = np.load(imp_dir / model_b / f"{dataset}.npz", allow_pickle=True)

    def _widen(p):
        if p.ndim == 2 and p.shape[1] == 1:
            p1 = p.ravel().astype(np.float32)
            return np.column_stack([1.0 - p1, p1]).astype(np.float32)
        return p
    preds_a, preds_b = _widen(imp_a["baseline_preds"]), _widen(imp_b["baseline_preds"])
    assert np.array_equal(imp_a["row_indices"], imp_b["row_indices"]), "row index mismatch"
    row_indices = imp_a["row_indices"]
    y_query = imp_a["y_query"]
    n_query = len(y_query)
    task = "classification" if preds_a.ndim == 2 else "regression"

    preds = {model_a: preds_a, model_b: preds_b}
    losses = {m: compute_per_row_loss(y_query, preds[m], task) for m in (model_a, model_b)}
    metric_a, mname = compute_importance_metric(y_query, preds_a, task)
    metric_b, _ = compute_importance_metric(y_query, preds_b, task)
    if mname == "degenerate" or metric_a == float("-inf") or metric_b == float("-inf"):
        return {"status": "degenerate"}
    strong, weak = (model_a, model_b) if metric_a >= metric_b else (model_b, model_a)
    if dataset not in norm_stats[strong]:
        return {"status": "no_norm_stats", "strong_model": strong, "weak_model": weak}

    strong_preds, weak_preds = preds[strong], preds[weak]
    strong_loss, weak_loss = losses[strong], losses[weak]

    # Active rows: strong wins per-row AND the models disagree past min_gap.
    active = np.zeros(n_query, dtype=bool)
    for r in range(n_query):
        if strong_loss[r] >= weak_loss[r]:
            continue
        if strong_preds.ndim == 2:
            y_r = int(y_query[r])
            pred_gap = abs(float(strong_preds[r, y_r] - weak_preds[r, y_r]))
        else:
            denom = abs(float(strong_preds[r]))
            pred_gap = abs(float(strong_preds[r] - weak_preds[r])) / denom if denom > 1e-8 else 0.0
        if pred_gap >= args.min_gap:
            active[r] = True
    n_active = int(active.sum())
    if n_active < args.min_active:
        return {"status": "too_few_active", "strong_model": strong,
                "weak_model": weak, "n_active": n_active}

    # Strong-model SAE activations + firing mask over all rows.
    emb_s = test_embeddings[strong][dataset]
    with torch.no_grad():
        activations = saes[strong].encode(torch.tensor(emb_s, dtype=torch.float32, device=device))
    firing_mask = (activations > 0).cpu().numpy()
    ds_mean, ds_std = norm_stats[strong][dataset]
    data_std = torch.tensor(ds_std, dtype=torch.float32, device=device)

    # Cached per-row importance (strong model) and the unmatched set for this pair.
    imp_s = {model_a: imp_a, model_b: imp_b}[strong]
    drops = imp_s["row_feature_drops"]            # (n_query, n_alive)
    feat_cols = [int(fi) for fi in imp_s["feature_indices"]]
    unmatched_set = set(unmatched.get(f"{strong}__{weak}", []))

    def _ranked_pool(members, train_rows):
        """Top-pool_n features in `members` by mean importance on `train_rows`,
        restricted to those firing on at least one train row."""
        imp_mean = drops[train_rows].mean(axis=0)
        fire_any = firing_mask[train_rows].any(axis=0)
        scored = [(feat_cols[i], imp_mean[i]) for i in range(len(feat_cols))
                  if feat_cols[i] in members and imp_mean[i] > 0 and fire_any[feat_cols[i]]]
        scored.sort(key=lambda x: -x[1])
        return [fi for fi, _ in scored[:args.pool_n]]

    # Build strong tail once (mirrors ablation_sweep).
    X_tr, y_tr, X_q, _, _, task_s = load_dataset_context(strong, dataset, splits)
    if y_tr.dtype == np.int32:
        y_tr = y_tr.astype(np.int64)
    cat_indices = None
    if strong in ("hyperfast", "tabpfn"):
        from data.preprocessing import load_preprocessed, CACHE_DIR
        try:
            cat_indices = load_preprocessed(strong, dataset, CACHE_DIR).cat_indices or None
        except Exception:
            pass
    torch.manual_seed(13); np.random.seed(13)
    tail = build_tail(strong, X_tr, y_tr, X_q, get_extraction_layer_taskaware(strong, dataset=dataset),
                      task_s, device, cat_indices=cat_indices,
                      target_name=splits.get(dataset, {}).get("target", "target"))
    use_mitra = isinstance(tail, MitraTail)
    use_seq = isinstance(tail, SEQUENTIAL_MODELS)
    common = dict(sae=saes[strong], tail=tail, activations=activations, X_query=X_q,
                  strong_preds=strong_preds, weak_preds=weak_preds, y_query=y_query,
                  task=task, data_std=data_std, firing_mask=firing_mask, device=device,
                  use_mitra=use_mitra, use_seq=use_seq, max_K=args.max_K)

    # Cross-fit over the active rows. One arm: UNMATCHED trained-SAE concepts.
    active_idx = np.where(active)[0]
    y_active = y_query[active_idx]
    is_loo = n_active < args.loo_threshold

    # Keep row identity so the honest OOF gc can be paired per-row against the
    # paper's circular per-row gc. Each active row is held out in exactly one fold.
    oof_row, oof_gc = [], []                       # absolute idx into n_query, honest gc
    fold_means, fold_feats, sel = [], [], Counter()

    for train_rows, held_rows in _make_folds(active_idx, y_active, task,
                                             args.n_folds, args.loo_threshold, args.seed):
        if not train_rows or not held_rows:
            continue
        pool = _ranked_pool(unmatched_set, train_rows)
        feats, _ = _greedy_select(pool, train_rows, m_max=args.m_max, **common)
        per_row = _eval_perrow(feats, held_rows, **common)
        final = per_row[:, -1] if feats else np.zeros(len(held_rows), dtype=np.float32)
        oof_row.extend(held_rows)
        oof_gc.extend(final.tolist())
        fold_means.append(float(final.mean()) if len(final) else 0.0)
        fold_feats.append(feats)
        sel.update(feats)

    n_folds_used = len(fold_means)

    # The paper's per-row circular gap-closed, joined by global row id on the active
    # rows, paired with the honest OOF gc row-for-row.
    abl = (Path(args.ablation_dir) if args.ablation_dir else FORWARD_ABLATION_DIR) / \
          f"{model_a}_vs_{model_b}" / f"{dataset}.npz"
    circ_by_row = {}
    paper_mean_gc = None
    if abl.exists():
        z = np.load(abl, allow_pickle=True)
        paper_mean_gc = float(z["mean_gap_closed"]) if "mean_gap_closed" in z else None
        if "gap_closed" in z and "row_indices" in z:
            cpos = {int(ri): i for i, ri in enumerate(z["row_indices"])}
            cgc = z["gap_closed"]
            for r in active_idx:
                gid = int(row_indices[r])
                if gid in cpos and np.isfinite(cgc[cpos[gid]]):
                    circ_by_row[int(r)] = float(cgc[cpos[gid]])

    honest_by_row = dict(zip([int(r) for r in oof_row], oof_gc))
    paired_rows = [int(row_indices[r]) for r in active_idx
                   if int(r) in honest_by_row and int(r) in circ_by_row]
    honest_arr = [honest_by_row[r] for r in active_idx
                  if int(r) in honest_by_row and int(r) in circ_by_row]
    circ_arr = [circ_by_row[int(r)] for r in active_idx
                if int(r) in honest_by_row and int(r) in circ_by_row]
    honest_arr = np.asarray(honest_arr, dtype=np.float32)
    circ_arr = np.asarray(circ_arr, dtype=np.float32)

    honest_mean = float(np.mean(oof_gc)) if oof_gc else float("nan")
    fold_std = float(np.std(fold_means)) if len(fold_means) > 1 else 0.0
    circ_mean_paired = float(circ_arr.mean()) if len(circ_arr) else float("nan")
    honest_mean_paired = float(honest_arr.mean()) if len(honest_arr) else float("nan")
    # Retention: fraction of the circular closure that survives cross-fitting.
    retention = (honest_mean_paired / circ_mean_paired
                 if circ_mean_paired and circ_mean_paired > 1e-8 else float("nan"))

    return {
        "status": "ok", "strong_model": strong, "weak_model": weak,
        "task": task, "metric_name": mname,
        "n_query": n_query, "n_active": n_active, "n_paired": len(paired_rows),
        "n_folds": n_folds_used, "loo": is_loo, "seed": args.seed,
        # honest (cross-fit) results
        "honest_oof_gc": honest_mean, "oof_gc_fold_std": fold_std,
        "fold_means": json.dumps(fold_means),
        "fold_sizes": json.dumps([len(f) for f in fold_feats]),
        "consensus_freq": json.dumps(
            {str(fi): c / n_folds_used for fi, c in sel.most_common()}),
        # circular (paper, on-disk) comparison, paired on the same active rows
        "circ_mean_paired": circ_mean_paired, "honest_mean_paired": honest_mean_paired,
        "retention": retention, "paper_mean_gap_closed": paper_mean_gc,
        # per-row paired arrays for downstream paired stats / distributions
        "paired_row_ids": np.asarray(paired_rows, dtype=np.int64),
        "honest_gc_perrow": honest_arr, "circ_gc_perrow": circ_arr,
    }


def main():
    p = argparse.ArgumentParser(description="Cross-fitted held-out concept generality")
    p.add_argument("--models", nargs=2, required=True, metavar="MODEL")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--loo-threshold", type=int, default=10,
                   help="use leave-one-out when active rows < this")
    p.add_argument("--m-max", type=int, default=8, help="max concepts in the frozen set")
    p.add_argument("--pool-n", type=int, default=12, help="candidate pool size per arm")
    p.add_argument("--min-gap", type=float, default=0.01)
    p.add_argument("--min-active", type=int, default=10,
                   help="skip datasets with fewer active rows than this")
    p.add_argument("--max-K", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--matching-file", type=Path, default=DEFAULT_MATCHING_FILE)
    p.add_argument("--importance-dir", default=None)
    p.add_argument("--ablation-dir", default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    model_a, model_b = sorted(args.models)
    pair_name = f"{model_a}_vs_{model_b}"
    splits = json.loads(SPLITS_PATH.read_text())

    saes, norm_stats, test_embeddings = {}, {}, {}
    for m in (model_a, model_b):
        sae, _ = load_sae(m, device=args.device); sae.eval()
        saes[m] = sae
        norm_stats[m] = load_norm_stats_matching(m)
        test_embeddings[m] = load_test_embeddings(m)

    unmatched = {}
    for src, tgt in [(model_a, model_b), (model_b, model_a)]:
        unmatched[f"{src}__{tgt}"] = get_unmatched_features(src, tgt, matching_file=args.matching_file)

    imp_dir = Path(args.importance_dir) if args.importance_dir else IMPORTANCE_DIR
    ds_a = {d.stem for d in (imp_dir / model_a).glob("*.npz")}
    ds_b = {d.stem for d in (imp_dir / model_b).glob("*.npz")}
    datasets = sorted(ds_a & ds_b)
    if args.datasets:
        datasets = [d for d in datasets if d in args.datasets]

    out_dir = (args.output_dir or OUTPUT_DIR) / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cross-fitted held-out generality: {pair_name}  ({len(datasets)} datasets, "
                f"{args.n_folds}-fold, LOO<{args.loo_threshold})")

    for i, ds in enumerate(datasets):
        out_path = out_dir / f"{ds}.npz"
        if args.resume and out_path.exists():
            logger.info(f"[{i+1}/{len(datasets)}] {ds}: SKIP (exists)"); continue
        logger.info(f"\n[{i+1}/{len(datasets)}] {ds}")
        t0 = time.time()
        try:
            result = run_dataset(model_a, model_b, ds, saes, splits, norm_stats,
                                 test_embeddings, unmatched, args.device, args)
            np.savez_compressed(str(out_path), **result)
            if result.get("status") == "ok":
                ret = result.get("retention")
                ret_s = f"{ret:.0%}" if isinstance(ret, float) and np.isfinite(ret) else "n/a"
                logger.info(
                    f"  {result['strong_model']}>{result['weak_model']} "
                    f"active={result['n_active']} paired={result['n_paired']} "
                    f"folds={result['n_folds']}{' (LOO)' if result['loo'] else ''} | "
                    f"honest OOF gc={result['honest_mean_paired']:.2f}"
                    f"±{result['oof_gc_fold_std']:.2f} vs "
                    f"circular={result['circ_mean_paired']:.2f} "
                    f"(retention {ret_s}) ({time.time()-t0:.0f}s)")
            else:
                logger.info(f"  status={result.get('status')} ({time.time()-t0:.0f}s)")
        except Exception as e:
            logger.error(f"  FAIL: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
