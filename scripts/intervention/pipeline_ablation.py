#!/usr/bin/env python3
"""Pipeline ablation: which knob actually closes the strong->weak gap?

The paper's per-row greedy (ablation_sweep.py) combines three choices. This
script ablates them one at a time, holding the model/pair/candidate-pool fixed,
so we can read off how much of the closed gap each choice is responsible for --
the non-circular version of "2-5 concepts close the gap".

Three knobs:
    granularity : row (each row its own set) vs dataset (one shared set)
    ordering    : importance (rank by LOO drop) vs uniform (random)
    selection   : greedy (adaptive search + 0.99 early-stop) vs none (fixed top-m)

Arms (the paper is the reference point; the other four are computed here):

    arm                        isolates
    (importance,row,greedy)    THE PAPER  -- read from disk, not recomputed
    (importance,row,none)      what the greedy search + early-stop add
    (importance,dataset,none)  what per-row tailoring adds over one shared set
    (uniform,dataset,none)     what the importance ranking adds
    (uniform,row,greedy)       whether ordering matters given the search

Candidate pools (fixed per arm, by first principles -- vary one knob):
    importance,row      : firing & unmatched & drop>0, ranked by drop desc
                          (exactly the paper's per-row candidate ranking)
    uniform,row         : firing & unmatched (no drop screen -- uniform has no
                          importance to screen on), random order
    importance,dataset  : unmatched & fires-on>=1-active-row, ranked by MEAN drop
                          over active rows; the same top-m set applied to every row
    uniform,dataset     : unmatched & fires-on>=1-active-row, random m, shared

Set-ablation gc is NOT additive from per-concept LOO drops, so every non-paper
arm needs real forward passes. This script reuses the library's forward/delta/gc
path (intervene_lib) WITHOUT touching ablation_sweep.py -- the paper scripts stay
the canonical reproducer. Two self-checks guard that the replicated path is
faithful (run with --validate):
    replay : ablate the paper's own selected_features -> gc must match on-disk
    greedy : rerun the greedy in importance order -> optimal_k/gc must match

Usage:
    python -m scripts.intervention.pipeline_ablation --models tabpfn tabicl \
        --datasets credit-g --device cuda --validate
"""
import argparse
import json
import logging
import time
from itertools import combinations as combos
from pathlib import Path

import numpy as np
import torch

from scripts.round_paths import (
    FORWARD_ABLATION_DIR, IMPORTANCE_DIR,
    FORWARD_ABLATION_RANDOM_DIR, IMPORTANCE_RANDOM_DIR, RESULTS_DIR, random_sae_dir,
)
from scripts.intervention.intervene_lib import (
    SPLITS_PATH, DEFAULT_SAE_DIR,
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

OUTPUT_DIR = Path(str(FORWARD_ABLATION_DIR)).parent / "pipeline_ablation"
# --random control: swap trained SAE/importance/matching/paper-arm for the random-
# init baselines (the same dirs the on-disk random greedy + transfer used), so the
# three none arms are apples-to-apples with the random greedy already on disk.
RANDOM_OUTPUT_DIR = Path(str(FORWARD_ABLATION_DIR)).parent / "pipeline_ablation_random"
RANDOM_MATCHING_FILE = RESULTS_DIR / "sae_feature_matching_mnn_t0.001_random.json"
M_GRID = (1, 2, 5, 10, 20, 50)          # top-m sweep for the "none" arms
# endpoint = |U|, the filtered pool size per dataset (unmatched & fires on >=1
# active row). Every row's pool is a subset of it, so top-|U| is each row's full
# list for the per-row arms and the whole shared set for the dataset arms.
M_LABELS = [str(m) for m in M_GRID] + ["|U|"]
GC_TOLERANCE = 0.99                      # greedy early-stop (matches paper default)
EPS = 1e-7


def _widen_binary(p):
    """Mitra-style (N,1) P(y=1) -> (N,2), matching ablation_sweep._widen_binary."""
    if p.ndim == 2 and p.shape[1] == 1:
        p1 = p.ravel().astype(np.float32)
        return np.column_stack([1.0 - p1, p1]).astype(np.float32)
    return p


# ── setup: replicate run_dataset's setup half (ablation_sweep.py:86-249) ───────
def setup(model_a, model_b, dataset, device, max_K, matching_file, importance_dir,
          sae_dir=DEFAULT_SAE_DIR):
    imp_a = np.load(importance_dir / model_a / f"{dataset}.npz", allow_pickle=True)
    imp_b = np.load(importance_dir / model_b / f"{dataset}.npz", allow_pickle=True)
    preds_a = _widen_binary(imp_a["baseline_preds"])
    preds_b = _widen_binary(imp_b["baseline_preds"])
    assert np.array_equal(imp_a["row_indices"], imp_b["row_indices"])
    row_indices = imp_a["row_indices"]
    y_query = imp_a["y_query"]
    n_query = len(y_query)
    task = "classification" if preds_a.ndim == 2 else "regression"

    preds = {model_a: preds_a, model_b: preds_b}
    metric_a, _ = compute_importance_metric(y_query, preds_a, task)
    metric_b, _ = compute_importance_metric(y_query, preds_b, task)
    strong, weak = (model_a, model_b) if metric_a >= metric_b else (model_b, model_a)
    imp_s = imp_a if strong == model_a else imp_b

    losses_s = compute_per_row_loss(y_query, preds[strong], task)
    losses_w = compute_per_row_loss(y_query, preds[weak], task)

    splits = json.loads(SPLITS_PATH.read_text())
    X_train, y_train, X_query, _, _, task_s = load_dataset_context(strong, dataset, splits)
    if y_train.dtype == np.int32:
        y_train = y_train.astype(np.int64)
    layer = get_extraction_layer_taskaware(strong, dataset=dataset)
    cat_indices = None
    if strong in ("hyperfast", "tabpfn"):
        from data.preprocessing import load_preprocessed, CACHE_DIR
        try:
            pre = load_preprocessed(strong, dataset, CACHE_DIR)
            cat_indices = pre.cat_indices if pre.cat_indices else None
        except Exception:
            pass
    target_name = splits.get(dataset, {}).get("target", "target")
    torch.manual_seed(13)
    np.random.seed(13)
    sae, _ = load_sae(strong, sae_dir=sae_dir, device=device)
    sae.eval()
    tail = build_tail(strong, X_train, y_train, X_query, layer, task_s, device,
                      cat_indices=cat_indices, target_name=target_name)

    emb = load_test_embeddings(strong)[dataset]
    with torch.no_grad():
        acts = sae.encode(torch.tensor(emb, dtype=torch.float32, device=device))
    firing = (acts > 0).cpu().numpy()

    _, ds_std = load_norm_stats_matching(strong)[dataset]
    data_std_t = torch.tensor(ds_std, dtype=torch.float32, device=device)

    unmatched = set(int(c) for c in get_unmatched_features(strong, weak, matching_file=matching_file))
    feat_idx = [int(fi) for fi in imp_s["feature_indices"]]   # position -> latent id

    return dict(
        strong=strong, weak=weak, task=task, dataset=dataset,
        y_query=y_query, n_query=n_query, row_indices=row_indices,
        baseline_preds_s=preds[strong], weak_preds=preds[weak],
        baseline_loss_s=losses_s, weak_loss=losses_w,
        sae=sae, acts=acts, firing=firing, data_std_t=data_std_t,
        tail=tail, X_query=X_query, max_K=max_K,
        use_seq=isinstance(tail, SEQUENTIAL_MODELS), use_mitra=isinstance(tail, MitraTail),
        unmatched=unmatched, feat_idx=feat_idx, drops=imp_s["row_feature_drops"],
    )


# ── forward + gap_closed (identical math to ablation_sweep) ────────────────────
def _row_preds(ctx, r, sets):
    """Ablate each latent-id set on row r in ONE batched forward; return preds."""
    sae, h_row = ctx["sae"], ctx["acts"][r]
    n = len(sets)
    with torch.no_grad():
        recon_full = sae.decode(h_row.unsqueeze(0))
        h_batch = h_row.unsqueeze(0).expand(n, -1).clone()
        for c, s in enumerate(sets):
            for fi in s:
                h_batch[c, fi] = 0.0
        recon_abl = sae.decode(h_batch)
        deltas = (recon_abl - recon_full) * ctx["data_std_t"].unsqueeze(0)
    X_row = ctx["X_query"][r:r + 1]
    if ctx["use_seq"]:
        return batched_ablation_sequential(ctx["tail"], X_row, deltas, query_idx=r)
    return batched_ablation(ctx["tail"], X_row, deltas, max_K=ctx["max_K"])


def _clip(a):
    return np.clip(a, 0.0, 1.0)


def _gc(ctx, r, preds):
    """RAW gap_closed per pred: fraction of the strong->weak loss gap recovered on
    the true class, UNCLIPPED. ablation_sweep stores min(1,max(0,.)) of this; the
    raw ratio exceeds 1 when an ablation overshoots the weak model's loss (<0 if it
    moves away). Callers clamp with _clip to match the paper's stored gap_closed."""
    row_sp, row_wp = ctx["baseline_preds_s"][r], ctx["weak_preds"][r]
    if ctx["task"] == "classification":
        yr = int(ctx["y_query"][r])
        orig = -np.log(np.clip(row_sp[yr], EPS, 1 - EPS))
        target = -np.log(np.clip(row_wp[yr], EPS, 1 - EPS))
        gap = target - orig
        out = [((-np.log(np.clip(p[yr], EPS, 1 - EPS)) - orig) / gap if gap > 1e-8 else 1.0)
               for p in preds]
    else:
        orig = float((row_sp - row_wp) ** 2)
        out = [(abs(float((np.asarray(p).ravel()[0] - row_wp) ** 2) - orig) / orig
                if orig > 1e-8 else 1.0) for p in preds]
    return np.asarray(out, dtype=np.float64)


# ── candidate pools / orderings ───────────────────────────────────────────────
def _ranked_imp_row(ctx, r):
    """firing & unmatched & drop>0, ranked by drop desc (the paper's ranking)."""
    fi, dr, fire, um = ctx["feat_idx"], ctx["drops"][r], ctx["firing"][r], ctx["unmatched"]
    scored = [(fi[i], dr[i]) for i in range(len(fi))
              if dr[i] > 0 and fire[fi[i]] and fi[i] in um]
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored]


def _pool_uni_row(ctx, r):
    """firing & unmatched (no drop screen)."""
    fi, fire, um = ctx["feat_idx"], ctx["firing"][r], ctx["unmatched"]
    return [fi[i] for i in range(len(fi)) if fire[fi[i]] and fi[i] in um]


def _dataset_pool_positions(ctx, active_mask):
    """positions of unmatched concepts that fire on >=1 active row."""
    fi, um = ctx["feat_idx"], ctx["unmatched"]
    fires_any = ctx["firing"][active_mask].any(axis=0)
    return [i for i in range(len(fi)) if fi[i] in um and fires_any[fi[i]]]


def _ranked_imp_dataset(ctx, active_mask):
    pos = _dataset_pool_positions(ctx, active_mask)
    mean_drop = ctx["drops"][active_mask].mean(axis=0)
    pos.sort(key=lambda i: -mean_drop[i])
    return [ctx["feat_idx"][i] for i in pos]


def _latents_dataset(ctx, active_mask):
    return [ctx["feat_idx"][i] for i in _dataset_pool_positions(ctx, active_mask)]


# ── arm runners ────────────────────────────────────────────────────────────────
def _none_curve(ctx, active_rows, ranked_for_row):
    """Fixed top-m (no search): gc at each m per active row. ranked_for_row(r)
    returns that row's ordered latent list (per-row arms) or the shared list
    (dataset arms)."""
    gc = {lab: [] for lab in M_LABELS}
    for r in active_rows:
        ranked = ranked_for_row(r)
        if not ranked:
            for lab in M_LABELS:
                gc[lab].append(0.0)
            continue
        sets = [ranked[:min(m, len(ranked))] for m in M_GRID] + [ranked[:]]
        gcs = _gc(ctx, r, _row_preds(ctx, r, sets))
        for lab, g in zip(M_LABELS, gcs):
            gc[lab].append(float(g))
    return {lab: np.asarray(v) for lab, v in gc.items()}


def _mean_curves(curves):
    """Average a list of per-seed {label: per-row array} dicts, per row per m."""
    labs = curves[0].keys()
    return {lab: np.mean([c[lab] for c in curves], axis=0) for lab in labs}


def _greedy(ctx, r, ranked, gc_tolerance=GC_TOLERANCE):
    """Faithful copy of ablation_sweep's per-row greedy (lines 335-432), but with
    the candidate list `ranked` supplied by the caller (importance or uniform)."""
    K = len(ranked)
    if K == 0:
        return [], 0.0
    sae, h_row = ctx["sae"], ctx["acts"][r]
    row_sp, row_wp = ctx["baseline_preds_s"][r], ctx["weak_preds"][r]
    if ctx["task"] == "classification":
        yr = int(ctx["y_query"][r])
        target = -np.log(np.clip(row_wp[yr], EPS, 1 - EPS))
        def dist(p):
            return (-np.log(np.clip(p[yr], EPS, 1 - EPS)) - target) ** 2
    else:
        def dist(p):
            return float((np.asarray(p).ravel()[0] - row_wp) ** 2)
    with torch.no_grad():
        recon_full = sae.decode(h_row.unsqueeze(0))
    batch_size = min(12, K)
    current = dist(row_sp)
    best_pred = row_sp
    accepted = ()
    offset = 0
    while offset < K:
        batch_end = min(offset + batch_size, K)
        subsets = []
        for order in range(1, min(4, (batch_end - offset) + 1)):
            subsets.extend(combos(range(offset, batch_end), order))
        if not subsets:
            break
        with torch.no_grad():
            h_batch = h_row.unsqueeze(0).expand(len(subsets), -1).clone()
            for c, subset in enumerate(subsets):
                for j in accepted:
                    h_batch[c, ranked[j]] = 0.0
                for j in subset:
                    h_batch[c, ranked[j]] = 0.0
            deltas = (sae.decode(h_batch) - recon_full) * ctx["data_std_t"].unsqueeze(0)
        X_row = ctx["X_query"][r:r + 1]
        if ctx["use_seq"]:
            cps = batched_ablation_sequential(ctx["tail"], X_row, deltas, query_idx=r)
        else:
            cps = batched_ablation(ctx["tail"], X_row, deltas, max_K=ctx["max_K"])
        best_c, best_c_dist = None, current
        for c in range(len(subsets)):
            d = dist(cps[c])
            if d < best_c_dist:
                best_c, best_c_dist = c, d
        if best_c is not None:
            accepted = tuple(accepted) + subsets[best_c]
            current = best_c_dist
            best_pred = cps[best_c]
            if _clip(_gc(ctx, r, [best_pred])[0]) >= gc_tolerance:  # goal is clamped
                break
        offset = batch_end
    sel = [ranked[j] for j in accepted]
    gc = float(_gc(ctx, r, [best_pred])[0]) if accepted else 0.0  # raw (may be >1)
    return sel, gc


def _greedy_arm(ctx, active_rows, pool_for_row, seeds):
    """Per-row greedy with a uniform-order candidate list, averaged over seeds."""
    per_row_gc, per_row_k = [], []
    for r in active_rows:
        pool = pool_for_row(r)
        gcs, ks = [], []
        for s in range(seeds):
            p = list(pool)
            np.random.RandomState(2000 + s).shuffle(p)
            sel, gc = _greedy(ctx, r, p)
            gcs.append(gc)
            ks.append(len(sel))
        per_row_gc.append(float(np.mean(gcs)))
        per_row_k.append(float(np.mean(ks)))
    return np.asarray(per_row_gc), np.asarray(per_row_k)


# ── validation (forward path + search faithfulness vs on-disk paper) ───────────
def _validate(ctx, abl, active_rows):
    gc_disk = abl["gap_closed"]
    sel_disk = abl["selected_features"]
    k_disk = abl["optimal_k"].astype(int)

    # replay: ablate the paper's own selected sets -> gc must match on-disk
    replay_diff = []
    for r in active_rows:
        sel = [int(f) for f in sel_disk[r][:k_disk[r]] if int(f) >= 0]
        if not sel:
            continue
        g = _clip(_gc(ctx, r, _row_preds(ctx, r, [sel]))[0])
        replay_diff.append(abs(g - float(gc_disk[r])))
    replay_diff = np.asarray(replay_diff)

    # greedy: rerun greedy in importance order -> k and gc must match on-disk
    k_mine, gc_mine, k_same = [], [], 0
    for r in active_rows:
        sel, gc = _greedy(ctx, r, _ranked_imp_row(ctx, r))
        k_mine.append(len(sel))
        gc_mine.append(gc)
        k_same += int(len(sel) == k_disk[r])
    k_mine, gc_mine = np.asarray(k_mine), _clip(np.asarray(gc_mine))
    gc_greedy_diff = np.abs(gc_mine - gc_disk[active_rows])

    print("\n── validation vs on-disk paper arm ──")
    print(f"  replay (paper sets):   max|Δgc|={replay_diff.max():.4f}  "
          f"mean={replay_diff.mean():.4f}  (n={len(replay_diff)})")
    print(f"  greedy (importance):   k match {k_same}/{len(active_rows)} rows  "
          f"max|Δgc|={gc_greedy_diff.max():.4f}  mean={gc_greedy_diff.mean():.4f}")
    return dict(replay_max=float(replay_diff.max()), replay_mean=float(replay_diff.mean()),
                greedy_k_match=int(k_same), greedy_gc_max=float(gc_greedy_diff.max()))


# ── driver ─────────────────────────────────────────────────────────────────────
def run(model_a, model_b, dataset, device, seeds, matching_file, do_validate,
        skip_greedy, quiet, out_dir, fwd_dir=FORWARD_ABLATION_DIR,
        importance_dir=IMPORTANCE_DIR, sae_dir=DEFAULT_SAE_DIR):
    pair = f"{model_a}_vs_{model_b}"
    abl_path = fwd_dir / pair / f"{dataset}.npz"
    if not abl_path.exists():
        print(f"  {dataset}: SKIP (no paper arm on disk)")
        return
    abl = np.load(abl_path, allow_pickle=True)
    if "optimal_k" not in abl:
        print(f"  {dataset}: SKIP (models tied -- no active rows)")
        return
    k_disk = abl["optimal_k"].astype(int)
    active_mask = k_disk > 0
    if active_mask.sum() == 0:
        print(f"  {dataset}: SKIP (0 active rows)")
        return

    t0 = time.time()
    ctx = setup(model_a, model_b, dataset, device, 512, matching_file, importance_dir,
                sae_dir=sae_dir)
    assert str(abl["strong_model"]) == ctx["strong"], "strong/weak orientation mismatch"
    assert np.array_equal(abl["row_indices"], ctx["row_indices"]), "row index mismatch"
    active_rows = np.where(active_mask)[0]
    gc_disk = abl["gap_closed"][active_mask]
    n = len(active_rows)

    vinfo = _validate(ctx, abl, active_rows) if do_validate else None

    # three "none" arms (fixed top-m, no search): raw (unclipped) gc per row
    imp_row = _none_curve(ctx, active_rows, lambda r: _ranked_imp_row(ctx, r))
    ranked_ds = _ranked_imp_dataset(ctx, active_mask)
    imp_ds = _none_curve(ctx, active_rows, lambda r, L=ranked_ds: L)
    uni_pool_ds = _latents_dataset(ctx, active_mask)
    uni_ds_seeds = []
    for s in range(seeds):
        perm = list(uni_pool_ds)
        np.random.RandomState(1000 + s).shuffle(perm)
        uni_ds_seeds.append(_none_curve(ctx, active_rows, lambda r, L=perm: L))
    uni_ds = _mean_curves(uni_ds_seeds)

    uni_row_gc = uni_row_k = None
    if not skip_greedy:
        uni_row_gc, uni_row_k = _greedy_arm(ctx, active_rows, lambda r: _pool_uni_row(ctx, r), seeds)

    cmean = lambda a: float(np.mean(_clip(a)))
    rmean = lambda a: float(np.mean(a))
    reach = lambda a: float(np.mean(a >= GC_TOLERANCE))

    if quiet:
        print(f"  {dataset:30.30} strong={ctx['strong']:9.9} n={n:>4} |U|={len(uni_pool_ds):>4}  "
              f"UNCLIPPED@m=1  imp_row={rmean(imp_row['1']):7.2f}  imp_ds={rmean(imp_ds['1']):6.2f}  "
              f"uni_ds={rmean(uni_ds['1']):6.2f}  [{time.time()-t0:.0f}s]")
    else:
        _report_full(dataset, ctx, imp_row, imp_ds, uni_ds, uni_row_gc, uni_row_k,
                     ranked_ds, uni_pool_ds, gc_disk, k_disk[active_mask], n, t0,
                     cmean, rmean, reach)

    out_dir.mkdir(parents=True, exist_ok=True)
    save = dict(
        strong=ctx["strong"], weak=ctx["weak"], dataset=dataset,
        active_rows=active_rows, gc_disk=gc_disk, k_disk=k_disk[active_mask],
        ranked_ds=np.asarray(ranked_ds), uni_pool_ds=np.asarray(uni_pool_ds),
        m_labels=np.asarray(M_LABELS),
    )
    for lab in M_LABELS:
        save[f"imp_row__{lab}"] = imp_row[lab]
        save[f"imp_ds__{lab}"] = imp_ds[lab]
        save[f"uni_ds__{lab}"] = uni_ds[lab]
    if not skip_greedy:
        save["uni_row_gc"] = uni_row_gc
        save["uni_row_k"] = uni_row_k
    if vinfo:
        save["validation"] = json.dumps(vinfo)
    np.savez_compressed(out_dir / f"{dataset}.npz", **save)


def _report_full(dataset, ctx, imp_row, imp_ds, uni_ds, uni_row_gc, uni_row_k,
                 ranked_ds, uni_pool_ds, gc_disk, k_active, n, t0,
                 cmean, rmean, reach):
    def row(curve, fn):
        return "  ".join(f"{fn(curve[lab]):>4.2f}" for lab in M_LABELS)
    hdr = "  ".join(f"{lab:>4}" for lab in M_LABELS)
    print(f"\n=== {ctx['strong']}(strong) vs {ctx['weak']} / {dataset}  active rows={n} ===")

    print(f"\nclamped gap_closed (paper convention, min(1,max(0,.)))")
    print(f"{'arm':38} {hdr}")
    print("-" * (38 + len(hdr) + 2))
    print(f"{'importance, row,     none':38} {row(imp_row, cmean)}")
    print(f"{'importance, dataset, none':38} {row(imp_ds, cmean)}")
    print(f"{'uniform,    dataset, none':38} {row(uni_ds, cmean)}")

    print(f"\nUNCLIPPED gap_closed (raw moved/gap; >1 = overshoot)")
    print(f"{'arm':38} {hdr}")
    print("-" * (38 + len(hdr) + 2))
    print(f"{'importance, row,     none':38} {row(imp_row, rmean)}")
    print(f"{'importance, dataset, none':38} {row(imp_ds, rmean)}")
    print(f"{'uniform,    dataset, none':38} {row(uni_ds, rmean)}")

    print(f"\nfraction of rows that ACTUALLY reach gc>={GC_TOLERANCE} (no clamp hiding it)")
    print(f"{'importance, row,     none':38} {row(imp_row, reach)}")
    print(f"{'importance, dataset, none':38} {row(imp_ds, reach)}")

    g1 = imp_row["1"]
    print(f"\nimportance,row,none @ m=1 per-row raw gc (n={n}): "
          f"min={g1.min():.2f} median={np.median(g1):.2f} mean={g1.mean():.2f} max={g1.max():.2f}")
    print(f"  rows reaching goal (>= {GC_TOLERANCE}): {int((g1>=GC_TOLERANCE).sum())}/{n}   "
          f"overshoot (>1): {int((g1>1.0).sum())}/{n}   "
          f"undershoot (< {GC_TOLERANCE}): {int((g1<GC_TOLERANCE).sum())}/{n}")

    if uni_row_gc is not None:
        print(f"\n{'uniform,    row,     greedy':38} gc(clamped)={cmean(uni_row_gc):.2f}  "
              f"gc(raw)={rmean(uni_row_gc):.2f}  mean k={uni_row_k.mean():.2f}")
    print(f"{'PAPER: importance, row, greedy (disk)':38} gc(clamped)={gc_disk.mean():.2f}  "
          f"mean k={k_active.mean():.2f}  median k={np.median(k_active):.0f}")
    print(f"\n|U| = {len(ranked_ds)}  (unmatched & fires on >=1 active row; the m-grid endpoint)")
    print(f"[{time.time() - t0:.0f}s]")


def main():
    p = argparse.ArgumentParser(description="Pipeline-knob ablation of the per-row greedy")
    p.add_argument("--models", nargs=2, required=True)
    p.add_argument("--datasets", nargs="+", default=None,
                   help="default: every dataset with a paper arm on disk for this pair")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", type=int, default=5, help="random draws for the uniform arms")
    p.add_argument("--matching-file", type=Path, default=None)
    p.add_argument("--random", action="store_true",
                   help="random-init SAE control: read the random SAE/importance/"
                        "matching/paper-arm dirs (the same inputs the on-disk random "
                        "greedy used); write to pipeline_ablation_random/")
    p.add_argument("--validate", action="store_true",
                   help="check the replicated forward/greedy path against on-disk paper arm")
    p.add_argument("--skip-greedy", action="store_true",
                   help="skip the uniform,row,greedy arm -- compute the three none arms only")
    p.add_argument("--quiet", action="store_true", help="one summary line per dataset (sweep)")
    p.add_argument("--resume", action="store_true", help="skip datasets already written")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    model_a, model_b = sorted(args.models)
    pair = f"{model_a}_vs_{model_b}"

    fwd_dir = FORWARD_ABLATION_RANDOM_DIR if args.random else FORWARD_ABLATION_DIR
    importance_dir = IMPORTANCE_RANDOM_DIR if args.random else IMPORTANCE_DIR
    sae_dir = random_sae_dir() if args.random else DEFAULT_SAE_DIR
    default_out = RANDOM_OUTPUT_DIR if args.random else OUTPUT_DIR
    matching_file = args.matching_file or (RANDOM_MATCHING_FILE if args.random
                                           else DEFAULT_MATCHING_FILE)
    out_dir = (args.output_dir or default_out) / pair

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = sorted(pp.stem for pp in (fwd_dir / pair).glob("*.npz"))
    print(f"{pair}: {len(datasets)} datasets  seeds={args.seeds}  "
          f"skip_greedy={args.skip_greedy}  random={args.random}")
    for i, ds in enumerate(datasets):
        if args.resume and (out_dir / f"{ds}.npz").exists():
            print(f"[{i+1}/{len(datasets)}] {ds}: SKIP (exists)")
            continue
        try:
            run(model_a, model_b, ds, args.device, args.seeds, matching_file,
                args.validate, args.skip_greedy, args.quiet, out_dir,
                fwd_dir=fwd_dir, importance_dir=importance_dir, sae_dir=sae_dir)
        except Exception as e:
            logger.error(f"[{i+1}/{len(datasets)}] FAIL {ds}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
