#!/usr/bin/env python3
"""Transfer-panel decomposition: gap_closed vs top-m as the MAP and the
selection knobs vary, mirroring the ablation panel (pipeline_ablation.py).

Transfer injects a strong-model concept, mapped into the weak model's embedding
space, into the weak model and measures how much of the weak->strong gap it
closes. The ablation panel asked "what closes the strong->weak gap"; this panel
asks "does the learned concept correspondence (the map) earn its keep, or would
a shuffled / geometry-only / random map do as well."

Two axes, both reported as UNCLIPPED gap_closed (raw moved/gap) pooled row-
weighted over S-wins rows, on the m grid {1,2,5,10,20,50,|U|}:

  Selection arms (map fixed = learned ridge):  vary dictionary/ordering/
    granularity exactly as the ablation panel (the none arms; the greedy arms
    T1/T6 are read from the on-disk transfer sweep, not recomputed here).

  Map arms (selection fixed = importance, dataset, none):  vary the map
    M1 ridge          learned ridge on matched, sign-corrected landmark atoms
    M2 shuffled       same ridge pipeline, landmark target pairing permuted
    M3 procrustes     semi-orthogonal SVD(X^T Y) on <=500 paired SAE-train rows
    M4 random_orth    random semi-orthogonal (QR of a Gaussian)
    M5 schnaus        unpaired Wasserstein Procrustes (optional; needs the repo)

Every map produces a d_weak x d_strong matrix applied to the strong concept's
decoder atom; the injected direction is unit-normalized so map scale never
confounds the comparison (this also makes M1 == the ridge selection arm T3).
Maps M3/M5 are fit on the dataset's SAE-TRAINING embeddings so the scored
SAE-TEST rows never enter any map.

Mirrors pipeline_ablation.py: reuses its m-grid, pooling, and none-curve metric,
swapping ablation-of-strong for injection-into-weak. Usage:
    python -m scripts.intervention.transfer_decomposition --models tabpfn tabicl \
        --datasets credit-g --map ridge --device cuda
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from scripts.round_paths import (
    FORWARD_ABLATION_DIR, IMPORTANCE_DIR, IMPORTANCE_RANDOM_DIR,
    SYMMETRIC_TRANSFER_DIR, RESULTS_DIR, random_sae_dir, sae_training_dir,
)
from scripts.intervention.intervene_lib import (
    SPLITS_PATH, DEFAULT_SAE_DIR,
    load_sae, get_extraction_layer_taskaware, build_tail,
    load_dataset_context, load_test_embeddings,
    compute_per_row_loss, compute_importance_metric,
    batched_intervention, batched_intervention_sequential,
    MitraTail, SEQUENTIAL_MODELS,
)
from scripts.matching.utils import load_norm_stats as load_norm_stats_matching
from scripts.intervention.ablation_sweep import DEFAULT_MATCHING_FILE
from scripts.rebuttal.transfer_sweep_symmetric import (
    get_matched_pairs, get_unmatched_features,
)
from scripts.intervention.transfer_virtual_nodes import (
    extract_decoder_atoms, filter_landmarks, fit_concept_map,
)
# reuse the ablation panel's grid, metric helpers, and pooling contract
from scripts.intervention.pipeline_ablation import (
    M_GRID, M_LABELS, GC_TOLERANCE, EPS, _clip, _mean_curves, _widen_binary,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(str(FORWARD_ABLATION_DIR)).parent / "transfer_decomposition"
RANDOM_MATCHING_FILE = RESULTS_DIR / "sae_feature_matching_mnn_t0.001_random.json"
MIN_COSINE = 0.0        # filter_landmarks threshold (matches transfer_sweep default)
MAP_VARIANTS = ("ridge", "shuffled", "procrustes", "random_orth", "schnaus")


# ── map builders: each returns M (d_weak x d_strong) mapping strong atoms -> weak ──
def _ridge_map(ctx, shuffle=False):
    """Learned ridge on matched, sign-corrected landmark decoder atoms (M1/M2)."""
    m_pairs = ctx["matched_pairs"]
    src = ctx["atoms_s"][[si for si, _ in m_pairs]]
    tgt = ctx["atoms_w"][[ti for _, ti in m_pairs]]
    filt_src, filt_tgt, _, q = filter_landmarks(src, tgt, m_pairs, min_cosine=MIN_COSINE)
    if shuffle:
        rng = np.random.RandomState(0)
        filt_tgt = filt_tgt[rng.permutation(len(filt_tgt))]
    M, r2 = fit_concept_map(filt_src, filt_tgt, alpha=1.0)   # (d_weak, d_strong)
    return M, {"n_landmarks": int(q["n_kept"]), "r2": float(r2)}


def _procrustes_map(ctx):
    """Semi-orthogonal map from SVD(X^T Y) on <=500 paired SAE-training rows (M3)."""
    X, Y = ctx["train_emb_s"], ctx["train_emb_w"]          # (n, d_s), (n, d_w) normalized
    U, _, Vt = np.linalg.svd(X.T @ Y, full_matrices=False)  # d_s x k, k x d_w
    M_fwd = U @ Vt                                          # (d_s, d_w): X @ M_fwd ~ Y
    return M_fwd.T.astype(np.float32), {"n_rows": int(X.shape[0])}  # (d_weak, d_strong)


def _random_orth_map(ctx):
    """Random semi-orthogonal d_weak x d_strong via QR of a Gaussian (M4)."""
    d_s, d_w = ctx["d_strong"], ctx["d_weak"]
    rng = np.random.RandomState(0)
    G = rng.randn(max(d_s, d_w), min(d_s, d_w))
    Q, _ = np.linalg.qr(G)                                  # columns orthonormal
    M = Q[:, :min(d_s, d_w)]
    M = M if d_w >= d_s else M.T
    # shape it to (d_weak, d_strong)
    if M.shape != (d_w, d_s):
        M = M[:d_w, :d_s] if M.shape[0] >= d_w else M.T[:d_w, :d_s]
    return M.astype(np.float32), {}


def _schnaus_map(ctx):
    """Unpaired Wasserstein Procrustes (optional; needs the external repo)."""
    raise NotImplementedError(
        "schnaus (unpaired Wasserstein Procrustes) not installed; run M1-M4 first")


def build_map(variant, ctx):
    if variant == "ridge":
        return _ridge_map(ctx, shuffle=False)
    if variant == "shuffled":
        return _ridge_map(ctx, shuffle=True)
    if variant == "procrustes":
        return _procrustes_map(ctx)
    if variant == "random_orth":
        return _random_orth_map(ctx)
    if variant == "schnaus":
        return _schnaus_map(ctx)
    raise ValueError(f"unknown map variant {variant}")


def virtual_atoms(ctx, M):
    """{unmatched fi -> UNIT direction in normalized weak space} = normalize(M @ atom_s).
    Unit-normalizing removes each map's scale so the arms are comparable (and the
    injection magnitude is governed by the SAE activation a_s and weak std only)."""
    out = {}
    for fi in ctx["unmatched"]:
        v = M @ ctx["atoms_s"][fi]
        n = np.linalg.norm(v)
        if n > 1e-8:
            out[fi] = (v / n).astype(np.float32)
    return out


# ── setup (mirrors transfer_sweep_symmetric's per-dataset setup) ────────────────
def setup(model_a, model_b, dataset, device, matching_file, importance_dir,
          sae_dir, variant):
    imp_a = np.load(importance_dir / model_a / f"{dataset}.npz", allow_pickle=True)
    imp_b = np.load(importance_dir / model_b / f"{dataset}.npz", allow_pickle=True)
    preds_a = _widen_binary(imp_a["baseline_preds"])
    preds_b = _widen_binary(imp_b["baseline_preds"])
    assert np.array_equal(imp_a["row_indices"], imp_b["row_indices"])
    row_indices = imp_a["row_indices"]
    y_query = imp_a["y_query"]
    task = "classification" if preds_a.ndim == 2 else "regression"

    preds = {model_a: preds_a, model_b: preds_b}
    metric_a, _ = compute_importance_metric(y_query, preds_a, task)
    metric_b, _ = compute_importance_metric(y_query, preds_b, task)
    strong, weak = (model_a, model_b) if metric_a >= metric_b else (model_b, model_a)
    imp_s = imp_a if strong == model_a else imp_b

    # strong -> weak transfer: inject strong's concepts into the WEAK model.
    splits = json.loads(SPLITS_PATH.read_text())
    Xtr_w, ytr_w, Xq_w, _, _, task_w = load_dataset_context(weak, dataset, splits)
    if ytr_w.dtype == np.int32:
        ytr_w = ytr_w.astype(np.int64)
    layer_w = get_extraction_layer_taskaware(weak, dataset=dataset)
    cat_w = None
    if weak in ("hyperfast", "tabpfn"):
        from data.preprocessing import load_preprocessed, CACHE_DIR
        try:
            pre = load_preprocessed(weak, dataset, CACHE_DIR)
            cat_w = pre.cat_indices if pre.cat_indices else None
        except Exception:
            pass
    target_name = splits.get(dataset, {}).get("target", "target")
    torch.manual_seed(13); np.random.seed(13)
    sae_s, _ = load_sae(strong, sae_dir=sae_dir, device=device); sae_s.eval()
    sae_w, _ = load_sae(weak, sae_dir=sae_dir, device=device); sae_w.eval()
    tail_w = build_tail(weak, Xtr_w, ytr_w, Xq_w, layer_w, task_w, device,
                        cat_indices=cat_w, target_name=target_name)

    # strong SAE activations on the test rows (what fires, and how strongly)
    emb_s = load_test_embeddings(strong)[dataset]
    with torch.no_grad():
        acts_s = sae_s.encode(torch.tensor(emb_s, dtype=torch.float32, device=device))
    h_strong = acts_s.cpu().numpy()
    firing = h_strong > 0

    _, ds_std_w = load_norm_stats_matching(weak)[dataset]
    ds_std_w_t = torch.tensor(ds_std_w, dtype=torch.float32, device=device)

    atoms_s = extract_decoder_atoms(sae_s).numpy()          # (n_latent_s, d_strong)
    atoms_w = extract_decoder_atoms(sae_w).numpy()          # (n_latent_w, d_weak)
    m_pairs = get_matched_pairs(strong, weak, matching_file=matching_file)
    unmatched = [int(c) for c in get_unmatched_features(strong, weak, matching_file=matching_file)]
    feat_idx = [int(fi) for fi in imp_s["feature_indices"]]

    ctx = dict(
        strong=strong, weak=weak, task=task, dataset=dataset,
        y_query=y_query, row_indices=row_indices,
        strong_preds=preds[strong], weak_preds=preds[weak],
        tail_w=tail_w, X_query_w=Xq_w,
        use_seq=isinstance(tail_w, SEQUENTIAL_MODELS), use_mitra=isinstance(tail_w, MitraTail),
        h_strong=h_strong, firing=firing, ds_std_w=ds_std_w_t,
        atoms_s=atoms_s, atoms_w=atoms_w, matched_pairs=m_pairs,
        unmatched=[u for u in unmatched], feat_idx=feat_idx,
        drops=imp_s["row_feature_drops"],
        d_strong=atoms_s.shape[1], d_weak=atoms_w.shape[1],
    )
    # paired SAE-training embeddings (<=500) for geometry maps; scored rows excluded
    if variant in ("procrustes", "schnaus"):
        ctx["train_emb_s"], ctx["train_emb_w"] = _train_pairs(strong, weak, dataset, 500)

    M, info = build_map(variant, ctx)
    ctx["vatoms"] = virtual_atoms(ctx, M)
    # restrict unmatched to those with a usable mapped direction AND importance
    ctx["unmatched"] = [u for u in ctx["unmatched"] if u in ctx["vatoms"]]
    ctx["map_info"] = info
    return ctx


def _train_pairs(strong, weak, dataset, n_max):
    """Normalized SAE-TRAINING embeddings for strong & weak on this dataset's rows.

    The corpus is one (N, d) `embeddings` array sliced per dataset via
    `source_datasets`/`samples_per_dataset`; `norm_stats` carries per-dataset
    means/stds. Only carte/tabdpt have a training corpus on disk, so geometry
    maps (procrustes/schnaus) are only available for pairs of those models;
    otherwise this raises and the dataset is skipped for that map."""
    def _load(model):
        p = sae_training_dir() / f"{model}_taskaware_sae_training.npz"
        if not p.exists():
            raise RuntimeError(f"no SAE-training corpus for {model} "
                               f"(procrustes/schnaus need it)")
        z = np.load(p, allow_pickle=True)
        spd = z["samples_per_dataset"]                      # structured (dataset, count)
        names = [str(d) for d in spd["dataset"]]
        counts = np.asarray(spd["count"], dtype=int)
        if dataset not in names:
            raise RuntimeError(f"{dataset} not in {model} SAE-training corpus")
        offs = np.concatenate([[0], np.cumsum(counts)])
        i = names.index(dataset)
        sl = slice(offs[i], offs[i + 1])
        emb = z["embeddings"][sl]
        ridx = np.asarray(z["row_indices"])[sl]             # original dataset row ids
        ns = np.load(sae_training_dir() / f"{model}_taskaware_norm_stats.npz",
                     allow_pickle=True)
        nds = [str(x) for x in ns["datasets"]]
        j = nds.index(dataset)
        emb = (emb - np.asarray(ns["means"])[j]) / np.asarray(ns["stds"])[j]
        return emb, ridx
    Es, rs = _load(strong)
    Ew, rw = _load(weak)
    # pair rows by shared original row id (keep strong's order), cap at n_max
    pos_w = {int(r): k for k, r in enumerate(rw)}
    pair_s, pair_w = [], []
    for k, r in enumerate(rs):
        if int(r) in pos_w:
            pair_s.append(k); pair_w.append(pos_w[int(r)])
            if len(pair_s) >= n_max:
                break
    if len(pair_s) < 10:
        raise RuntimeError(f"too few paired SAE-training rows ({len(pair_s)}) "
                           f"for {strong}/{weak} on {dataset}")
    return Es[pair_s].astype(np.float32), Ew[pair_w].astype(np.float32)


# ── transfer forward + gap_closed toward the strong model ──────────────────────
def _row_preds(ctx, r, sets):
    """Inject each concept set's summed mapped delta into the weak model on row r."""
    n = len(sets)
    deltas = []
    for s in sets:
        d = torch.zeros(ctx["d_weak"], dtype=torch.float32, device=ctx["ds_std_w"].device)
        for fi in s:
            a_s = float(ctx["h_strong"][r, fi])
            v = torch.tensor(ctx["vatoms"][fi], device=d.device)
            d = d + a_s * v * ctx["ds_std_w"]
        deltas.append(d)
    deltas = torch.stack(deltas) if deltas else torch.zeros((0, ctx["d_weak"]))
    X_row = ctx["X_query_w"][r:r + 1]
    if ctx["use_seq"]:
        return batched_intervention_sequential(ctx["tail_w"], X_row, deltas, query_idx=r)
    return batched_intervention(ctx["tail_w"], X_row, deltas, inject_context=False)


def _gc(ctx, r, preds):
    """RAW gap_closed toward strong: fraction of the weak->strong true-class loss
    gap recovered by injection, UNCLIPPED (>1 = overshot past the strong model)."""
    sp, wp = ctx["strong_preds"][r], ctx["weak_preds"][r]
    if ctx["task"] == "classification":
        yr = int(ctx["y_query"][r])
        orig = -np.log(np.clip(wp[yr], EPS, 1 - EPS))       # weak loss (start)
        target = -np.log(np.clip(sp[yr], EPS, 1 - EPS))     # strong loss (goal)
        gap = target - orig
        out = [((-np.log(np.clip(p[yr], EPS, 1 - EPS)) - orig) / gap if abs(gap) > 1e-8 else 1.0)
               for p in preds]
    else:
        orig = float((wp - sp) ** 2)
        out = [(abs(float((np.asarray(p).ravel()[0] - sp) ** 2) - orig) / orig
                if orig > 1e-8 else 1.0) for p in preds]
    return np.asarray(out, dtype=np.float64)


# S-wins rows: the strong model beats the weak on this row's true class.
def _swins(ctx):
    sp, wp, y = ctx["strong_preds"], ctx["weak_preds"], ctx["y_query"]
    if ctx["task"] == "classification":
        yr = y.astype(int)
        s = sp[np.arange(len(y)), yr]; w = wp[np.arange(len(y)), yr]
        return (s > w + 1e-6)
    return (np.abs(sp - y) < np.abs(wp - y))


# ── candidate pools / orderings (strong concepts, ranked by strong importance) ──
def _ranked_imp_row(ctx, r):
    fi, dr, fire, vm = ctx["feat_idx"], ctx["drops"][r], ctx["firing"][r], ctx["vatoms"]
    scored = [(fi[i], dr[i]) for i in range(len(fi))
              if dr[i] > 0 and fire[fi[i]] and fi[i] in vm]
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored]


def _dataset_pool_positions(ctx, active_mask):
    fi, vm = ctx["feat_idx"], ctx["vatoms"]
    fires_any = ctx["firing"][active_mask].any(axis=0)
    return [i for i in range(len(fi)) if fi[i] in vm and fires_any[fi[i]]]


def _ranked_imp_dataset(ctx, active_mask):
    pos = _dataset_pool_positions(ctx, active_mask)
    mean_drop = ctx["drops"][active_mask].mean(axis=0)
    pos.sort(key=lambda i: -mean_drop[i])
    return [ctx["feat_idx"][i] for i in pos]


def _latents_dataset(ctx, active_mask):
    return [ctx["feat_idx"][i] for i in _dataset_pool_positions(ctx, active_mask)]


def _none_curve(ctx, active_rows, ranked_for_row):
    gc = {lab: [] for lab in M_LABELS}
    for r in active_rows:
        ranked = ranked_for_row(r)
        if not ranked:
            for lab in M_LABELS:
                gc[lab].append(0.0)
            continue
        ms = [min(m, len(ranked)) for m in M_GRID] + [len(ranked)]
        sets = [ranked[:m] for m in ms]
        preds = _row_preds(ctx, r, sets)
        vals = _gc(ctx, r, preds)
        for lab, v in zip(M_LABELS, vals):
            gc[lab].append(float(v))
    return {lab: np.asarray(v) for lab, v in gc.items()}


# ── driver ─────────────────────────────────────────────────────────────────────
def run(model_a, model_b, dataset, device, seeds, matching_file, quiet, out_dir,
        importance_dir, sae_dir, variant):
    t0 = time.time()
    ctx = setup(model_a, model_b, dataset, device, matching_file, importance_dir,
                sae_dir, variant)
    swins = _swins(ctx)
    active_rows = np.where(swins)[0]
    if len(active_rows) == 0:
        print(f"  {dataset}: SKIP (no S-wins rows)")
        return
    n = len(active_rows)

    imp_row = _none_curve(ctx, active_rows, lambda r: _ranked_imp_row(ctx, r))
    ranked_ds = _ranked_imp_dataset(ctx, swins)
    imp_ds = _none_curve(ctx, active_rows, lambda r, L=ranked_ds: L)
    uni_pool_ds = _latents_dataset(ctx, swins)
    uni_seeds = []
    for s in range(seeds):
        perm = list(uni_pool_ds)
        np.random.RandomState(1000 + s).shuffle(perm)
        uni_seeds.append(_none_curve(ctx, active_rows, lambda r, L=perm: L))
    uni_ds = _mean_curves(uni_seeds)

    rmean = lambda a: float(np.mean(a))
    if quiet:
        print(f"  {dataset:28.28} {ctx['strong']:8.8}->{ctx['weak']:8.8} n={n:>4} "
              f"|U|={len(uni_pool_ds):>4} map={variant:11.11}  UNCLIP@m=1 "
              f"imp_row={rmean(imp_row['1']):6.2f} imp_ds={rmean(imp_ds['1']):6.2f} "
              f"uni_ds={rmean(uni_ds['1']):6.2f}  [{time.time()-t0:.0f}s]")
    else:
        _report(dataset, ctx, imp_row, imp_ds, uni_ds, n, variant, t0)

    out_dir.mkdir(parents=True, exist_ok=True)
    save = dict(strong=ctx["strong"], weak=ctx["weak"], dataset=dataset, map=variant,
                active_rows=active_rows, map_info=json.dumps(ctx["map_info"]),
                uni_pool_ds=np.asarray(uni_pool_ds), m_labels=np.asarray(M_LABELS))
    for lab in M_LABELS:
        save[f"imp_row__{lab}"] = imp_row[lab]
        save[f"imp_ds__{lab}"] = imp_ds[lab]
        save[f"uni_ds__{lab}"] = uni_ds[lab]
    np.savez_compressed(out_dir / f"{dataset}.npz", **save)


def _report(dataset, ctx, imp_row, imp_ds, uni_ds, n, variant, t0):
    hdr = "  ".join(f"{lab:>5}" for lab in M_LABELS)
    print(f"\n=== {ctx['strong']}->{ctx['weak']} / {dataset}  map={variant}  "
          f"S-wins rows={n}  (landmarks={ctx['map_info']}) ===")
    print(f"UNCLIPPED gap_closed toward strong (raw; >1 = overshoot)")
    print(f"{'arm':26} {hdr}")
    for name, cur in [("importance, row,     none", imp_row),
                      ("importance, dataset, none", imp_ds),
                      ("uniform,    dataset, none", uni_ds)]:
        print(f"{name:26} " + "  ".join(f"{cur[lab].mean():>5.2f}" for lab in M_LABELS))
    print(f"[{time.time()-t0:.0f}s]")


def main():
    p = argparse.ArgumentParser(description="Transfer-panel decomposition (gc vs m, maps)")
    p.add_argument("--models", nargs=2, required=True)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--map", choices=MAP_VARIANTS, default="ridge")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--matching-file", type=Path, default=None)
    p.add_argument("--random", action="store_true",
                   help="random-init SAE dictionary (T5/T6 control)")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    model_a, model_b = sorted(args.models)
    pair = f"{model_a}_vs_{model_b}"

    importance_dir = IMPORTANCE_RANDOM_DIR if args.random else IMPORTANCE_DIR
    sae_dir = random_sae_dir() if args.random else DEFAULT_SAE_DIR
    matching_file = args.matching_file or (RANDOM_MATCHING_FILE if args.random
                                           else DEFAULT_MATCHING_FILE)
    tag = args.map + ("_random" if args.random else "")
    out_dir = (args.output_dir or (OUTPUT_DIR / tag)) / pair

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = sorted(pp.stem for pp in (FORWARD_ABLATION_DIR / pair).glob("*.npz"))
    print(f"{pair}: {len(datasets)} datasets  map={args.map}  random={args.random}")
    for i, ds in enumerate(datasets):
        if args.resume and (out_dir / f"{ds}.npz").exists():
            print(f"[{i+1}/{len(datasets)}] {ds}: SKIP (exists)")
            continue
        try:
            run(model_a, model_b, ds, args.device, args.seeds, matching_file,
                args.quiet, out_dir, importance_dir, sae_dir, args.map)
        except Exception as e:
            logger.error(f"[{i+1}/{len(datasets)}] FAIL {ds}: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
