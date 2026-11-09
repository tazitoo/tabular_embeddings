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
SKIP_RECIPIENTS = {"mitra"}   # weak models whose forwards are intractable here


class _SkipDataset(Exception):
    """Dataset intentionally skipped (not an error) -- e.g. intractable recipient."""


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


def virtual_atoms(ctx, M, scale="unit", pool=None):
    """{strong fi -> direction in normalized weak space} = normalize(M @ atom_s),
    over `pool` strong concept ids (default ctx['unmatched']), scaled per `scale`:
      'unit'    : unit direction (map scale removed) -- comparable across MAP arms.
      'natural' : the paper/greedy scaling -- unit_dir * ||atom_s|| * median matched
                  (weak/strong) norm ratio. Preserves the DICTIONARY's own injection
                  magnitude (trained vs random SAE), for the norm/magnitude control.
    Injection magnitude is then this * the SAE activation a_s * weak std."""
    nr = float(ctx.get("median_norm_ratio", 1.0))
    pool = ctx["unmatched"] if pool is None else pool
    out = {}
    for fi in pool:
        a = ctx["atoms_s"][fi]
        v = M @ a
        n = np.linalg.norm(v)
        if n <= 1e-8:
            continue
        u = v / n
        if scale == "natural":
            u = u * float(np.linalg.norm(a)) * nr
        out[fi] = u.astype(np.float32)
    return out


# ── setup (mirrors transfer_sweep_symmetric's per-dataset setup) ────────────────
def setup(model_a, model_b, dataset, device, matching_file, importance_dir,
          sae_dir, variant, scale="unit", arm="unmatched"):
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

    # Mitra as RECIPIENT is intractable here: on 24 GB it OOM-retries every
    # inference (reducing its support set), and the none-curve path fires
    # thousands of forwards per dataset (vs the on-disk greedy's handful). Skip
    # these datasets; the greedy T1/T6 references for them stay on disk.
    if weak in SKIP_RECIPIENTS:
        raise _SkipDataset(f"recipient={weak} intractable (OOM-retry forwards)")

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
    n_ls, n_lw = atoms_s.shape[0], atoms_w.shape[0]
    # Drop matched/unmatched ids that fall outside the loaded SAE latent ranges:
    # some matching files carry stale concept ids from a larger pre-rebuild SAE
    # (matching_file_provenance). Ablation only ever indexed the strong SAE, so it
    # never tripped; transfer indexes the WEAK SAE's atoms via matched targets.
    m_pairs = [(int(si), int(ti)) for si, ti in
               get_matched_pairs(strong, weak, matching_file=matching_file)
               if int(si) < n_ls and int(ti) < n_lw]
    unmatched = [int(c) for c in
                 get_unmatched_features(strong, weak, matching_file=matching_file)
                 if int(c) < n_ls]
    # feat_idx stays aligned with drops columns (same SAE as importance -> in range)
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
    # matched-concept (weak/strong) norm ratio -- the paper/greedy injection scale
    # (used only for scale='natural'); computed from this SAE's matched landmarks.
    _fs, _ft, _, _ = filter_landmarks(atoms_s[[si for si, _ in m_pairs]],
                                      atoms_w[[ti for _, ti in m_pairs]], m_pairs,
                                      min_cosine=MIN_COSINE)
    _sn = np.linalg.norm(_fs, axis=1); _tn = np.linalg.norm(_ft, axis=1)
    _ok = (_sn > 1e-8) & (_tn > 1e-8)
    ctx["median_norm_ratio"] = float(np.median(_tn[_ok] / _sn[_ok])) if _ok.sum() else 1.0

    ctx["arm"] = arm
    if arm == "matched_wdec":
        # No map: amplify W's OWN usage of each matched concept. Matched concepts
        # cofire, so W already fires ti on the S-relevant rows; we turn it up by its
        # own activation and decode through W's OWN atom (underweighting-as-operation).
        # delta = a_ti^W * atoms_w[ti] * std_w -- scale-free (all in W's SAE), no map,
        # no csign; sidesteps every cross-model correspondence AND cross-SAE scale question.
        emb_w = load_test_embeddings(weak)[dataset]
        with torch.no_grad():
            acts_w = sae_w.encode(torch.tensor(emb_w, dtype=torch.float32, device=device))
        ctx["h_weak"] = acts_w.cpu().numpy()
        ctx["pair_of"] = {int(si): int(ti) for si, ti in m_pairs}
        ctx["vatoms"] = {int(si): atoms_w[ti].astype(np.float32) for si, ti in m_pairs}
        ctx["unmatched"] = [int(si) for si, _ in m_pairs if int(si) in ctx["vatoms"]]
        ctx["map_info"] = {"arm": "matched_wdec", "n_pairs": len(m_pairs)}
        return ctx

    # paired SAE-training embeddings (<=500) for geometry maps; scored rows excluded
    if variant in ("procrustes", "schnaus"):
        ctx["train_emb_s"], ctx["train_emb_w"] = _train_pairs(strong, weak, dataset, 500)

    # map-based arms: inject UNMATCHED strong concepts (default) or the MATCHED ones
    # (matched_map -- the concepts the map was fit on, directly parallel to unmatched)
    pool = ([int(si) for si, _ in m_pairs] if arm == "matched_map"
            else ctx["unmatched"])
    M, info = build_map(variant, ctx)
    ctx["vatoms"] = virtual_atoms(ctx, M, scale=scale, pool=pool)
    # iteration pool = concepts with a usable mapped direction AND importance
    ctx["unmatched"] = [u for u in pool if u in ctx["vatoms"]]
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
    # need enough ALIGNED rows for a usable d_strong x d_weak Procrustes map;
    # aligned pairs give ~500, misaligned (tabdpt, partial datasets) far fewer ->
    # skip those datasets for the geometry maps.
    if len(pair_s) < 100:
        raise RuntimeError(f"too few paired SAE-training rows ({len(pair_s)}) "
                           f"for {strong}/{weak} on {dataset}")
    return Es[pair_s].astype(np.float32), Ew[pair_w].astype(np.float32)


# ── transfer forward + gap_closed toward the strong model ──────────────────────
def _forward(ctx, r, deltas):
    """Inject each delta into the weak model on row r -> one prediction per delta."""
    X_row = ctx["X_query_w"][r:r + 1]
    if ctx["use_seq"]:
        return batched_intervention_sequential(ctx["tail_w"], X_row, deltas, query_idx=r)
    return batched_intervention(ctx["tail_w"], X_row, deltas, inject_context=False)


def _concept_delta(ctx, r, fi):
    """Signed raw-weak-space delta for injecting concept fi on row r.
    The per-dataset sign (ctx['csign']) resolves the arbitrary SAE/map sign once
    per concept -- the dataset-granularity analog of the greedy's per-row flip.
    matched_wdec: no map, no csign. Matched concepts COFIRE, so W already has the
    concept active on these rows; we amplify W's OWN usage of the matched partner ti
    by its own activation (in W's units -- no cross-SAE scale gap), direction = W's
    own decoder atom. delta = a_ti^W * d_ti^W * std_w (one extra unit of what W fires;
    top-m sweeps how many matched concepts are turned up)."""
    v = torch.tensor(ctx["vatoms"][fi], device=ctx["ds_std_w"].device)
    if ctx.get("arm") == "matched_wdec":
        ti = ctx["pair_of"][fi]
        amp = float(ctx["h_weak"][r, ti])          # W's own activation of ti
        return amp * v * ctx["ds_std_w"]
    a_s = float(ctx["h_strong"][r, fi])
    sgn = ctx.get("csign", {}).get(fi, 1.0)
    return a_s * sgn * v * ctx["ds_std_w"]


def _row_preds(ctx, r, sets):
    """Inject each concept set's summed (signed) mapped delta on row r."""
    deltas = []
    for s in sets:
        d = torch.zeros(ctx["d_weak"], dtype=torch.float32, device=ctx["ds_std_w"].device)
        for fi in s:
            d = d + _concept_delta(ctx, r, fi)
        deltas.append(d)
    deltas = torch.stack(deltas) if deltas else torch.zeros((0, ctx["d_weak"]))
    return _forward(ctx, r, deltas)


def _dataset_signs(ctx, active_rows, chunk=96, max_rows=24, max_concepts=128):
    """Choose each concept's injection sign ONCE per dataset: the sign (+/-) whose
    single-concept injection yields higher mean gap_closed toward strong over the
    active rows where it fires. Applied uniformly to all rows -- no per-row search,
    so the none arms match the greedy's sign freedom at dataset granularity.

    The sign is a dataset-level majority vote, estimated from a bounded sample of
    rows AND only for the top-max_concepts concepts by importance -- otherwise it
    costs 2*n_firing forwards per row, which wedges mitra-weak pairs (~1300
    unmatched, with mitra OOM-retrying each forward). Deep-tail concepts (reached
    only at the |U| endpoint) default to +sign. The +/- deltas within a row are
    forwarded in chunks to bound the sequential (carte) batch and VRAM."""
    # rank the mapped pool by dataset-mean importance; the none curves only reach
    # the top concepts at m<=50, so signs past max_concepts don't move those points.
    pos = {fi: i for i, fi in enumerate(ctx["feat_idx"])}
    mean_drop = ctx["drops"][active_rows].mean(axis=0)
    pool = sorted(ctx["vatoms"].keys(),
                  key=lambda fi: -(mean_drop[pos[fi]] if fi in pos else 0.0))[:max_concepts]
    acc = {fi: [0.0, 0.0] for fi in pool}          # [sum gc(+), sum gc(-)]
    pool_set = set(pool)
    sign_rows = (np.random.RandomState(7).choice(active_rows, max_rows, replace=False)
                 if len(active_rows) > max_rows else active_rows)
    for r in sign_rows:
        firing = [fi for fi in pool if ctx["firing"][r, fi]]
        if not firing:
            continue
        deltas, meta = [], []
        for fi in firing:
            a_s = float(ctx["h_strong"][r, fi])
            v = torch.tensor(ctx["vatoms"][fi], device=ctx["ds_std_w"].device)
            base = a_s * v * ctx["ds_std_w"]
            deltas.append(base); meta.append((fi, 0))
            deltas.append(-base); meta.append((fi, 1))
        for i in range(0, len(deltas), chunk):
            gc = _gc(ctx, r, _forward(ctx, r, torch.stack(deltas[i:i + chunk])))
            for (fi, slot), g in zip(meta[i:i + chunk], gc):
                acc[fi][slot] += float(g)
    return {fi: (1.0 if s_plus >= s_minus else -1.0) for fi, (s_plus, s_minus) in acc.items()}


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
def _deployed_delta(ctx, active_rows, ranked_ds, m):
    """The SUMMED injection actually added to the weak model per active row at menu
    size m (imp,dataset ordering) -- the deployed-delta analog of the greedy's
    deployed_delta, in RAW weak space. _concept_delta already returns raw-space
    deltas (normalized-space dir * std_w), with per-dataset signs for the map arms
    and W's own usage for matched_wdec. Rows whose top-m concepts don't fire give a
    zero vector and are dropped by the downstream norm filter."""
    sel = ranked_ds[:m]
    out = np.zeros((len(active_rows), ctx["d_weak"]), dtype=np.float32)
    for k, r in enumerate(active_rows):
        d = torch.zeros(ctx["d_weak"], dtype=torch.float32, device=ctx["ds_std_w"].device)
        for fi in sel:
            if ctx["firing"][r, fi] if ctx.get("arm") != "matched_wdec" else True:
                d = d + _concept_delta(ctx, r, fi)
        out[k] = d.cpu().numpy()
    return out


def run(model_a, model_b, dataset, device, seeds, matching_file, quiet, out_dir,
        importance_dir, sae_dir, variant, scale="unit", arm="unmatched", dump_m=None):
    t0 = time.time()
    try:
        ctx = setup(model_a, model_b, dataset, device, matching_file, importance_dir,
                    sae_dir, variant, scale=scale, arm=arm)
    except _SkipDataset as e:
        print(f"  {dataset}: SKIP ({e})")
        return
    swins = _swins(ctx)
    active_rows = np.where(swins)[0]
    if len(active_rows) == 0:
        print(f"  {dataset}: SKIP (no S-wins rows)")
        return
    n = len(active_rows)

    # resolve each concept's injection sign once per dataset (see _dataset_signs);
    # matched_wdec has no sign freedom (the activation difference is directional)
    if arm == "matched_wdec":
        ctx["csign"] = {}
    else:
        ctx["csign"] = _dataset_signs(ctx, active_rows)
    n_flip = sum(1 for s in ctx["csign"].values() if s < 0)

    # deployed-delta dump: the summed injection the model actually saw at menu m
    # (imp,dataset). Reuses the exact signs; skips the none-curve forwards entirely.
    if dump_m is not None:
        ranked_ds = _ranked_imp_dataset(ctx, swins)
        dd = _deployed_delta(ctx, active_rows, ranked_ds, dump_m)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"{dataset}.npz",
                            deployed_delta=dd, strong_model=ctx["strong"],
                            weak_model=ctx["weak"], dataset=dataset, menu_m=dump_m,
                            active_rows=active_rows, n_sign_flip=n_flip,
                            n_pool_signed=len(ctx["csign"]))
        print(f"  {dataset:28.28} {ctx['strong']:8.8}->{ctx['weak']:8.8} n={n:>4} "
              f"m={dump_m} dumped deployed_delta {dd.shape}  [{time.time()-t0:.0f}s]")
        return

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
              f"|U|={len(uni_pool_ds):>4} flip={n_flip}/{len(ctx['csign'])} map={variant:9.9}  "
              f"UNCLIP@m=1 imp_row={rmean(imp_row['1']):6.2f} imp_ds={rmean(imp_ds['1']):6.2f} "
              f"uni_ds={rmean(uni_ds['1']):6.2f}  [{time.time()-t0:.0f}s]")
    else:
        _report(dataset, ctx, imp_row, imp_ds, uni_ds, n, variant, t0)

    out_dir.mkdir(parents=True, exist_ok=True)
    save = dict(strong=ctx["strong"], weak=ctx["weak"], dataset=dataset, map=variant,
                active_rows=active_rows, map_info=json.dumps(ctx["map_info"]),
                n_sign_flip=n_flip, n_pool_signed=len(ctx["csign"]),
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
    p.add_argument("--scale", choices=["unit", "natural"], default="unit",
                   help="injection scale: unit (map comparison) or natural "
                        "(paper/greedy: atom_norm x matched weak/strong ratio) for "
                        "the dictionary magnitude control")
    p.add_argument("--arm", choices=["unmatched", "matched_map", "matched_wdec"],
                   default="unmatched",
                   help="which concepts to inject: unmatched strong concepts via the "
                        "map (default); matched strong concepts via the SAME map "
                        "(matched_map); or matched concepts via W's OWN decoder at S's "
                        "activation level, no map (matched_wdec -- underweighting test)")
    p.add_argument("--dump-deltas", type=int, default=None, metavar="M",
                   help="instead of the gc curves, dump the SUMMED deployed delta at "
                        "menu size M (imp,dataset ordering) for the deployed-delta "
                        "subspace analysis; writes to the <tag>_deltas dir")
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
    if args.arm == "matched_wdec":
        tag = "matched_wdec" + ("_random" if args.random else "")   # no map, no scale
    else:
        tag = (args.map + ("_random" if args.random else "")
               + ("_natural" if args.scale == "natural" else "")
               + ("_matched" if args.arm == "matched_map" else ""))
    if args.dump_deltas is not None:
        tag += "_deltas"
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
                args.quiet, out_dir, importance_dir, sae_dir, args.map,
                scale=args.scale, arm=args.arm, dump_m=args.dump_deltas)
        except Exception as e:
            logger.error(f"[{i+1}/{len(datasets)}] FAIL {ds}: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
