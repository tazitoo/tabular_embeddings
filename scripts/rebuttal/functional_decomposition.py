#!/usr/bin/env python3
"""REBUTTAL: functional (not energy) decomposition of the transfer delta.

The subspace/energy analysis (subspace_analysis.py) measures where the deployed
delta's ENERGY sits relative to the recipient's activation-variance eigenbasis.
But energy in the variance basis need not translate to impact on the PREDICTION:
variance directions are data-driven, prediction impact depends on the head's
readout directions, and the two can diverge (a low-variance direction can be the
decision boundary). So the energy split can't settle "recombination vs novel
capacity" functionally.

This measures function directly. For each deployed transfer delta Delta (forward
transfer, recipient = weak model), split it against the recipient's active
subspace E (top-k_e eigenvectors capturing 90% of activation variance):

    Delta_active = E E^T Delta        (the dominant / "existing structure" part)
    Delta_tail   = Delta - Delta_active   (the low-variance / "novel" part)

Inject each SEPARATELY into the recipient and measure how much of the transfer's
gap-closure (recipient -> donor prediction) each one produces:

    gc_active : does the dominant-structure component move the prediction?
    gc_tail   : does the low-variance "novel" component move the prediction?
    gc_full   : the full delta (sanity vs the stored preds_intervened)

If gc_tail ~ 0 while gc_active ~ gc_full, the "novel" energy is functionally
inert -> recombination of existing structure does the work. If gc_tail is large
despite its low energy, low-variance directions ARE prediction-relevant ->
genuine functional novelty the energy view hid.

Reads deployed_delta from output/rebuttal/forward_deltas/<pair>/<dataset>.npz.
Needs the recipient base model (GPU).

Usage:
    python -m scripts.rebuttal.functional_decomposition --models tabpfn tabdpt --dataset credit-g
    python -m scripts.rebuttal.functional_decomposition --models tabpfn tabdpt          # all datasets w/ deltas
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import (
    SPLITS_PATH, get_extraction_layer_taskaware, build_tail,
    load_dataset_context, load_test_embeddings, batched_intervention,
)
from scripts.matching.utils import load_norm_stats as load_norm_stats_matching
from scripts.rebuttal.subspace_analysis import _eig_cov, _k_for_variance

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

FWD_DIR = PROJECT_ROOT / "output" / "rebuttal" / "forward_deltas"
OUT_DIR = PROJECT_ROOT / "output" / "rebuttal" / "functional_decomposition"
EPS = 1e-7


def _loss(p, y):
    return -np.log(np.clip(p[y], EPS, 1 - EPS))


def _gc(base_p, inter_p, target_p, y):
    """Gap closed from base->target by an intervention, per the transfer sweep."""
    ol = _loss(base_p, y); tl = _loss(target_p, y); il = _loss(inter_p, y)
    gap = ol - tl
    return float(np.clip((ol - il) / gap, 0.0, 1.0)) if gap > 1e-8 else np.nan


def run_dataset(strong, weak, dataset, device, emb_cache, norm_cache, fwd_dir=FWD_DIR):
    npz = fwd_dir / f"{min(strong,weak)}_vs_{max(strong,weak)}" / f"{dataset}.npz"
    if not npz.exists():
        return None
    d = np.load(npz, allow_pickle=True)
    if "deployed_delta" not in d.files:
        return None
    recipient = str(d["weak_model"])          # forward: recipient = weak
    if recipient != weak:
        # orientation in the npz can differ; trust the npz's recipient
        weak = recipient
    dd = np.asarray(d["deployed_delta"], dtype=np.float64)
    preds_strong = np.asarray(d["preds_strong"], dtype=np.float64)   # target (donor)
    preds_weak = np.asarray(d["preds_weak"], dtype=np.float64)       # recipient baseline
    if preds_strong.ndim != 2:            # classification-only (baseline-swap filter is argmax-based)
        return None
    y_query = np.asarray(d["y_query"]).astype(int)
    rows = np.where(np.linalg.norm(dd, axis=1) > 1e-12)[0]
    if len(rows) < 5:
        return None

    # Recipient active subspace E (top-k_e of raw-embedding variance).
    if recipient not in emb_cache:
        emb_cache[recipient] = load_test_embeddings(recipient)
        norm_cache[recipient] = load_norm_stats_matching(recipient)
    Xn = np.asarray(emb_cache[recipient][dataset], dtype=np.float64)
    mean, std = norm_cache[recipient][dataset]
    Xraw = Xn * np.asarray(std) + np.asarray(mean)
    _, lam_e, V_e = _eig_cov(Xraw, center=True)
    ke = max(1, min(_k_for_variance(lam_e, 0.90), V_e.shape[1]))
    E = V_e[:, :ke]                             # (d, ke)

    # Recipient tail (to re-run predictions under each injected component).
    splits = json.loads(SPLITS_PATH.read_text())
    Xtr, ytr, Xq, _, _, task = load_dataset_context(recipient, dataset, splits)
    if ytr.dtype == np.int32:
        ytr = ytr.astype(np.int64)
    layer = get_extraction_layer_taskaware(recipient, dataset=dataset)
    cat_indices = None
    if recipient in ("hyperfast", "tabpfn"):
        from data.preprocessing import load_preprocessed, CACHE_DIR
        try:
            cat_indices = load_preprocessed(recipient, dataset, CACHE_DIR).cat_indices or None
        except Exception:
            pass
    target_name = splits.get(dataset, {}).get("target", "target")
    torch.manual_seed(13); np.random.seed(13)
    tail = build_tail(recipient, Xtr, ytr, Xq, layer, task, device,
                      cat_indices=cat_indices, target_name=target_name)

    recs = []
    for r in rows:
        Delta = dd[r]
        d_act = E @ (E.T @ Delta)               # active-subspace component
        d_tail = Delta - d_act                  # low-variance / "novel" component
        deltas = torch.tensor(np.vstack([d_act, d_tail, Delta]),
                              dtype=torch.float32, device=device)
        preds = np.asarray(batched_intervention(tail, Xq[r:r+1], deltas,
                                                inject_context=False), dtype=np.float64)
        y = int(y_query[r])
        b, t = preds_weak[r], preds_strong[r]
        recs.append((
            _gc(b, preds[0], t, y),   # active
            _gc(b, preds[1], t, y),   # tail
            _gc(b, preds[2], t, y),   # full
            float(np.linalg.norm(d_act)**2 / (np.linalg.norm(Delta)**2 + 1e-12)),  # active energy frac
        ))
    A = np.array(recs)
    good = ~np.isnan(A[:, :3]).any(1)
    kept_rows = rows[good]
    A = A[good]
    if not len(A):
        return None
    return {
        "recipient": recipient, "donor": strong, "dataset": dataset, "n_rows": int(len(A)),
        "gc_active": float(A[:, 0].mean()), "gc_tail": float(A[:, 1].mean()),
        "gc_full": float(A[:, 2].mean()), "active_energy_frac": float(A[:, 3].mean()),
        # per-row arrays (row index into the n_query test set) so aggregation can
        # drop baseline-swap rows and match recipients across trained/random.
        "row_idx": [int(x) for x in kept_rows],
        "gc_active_rows": [float(x) for x in A[:, 0]],
        "gc_tail_rows": [float(x) for x in A[:, 1]],
        "gc_full_rows": [float(x) for x in A[:, 2]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs=2, required=True, metavar=("STRONG", "WEAK"))
    ap.add_argument("--dataset", default=None, help="single dataset; default = all with deltas")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--delta-dir", type=Path, default=FWD_DIR,
                    help="dir of deployed deltas (default forward_deltas; forward_deltas_random for the random arm)")
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    strong, weak = args.models
    pair = f"{min(strong,weak)}_vs_{max(strong,weak)}"
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = sorted(p.stem for p in (args.delta_dir / pair).glob("*.npz"))

    emb_cache, norm_cache = {}, {}
    results = []
    for ds in datasets:
        try:
            r = run_dataset(strong, weak, ds, args.device, emb_cache, norm_cache, fwd_dir=args.delta_dir)
        except Exception as e:
            logger.info(f"  {ds}: FAIL {e}")
            continue
        if r:
            results.append(r)
            logger.info(f"  {ds}: gc_active={r['gc_active']:.3f} gc_tail={r['gc_tail']:.3f} "
                        f"gc_full={r['gc_full']:.3f}  (active energy {r['active_energy_frac']:.2f}, "
                        f"n={r['n_rows']})")
    if not results:
        print("No datasets produced results."); return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{pair}.json").write_text(json.dumps(results, indent=2))

    a = np.array([r["gc_active"] for r in results])
    t = np.array([r["gc_tail"] for r in results])
    f = np.array([r["gc_full"] for r in results])
    ef = np.array([r["active_energy_frac"] for r in results])
    print(f"\n{'='*70}\nFUNCTIONAL DECOMPOSITION  {weak} <- {strong}  ({len(results)} datasets)\n{'='*70}")
    print(f"  gap closed by ACTIVE (dominant-structure) component: mean={a.mean():.3f}")
    print(f"  gap closed by TAIL ('novel' low-variance) component: mean={t.mean():.3f}")
    print(f"  gap closed by FULL delta:                            mean={f.mean():.3f}")
    print(f"  (active component holds {ef.mean():.0%} of the delta's ENERGY)")
    print(f"\n  => tail carries {ef.mean()*0+ (1-ef.mean()):.0%} of energy but "
          f"{t.mean()/max(f.mean(),1e-9):.0%} of the functional gap-closure")


if __name__ == "__main__":
    main()
