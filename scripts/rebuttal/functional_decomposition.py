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
transfer, recipient = weak model), split it against the recipient's ON-MANIFOLD
subspace E — the top-k_e eigenvectors capturing 90% of activation variance, a
linear proxy for the data manifold's dominant directions:

    Delta_on  = E E^T Delta            (ON-manifold: high-variance directions)
    Delta_off = Delta - Delta_on       (OFF-manifold: orthogonal low-variance)

Inject each SEPARATELY into the recipient and measure how much of the transfer's
gap-closure (recipient -> donor prediction) each one produces:

    gc_on_manifold  : does the high-variance (on-manifold) component move it?
    gc_off_manifold : does the low-variance (off-manifold) component move it?
    gc_full         : the full delta (sanity vs the stored preds_intervened)

If gc_off ~ 0 while gc_on ~ gc_full, off-manifold energy is functionally inert ->
recombination of existing (on-manifold) structure does the work. If gc_off is
large despite its low energy, off-manifold directions ARE prediction-relevant ->
genuine functional novelty the energy view hid. ("on/off-manifold" here always
means the high/low-variance split above, distinct from the nearest-neighbour
data-proximity notion in transfer_direction.py.)

Reads deployed_delta from output/rebuttal/forward_deltas/<pair>/<dataset>.npz.
Needs the recipient base model (GPU).

Usage:
    python -m scripts.rebuttal.functional_decomposition --models tabpfn tabdpt --dataset credit-g
    python -m scripts.rebuttal.functional_decomposition --models tabpfn tabdpt          # all datasets w/ deltas
"""
import argparse
import json
import logging
import os
from pathlib import Path

# Must precede CUDA initialisation, hence before torch is imported. Required by
# torch.use_deterministic_algorithms for cuBLAS ops; see docs/reproducibility.md.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import (
    SPLITS_PATH, get_extraction_layer_taskaware, build_tail,
    load_dataset_context, load_test_embeddings, batched_intervention,
    batched_ablation_sequential, SEQUENTIAL_MODELS,
)
from scripts.matching.utils import load_norm_stats as load_norm_stats_matching
from scripts.rebuttal.subspace_analysis import _eig_cov, _k_for_variance

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

FWD_DIR = PROJECT_ROOT / "output" / "rebuttal" / "forward_deltas"
OUT_DIR = PROJECT_ROOT / "output" / "rebuttal" / "functional_decomposition"
EPS = 1e-7


def _gc(base_p, inter_p, strong_p, y):
    """Gap closed base->strong by an intervention, matching the transfer sweep.
    Classification: cross-entropy toward y_true. Regression: squared distance to
    the strong (donor) prediction."""
    base_p = np.asarray(base_p); inter_p = np.asarray(inter_p); strong_p = np.asarray(strong_p)
    if base_p.ndim >= 1 and base_p.size > 1:                 # classification prob vector
        ol = -np.log(np.clip(base_p[y], EPS, 1 - EPS))
        tl = -np.log(np.clip(strong_p[y], EPS, 1 - EPS))
        il = -np.log(np.clip(inter_p[y], EPS, 1 - EPS))
        gap = ol - tl
        return float(np.clip((ol - il) / gap, 0.0, 1.0)) if gap > 1e-8 else np.nan
    orig = (float(base_p) - float(strong_p)) ** 2           # regression: distance to strong pred
    best = (float(inter_p) - float(strong_p)) ** 2
    return float(np.clip(1.0 - best / orig, 0.0, 1.0)) if orig > 1e-12 else np.nan


def run_dataset(strong, weak, dataset, device, emb_cache, norm_cache, fwd_dir=FWD_DIR,
                var_threshold=0.90):
    npz = fwd_dir / f"{min(strong,weak)}_vs_{max(strong,weak)}" / f"{dataset}.npz"
    if not npz.exists():
        return None
    d = np.load(npz, allow_pickle=True)
    if "deployed_delta" not in d.files:
        return None
    recipient = str(d["weak_model"])          # forward: recipient = weak (per-dataset, from the npz)
    donor = str(d["strong_model"])            # forward: donor  = strong (per-dataset, from the npz)
    if donor == recipient:
        raise ValueError(f"{dataset}: degenerate delta, donor == recipient == {donor}")
    if {donor, recipient} != {strong, weak}:
        raise ValueError(
            f"{dataset}: npz models {{{donor}, {recipient}}} do not match requested pair "
            f"{{{strong}, {weak}}} -- wrong --models or wrong delta dir")
    dd = np.asarray(d["deployed_delta"], dtype=np.float64)
    preds_strong = np.asarray(d["preds_strong"], dtype=np.float64)   # target (donor)
    preds_weak = np.asarray(d["preds_weak"], dtype=np.float64)       # recipient baseline
    # all task types now (strong-wins population, matching the paper's gc). y_query
    # is only used as a class index for classification; ignored for regression.
    y_query = np.asarray(d["y_query"])
    y_query = y_query.astype(int) if preds_strong.ndim == 2 else y_query
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
    ke = max(1, min(_k_for_variance(lam_e, var_threshold), V_e.shape[1]))
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

    # on-manifold = projection onto the recipient's high-variance active subspace
    # E (top eigenvectors capturing 90% of embedding variance, a linear proxy for
    # the data manifold's dominant directions); off-manifold = the orthogonal,
    # low-variance complement.
    recs = []
    for r in rows:
        Delta = dd[r]
        d_on = E @ (E.T @ Delta)                # on-manifold (high-variance) component
        d_off = Delta - d_on                    # off-manifold (low-variance) component
        deltas = torch.tensor(np.vstack([d_on, d_off, Delta]),
                              dtype=torch.float32, device=device)
        # Query-only injection of the 3 delta variants. CARTE (and the other
        # SEQUENTIAL_MODELS) have no `recapture`; they inject via a central-node
        # hook, so route them through the same per-row batched path the ablation
        # pipeline uses (batched_ablation_sequential -> CARTETail.predict_row_batched).
        # Everything else uses the generic recapture path (inject_context=False =
        # query positions only, matching the sequential models' central-node inject).
        if isinstance(tail, SEQUENTIAL_MODELS):
            preds = np.asarray(batched_ablation_sequential(tail, Xq[r:r+1], deltas,
                                                           query_idx=r), dtype=np.float64)
        else:
            preds = np.asarray(batched_intervention(tail, Xq[r:r+1], deltas,
                                                    inject_context=False), dtype=np.float64)
        y = int(y_query[r])
        b, t = preds_weak[r], preds_strong[r]
        recs.append((
            _gc(b, preds[0], t, y),   # on-manifold
            _gc(b, preds[1], t, y),   # off-manifold
            _gc(b, preds[2], t, y),   # full
            float(np.linalg.norm(d_on)**2 / (np.linalg.norm(Delta)**2 + 1e-12)),  # on-manifold energy frac
        ))
    A = np.array(recs)
    good = ~np.isnan(A[:, :3]).any(1)
    kept_rows = rows[good]
    A = A[good]
    if not len(A):
        return None
    return {
        "recipient": recipient, "donor": donor, "dataset": dataset, "n_rows": int(len(A)),
        "var_threshold": float(var_threshold), "ke": int(ke), "emb_dim": int(V_e.shape[1]),
        "gc_on_manifold": float(A[:, 0].mean()), "gc_off_manifold": float(A[:, 1].mean()),
        "gc_full": float(A[:, 2].mean()), "on_manifold_energy": float(A[:, 3].mean()),
        # per-row arrays (row index into the n_query test set) so aggregation can
        # drop baseline-swap rows and match recipients across trained/random.
        "row_idx": [int(x) for x in kept_rows],
        "gc_on_manifold_rows": [float(x) for x in A[:, 0]],
        "gc_off_manifold_rows": [float(x) for x in A[:, 1]],
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
    ap.add_argument("--var-threshold", type=float, default=0.90,
                    help="cumulative activation-variance fraction defining the on-manifold "
                         "subspace E (default 0.90). Sweep this to test the split's sensitivity.")
    args = ap.parse_args()
    # Seeding alone does not make carte reproducible: it is the only recipient
    # whose tail is TRAINED, and CARTE re-seeds torch internally anyway. This is
    # the operative knob (covers the scatter/index_add ops PyG uses), and it is
    # verified not to perturb the other five models. Pairs with the
    # CUBLAS_WORKSPACE_CONFIG set at the top of this module.
    # See docs/reproducibility.md.
    torch.use_deterministic_algorithms(True)
    strong, weak = args.models
    pair = f"{min(strong,weak)}_vs_{max(strong,weak)}"
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = sorted(p.stem for p in (args.delta_dir / pair).glob("*.npz"))

    emb_cache, norm_cache = {}, {}
    results = []
    for ds in datasets:
        # No blanket except: a real failure must crash with its traceback rather
        # than be silently dropped (that swallow once hid every carte-recipient
        # dataset). "Nothing to compute" is signalled by run_dataset -> None.
        r = run_dataset(strong, weak, ds, args.device, emb_cache, norm_cache,
                        fwd_dir=args.delta_dir, var_threshold=args.var_threshold)
        if r:
            results.append(r)
            logger.info(f"  {ds}: gc_on={r['gc_on_manifold']:.3f} gc_off={r['gc_off_manifold']:.3f} "
                        f"gc_full={r['gc_full']:.3f}  (on-manifold energy {r['on_manifold_energy']:.2f}, "
                        f"n={r['n_rows']})")
        else:
            logger.info(f"  {ds}: no result (no delta file or <5 active rows)")
    if not results:
        print("No datasets produced results."); return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{pair}.json").write_text(json.dumps(results, indent=2))

    on = np.array([r["gc_on_manifold"] for r in results])
    off = np.array([r["gc_off_manifold"] for r in results])
    f = np.array([r["gc_full"] for r in results])
    ef = np.array([r["on_manifold_energy"] for r in results])
    print(f"\n{'='*70}\nFUNCTIONAL DECOMPOSITION  {weak} <- {strong}  ({len(results)} datasets)\n{'='*70}")
    print(f"  gap closed by ON-MANIFOLD  (high-variance) component: mean={on.mean():.3f}")
    print(f"  gap closed by OFF-MANIFOLD (low-variance)  component: mean={off.mean():.3f}")
    print(f"  gap closed by FULL delta:                            mean={f.mean():.3f}")
    print(f"  (on-manifold component holds {ef.mean():.0%} of the delta's ENERGY)")
    print(f"\n  => off-manifold carries {(1-ef.mean()):.0%} of energy but "
          f"{off.mean()/max(f.mean(),1e-9):.0%} of the functional gap-closure")


if __name__ == "__main__":
    main()
