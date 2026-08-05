#!/usr/bin/env python3
"""Leave-one-out spectrum: how is a row's gap closure split across its concepts?

A concept is ACCEPTED by the greedy on its MARGINAL contribution -- injecting it
strictly reduced dist_to_strong given whatever was already in the delta. The ablation
ceiling measures something else: leave-one-out from the COMPLETED delta. With correlated
concepts these diverge, because the remaining concepts cover for the one removed. So a
tiny ceiling does not by itself mean the concept was worthless when accepted.

This computes LOO for every concept at a row, which distinguishes the possibilities:

  all LOO tiny, sum << deployed   -> heavy redundancy; no concept is individually
                                     necessary and single-concept attribution is not
                                     meaningful at this row, however good the search.
  c tiny, others large            -> c really is a minor contributor here.
  all comparable, total is small  -> c is an ordinary contributor to a small effect;
                                     the right normalisation is its share, not an
                                     absolute threshold.

Needs no donor forward: the delta is linear in the activations, so each LOO delta is
arithmetic, and only the recipient tail runs.

Usage:
    python -m scripts.rebuttal.loo_spectrum --donor tabicl --feat 158 --device cuda
"""
import argparse
import glob
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import (
    SPLITS_PATH, batched_intervention, build_tail, get_extraction_layer_taskaware,
    load_dataset_context, load_norm_stats, load_sae, load_test_embeddings,
)

FWD = PROJECT_ROOT / "output" / "rebuttal" / "forward_deltas"
ATOMS = PROJECT_ROOT / "output" / "transfer_caches" / "global_trained"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", default="tabicl")
    ap.add_argument("--feat", type=int, default=158)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--n-rows", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.use_deterministic_algorithms(True)

    best = None
    for f in sorted(glob.glob(str(FWD / "*" / "*.npz"))):
        if args.dataset and os.path.basename(f)[:-4] != args.dataset:
            continue
        z = np.load(f, allow_pickle=True)
        if str(z["strong_model"]) != args.donor or z["selected_features"].size == 0:
            continue
        sel = z["selected_features"]
        rows = [r for r in range(sel.shape[0])
                if args.feat in set(sel[r][sel[r] >= 0].tolist())]
        if rows and (best is None or len(rows) > best[0]):
            best = (len(rows), f, os.path.basename(f)[:-4], str(z["weak_model"]), rows, z)
    if best is None:
        print("no cell"); return
    _, path, dataset, recipient, rows, z = best
    print(f"{args.donor} f{args.feat} -> {recipient} / {dataset}  ({len(rows)} rows)\n")

    dd = np.asarray(z["deployed_delta"], dtype=np.float64)
    sel = np.asarray(z["selected_features"])
    zc = np.load(ATOMS / f"{args.donor}_to_{recipient}.npz", allow_pickle=True)
    V = np.asarray(zc["virtual_atoms"], dtype=np.float64)
    fmap = {int(f): i for i, f in enumerate(np.asarray(zc["feature_ids"]))}
    _, std_w = load_norm_stats(recipient, dataset, device=args.device)
    std_w = np.asarray(std_w.cpu(), dtype=np.float64)
    sae, _ = load_sae(args.donor, device=args.device)
    with torch.no_grad():
        A = sae.encode(torch.tensor(np.asarray(load_test_embeddings(args.donor)[dataset],
                                               dtype=np.float32),
                                    device=args.device)).cpu().numpy().astype(np.float64)

    splits = json.loads(SPLITS_PATH.read_text())
    Xtr, ytr, Xq, _, _, task = load_dataset_context(recipient, dataset, splits)
    if ytr.dtype == np.int32:
        ytr = ytr.astype(np.int64)
    layer = get_extraction_layer_taskaware(recipient, dataset=dataset)
    torch.manual_seed(13); np.random.seed(13)
    tail = build_tail(recipient, Xtr, ytr, Xq, layer, task, args.device,
                      target_name=splits.get(dataset, {}).get("target", "target"))
    from scripts.rebuttal.functional_decomposition import _gc
    from scripts.intervention.intervene_lib import (
        SEQUENTIAL_MODELS, batched_ablation_sequential)

    out = []
    for row in rows[:args.n_rows]:
        fids = [int(f) for f in np.unique(sel[row][sel[row] >= 0]) if int(f) in fmap]
        B = np.stack([V[fmap[f]] * std_w for f in fids])
        c, *_ = np.linalg.lstsq(B.T, dd[row], rcond=None)
        signs = np.sign(c)
        a = np.array([A[row, f] for f in fids])
        # deployed, then one LOO delta per concept
        variants = [dd[row]]
        for i in range(len(fids)):
            r = np.ones(len(fids)); r[i] = 0.0
            variants.append((signs * a * r) @ B)
        deltas = torch.tensor(np.vstack(variants), dtype=torch.float32, device=args.device)
        if isinstance(tail, SEQUENTIAL_MODELS):
            preds = np.asarray(batched_ablation_sequential(tail, Xq[row:row+1], deltas,
                                                           query_idx=row), dtype=np.float64)
        else:
            preds = np.asarray(batched_intervention(tail, Xq[row:row+1], deltas,
                                                    inject_context=False), dtype=np.float64)
        y = int(np.asarray(z["y_query"])[row])
        b, t = np.asarray(z["preds_weak"])[row], np.asarray(z["preds_strong"])[row]
        gc_dep = float(_gc(b, preds[0], t, y))
        loo = np.array([gc_dep - float(_gc(b, preds[i + 1], t, y)) for i in range(len(fids))])
        i_c = fids.index(args.feat)
        rank = int((loo > loo[i_c]).sum()) + 1
        print(f"  row {row:4d}  n_concepts={len(fids):3d}  gc_deployed={gc_dep:8.4f}")
        print(f"     LOO effects: median={np.median(loo):9.5f}  max={loo.max():9.5f}  "
              f"sum={loo.sum():9.5f}  (sum/gc = {loo.sum()/(gc_dep+1e-12):6.2f})")
        print(f"     concept f{args.feat}: LOO={loo[i_c]:9.5f}  rank {rank}/{len(fids)}  "
              f"share={loo[i_c]/(gc_dep+1e-12):6.2%} of deployed gc")
        out.append({"row": int(row), "n_concepts": len(fids), "gc_deployed": gc_dep,
                    "loo_target": float(loo[i_c]), "loo_median": float(np.median(loo)),
                    "loo_max": float(loo.max()), "loo_sum": float(loo.sum()),
                    "rank_of_target": rank})

    p = args.out or str(PROJECT_ROOT / "output" / "rebuttal" /
                        f"loo_spectrum_{args.donor}_f{args.feat}.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")
    print("\nsum/gc << 1 means the concepts are redundant: removing any one is covered by")
    print("the rest, so no single concept is individually necessary at that row.")


if __name__ == "__main__":
    main()
