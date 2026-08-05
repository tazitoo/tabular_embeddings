#!/usr/bin/env python3
"""Does re-injecting the stored deployed_delta reproduce the transfer's gap_closed?

The patch readout computes gc by pushing deployed_delta through the recipient tail and
comparing against preds_weak/preds_strong. The transfer already recorded gap_closed for
the same rows. If the two disagree, every ceiling and capture number is measured on a
path that does not reproduce the run it claims to explain.

Motivating discrepancy: the cached gap_closed is nonzero on ~100% of accepted rows, yet
the readout reported gc_deployed = 0.0000 on several of them.

Usage:
    python -m scripts.rebuttal.check_gc_reproduction --donor tabpfn --device cuda
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
    load_dataset_context,
)

FWD = PROJECT_ROOT / "output" / "rebuttal" / "forward_deltas"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", default="tabpfn")
    ap.add_argument("--dataset", default="MIC")
    ap.add_argument("--n-rows", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.use_deterministic_algorithms(True)

    pick = None
    for f in sorted(glob.glob(str(FWD / "*" / "*.npz"))):
        if os.path.basename(f)[:-4] != args.dataset:
            continue
        z = np.load(f, allow_pickle=True)
        if str(z["strong_model"]) == args.donor and z["selected_features"].size:
            pick = (f, str(z["weak_model"]), z); break
    if pick is None:
        print("no cell"); return
    path, recipient, z = pick
    dd = np.asarray(z["deployed_delta"], dtype=np.float64)
    gcc = np.asarray(z["gap_closed"], dtype=np.float64)
    pw, ps, yq = z["preds_weak"], z["preds_strong"], z["y_query"]
    sw = np.asarray(z["strong_wins"], dtype=bool)
    rows = [r for r in range(len(dd)) if np.abs(dd[r]).any()][:args.n_rows]
    print(f"{args.donor} -> {recipient} / {args.dataset}   {len(rows)} rows\n")

    splits = json.loads(SPLITS_PATH.read_text())
    Xtr, ytr, Xq, _, _, task = load_dataset_context(recipient, dataset := args.dataset, splits)
    if ytr.dtype == np.int32:
        ytr = ytr.astype(np.int64)
    layer = get_extraction_layer_taskaware(recipient, dataset=dataset)
    cat_idx = None
    if recipient in ("hyperfast", "tabpfn"):
        from data.preprocessing import load_preprocessed, CACHE_DIR
        try:
            cat_idx = load_preprocessed(recipient, dataset, CACHE_DIR).cat_indices or None
        except Exception:
            pass
    torch.manual_seed(13); np.random.seed(13)
    tail = build_tail(recipient, Xtr, ytr, Xq, layer, task, args.device, cat_indices=cat_idx,
                      target_name=splits.get(dataset, {}).get("target", "target"))
    from scripts.rebuttal.functional_decomposition import _gc
    from scripts.intervention.intervene_lib import (
        SEQUENTIAL_MODELS, batched_ablation_sequential)

    # Log the PREDICTION difference, not just the derived gc: gc clamps to [0,1] and
    # divides by gap, so a gc difference could come from a tiny prediction change, a
    # near-zero denominator, or a clamp, and those are indistinguishable downstream.
    # preds_intervened is cached, so the transfer's own prediction is directly
    # comparable. gap and moved are logged because gc is unstable when gap is small.
    pi = np.asarray(z["preds_intervened"], dtype=np.float64)
    EPS = 1e-7
    print(f"  {'row':>5s} {'|pred-cached|':>14s} {'gap':>9s} {'moved_cache':>12s} "
          f"{'moved_mine':>11s} {'gc_cache':>9s} {'gc_mine':>9s} {'clamp':>7s}")
    diffs, pdiffs = [], []
    for r in rows:
        d = torch.tensor(dd[r][None, :], dtype=torch.float32, device=args.device)
        if isinstance(tail, SEQUENTIAL_MODELS):
            p = np.asarray(batched_ablation_sequential(tail, Xq[r:r+1], d, query_idx=r),
                           dtype=np.float64)
        else:
            p = np.asarray(batched_intervention(tail, Xq[r:r+1], d, inject_context=False),
                           dtype=np.float64)
        y = int(np.asarray(yq)[r])
        g = float(_gc(np.asarray(pw)[r], p[0], np.asarray(ps)[r], y))
        pd = float(np.abs(p[0] - pi[r]).max())        # my injection vs the transfer's own
        bw, bs = np.asarray(pw)[r], np.asarray(ps)[r]
        if bw.ndim >= 1 and bw.size > 1:              # classification
            ol = -np.log(np.clip(bw[y], EPS, 1 - EPS))
            tl = -np.log(np.clip(bs[y], EPS, 1 - EPS))
            mv_mine = ol - (-np.log(np.clip(p[0][y], EPS, 1 - EPS)))
            mv_cache = ol - (-np.log(np.clip(pi[r][y], EPS, 1 - EPS)))
            gap = ol - tl
        else:
            gap = float((float(bw) - float(bs)) ** 2)
            mv_mine = gap - float((float(p[0]) - float(bs)) ** 2)
            mv_cache = gap - float((float(pi[r]) - float(bs)) ** 2)
        clamp = "low" if mv_mine <= 0 else ("high" if gap > 1e-8 and mv_mine >= gap else "-")
        diffs.append(g - gcc[r]); pdiffs.append(pd)
        print(f"  {r:5d} {pd:14.2e} {gap:9.4f} {mv_cache:12.4f} {mv_mine:11.4f} "
              f"{gcc[r]:9.4f} {g:9.4f} {clamp:>7s}")
    d, q = np.array(diffs), np.array(pdiffs)
    print(f"\n  prediction drift |mine - cached preds_intervened|: median {np.median(q):.2e}  max {q.max():.2e}")
    print(f"  gc difference:                                     median {np.median(np.abs(d)):.4f}  max {np.abs(d).max():.4f}")
    print("\n  Read together: small prediction drift with large gc differences means gc is")
    print("  unstable (small gap or a clamp), not that the injection path is wrong.")


if __name__ == "__main__":
    main()
