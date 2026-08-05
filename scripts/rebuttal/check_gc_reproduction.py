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

    print(f"  {'row':>5s} {'cached gap_closed':>18s} {'recomputed gc':>15s} {'diff':>10s} {'strong_win':>11s}")
    diffs = []
    for r in rows:
        d = torch.tensor(dd[r][None, :], dtype=torch.float32, device=args.device)
        if isinstance(tail, SEQUENTIAL_MODELS):
            p = np.asarray(batched_ablation_sequential(tail, Xq[r:r+1], d, query_idx=r),
                           dtype=np.float64)
        else:
            p = np.asarray(batched_intervention(tail, Xq[r:r+1], d, inject_context=False),
                           dtype=np.float64)
        g = float(_gc(np.asarray(pw)[r], p[0], np.asarray(ps)[r], int(np.asarray(yq)[r])))
        diffs.append(g - gcc[r])
        print(f"  {r:5d} {gcc[r]:18.4f} {g:15.4f} {g-gcc[r]:10.4f} {str(bool(sw[r])):>11s}")
    d = np.array(diffs)
    print(f"\n  median |diff| = {np.median(np.abs(d)):.4f}   max |diff| = {np.abs(d).max():.4f}")
    print("  Nonzero differences mean the readout does not reproduce the transfer, so")
    print("  ceiling and capture are measured against a baseline the run never had.")


if __name__ == "__main__":
    main()
