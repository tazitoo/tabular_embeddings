#!/usr/bin/env python3
"""What is preds_weak in the forward-delta files: a model prediction, or an injection?

Gap closure is computed as

    gc = (loss(preds_weak) - loss(preds_intervened)) / (loss(preds_weak) - loss(preds_strong))

with preds_weak read from the npz and preds_intervened produced by the injection path.
Injecting a ZERO delta does not reproduce the tail's own baseline (measured: tabpfn
1.26e-02, tabdpt 3.0e-03 at 100k train rows, deterministic in every case). So the two
sides of that subtraction may not sit on the same footing.

Which it is decides whether anything is wrong:

  preds_weak == zero-delta injection   -> both sides share the mechanism, the offset
                                          cancels, gc is sound as published.
  preds_weak == tail baseline_preds    -> gc subtracts a model prediction from an
                                          injection-path prediction and carries a
                                          systematic offset.

This compares preds_weak against BOTH candidates on the same rows, so the answer is
read off directly rather than inferred.

Usage:
    python -m scripts.rebuttal.check_preds_weak_baseline --recipient tabpfn --device cuda
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
    ap.add_argument("--recipient", default="tabpfn")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--n-rows", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.use_deterministic_algorithms(True)
    pick = None
    for f in sorted(glob.glob(str(FWD / "*" / "*.npz"))):
        if args.dataset and os.path.basename(f)[:-4] != args.dataset:
            continue
        z = np.load(f, allow_pickle=True)
        if str(z["weak_model"]) == args.recipient and z["selected_features"].size:
            pick = (f, os.path.basename(f)[:-4], str(z["strong_model"]), z)
            break
    if pick is None:
        print("no matching cell"); return
    path, dataset, donor, z = pick
    pw = np.asarray(z["preds_weak"], dtype=np.float64)
    dd = np.asarray(z["deployed_delta"], dtype=np.float64)
    print(f"{donor} -> {args.recipient} / {dataset}   preds_weak {pw.shape}\n")

    splits = json.loads(SPLITS_PATH.read_text())
    Xtr, ytr, Xq, _, _, task = load_dataset_context(args.recipient, dataset, splits)
    if ytr.dtype == np.int32:
        ytr = ytr.astype(np.int64)
    layer = get_extraction_layer_taskaware(args.recipient, dataset=dataset)
    cat_idx = None
    if args.recipient in ("hyperfast", "tabpfn"):
        from data.preprocessing import load_preprocessed, CACHE_DIR
        try:
            cat_idx = load_preprocessed(args.recipient, dataset, CACHE_DIR).cat_indices or None
        except Exception:
            pass
    torch.manual_seed(13); np.random.seed(13)
    tail = build_tail(args.recipient, Xtr, ytr, Xq, layer, task, args.device,
                      cat_indices=cat_idx,
                      target_name=splits.get(dataset, {}).get("target", "target"))
    base = np.asarray(getattr(tail, "baseline_preds", None), dtype=np.float64)

    rows = np.linspace(0, min(len(pw), len(Xq)) - 1, args.n_rows).astype(int)
    d_base, d_zero = [], []
    for r in rows:
        zed = torch.zeros((1, dd.shape[1]), dtype=torch.float32, device=args.device)
        pz = np.asarray(batched_intervention(tail, Xq[r:r + 1], zed, inject_context=False),
                        dtype=np.float64)[0]
        d_zero.append(float(np.abs(pz - pw[r]).max()))
        if base.ndim:
            d_base.append(float(np.abs(base[r] - pw[r]).max()))

    print(f"  {'row':>5s} {'|preds_weak - baseline|':>24s} {'|preds_weak - zeroInject|':>26s}")
    for i, r in enumerate(rows):
        b = f"{d_base[i]:.3e}" if d_base else "n/a"
        print(f"  {r:5d} {b:>24s} {d_zero[i]:26.3e}")
    print()
    if d_base:
        print(f"  median |preds_weak - baseline_preds| = {np.median(d_base):.3e}")
    print(f"  median |preds_weak - zero-injection|  = {np.median(d_zero):.3e}")
    verdict = ("preds_weak IS the injection path -> offset cancels in gc"
               if d_base and np.median(d_zero) < np.median(d_base)
               else "preds_weak is the MODEL prediction -> gc mixes two footings")
    print(f"\n  => {verdict}")

    out = args.out or str(PROJECT_ROOT / "output" / "rebuttal" /
                          f"preds_weak_check_{args.recipient}.json")
    json.dump({"recipient": args.recipient, "dataset": dataset, "donor": donor,
               "rows": rows.tolist(), "d_baseline": d_base, "d_zero_injection": d_zero,
               "verdict": verdict}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
