#!/usr/bin/env python3
"""Is CARTETail reproducible run-to-run under identical seeds? (§E carte drift)

Two runs of `functional_decomposition.py` at the same threshold, same commit,
same config agree to ~0.001-0.007 mean |dgc| on every recipient EXCEPT carte,
which posts 0% exact matches and 0.105 mean / 0.68 max. Seeds are not missing:
functional_decomposition.py sets torch.manual_seed(13)/np.random.seed(13) before
build_tail, and CARTEClassifier defaults to random_state=0.

The candidate explanation is that carte is the only recipient whose tail is
TRAINED rather than loaded: CARTETail.from_data calls clf.fit(...) for up to
max_epoch epochs with an internal val split + early stopping. CUDA scatter ops
in the GNN use atomicAdd, whose float accumulation order is not reproducible, so
seeding fixes initialization but not the training trajectory.

This builds the SAME tail twice in one process and compares baseline predictions.
  identical      -> training is reproducible; the drift is elsewhere.
  differing      -> confirmed: seeds do not make CARTETail reproducible.
Also builds a non-training recipient (tabpfn) twice as a control.

Usage (on a worker):
    python -m scripts.rebuttal.carte_determinism_check --dataset <name>
"""
import argparse
import json

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import (
    SPLITS_PATH, get_extraction_layer_taskaware, build_tail, load_dataset_context,
)


def build_once(model, dataset, device, splits):
    """Build a tail under the exact seeding functional_decomposition.py uses."""
    Xtr, ytr, Xq, _, _, task = load_dataset_context(model, dataset, splits)
    if ytr.dtype == np.int32:
        ytr = ytr.astype(np.int64)
    layer = get_extraction_layer_taskaware(model, dataset=dataset)
    cat_indices = None
    if model in ("hyperfast", "tabpfn"):
        from data.preprocessing import load_preprocessed, CACHE_DIR
        cat_indices = load_preprocessed(model, dataset, CACHE_DIR).cat_indices or None
    target_name = splits.get(dataset, {}).get("target", "target")
    torch.manual_seed(13); np.random.seed(13)          # same as the decomposition
    tail = build_tail(model, Xtr, ytr, Xq, layer, task, device,
                      cat_indices=cat_indices, target_name=target_name)
    return np.asarray(tail.baseline_preds, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", nargs="+", default=["carte", "tabpfn"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    splits = json.loads(SPLITS_PATH.read_text())
    for model in args.models:
        a = build_once(model, args.dataset, args.device, splits)
        b = build_once(model, args.dataset, args.device, splits)
        d = np.abs(a - b)
        same = bool(np.array_equal(a, b))
        print(f"\n=== {model} / {args.dataset} ===")
        print(f"  shape={a.shape}  bit-identical={same}")
        print(f"  |d|: mean={d.mean():.3e}  max={d.max():.3e}  "
              f"rows differing={int((d.max(axis=1) > 0).sum()) if d.ndim > 1 else int((d>0).sum())}"
              f"/{len(d)}")
        if not same:
            print(f"  -> NOT reproducible under identical seeds")


if __name__ == "__main__":
    main()
