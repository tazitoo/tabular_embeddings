#!/usr/bin/env python3
"""Does the delta injection land on TabDPT's query row, or spread over its context?

_inject_query_deltas indexes TabDPT's cached state as state[n_ctx + k], and recapture()
derives n_ctx as hidden_state.shape[0] - n_query -- both assume dimension 0 is the
SEQUENCE (context rows, then query rows), the layout TabPFN and TabICL use.

TabDPT retrieves a per-query context (up to context_size neighbours) and forwards
(batch, retrieved_context + 1, H). If that is the real layout, dim 0 is the batch, the
derived n_ctx collapses, and the delta broadcasts across every retrieved neighbour
instead of hitting the query row. A uniform shift applied to the whole context largely
cancels in attention, which would explain why ABLATING a concept moves tabdpt only
0.0015 against 0.010-0.017 for the other recipients -- an attenuation upstream of any
search, that silently disqualifies 89% of tabdpt-recipient rows as sub-floor.

Prints the shapes and checks the injection actually changes the layer output where
intended. Cheap: one tail build.

Usage:
    python -m scripts.rebuttal.probe_tabdpt_injection --dataset anneal
"""
import argparse
import json

import numpy as np
import torch

from scripts.rebuttal.patch_search import SPLITS_PATH, load_dataset_context
from scripts.intervention.intervene_sae import build_tail
from scripts.intervention.intervene_lib import batched_intervention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="anneal")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    splits = json.loads(SPLITS_PATH.read_text())
    Xtr, ytr, Xq, _, _, task = load_dataset_context("tabdpt", args.dataset, splits)
    if ytr.dtype == np.int32:
        ytr = ytr.astype(np.int64)
    from scripts.rebuttal.patch_search import get_extraction_layer_taskaware
    layer = args.layer if args.layer is not None else \
        get_extraction_layer_taskaware("tabdpt", dataset=args.dataset)
    print(f"dataset={args.dataset} task={task} layer={layer} "
          f"n_context={len(Xtr)} n_query={len(Xq)}")

    tail = build_tail("tabdpt", Xtr, ytr, Xq, layer, task, args.device)
    hs = tail.hidden_state
    print(f"\nAFTER BUILD: hidden_state.shape={tuple(hs.shape)}  "
          f"n_ctx={tail.n_ctx}  n_query={tail.n_query}")
    print(f"  dim0 == n_ctx + n_query ? {hs.shape[0] == tail.n_ctx + tail.n_query}"
          f"   (the layout the injection code assumes)")

    # recapture with a single row, exactly as the patch search does
    row = 0
    tail.recapture(np.tile(Xq[row:row + 1], (4, 1)))
    hs2 = tail.hidden_state
    print(f"\nAFTER RECAPTURE with K=4 copies of one row: shape={tuple(hs2.shape)}  "
          f"n_ctx={tail.n_ctx}  n_query={tail.n_query}")
    if tail.n_ctx <= 0:
        print("  *** n_ctx <= 0: dim 0 is NOT the sequence, so state[n_ctx + k] "
              "indexes the batch and the delta hits every position ***")

    # does an injected delta actually move the prediction, and by how much?
    H = hs2.shape[-1]
    base = tail._predict_with_modified_state(hs2.clone())
    for scale in (1.0, 10.0, 100.0):
        d = torch.zeros((4, H), dtype=hs2.dtype, device=hs2.device)
        d[0] = scale * torch.randn(H, generator=torch.Generator(device=hs2.device)
                                   .manual_seed(0), device=hs2.device, dtype=hs2.dtype)
        preds = batched_intervention(tail, Xq[row:row + 1], d, inject_context=False)
        moved = float(np.abs(np.asarray(preds)[0] - np.asarray(base)[0]).max())
        print(f"  |delta|={scale * float(torch.norm(torch.randn(H, generator=torch.Generator(device='cpu').manual_seed(0)))):9.2f}"
              f"   max prediction change = {moved:.6f}")


if __name__ == "__main__":
    main()
