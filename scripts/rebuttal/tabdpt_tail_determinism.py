#!/usr/bin/env python3
"""Is the TabDPT intervention TAIL deterministic?

TabDPT's ensemble draw comes from np.random.SeedSequence(seed) at
tabdpt/estimator.py:229. With seed=None it pulls fresh OS entropy, ignoring the global
NumPy RNG and unaffected by torch.use_deterministic_algorithms. We already showed the
EXTRACTION path is hit by this. TabDPTTail.from_data calls clf.predict(X_query) with no
seed either, so the intervention path plausibly has the same problem.

That matters well beyond patching. Every recipient tail is built this way, so if the
tail is nondeterministic then any result that injects a delta into tabdpt-as-RECIPIENT
carries that noise -- including functional_decomposition, which produced the published
rel_on/rel_off numbers. docs/reproducibility.md reports repeated same-host runs as
bit-identical for all 6 models, so either that check did not exercise this path or
something else pins it. Worth settling by measurement rather than inference.

Three checks:
  build     two tails from identical inputs -> same baseline predictions?
  predict   one tail, same delta twice      -> same output?
  zero      injecting a zero delta          -> reproduces the baseline?

Usage:
    python -m scripts.rebuttal.tabdpt_tail_determinism --device cuda
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
    SPLITS_PATH, build_tail, get_extraction_layer_taskaware, load_dataset_context,
    batched_intervention,
)

FWD = PROJECT_ROOT / "output" / "rebuttal" / "forward_deltas"


def find_cell(recipient, dataset=None):
    for f in sorted(glob.glob(str(FWD / "*" / "*.npz"))):
        if dataset and os.path.basename(f)[:-4] != dataset:
            continue
        z = np.load(f, allow_pickle=True)
        if str(z["weak_model"]) == recipient and z["selected_features"].size:
            rows = [r for r in range(z["deployed_delta"].shape[0])
                    if np.abs(z["deployed_delta"][r]).any()]
            if rows:
                return f, os.path.basename(f)[:-4], str(z["strong_model"]), rows[0], z
    return None


def make_tail(recipient, dataset, device):
    splits = json.loads(SPLITS_PATH.read_text())
    Xtr, ytr, Xq, _, _, task = load_dataset_context(recipient, dataset, splits)
    if ytr.dtype == np.int32:
        ytr = ytr.astype(np.int64)
    layer = get_extraction_layer_taskaware(recipient, dataset=dataset)
    # functional_decomposition passes cat_indices for these models; omitting it builds a
    # DIFFERENT model and inflates any baseline-vs-injection discrepancy.
    cat_idx = None
    if recipient in ("hyperfast", "tabpfn"):
        from data.preprocessing import load_preprocessed, CACHE_DIR
        try:
            cat_idx = load_preprocessed(recipient, dataset, CACHE_DIR).cat_indices or None
        except Exception:
            pass
    torch.manual_seed(13); np.random.seed(13)
    tail = build_tail(recipient, Xtr, ytr, Xq, layer, task, device, cat_indices=cat_idx,
                      target_name=splits.get(dataset, {}).get("target", "target"))
    return tail, Xq, task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipient", default="tabdpt")
    ap.add_argument("--dataset", default=None,
                    help="pin the dataset; TabDPT only SAMPLES when train > context_size "
                         "(2048), so a small-train dataset can look deterministic by luck")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(
        PROJECT_ROOT / "output" / "rebuttal" / "tabdpt_tail_determinism.json"))
    args = ap.parse_args()

    torch.use_deterministic_algorithms(True)
    found = find_cell(args.recipient, args.dataset)
    if not found:
        print(f"no cell with recipient={args.recipient} dataset={args.dataset}"); return
    path, dataset, donor, row, z = found
    dd = np.asarray(z["deployed_delta"], dtype=np.float64)[row]
    print(f"{donor} -> {args.recipient} / {dataset}, row {row}\n")

    res = {"recipient": args.recipient, "dataset": dataset, "donor": donor, "row": row}

    # 1. build twice from identical inputs
    t1, Xq, task = make_tail(args.recipient, dataset, args.device)
    t2, _, _ = make_tail(args.recipient, dataset, args.device)
    b1 = np.asarray(getattr(t1, "baseline_preds", None), dtype=np.float64)
    b2 = np.asarray(getattr(t2, "baseline_preds", None), dtype=np.float64)
    if b1.ndim:
        d = float(np.abs(b1 - b2).max())
        res["build_max_abs_diff"] = d
        print(f"  build     two tails, baseline preds: max|d| = {d:.3e}  "
              f"-> {'DETERMINISTIC' if d < 1e-8 else 'NONDETERMINISTIC'}")

    # 2. same tail, same delta, twice
    deltas = torch.tensor(np.vstack([dd, dd]), dtype=torch.float32, device=args.device)
    p = np.asarray(batched_intervention(t1, Xq[row:row + 1], deltas, inject_context=False),
                   dtype=np.float64)
    d2 = float(np.abs(p[0] - p[1]).max())
    res["same_delta_twice_max_abs_diff"] = d2
    print(f"  predict   same delta twice:            max|d| = {d2:.3e}  "
          f"-> {'DETERMINISTIC' if d2 < 1e-8 else 'NONDETERMINISTIC'}")

    # 3. zero delta should reproduce the baseline
    zed = torch.zeros((1, dd.shape[0]), dtype=torch.float32, device=args.device)
    pz = np.asarray(batched_intervention(t1, Xq[row:row + 1], zed, inject_context=False),
                    dtype=np.float64)
    if b1.ndim:
        d3 = float(np.abs(pz[0] - b1[row]).max())
        res["zero_delta_vs_baseline_max_abs_diff"] = d3
        print(f"  zero      zero delta vs baseline:      max|d| = {d3:.3e}  "
              f"-> {'consistent' if d3 < 1e-8 else 'INCONSISTENT'}")

    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    print("\nIf any check is NONDETERMINISTIC, every result injecting a delta into this")
    print("recipient carries that noise -- including functional_decomposition's published")
    print("rel_on/rel_off, not just the patch readout.")


if __name__ == "__main__":
    main()
