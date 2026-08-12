#!/usr/bin/env python3
"""Do the per-concept ablations SUM to the transfer's effect, or not?

The patch's target is what ablating one concept achieves. That target only means "this
concept's share" if the concepts are additive at that row. Otherwise there is no share to
reproduce, and asking the recipient to move by c's amount is asking for something that
does not exist.

    sum(LOO) ~ gc_deployed   additive -- each ablation is genuinely that concept's share,
                             and a patch reaching it has reproduced that contribution
    sum(LOO) << gc_deployed  redundant -- the remaining concepts cover for whichever is
                             removed, no concept is individually necessary, and
                             single-concept attribution is not meaningful at that row
    sum(LOO) >> gc_deployed  interference -- the concepts partly cancel, so removing one
                             releases more than it contributed

Cheap at scale for the same reason loo_spectrum is: delta_r is linear in the activations,
so every leave-one-out delta is arithmetic and only the RECIPIENT tail runs. One batched
recipient call per row covers the deployed delta plus every LOO variant.

Usage:
    python -m scripts.rebuttal.loo_additivity_sweep --n-tails 12 --n-rows 8
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

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
BURNDOWN = PROJECT_ROOT / "output" / "rebuttal" / "patching_burndown.csv"


def universe(n_tails, n_rows):
    """(donor, recipient, dataset, path) -> rows, restricted to the patched concepts."""
    want = {(r["donor"], int(r["feat_id"])) for r in csv.DictReader(open(BURNDOWN))}
    cells = defaultdict(set)
    for f in sorted(glob.glob(str(FWD / "*" / "*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "selected_features" not in z.files or z["selected_features"].size == 0:
            continue
        donor, recipient = str(z["strong_model"]), str(z["weak_model"])
        dataset = os.path.basename(f)[:-4]
        if recipient == "carte":
            continue
        sel = z["selected_features"]
        for r in range(sel.shape[0]):
            acc = {int(x) for x in sel[r] if x >= 0}
            if any((donor, fid) in want for fid in acc):
                cells[(donor, recipient, dataset, f)].add(int(r))
    out = sorted(cells.items(), key=lambda kv: -len(kv[1]))[:n_tails]
    return [(k, sorted(v)[:n_rows]) for k, v in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tails", type=int, default=12)
    ap.add_argument("--n-rows", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output" / "rebuttal" /
                                         "loo_additivity.json"))
    args = ap.parse_args()
    torch.use_deterministic_algorithms(True)

    splits = json.loads(SPLITS_PATH.read_text())
    from scripts.rebuttal.functional_decomposition import _gc
    from scripts.intervention.intervene_lib import (
        SEQUENTIAL_MODELS, batched_ablation_sequential)

    records = []
    for (donor, recipient, dataset, path), rows in universe(args.n_tails, args.n_rows):
        try:
            z = np.load(path, allow_pickle=True)
            dd = np.asarray(z["deployed_delta"], dtype=np.float64)
            sel = np.asarray(z["selected_features"])
            zc = np.load(ATOMS / f"{donor}_to_{recipient}.npz", allow_pickle=True)
            V = np.asarray(zc["virtual_atoms"], dtype=np.float64)
            fmap = {int(f): i for i, f in enumerate(np.asarray(zc["feature_ids"]))}
            _, std_w = load_norm_stats(recipient, dataset, device=args.device)
            std_w = np.asarray(std_w.cpu(), dtype=np.float64)
            sae, _ = load_sae(donor, device=args.device)
            with torch.no_grad():
                A = sae.encode(torch.tensor(
                    np.asarray(load_test_embeddings(donor)[dataset], dtype=np.float32),
                    device=args.device)).cpu().numpy().astype(np.float64)
            Xtr, ytr, Xq, _, _, task = load_dataset_context(recipient, dataset, splits)
            if ytr.dtype == np.int32:
                ytr = ytr.astype(np.int64)
            layer = get_extraction_layer_taskaware(recipient, dataset=dataset)
            torch.manual_seed(13); np.random.seed(13)
            tail = build_tail(recipient, Xtr, ytr, Xq, layer, task, args.device,
                              target_name=splits.get(dataset, {}).get("target", "target"))
        except Exception as exc:
            print(f"  {donor}->{recipient}/{dataset}: SKIP {type(exc).__name__}: {exc}",
                  flush=True)
            continue

        for row in rows:
            fids = [int(f) for f in np.unique(sel[row][sel[row] >= 0]) if int(f) in fmap]
            if len(fids) < 2:
                continue
            B = np.stack([V[fmap[f]] * std_w for f in fids])
            c, *_ = np.linalg.lstsq(B.T, dd[row], rcond=None)
            signs, a = np.sign(c), np.array([A[row, f] for f in fids])
            variants = [dd[row]]
            for i in range(len(fids)):
                keep = np.ones(len(fids)); keep[i] = 0.0
                variants.append((signs * a * keep) @ B)
            d = torch.tensor(np.vstack(variants), dtype=torch.float32, device=args.device)
            if isinstance(tail, SEQUENTIAL_MODELS):
                preds = np.asarray(batched_ablation_sequential(tail, Xq[row:row+1], d,
                                                               query_idx=row), dtype=np.float64)
            else:
                preds = np.asarray(batched_intervention(tail, Xq[row:row+1], d,
                                                        inject_context=False), dtype=np.float64)
            y = int(np.asarray(z["y_query"])[row])
            b, t = np.asarray(z["preds_weak"])[row], np.asarray(z["preds_strong"])[row]
            gc_dep = float(_gc(b, preds[0], t, y))
            loo = np.array([gc_dep - float(_gc(b, preds[i + 1], t, y))
                            for i in range(len(fids))])
            records.append({"donor": donor, "recipient": recipient, "dataset": dataset,
                            "row": int(row), "n_concepts": len(fids),
                            "gc_deployed": gc_dep, "loo_sum": float(loo.sum()),
                            "loo_max": float(loo.max()), "loo_median": float(np.median(loo)),
                            "ratio": float(loo.sum() / gc_dep) if abs(gc_dep) > 1e-9 else None})
        print(f"  {donor}->{recipient}/{dataset}: {len(rows)} rows", flush=True)
        json.dump(records, open(args.out, "w"), indent=2)

    r = np.array([x["ratio"] for x in records
                  if x["ratio"] is not None and np.isfinite(x["ratio"])])
    k = np.array([x["n_concepts"] for x in records])
    print(f"\n{len(records)} rows, {len(r)} with a usable ratio")
    if r.size:
        print(f"  sum(LOO)/gc_deployed: " +
              "  ".join(f"p{q}={np.percentile(r, q):.3f}" for q in (10, 25, 50, 75, 90)))
        for lo, hi, lbl in [(-np.inf, 0.5, "REDUNDANT  (<0.5)"),
                            (0.5, 1.5, "additive (0.5-1.5)"),
                            (1.5, np.inf, "interfering (>1.5)")]:
            m = (r >= lo) & (r < hi)
            print(f"    {lbl:<20s} {int(m.sum()):5d} ({m.mean():6.1%})")
        print(f"  concepts per row: p50={np.percentile(k,50):.0f} p90={np.percentile(k,90):.0f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
