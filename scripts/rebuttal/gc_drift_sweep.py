#!/usr/bin/env python3
"""Does the patch readout reproduce the transfer, across the whole patching universe?

The readout computes gap closure by pushing a delta through a freshly built recipient
tail. The transfer recorded gap_closed and preds_intervened for the same rows on
2026-07-24/25 -- before the 2026-08-02/03 env lockdown, on hosts the npz does not
record. So a difference now is a joint function of the injection path, the env, and the
hardware, and none of the previously measured drift figures apply: +0.0114 (GPU) and
0.0000-0.0016 (env migration) were measured on POOLED rel_off from the functional
decomposition, not on per-row gc over these rows.

Scope is the patching universe rather than a spot check, because drift may vary by
recipient, dataset and row, and a probe on one cell says nothing about the rest. This is
cheap: gc reproduction is independent of the concept (every concept at a row shares the
same deployed_delta), so it dedupes across concepts, and it needs NO donor forward --
only a recipient tail per (recipient, dataset).

Per row it logs a ladder, so a difference can be attributed rather than just observed:

  tail      |my tail baseline - cached preds_weak|        tail reproduction
  inject    |zero-delta injection - my tail baseline|     injection identity
  full      |deployed-delta injection - preds_intervened| the path the readout uses
  plus gap, moved (cached and mine), both gc values, and which clamp fired.

Run one arm per host. Two same-architecture hosts give repeat-vs-repeat (the current
noise floor, predicted bit-identical by docs/reproducibility.md); a different
architecture prices the GPU term. What cannot be separated: the old env from the
unrecorded producing host, since neither is recoverable.

Usage (one arm per host, all three in the locked env):
    python -m scripts.rebuttal.gc_drift_sweep --tag surfer
"""
import argparse
import glob
import json
import os
import socket
import subprocess
import time
from collections import defaultdict

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import (
    SPLITS_PATH, batched_intervention, build_tail, get_extraction_layer_taskaware,
    load_dataset_context,
)

FWD = PROJECT_ROOT / "output" / "rebuttal" / "forward_deltas"
BURNDOWN = PROJECT_ROOT / "output" / "rebuttal" / "patching_burndown.csv"
EPS = 1e-7


def provenance():
    """Recorded per run: a drift number is meaningless without what produced it."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        commit = "?"
    gpu = "?"
    try:
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {"host": socket.gethostname(), "gpu": gpu, "torch": torch.__version__,
            "numpy": np.__version__, "commit": commit,
            "python": os.environ.get("CONDA_PREFIX", ""), "time": time.strftime("%F %T")}


def universe(n_datasets, n_rows):
    """The (donor, recipient, dataset, row) tuples the patch selection would touch.

    Mirrors patch_search: per concept, the datasets where it is accepted earliest, then
    rows by acceptance position. Deduped across concepts, since gc reproduction does not
    depend on which concept we are studying.
    """
    import csv
    want = {(r["donor"], int(r["feat_id"]))
            for r in csv.DictReader(open(BURNDOWN))}
    cells = defaultdict(list)
    for f in sorted(glob.glob(str(FWD / "*" / "*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "selected_features" not in z.files or z["selected_features"].size == 0:
            continue
        donor, recipient = str(z["strong_model"]), str(z["weak_model"])
        dataset = os.path.basename(f)[:-4]
        sel = z["selected_features"]
        for r in range(sel.shape[0]):
            acc = [int(x) for x in sel[r] if x >= 0]
            for i, fid in enumerate(acc):
                if (donor, fid) in want:
                    cells[((donor, fid), recipient, dataset, f)].append((r, i))
    per_concept = defaultdict(list)
    for (c, rec, ds, f), rows in cells.items():
        per_concept[c].append((min(i for _, i in rows), rec, ds, f, rows))
    rows_needed = defaultdict(set)          # (donor, recipient, dataset, path) -> rows
    for c, lst in per_concept.items():
        for _, rec, ds, f, rows in sorted(lst, key=lambda t: t[0])[:n_datasets]:
            for r, _ in sorted(rows, key=lambda t: t[1])[:n_rows]:
                rows_needed[(c[0], rec, ds, f)].add(int(r))
    return rows_needed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="arm name, e.g. the host")
    ap.add_argument("--n-datasets", type=int, default=3)
    ap.add_argument("--n-rows", type=int, default=10)
    ap.add_argument("--shard", default=None, help="i/n over (recipient, dataset) tails")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.use_deterministic_algorithms(True)
    prov = provenance()
    out_path = args.out or str(PROJECT_ROOT / "output" / "rebuttal" /
                               f"gc_drift_{args.tag}.json")
    print(json.dumps(prov, indent=2))

    need = universe(args.n_datasets, args.n_rows)
    by_tail = defaultdict(list)
    for (donor, rec, ds, f), rows in need.items():
        by_tail[(rec, ds)].append((donor, f, sorted(rows)))
    tails = sorted(by_tail)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        tails = tails[i::n]
    print(f"\n{sum(len(v) for k, v in by_tail.items() if k in set(tails))} cells over "
          f"{len(tails)} (recipient, dataset) tails\n", flush=True)

    splits = json.loads(SPLITS_PATH.read_text())
    from scripts.rebuttal.functional_decomposition import _gc
    from scripts.intervention.intervene_lib import (
        SEQUENTIAL_MODELS, batched_ablation_sequential)

    records, done = [], 0
    for (recipient, dataset) in tails:
        try:
            Xtr, ytr, Xq, _, _, task = load_dataset_context(recipient, dataset, splits)
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
            tail = build_tail(recipient, Xtr, ytr, Xq, layer, task, args.device,
                              cat_indices=cat_idx,
                              target_name=splits.get(dataset, {}).get("target", "target"))
            base = np.asarray(getattr(tail, "baseline_preds", None), dtype=np.float64)
        except Exception as exc:
            print(f"  {recipient}/{dataset}: TAIL FAILED {type(exc).__name__}: {exc}", flush=True)
            records.append({"recipient": recipient, "dataset": dataset,
                            "error": f"tail: {type(exc).__name__}: {exc}"})
            continue

        for donor, f, rows in by_tail[(recipient, dataset)]:
            z = np.load(f, allow_pickle=True)
            dd = np.asarray(z["deployed_delta"], dtype=np.float64)
            gcc = np.asarray(z["gap_closed"], dtype=np.float64)
            pi = np.asarray(z["preds_intervened"], dtype=np.float64)
            pw = np.asarray(z["preds_weak"], dtype=np.float64)
            ps = np.asarray(z["preds_strong"], dtype=np.float64)
            yq = np.asarray(z["y_query"])
            for r in rows:
                try:
                    two = torch.tensor(np.vstack([dd[r], np.zeros_like(dd[r])]),
                                       dtype=torch.float32, device=args.device)
                    if isinstance(tail, SEQUENTIAL_MODELS):
                        p = np.asarray(batched_ablation_sequential(tail, Xq[r:r+1], two,
                                                                   query_idx=r), dtype=np.float64)
                    else:
                        p = np.asarray(batched_intervention(tail, Xq[r:r+1], two,
                                                            inject_context=False), dtype=np.float64)
                    y = int(yq[r])
                    g = float(_gc(pw[r], p[0], ps[r], y))
                    bw, bs = pw[r], ps[r]
                    if bw.ndim >= 1 and bw.size > 1:
                        ol = -np.log(np.clip(bw[y], EPS, 1 - EPS))
                        tl = -np.log(np.clip(bs[y], EPS, 1 - EPS))
                        gap = float(ol - tl)
                        mv_mine = float(ol + np.log(np.clip(p[0][y], EPS, 1 - EPS)))
                        mv_cache = float(ol + np.log(np.clip(pi[r][y], EPS, 1 - EPS)))
                    else:
                        gap = float((float(bw) - float(bs)) ** 2)
                        mv_mine = gap - float((float(p[0]) - float(bs)) ** 2)
                        mv_cache = gap - float((float(pi[r]) - float(bs)) ** 2)
                    records.append({
                        "donor": donor, "recipient": recipient, "dataset": dataset, "row": int(r),
                        # the ladder
                        "d_tail_baseline": (float(np.abs(base[r] - pw[r]).max())
                                            if base.ndim else None),
                        "d_zero_vs_baseline": (float(np.abs(p[1] - base[r]).max())
                                               if base.ndim else None),
                        "d_full_vs_cached": float(np.abs(p[0] - pi[r]).max()),
                        # gc and its components
                        "gap": gap, "moved_cache": mv_cache, "moved_mine": mv_mine,
                        "gc_cache": float(gcc[r]), "gc_mine": g,
                        "gc_diff": float(g - gcc[r]),
                        "clamp": ("low" if mv_mine <= 0 else
                                  "high" if gap > 1e-8 and mv_mine >= gap else "-"),
                    })
                except Exception as exc:
                    records.append({"donor": donor, "recipient": recipient,
                                    "dataset": dataset, "row": int(r),
                                    "error": f"{type(exc).__name__}: {exc}"})
        done += 1
        ok = [r for r in records if "gc_diff" in r]
        if ok:
            print(f"  [{done}/{len(tails)}] {recipient}/{dataset}: {len(ok)} rows so far, "
                  f"median |gc diff| {np.median([abs(r['gc_diff']) for r in ok]):.4f}, "
                  f"median pred drift {np.median([r['d_full_vs_cached'] for r in ok]):.2e}",
                  flush=True)
        json.dump({"provenance": prov, "records": records}, open(out_path, "w"), indent=2)

    ok = [r for r in records if "gc_diff" in r]
    if ok:
        gd = np.abs([r["gc_diff"] for r in ok])
        pdz = np.array([r["d_full_vs_cached"] for r in ok])
        tb = np.array([r["d_tail_baseline"] for r in ok if r["d_tail_baseline"] is not None])
        zb = np.array([r["d_zero_vs_baseline"] for r in ok if r["d_zero_vs_baseline"] is not None])
        print(f"\n{len(ok)} rows")
        if len(tb):
            print(f"  tail   |baseline - preds_weak|      median {np.median(tb):.2e}  p95 {np.percentile(tb,95):.2e}")
        if len(zb):
            print(f"  inject |zero - baseline|            median {np.median(zb):.2e}  p95 {np.percentile(zb,95):.2e}")
        print(f"  full   |deployed - preds_intervened|  median {np.median(pdz):.2e}  p95 {np.percentile(pdz,95):.2e}")
        print(f"  gc     |mine - cached|                median {np.median(gd):.4f}  p95 {np.percentile(gd,95):.4f}")
        from collections import Counter
        print(f"  clamps: {dict(Counter(r['clamp'] for r in ok))}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
