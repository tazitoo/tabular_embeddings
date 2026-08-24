#!/usr/bin/env python3
"""Re-run the readout on finished sweeps to add MEASURED attribution.

The search subtracts a linear estimate of the bystander contribution (leave-one-out
effects scaled by ratio change) because it cannot afford a counterfactual per
candidate. That estimate assumes per-concept effects add, which row_additivity says
they mostly do not, so the reported movement is biased. The readout can afford the
counterfactual: restore c to its corpus value with every bystander held at its
patched ratio, predict once, and c's contribution is measured.

No search re-runs -- only forward passes -- so a finished round can be rescored.
Source files are never modified; augmented copies go to --out-dir.

Usage:
    python -m scripts.rebuttal.backfill_exact_attribution \
        --inputs "output/rebuttal/v30q/*.json" --out-dir output/rebuttal/v30q_exact \
        --shard 0/4 --device cuda
"""
import argparse
import glob
import json
import os
import time
import traceback

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.rebuttal.patch_search import FWD, readout

NEW_FIELDS = ("toward_exact", "toward_unattributed", "capture_exact", "gc_restored_c",
              "movement_from_c", "movement_total_measured",
              "movement_bystanders_measured", "interval_readout")


def npz_index():
    """(strong, weak, dataset) -> path. Pair directories are alphabetically named, so
    the direction lives in the file, not the folder."""
    idx = {}
    for f in glob.glob(str(FWD / "*" / "*.npz")):
        z = np.load(f, allow_pickle=True)
        idx[(str(z["strong_model"]), str(z["weak_model"]),
             os.path.basename(f)[:-4])] = f
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default=None, help="i/n, 0-based")
    ap.add_argument("--limit-rows", type=int, default=None,
                    help="stop after this many rows (smoke)")
    args = ap.parse_args()

    outdir = PROJECT_ROOT / args.out_dir
    outdir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for pat in args.inputs for p in glob.glob(pat))
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        paths = paths[i::n]
    print(f"{len(paths)} files this shard -> {outdir}", flush=True)

    idx = npz_index()
    print(f"forward_deltas index: {len(idx)} cells", flush=True)

    n_rows = n_done = n_err = n_skip = 0
    t0 = time.time()
    for p in paths:
        dest = outdir / os.path.basename(p)
        if dest.exists():
            continue                       # resumable: finished files are immutable
        data = json.load(open(p))
        for c in data:
            for ds in c.get("datasets") or []:
                for r in ds.get("rows") or []:
                    if not (r.get("best") and r.get("accepted_ratios")
                            and r.get("readout") and not r["readout"].get("error")):
                        continue
                    n_rows += 1
                    if args.limit_rows and n_rows > args.limit_rows:
                        break
                    key = (c["donor"], ds.get("recipient"), ds.get("dataset"))
                    npz = idx.get(key)
                    if npz is None:
                        r["readout"]["exact_error"] = f"no forward_deltas for {key}"
                        n_skip += 1
                        continue
                    try:
                        ratios = {int(k): v for k, v in r["accepted_ratios"].items()}
                        new = readout(npz, c["feat"], r["row"], ratios, args.device)
                        if new is None:
                            r["readout"]["exact_error"] = "readout returned None"
                            n_skip += 1
                            continue
                        for k in NEW_FIELDS:
                            r["readout"][k] = new[k]
                        n_done += 1
                    except Exception as exc:
                        # loud and per row: a silent hole here would look like a
                        # rescored round that simply had fewer rows
                        r["readout"]["exact_error"] = f"{type(exc).__name__}: {exc}"
                        print(f"  ROW FAILED {key} row {r['row']}: {exc}\n"
                              f"{traceback.format_exc()}", flush=True)
                        n_err += 1
        dest.write_text(json.dumps(data))
        rate = n_done / max(time.time() - t0, 1e-9)
        print(f"  wrote {dest.name}  rows={n_rows} ok={n_done} err={n_err} "
              f"skip={n_skip}  ({rate * 3600:.0f} rows/h)", flush=True)

    print(f"DONE {outdir} rows={n_rows} rescored={n_done} errors={n_err} "
          f"skipped={n_skip} elapsed={(time.time() - t0) / 60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
