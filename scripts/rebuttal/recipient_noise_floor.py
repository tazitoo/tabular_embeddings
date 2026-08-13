#!/usr/bin/env python3
"""What is the smallest recipient prediction difference the patch readout can resolve?

`reversal` divides by the interval L_ablated - L_transfer. Now that both endpoints are
predictions (patch_search.build_recip.loss), that interval is a difference between two
numbers produced by the SAME recipient forward path -- so the interval is only meaningful
above that path's own reproduction error. Below it, numerator and denominator are both
noise and the ratio is arbitrary. That error is the floor `min_interval` should be set
from, and it was already measured: gc_drift_sweep.py logged, per row, a ladder in
prediction units.

  d_zero_vs_baseline  |zero-delta injection - tail baseline|   the injection path alone,
                      same process, same run -- pure path noise
  d_full_vs_cached    |deployed-delta injection - cached preds_intervened|  the same path
                      against the transfer's own stored predictions, so it also carries
                      the env migration and the unrecorded producing host

Both are max-abs over classes, which is the conservative reading for an interval taken on
one class. Three hosts ran it (firelord 4090, surfer/terrax 3090), so the spread of the
same row across hosts prices the hardware term: |d_A - d_B| is a LOWER bound on
|pred_A - pred_B| (both are distances to a common reference), reported as such.

The floor is per (recipient, dataset): it is a property of a tail, and pooling across
recipients would set one model's threshold from another's arithmetic.

Classification and regression are reported SEPARATELY and never pooled. A classification
prediction difference is bounded by 1; a regression one carries the target's own units, so
a pooled quantile is an average over incommensurable scales -- pooling put the max at
2.6e+04, which no probability difference can reach.

carte is reported but flagged -- patch_search puts it in READOUT_EXCLUDED, so its rows
never reach `reversal` and its floor is not a constraint on anything.

Usage:
    python -m scripts.rebuttal.recipient_noise_floor
"""
import json
from collections import defaultdict

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.rebuttal.patch_search import _task_of

DRIFT = PROJECT_ROOT / "output" / "rebuttal"
HOSTS = ("firelord", "surfer", "terrax")
OUT = DRIFT / "recipient_noise_floor.json"
EXCLUDED = ("carte",)


def load():
    """Records per host, keyed so the same row can be compared across hosts."""
    per_host = {}
    for h in HOSTS:
        p = DRIFT / f"gc_drift_{h}.json"
        if not p.exists():
            continue
        recs = json.loads(p.read_text())["records"]
        per_host[h] = {(r["donor"], r["recipient"], r["dataset"], r["row"]): r
                       for r in recs if "gc_diff" in r and r.get("d_full_vs_cached") is not None}
    return per_host


def summarize(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return None
    return {"n": int(len(v)), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def main():
    per_host = load()
    if not per_host:
        raise SystemExit(f"no gc_drift_*.json in {DRIFT}")

    # Per (recipient, dataset), pooled over hosts and donors: the path's own error.
    by_cell = defaultdict(lambda: {"zero": [], "full": []})
    task = {}
    for recs in per_host.values():
        for (_, recipient, dataset, _), r in recs.items():
            c = by_cell[(recipient, dataset)]
            c["zero"].append(r.get("d_zero_vs_baseline"))
            c["full"].append(r.get("d_full_vs_cached"))
            task[dataset] = _task_of(dataset)

    # Cross-host: same row, two hosts. Lower bound on the hardware/env term.
    cross, cross_all = defaultdict(list), defaultdict(list)
    hs = sorted(per_host)
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            a, b = per_host[hs[i]], per_host[hs[j]]
            for k in set(a) & set(b):
                d = abs(a[k]["d_full_vs_cached"] - b[k]["d_full_vs_cached"])
                cross[(k[1], k[2])].append(d)
                cross_all[(task[k[2]], k[1], f"{hs[i]}~{hs[j]}")].append(d)

    cells = {}
    for (recipient, dataset), c in sorted(by_cell.items()):
        cells[f"{recipient}/{dataset}"] = {
            "recipient": recipient, "dataset": dataset, "task": task[dataset],
            "readout_excluded": recipient in EXCLUDED,
            "injection_identity": summarize(c["zero"]),
            "path_vs_cached": summarize(c["full"]),
            "cross_host_lower_bound": summarize(cross.get((recipient, dataset), [])),
        }

    by_recipient = defaultdict(lambda: {"zero": [], "full": [], "cells": 0})
    for (recipient, dataset), c in by_cell.items():
        k = (task[dataset], recipient)
        by_recipient[k]["zero"] += c["zero"]
        by_recipient[k]["full"] += c["full"]
        by_recipient[k]["cells"] += 1

    rec_out = {}
    for t in ("classification", "regression"):
        print(f"\n=== {t} " + "=" * 62)
        print(f"{'recipient':<12} {'cells':>5} {'rows':>6}  "
              f"{'inject-identity (med/p95/max)':<34}  {'vs-cached (med/p95/max)':<34}")
        for (tt, recipient), c in sorted(by_recipient.items()):
            if tt != t:
                continue
            z, f = summarize(c["zero"]), summarize(c["full"])
            flag = "  [readout-excluded]" if recipient in EXCLUDED else ""
            zs = (f"{z['median']:.2e} {z['p95']:.2e} {z['max']:.2e}" if z else "-")
            fs = (f"{f['median']:.2e} {f['p95']:.2e} {f['max']:.2e}" if f else "-")
            print(f"{recipient:<12} {c['cells']:>5} {f['n'] if f else 0:>6}  "
                  f"{zs:<34}  {fs:<34}{flag}")
            rec_out[f"{t}/{recipient}"] = {
                "task": t, "recipient": recipient, "n_cells": c["cells"],
                "injection_identity": z, "path_vs_cached": f,
                "readout_excluded": recipient in EXCLUDED}

    # Per recipient, never pooled: carte is 36 of the 90 classification cells and 5730 of
    # the 7680 rows, so a pooled cross-host figure is mostly a model whose rows never
    # reach `reversal`.
    print("\ncross-host |d_A - d_B| on the same row, LOWER bound on |pred_A - pred_B|")
    for t in ("classification", "regression"):
        for (tt, recipient, pair), vals in sorted(cross_all.items()):
            if tt != t:
                continue
            s = summarize(vals)
            flag = "  [readout-excluded]" if recipient in EXCLUDED else ""
            print(f"  {t:<15} {recipient:<9} {pair:<18} n={s['n']:<6} "
                  f"median {s['median']:.2e}  p95 {s['p95']:.2e}  max {s['max']:.2e}{flag}")
            rec_out.setdefault(f"{t}/{recipient}", {}).setdefault("cross_host", {})[pair] = s

    OUT.write_text(json.dumps({"per_cell": cells, "per_recipient": rec_out}, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
