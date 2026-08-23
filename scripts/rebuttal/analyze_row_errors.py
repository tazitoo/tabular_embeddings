#!/usr/bin/env python3
"""Which rows failed, why, and whether the failures correlate with the host.

The determinism canary in evaluate() rejects a window whose unmodified control
row moved more than tol, so canary trips read as a probe of numerical stability
-- a per-machine property (GPU model, clocks, kernel selection).

Counts are printed over the rows each host actually searched: a host running ten
times the rows shows ten times the trips at the same rate.

Usage:
    python -m scripts.rebuttal.analyze_row_errors \
        --inputs "output/rebuttal/v30q/*.json" "output/rebuttal/patchv30clf_*.json"
"""
import argparse
import glob
import json
import re
from collections import Counter, defaultdict

CANARY = re.compile(r"unmodified row moved ([0-9.e+-]+) \(tol ([0-9.e+-]+)\)")


def classify(err):
    if "unmodified row moved" in err:
        return "determinism_canary"
    if "no raw" in err or "Incomplete cache" in err:
        return "missing_cache"
    if "out of memory" in err.lower():
        return "oom"
    return err.split(":")[0] or "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    args = ap.parse_args()

    rows_by_host = Counter()
    errs = defaultdict(Counter)          # host -> kind -> n
    magnitudes = []                      # (host, moved, tol, donor, recipient, dataset)
    cells = Counter()                    # cell-level (dataset) failures by status
    for pat in args.inputs:
        for p in sorted(glob.glob(pat)):
            for c in json.load(open(p)):
                for ds in c.get("datasets") or []:
                    if ds.get("error") or ds.get("status"):
                        cells[classify(str(ds.get("error") or ds.get("status")))] += 1
                    for r in ds.get("rows") or []:
                        host = r.get("host") or "unknown"
                        rows_by_host[host] += 1
                        e = r.get("error")
                        if not e:
                            continue
                        kind = classify(str(e))
                        errs[host][kind] += 1
                        m = CANARY.search(str(e))
                        if m:
                            magnitudes.append((host, float(m.group(1)),
                                               float(m.group(2)), c["donor"],
                                               ds.get("recipient"), ds.get("dataset")))

    total_rows = sum(rows_by_host.values())
    total_err = sum(sum(v.values()) for v in errs.values())
    print(f"{total_rows} rows searched, {total_err} with an error "
          f"({total_err / max(total_rows, 1):.2%})")

    print(f"\n=== per host (rate over that host's own rows)")
    print(f"  {'host':10s} {'rows':>7s} {'errors':>7s} {'rate':>7s}   breakdown")
    for host in sorted(rows_by_host, key=lambda h: -rows_by_host[h]):
        n, e = rows_by_host[host], sum(errs[host].values())
        bd = ", ".join(f"{k}={v}" for k, v in errs[host].most_common())
        print(f"  {host:10s} {n:7d} {e:7d} {e / max(n, 1):6.2%}   {bd or '-'}")

    if magnitudes:
        print(f"\n=== determinism canary: {len(magnitudes)} trips")
        by_host = Counter(h for h, *_ in magnitudes)
        for h, n in by_host.most_common():
            mv = [m for hh, m, *_ in magnitudes if hh == h]
            print(f"  {h:10s} trips {n:4d} over {rows_by_host[h]:5d} rows "
                  f"= {n / max(rows_by_host[h], 1):.2%}   "
                  f"moved min {min(mv):.2e} max {max(mv):.2e}")
        print("  by recipient: "
              + ", ".join(f"{k}={v}" for k, v in
                          Counter(r for *_, r, _ in magnitudes).most_common()))
        print("  by dataset:   "
              + ", ".join(f"{k}={v}" for k, v in
                          Counter(d for *_, d in magnitudes).most_common(8)))

    if cells:
        print(f"\n=== cell-level failures: "
              + ", ".join(f"{k}={v}" for k, v in cells.most_common()))


if __name__ == "__main__":
    main()
