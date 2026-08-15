#!/usr/bin/env python3
"""Would LOO-weighted column EFFECTIVENESS feed the greedy different candidates?

Pass 1 currently ranks columns by how much each moves concept c's activation per unit
of log frequency -- blast-blind by design, because the old selectivity ranking
penalised columns for probe-sized collateral. The proposal under test (2026-08-15):
rank instead by an effectiveness built from the MAIN-EFFECT LOO values,

    gain(probe)  = |da_c| / a_c x |interval_c|             c's predicted prediction-effect
    spend(probe) = sum_{j != c, live} |da_j| / a_j x loo_j  bystanders' predicted spend
    net rate     = (gain - spend) / dL                      per unit log frequency

per-unit and LOO-weighted, so the two old failure modes are absent: probe size divides
out of the rate, and a column that disturbs concepts the prediction ignores is not
penalised. Magnitudes and interactions stay the greedy's job; this only reorders the
menu it sees.

For a handful of rows, three rankings are compared -- current (slope on c alone), net
effectiveness, and the gain/spend ratio -- reporting the top-K sets, their overlap,
and where the columns the LAST SWEEP'S SEARCH ACTUALLY CHOSE sit under each ranking.
The chosen columns had to survive pass 2's full objective, so a ranking that places
them higher is offering the greedy a better menu.

Rows are sampled from the sweep output: crowded rows (many co-accepted concepts, where
blast bites) with a recipient readout, spread over cells, restricted to cells runnable
in the current env.

Usage (one GPU, ~minutes per row):
    python -m scripts.rebuttal.column_effectiveness_probe --device cuda
"""
import argparse
import glob
import json
import os
from collections import defaultdict

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

from scripts._project_root import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "output" / "rebuttal"


def sample_rows(sweep_glob, n_cells, rows_per_cell, min_others):
    """Crowded, readout-bearing rows, spread over cells, largest concept count first."""
    by_cell = defaultdict(list)
    for p in sorted(glob.glob(str(OUT_DIR / sweep_glob))):
        for c in json.load(open(p)):
            for ds in c.get("datasets") or []:
                for r in ds.get("rows") or []:
                    b = r.get("best")
                    if not b or (r.get("n_other_concepts") or 0) < min_others:
                        continue
                    if b.get("movement_observed") is None:      # needs a readout
                        continue
                    by_cell[(c["donor"], int(c["feat"]), ds["recipient"],
                             ds["dataset"])].append(
                        {"row": int(r["row"]),
                         "n_others": int(r["n_other_concepts"]),
                         "chosen_cols": list(b.get("columns") or [])})
    ranked = sorted(by_cell.items(),
                    key=lambda kv: -max(x["n_others"] for x in kv[1]))
    picked, seen_ds = [], set()
    for key, rows in ranked:
        if key[3] in seen_ds:                                   # spread over datasets
            continue
        rows = sorted(rows, key=lambda x: -x["n_others"])[:rows_per_cell]
        picked.append((key, rows)); seen_ds.add(key[3])
        if len(picked) >= n_cells:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="patchv20clf_*.json",
                    help="sweep to sample rows (and their chosen columns) from")
    ap.add_argument("--n-cells", type=int, default=4)
    ap.add_argument("--rows-per-cell", type=int, default=2)
    ap.add_argument("--min-others", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=8, help="menu size, the sweep's top-cols")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(OUT_DIR / "column_effectiveness_probe.json"))
    args = ap.parse_args()

    import torch
    torch.use_deterministic_algorithms(True)
    from scripts.rebuttal.patch_search import (
        ACTIVE_FLOOR, EPS, build_recip, build_recip_shared, column_sensitivity,
        current_env, extract_acts, load_dataset_context, make_evaluator, raw_space,
        required_env, FWD)

    picked = sample_rows(args.sweep, args.n_cells, args.rows_per_cell, args.min_others)
    runnable = [(k, v) for k, v in picked
                if required_env(k[0], k[2]) == current_env() and k[2] != "carte"]
    print(f"sampled {len(picked)} cells, {len(runnable)} runnable in this env\n")

    results = []
    for (donor, feat, recipient, dataset), rows in runnable:
        npzs = [f for f in glob.glob(str(FWD / "*" / f"{dataset}.npz"))
                if str(np.load(f, allow_pickle=True)["strong_model"]) == donor
                and str(np.load(f, allow_pickle=True)["weak_model"]) == recipient]
        if not npzs:
            print(f"{donor} f{feat} -> {recipient}/{dataset}: no npz"); continue
        try:
            X_ctx, y_ctx, X_query, _, row_indices, task = load_dataset_context(
                donor, dataset, query_source="holdout")
            space = raw_space(donor, dataset, row_indices, X_query)
            a0, _ = extract_acts(donor, dataset, X_ctx, y_ctx, X_query, task, args.device)
            shared = build_recip_shared(donor, recipient, dataset, args.device)
        except Exception as exc:
            print(f"{donor} f{feat} -> {recipient}/{dataset}: SETUP {type(exc).__name__}: {exc}")
            continue
        if shared is None:
            continue
        for rinfo in rows:
            row = rinfo["row"]
            try:
                a_base = a0[row].copy()
                recip = build_recip(shared, donor, recipient, dataset, npzs[0],
                                    row, a_base, feat, args.device)
                if recip is None or not np.isfinite(recip.get("interval", np.nan)):
                    continue
                others = np.array([f for f in recip["fids"] if f != feat], dtype=int)
                ev = make_evaluator(donor, dataset, X_ctx, y_ctx, X_query, space,
                                    task, args.device, row, a_ref=a0[row])
                probes = column_sensitivity(ev, space, space.row(row), a_base, feat,
                                            others, max_levels=None, keep_vectors=True)
                loo_c = abs(float(recip["interval"]))
                loo = recip["loo_by_fid"]
                a_c = abs(float(a_base[feat]))
                per_col = defaultdict(lambda: {"slope": -np.inf, "net": -np.inf,
                                               "ratio": -np.inf})
                for pr in probes:
                    dL = pr["delta_log_freq"]
                    if not (np.isfinite(pr.get("slope", np.nan)) and dL > 0):
                        continue
                    av = pr["_a_vec"]
                    gain = abs(float(a_base[feat] - av[feat])) / max(a_c, EPS) * loo_c
                    spend = 0.0
                    for j in others:
                        aj = abs(float(a_base[j]))
                        if aj <= ACTIVE_FLOOR:
                            continue
                        spend += abs(float(av[j] - a_base[j])) / aj * float(loo.get(int(j), 0.0))
                    c = per_col[pr["column"]]
                    c["slope"] = max(c["slope"], pr["slope"])
                    c["net"] = max(c["net"], (gain - spend) / dL)
                    c["ratio"] = max(c["ratio"], gain / (spend + EPS))
                    c["name"] = pr["column_name"]

                def order_of(key):
                    return [c for c, _ in sorted(per_col.items(),
                                                 key=lambda kv: -kv[1][key])]

                def rank_of(cols, order):
                    return [order.index(c) + 1 if c in order else None for c in cols]

                orders = {k: order_of(v) for k, v in
                          (("current", "slope"), ("net", "net"), ("ratio", "ratio"))}
                # It is the RANKING that matters, not set overlap: the greedy visits
                # columns in rank order and every commit changes the base the rest are
                # evaluated against, so two identical top-K SETS in different orders
                # can end at different patches. Kendall's tau over the full ordering is
                # the agreement statistic; the first visit is the path's anchor.
                from scipy.stats import kendalltau
                rank_cur = {c: i for i, c in enumerate(orders["current"])}
                taus = {}
                for k in ("net", "ratio"):
                    rk = {c: i for i, c in enumerate(orders[k])}
                    common = list(rank_cur)
                    taus[k] = float(kendalltau([rank_cur[c] for c in common],
                                               [rk[c] for c in common]).statistic)
                chosen = rinfo["chosen_cols"]
                rec = {"donor": donor, "feat": feat, "recipient": recipient,
                       "dataset": dataset, "row": row, "n_probed_cols": len(per_col),
                       "n_others": rinfo["n_others"],
                       "order_current": orders["current"], "order_net": orders["net"],
                       "order_ratio": orders["ratio"],
                       "tau_net": taus["net"], "tau_ratio": taus["ratio"],
                       "first_visit": {k: orders[k][0] for k in orders},
                       "chosen_cols": chosen,
                       "chosen_rank_current": rank_of(chosen, orders["current"]),
                       "chosen_rank_net": rank_of(chosen, orders["net"]),
                       "chosen_rank_ratio": rank_of(chosen, orders["ratio"]),
                       "names": {str(c): per_col[c]["name"] for c in per_col}}
                results.append(rec)
                print(f"{donor} f{feat} -> {recipient}/{dataset} row {row} "
                      f"(k-1={rinfo['n_others']}, {len(per_col)} cols): "
                      f"tau current~net {taus['net']:+.2f}, current~ratio {taus['ratio']:+.2f}"
                      f" | first visit {orders['current'][0]} -> net {orders['net'][0]}"
                      f" -> ratio {orders['ratio'][0]}")
                print(f"    chosen cols {chosen}: rank now {rec['chosen_rank_current']} "
                      f"-> net {rec['chosen_rank_net']} -> ratio {rec['chosen_rank_ratio']}",
                      flush=True)
            except Exception as exc:
                print(f"  row {row}: {type(exc).__name__}: {exc}", flush=True)
        json.dump(results, open(args.out, "w"), indent=2)

    if results:
        print(f"\n{len(results)} rows probed")
        for k in ("net", "ratio"):
            t = np.array([r[f"tau_{k}"] for r in results])
            same_first = np.mean([r["first_visit"]["current"] == r["first_visit"][k]
                                  for r in results])
            print(f"  rank agreement current vs {k}: Kendall tau med {np.median(t):+.2f} "
                  f"(min {t.min():+.2f}, max {t.max():+.2f}); "
                  f"same FIRST visit on {same_first:.0%} of rows")
        # does either ranking place the search's own final choices higher?
        for key in ("current", "net", "ratio"):
            rk = [r_ for r in results for r_ in r[f"chosen_rank_{key}"] if r_ is not None]
            print(f"  chosen columns' median rank under {key}: {np.median(rk):.1f} "
                  f"(in-menu at top-{args.top_k}: "
                  f"{np.mean([r_ <= args.top_k for r_ in rk]):.0%})")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
