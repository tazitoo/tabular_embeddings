#!/usr/bin/env python3
"""How much did the recipient's prediction actually change -- and whose doing was it?

Per chosen patch, in true-class probability units:

  planned      suppression_frac x interval: what removing c's suppressed share should
               move the prediction, from LOO
  observed     p_patched - p_transfer: what the recipient actually did. Recorded in
               v20 (movement_observed); reconstructed for earlier sweeps from
               toward_ablation x its floored denominator, exact because that is how
               toward was computed
  bystanders   the signed first-order share of the observed movement caused by the
               other k-1 concepts (v20+ only -- est_bystander was not recorded before)
  attributed   observed - bystanders, the movement credited to c (v20+; equals
               observed on fallback rows)

The v19 bystander/attributed columns are BLANK, not zero: filling them needs the
re-scoring run (one recipient forward per row). The observed and planned columns are
instrument-independent and comparable across sweeps.

Usage:
    python -m scripts.rebuttal.movement_decomposition
"""
import argparse
import glob
import json
from collections import defaultdict

import numpy as np

from scripts._project_root import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "output" / "rebuttal"
FLOOR = 0.01


_npz_cache = {}


def transfer_moved(donor, recipient, dataset, row):
    """|p_transfer - p_weak| for this row: what the WHOLE k-concept transfer moved the
    true-class probability. The reference every per-concept movement should be read
    against -- a single concept can only move its own share of this."""
    key = (donor, recipient, dataset)
    if key not in _npz_cache:
        _npz_cache[key] = None
        for f in glob.glob(str(OUT_DIR / "forward_deltas" / "*" / f"{dataset}.npz")):
            z = np.load(f, allow_pickle=True)
            if str(z["strong_model"]) == donor and str(z["weak_model"]) == recipient:
                _npz_cache[key] = {"pw": np.asarray(z["preds_weak"]),
                                   "pi": np.asarray(z["preds_intervened"]),
                                   "ps": np.asarray(z["preds_strong"]),
                                   "y": np.asarray(z["y_query"])}
                break
    z = _npz_cache[key]
    if z is None:
        return None
    pw, pi, ps, y = z["pw"][row], z["pi"][row], z["ps"][row], int(z["y"][row])
    if np.asarray(pw).ndim >= 1 and np.asarray(pw).size > 1:
        return {"moved": float(abs(float(pi[y]) - float(pw[y]))),
                # the ORIGINAL donor-recipient disagreement, the row's own scale;
                # admission required >= min_gap, so this is floored by construction
                "gap": float(abs(float(ps[y]) - float(pw[y]))),
                # +1 direction = toward the recipient's own (weak) prediction = OPENING
                "open_sign": float(np.sign(float(pw[y]) - float(pi[y])))}
    return None                       # regression rows are not in these sweeps


def load(pattern):
    rows = []
    for p in sorted(glob.glob(str(OUT_DIR / pattern))):
        for c in json.load(open(p)):
            for ds in c.get("datasets") or []:
                for r in ds.get("rows") or []:
                    b = r.get("best")
                    iv, tw = r.get("ablation_interval"), (b or {}).get("toward_ablation")
                    if not b or iv is None or tw is None:
                        continue
                    if not (np.isfinite(iv) and np.isfinite(tw)):
                        continue
                    denom = float(np.copysign(max(abs(iv), FLOOR), iv if iv != 0 else 1.0))
                    obs = b.get("movement_observed")
                    est = b.get("est_bystander")
                    fb = b.get("attribution_fallback")
                    tm = transfer_moved(c["donor"], ds["recipient"],
                                        ds["dataset"], int(r["row"]))
                    rows.append({
                        "cell": (c["donor"], ds["dataset"]),
                        "cols": list(b.get("columns") or []),
                        "gap_opened": r.get("gap_opened"),
                        "planned": float(b["suppression_frac"]) * float(iv),
                        # v20 records observed directly; pre-v20 toward was the RAW
                        # movement over denom, so toward x denom recovers it exactly
                        "observed": float(obs) if obs is not None else float(tw) * denom,
                        "est_bystander": float(est) if est is not None else None,
                        "fallback": fb,
                        "interval": float(iv),
                        "transfer": tm["moved"] if tm else None,
                        "gap": tm["gap"] if tm else None,
                        "open_sign": tm["open_sign"] if tm else None,
                    })
    return rows


TYPE_CACHE = OUT_DIR / "raw_space_column_types.json"


def column_type_map(cells):
    """(donor, dataset) -> set of categorical column indices, from the SPACE'S OWN
    classification: raw_space rebuilds the fitted generator, so the types are exactly
    the ones the search saw -- never re-derived from values (a v10-era bug class) and
    never borrowed from another model's preprocessing. Cached to disk incrementally;
    rebuilding is CPU-only but costs a generator refit per cell."""
    cache = json.loads(TYPE_CACHE.read_text()) if TYPE_CACHE.exists() else {}
    missing = [c for c in cells if f"{c[0]}/{c[1]}" not in cache]
    if missing:
        from scripts.rebuttal.patch_search import load_dataset_context, raw_space
        for i, (donor, dataset) in enumerate(missing):
            key = f"{donor}/{dataset}"
            try:
                _, _, X_query, _, row_indices, _ = load_dataset_context(
                    donor, dataset, query_source="holdout")
                space = raw_space(donor, dataset, row_indices, X_query)
                cache[key] = {"cat": sorted(int(j) for j in space.cat),
                              "n_cols": len(space.names)}
            except Exception as exc:
                cache[key] = {"error": f"{type(exc).__name__}: {exc}"}
            if (i + 1) % 10 == 0 or i + 1 == len(missing):
                TYPE_CACHE.write_text(json.dumps(cache, indent=1))
                print(f"  [type map] {i + 1}/{len(missing)} cells", flush=True)
        TYPE_CACHE.write_text(json.dumps(cache, indent=1))
    return {tuple(k.split("/", 1)): set(v["cat"]) for k, v in cache.items()
            if "cat" in v}


def row_gap_opened(r):
    """The recorded gap_opened (v21+), or the identical quantity computed from the
    row's own components for sweeps that predate the recording."""
    g = r.get("gap_opened")
    if g is not None and np.isfinite(g):
        return float(g)
    if not r.get("gap") or r["gap"] <= 0 or r.get("open_sign") is None:
        return None
    att = (r["observed"] if r.get("fallback") or r.get("est_bystander") is None
           else r["observed"] - r["est_bystander"])
    return float(att * r["open_sign"] / r["gap"])


def med_q(vals, lo=25, hi=75):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return "      -              "
    return f"{np.percentile(v, lo):7.4f} {np.median(v):7.4f} {np.percentile(v, hi):7.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", nargs="*", default=["v19", "v20"],
                    help="sweep tags to compare, e.g. v20 v21; each expands to "
                         "patch<TAG>clf_*.json")
    ap.add_argument("--build-type-cache", action="store_true",
                    help="only build the raw_space column-type cache for these sweeps' "
                         "cells and exit. Needs autogluon, so this runs on a WORKER; "
                         "the cache json is then pulled back and the table runs locally.")
    args = ap.parse_args()
    data = {name: load(f"patch{name}clf_*.json") for name in args.sweeps}
    if args.build_type_cache:
        cells = sorted({r["cell"] for rows in data.values() for r in rows})
        tmap = column_type_map(cells)
        print(f"type cache: {len(tmap)}/{len(cells)} cells typed -> {TYPE_CACHE}")
        return
    for name, rows in data.items():
        print(f"{name}: {len(rows)} chosen patches with a recipient readout")

    print(f"\n|movement|, probability units          {'p25':>7} {'med':>7} {'p75':>7}")
    for name, rows in data.items():
        print(f"  {name} donor-recipient gap (original) "
              + med_q([r["gap"] for r in rows]))
        print(f"  {name} whole transfer (k concepts)    "
              + med_q([r["transfer"] for r in rows]))
        print(f"  {name} c's ceiling (full ablation)    "
              + med_q([abs(r["interval"]) for r in rows]))
        print(f"  {name} planned (supp x interval)      "
              + med_q([abs(r["planned"]) for r in rows]))
        print(f"  {name} observed (recipient moved)     "
              + med_q([abs(r["observed"]) for r in rows]))
    for name, rows in data.items():
        est = [r["est_bystander"] for r in rows if r["est_bystander"] is not None]
        if not est:
            print(f"\n  {name}: bystander share not recorded (needs the re-scoring run)")
            continue
        att = [r["observed"] - r["est_bystander"] for r in rows
               if r["est_bystander"] is not None and not r["fallback"]]
        n_fb = sum(1 for r in rows if r["fallback"])
        dom = sum(1 for r in rows
                  if r["est_bystander"] is not None and not r["fallback"]
                  and abs(r["est_bystander"]) > abs(r["observed"] - r["est_bystander"]))
        n_att = len(att)
        print(f"\n  {name} bystanders (signed share)      " + med_q([abs(x) for x in est]))
        print(f"  {name} attributed to c                " + med_q([abs(x) for x in att]))
        print(f"  {name} attribution fallback: {n_fb}/{len(rows)} = {n_fb / len(rows):.1%}"
              f"   bystanders outweigh c's share: {dom}/{n_att} = {dom / max(n_att, 1):.1%}")

    # GAP OPENED: everything above, renormalised by the row's ORIGINAL donor-recipient
    # disagreement, the transfer pipeline's own convention (gap_closed) pointed back at
    # the patch. Unclamped -- the _gc clamp is what made capture_of_ceiling unreadable --
    # so >1 rows are counted, not hidden. Signed: + re-opens the gap, - closes it further.
    print(f"\nGAP OPENED, fraction of the row's original weak-strong disagreement")
    print(f"                                       {'p25':>7} {'med':>7} {'p75':>7}")
    for name, rows in data.items():
        ok = [r for r in rows if r["gap"] and r["gap"] > 0 and r["open_sign"] is not None]
        below = sum(1 for r in ok if r["gap"] < 0.01)
        gc = [r["transfer"] / r["gap"] for r in ok]
        ceil = [abs(r["interval"]) / r["gap"] for r in ok]
        print(f"  {name} transfer had CLOSED            " + med_q(gc))
        print(f"  {name} c's ceiling could re-open      " + med_q(ceil))
        if any(r["est_bystander"] is not None for r in ok):
            go = [(r["observed"] - r["est_bystander"]) * r["open_sign"] / r["gap"]
                  for r in ok if r["est_bystander"] is not None and not r["fallback"]]
        else:
            go = [r["observed"] * r["open_sign"] / r["gap"] for r in ok]
        tag = "attributed" if any(r["est_bystander"] is not None for r in ok) else "observed (uncorrected)"
        print(f"  {name} patch RE-OPENED ({tag:<22})" + med_q(go))
        print(f"  {name} re-opened > its own ceiling: "
              f"{sum(1 for g_, r in zip(go, ok) if abs(g_) > abs(r['interval']) / r['gap']):>4}"
              f"/{len(go)}   gap below admission floor 0.01: {below}/{len(ok)}")

    # PATCH SIZE: how many columns each chosen patch edits, and what that buys in
    # gap_opened -- the size distribution is where a better menu would first show.
    print(f"\nPATCH SIZE, share of chosen patches (and median attributed gap_opened)")
    print("  " + " ".join(f"{'':>6}" if i == 0 else f"{i if i < 6 else '6+':>10}"
                          for i in range(7)))
    for name, rows in data.items():
        cells_line, med_line = [f"{name:<6}"], [f"{'':6}"]
        for size in (1, 2, 3, 4, 5, 6):
            grp = [r for r in rows if (len(r["cols"]) >= 6 if size == 6
                                       else len(r["cols"]) == size)]
            share = len(grp) / max(len(rows), 1)
            go = [g for g in (row_gap_opened(r) for r in grp) if g is not None]
            cells_line.append(f"{share:>10.1%}")
            med_line.append(f"{np.median(go):>10.3f}" if go else f"{'-':>10}")
        print("  " + " ".join(cells_line))
        print("  " + " ".join(med_line) + "   <- median gap_opened at that size")

    # PATCH TYPE: composition of the committed columns, categorical vs continuous,
    # against the space's own base rate. Types come from raw_space's generator, cached.
    all_cells = sorted({r["cell"] for rows in data.values() for r in rows})
    tmap = column_type_map(all_cells)
    cache = json.loads(TYPE_CACHE.read_text()) if TYPE_CACHE.exists() else {}
    print(f"\nPATCH TYPE, committed columns ({len(tmap)}/{len(all_cells)} cells typed)")
    print(f"  {'sweep':<6} {'all-categorical':>16} {'mixed':>8} {'all-continuous':>15}"
          f" {'cat share of slots':>19} {'cat base rate':>14}")
    for name, rows in data.items():
        typed = [r for r in rows if r["cell"] in tmap and r["cols"]]
        comp = {"cat": 0, "mix": 0, "cont": 0}
        slots = cat_slots = 0
        for r in typed:
            cats = tmap[r["cell"]]
            k = sum(1 for j in r["cols"] if j in cats)
            comp["cat" if k == len(r["cols"]) else "cont" if k == 0 else "mix"] += 1
            slots += len(r["cols"]); cat_slots += k
        # base rate: the searched cells' own categorical share, patch-weighted, so a
        # cat-heavy patch mix can be read against what the spaces offered
        base = [len(tmap[r["cell"]]) / max(cache[f"{r['cell'][0]}/{r['cell'][1]}"]["n_cols"], 1)
                for r in typed]
        n = max(len(typed), 1)
        print(f"  {name:<6} {comp['cat'] / n:>16.1%} {comp['mix'] / n:>8.1%} "
              f"{comp['cont'] / n:>15.1%} {cat_slots / max(slots, 1):>19.1%}"
              f" {np.mean(base) if base else float('nan'):>14.1%}")


if __name__ == "__main__":
    main()
