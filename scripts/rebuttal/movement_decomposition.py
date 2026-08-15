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
import glob
import json
from collections import defaultdict

import numpy as np

from scripts._project_root import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "output" / "rebuttal"
FLOOR = 0.01
SWEEPS = [("v19", "patchv19clf_*.json"), ("v20", "patchv20clf_*.json")]


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


def med_q(vals, lo=25, hi=75):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return "      -              "
    return f"{np.percentile(v, lo):7.4f} {np.median(v):7.4f} {np.percentile(v, hi):7.4f}"


def main():
    data = {name: load(pat) for name, pat in SWEEPS}
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


if __name__ == "__main__":
    main()
