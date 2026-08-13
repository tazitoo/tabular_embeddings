#!/usr/bin/env python3
"""Where do patched rows land in the REAL rows' reconstruction error, per (donor, dataset)?

The objective trades activation drop against staying in-sample, and prices the second as

    rex = max(0, rel_patched / rel_start - 1)

against the row's OWN unpatched error. That reference makes two patches ending at the same
reconstruction error score differently: a row going 0.10 -> 0.50 pays 400% and one going
0.40 -> 0.50 pays 25%, though the dictionary represents the endpoint equally well (or
badly) in both cases. The alternative reference is the population -- what reconstruction
error do REAL rows of this (donor, dataset) have -- which is a property of the endpoint
alone.

That null needs no new compute. patch_search records `recon_rel_start` for every row it
searches, and that IS a real, unpatched row, so the rows of a sweep supply a sample of the
population per (donor, dataset). Deduped by (donor, dataset, row), since every concept
accepted at a row records the same start.

Reported per cell:
  n_real         distinct real rows contributing to the null
  real           median / p95 / max of their reconstruction error
  patched        median / p95 / max of the CHOSEN patch's error
  above_max      fraction of chosen patches whose error exceeds every real row's
  pos            median percentile position of a chosen patch within the null

The question this answers is whether the term is live at all: if patches land inside the
real-row range, its reference barely matters, and if they land past the maximum, it is the
term deciding what the sweep reports.

Usage:
    python -m scripts.rebuttal.patch_recon_position
"""
import glob
import json
from collections import defaultdict

import numpy as np

from scripts._project_root import PROJECT_ROOT

SWEEP = "patchv15clf"
# NOT f"{SWEEP}_*": the shard glob below would swallow this file on the second run.
OUT = PROJECT_ROOT / "output" / "rebuttal" / f"recon_position_{SWEEP}.json"


def load(sweep):
    """(donor, dataset) -> {row: recon_rel_start} and the list of chosen patches."""
    null = defaultdict(dict)
    patched = defaultdict(list)
    files = sorted(glob.glob(str(PROJECT_ROOT / "output" / "rebuttal" / f"{sweep}_*.json")))
    if not files:
        raise SystemExit(f"no {sweep}_*.json")
    for f in files:
        for concept in json.loads(open(f).read()):
            for cell in concept["datasets"]:
                key = (concept["donor"], cell["dataset"])
                for row in cell.get("rows") or []:
                    start = row.get("recon_rel_start")
                    if start is not None and np.isfinite(start):
                        null[key][int(row["row"])] = float(start)   # dedup by row
                    best = row.get("best")
                    if best and best.get("recon_rel") is not None:
                        patched[key].append({
                            "row": int(row["row"]), "feat": concept["feat"],
                            "recipient": cell["recipient"],
                            "start": start, "final": float(best["recon_rel"]),
                            "recon_excess": best.get("recon_excess"),
                            "drop_frac": best.get("drop_frac"),
                            "n_cols": len(best.get("columns") or [])})
    return files, null, patched


def main():
    files, null, patched = load(SWEEP)
    print(f"{SWEEP}: {len(files)} shards, {len(null)} (donor, dataset) cells\n")

    print(f"{'donor/dataset':<34} {'n_real':>6} {'real med/p95/max':<22} "
          f"{'patched med/p95/max':<22} {'>max':>6} {'pos':>6} {'n':>5}")
    out, all_pos, all_above = {}, [], []
    for key in sorted(null, key=lambda k: (k[0], k[1])):
        real = np.array(sorted(null[key].values()))
        ps = patched.get(key, [])
        if len(real) < 2 or not ps:
            continue
        fin = np.array([p["final"] for p in ps])
        # percentile position of each patch inside the real-row sample
        pos = np.searchsorted(real, fin, side="right") / len(real)
        above = float((fin > real.max()).mean())
        all_pos += list(pos); all_above.append(above)
        name = f"{key[0]}/{key[1]}"
        print(f"{name:<34} {len(real):>6} "
              f"{np.median(real):.3f}/{np.percentile(real, 95):.3f}/{real.max():.3f}".ljust(63)
              + f"{np.median(fin):.3f}/{np.percentile(fin, 95):.3f}/{fin.max():.3f}".ljust(22)
              + f"{above:>6.1%} {np.median(pos):>6.2f} {len(ps):>5}")
        out[name] = {
            "n_real": len(real), "n_patched": len(ps),
            "real": {"median": float(np.median(real)),
                     "p95": float(np.percentile(real, 95)), "max": float(real.max())},
            "patched": {"median": float(np.median(fin)),
                        "p95": float(np.percentile(fin, 95)), "max": float(fin.max())},
            "frac_above_real_max": above,
            "median_position": float(np.median(pos)),
        }

    pos = np.array(all_pos)
    print(f"\npooled over {len(out)} cells, {len(pos)} chosen patches")
    print(f"  position in the real-row null: median {np.median(pos):.2f}, "
          f"p90 {np.percentile(pos, 90):.2f}, p10 {np.percentile(pos, 10):.2f}")
    print(f"  at or above every real row:    {float((pos >= 1.0).mean()):.1%} of patches")
    print(f"  at or below every real row:    {float((pos <= 0.0).mean()):.1%} of patches")
    print(f"  per-cell fraction above max:   median {np.median(all_above):.1%}, "
          f"max {max(all_above):.1%}")

    # Is the DOWNWARD tail populated? A monotone term in rel treats "reconstructed better
    # than any real row" as the best possible outcome, though such a row is as atypical as
    # one past the upper end -- it is sitting where the dictionary is unusually happy, not
    # where real rows live. Whether that matters is empirical: it matters if patches go
    # there.
    below = [p for key, ps in patched.items() for p in ps
             if len(null[key]) >= 2 and p["final"] < min(null[key].values())]
    n_all = sum(len(ps) for key, ps in patched.items() if len(null[key]) >= 2)
    print(f"\n  BELOW every real row's reconstruction error: {len(below)}/{n_all} = "
          f"{len(below) / max(n_all, 1):.1%} of chosen patches")
    if below:
        d = np.array([p["drop_frac"] for p in below if p.get("drop_frac") is not None])
        print(f"    their drop_frac: median {np.median(d):.3f} vs "
              f"{np.median([p['drop_frac'] for key, ps in patched.items() for p in ps if p.get('drop_frac') is not None]):.3f} overall")

    # What would each candidate REFERENCE charge the same chosen patches?
    #
    #   own      1 + max(0, rel/rel_start - 1)      what v15 used: the row's own error
    #   typical  rel / median(real)                 continuous, two-sided, population
    #   excursion max(1, rel / p95(real))           one-sided, prices only excursions
    #
    # Same patches, three prices, so the disagreement is the reference's doing and not the
    # search's. What this CANNOT show is whether the argmax would move: the sweep records
    # only the winning candidate per step, so the losers cannot be rescored. That is a
    # recording gap, not a null result.
    print("\nwhat each reference charges the SAME chosen patches (divisor, >1 = penalty)")
    charge = {"own": [], "typical": [], "excursion": []}
    for key, ps in patched.items():
        real = np.array(sorted(null[key].values()))
        if len(real) < 2:
            continue
        med, p95 = float(np.median(real)), float(np.percentile(real, 95))
        for p in ps:
            if p.get("recon_excess") is None or not p.get("start"):
                continue
            charge["own"].append(1.0 + max(0.0, p["recon_excess"]))
            charge["typical"].append(p["final"] / med if med > 0 else np.nan)
            charge["excursion"].append(max(1.0, p["final"] / p95) if p95 > 0 else np.nan)
    for name, vals in charge.items():
        v = np.asarray([x for x in vals if np.isfinite(x)])
        inert = float((v <= 1.0 + 1e-12).mean())
        print(f"  {name:<10} median {np.median(v):.3f}  p95 {np.percentile(v, 95):.3f}  "
              f"max {v.max():.3f}   inert on {inert:.1%} of patches   n={len(v)}")

    own, typ = np.asarray(charge["own"]), np.asarray(charge["typical"])
    ok = np.isfinite(own) & np.isfinite(typ)
    print(f"\n  own vs typical, per patch: they disagree by a factor of "
          f"{np.median(np.abs(np.log(typ[ok] / own[ok]))):.3f} in log terms (median), "
          f"and rank-correlate {np.corrcoef(np.argsort(np.argsort(own[ok])), np.argsort(np.argsort(typ[ok])))[0, 1]:.3f}")

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
