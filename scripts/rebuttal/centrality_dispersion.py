#!/usr/bin/env python3
"""Is centrality behaving the same across datasets, or does it need norming?

centrality is rank-based, so its VALUE is scale-free within a cell by construction.
What is not guaranteed uniform across cells:

  sensitivity   a tight reconstruction-loss histogram turns small loss changes into
                large rank jumps; a wide one is sluggish. The term's within-row
                discrimination power -- its effective weight against suppression and
                blast -- then varies by dataset.
  resolution    the smoothed rank floor is ~1/(n+1); dataset size caps the achievable
                ratio range.
  start bias    real rows' centrality is uniform by construction, but the SEARCHED rows
                are sampled from accepted rows -- if acceptance correlates with recon
                loss, starts are not uniform and the ratio's baseline shifts per cell.

Measured on chosen patches (v19 by default; the machinery is unchanged in v20):

  1. per-cell dispersion of log(centrality_ratio), against the cell's null width
  2. variance decomposition: how much of log(ratio) variance is BETWEEN cells --
     the share a per-dataset norm could even address
  3. centrality_start quartiles per cell vs the uniform ideal (0.25 / 0.50 / 0.75)
  4. rank-correlation of null width vs ratio dispersion across cells

The null width per cell is estimated from the searched rows' own recon_loss_start
(deduped by row) -- a SAMPLE of the runtime null, whose size is also reported so the
bias is visible.

Usage:
    python -m scripts.rebuttal.centrality_dispersion --inputs output/rebuttal/patchv19clf_*.json
"""
import argparse
import glob
import json
from collections import defaultdict

import numpy as np

from scripts._project_root import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "output" / "rebuttal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=None)
    ap.add_argument("--min-rows", type=int, default=5,
                    help="cells with fewer chosen patches are pooled into the variance "
                         "decomposition but not printed individually")
    args = ap.parse_args()
    paths = args.inputs or sorted(glob.glob(str(OUT_DIR / "patchv19clf_*.json")))

    cells = defaultdict(lambda: {"ratio": [], "start": [], "null": {}})
    for p in paths:
        for c in json.load(open(p)):
            for ds in c.get("datasets") or []:
                for r in ds.get("rows") or []:
                    b = r.get("best")
                    key = (c["donor"], ds["dataset"])
                    if r.get("recon_loss_start") is not None:
                        cells[key]["null"][int(r["row"])] = float(r["recon_loss_start"])
                    if not b or b.get("centrality_ratio") is None:
                        continue
                    cr = float(b["centrality_ratio"])
                    if not (np.isfinite(cr) and cr > 0):
                        continue
                    cells[key]["ratio"].append(np.log(cr))
                    if r.get("centrality_start") is not None:
                        cells[key]["start"].append(float(r["centrality_start"]))

    print(f"{len(paths)} shards, {len(cells)} (donor, dataset) cells\n")
    print(f"{'donor/dataset':<40} {'n':>4} {'null n':>6} {'null IQR/med':>12} "
          f"{'log-ratio IQR':>13} {'start med':>9}")
    widths, disps, printed = [], [], 0
    for key in sorted(cells):
        d = cells[key]
        lr = np.asarray(d["ratio"])
        nul = np.asarray(sorted(d["null"].values()))
        if len(nul) >= 2 and len(lr) >= 2:
            width = (np.percentile(nul, 75) - np.percentile(nul, 25)) / np.median(nul)
            disp = np.percentile(lr, 75) - np.percentile(lr, 25)
            widths.append(width); disps.append(disp)
        if len(lr) < args.min_rows:
            continue
        printed += 1
        st = np.asarray(d["start"])
        print(f"{key[0] + '/' + key[1]:<40} {len(lr):>4} {len(nul):>6} "
              f"{width:>12.3f} {disp:>13.3f} "
              f"{np.median(st) if len(st) else float('nan'):>9.2f}")
    hidden = len(cells) - printed
    print(f"({hidden} cells under {args.min_rows} chosen patches not printed; "
          f"all cells enter the aggregates below)")

    # 2. variance decomposition of log(centrality_ratio)
    groups = [np.asarray(d["ratio"]) for d in cells.values() if len(d["ratio"]) >= 2]
    allv = np.concatenate(groups)
    grand = allv.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((allv - grand) ** 2).sum()
    print(f"\nlog(centrality_ratio) over {len(allv)} chosen patches in {len(groups)} cells:")
    print(f"  between-cell share of variance (eta^2): {ss_between / ss_total:.1%}")
    print(f"  pooled IQR {np.percentile(allv, 75) - np.percentile(allv, 25):.3f}, "
          f"per-cell IQR median {np.median(disps):.3f} "
          f"(p10 {np.percentile(disps, 10):.3f}, p90 {np.percentile(disps, 90):.3f})")

    # 3. start uniformity, pooled
    st = np.concatenate([np.asarray(d["start"]) for d in cells.values() if d["start"]])
    q = np.percentile(st, [25, 50, 75])
    print(f"\ncentrality_start quartiles over {len(st)} rows: "
          f"{q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}   (uniform ideal 0.25 / 0.50 / 0.75)")

    # 4. does null width predict ratio dispersion?
    if len(widths) > 2:
        rw = np.argsort(np.argsort(widths)); rd = np.argsort(np.argsort(disps))
        rho = np.corrcoef(rw, rd)[0, 1]
        print(f"\nnull width vs per-cell log-ratio dispersion, rank corr over "
              f"{len(widths)} cells: {rho:+.2f}")


if __name__ == "__main__":
    main()
