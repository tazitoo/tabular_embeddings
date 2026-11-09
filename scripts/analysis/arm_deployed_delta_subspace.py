#!/usr/bin/env python3
"""Deployed-DELTA (summed-injection) on/off-manifold view, per transfer ARM.

arm_direction_manifold.py measures each injected concept direction SEPARATELY.
This complements it with the object the recipient model actually sees: the SUMMED
injection Δ(x) = Σ_i a_i(x)·sign_i·v_i·std_w over the top-m concepts (imp,dataset
ordering), the deployed-delta analog of the greedy's `deployed_delta`. The arms'
deltas are dumped by `transfer_decomposition --dump-deltas M` (<tag>_deltas dirs);
the greedy references already carry `deployed_delta` on disk.

For each (donor->recipient, dataset) unit we reuse the paper's intervention-vs-
embedding subspace split (subspace_analysis.analyze_unit):
  aligned_fraction = share of deployed-Δ energy inside the recipient's active 90%
    subspace (reuses existing structure);  novel = 1 - aligned (orthogonal/new).
  principal angle between the significant Δ subspace and the active subspace.
We aggregate the DISTRIBUTION across units, broken down by recipient (the basis is
recipient-specific; deltas of different recipients can't be pooled).

Mac-only CPU (recipient embeddings + norm stats are cached). Usage:
    python -m scripts.analysis.arm_deployed_delta_subspace
"""
import argparse
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.round_paths import (
    FORWARD_ABLATION_DIR, FORWARD_DELTAS_DIR, FORWARD_DELTAS_RANDOM_DIR,
)
from scripts.rebuttal.subspace_analysis import analyze_unit

TD_DIR = Path(str(FORWARD_ABLATION_DIR)).parent / "transfer_decomposition"

# (label, delta dir) -- the none-curve arm dumps plus the on-disk greedy references.
ARMS = [
    ("greedy trained",          FORWARD_DELTAS_DIR),
    ("greedy random",           FORWARD_DELTAS_RANDOM_DIR),
    ("unmatched (ridge)",       TD_DIR / "ridge_deltas"),
    ("unmatched random",        TD_DIR / "ridge_random_deltas"),
    ("matched_map (ridge)",     TD_DIR / "ridge_matched_deltas"),
    ("matched_wdec",            TD_DIR / "matched_wdec_deltas"),
]


def _units(delta_dir, emb_cache, norm_cache):
    units = []
    for f in sorted(glob.glob(str(Path(delta_dir) / "*_vs_*" / "*.npz"))):
        u = analyze_unit(Path(f), emb_cache, norm_cache)
        if u:
            units.append(u)
    return units


def _med(xs):
    a = np.asarray([x for x in xs if x == x], dtype=np.float64)
    return (float(np.median(a)), len(a)) if len(a) else (float("nan"), 0)


def main():
    argparse.ArgumentParser(description="Deployed-delta subspace split per arm").parse_args()
    emb_cache, norm_cache = {}, {}
    recips = sorted({"carte", "mitra", "tabdpt", "tabicl", "tabicl_v2", "tabpfn"})

    rows = []
    per_recip = {}
    for label, d in ARMS:
        if not Path(d).exists():
            rows.append((label, None)); per_recip[label] = {}; continue
        us = _units(d, emb_cache, norm_cache)
        rows.append((label, us))
        by = defaultdict(list)
        for u in us:
            by[u["recipient"]].append(u)
        per_recip[label] = by

    print("\n=== DEPLOYED-DELTA (summed injection) vs ACTIVE SUBSPACE ===")
    print("aligned = share of deployed-Δ energy inside recipient's active 90% subspace; "
          "novel = 1-aligned\n")
    print(f"{'arm':22} {'n':>5} {'aligned':>8} {'novel':>7} {'angle°':>7} {'Δdim90':>7}")
    print("-" * 62)
    for label, us in rows:
        if not us:
            print(f"{label:22} {'(no deltas on disk)':>38}"); continue
        al, n = _med([u["aligned_fraction"] for u in us])
        ang, _ = _med([u["median_principal_angle_deg"] for u in us])
        kd, _ = _med([u["delta_sig_dim_90"] for u in us])
        print(f"{label:22} {n:>5} {al:>8.2f} {1-al:>7.2f} {ang:>7.1f} {kd:>7.0f}")

    print(f"\naligned(median) BY RECIPIENT")
    print(f"{'arm':22} " + "  ".join(f"{r:>9}" for r in recips))
    for label, us in rows:
        if not us:
            continue
        by = per_recip[label]
        cells = []
        for r in recips:
            if by.get(r):
                m, _ = _med([u["aligned_fraction"] for u in by[r]])
                cells.append(f"{m:>9.2f}")
            else:
                cells.append(f"{'--':>9}")
        print(f"{label:22} " + "  ".join(cells))


if __name__ == "__main__":
    main()
