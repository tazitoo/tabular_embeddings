#!/usr/bin/env python3
"""REBUTTAL (ofnL Q2): per-recipient on/off-manifold split of the transfer delta.

Replaces the high-dimensional single-vector cosine ("median cos = 0.03 =>
nearly orthogonal") with the aligned/novel ENERGY split of the deployed transfer
delta against the recipient's own active manifold. Reported PER RECIPIENT
because the manifold is the recipient's own eigenbasis -- pooling across
recipients averages quantities defined in different bases (see the camera-ready
TODO caveat), whereas within a fixed recipient the split is well defined.

Source: output/rebuttal/subspace_summary.json (units list, one entry per
donor->recipient x dataset). Each unit carries aligned_fraction (on-manifold),
novel_fraction (off-manifold), and median_principal_angle_deg.

on-manifold  = median aligned_fraction over the recipient's units
off-manifold = median novel_fraction   (= 1 - on, up to per-unit rounding)
angle        = median median_principal_angle_deg (0 deg aligned, 90 deg off)

Deterministic: no CLI args; canonical path.

Usage:
    python -m scripts.rebuttal.subspace_by_recipient
"""
import glob
import json
from statistics import median

import numpy as np

from scripts._project_root import PROJECT_ROOT

SUMMARY = PROJECT_ROOT / "output" / "rebuttal" / "subspace_summary.json"
FORWARD_DELTAS = PROJECT_ROOT / "output" / "rebuttal" / "forward_deltas"
# Display order: transformer-ICL cluster first, graph model (carte) last.
ORDER = ["tabpfn", "tabicl", "tabicl_v2", "mitra", "tabdpt", "carte"]


def median_k_by_recipient() -> dict:
    """Median transferred-concept count per recipient, over the same below-diagonal
    injections the subspace split uses: optimal_k on accepted rows (k > 0)."""
    pooled = {}
    for f in sorted(glob.glob(str(FORWARD_DELTAS / "*/*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "optimal_k" not in z.files:
            continue
        rec = str(z["recipient_model"])
        k = np.asarray(z["optimal_k"])
        pooled.setdefault(rec, []).extend(k[k > 0].tolist())
    return {r: (median(v), len(v)) for r, v in pooled.items()}


def main():
    units = json.loads(SUMMARY.read_text())["units"]
    by_rec = {}
    for u in units:
        by_rec.setdefault(u["recipient"], []).append(u)
    kmed = median_k_by_recipient()

    rows = []
    for rec in ORDER:
        us = by_rec.get(rec, [])
        if not us:
            continue
        off = median(u["novel_fraction"] for u in us)
        ang = median(u["median_principal_angle_deg"] for u in us)
        km, n_rows = kmed.get(rec, (float("nan"), 0))
        rows.append((rec, len(us), off, ang, km, n_rows))

    print(f"{'recipient':<12}{'n_units':>8}{'off-manifold':>14}{'angle_deg':>11}"
          f"{'median_K':>10}{'n_rows':>8}")
    for rec, n, off, ang, km, n_rows in rows:
        print(f"{rec:<12}{n:>8}{off:>14.2f}{ang:>11.1f}{km:>10.0f}{n_rows:>8}")

    print("\noff-manifold range:",
          f"{min(r[2] for r in rows):.2f}-{max(r[2] for r in rows):.2f}")


if __name__ == "__main__":
    main()
