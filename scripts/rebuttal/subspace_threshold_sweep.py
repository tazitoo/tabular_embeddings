#!/usr/bin/env python3
"""REBUTTAL: threshold sensitivity of the intervention-vs-embedding split.

subspace_analysis.py defines the recipient's "active subspace" as the top-k
eigenvectors capturing 90% of the embedding variance, then reports what fraction
of the deployed-intervention energy lies inside it (aligned) vs orthogonal
(novel). That 90% cut is a choice, so this sweeps the cut and reports how the
median aligned fraction moves -- so we can state the split's robustness (or show
the whole curve) rather than lean on a single arbitrary threshold.

Deterministic: reads output/rebuttal/forward_deltas/*/*.npz (same units as
subspace_analysis). One eigendecomposition per unit; aligned fraction evaluated
at every threshold from that single decomposition.

Usage:
    python -m scripts.rebuttal.subspace_threshold_sweep
"""
import json
from collections import defaultdict

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import load_test_embeddings
from scripts.matching.utils import load_norm_stats as load_norm_stats_matching
from scripts.rebuttal.subspace_analysis import _eig_cov, _k_for_variance, DELTA_DIR

THRESHOLDS = [0.70, 0.80, 0.90, 0.95, 0.99]
DELTA_VAR = 0.90   # keep the delta-significant-dim cut fixed (the question is the active cut)


def unit_aligned(npz_path, emb_cache, norm_cache):
    d = np.load(npz_path, allow_pickle=True)
    if "deployed_delta" not in d.files:
        return None
    recipient = str(d["weak_model"])
    dataset = npz_path.stem
    dd = np.asarray(d["deployed_delta"], dtype=np.float64)
    D = dd[np.linalg.norm(dd, axis=1) > 1e-12]
    if len(D) < 5:
        return None
    if recipient not in emb_cache:
        emb_cache[recipient] = load_test_embeddings(recipient)
        norm_cache[recipient] = load_norm_stats_matching(recipient)
    if dataset not in emb_cache[recipient] or dataset not in norm_cache[recipient]:
        return None
    Xn = np.asarray(emb_cache[recipient][dataset], dtype=np.float64)
    mean, std = norm_cache[recipient][dataset]
    Xraw = Xn * np.asarray(std) + np.asarray(mean)

    _, lam_e, V_e = _eig_cov(Xraw, center=True)
    if lam_e.sum() <= 0:
        return None
    Du = D / np.linalg.norm(D, axis=1, keepdims=True)
    _, lam_d, V_d = _eig_cov(Du, center=False)
    rank_d = int((lam_d > 1e-10 * lam_d.max()).sum())
    kd = max(1, min(_k_for_variance(lam_d, DELTA_VAR), rank_d))
    U = V_d[:, :kd]
    w = lam_d[:kd] / lam_d[:kd].sum()

    out = {"recipient": recipient}
    for tau in THRESHOLDS:
        ke = max(1, min(_k_for_variance(lam_e, tau), V_e.shape[1]))
        E = V_e[:, :ke]
        aligned = float(((E.T @ U) ** 2).sum(0) @ w)
        out[tau] = aligned
    return out


def main():
    emb_cache, norm_cache = {}, {}
    units = []
    for pair_dir in sorted(DELTA_DIR.glob("*_vs_*")):
        for npz_path in sorted(pair_dir.glob("*.npz")):
            u = unit_aligned(npz_path, emb_cache, norm_cache)
            if u:
                units.append(u)
    if not units:
        print(f"No units under {DELTA_DIR}")
        return

    print(f"\nThreshold sensitivity of the aligned/novel split (n={len(units)} units)")
    print(f"{'active-subspace variance cut':<30}{'aligned(med)':>13}{'novel(med)':>12}")
    for tau in THRESHOLDS:
        a = np.array([u[tau] for u in units])
        print(f"  top-k capturing {int(tau*100)}% variance{'':<6}{np.median(a):>10.3f}{1-np.median(a):>12.3f}")

    print(f"\nby recipient (aligned median at each cut):")
    byr = defaultdict(list)
    for u in units:
        byr[u["recipient"]].append(u)
    hdr = "  ".join(f"{int(t*100)}%" for t in THRESHOLDS)
    print(f"  {'recipient':<12}{hdr:>34}")
    for r, us in sorted(byr.items()):
        cells = "  ".join(f"{np.median([u[t] for u in us]):.2f}" for t in THRESHOLDS)
        print(f"  {r:<12}{cells:>34}  (n={len(us)})")

    out = {"n_units": len(units), "thresholds": THRESHOLDS,
           "aligned_median": {str(t): float(np.median([u[t] for u in units])) for t in THRESHOLDS}}
    p = PROJECT_ROOT / "output" / "rebuttal" / "subspace_threshold_sweep.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
