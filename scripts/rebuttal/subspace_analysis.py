#!/usr/bin/env python3
"""REBUTTAL: intervention-vs-embedding subspace comparison (ofnL Q2).

ofnL asks whether the "injected direction is orthogonal to the weak model" (the
paper's median cos 0.03) is measured against the row-specific hidden state or
against the subspace the weak model's dictionary actually spans. A cosine
against a single vector is near-zero for any two vectors in high dim, so it
proves nothing. The honest test compares two SUBSPACES on the SAME samples:

  - embedding eigenbasis: PCA of the recipient's activations on a dataset
    (where the model's representations actually vary; intrinsic dim is low).
  - intervention eigenbasis: PCA of the ACTUAL deployed transfer deltas
    (deployed_delta from transfer_sweep_symmetric --forward), the real
    transferred contribution injected into that recipient.

Aggregation (see below): the embedding eigenbasis is a property of
(recipient, dataset) ONLY — fixed across donors. The deployed delta is a
property of (donor->recipient, dataset). Deltas for different recipients live
in different spaces (different dims, per-dataset normalization), so they CANNOT
be pooled into one global basis. Instead each (donor, recipient, dataset) unit
is collapsed to one scale-free scalar and the DISTRIBUTION is aggregated across
units, broken down by recipient model.

Per-unit scalar — variance capture:

  c = sum_j (u_j^T C_emb u_j) * w_j / trace(C_emb)

where u_j are the deployed-delta principal directions, w_j = lambda_delta_j /
sum(lambda_delta) their normalized weights, and C_emb is the recipient's
(centered) activation covariance. c in [0,1] is the fraction of the recipient's
activation variance that lies along the deployed-delta subspace. Low c across
units => transfer directions avoid the active subspace (the measured version of
"latent capacity"); high c => they ride the active manifold (concedes W1).

Also reports the principal-angle summary between the top-k delta subspace and
the top-k90 embedding subspace (dims capturing 90% of activation variance) as
the per-unit geometric picture.

Deterministic: reads output/rebuttal/forward_deltas/*/*.npz (recipient raw
embeddings + norm stats are cached; no GPU, no base models).

Usage:
    python -m scripts.rebuttal.subspace_analysis
    python -m scripts.rebuttal.subspace_analysis --json out.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import load_test_embeddings
from scripts.matching.utils import load_norm_stats as load_norm_stats_matching

DELTA_DIR = PROJECT_ROOT / "output" / "rebuttal" / "forward_deltas"


def _eig_cov(M: np.ndarray, center: bool):
    """Symmetric eigendecomposition of a covariance, eigenvalues descending."""
    X = M - M.mean(0, keepdims=True) if center else M
    C = (X.T @ X) / max(len(X), 1)
    w, V = np.linalg.eigh(C)          # ascending
    order = np.argsort(w)[::-1]
    return C, w[order], V[:, order]


def _k_for_variance(evals: np.ndarray, frac: float) -> int:
    """Number of leading eigenvalues capturing `frac` of the total variance."""
    tot = evals.sum()
    if tot <= 0:
        return 0
    csum = np.cumsum(evals) / tot
    return int(np.searchsorted(csum, frac) + 1)


def _principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between the column spans of A, B.

    A, B have orthonormal columns. cos(theta_i) = singular values of A^T B.
    """
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)


def analyze_unit(npz_path: Path, embeddings_cache: dict, norm_cache: dict) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    if "deployed_delta" not in d.files:
        return None
    recipient = str(d["weak_model"])       # forward: recipient = weak
    donor = str(d["strong_model"])
    dataset = npz_path.stem

    dd = np.asarray(d["deployed_delta"], dtype=np.float64)   # (n_query, d) raw recip space
    norms = np.linalg.norm(dd, axis=1)
    D = dd[norms > 1e-12]
    if len(D) < 5:
        return None

    # Recipient RAW embeddings on this dataset (delta lives in raw space).
    if recipient not in embeddings_cache:
        embeddings_cache[recipient] = load_test_embeddings(recipient)
        norm_cache[recipient] = load_norm_stats_matching(recipient)
    if dataset not in embeddings_cache[recipient] or dataset not in norm_cache[recipient]:
        return None
    Xn = np.asarray(embeddings_cache[recipient][dataset], dtype=np.float64)
    mean, std = norm_cache[recipient][dataset]
    Xraw = Xn * np.asarray(std) + np.asarray(mean)

    # Embedding eigenbasis (centered) and delta eigenbasis (unit-normed, uncentered).
    C_emb, lam_e, V_e = _eig_cov(Xraw, center=True)
    trace_emb = float(lam_e.sum())
    if trace_emb <= 0:
        return None
    Du = D / np.linalg.norm(D, axis=1, keepdims=True)
    _, lam_d, V_d = _eig_cov(Du, center=False)

    # SIGNIFICANT dimensions of each basis (top directions holding 90% of that
    # basis's own spectrum). Using the significant delta subspace (not the full
    # rank) avoids a rank confound when comparing to a small active subspace.
    rank_d = int((lam_d > 1e-10 * lam_d.max()).sum())
    kd = max(1, min(_k_for_variance(lam_d, 0.90), rank_d))
    ke = max(1, min(_k_for_variance(lam_e, 0.90), V_e.shape[1]))
    U = V_d[:, :kd]                                  # (d, kd) significant delta dirs
    E = V_e[:, :ke]                                  # (d, ke) active subspace

    # Primary metric: principal angles between the two significant subspaces.
    # 0 deg == aligned (mechanical); ~90 deg == orthogonal (capacity). The
    # baseline for "how orthogonal is unrelated structure" comes from running
    # this same script on the random-transfer deltas, NOT a synthetic subspace.
    angles = _principal_angles(E, U)
    kk = min(kd, ke)
    median_angle = float(np.degrees(np.median(angles[:kk]))) if kk else float("nan")
    min_angle = float(np.degrees(angles[:kk].min())) if kk else float("nan")

    # PRIMARY split: fraction of deployed-intervention energy inside the active
    # subspace (E E^T projection), weighted by delta-PC importance. This is the
    # measured decomposition that replaces the "purely new capacity" claim:
    #   aligned_fraction  = share of interventions that reuse EXISTING structure
    #   1 - aligned       = share that is genuinely NOVEL (orthogonal to active)
    w = lam_d[:kd] / lam_d[:kd].sum()
    overlap_per_pc = (E.T @ U) ** 2          # ||P_E u_j||^2 per delta PC, (ke, kd)->sum
    aligned_fraction = float((overlap_per_pc.sum(0) * w).sum())

    # Secondary: variance-capture scalar (activation variance along the
    # significant delta subspace, weighted by delta importance).
    var_along = np.einsum("di,dj,ji->i", U, C_emb, U)
    c = float((var_along * w).sum() / trace_emb)

    return {
        "donor": donor, "recipient": recipient, "dataset": dataset,
        "n_deltas": int(len(D)), "d_emb": int(Xraw.shape[1]),
        "intrinsic_dim_90": ke, "delta_sig_dim_90": kd, "delta_rank": rank_d,
        "aligned_fraction": aligned_fraction,         # PRIMARY: existing-structure share
        "novel_fraction": 1.0 - aligned_fraction,     # orthogonal / new-capacity share
        "median_principal_angle_deg": median_angle,
        "min_principal_angle_deg": min_angle,
        "variance_capture": c,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta-dir", type=Path, default=DELTA_DIR,
                    help="Directory of <pair>/<dataset>.npz with deployed_delta. "
                         "Run on forward_deltas (trained) or the random-transfer "
                         "equivalent for the matched baseline.")
    ap.add_argument("--json", type=Path,
                    default=PROJECT_ROOT / "output" / "rebuttal" / "subspace_summary.json")
    args = ap.parse_args()

    embeddings_cache, norm_cache = {}, {}
    units = []
    for pair_dir in sorted(args.delta_dir.glob("*_vs_*")):
        for npz_path in sorted(pair_dir.glob("*.npz")):
            u = analyze_unit(npz_path, embeddings_cache, norm_cache)
            if u:
                units.append(u)

    if not units:
        print(f"No deployed-delta units found under {args.delta_dir}")
        return

    def _stats(vals):
        a = np.asarray([v for v in vals if v == v], dtype=np.float64)  # drop NaN
        if not len(a):
            return None
        return {"median": float(np.median(a)), "mean": float(a.mean()),
                "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
                "n": int(len(a))}

    by_recip = defaultdict(list)
    for u in units:
        by_recip[u["recipient"]].append(u)

    out = {
        "n_units": len(units),
        "overall_aligned_fraction": _stats([u["aligned_fraction"] for u in units]),
        "overall_principal_angle_deg": _stats([u["median_principal_angle_deg"] for u in units]),
        "by_recipient": {
            r: {"aligned_fraction": _stats([u["aligned_fraction"] for u in us]),
                "principal_angle_deg": _stats([u["median_principal_angle_deg"] for u in us]),
                "median_intrinsic_dim_90": float(np.median([u["intrinsic_dim_90"] for u in us])),
                "median_delta_sig_dim_90": float(np.median([u["delta_sig_dim_90"] for u in us])),
                "median_d_emb": float(np.median([u["d_emb"] for u in us]))}
            for r, us in sorted(by_recip.items())
        },
        "units": units,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=2))

    af = out["overall_aligned_fraction"]
    ang = out["overall_principal_angle_deg"]
    print(f"\n{'='*70}\nINTERVENTION vs EMBEDDING SUBSPACE (ofnL Q2)\n{'='*70}")
    print(f"  source: {args.delta_dir}")
    print(f"  units (donor->recipient, dataset): {out['n_units']}")
    print(f"  SPLIT — deployed-intervention energy decomposition:")
    print(f"     aligned (existing structure): median={af['median']:.2f}  "
          f"IQR[{af['p25']:.2f}, {af['p75']:.2f}]")
    print(f"     novel   (orthogonal/new cap): median={1-af['median']:.2f}")
    print(f"  principal angle (Δ vs active subspace): median={ang['median']:.1f}° "
          f"IQR[{ang['p25']:.1f}, {ang['p75']:.1f}]°")
    print(f"\n  by recipient:")
    print(f"  {'recipient':<12} {'n':>4} {'d':>5} {'idim90':>7} {'Δdim90':>7} "
          f"{'aligned(med)':>13} {'angle°(med)':>12}")
    for r, s in out["by_recipient"].items():
        print(f"  {r:<12} {s['aligned_fraction']['n']:>4} {int(s['median_d_emb']):>5} "
              f"{int(s['median_intrinsic_dim_90']):>7} {int(s['median_delta_sig_dim_90']):>7} "
              f"{s['aligned_fraction']['median']:>13.2f} {s['principal_angle_deg']['median']:>12.1f}")
    print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
