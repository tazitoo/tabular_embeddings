#!/usr/bin/env python3
"""REBUTTAL prerequisite for the OMP upper-bound: are the RANDOM SAE decoder
atoms actually isotropic random directions, or data-aligned?

The sparse-approximation / OMP bound on how many random atoms are needed to
recover a trained transfer delta rests on the atoms being near-isotropic in R^d
(then E[best cos] ~ sqrt(2 ln N / d) and energy accrues ~that much per near-
orthogonal atom). But if the random SAE is archetypal and retains data-derived
k-means centroids in its decoder, its columns are convex combinations of those
centroids -> data-aligned, clustered, low effective rank. Then the isotropic
count estimates are wrong and the atoms reconstruct data-space deltas far more
cheaply than the isotropic table predicts. (This would also help explain the
~0.29 random-through-M alignment in map_alignment_null.)

This measures, per recipient model's random SAE decoder:
  - N (dict size), d (embed dim)
  - E[max cos] isotropic prediction  sqrt(2 ln N / d)
  - OBSERVED mean nearest-neighbour |cos| over atoms (data-aligned >> isotropic)
  - mean |cos| over a random sample of atom pairs
  - effective rank of the atom set (participation ratio of singular values):
    ~d for isotropic, << d if clustered/low-rank

Usage:
    python -m scripts.rebuttal.random_sae_isotropy
    python -m scripts.rebuttal.random_sae_isotropy --sae-dir output/sae_random_baseline --models tabpfn mitra
"""
import argparse
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import load_sae
from scripts.intervention.transfer_virtual_nodes import extract_decoder_atoms

RECIPIENTS = ["tabpfn", "tabicl", "tabicl_v2", "mitra", "tabdpt", "carte"]
RNG = np.random.default_rng(0)


def eff_rank(A):
    """Participation ratio of singular values: (sum s)^2 / sum s^2. ~d if the
    atoms span isotropically, small if they cluster onto few directions."""
    s = np.linalg.svd(A, compute_uv=False)
    s2 = s ** 2
    return float((s.sum() ** 2) / (s2.sum() + 1e-12))


def _metrics(U):
    """(mean NN |cos|, mean random-pair |cos|, eff_rank) for unit-row matrix U."""
    N = U.shape[0]
    m = min(N, 800)
    idx = RNG.choice(N, m, replace=False)
    G = U[idx] @ U.T                                            # (m, N)
    G[np.arange(m), idx] = 0.0                                  # mask self (not -inf!)
    nn = np.abs(G).max(1)
    a = RNG.choice(N, 4000); b = RNG.choice(N, 4000)
    keep = a != b
    pair = np.abs((U[a[keep]] * U[b[keep]]).sum(1))
    return float(nn.mean()), float(pair.mean()), eff_rank(U)


def analyse(model, sae_dir):
    sae, _ = load_sae(model, device="cpu", **({"sae_dir": sae_dir} if sae_dir else {}))
    sae.eval()
    A = extract_decoder_atoms(sae).numpy().astype(np.float64)   # (N, d)
    N, d = A.shape
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    U = A / norms                                               # unit atoms
    nn_o, pair_o, er_o = _metrics(U)
    # matched isotropic reference: Gaussian unit rows, same (N, d)
    R = RNG.standard_normal((N, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    nn_r, pair_r, er_r = _metrics(R)
    return {
        "model": model, "N": N, "d": d,
        "nn_obs": nn_o, "nn_iso": nn_r,
        "pair_obs": pair_o, "pair_iso": pair_r,
        "er_obs": er_o, "er_iso": er_r,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", default=str(PROJECT_ROOT / "output" / "sae_random_baseline"))
    ap.add_argument("--models", nargs="+", default=RECIPIENTS)
    args = ap.parse_args()
    sae_dir = Path(args.sae_dir) if args.sae_dir else None

    print(f"\nRandom-SAE decoder isotropy vs matched Gaussian reference  (sae-dir={args.sae_dir})")
    print(f"{'model':11}{'N':>6}{'d':>5}   {'NN|cos| obs/iso':>16}   "
          f"{'pair|cos| obs/iso':>18}   {'effRank obs/iso':>18}")
    for m in args.models:
        try:
            r = analyse(m, sae_dir)
        except Exception as e:
            print(f"  {m:9} FAIL {type(e).__name__}: {e}")
            continue
        print(f"{r['model']:11}{r['N']:>6}{r['d']:>5}   "
              f"{r['nn_obs']:>7.3f}/{r['nn_iso']:<7.3f}  "
              f"{r['pair_obs']:>8.3f}/{r['pair_iso']:<8.3f}  "
              f"{r['er_obs']:>8.1f}/{r['er_iso']:<8.1f}")
    print("\nRead: obs ~= iso  =>  atoms are isotropic random (OMP count table applies).")
    print("      obs NN/pair |cos| >> iso, or effRank obs << iso  =>  data-aligned/clustered.")


if __name__ == "__main__":
    main()
