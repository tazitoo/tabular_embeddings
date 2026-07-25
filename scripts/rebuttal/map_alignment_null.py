#!/usr/bin/env python3
"""REBUTTAL triage: is the deployed-delta alignment a ridge-map artifact?

The concept map M is a regularized ridge fit, whose outputs are biased toward
the dominant directions of the target it was fit on. So the observed ~51%
alignment of deployed deltas with the recipient's top-variance subspace might be
a MECHANICAL property of M (it aligns anything), not of the transferred concepts.

Cheap isolation (pure matrix multiply, no GPU, no greedy): push ARBITRARY
directions through M and measure their aligned/novel split against the same
recipient active subspace used in subspace_analysis. Compare to:
  - trained deployed deltas (~0.51, the number in question)
  - chance k_e/d (a random direction's expected alignment)

Outcomes:
  random-through-M ~ 0.51  => M is the artifact; the 0.51 means nothing.
  random-through-M ~ k_e/d => M is input-faithful; the trained 0.51 is real.

M is the SAME map the forward transfer uses (donor->recipient, built from the
round-10 SAE decoder atoms via filter_landmarks + fit_concept_map). Random
inputs are (a) isotropic unit vectors in donor space and (b) the donor's own
unmatched decoder atoms (a realistic but unselected input set).

Usage:
    python -m scripts.rebuttal.map_alignment_null                 # all pairs w/ deltas
    python -m scripts.rebuttal.map_alignment_null --models tabpfn tabdpt
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import load_sae, load_test_embeddings
from scripts.matching.utils import load_norm_stats as load_norm_stats_matching
from scripts.intervention.transfer_virtual_nodes import (
    extract_decoder_atoms, fit_concept_map, filter_landmarks,
)
from scripts.rebuttal.transfer_sweep_symmetric import (
    get_matched_pairs, get_unmatched_features, DEFAULT_MATCHING_FILE,
)
from scripts.rebuttal.subspace_analysis import _eig_cov, _k_for_variance

FWD_DIR = PROJECT_ROOT / "output" / "rebuttal" / "forward_deltas"
RNG = np.random.default_rng(0)


def _aligned(V_active, dirs):
    """Mean aligned fraction ||P_E d||^2/||d||^2 over rows of `dirs` (each a dir)."""
    P = (dirs @ V_active)                       # (n, ke) coords in active subspace
    num = (P ** 2).sum(1)
    den = (dirs ** 2).sum(1) + 1e-12
    return float(np.mean(num / den))


def build_map(donor, recipient, saes, matching_file):
    """The forward transfer's M (donor->recipient) + donor unmatched atoms."""
    atoms_d = extract_decoder_atoms(saes[donor]).numpy()
    atoms_r = extract_decoder_atoms(saes[recipient]).numpy()
    m_pairs = get_matched_pairs(donor, recipient, matching_file=matching_file)
    if len(m_pairs) < 5:
        return None, None, None
    src = [s for s, _ in m_pairs]; tgt = [t for _, t in m_pairs]
    fs, ft, fp, _ = filter_landmarks(atoms_d[src], atoms_r[tgt], m_pairs, min_cosine=0.0, alpha=1.0)
    if len(fp) < 5:
        return None, None, None
    M, _ = fit_concept_map(fs, ft, alpha=1.0)          # recipient = donor @ M.T
    unmatched = get_unmatched_features(donor, recipient, matching_file=matching_file)
    return M, atoms_d, unmatched


def analyze_pair(donor, recipient, saes, emb, norm, matching_file, n_rand=400):
    M, atoms_d, unmatched = build_map(donor, recipient, saes, matching_file)
    if M is None:
        return []
    d_src = atoms_d.shape[1]
    # (a) isotropic random directions in donor space, mapped through M
    G = RNG.standard_normal((n_rand, d_src))
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    rand_mapped = G @ M.T                                # (n_rand, d_recipient)
    # (b) donor's own unmatched decoder atoms, mapped through M
    um_mapped = atoms_d[unmatched] @ M.T if unmatched else None

    out = []
    for dataset in sorted(emb[recipient]):
        if dataset not in norm[recipient]:
            continue
        Xn = np.asarray(emb[recipient][dataset], dtype=np.float64)
        mean, std = norm[recipient][dataset]
        Xraw = Xn * np.asarray(std) + np.asarray(mean)
        _, lam_e, V_e = _eig_cov(Xraw, center=True)
        ke = max(1, min(_k_for_variance(lam_e, 0.90), V_e.shape[1]))
        E = V_e[:, :ke]
        rec = {"donor": donor, "recipient": recipient, "dataset": dataset,
               "k_e": int(ke), "d": int(V_e.shape[1]), "chance": ke / V_e.shape[1],
               "aligned_random_through_M": _aligned(E, rand_mapped)}
        if um_mapped is not None and len(um_mapped):
            rec["aligned_unmatched_atoms_through_M"] = _aligned(E, um_mapped)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs=2, default=None, metavar=("STRONG", "WEAK"))
    ap.add_argument("--matching-file", type=str, default=str(DEFAULT_MATCHING_FILE))
    args = ap.parse_args()

    if args.models:
        pairs = [tuple(args.models)]
    else:
        pairs = []
        for pd in sorted(FWD_DIR.glob("*_vs_*")):
            a, b = pd.name.split("_vs_")
            pairs.append((a, b)); pairs.append((b, a))   # both directions; recipient=weak per data

    saes, emb, norm = {}, {}, {}
    def _load(m):
        if m not in saes:
            s, _ = load_sae(m, device="cpu"); s.eval(); saes[m] = s
            emb[m] = load_test_embeddings(m); norm[m] = load_norm_stats_matching(m)

    results = []
    for donor, recipient in pairs:
        # forward transfer recipient = weaker model; skip the (donor,recipient) whose
        # forward deltas we don't have (keeps this aligned with subspace_analysis units)
        try:
            _load(donor); _load(recipient)
        except Exception:
            continue
        results.extend(analyze_pair(donor, recipient, saes, emb, norm, args.matching_file))

    if not results:
        print("No results."); return
    ar = np.array([r["aligned_random_through_M"] for r in results])
    ch = np.array([r["chance"] for r in results])
    print(f"\n{'='*70}\nMAP-ALIGNMENT NULL — is the ~0.51 a ridge-map artifact?\n{'='*70}")
    print(f"  units: {len(results)}")
    print(f"  aligned of RANDOM directions through M : median={np.median(ar):.3f}  IQR[{np.percentile(ar,25):.3f},{np.percentile(ar,75):.3f}]")
    print(f"  chance k_e/d                           : median={np.median(ch):.3f}")
    if any("aligned_unmatched_atoms_through_M" in r for r in results):
        au = np.array([r["aligned_unmatched_atoms_through_M"] for r in results if "aligned_unmatched_atoms_through_M" in r])
        print(f"  aligned of donor UNMATCHED atoms thru M: median={np.median(au):.3f}")
    print(f"\n  interpretation:")
    print(f"    random-through-M ~ 0.51  => M mechanically aligns anything (artifact)")
    print(f"    random-through-M ~ {np.median(ch):.3f} (chance) => M is input-faithful; trained 0.51 is real")
    OUT = PROJECT_ROOT / "output" / "rebuttal" / "map_alignment_null.json"
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
