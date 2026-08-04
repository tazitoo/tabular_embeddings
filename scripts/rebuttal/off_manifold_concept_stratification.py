#!/usr/bin/env python3
"""REBUTTAL / camera-ready: are OFF-MANIFOLD concepts more labelable?

The functional decomposition splits each deployed transfer DELTA into on/off-manifold
components. This script pushes that attribution down to individual CONCEPTS: for every
accepted concept, project its mapped direction (the virtual atom = donor concept mapped
into the recipient's embedding space) onto the recipient's on-manifold subspace E, and
record its OFF-manifold energy fraction. A concept whose mapped direction lives mostly
off-manifold is a "latent-capacity" concept -- the transfer's most distinctive
contribution.

We then cross that with FIRING DENSITY (fraction of the donor's rows the concept fires
on -- the labelability proxy: dense = little contrast = hard to label). The question:
are off-manifold concepts the sparse, labelable ones? If so, labeling a stratum of them
is the highest-leverage interpretability offer.

off-manifold fraction is acceptance-weighted over every (recipient, dataset) where the
concept was accepted (E is per recipient x dataset, matching functional_decomposition).
All CPU -- no model forward passes.

Usage:
    python -m scripts.rebuttal.off_manifold_concept_stratification            # trained
    python -m scripts.rebuttal.off_manifold_concept_stratification --arm random
"""
import argparse
import glob
import os
from collections import defaultdict

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import load_test_embeddings
from scripts.matching.utils import load_norm_stats as load_norm
from scripts.rebuttal.subspace_analysis import _eig_cov, _k_for_variance

ACT_NORM = {"carte": "carte", "mitra": "mitra", "tabdpt": "tabdpt",
            "tabicl": "tabicl", "tabicl_v2": "tabicl_v2", "tabpfn": "tabpfn"}


def get_E(recipient, dataset, emb_cache, norm_cache, var=0.90):
    if recipient not in emb_cache:
        emb_cache[recipient] = load_test_embeddings(recipient)
        norm_cache[recipient] = load_norm(recipient)
    Xn = np.asarray(emb_cache[recipient][dataset], dtype=np.float64)
    mean, std = norm_cache[recipient][dataset]
    Xraw = Xn * np.asarray(std) + np.asarray(mean)
    _, lam, V = _eig_cov(Xraw, center=True)
    ke = max(1, min(_k_for_variance(lam, var), V.shape[1]))
    return V[:, :ke]


def load_vatoms(cache_dir, donor, recipient):
    f = os.path.join(cache_dir, f"{donor}_to_{recipient}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    V = np.asarray(z["virtual_atoms"], dtype=np.float64)
    fids = np.asarray(z["feature_ids"])
    return V, {int(fid): i for i, fid in enumerate(fids)}


def firing_density(donor_models):
    """Per (donor, feat_id) fraction of the donor's rows the concept fires on (>0)."""
    base = PROJECT_ROOT / "output" / "concept_activations_cache"
    dens = {}
    for m in donor_models:
        fs = sorted(glob.glob(str(base / ACT_NORM[m] / "*.npz")))
        A = [d["activations"] for d in (np.load(f, allow_pickle=True) for f in fs)
             if "activations" in d.files]
        if not A:
            continue
        A = np.concatenate(A, axis=0)
        fire = (A > 0).mean(axis=0)
        for fid in range(A.shape[1]):
            dens[(m, fid)] = float(fire[fid])
    return dens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["trained", "random"], default="trained")
    ap.add_argument("--var-threshold", type=float, default=0.90)
    ap.add_argument("--dump", action="store_true",
                     help="write the locked patching cell (off in [--dump-off-lo, "
                          "--dump-off-hi), acceptance in [--dump-acc-lo, --dump-acc-hi]) "
                          "to --dump-out as CSV: donor,feat_id,off_frac,density,"
                          "universality,n_datasets,acceptance")
    ap.add_argument("--dump-off-lo", type=float, default=0.6)
    ap.add_argument("--dump-off-hi", type=float, default=0.8)
    ap.add_argument("--dump-acc-lo", type=float, default=200)
    ap.add_argument("--dump-acc-hi", type=float, default=499)
    ap.add_argument("--dump-out", default=None,
                     help="default: output/rebuttal/off_manifold_concept_dump_<arm>.csv")
    args = ap.parse_args()

    fwd = PROJECT_ROOT / "output" / "rebuttal" / (
        "forward_deltas_random" if args.arm == "random" else "forward_deltas")
    cache = PROJECT_ROOT / "output" / "transfer_caches" / (
        "global_random_randomSAE_p90" if args.arm == "random" else "global_trained")

    off_w = defaultdict(float)   # sum(off_frac * n_accepted)
    acc_w = defaultdict(float)   # sum(n_accepted)
    recips = defaultdict(set)
    dsets = defaultdict(set)
    emb_cache, norm_cache, vcache = {}, {}, {}

    for f in glob.glob(str(fwd / "*" / "*.npz")):
        try:
            z = np.load(f, allow_pickle=True)
        except Exception:
            continue
        if "feature_acceptance" not in z.files:
            continue
        donor = str(z["strong_model"]); recipient = str(z["weak_model"])
        dataset = os.path.basename(f)[:-4]
        key = (donor, recipient)
        if key not in vcache:
            vcache[key] = load_vatoms(str(cache), donor, recipient)
        if vcache[key] is None:
            continue
        V, fmap = vcache[key]
        try:
            E = get_E(recipient, dataset, emb_cache, norm_cache, args.var_threshold)
        except Exception:
            continue
        for fid, _nt, na in np.asarray(z["feature_acceptance"]):
            fid = int(fid); na = int(na)
            if na <= 0 or fid not in fmap:
                continue
            v = V[fmap[fid]]
            nv = float(v @ v)
            if nv <= 1e-12:
                continue
            d_on = E @ (E.T @ v)
            off = 1.0 - float(d_on @ d_on) / nv
            off_w[(donor, fid)] += off * na
            acc_w[(donor, fid)] += na
            recips[(donor, fid)].add(recipient)
            dsets[(donor, fid)].add(dataset)

    concepts = list(acc_w)
    dens = firing_density(set(c[0] for c in concepts))

    rows = []
    for c in concepts:
        rows.append((
            c[0], c[1],
            off_w[c] / acc_w[c],          # acceptance-weighted off-manifold fraction
            int(acc_w[c]),                 # total acceptance
            len(recips[c]),                # universality (# distinct recipients)
            dens.get(c, np.nan),           # firing density (labelability)
            len(dsets[c]),                  # # distinct donor datasets
        ))
    off = np.array([r[2] for r in rows])
    den = np.array([r[5] for r in rows])
    acc = np.array([r[3] for r in rows])
    ok = ~np.isnan(den)

    print(f"\n{args.arm} arm: {len(rows)} accepted concepts, off-manifold fraction of the "
          f"mapped direction (acc-weighted) vs firing density\n")
    print("off-manifold band     n   med-density   med-acc   med-univ   %low-density(<30%)")
    for lo, hi in [(0.0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
        m = ok & (off >= lo) & (off < hi)
        if not m.sum():
            print(f"  [{lo:.1f},{hi:.1f})          0")
            continue
        d = den[m]
        print(f"  [{lo:.1f},{hi:.1f})   {m.sum():5d}   {np.median(d):9.2f}   "
              f"{int(np.median(acc[m])):6d}   {int(np.median([rows[i][4] for i in np.where(m)[0]])):6d}     "
              f"{100*np.mean(d < 0.30):5.0f}%")
    r = np.corrcoef(off[ok], den[ok])[0, 1]
    print(f"\n  corr(off-manifold fraction, firing density) = {r:+.3f}")
    print(f"  (negative => off-manifold concepts fire less => easier to label)")
    # the labelable off-manifold population
    for omin, dmax in [(0.6, 0.30), (0.5, 0.30), (0.6, 0.50)]:
        m = ok & (off >= omin) & (den < dmax)
        print(f"  off>={omin:.1f} & density<{dmax:.2f}: {m.sum():4d} concepts "
              f"(univ>=5: {(m & (np.array([r[4] for r in rows])>=5)).sum()})")

    if args.dump:
        cell = [r for r in rows
                if args.dump_off_lo <= r[2] < args.dump_off_hi
                and args.dump_acc_lo <= r[3] <= args.dump_acc_hi]
        # per-concept share of total off-manifold "contribution" mass. off_w[c] = sum
        # over accepted (recipient, dataset) instances of (off_frac * n_accepted) -- an
        # acceptance-weighted off-manifold energy-fraction mass. The 20.5% headline is
        # this cell's share of that mass across ALL accepted concepts (not a true
        # energy decomposition -- that was dropped as confounded, see 2026-08-02 handoff).
        total_off_mass = sum(off_w.values())
        cell_off_mass = sum(off_w[(r[0], r[1])] for r in cell)
        out = args.dump_out or str(
            PROJECT_ROOT / "output" / "rebuttal" / f"off_manifold_concept_dump_{args.arm}.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            fh.write("donor,feat_id,off_frac,density,universality,n_datasets,acceptance,"
                      "off_mass,off_mass_share,recipients\n")
            for r in sorted(cell, key=lambda r: -off_w[(r[0], r[1])]):
                c = (r[0], r[1])
                fh.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[5]:.4f},{r[4]},{r[6]},{r[3]},"
                          f"{off_w[c]:.4f},{off_w[c] / total_off_mass:.6f},"
                          f"\"{';'.join(sorted(recips[c]))}\"\n")
        print(f"\n  --dump: {len(cell)} concepts in off in [{args.dump_off_lo},"
              f"{args.dump_off_hi}) x acceptance [{args.dump_acc_lo:.0f},"
              f"{args.dump_acc_hi:.0f}] -> {out}")
        print(f"  cell off-mass share of all accepted concepts' off-mass: "
              f"{cell_off_mass / total_off_mass:.1%}  "
              f"(cell_off_mass={cell_off_mass:.1f}, total_off_mass={total_off_mass:.1f})")


if __name__ == "__main__":
    main()
