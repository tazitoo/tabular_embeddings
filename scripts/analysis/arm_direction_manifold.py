#!/usr/bin/env python3
"""Where do each transfer ARM's injected directions land -- ON or OFF the weak
model's manifold? That decides whether a gap-closing direction is NOVEL to the
weak model (off-manifold -> latent capacity it lacks) or something it already
represents (on-manifold -> the concept is there, just underweighted).

Reuses the paper's canonical on/off-manifold measure (subspace_analysis /
off_manifold_concept_stratification.get_E): E is the top-k PCA subspace of the
recipient's RAW embeddings capturing 90% of the variance (per recipient x
dataset). For an injected direction v (in raw weak space),

    off = 1 - ||E^T v||^2 / ||v||^2          (energy outside the active subspace)

The arms (directions reconstructed from the SAEs + ridge map, no model forwards):
  unmatched   : unit(M_ridge @ atom_s[fi])          -- the main transfer arm
  matched_map : unit(M_ridge @ atom_s[si])          -- matched concepts, same map
  matched_wdec: atoms_w[ti]                          -- W's OWN atom (on-manifold
                by construction -- the control; "turn up a concept W already has")
Each normalized direction is brought to raw weak space by elementwise * std_w
(the injection is a_s * dir * std_w, so std_w sets the raw-space direction), then
projected onto E. --random swaps in the random-init SAE dictionary + its matching.

We report the off-manifold fraction distribution per arm, for ALL pooled
concepts and for the TOP-20 by donor importance (the directions the none-curves
actually inject at m<=20). Mac-only CPU. Usage:
    python -m scripts.analysis.arm_direction_manifold
    python -m scripts.analysis.arm_direction_manifold --random
"""
import argparse
from pathlib import Path

import numpy as np

from scripts.round_paths import (
    FORWARD_ABLATION_DIR, IMPORTANCE_DIR, IMPORTANCE_RANDOM_DIR,
    RESULTS_DIR, random_sae_dir,
)
from scripts.intervention.intervene_lib import (
    load_sae, DEFAULT_SAE_DIR, load_test_embeddings,
)
from scripts.matching.utils import load_norm_stats as load_norm
from scripts.intervention.ablation_sweep import DEFAULT_MATCHING_FILE
from scripts.rebuttal.transfer_sweep_symmetric import (
    get_matched_pairs, get_unmatched_features,
)
from scripts.intervention.transfer_virtual_nodes import extract_decoder_atoms
from scripts.intervention.transfer_decomposition import _ridge_map, virtual_atoms
from scripts.rebuttal.off_manifold_concept_stratification import get_E

RANDOM_MATCHING_FILE = RESULTS_DIR / "sae_feature_matching_mnn_t0.001_random.json"
ARMS = ["unmatched", "matched_map", "matched_wdec"]
TOP_N = 20


def _pair_ctx(strong, weak, matching_file, sae_dir):
    """Minimal ctx: decoder atoms + matched/unmatched (no model), SAE-range-filtered."""
    sae_s, _ = load_sae(strong, sae_dir=sae_dir, device="cpu"); sae_s.eval()
    sae_w, _ = load_sae(weak, sae_dir=sae_dir, device="cpu"); sae_w.eval()
    atoms_s = extract_decoder_atoms(sae_s).numpy()
    atoms_w = extract_decoder_atoms(sae_w).numpy()
    n_ls, n_lw = atoms_s.shape[0], atoms_w.shape[0]
    m_pairs = [(int(si), int(ti)) for si, ti in
               get_matched_pairs(strong, weak, matching_file=matching_file)
               if int(si) < n_ls and int(ti) < n_lw]
    unmatched = [int(c) for c in
                 get_unmatched_features(strong, weak, matching_file=matching_file)
                 if int(c) < n_ls]
    return dict(atoms_s=atoms_s, atoms_w=atoms_w, matched_pairs=m_pairs,
                unmatched=unmatched)


def _importance(strong, dataset, importance_dir):
    """Per strong-concept mean importance (|row_feature_drops| mean over rows)."""
    z = np.load(importance_dir / strong / f"{dataset}.npz", allow_pickle=True)
    drops = z["row_feature_drops"]                       # (rows, n_feat)
    fi = [int(x) for x in z["feature_indices"]]
    mean_drop = np.asarray(drops).mean(axis=0)
    return {fi[i]: float(mean_drop[i]) for i in range(len(fi))}


def _off(E, v_raw):
    nv = float(v_raw @ v_raw)
    if nv <= 1e-12:
        return None
    d_on = E @ (E.T @ v_raw)
    return 1.0 - float(d_on @ d_on) / nv


def _arm_dirs(ctx, arm):
    """{strong fi -> UNIT direction in NORMALIZED weak space} for one arm, plus the
    matched partner ti where relevant (for wdec we use W's own atom directly)."""
    if arm == "matched_wdec":
        return {int(si): ctx["atoms_w"][ti].astype(np.float64) for si, ti in ctx["matched_pairs"]}
    M, _ = _ridge_map(ctx, shuffle=False)
    pool = ([int(si) for si, _ in ctx["matched_pairs"]] if arm == "matched_map"
            else ctx["unmatched"])
    return {fi: v.astype(np.float64) for fi, v in virtual_atoms(ctx, M, pool=pool).items()}


def collect(random=False, var_threshold=0.90, pairs=None, verbose=False):
    """Per-arm off-manifold fractions of the injected directions. Returns
    (offs, offs_top, offs_recip, n_pairs): pooled-all, pooled top-20-by-importance,
    and per-recipient top-20 (recipient's own basis). Shared by main() and the
    manifold table generator."""
    import glob
    from collections import defaultdict
    sae_dir = random_sae_dir() if random else DEFAULT_SAE_DIR
    matching_file = RANDOM_MATCHING_FILE if random else DEFAULT_MATCHING_FILE
    importance_dir = IMPORTANCE_RANDOM_DIR if random else IMPORTANCE_DIR

    pair_dirs = sorted(d.name for d in Path(str(FORWARD_ABLATION_DIR)).iterdir() if d.is_dir())
    if pairs:
        pair_dirs = [d for d in pair_dirs if d in pairs]

    emb_cache, norm_cache = {}, {}
    offs = {a: [] for a in ARMS}
    offs_top = {a: [] for a in ARMS}
    offs_recip = defaultdict(lambda: {a: [] for a in ARMS})
    n_pairs = 0
    for pair in pair_dirs:
        fs = sorted(glob.glob(str(FORWARD_ABLATION_DIR / pair / "*.npz")))
        if not fs:
            continue
        z0 = np.load(fs[0], allow_pickle=True)
        strong, weak = str(z0["strong_model"]), str(z0["weak_model"])
        try:
            ctx = _pair_ctx(strong, weak, matching_file, sae_dir)
            ctx["median_norm_ratio"] = 1.0
            std_w_all = load_norm(weak)
        except Exception as e:
            if verbose:
                print(f"{pair:28} SKIP ({str(e)[:50]})")
            continue
        dirs = {a: _arm_dirs(ctx, a) for a in ARMS}
        datasets = sorted(Path(f).stem for f in fs)
        used = 0
        for ds in datasets:
            try:
                E = get_E(weak, ds, emb_cache, norm_cache, var_threshold)
                _, std_w = std_w_all[ds]
            except Exception:
                continue
            std_w = np.asarray(std_w, dtype=np.float64)
            imp = _importance(strong, ds, importance_dir)
            for a in ARMS:
                rows = []
                for fi, v in dirs[a].items():
                    o = _off(E, v * std_w)           # raw-space injection direction
                    if o is not None:
                        rows.append((imp.get(fi, 0.0), o))
                if not rows:
                    continue
                offs[a].extend(o for _, o in rows)
                rows.sort(key=lambda t: -t[0])       # by donor importance
                top = [o for _, o in rows[:TOP_N]]
                offs_top[a].extend(top)
                offs_recip[weak][a].extend(top)
            used += 1
        if used:
            n_pairs += 1
            if verbose:
                print(f"{pair:28} {strong}->{weak}  {used} ds  "
                      f"(|unmatched|={len(dirs['unmatched'])} |matched|={len(dirs['matched_wdec'])})")
    return offs, offs_top, offs_recip, n_pairs


def main():
    p = argparse.ArgumentParser(description="On/off-manifold of transfer-arm directions")
    p.add_argument("--random", action="store_true", help="random-init SAE dictionary")
    p.add_argument("--pairs", nargs="+", default=None)
    p.add_argument("--var-threshold", type=float, default=0.90)
    args = p.parse_args()

    offs, offs_top, offs_recip, n_pairs = collect(
        args.random, args.var_threshold, args.pairs, verbose=True)

    print(f"\n=== OFF-MANIFOLD FRACTION of injected directions ({'RANDOM' if args.random else 'TRAINED'} dict) ===")
    print(f"E = top-k PCA of recipient RAW embeddings @ {args.var_threshold:.0%} variance; "
          f"{n_pairs} pairs\n")
    hdr = f"{'arm':14} {'n_dir':>7} {'med':>6} {'p25':>6} {'p75':>6} {'frac>0.5':>9} {'frac>0.8':>9}"
    print("ALL pooled concepts")
    print(hdr)
    for a in ARMS:
        x = np.asarray(offs[a])
        if len(x):
            print(f"{a:14} {len(x):>7} {np.median(x):>6.2f} {np.percentile(x,25):>6.2f} "
                  f"{np.percentile(x,75):>6.2f} {(x>0.5).mean():>9.2f} {(x>0.8).mean():>9.2f}")
    print(f"\nTOP-{TOP_N} by donor importance (the directions injected at m<={TOP_N})")
    print(hdr)
    for a in ARMS:
        x = np.asarray(offs_top[a])
        if len(x):
            print(f"{a:14} {len(x):>7} {np.median(x):>6.2f} {np.percentile(x,25):>6.2f} "
                  f"{np.percentile(x,75):>6.2f} {(x>0.5).mean():>9.2f} {(x>0.8).mean():>9.2f}")

    print(f"\nTOP-{TOP_N} median off-manifold BY RECIPIENT (its own basis; not pooled)")
    print(f"{'recipient':12} {'n':>6}  " + "  ".join(f"{a:>12}" for a in ARMS))
    for recip in sorted(offs_recip, key=lambda r: -np.median(offs_recip[r]['unmatched'] or [0])):
        d = offs_recip[recip]
        n = len(d['unmatched'])
        print(f"{recip:12} {n:>6}  "
              + "  ".join(f"{np.median(d[a]):>12.2f}" if d[a] else f"{'--':>12}" for a in ARMS))


if __name__ == "__main__":
    main()
