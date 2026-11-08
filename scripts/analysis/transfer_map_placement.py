#!/usr/bin/env python3
"""Do the learned and shuffled ridge maps PLACE each concept in the same spot?

The transfer maps agree in per-row gap_closed (transfer_map_agreement.py). This
asks WHY: for each unmatched strong concept d_c, where does each map put it in
the weak embedding space? We compare the mapped directions

    v1 = normalize(M1 @ atom(d_c))   # learned ridge (matched landmarks)
    v2 = normalize(M2 @ atom(d_c))   # same ridge pipeline, landmark TARGETS shuffled

by cosine similarity, per concept, pooled over pairs.

  cos(v1, v2) ~ 1  -> shuffling the landmark pairing barely moves the concept, so
    the pairing never determined placement -- the ridge's per-dimension scaling
    did -- and the gc agreement is fully explained.
  cos(v1, v2) low (and gc still agrees) -> the effect is not about placement.

Maps are per-PAIR (fit on decoder atoms + matched landmarks), so no embeddings or
model forwards are needed -- Mac-only. (M3/procrustes placement is per-dataset and
needs the training embeddings; a separate pass.)

Usage:
    python -m scripts.analysis.transfer_map_placement
"""
import argparse
from pathlib import Path

import numpy as np

from scripts.round_paths import FORWARD_ABLATION_DIR
from scripts.intervention.intervene_lib import load_sae
from scripts.intervention.ablation_sweep import DEFAULT_MATCHING_FILE
from scripts.intervention.transfer_decomposition import _ridge_map, virtual_atoms
from scripts.rebuttal.transfer_sweep_symmetric import get_matched_pairs, get_unmatched_features
from scripts.intervention.transfer_virtual_nodes import extract_decoder_atoms


def _pair_ctx(strong, weak):
    """Minimal ctx for the ridge maps: decoder atoms, matched/unmatched (no model)."""
    sae_s, _ = load_sae(strong, device="cpu"); sae_s.eval()
    sae_w, _ = load_sae(weak, device="cpu"); sae_w.eval()
    atoms_s = extract_decoder_atoms(sae_s).numpy()
    atoms_w = extract_decoder_atoms(sae_w).numpy()
    n_ls, n_lw = atoms_s.shape[0], atoms_w.shape[0]
    m_pairs = [(int(si), int(ti)) for si, ti in
               get_matched_pairs(strong, weak, matching_file=DEFAULT_MATCHING_FILE)
               if int(si) < n_ls and int(ti) < n_lw]
    unmatched = [int(c) for c in
                 get_unmatched_features(strong, weak, matching_file=DEFAULT_MATCHING_FILE)
                 if int(c) < n_ls]
    return dict(atoms_s=atoms_s, atoms_w=atoms_w, matched_pairs=m_pairs, unmatched=unmatched)


def main():
    p = argparse.ArgumentParser(description="Placement agreement of ridge vs shuffled maps")
    p.add_argument("--pairs", nargs="+", default=None)
    args = p.parse_args()

    pair_dirs = sorted(d.name for d in (Path(str(FORWARD_ABLATION_DIR))).iterdir() if d.is_dir())
    if args.pairs:
        pair_dirs = [p for p in pair_dirs if p in args.pairs]

    all_cos = []
    print(f"{'pair':26} {'strong->weak':22} {'r2(M1)':>7} {'n_concepts':>10} "
          f"{'cos(M1,M2) median':>18} {'p10':>6} {'frac>0.9':>9}")
    print("-" * 100)
    for pair in pair_dirs:
        # orientation from the on-disk transfer (strong=metric winner)
        import glob
        fs = sorted(glob.glob(str(FORWARD_ABLATION_DIR / pair / "*.npz")))
        if not fs:
            continue
        z = np.load(fs[0], allow_pickle=True)
        strong, weak = str(z["strong_model"]), str(z["weak_model"])
        try:
            ctx = _pair_ctx(strong, weak)
        except Exception as e:
            print(f"{pair:26} SKIP ({str(e)[:40]})"); continue
        M1, info1 = _ridge_map(ctx, shuffle=False)
        M2, _ = _ridge_map(ctx, shuffle=True)
        v1 = virtual_atoms(ctx, M1)
        v2 = virtual_atoms(ctx, M2)
        fis = [fi for fi in v1 if fi in v2]
        cos = np.array([float(np.dot(v1[fi], v2[fi])) for fi in fis])  # unit vectors
        all_cos.append(cos)
        print(f"{pair:26} {strong+'->'+weak:22} {info1['r2']:>7.2f} {len(fis):>10} "
              f"{np.median(cos):>18.3f} {np.percentile(cos,10):>6.2f} {(cos>0.9).mean():>9.2f}")

    if all_cos:
        c = np.concatenate(all_cos)
        print("-" * 100)
        print(f"POOLED {len(c)} concepts: cos(M1,M2) median={np.median(c):.3f}  "
              f"mean={c.mean():.3f}  p10={np.percentile(c,10):.2f}  "
              f"frac>0.9={(c>0.9).mean():.2f}  frac>0.99={(c>0.99).mean():.2f}")


if __name__ == "__main__":
    main()
