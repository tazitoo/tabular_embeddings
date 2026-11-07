#!/usr/bin/env python3
"""How many UNMATCHED concepts fire on each active row? (the sufficient pool size)

Firing = the strong SAE latent is active (>0) on that row. Because the unmatched
set is fixed per pair, the unmatched concepts that fire on a row are the candidate
("sufficient") pool the per-row greedy then down-selects to its accepted ("necessary")
set of size k (see perrow_selection_overlap.py). This script reports, for one
pair/dataset, the strong SAE top-k sparsity and the per-row distribution of the
number of unmatched concepts firing on the active rows.

Mac-only: SAE encode of cached test embeddings; no model forward pass.

Usage:
    python -m scripts.intervention.perrow_firing_counts --models tabpfn tabicl --datasets credit-g
"""
import argparse
from collections import Counter

import numpy as np
import torch

from scripts.round_paths import FORWARD_ABLATION_DIR
from scripts.intervention.intervene_lib import load_sae, load_test_embeddings
from scripts.intervention.ablation_sweep import get_unmatched_features, DEFAULT_MATCHING_FILE


def run(model_a, model_b, dataset, matching_file, device):
    pair = f"{model_a}_vs_{model_b}"
    abl = np.load(FORWARD_ABLATION_DIR / pair / f"{dataset}.npz", allow_pickle=True)
    strong, weak = str(abl["strong_model"]), str(abl["weak_model"])
    k = abl["optimal_k"].astype(int)
    active = k > 0

    sae, _ = load_sae(strong, device=device); sae.eval()
    topk = getattr(getattr(sae, "config", sae), "topk", None)
    n_latents = getattr(getattr(sae, "config", sae), "hidden_dim", None)

    emb = load_test_embeddings(strong)[dataset]
    with torch.no_grad():
        acts = sae.encode(torch.tensor(emb, dtype=torch.float32, device=device))
    firing = (acts > 0).cpu().numpy()                    # (n_rows, n_latents)
    n_rows = firing.shape[0]

    unmatched = sorted(int(c) for c in get_unmatched_features(strong, weak, matching_file=matching_file))
    um = np.array(unmatched, dtype=int)

    l0_all = firing.sum(axis=1)                          # total active latents per row (~topk)
    fire_um = firing[:, um].sum(axis=1)                  # unmatched concepts firing per row

    print(f"=== {pair} / {dataset} ===")
    print(f"strong={strong}  weak={weak}  SAE top-k (sparsity)={topk}  n_latents={n_latents}")
    print(f"unmatched set size={len(unmatched)}  (of {n_latents} latents)")
    print(f"rows: n_test={n_rows}  active(optimal_k>0)={int(active.sum())}")
    print(f"total firing latents/row (L0): mean={l0_all.mean():.1f} "
          f"[{l0_all.min()}, {l0_all.max()}]  (should track top-k)")
    print()
    fa = fire_um[active]
    print(f"# UNMATCHED concepts firing per ACTIVE row:")
    print(f"  mean={fa.mean():.1f}  median={np.median(fa):.0f}  min={fa.min()}  max={fa.max()}")
    print(f"  distribution: {dict(sorted(Counter(fa.tolist()).items()))}")
    print()
    print(f"for contrast, accepted k per active row: "
          f"mean={k[active].mean():.2f} median={np.median(k[active]):.0f} "
          f"dist={dict(sorted(Counter(k[active].tolist()).items()))}")


def main():
    p = argparse.ArgumentParser(description="Per-row unmatched-concept firing counts")
    p.add_argument("--models", nargs=2, required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--matching-file", default=DEFAULT_MATCHING_FILE)
    args = p.parse_args()
    model_a, model_b = sorted(args.models)
    for ds in args.datasets:
        run(model_a, model_b, ds, args.matching_file, args.device)


if __name__ == "__main__":
    main()
