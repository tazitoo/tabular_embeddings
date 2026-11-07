#!/usr/bin/env python3
"""Does the importance-ranked candidate pool agree across rows, even when the
accepted (k~1) set does not?

The paper's per-row greedy (ablation_sweep.py, lines 297-309) ranks each active
row's firing UNMATCHED concepts by LOO importance (row_feature_drops > 0, the
"positive = helps the strong model" screen), then accepts from the top. The
accepted sets are near-disjoint across rows (perrow_selection_overlap.py: Jaccard
~0.04). This script asks the complementary question with the SAME screen: take the
top-m of each row's importance-ranked firing-unmatched pool and measure cross-row
agreement as m grows.

  overlap rises with m -> rows share a common important pool; the greedy just
    picks a different substitute from it (redundancy, not heterogeneity).
  overlap stays ~0 at every m -> the gap is genuinely per-row.

Common index space is the SAE latent id (confirmed in ablation_sweep: selected_
features, unmatched_set, firing columns, and feature_indices are all latent ids).
Mac-only: SAE encode of cached embeddings + cached importance; no model forward.

Usage:
    python -m scripts.intervention.perrow_ranked_pool_overlap --models tabpfn tabicl --datasets credit-g
"""
import argparse
from collections import Counter
from itertools import combinations

import numpy as np
import torch

from scripts.round_paths import FORWARD_ABLATION_DIR, IMPORTANCE_DIR
from scripts.intervention.intervene_lib import load_sae, load_test_embeddings
from scripts.intervention.ablation_sweep import get_unmatched_features, DEFAULT_MATCHING_FILE

DEPTHS = (1, 2, 3, 5, 10, 20)


def _ranked_pools(model_a, model_b, dataset, matching_file, device):
    """Per active row, the importance-ranked list of firing-unmatched latent ids
    (row_drops > 0, descending) -- exactly the greedy's candidate ranking."""
    pair = f"{model_a}_vs_{model_b}"
    abl = np.load(FORWARD_ABLATION_DIR / pair / f"{dataset}.npz", allow_pickle=True)
    strong, weak = str(abl["strong_model"]), str(abl["weak_model"])
    kacc = abl["optimal_k"].astype(int)
    active = kacc > 0

    imp = np.load(IMPORTANCE_DIR / strong / f"{dataset}.npz", allow_pickle=True)
    feat_idx = [int(fi) for fi in imp["feature_indices"]]       # positions -> latent ids
    drops = imp["row_feature_drops"]                            # (n_query, n_alive)

    sae, _ = load_sae(strong, device=device); sae.eval()
    emb = load_test_embeddings(strong)[dataset]
    with torch.no_grad():
        firing = (sae.encode(torch.tensor(emb, dtype=torch.float32, device=device)) > 0).cpu().numpy()

    unmatched = set(int(c) for c in get_unmatched_features(strong, weak, matching_file=matching_file))

    pools = []
    for r in np.where(active)[0]:
        scored = [(feat_idx[i], drops[r, i]) for i in range(len(feat_idx))
                  if drops[r, i] > 0 and firing[r, feat_idx[i]] and feat_idx[i] in unmatched]
        scored.sort(key=lambda x: -x[1])
        pools.append([fi for fi, _ in scored])
    return strong, weak, pools, kacc[active]


def _overlap(sets):
    n = len(sets)
    cc = Counter(c for s in sets for c in s)
    top_cov = max(cc.values()) / n if cc else float("nan")
    core = sum(1 for v in cc.values() if v / n >= 0.5)
    sets = [set(s) for s in sets if s]
    if len(sets) >= 2:
        jac = float(np.mean([len(a & b) / len(a | b) for a, b in combinations(sets, 2) if (a | b)]))
    else:
        jac = float("nan")
    return top_cov, core, jac, len(cc)


def run(model_a, model_b, dataset, matching_file, device):
    strong, weak, pools, kacc = _ranked_pools(model_a, model_b, dataset, matching_file, device)
    sizes = np.array([len(p) for p in pools])
    n = len(pools)
    print(f"=== {model_a}_vs_{model_b} / {dataset} ===")
    print(f"strong={strong} weak={weak}  active rows={n}")
    print(f"positive-importance firing-unmatched pool size per row: "
          f"mean={sizes.mean():.1f} median={np.median(sizes):.0f} min={sizes.min()} max={sizes.max()}")
    # Concepts ACTUALLY accepted to close each row's gap (what the paper uses).
    kc = Counter(int(k) for k in kacc)
    print(f"accepted k per row (concepts needed to close gc~0.99): "
          f"mean={kacc.mean():.2f} median={np.median(kacc):.0f} max={kacc.max()}  "
          f"dist={dict(sorted(kc.items()))}")
    cum = {t: int((kacc <= t).sum()) for t in (1, 2, 3, 5, 10, 20)}
    print(f"rows with accepted k<=t: " + "  ".join(f"<={t}:{c}/{n}" for t, c in cum.items()))
    print()
    # overlap-vs-depth: 'rows>=m' is how many rows' pools even reach depth m, i.e.
    # the sample count behind that row (truncation makes deep m describe a candidate
    # ranking, NOT the concepts rows actually use -- see accepted k above).
    print(f"{'depth m':>8} {'rows>=m':>8} {'topcov':>7} {'core@.5':>8} {'jaccard':>8} {'pool_used':>10}")
    print("-" * 54)
    for m in DEPTHS:
        rows_m = int((sizes >= m).sum())
        topm = [p[:m] for p in pools]
        tc, core, jac, used = _overlap(topm)
        print(f"{m:>8} {rows_m:>8} {tc:>7.2f} {core:>8} {jac:>8.2f} {used:>10}")
    tc, core, jac, used = _overlap(pools)
    print(f"{'all':>8} {n:>8} {tc:>7.2f} {core:>8} {jac:>8.2f} {used:>10}")


def main():
    p = argparse.ArgumentParser(description="Importance-ranked pool overlap vs screen depth")
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
