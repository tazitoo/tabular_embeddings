#!/usr/bin/env python3
"""Compare cross-model matching noise floors across the random-SAE rungs.

The matching noise floor is the distribution of each trained feature's max |r|
against a random SAE's features (trained-A vs random-B, all directed pairs). Two
random arms exist:

  - archetypal-random: random convex combos of the model's real K-means centroids
    (data-informed dictionary)  -> sae_cross_model_random_baseline.json
  - random-topk:       unit-norm random directions, no data geometry
                       -> sae_cross_model_random_baseline_randomtopk.json

If the data geometry is what makes the random baseline look strong, the random-topk
floor should sit clearly BELOW the archetypal-random floor. This prints the per-pair
p90/mean for both and the aggregate drop.

Usage:
    python -m scripts.analysis.compare_random_floors
"""
import json

from scripts.round_paths import CROSS_MODEL_BASELINE_FILE


def _load(path):
    with open(path) as f:
        return json.load(f)["pairs"]


def main():
    arch_path = CROSS_MODEL_BASELINE_FILE
    rtopk_path = arch_path.with_name(arch_path.stem + "_randomtopk" + arch_path.suffix)
    arch = _load(arch_path)
    rtopk = _load(rtopk_path)

    pairs = sorted(set(arch) & set(rtopk))
    print(f"archetypal-random: {arch_path.name}")
    print(f"random-topk:       {rtopk_path.name}")
    print(f"shared pairs: {len(pairs)} (arch={len(arch)}, rtopk={len(rtopk)})\n")

    hdr = f"{'pair':<40} {'arch_p90':>9} {'rtopk_p90':>10} {'Δp90':>8}   {'arch_mean':>9} {'rtopk_mean':>10} {'Δmean':>8}"
    print(hdr)
    print("-" * len(hdr))
    dp90, dmean = [], []
    for k in pairs:
        a, r = arch[k], rtopk[k]
        d90 = r["p90"] - a["p90"]
        dm = r["mean"] - a["mean"]
        dp90.append(d90)
        dmean.append(dm)
        print(f"{k:<40} {a['p90']:>9.3f} {r['p90']:>10.3f} {d90:>8.3f}   "
              f"{a['mean']:>9.3f} {r['mean']:>10.3f} {dm:>8.3f}")

    n = len(pairs)
    print("-" * len(hdr))
    print(f"{'MEAN over pairs':<40} "
          f"{sum(arch[k]['p90'] for k in pairs)/n:>9.3f} "
          f"{sum(rtopk[k]['p90'] for k in pairs)/n:>10.3f} "
          f"{sum(dp90)/n:>8.3f}   "
          f"{sum(arch[k]['mean'] for k in pairs)/n:>9.3f} "
          f"{sum(rtopk[k]['mean'] for k in pairs)/n:>10.3f} "
          f"{sum(dmean)/n:>8.3f}")
    worse = sum(1 for d in dp90 if d > 0)
    print(f"\npairs where random-topk floor is LOWER (expected): {n - worse}/{n}")


if __name__ == "__main__":
    main()
