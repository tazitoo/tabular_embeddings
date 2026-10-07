#!/usr/bin/env python3
"""Three-rung forward-ablation comparison: trained vs archetypal-random vs random-topk.

Reads mean_gap_closed (gc) per (pair, dataset) from the three forward-ablation arms
and prints them side by side. The hypothesis: the archetypal data scaffold is what
makes the random baseline close the gap, so

    random-topk gc  <  archetypal-random gc  <  trained gc        (roughly)

gc is the fraction of the strong-weak performance gap closed by ablating the strong
model's firing, UNMATCHED concepts. A low random-topk gc means a data-free dictionary's
concepts don't carry the signal -- the archetypal arm's strength was the data geometry.

Usage:
    python -m scripts.analysis.compare_ablation_arms
    python -m scripts.analysis.compare_ablation_arms --pairs carte_vs_tabdpt mitra_vs_tabpfn \
        --datasets blood-transfusion-service-center diabetes GiveMeSomeCredit churn
"""
import argparse

import numpy as np

from scripts.round_paths import RESULTS_DIR

ARMS = {
    "trained": RESULTS_DIR / "forward_ablation",
    "arch-rand": RESULTS_DIR / "forward_ablation_random",
    "rtopk": RESULTS_DIR / "forward_ablation_randomtopk",
}
PILOT_PAIRS = ["carte_vs_tabdpt", "mitra_vs_tabpfn"]
PILOT_DATASETS = ["blood-transfusion-service-center", "diabetes",
                  "GiveMeSomeCredit", "churn"]


def _gc(arm_dir, pair, dataset):
    f = arm_dir / pair / f"{dataset}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return float(d["mean_gap_closed"]), float(d["mean_optimal_k"]), str(d["metric_name"])


def _discover_pairs():
    base = ARMS["trained"]
    return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []


def _discover_datasets(pair):
    """Datasets whose npz exist in BOTH always-present arms (trained, arch-rand)."""
    core = ["trained", "arch-rand"]
    sets = []
    for a in core:
        d = ARMS[a] / pair
        sets.append({f.stem for f in d.glob("*.npz")} if d.is_dir() else set())
    return sorted(set.intersection(*sets)) if all(sets) else []


def _eff(gc, k):
    return gc / k if k and k > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="default: all pairs present in the trained arm")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="default: per-pair, datasets common to trained & arch-rand")
    args = ap.parse_args()

    pairs = args.pairs or _discover_pairs()
    arms = list(ARMS)
    col = {a: {"gc": [], "k": [], "eff": []} for a in arms}
    detail = len(pairs) <= 2  # per-cell dump only for a small (pilot) view

    print(f"{'pair':<22} {'n':>3}  " + "  ".join(f"{a+' med gc/K':>14}" for a in arms))
    print("-" * 74)
    for pair in pairs:
        datasets = args.datasets or _discover_datasets(pair)
        per = {a: {"gc": [], "k": [], "eff": []} for a in arms}
        for ds in datasets:
            for a in arms:
                r = _gc(ARMS[a], pair, ds)
                if r is None:
                    continue
                gc, k, _ = r
                e = _eff(gc, k)
                per[a]["gc"].append(gc); per[a]["k"].append(k); per[a]["eff"].append(e)
                col[a]["gc"].append(gc); col[a]["k"].append(k); col[a]["eff"].append(e)
        n = max(len(per[a]["gc"]) for a in arms)
        cells = "  ".join(
            f"{np.nanmedian(per[a]['eff']):>14.3f}" if per[a]["eff"] else f"{'--':>14}"
            for a in arms)
        print(f"{pair:<22} {n:>3}  {cells}")

    print("\n=== pooled over all cells (mean | median) ===")
    print(f"  {'arm':<12} {'gc':>15} {'K':>15} {'gc/K':>17}   "
          f"{'gc/K ratio-of-means':>20} {'n':>5}")
    for a in arms:
        g, k, e = col[a]["gc"], col[a]["k"], col[a]["eff"]
        if not g:
            print(f"  {a:<12} (no data)")
            continue
        rom = np.mean(g) / np.mean(k) if np.mean(k) else float("nan")
        print(f"  {a:<12} "
              f"{np.mean(g):>7.3f}|{np.median(g):<7.3f} "
              f"{np.mean(k):>7.2f}|{np.median(k):<7.2f} "
              f"{np.nanmean(e):>8.3f}|{np.nanmedian(e):<8.3f} "
              f"{rom:>20.3f} {len(g):>5}")
    tr = col["trained"]; ar = col["arch-rand"]
    if tr["gc"] and ar["gc"]:
        med = np.nanmedian(tr["eff"]) / np.nanmedian(ar["eff"])
        rom = (np.mean(tr["gc"]) / np.mean(tr["k"])) / (np.mean(ar["gc"]) / np.mean(ar["k"]))
        print(f"\n  trained vs arch-rand efficiency ratio: "
              f"{med:.1f}x (median gc/K), {rom:.1f}x (ratio-of-means)")
    print("  (ratio-of-means = mean_gc / mean_K; robust to tiny-K cells that inflate per-cell gc/K)")


if __name__ == "__main__":
    main()
