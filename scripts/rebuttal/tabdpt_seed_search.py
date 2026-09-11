#!/usr/bin/env python3
"""Does TabDPT's corpus extraction path reproduce, and does the seed pin it?

TabDPT has two random draws that predict()-level seeding does not fix. Over its
100-feature cap it fits a randomized PCA (torch.pca_lowrank) that nothing seeds, and
regression predict() is an 8-member ensemble with OS-entropy member seeds and a
per-member feature permutation. Seeding predict() switches ON that permutation, which
is a different forward from the one the corpus used, so it cannot reproduce the corpus.

`load_and_fit(seed=...)` now pins the RNG before fit (PCA) and `predict` runs a single
unpermuted member on both task types. This script measures what that buys: each entry
in --seeds is the fit seed for one re-extraction (`none` = leave the RNG alone), and
every run is compared against the stored corpus activations and against each other.

Expected under the fixed path: repeated equal seeds are bit-identical; narrow
classification datasets (<=100 features, no random draw at all) match the corpus at
~0.998 regardless of seed; wide-classification and regression corpora are one arbitrary
historical draw and no seed matches them -- those 18 datasets need re-extraction.

Reports, per run, against the stored corpus activations:
  cosine, firing agreement, max|d|, and the per-row cosine distribution.
Plus the run-to-run spread.

Usage:
    python -m scripts.rebuttal.tabdpt_seed_search --dataset APSFailure --seeds 13 13 none
    python -m scripts.rebuttal.tabdpt_seed_search --dataset miami_housing --seeds 13 13 7
"""
import argparse
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import load_dataset_context
from scripts.rebuttal.patch_baseline_gate import _sae_acts, cached_acts


def acts_at_seed(model, dataset, device, n_rows, seed):
    from models.layer_extraction import extract_all_layers, load_and_fit, sort_layer_names
    from scripts.intervention.intervene_lib import get_extraction_layer_taskaware

    X_train, y_train, X_query, _, _, task = load_dataset_context(
        model, dataset, query_source="holdout")
    q = X_query[:n_rows] if not hasattr(X_query, "iloc") else X_query.iloc[:n_rows]
    clf = load_and_fit(model, X_train, y_train, task=task, device=device, seed=seed)
    embs = extract_all_layers(model, clf, q, task=task)
    names = sort_layer_names(list(embs.keys()))
    idx = min(max(get_extraction_layer_taskaware(model, dataset), 0), len(names) - 1)
    return _sae_acts(model, np.asarray(embs[names[idx]], dtype=np.float32), dataset, device)


def compare(a, ref):
    n = min(len(a), len(ref))
    a, ref = a[:n], ref[:n]
    per_row = np.array([
        float(a[i] @ ref[i] / (np.linalg.norm(a[i]) * np.linalg.norm(ref[i]) + 1e-12))
        for i in range(n)])
    return {
        "cosine": float(a.ravel() @ ref.ravel() /
                        (np.linalg.norm(a.ravel()) * np.linalg.norm(ref.ravel()) + 1e-12)),
        "firing_agreement": float(((ref > 0) == (a > 0)).mean()),
        "max_abs_diff": float(np.abs(a - ref).max()),
        "per_row_cos_median": float(np.median(per_row)),
        "per_row_cos_min": float(per_row.min()),
        "per_row_cos_max": float(per_row.max()),
        "rows_above_0.99": int((per_row > 0.99).sum()),
        "n_rows": int(n),
    }


def _seed(s):
    return None if s.lower() == "none" else int(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tabdpt")
    ap.add_argument("--dataset", default="miami_housing")
    ap.add_argument("--seeds", nargs="+", type=_seed, default=[0, 1, 2, 7, 13, 42, 123, 2024],
                    help="fit seeds, one re-extraction each; `none` leaves the RNG "
                         "unpinned. Repeat a value to measure reproducibility")
    ap.add_argument("--n-rows", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(
        PROJECT_ROOT / "output" / "rebuttal" / "tabdpt_seed_search.json"))
    args = ap.parse_args()

    torch.use_deterministic_algorithms(True)
    ref = cached_acts(args.model, args.dataset, args.device)[:args.n_rows]

    print(f"{args.model} / {args.dataset}: {len(args.seeds)} seeds vs the stored corpus\n")
    print(f"  {'seed':>6s} {'cosine':>8s} {'fire-agr':>9s} {'max|d|':>8s} "
          f"{'row-cos med':>11s} {'rows>0.99':>9s}")
    runs, results = {}, []
    for i, s in enumerate(args.seeds):
        a = acts_at_seed(args.model, args.dataset, args.device, args.n_rows, s)
        runs[i] = a  # keyed by position: repeated `none` draws must stay distinct
        m = compare(a, ref)
        m["seed"] = s
        results.append(m)
        print(f"  {str(s):>6s} {m['cosine']:8.4f} {m['firing_agreement']:9.1%} "
              f"{m['max_abs_diff']:8.3f} {m['per_row_cos_median']:11.4f} "
              f"{m['rows_above_0.99']:>6d}/{m['n_rows']}")

    best = max(results, key=lambda r: r["cosine"])
    print(f"\n  best seed = {best['seed']} at cosine {best['cosine']:.4f}")

    # seed-to-seed spread: what a seed search is competing against
    seeds = list(runs)
    pair_cos = [float(runs[x].ravel() @ runs[y].ravel() /
                      (np.linalg.norm(runs[x].ravel()) * np.linalg.norm(runs[y].ravel()) + 1e-12))
                for i, x in enumerate(seeds) for y in seeds[i + 1:]]
    print(f"  seed-to-seed cosine: median={np.median(pair_cos):.4f} "
          f"min={min(pair_cos):.4f} max={max(pair_cos):.4f}")
    print(f"  vs-corpus cosine:    median={np.median([r['cosine'] for r in results]):.4f} "
          f"min={min(r['cosine'] for r in results):.4f} "
          f"max={max(r['cosine'] for r in results):.4f}")
    print("\n  Equal seeds should be bit-identical. If no seed approaches the corpus, the")
    print("  corpus is a historical draw this dataset cannot reproduce and must be re-extracted.")

    with open(args.out, "w") as fh:
        json.dump({"model": args.model, "dataset": args.dataset, "n_rows": args.n_rows,
                   "per_seed": results, "seed_to_seed_cos": pair_cos}, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
