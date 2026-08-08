#!/usr/bin/env python3
"""
Section 4 ablation summary: mean gap closed when ablating unmatched
concepts, by strong model. Rendered as Table 2 in the paper draft.

Reads all ablation sweep NPZ files, groups by which model is "strong"
(the one being ablated), and reports mean/median gap_closed.

Usage:
    python -m scripts.tables.ablation_summary.ablation_summary
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.paper._paper_repo import paper_table_path

SWEEP_DIR = PROJECT_ROOT / "output" / "ablation_sweep_tols"
RANDOM_DIR = PROJECT_ROOT / "output" / "ablation_sweep_random_tols"
OUTPUT_TEX = Path(__file__).parent / "ablation_summary.tex"
PAPER_OUTPUT_TEX = paper_table_path("section4_summary.tex")

# Display name mapping
DISPLAY = {
    "tabpfn": "TabPFN", "mitra": "Mitra", "tabicl": "TabICL",
    "tabicl_v2": "TabICL-v2", "tabdpt": "TabDPT", "carte": "CARTE",
}

# Exclude from main table
EXCLUDE = {"hyperfast", "tabula8b"}


def load_ablation_results(sweep_dir):
    """Load all ablation NPZ files from a sweep directory.

    Returns list of (strong_model, pair_dir_name, dataset, gc, mean_k) tuples.
    """
    results = []
    for pair_dir in sorted(sweep_dir.iterdir()):
        if not pair_dir.is_dir():
            continue
        parts = pair_dir.name.split("_vs_")
        if len(parts) != 2:
            continue
        model_a, model_b = parts
        if model_a in EXCLUDE or model_b in EXCLUDE:
            continue

        for npz_path in sorted(pair_dir.glob("*.npz")):
            try:
                data = np.load(npz_path, allow_pickle=True)
            except Exception:
                continue

            strong = str(data["strong_model"]) if "strong_model" in data else None
            if strong is None:
                continue

            gc = float(data["mean_gap_closed"]) if "mean_gap_closed" in data else None
            if gc is None:
                continue

            n_strong = int(data["n_strong_wins"]) if "n_strong_wins" in data else 0
            if n_strong == 0:
                continue

            mean_k = float(data["mean_optimal_k"]) if "mean_optimal_k" in data else None

            dataset = npz_path.stem
            results.append((strong, pair_dir.name, dataset, gc, mean_k))

    return results


def main():
    trained = load_ablation_results(SWEEP_DIR)
    print(f"Trained: {len(trained)} entries")

    # Load random baseline and index by (pair, dataset)
    random_gc = {}
    random_k = {}
    if RANDOM_DIR.exists():
        random_results = load_ablation_results(RANDOM_DIR)
        print(f"Random:  {len(random_results)} entries")
        for strong, pair, dataset, gc, mean_k in random_results:
            random_gc[(pair, dataset)] = gc
            if mean_k is not None:
                random_k[(pair, dataset)] = mean_k
    else:
        print("WARNING: no random baseline directory, gc_R/K_R will be missing")

    # Group by strong model
    by_model_gc = defaultdict(list)
    by_model_k = defaultdict(list)
    by_model_random = defaultdict(list)
    by_model_random_k = defaultdict(list)
    for strong, pair, dataset, gc, mean_k in trained:
        by_model_gc[strong].append(gc)
        if mean_k is not None:
            by_model_k[strong].append(mean_k)
        rgc = random_gc.get((pair, dataset))
        if rgc is not None:
            by_model_random[strong].append(rgc)
        rk = random_k.get((pair, dataset))
        if rk is not None:
            by_model_random_k[strong].append(rk)

    # Sort by N descending
    model_stats = []
    for model, gcs in by_model_gc.items():
        display = DISPLAY.get(model, model)
        ks = by_model_k.get(model, [])
        randoms = by_model_random.get(model, [])
        random_ks = by_model_random_k.get(model, [])
        model_stats.append({
            "key": model,
            "display": display,
            "n": len(gcs),
            "mean_gc": np.mean(gcs),
            "std_gc": np.std(gcs),
            "mean_k": np.mean(ks) if ks else 0,
            "std_k": np.std(ks) if ks else 0,
            "mean_gc_r": np.mean(randoms) if randoms else None,
            "std_gc_r": np.std(randoms) if randoms else None,
            "n_random": len(randoms),
            "mean_k_r": np.mean(random_ks) if random_ks else None,
            "std_k_r": np.std(random_ks) if random_ks else None,
        })
    model_stats.sort(key=lambda x: -x["n"])

    # Print summary
    print(f"\n{'Model':<15s} {'N':>4s} {'gc':>12s} {'gc_R':>12s} {'K':>12s} {'K_R':>12s}")
    print("-" * 75)
    for s in model_stats:
        gc_r_str = (f"{s['mean_gc_r']:.3f}±{s['std_gc_r']:.3f}"
                    if s["mean_gc_r"] is not None else "---")
        k_r_str = (f"{s['mean_k_r']:.1f}±{s['std_k_r']:.1f}"
                   if s["mean_k_r"] is not None else "---")
        print(f"{s['display']:<15s} {s['n']:>4d} "
              f"{s['mean_gc']:.3f}±{s['std_gc']:.3f} "
              f"{gc_r_str:>12s} "
              f"{s['mean_k']:.1f}±{s['std_k']:.1f}".ljust(60) +
              f"{k_r_str:>12s}")
    all_gcs = [gc for _, _, _, gc, _ in trained]
    all_ks = [k for _, _, _, _, k in trained if k is not None]
    all_randoms = [random_gc[(pair, ds)] for _, pair, ds, _, _ in trained
                   if (pair, ds) in random_gc]
    all_random_ks = [random_k[(pair, ds)] for _, pair, ds, _, _ in trained
                     if (pair, ds) in random_k]
    print("-" * 75)
    print(f"{'Overall':<15s} {len(all_gcs):>4d} "
          f"{np.mean(all_gcs):.3f}±{np.std(all_gcs):.3f} "
          f"{np.mean(all_randoms):.3f}±{np.std(all_randoms):.3f} "
          f"{np.mean(all_ks):.1f}±{np.std(all_ks):.1f}".ljust(60) +
          f"{np.mean(all_random_ks):.1f}±{np.std(all_random_ks):.1f}".rjust(12))

    # Generate LaTeX
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Ablation results by strong model, sorted by $N$. "
        r"$gc/gc_R$ = mean gap closed using trained/random SAEs. "
        r"$K/K_R$ = mean concepts ablated using trained/random SAEs.}"
    )
    lines.append(r"\label{tab:ablation_summary}")
    lines.append(r"\begin{tabular}{lrllll}")
    lines.append(r"\toprule")
    lines.append(r"Model (when strong) & $N$ & gc & $gc_R$ & $K$ & $K_R$ \\")
    lines.append(r"\midrule")

    for s in model_stats:
        gc_r_str = (f"{s['mean_gc_r']:.2f} $\\pm$ {s['std_gc_r']:.2f}"
                    if s["mean_gc_r"] is not None else "---")
        k_r_str = (f"{s['mean_k_r']:.1f} $\\pm$ {s['std_k_r']:.1f}"
                   if s["mean_k_r"] is not None else "---")
        lines.append(
            f"{s['display']} & {s['n']} & "
            f"{s['mean_gc']:.2f} $\\pm$ {s['std_gc']:.2f} & "
            f"{gc_r_str} & "
            f"{s['mean_k']:.1f} $\\pm$ {s['std_k']:.1f} & "
            f"{k_r_str} \\\\"
        )

    lines.append(r"\midrule")
    gc_r_overall = (f"{np.mean(all_randoms):.2f} $\\pm$ {np.std(all_randoms):.2f}"
                    if all_randoms else "---")
    k_r_overall = (f"{np.mean(all_random_ks):.1f} $\\pm$ {np.std(all_random_ks):.1f}"
                   if all_random_ks else "---")
    lines.append(
        f"Overall & {len(all_gcs)} & "
        f"{np.mean(all_gcs):.2f} $\\pm$ {np.std(all_gcs):.2f} & "
        f"{gc_r_overall} & "
        f"{np.mean(all_ks):.1f} $\\pm$ {np.std(all_ks):.1f} & "
        f"{k_r_overall} \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    OUTPUT_TEX.write_text(tex + "\n")
    print(f"\nSaved to {OUTPUT_TEX}")
    PAPER_OUTPUT_TEX.write_text(tex + "\n")
    print(f"  → also wrote {PAPER_OUTPUT_TEX}")
    print(tex)


if __name__ == "__main__":
    main()
