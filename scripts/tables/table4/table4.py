#!/usr/bin/env python3
"""
Table 4: Mean gap closed when ablating unmatched concepts, by strong model.

Reads all ablation sweep NPZ files, groups by which model is "strong"
(the one being ablated), and reports mean/median gap_closed.

Usage:
    python -m scripts.tables.table4.table4
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT

SWEEP_DIR = PROJECT_ROOT / "output" / "ablation_sweep"
OUTPUT_TEX = Path(__file__).parent / "ablation_summary.tex"

# Display name mapping
DISPLAY = {
    "tabpfn": "TabPFN", "mitra": "Mitra", "tabicl": "TabICL",
    "tabicl_v2": "TabICL-v2", "tabdpt": "TabDPT", "carte": "CARTE",
}

# Exclude from main table
EXCLUDE = {"hyperfast", "tabula8b"}


def load_ablation_results():
    """Load all ablation NPZ files, return list of (strong_model, dataset, gc) tuples."""
    results = []
    for pair_dir in sorted(SWEEP_DIR.iterdir()):
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

            # Determine strong model
            strong = str(data["strong_model"]) if "strong_model" in data else None
            if strong is None:
                continue

            gc = float(data["mean_gap_closed"]) if "mean_gap_closed" in data else None
            if gc is None:
                continue

            # Skip degenerate cases (no strong wins)
            n_strong = int(data["n_strong_wins"]) if "n_strong_wins" in data else 0
            if n_strong == 0:
                continue

            mean_k = float(data["mean_optimal_k"]) if "mean_optimal_k" in data else None

            dataset = npz_path.stem
            results.append((strong, dataset, gc, mean_k))

    return results


def main():
    results = load_ablation_results()
    print(f"Loaded {len(results)} (model, dataset, gc) entries")

    # Group by strong model
    by_model_gc = defaultdict(list)
    by_model_k = defaultdict(list)
    for strong, dataset, gc, mean_k in results:
        by_model_gc[strong].append(gc)
        if mean_k is not None:
            by_model_k[strong].append(mean_k)

    # Sort by mean gc descending
    model_stats = []
    for model, gcs in by_model_gc.items():
        display = DISPLAY.get(model, model)
        ks = by_model_k.get(model, [])
        model_stats.append({
            "key": model,
            "display": display,
            "n": len(gcs),
            "mean_gc": np.mean(gcs),
            "std_gc": np.std(gcs),
            "mean_k": np.mean(ks) if ks else 0,
            "std_k": np.std(ks) if ks else 0,
        })
    model_stats.sort(key=lambda x: -x["mean_gc"])

    # Print summary
    print(f"\n{'Model':<15s} {'N':>4s} {'Mean gc':>12s} {'Mean K':>12s}")
    print("-" * 48)
    for s in model_stats:
        print(f"{s['display']:<15s} {s['n']:>4d} "
              f"{s['mean_gc']:.3f}±{s['std_gc']:.3f} "
              f"{s['mean_k']:.1f}±{s['std_k']:.1f}")
    all_gcs = [gc for _, _, gc, _ in results]
    all_ks = [k for _, _, _, k in results if k is not None]
    print("-" * 48)
    print(f"{'Overall':<15s} {len(all_gcs):>4d} "
          f"{np.mean(all_gcs):.3f}±{np.std(all_gcs):.3f} "
          f"{np.mean(all_ks):.1f}±{np.std(all_ks):.1f}")

    # Generate LaTeX
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Mean gap closed when ablating unmatched concepts, sorted by "
        r"explanatory power. $N$ is the number of datasets where the model is "
        r"strong. $K$ is the mean number of concepts ablated to close the gap.}"
    )
    lines.append(r"\label{tab:ablation_summary}")
    lines.append(r"\begin{tabular}{lrll}")
    lines.append(r"\toprule")
    lines.append(r"Model (when strong) & $N$ & Mean gc & Mean $K$ \\")
    lines.append(r"\midrule")

    for s in model_stats:
        lines.append(
            f"{s['display']} & {s['n']} & "
            f"{s['mean_gc']:.2f} $\\pm$ {s['std_gc']:.2f} & "
            f"{s['mean_k']:.1f} $\\pm$ {s['std_k']:.1f} \\\\"
        )

    lines.append(r"\midrule")
    lines.append(
        f"Overall & {len(all_gcs)} & "
        f"{np.mean(all_gcs):.2f} $\\pm$ {np.std(all_gcs):.2f} & "
        f"{np.mean(all_ks):.1f} $\\pm$ {np.std(all_ks):.1f} \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    OUTPUT_TEX.write_text(tex + "\n")
    print(f"\nSaved to {OUTPUT_TEX}")
    print(tex)


if __name__ == "__main__":
    main()
