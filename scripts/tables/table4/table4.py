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

            dataset = npz_path.stem
            results.append((strong, dataset, gc))

    return results


def main():
    results = load_ablation_results()
    print(f"Loaded {len(results)} (model, dataset, gc) entries")

    # Group by strong model
    by_model = defaultdict(list)
    for strong, dataset, gc in results:
        by_model[strong].append(gc)

    # Sort by mean gc descending
    model_stats = []
    for model, gcs in by_model.items():
        display = DISPLAY.get(model, model)
        model_stats.append({
            "key": model,
            "display": display,
            "n": len(gcs),
            "mean_gc": np.mean(gcs),
            "median_gc": np.median(gcs),
        })
    model_stats.sort(key=lambda x: -x["mean_gc"])

    # Print summary
    print(f"\n{'Model':<15s} {'N':>4s} {'Mean gc':>8s} {'Median gc':>10s}")
    print("-" * 40)
    for s in model_stats:
        print(f"{s['display']:<15s} {s['n']:>4d} {s['mean_gc']:>8.3f} {s['median_gc']:>10.3f}")
    all_gcs = [gc for _, _, gc in results]
    print("-" * 40)
    print(f"{'Overall':<15s} {len(all_gcs):>4d} {np.mean(all_gcs):>8.3f} {np.median(all_gcs):>10.3f}")

    # Generate LaTeX
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Mean gap closed when ablating unmatched concepts, sorted by "
        r"explanatory power. $N$ is the number of datasets where the model is strong.}"
    )
    lines.append(r"\label{tab:ablation_summary}")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model (when strong) & N datasets & Mean gc & Median gc \\")
    lines.append(r"\midrule")

    for s in model_stats:
        lines.append(
            f"{s['display']} & {s['n']} & {s['mean_gc']:.3f} & {s['median_gc']:.3f} \\\\"
        )

    lines.append(r"\midrule")
    lines.append(
        f"Overall & {len(all_gcs)} & {np.mean(all_gcs):.3f} & {np.median(all_gcs):.3f} \\\\"
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
