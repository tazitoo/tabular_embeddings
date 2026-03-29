#!/usr/bin/env python3
"""
Analyze pre-extracted layerwise CKA matrices to find optimal extraction layers.

Reads CKA matrices from output/layerwise_cka_v2/ (produced by extract_all_layers_batch.py),
computes per-dataset depth metrics (L0 profile, critical layer, drift), and saves
aggregated JSON files consumed by the plotting scripts.

Output:
    output/layerwise_depth_analysis_{model}.json  — per-dataset depth metrics
    Printed table of recommended optimal layers per model

Usage:
    python scripts/embeddings/analyze_layerwise_cka.py
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "figures" / "4_results"))

from layerwise_cka_analysis import compute_critical_depth  # noqa: E402

INPUT_DIR = PROJECT_ROOT / "output" / "layerwise_cka_v2"
OUTPUT_DIR = PROJECT_ROOT / "output"

MODELS = [
    "tabpfn",
    "mitra",
    "tabdpt",
    "carte",
    "tabicl",
    "tabicl_v2",
    "hyperfast",
    "tabula8b",
]


def analyze_model(model: str) -> dict:
    """Load all CKA matrices for a model and compute depth metrics."""
    # Exclude sub-variants (e.g. tabicl_v2 when model=tabicl)
    prefix = f"layerwise_cka_{model}_"
    files = sorted(f for f in INPUT_DIR.glob(f"{prefix}*.npz")
                   if not f.name.startswith(f"{prefix}v2_"))
    if not files:
        return {}

    results = {}
    for f in files:
        dataset = f.stem.replace(f"layerwise_cka_{model}_", "")
        data = np.load(f)
        cka_matrix = data["cka_matrix"]
        metrics = compute_critical_depth(cka_matrix)
        metrics["dataset"] = dataset
        metrics["model"] = model
        results[dataset] = metrics

    return results


def recommend_layer(results: dict, model: str) -> dict:
    """Recommend optimal extraction layer from aggregated depth metrics."""
    if not results:
        return {}

    n_layers_list = [r["n_layers"] for r in results.values()]
    n_layers = max(set(n_layers_list), key=n_layers_list.count)

    critical_depths = [r["critical_depth_frac"] for r in results.values()]
    mean_depth = np.mean(critical_depths)
    std_depth = np.std(critical_depths)

    # Round to nearest layer
    optimal_layer = int(round(mean_depth * (n_layers - 1)))
    final_ckas = [r["final_cka"] for r in results.values()]

    return {
        "model": model,
        "n_datasets": len(results),
        "n_layers": n_layers,
        "optimal_layer": optimal_layer,
        "mean_critical_depth": float(mean_depth),
        "std_critical_depth": float(std_depth),
        "mean_final_cka": float(np.mean(final_ckas)),
    }


def main():
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")

    summary = []

    for model in MODELS:
        results = analyze_model(model)
        if not results:
            print(f"{model}: no data found")
            continue

        rec = recommend_layer(results, model)

        # Save per-dataset JSON
        out_path = OUTPUT_DIR / f"layerwise_depth_analysis_{model}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        summary.append(rec)
        print(
            f"{model:<14} "
            f"{rec['n_datasets']:>3} datasets  "
            f"{rec['n_layers']:>3} layers  "
            f"optimal=L{rec['optimal_layer']:<3}  "
            f"depth={rec['mean_critical_depth']:.2f}±{rec['std_critical_depth']:.2f}  "
            f"final_cka={rec['mean_final_cka']:.3f}"
        )

    # Save summary
    summary_path = OUTPUT_DIR / "layerwise_optimal_layers_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
