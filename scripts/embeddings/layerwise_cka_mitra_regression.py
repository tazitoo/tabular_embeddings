#!/usr/bin/env python3
"""
Layer-wise CKA depth analysis for Mitra REGRESSOR checkpoint.

The original layerwise_cka_analysis.py batch_analyze() did not pass task= to
extract_mitra_all_layers(), so all 51 TabArena datasets (including 13 regression)
were analyzed using the MitraClassifier checkpoint.  This script re-runs the
depth analysis on regression datasets using MitraRegressor.

Compare the CKA profiles (especially critical depth) against the classifier
results to see whether the regressor checkpoint has different geometry.

Usage:
    python scripts/layerwise_cka_mitra_regression.py --device cuda
    python scripts/layerwise_cka_mitra_regression.py --device cuda --datasets diamonds houses
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT

from data.extended_loader import TABARENA_DATASETS
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "figures" / "4_results"))
from layerwise_cka_analysis import (
    extract_mitra_all_layers,
    compute_layerwise_cka,
    compute_critical_depth,
    sort_layer_names,
    load_dataset,
)


def get_regression_datasets() -> list[str]:
    """Return sorted list of TabArena regression dataset names."""
    return sorted(
        k for k, v in TABARENA_DATASETS.items()
        if v.get("task") == "regression"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Mitra regressor layerwise CKA depth analysis"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets (default: all 13 regression)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    datasets = args.datasets or get_regression_datasets()
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mitra REGRESSOR layerwise CKA analysis")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Device: {args.device}")
    print()

    results = {}

    for i, dataset_name in enumerate(datasets):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(datasets)}] {dataset_name}")
        print("=" * 60)

        t0 = time.time()
        try:
            X_context, y_context, X_query, _ = load_dataset(
                dataset_name, max_samples=args.n_samples
            )

            # Key difference: task="regression" → loads MitraRegressor checkpoint
            layer_embeddings = extract_mitra_all_layers(
                X_context, y_context, X_query,
                device=args.device,
                task="regression",
            )

            if not layer_embeddings:
                print(f"  No embeddings extracted")
                continue

            # Print per-layer stats
            for name in sort_layer_names(list(layer_embeddings.keys())):
                e = layer_embeddings[name]
                print(f"  {name:12s}: shape={e.shape}  "
                      f"mean={e.mean():.3f}  std={e.std():.3f}")

            # Compute CKA
            cka_matrix, layer_names = compute_layerwise_cka(layer_embeddings)
            depth_metrics = compute_critical_depth(cka_matrix)
            depth_metrics["dataset"] = dataset_name
            depth_metrics["model"] = "mitra_regressor"
            depth_metrics["layer_names"] = layer_names

            results[dataset_name] = depth_metrics

            # Save per-dataset CKA
            suffix = f"mitra_regressor_{dataset_name}"
            np.savez(
                output_dir / f"layerwise_cka_{suffix}.npz",
                cka_matrix=cka_matrix,
                layer_names=layer_names,
                dataset=dataset_name,
            )

            dt = time.time() - t0
            print(f"  Critical layer: L{depth_metrics['critical_layer']} "
                  f"({depth_metrics['critical_depth_frac']:.1%})")
            print(f"  Final CKA with L0: {depth_metrics['final_cka']:.3f}")
            print(f"  Time: {dt:.1f}s")

        except Exception as e:
            dt = time.time() - t0
            print(f"  ERROR ({dt:.1f}s): {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("\nNo results!")
        return

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Mitra Regressor Depth Analysis")
    print("=" * 60)

    critical_layers = []
    critical_fracs = []
    for ds, r in sorted(results.items()):
        cl = r["critical_layer"]
        cf = r["critical_depth_frac"]
        fc = r["final_cka"]
        critical_layers.append(cl)
        critical_fracs.append(cf)
        print(f"  {ds:40s}  critical=L{cl:2d} ({cf:.1%})  final_cka={fc:.3f}")

    mean_cl = np.mean(critical_layers)
    std_cl = np.std(critical_layers)
    mean_cf = np.mean(critical_fracs)
    std_cf = np.std(critical_fracs)
    print(f"\n  Mean critical layer: {mean_cl:.1f} +/- {std_cl:.1f}")
    print(f"  Mean critical depth: {mean_cf:.1%} +/- {std_cf:.1%}")
    n_layers = results[list(results.keys())[0]]["n_layers"]
    optimal = int(round(mean_cl)) - 1  # critical - 1
    print(f"  Suggested optimal layer: {optimal} (critical - 1)")
    print(f"  Total layers: {n_layers}")

    # Save aggregate results
    output_path = output_dir / "layerwise_depth_analysis_mitra_regression.json"
    serializable = {}
    for ds, r in results.items():
        serializable[ds] = {k: v for k, v in r.items()}
    serializable["_summary"] = {
        "mean_critical_layer": float(mean_cl),
        "std_critical_layer": float(std_cl),
        "mean_critical_depth_frac": float(mean_cf),
        "std_critical_depth_frac": float(std_cf),
        "suggested_optimal_layer": optimal,
        "n_layers": n_layers,
        "n_datasets": len(results),
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
