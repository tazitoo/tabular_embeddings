#!/usr/bin/env python3
"""
Extract embeddings at optimal layers for all models.

Reads pre-extracted layerwise data from output/embeddings/tabarena_layerwise/{model}/
and selects the optimal layer specified in config/optimal_extraction_layers.json.

Saves single-layer embeddings to output/embeddings/tabarena/{model}/ for SAE training.

Requires: extract_all_layers.py must have been run first for each model.

Usage:
    python scripts/extract_optimal_layers.py
    python scripts/extract_optimal_layers.py --models tabpfn mitra
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_optimal_layers
from data.extended_loader import TABARENA_DATASETS
from scripts.extract_layer_embeddings import sort_layer_names


def get_dataset_task(dataset_name: str) -> str:
    """Look up task type ('classification' or 'regression') from the catalog."""
    info = TABARENA_DATASETS.get(dataset_name, {})
    return info.get("task", "classification")


def main():
    parser = argparse.ArgumentParser(
        description="Extract optimal-layer embeddings from layerwise data"
    )
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to extract (default: all in config)")
    args = parser.parse_args()

    config = load_optimal_layers()
    models = args.models or list(config.keys())

    layerwise_base = PROJECT_ROOT / "output" / "embeddings" / "tabarena_layerwise"
    output_base = PROJECT_ROOT / "output" / "embeddings" / "tabarena"

    print(f"{'Model':<12} {'Layer':<10} {'Datasets':<10}")
    print("-" * 35)

    for model in models:
        cls_layer = config[model]["optimal_layer"]
        reg_layer = config[model].get("regression_layer")
        n_layers = config[model]["n_layers"]
        layerwise_dir = layerwise_base / model
        output_dir = output_base / model
        output_dir.mkdir(parents=True, exist_ok=True)

        npz_files = sorted(layerwise_dir.glob("tabarena_*.npz"))
        assert len(npz_files) > 0, f"No layerwise data for {model} in {layerwise_dir}"

        n_cls, n_reg = 0, 0
        for npz_path in npz_files:
            out_path = output_dir / f"{npz_path.stem}.npz"
            data = np.load(npz_path, allow_pickle=True)
            layer_names = sort_layer_names(list(data["layer_names"]))

            # Pick layer based on dataset task type
            ds_name = npz_path.stem.replace("tabarena_", "")
            task = get_dataset_task(ds_name)
            if task == "regression" and reg_layer is not None:
                optimal_layer = reg_layer
                n_reg += 1
            else:
                optimal_layer = cls_layer
                n_cls += 1

            assert optimal_layer < len(layer_names), (
                f"{model}: layer {optimal_layer} out of range ({len(layer_names)} layers)"
            )

            layer_key = layer_names[optimal_layer]
            embeddings = data[layer_key]

            np.savez_compressed(
                str(out_path),
                embeddings=embeddings,
                extraction_point=np.array(layer_key),
                n_samples=np.array(embeddings.shape[0]),
                embedding_dim=np.array(embeddings.shape[1]),
            )

        layer_str = f"L{cls_layer}/{n_layers}"
        if reg_layer is not None:
            layer_str += f" (reg:L{reg_layer})"
        print(f"{model:<12} {layer_str:<20} {n_cls} cls + {n_reg} reg = {len(npz_files)}")

    print(f"\nOutput: {output_base}")


if __name__ == "__main__":
    main()
