#!/usr/bin/env python3
"""Stage 3: Build concept × dataset importance matrix.

For each model pair, builds a matrix of (top_concepts × datasets) showing
where each concept is most important. Enables dataset-specific storytelling:
"concept f92 is most important on hiva_agnostic but irrelevant on credit-g."

Also computes:
- For each dataset: top 3 concepts that explain most of its gap
- For each concept: top 3 datasets where it's most active

Usage:
    python -m scripts.causal_chain_analysis.03_concept_dataset_matrix
    python -m scripts.causal_chain_analysis.03_concept_dataset_matrix --pair tabicl_vs_tabicl_v2
"""
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ABLATION_DIR = PROJECT_ROOT / "output" / "ablation_sweep"
EVIDENCE_DIR = PROJECT_ROOT / "output" / "concept_evidence"


def build_matrix(pair_name, top_n=20):
    """Build concept × dataset importance matrix for one pair."""
    pair_dir = ABLATION_DIR / pair_name
    ranking_path = EVIDENCE_DIR / pair_name / "concept_ranking.json"

    if not ranking_path.exists():
        logger.warning(f"No concept ranking for {pair_name}, run Stage 1 first")
        return None

    ranking = json.loads(ranking_path.read_text())

    results = {}
    for model_key, model_data in ranking["models"].items():
        top_features = model_data["top_features"][:top_n]
        if not top_features:
            continue

        feature_indices = [f["feature_idx"] for f in top_features]
        feature_labels = {f["feature_idx"]: f["label"][:60] for f in top_features}
        feature_bands = {f["feature_idx"]: f["band"] for f in top_features}
        feature_unmatched = {f["feature_idx"]: f["is_unmatched"] for f in top_features}

        # Build per-dataset importance for these features
        # importance[fi][dataset] = mean importance on strong-win rows
        importance = defaultdict(lambda: defaultdict(float))
        dataset_gc = {}  # dataset -> mean gap_closed
        dataset_strong_count = {}  # dataset -> n_strong_wins

        for f in sorted(pair_dir.glob("*.npz")):
            d = np.load(f, allow_pickle=True)
            if str(d["strong_model"]) != model_key:
                continue
            if "strong_wins" not in d.files or "selected_features" not in d.files:
                continue

            sw = d["strong_wins"]
            if sw.sum() == 0:
                continue

            ds = f.stem
            gc = d["gap_closed"][sw]
            gc = gc[~np.isnan(gc)]
            dataset_gc[ds] = float(gc.mean()) if len(gc) else 0
            dataset_strong_count[ds] = int(sw.sum())

            sel = d["selected_features"]
            if sel.size == 0 or sel.ndim != 2:
                continue

            # Count how many strong-win rows select each feature
            for r in range(len(sw)):
                if not sw[r]:
                    continue
                for fi in sel[r]:
                    fi = int(fi)
                    if fi in feature_indices:
                        importance[fi][ds] += 1

        if not importance:
            continue

        # Normalize: fraction of strong-win rows that select this feature
        for fi in importance:
            for ds in importance[fi]:
                n = dataset_strong_count.get(ds, 1)
                importance[fi][ds] /= max(n, 1)

        # Get all datasets where this model is strong
        datasets = sorted(dataset_gc.keys())

        # Build matrix (features × datasets)
        matrix = np.zeros((len(feature_indices), len(datasets)))
        for i, fi in enumerate(feature_indices):
            for j, ds in enumerate(datasets):
                matrix[i, j] = importance[fi].get(ds, 0)

        # Per-dataset: top 3 concepts
        dataset_top_concepts = {}
        for j, ds in enumerate(datasets):
            col = matrix[:, j]
            top_idx = np.argsort(-col)[:3]
            dataset_top_concepts[ds] = [
                {"feature_idx": feature_indices[k],
                 "importance": float(col[k]),
                 "label": feature_labels.get(feature_indices[k], "")}
                for k in top_idx if col[k] > 0
            ]

        # Per-concept: top 3 datasets
        concept_top_datasets = {}
        for i, fi in enumerate(feature_indices):
            row = matrix[i, :]
            top_idx = np.argsort(-row)[:3]
            concept_top_datasets[fi] = [
                {"dataset": datasets[k],
                 "importance": float(row[k]),
                 "gc": dataset_gc.get(datasets[k], 0)}
                for k in top_idx if row[k] > 0
            ]

        results[model_key] = {
            "display_name": model_data["display_name"],
            "feature_indices": feature_indices,
            "feature_labels": feature_labels,
            "feature_bands": feature_bands,
            "feature_unmatched": feature_unmatched,
            "datasets": datasets,
            "matrix": matrix.tolist(),
            "dataset_gc": dataset_gc,
            "dataset_top_concepts": dataset_top_concepts,
            "concept_top_datasets": {str(k): v for k, v in concept_top_datasets.items()},
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Build concept × dataset importance matrix")
    parser.add_argument("--pair", default=None)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    if args.pair:
        pairs = [args.pair]
    else:
        pairs = sorted(d.name for d in EVIDENCE_DIR.iterdir()
                        if d.is_dir() and (d / "concept_ranking.json").exists())

    for pair in pairs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Pair: {pair}")

        result = build_matrix(pair, top_n=args.top_n)
        if result is None:
            continue

        out_path = EVIDENCE_DIR / pair / "concept_dataset_matrix.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        for model_key, data in result.items():
            disp = data["display_name"]
            n_feat = len(data["feature_indices"])
            n_ds = len(data["datasets"])
            logger.info(f"\n  {disp} (when strong): {n_feat} concepts × {n_ds} datasets")

            # Show top 3 concepts with their top datasets
            logger.info(f"  Top concepts and where they matter:")
            for fi in data["feature_indices"][:5]:
                label = data["feature_labels"].get(fi, "")[:40]
                band = data["feature_bands"].get(fi, -1)
                unm = "U" if data["feature_unmatched"].get(fi) else "M"
                top_ds = data["concept_top_datasets"].get(str(fi), [])
                ds_str = ", ".join(f"{d['dataset']}({d['importance']:.2f})"
                                   for d in top_ds[:3])
                logger.info(f"    f{fi} [B{band},{unm}]: {label}")
                logger.info(f"      top datasets: {ds_str}")

        logger.info(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
