#!/usr/bin/env python3
"""Global pairwise ablation: ablate concepts unique to the better-performing model.

For each model pair (A, B) where A outperforms B on a dataset, ablate concepts
that A has but B lacks, and measure whether A's accuracy drops toward B's level.

The "concept contribution ratio" = (baseline_A - ablated_A) / (baseline_A - baseline_B)
measures what fraction of the performance gap is explained by the unique concepts.

Usage:
    # Single pair, single dataset
    python scripts/ablate_unique_concepts.py --model-a tabpfn --model-b mitra \\
        --datasets adult --device cuda

    # Full sweep across all TabArena datasets
    python scripts/ablate_unique_concepts.py --model-a tabpfn --model-b mitra \\
        --device cuda

    # All model pairs
    python scripts/ablate_unique_concepts.py --all-pairs --device cuda
"""

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from scripts._project_root import PROJECT_ROOT

from scripts.intervention.intervene_sae import (
    intervene,
    load_sae,
    get_extraction_layer,
    INTERVENE_FN,
    DEFAULT_SAE_DIR,
    DEFAULT_TRAINING_DIR,
    DEFAULT_LAYERS_PATH,
)

logger = logging.getLogger(__name__)

DEFAULT_HIERARCHY_PATH = PROJECT_ROOT / "output" / "concept_hierarchy_full.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "pairwise_ablation_results.json"

# Models with intervention support (requires per-model hook in intervene_sae.py)
ABLATION_MODELS = [
    "tabpfn", "mitra", "tabicl", "tabicl_v2", "tabdpt",
    "hyperfast", "carte", "tabula8b",
]

# Display name mapping for hierarchy lookup (all 8 models for completeness)
DISPLAY_NAMES = {
    "tabpfn": "TabPFN",
    "mitra": "Mitra",
    "tabicl": "TabICL",
    "tabicl_v2": "TabICL-v2",
    "tabdpt": "TabDPT",
    "hyperfast": "HyperFast",
    "carte": "CARTE",
    "tabula8b": "Tabula-8B",
}


def load_hierarchy(path: Path) -> dict:
    """Load concept hierarchy JSON."""
    with open(path) as f:
        return json.load(f)


def get_unique_features(
    hierarchy: dict,
    model_a_display: str,
    model_b_display: str,
) -> Tuple[List[int], List[str]]:
    """Get feature indices unique to model A vs model B.

    Returns:
        (feature_indices, concept_labels) — features in A's SAE that correspond
        to concept groups A has but B lacks.
    """
    comp = hierarchy["model_comparison"]
    vs_key = f"vs_{model_b_display}"
    unique_info = comp.get("unique_to", {}).get(model_a_display, {}).get(vs_key, {})
    unique_groups = unique_info.get("groups", [])

    if not unique_groups:
        return [], []

    # Collect feature indices from the hierarchy
    feature_indices = []
    concept_labels = []

    h = hierarchy["hierarchy"]
    for band_data in h.values():
        for cat_data in band_data.values():
            for gid, group in cat_data.get("groups", {}).items():
                if gid in unique_groups:
                    model_features = group.get("features", {}).get(model_a_display, [])
                    feature_indices.extend(model_features)
                    if model_features:
                        concept_labels.append(group.get("label", f"group_{gid}"))

    return feature_indices, concept_labels


def evaluate_predictions(
    preds: np.ndarray,
    y_true: np.ndarray,
    task: str = "classification",
) -> float:
    """Compute accuracy (classification) or R² (regression)."""
    if task == "classification":
        if preds.ndim == 2:
            pred_labels = preds.argmax(axis=1)
        else:
            pred_labels = preds
        return float(np.mean(pred_labels == y_true))
    else:
        from sklearn.metrics import r2_score
        return float(r2_score(y_true, preds))


def ablate_pair_dataset(
    model_a: str,
    model_b: str,
    dataset: str,
    hierarchy: dict,
    device: str = "cuda",
    sae_dir: Path = DEFAULT_SAE_DIR,
    training_dir: Path = DEFAULT_TRAINING_DIR,
    layers_path: Path = DEFAULT_LAYERS_PATH,
) -> Optional[dict]:
    """Run pairwise ablation for one model pair on one dataset.

    Returns:
        Dict with baseline/ablated metrics, or None if skipped.
    """
    from scripts.embeddings.extract_layer_embeddings import get_dataset_task
    from data.extended_loader import load_tabarena_dataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    task = get_dataset_task(dataset)
    display_a = DISPLAY_NAMES[model_a]
    display_b = DISPLAY_NAMES[model_b]

    # Get unique features
    unique_features, concept_labels = get_unique_features(hierarchy, display_a, display_b)
    if not unique_features:
        logger.info("  No unique concepts for %s vs %s", display_a, display_b)
        return None

    # Load data
    result = load_tabarena_dataset(dataset, max_samples=1100)
    if result is None:
        logger.warning("  Failed to load dataset: %s", dataset)
        return None

    X, y, _ = result
    n = len(X)
    ctx_size = min(600, int(n * 0.7))
    q_size = min(500, n - ctx_size)

    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)
        query_frac = q_size / (ctx_size + q_size)
        try:
            X_ctx, X_q, y_ctx, y_q = train_test_split(
                X, y, test_size=query_frac, random_state=42, stratify=y,
            )
        except ValueError:
            X_ctx, X_q, y_ctx, y_q = train_test_split(
                X, y, test_size=query_frac, random_state=42,
            )
    else:
        X_ctx = X[:ctx_size]
        y_ctx = y[:ctx_size]
        X_q = X[ctx_size:ctx_size + q_size]
        y_q = y[ctx_size:ctx_size + q_size]

    X_ctx = X_ctx[:ctx_size]
    y_ctx = y_ctx[:ctx_size]
    X_q = X_q[:q_size]
    y_q = y_q[:q_size]

    # Run intervention on model A
    try:
        results_a = intervene(
            model_key=model_a,
            X_context=X_ctx,
            y_context=y_ctx,
            X_query=X_q,
            y_query=y_q,
            ablate_features=unique_features,
            device=device,
            task=task,
            sae_dir=sae_dir,
            layers_path=layers_path,
            training_dir=training_dir,
        )
    except Exception as e:
        logger.error("  Intervention failed for %s on %s: %s", model_a, dataset, e)
        return None

    # Get baseline prediction for model B (run with no ablation)
    try:
        results_b = intervene(
            model_key=model_b,
            X_context=X_ctx,
            y_context=y_ctx,
            X_query=X_q,
            y_query=y_q,
            ablate_features=[],  # No ablation — just get baseline
            device=device,
            task=task,
            sae_dir=sae_dir,
            layers_path=layers_path,
            training_dir=training_dir,
        )
    except Exception as e:
        logger.error("  Baseline failed for %s on %s: %s", model_b, dataset, e)
        return None

    # Evaluate
    baseline_a = evaluate_predictions(results_a["baseline_preds"], y_q, task)
    ablated_a = evaluate_predictions(results_a["ablated_preds"], y_q, task)
    baseline_b = evaluate_predictions(results_b["baseline_preds"], y_q, task)

    gap = baseline_a - baseline_b
    drop = baseline_a - ablated_a
    contribution_ratio = drop / gap if abs(gap) > 1e-6 else float("nan")

    return {
        "dataset": dataset,
        "task": task,
        "baseline_acc_a": baseline_a,
        "ablated_acc_a": ablated_a,
        "baseline_acc_b": baseline_b,
        "drop": drop,
        "gap": gap,
        "contribution_ratio": contribution_ratio,
        "n_unique_concepts": len(concept_labels),
        "n_features_ablated": len(unique_features),
        "ablated_concepts": concept_labels,
        "n_query": len(y_q),
    }


def ablate_pair(
    model_a: str,
    model_b: str,
    datasets: List[str],
    hierarchy: dict,
    device: str = "cuda",
    sae_dir: Path = DEFAULT_SAE_DIR,
    training_dir: Path = DEFAULT_TRAINING_DIR,
    layers_path: Path = DEFAULT_LAYERS_PATH,
) -> dict:
    """Run pairwise ablation across multiple datasets.

    Returns summary dict for this model pair.
    """
    display_a = DISPLAY_NAMES[model_a]
    display_b = DISPLAY_NAMES[model_b]
    logger.info("Ablating %s vs %s across %d datasets", display_a, display_b, len(datasets))

    dataset_results = {}
    for ds in datasets:
        logger.info("  Dataset: %s", ds)
        result = ablate_pair_dataset(
            model_a, model_b, ds, hierarchy,
            device=device, sae_dir=sae_dir, training_dir=training_dir,
            layers_path=layers_path,
        )
        if result is not None:
            dataset_results[ds] = result

    # Summary
    if dataset_results:
        drops = [r["drop"] for r in dataset_results.values()]
        contributions = [r["contribution_ratio"] for r in dataset_results.values()
                         if not np.isnan(r["contribution_ratio"])]
        n_where_drop = sum(1 for d in drops if d > 0)
    else:
        drops = []
        contributions = []
        n_where_drop = 0

    return {
        "model_a": display_a,
        "model_b": display_b,
        "datasets": dataset_results,
        "summary": {
            "mean_drop": float(np.mean(drops)) if drops else 0.0,
            "median_drop": float(np.median(drops)) if drops else 0.0,
            "mean_contribution_ratio": float(np.mean(contributions)) if contributions else 0.0,
            "n_datasets_where_drop": n_where_drop,
            "n_datasets_total": len(dataset_results),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Global pairwise concept ablation")
    parser.add_argument("--model-a", type=str, choices=ABLATION_MODELS)
    parser.add_argument("--model-b", type=str, choices=ABLATION_MODELS)
    parser.add_argument("--all-pairs", action="store_true",
                        help="Run all model pair combinations")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets (default: all TabArena)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hierarchy", type=Path, default=DEFAULT_HIERARCHY_PATH)
    parser.add_argument("--sae-dir", type=Path, default=DEFAULT_SAE_DIR)
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--layers-config", type=Path, default=DEFAULT_LAYERS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    hierarchy = load_hierarchy(args.hierarchy)

    # Get dataset list
    if args.datasets:
        datasets = args.datasets
    else:
        from data.extended_loader import TABARENA_DATASETS
        datasets = sorted(TABARENA_DATASETS.keys())

    # Get model pairs
    if args.all_pairs:
        pairs = list(combinations(ABLATION_MODELS, 2))
        # Run both directions for each pair
        pairs = [(a, b) for a, b in pairs] + [(b, a) for a, b in pairs]
    elif args.model_a and args.model_b:
        pairs = [(args.model_a, args.model_b)]
    else:
        parser.error("Specify --model-a/--model-b or --all-pairs")

    all_results = {"pairwise_ablations": {}}

    for model_a, model_b in pairs:
        pair_key = f"{DISPLAY_NAMES[model_a]}__{DISPLAY_NAMES[model_b]}"
        result = ablate_pair(
            model_a, model_b, datasets, hierarchy,
            device=args.device, sae_dir=args.sae_dir,
            training_dir=args.training_dir, layers_path=args.layers_config,
        )
        all_results["pairwise_ablations"][pair_key] = result

        # Print summary
        summary = result["summary"]
        print(f"\n{pair_key}:")
        print(f"  Mean drop: {summary['mean_drop']:.4f}")
        print(f"  Mean contribution ratio: {summary['mean_contribution_ratio']:.4f}")
        print(f"  Datasets with drop: {summary['n_datasets_where_drop']}/{summary['n_datasets_total']}")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
