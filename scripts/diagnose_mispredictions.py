#!/usr/bin/env python3
"""Per-dataset causal diagnostics: trace mispredictions to concept groups.

For each dataset, compare predictions sample-by-sample between model pairs.
Find systematic misprediction patterns, map to concepts, ablate, and verify.

Workflow for each (dataset, model_A, model_B):
1. Get baseline predictions for both: preds_A, preds_B, y_true
2. Find "A-correct, B-wrong" samples (A-wins) and vice versa
3. For A-wins samples: get SAE activations, find most-active concept groups
4. Ablate top-K active concept groups from A
5. Check: do A's predictions on A-wins samples flip to wrong?
6. If yes → causal evidence

Usage:
    # Single pair, single dataset
    python scripts/diagnose_mispredictions.py --model-a tabpfn --model-b mitra \\
        --dataset adult --device cuda

    # Sweep across datasets
    python scripts/diagnose_mispredictions.py --model-a tabpfn --model-b mitra \\
        --device cuda

    # Top-K concept groups to test
    python scripts/diagnose_mispredictions.py --model-a tabpfn --model-b mitra \\
        --dataset adult --top-k 5 --device cuda
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.intervene_sae import (
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "causal_diagnostics"

DISPLAY_NAMES = {
    "tabpfn": "TabPFN",
    "mitra": "Mitra",
    "tabicl": "TabICL",
    "tabdpt": "TabDPT",
    "hyperfast": "HyperFast",
}


def load_hierarchy(path: Path) -> dict:
    """Load concept hierarchy JSON."""
    with open(path) as f:
        return json.load(f)


def get_concept_group_features(hierarchy: dict, model_display: str) -> Dict[str, List[int]]:
    """Get mapping: group_id -> feature indices for a given model.

    Returns:
        Dict mapping group_id -> list of SAE feature indices belonging to that group.
    """
    group_features = {}
    h = hierarchy["hierarchy"]
    for band_data in h.values():
        for cat_data in band_data.values():
            for gid, group in cat_data.get("groups", {}).items():
                feats = group.get("features", {}).get(model_display, [])
                if feats:
                    group_features[gid] = feats
    return group_features


def get_concept_group_labels(hierarchy: dict) -> Dict[str, str]:
    """Get mapping: group_id -> label."""
    labels = {}
    h = hierarchy["hierarchy"]
    for band_data in h.values():
        for cat_data in band_data.values():
            for gid, group in cat_data.get("groups", {}).items():
                labels[gid] = group.get("label", f"group_{gid}")
    return labels


def get_sae_activations(
    model_key: str,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    device: str = "cuda",
    task: str = "classification",
    sae_dir: Path = DEFAULT_SAE_DIR,
    layers_path: Path = DEFAULT_LAYERS_PATH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract SAE activations for query samples.

    Uses the all-layers extraction functions from layerwise_cka_analysis.py
    to get embeddings at the optimal layer, then encodes through the SAE.

    Returns:
        (activations, embeddings) where activations is (n_query, hidden_dim)
    """
    import inspect
    from layerwise_cka_analysis import sort_layer_names
    from scripts.extract_layer_embeddings import EXTRACT_FN

    extract_fn = EXTRACT_FN[model_key]
    layer = get_extraction_layer(model_key, layers_path)

    # Pass task kwarg for models that support it
    sig = inspect.signature(extract_fn)
    kwargs = dict(device=device)
    if "task" in sig.parameters:
        kwargs["task"] = task

    layer_embeddings = extract_fn(X_context, y_context, X_query, **kwargs)

    available = sort_layer_names(list(layer_embeddings.keys()))
    if 0 <= layer < len(available):
        layer_key = available[layer]
    else:
        raise ValueError(f"Layer {layer} out of range for {model_key}: {available}")

    embeddings = layer_embeddings[layer_key]

    sae, _ = load_sae(model_key, sae_dir=sae_dir, device=device)
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)

    with torch.no_grad():
        h = sae.encode(emb_tensor)

    return h.cpu().numpy(), embeddings


def rank_concept_groups_by_activation(
    activations: np.ndarray,
    sample_mask: np.ndarray,
    group_features: Dict[str, List[int]],
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """Rank concept groups by mean activation on selected samples.

    Args:
        activations: (n_query, hidden_dim) SAE activations
        sample_mask: Boolean mask selecting samples of interest
        group_features: group_id -> feature indices
        top_k: Number of top groups to return

    Returns:
        List of (group_id, mean_activation) sorted by decreasing activation.
    """
    if sample_mask.sum() == 0:
        return []

    selected = activations[sample_mask]  # (n_selected, hidden_dim)

    group_scores = []
    for gid, feat_indices in group_features.items():
        # Mean activation of this group's features across selected samples
        group_act = selected[:, feat_indices].mean()
        group_scores.append((gid, float(group_act)))

    group_scores.sort(key=lambda x: x[1], reverse=True)
    return group_scores[:top_k]


def diagnose_dataset(
    model_a: str,
    model_b: str,
    dataset: str,
    hierarchy: dict,
    top_k: int = 10,
    device: str = "cuda",
    sae_dir: Path = DEFAULT_SAE_DIR,
    training_dir: Path = DEFAULT_TRAINING_DIR,
    layers_path: Path = DEFAULT_LAYERS_PATH,
) -> Optional[dict]:
    """Run per-dataset causal diagnostics for one model pair.

    Returns diagnostic results or None if skipped.
    """
    from scripts.extract_layer_embeddings import get_dataset_task
    from data.extended_loader import load_tabarena_dataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    task = get_dataset_task(dataset)
    if task != "classification":
        logger.info("  Skipping %s (regression not supported for misprediction analysis)", dataset)
        return None

    display_a = DISPLAY_NAMES[model_a]
    display_b = DISPLAY_NAMES[model_b]

    # Load data
    result = load_tabarena_dataset(dataset, max_samples=1100)
    if result is None:
        return None

    X, y, _ = result
    n = len(X)
    ctx_size = min(600, int(n * 0.7))
    q_size = min(500, n - ctx_size)

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

    X_ctx = X_ctx[:ctx_size]
    y_ctx = y_ctx[:ctx_size]
    X_q = X_q[:q_size]
    y_q = y_q[:q_size]

    # --- Get baseline predictions for both models ---
    try:
        results_a = intervene(
            model_key=model_a, X_context=X_ctx, y_context=y_ctx,
            X_query=X_q, y_query=y_q, ablate_features=[],
            device=device, task=task, sae_dir=sae_dir,
            training_dir=training_dir, layers_path=layers_path,
        )
        results_b = intervene(
            model_key=model_b, X_context=X_ctx, y_context=y_ctx,
            X_query=X_q, y_query=y_q, ablate_features=[],
            device=device, task=task, sae_dir=sae_dir,
            training_dir=training_dir, layers_path=layers_path,
        )
    except Exception as e:
        logger.error("  Baseline failed: %s", e)
        return None

    preds_a = results_a["baseline_preds"].argmax(axis=1)
    preds_b = results_b["baseline_preds"].argmax(axis=1)

    # --- Identify misprediction categories ---
    a_correct = preds_a == y_q
    b_correct = preds_b == y_q
    a_wins = a_correct & ~b_correct  # A right, B wrong
    b_wins = ~a_correct & b_correct  # B right, A wrong
    both_correct = a_correct & b_correct
    both_wrong = ~a_correct & ~b_correct

    n_a_wins = int(a_wins.sum())
    n_b_wins = int(b_wins.sum())

    if n_a_wins == 0:
        logger.info("  No A-wins samples for %s vs %s on %s", display_a, display_b, dataset)
        return {
            "dataset": dataset,
            "model_a": display_a,
            "model_b": display_b,
            "n_samples": len(y_q),
            "a_correct_b_wrong": 0,
            "b_correct_a_wrong": n_b_wins,
            "both_correct": int(both_correct.sum()),
            "both_wrong": int(both_wrong.sum()),
            "causal_interventions": [],
        }

    # --- Get SAE activations for model A ---
    group_features = get_concept_group_features(hierarchy, display_a)
    group_labels = get_concept_group_labels(hierarchy)

    try:
        activations, _ = get_sae_activations(
            model_a, X_ctx, y_ctx, X_q,
            device=device, task=task, sae_dir=sae_dir, layers_path=layers_path,
        )
    except Exception as e:
        logger.error("  SAE activation extraction failed: %s", e)
        return None

    # --- Rank concept groups by activity on A-wins samples ---
    top_groups = rank_concept_groups_by_activation(
        activations, a_wins, group_features, top_k=top_k,
    )

    # --- Ablate each top group and check for flips ---
    causal_interventions = []

    for gid, mean_act in top_groups:
        features = group_features[gid]

        try:
            ablated = intervene(
                model_key=model_a, X_context=X_ctx, y_context=y_ctx,
                X_query=X_q, y_query=y_q, ablate_features=features,
                device=device, task=task, sae_dir=sae_dir,
                training_dir=training_dir, layers_path=layers_path,
            )
        except Exception as e:
            logger.error("  Ablation failed for group %s: %s", gid, e)
            continue

        ablated_preds = ablated["ablated_preds"].argmax(axis=1)

        # Count how many A-wins samples flip to wrong after ablation
        flipped = a_wins & (ablated_preds != y_q)
        n_flipped = int(flipped.sum())
        flip_rate = n_flipped / n_a_wins if n_a_wins > 0 else 0.0

        # Simple binomial test: is flip rate significantly above chance?
        # Under null hypothesis (ablation doesn't matter), flip rate ≈ 0
        # Use normal approximation for p-value
        if n_a_wins > 0 and n_flipped > 0:
            from scipy.stats import binomtest
            p_value = float(binomtest(n_flipped, n_a_wins, 1.0 / max(2, len(le.classes_))).pvalue)
        else:
            p_value = 1.0

        causal_interventions.append({
            "concept_group": gid,
            "label": group_labels.get(gid, ""),
            "n_features": len(features),
            "mean_activation": mean_act,
            "n_a_wins_flipped": n_flipped,
            "flip_rate": flip_rate,
            "p_value": p_value,
        })

    return {
        "dataset": dataset,
        "model_a": display_a,
        "model_b": display_b,
        "n_samples": len(y_q),
        "a_correct_b_wrong": n_a_wins,
        "b_correct_a_wrong": n_b_wins,
        "both_correct": int(both_correct.sum()),
        "both_wrong": int(both_wrong.sum()),
        "causal_interventions": causal_interventions,
    }


def main():
    parser = argparse.ArgumentParser(description="Per-dataset causal misprediction diagnostics")
    parser.add_argument("--model-a", type=str, required=True, choices=list(INTERVENE_FN.keys()))
    parser.add_argument("--model-b", type=str, required=True, choices=list(INTERVENE_FN.keys()))
    parser.add_argument("--dataset", type=str, default=None, help="Single dataset")
    parser.add_argument("--datasets", nargs="+", default=None, help="Multiple datasets")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top concept groups to test per dataset")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hierarchy", type=Path, default=DEFAULT_HIERARCHY_PATH)
    parser.add_argument("--sae-dir", type=Path, default=DEFAULT_SAE_DIR)
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--layers-config", type=Path, default=DEFAULT_LAYERS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    hierarchy = load_hierarchy(args.hierarchy)

    # Get dataset list
    if args.dataset:
        datasets = [args.dataset]
    elif args.datasets:
        datasets = args.datasets
    else:
        from data.extended_loader import TABARENA_DATASETS
        datasets = sorted(TABARENA_DATASETS.keys())

    display_a = DISPLAY_NAMES[args.model_a]
    display_b = DISPLAY_NAMES[args.model_b]
    print(f"Diagnosing {display_a} vs {display_b} across {len(datasets)} datasets")
    print(f"Top-K concept groups: {args.top_k}")

    all_results = []
    for ds in datasets:
        print(f"\n--- {ds} ---")
        result = diagnose_dataset(
            args.model_a, args.model_b, ds, hierarchy,
            top_k=args.top_k, device=args.device,
            sae_dir=args.sae_dir, training_dir=args.training_dir,
            layers_path=args.layers_config,
        )
        if result is not None:
            all_results.append(result)

            # Print summary
            print(f"  Samples: {result['n_samples']}")
            print(f"  {display_a} wins: {result['a_correct_b_wrong']}, "
                  f"{display_b} wins: {result['b_correct_a_wrong']}")

            sig_interventions = [
                c for c in result["causal_interventions"]
                if c["p_value"] < 0.05
            ]
            if sig_interventions:
                print(f"  Significant causal interventions: {len(sig_interventions)}")
                for c in sig_interventions[:3]:
                    print(f"    {c['label']}: flip_rate={c['flip_rate']:.2f}, "
                          f"p={c['p_value']:.4f}")

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"diagnostics_{args.model_a}_vs_{args.model_b}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results to {output_path}")


if __name__ == "__main__":
    main()
