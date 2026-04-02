#!/usr/bin/env python3
"""Cross-model ablation sweep: how many concepts separate two models?

For each dataset, determines which model is stronger (dataset-level AUC/RMSE),
then ablates the stronger model's features until per-row loss reaches parity
with the weaker model.

Output:
    output/ablation_sweep/{model_a}_vs_{model_b}/{dataset}.npz

    The output is symmetric — only one directory per unordered pair.
    The NPZ records which model was stronger per dataset.

Usage:
    python -m scripts.intervention.ablation_sweep --models tabpfn tabicl --device cuda
    python -m scripts.intervention.ablation_sweep --models tabpfn tabicl --datasets credit-g
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import (
    SPLITS_PATH,
    load_sae, get_extraction_layer_taskaware, build_tail,
    load_dataset_context, load_test_embeddings,
    compute_per_row_loss, compute_importance_metric,
    batched_ablation, batched_ablation_sequential,
    MitraTail, SEQUENTIAL_MODELS,
    get_alive_features, MODEL_KEY_TO_LABEL_KEY, DEFAULT_CONCEPT_LABELS,
)
from scripts.matching.utils import load_norm_stats as load_norm_stats_matching

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "output" / "ablation_sweep"
IMPORTANCE_DIR = PROJECT_ROOT / "output" / "perrow_importance"

SUPPORTED_MODELS = ["tabpfn", "tabicl", "tabicl_v2", "mitra", "tabdpt", "hyperfast", "carte", "tabula8b"]


def get_unmatched_features(source_model: str, target_model: str,
                           concept_labels_path=None):
    """Get source features NOT matched to the target model.

    A feature is "matched" if it appears in a concept group that also contains
    at least one feature from the target model.  Everything else is unmatched —
    these are the features unique to the source model.
    """
    if concept_labels_path is None:
        concept_labels_path = DEFAULT_CONCEPT_LABELS

    with open(concept_labels_path) as f:
        data = json.load(f)

    src_key = MODEL_KEY_TO_LABEL_KEY.get(source_model, source_model)
    tgt_key = MODEL_KEY_TO_LABEL_KEY.get(target_model, target_model)

    matched_source = set()
    for group in data.get("concept_groups", {}).values():
        members = group.get("members", [])
        models_in_group = set(m for m, _ in members)
        if src_key in models_in_group and tgt_key in models_in_group:
            for m, f in members:
                if m == src_key:
                    matched_source.add(f)

    all_source = set(get_alive_features(source_model, concept_labels_path))
    return sorted(all_source - matched_source)


def run_dataset(
    model_a: str,
    model_b: str,
    dataset: str,
    saes: dict,
    splits: dict,
    norm_stats: dict,
    test_embeddings: dict,
    unmatched: dict,
    device: str,
    max_K: int,
    max_steps: int,
) -> dict:
    """Cross-model ablation for one dataset.

    Determines strong/weak from dataset-level metric, then ablates the
    strong model's features on rows where it outperforms the weak model.
    """

    # Load cached baseline predictions from perrow_importance output.
    # This avoids building the weak model's tail (saves GPU memory and time).
    imp_a = np.load(IMPORTANCE_DIR / model_a / f"{dataset}.npz", allow_pickle=True)
    imp_b = np.load(IMPORTANCE_DIR / model_b / f"{dataset}.npz", allow_pickle=True)
    preds_a = imp_a["baseline_preds"]
    preds_b = imp_b["baseline_preds"]
    row_indices_a = imp_a["row_indices"]
    row_indices_b = imp_b["row_indices"]
    assert np.array_equal(row_indices_a, row_indices_b), (
        f"Row index mismatch between {model_a} and {model_b} on {dataset}"
    )
    row_indices = row_indices_a
    y_query = imp_a["y_query"]
    n_query = len(y_query)
    # Infer task from prediction shape (classification: 2D probabilities, regression: 1D)
    task = "classification" if preds_a.ndim == 2 else "regression"

    imps = {model_a: imp_a, model_b: imp_b}
    preds = {model_a: preds_a, model_b: preds_b}
    losses = {
        model_a: compute_per_row_loss(y_query, preds_a, task),
        model_b: compute_per_row_loss(y_query, preds_b, task),
    }

    # Determine strong/weak from dataset-level metric (higher = better)
    metric_a, metric_name = compute_importance_metric(y_query, preds[model_a], task)
    metric_b, _ = compute_importance_metric(y_query, preds[model_b], task)

    # Skip if either model produces degenerate predictions
    if metric_name == "degenerate" or metric_a == float("-inf") or metric_b == float("-inf"):
        logger.info(f"  SKIP (degenerate predictions from one or both models)")
        return {
            "strong_model": model_a, "weak_model": model_b,
            "n_strong_wins": 0, "n_query": n_query,
            "metric_strong": 0.0, "metric_weak": 0.0, "metric_name": "degenerate",
        }

    if metric_a >= metric_b:
        strong, weak = model_a, model_b
        metric_strong, metric_weak = metric_a, metric_b
    else:
        strong, weak = model_b, model_a
        metric_strong, metric_weak = metric_b, metric_a

    logger.info(f"  {strong} ({metric_name}={metric_strong:.4f}) > "
                f"{weak} ({metric_name}={metric_weak:.4f})")

    # Need norm stats for the strong model (SAE encoding + delta denormalization)
    if dataset not in norm_stats[strong]:
        logger.info(f"  SKIP (strong model {strong} has no norm stats for {dataset})")
        return {
            "strong_model": strong, "weak_model": weak,
            "n_strong_wins": 0, "n_query": n_query,
            "metric_strong": float(metric_strong),
            "metric_weak": float(metric_weak), "metric_name": metric_name,
        }

    baseline_loss_s = losses[strong]
    weak_loss = losses[weak]
    baseline_preds_s = preds[strong]
    weak_preds = preds[weak]

    # Build tail only for the strong model
    t0 = time.time()
    X_train_s, y_train_s, X_query_s, _, _, task_s = load_dataset_context(strong, dataset, splits)
    if y_train_s.dtype == np.int32:
        y_train_s = y_train_s.astype(np.int64)
    layer_s = get_extraction_layer_taskaware(strong, dataset=dataset)
    cat_indices = None
    if strong in ("hyperfast", "tabpfn"):
        from data.preprocessing import load_preprocessed, CACHE_DIR
        try:
            pre = load_preprocessed(strong, dataset, CACHE_DIR)
            cat_indices = pre.cat_indices if pre.cat_indices else None
        except Exception:
            pass
    target_name = splits.get(dataset, {}).get("target", "target")
    tail_s = build_tail(strong, X_train_s, y_train_s, X_query_s, layer_s, task_s, device,
                        cat_indices=cat_indices, target_name=target_name)
    logger.info(f"  Strong tail ({strong}) built in {time.time() - t0:.1f}s")

    # Load importance arrays for the strong model (already in imps dict)
    imp_s = imps[strong]
    row_feature_drops = imp_s["row_feature_drops"]
    feature_indices = imp_s["feature_indices"]

    # Only ablate features unmatched to the weak model
    pair_key = f"{strong}__{weak}"
    unmatched_set = set(unmatched.get(pair_key, []))
    n_total_features = len(feature_indices)
    n_unmatched_features = sum(1 for fi in feature_indices if int(fi) in unmatched_set)
    logger.info(f"  Unmatched features: {n_unmatched_features}/{n_total_features}")

    emb_s = test_embeddings[strong][dataset]
    with torch.no_grad():
        emb_t = torch.tensor(emb_s, dtype=torch.float32, device=device)
        activations_s = saes[strong].encode(emb_t)
    firing_mask_s = (activations_s > 0).cpu().numpy()

    ds_mean_s, ds_std_s = norm_stats[strong][dataset]
    data_mean_t_s = torch.tensor(ds_mean_s, dtype=torch.float32, device=device)
    data_std_t_s = torch.tensor(ds_std_s, dtype=torch.float32, device=device)

    # Filter to rows where strong outperforms weak
    strong_wins = baseline_loss_s < weak_loss
    n_strong_wins = int(strong_wins.sum())
    logger.info(f"  Strong wins on {n_strong_wins}/{n_query} rows")

    if n_strong_wins == 0:
        return {
            "strong_model": strong, "weak_model": weak,
            "n_strong_wins": 0, "n_query": n_query,
            "metric_strong": float(metric_strong),
            "metric_weak": float(metric_weak),
            "metric_name": metric_name,
        }

    use_sequential = isinstance(tail_s, SEQUENTIAL_MODELS)
    use_mitra = isinstance(tail_s, MitraTail)

    # Per-row ablation search
    optimal_k = np.zeros(n_query, dtype=np.int32)
    gap_closed = np.full(n_query, np.nan, dtype=np.float32)
    preds_intervened = baseline_preds_s.copy()
    selected_features = [[] for _ in range(n_query)]  # per-row accepted feature indices

    t0 = time.time()
    for r in range(n_query):
        if not strong_wins[r]:
            optimal_k[r] = 0
            gap_closed[r] = 1.0
            continue

        # Original distance from strong to weak prediction
        y_r = int(y_query[r])
        if baseline_preds_s.ndim == 2:
            eps = 1e-7
            # Per-row log loss on the correct class
            orig_dist = -np.log(np.clip(baseline_preds_s[r, y_r], eps, 1 - eps))
            target_dist = -np.log(np.clip(weak_preds[r, y_r], eps, 1 - eps))
        else:
            orig_dist = float((baseline_preds_s[r] - weak_preds[r]) ** 2)
            target_dist = 0.0
        if abs(orig_dist - target_dist) < 1e-8:
            optimal_k[r] = 0
            gap_closed[r] = 1.0
            continue

        # Rank this row's firing UNMATCHED features by importance
        row_drops = row_feature_drops[r]
        row_firing = [i for i, fi in enumerate(feature_indices)
                      if firing_mask_s[r, fi] and int(fi) in unmatched_set]
        if not row_firing:
            continue

        # Rank by importance (positive = feature helps model predict well).
        # Only remove helpful features — removing them degrades the model.
        firing_importance = [(i, row_drops[i]) for i in row_firing if row_drops[i] > 0]
        firing_importance.sort(key=lambda x: -x[1])
        ranked = [int(feature_indices[i]) for i, _ in firing_importance[:max_steps]]
        K = len(ranked)

        h_row = activations_s[r]
        X_row = X_query_s[r:r + 1]

        # Compute per-feature deltas (individual, not cumulative)
        with torch.no_grad():
            recon_full = saes[strong].decode(h_row.unsqueeze(0))
            h_batch = h_row.unsqueeze(0).expand(K, -1).clone()
            for k in range(K):
                h_batch[k, ranked[k]] = 0.0  # each row zeros ONE feature
            recon_abl = saes[strong].decode(h_batch)
            per_feature_deltas = (recon_abl - recon_full) * data_std_t_s.unsqueeze(0)

        # Distance metric: squared distance of per-row loss to the weak
        # model's loss.  Lower = closer to matching the weak model.
        if baseline_preds_s.ndim == 2:
            eps = 1e-7
            def dist_to_weak(p):
                p_loss = -np.log(np.clip(p[y_r], eps, 1 - eps))
                return (p_loss - target_dist) ** 2
        else:
            def dist_to_weak(p):
                return float((p - weak_preds[r]) ** 2)

        # Combinatorial batching: enumerate all subsets up to size 3 from
        # the top-N features, test them all in one batched forward pass,
        # pick the best subset. If gap remains, repeat with next batch.
        from itertools import combinations as combos

        batch_size = min(12, K)  # top-N features per batch
        current_dist = dist_to_weak(baseline_preds_s[r])
        best_pred = baseline_preds_s[r]
        accepted_combo = ()
        offset = 0

        while offset < K:
            batch_end = min(offset + batch_size, K)
            batch_indices = list(range(offset, batch_end))
            n_batch = len(batch_indices)

            # Build all subsets: C(n,1) + C(n,2) + C(n,3)
            all_subsets = []
            for order in range(1, min(4, n_batch + 1)):
                all_subsets.extend(combos(batch_indices, order))

            # Each subset is combined with previously accepted features
            n_combos = len(all_subsets)
            if n_combos == 0:
                break

            with torch.no_grad():
                h_batch = h_row.unsqueeze(0).expand(n_combos, -1).clone()
                for c, subset in enumerate(all_subsets):
                    for j in accepted_combo:
                        h_batch[c, ranked[j]] = 0.0
                    for j in subset:
                        h_batch[c, ranked[j]] = 0.0
                recon_batch = saes[strong].decode(h_batch)
                deltas = (recon_batch - recon_full) * data_std_t_s.unsqueeze(0)

            if use_mitra:
                cand_preds = batched_ablation(tail_s, X_row, deltas, max_K=max_K)
            elif use_sequential:
                cand_preds = batched_ablation_sequential(tail_s, X_row, deltas, query_idx=r)
            else:
                cand_preds = batched_ablation(tail_s, X_row, deltas, max_K=max_K)

            # Find best subset
            best_c = None
            best_c_dist = current_dist
            for c in range(n_combos):
                d = dist_to_weak(cand_preds[c])
                if d < best_c_dist:
                    best_c = c
                    best_c_dist = d

            if best_c is not None:
                accepted_combo = tuple(accepted_combo) + all_subsets[best_c]
                current_dist = best_c_dist
                best_pred = cand_preds[best_c]

            # Move to next batch of features
            offset = batch_end

        accepted_k = len(accepted_combo)
        # Map back to global feature indices
        selected_features_r = [ranked[j] for j in accepted_combo] if accepted_combo else []

        if accepted_k > 0:
            optimal_k[r] = accepted_k
            if baseline_preds_s.ndim == 2:
                # Classification: signed distance in loss space
                best_loss = -np.log(np.clip(best_pred[y_r], eps, 1 - eps))
                gap = target_dist - orig_dist       # positive (strong < weak loss)
                moved = best_loss - orig_dist       # positive = toward weak
                gap_closed[r] = min(1.0, max(0.0, moved / gap)) if gap > 1e-8 else 1.0
            else:
                gap = abs(orig_dist - target_dist)
                moved = abs(current_dist - orig_dist)
                gap_closed[r] = min(1.0, moved / gap) if gap > 1e-8 else 1.0
            preds_intervened[r] = best_pred
            selected_features[r] = selected_features_r
        else:
            optimal_k[r] = 0
            gap_closed[r] = 0.0

        if (r + 1) % 50 == 0 or r == n_query - 1:
            elapsed = time.time() - t0
            rate = (r + 1) / elapsed
            eta = (n_query - r - 1) / rate if rate > 0 else 0
            valid = optimal_k[:r+1][strong_wins[:r+1]]
            mean_k = valid.mean() if len(valid) else 0
            logger.info(f"    row {r+1}/{n_query}: mean_optimal_k={mean_k:.1f} "
                        f"({rate:.1f} rows/s, ETA {eta:.0f}s)")

    logger.info(f"  Done in {time.time() - t0:.1f}s")

    valid_k = optimal_k[strong_wins]
    valid_gc = gap_closed[strong_wins]

    # Pad selected features to fixed width for NPZ storage
    # -1 = unused slot
    max_selected = max((len(sf) for sf in selected_features), default=0)
    if max_selected > 0:
        selected_arr = np.full((n_query, max_selected), -1, dtype=np.int32)
        for r, sf in enumerate(selected_features):
            for j, fi in enumerate(sf):
                selected_arr[r, j] = fi
    else:
        selected_arr = np.array([], dtype=np.int32)

    return {
        "strong_model": strong,
        "weak_model": weak,
        "optimal_k": optimal_k,
        "gap_closed": gap_closed,
        "strong_wins": strong_wins,
        "preds_strong": baseline_preds_s,
        "preds_weak": weak_preds,
        "preds_intervened": preds_intervened,
        "baseline_loss_strong": baseline_loss_s,
        "baseline_loss_weak": weak_loss,
        "n_query": n_query,
        "n_strong_wins": n_strong_wins,
        "mean_optimal_k": float(valid_k.mean()) if len(valid_k) else 0.0,
        "median_optimal_k": float(np.median(valid_k)) if len(valid_k) else 0.0,
        "mean_gap_closed": float(valid_gc.mean()) if len(valid_gc) else 0.0,
        "metric_strong": float(metric_strong),
        "metric_weak": float(metric_weak),
        "metric_name": metric_name,
        "feature_indices": feature_indices,
        "selected_features": selected_arr,
        "y_query": y_query.astype(np.float32),
        "row_indices": row_indices.astype(np.int32),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model ablation: how many concepts separate two models?")
    parser.add_argument("--models", nargs=2, required=True, metavar="MODEL",
                        help="Two models to compare (order does not matter)")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-K", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=10000)
    args = parser.parse_args()

    model_a, model_b = sorted(args.models)
    pair_name = f"{model_a}_vs_{model_b}"

    splits = json.loads(SPLITS_PATH.read_text())

    # Load SAEs and norm stats for both models
    saes = {}
    norm_stats = {}
    test_embeddings = {}
    for m in (model_a, model_b):
        sae, _ = load_sae(m, device=args.device)
        sae.eval()
        saes[m] = sae
        norm_stats[m] = load_norm_stats_matching(m)
        test_embeddings[m] = load_test_embeddings(m)

    # Load unmatched features for both directions
    unmatched = {}
    for source, target in [(model_a, model_b), (model_b, model_a)]:
        key = f"{source}__{target}"
        unmatched[key] = get_unmatched_features(source, target)
        logger.info(f"  {key}: {len(unmatched[key])} unmatched features")

    # Find datasets where both models have importance data
    ds_a = set(d.stem for d in (IMPORTANCE_DIR / model_a).glob("*.npz"))
    ds_b = set(d.stem for d in (IMPORTANCE_DIR / model_b).glob("*.npz"))
    available = sorted(ds_a & ds_b)

    if args.datasets:
        datasets = [d for d in available if d in args.datasets]
    else:
        datasets = available

    out_dir = OUTPUT_DIR / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Ablation sweep: {model_a} vs {model_b}")
    logger.info(f"  Datasets: {len(datasets)} (of {len(ds_a)} / {len(ds_b)} with importance)")
    logger.info(f"  Max steps: {args.max_steps}")

    for i, ds in enumerate(datasets):
        out_path = out_dir / f"{ds}.npz"
        if args.resume and out_path.exists():
            logger.info(f"[{i+1}/{len(datasets)}] {ds}: SKIP (exists)")
            continue

        logger.info(f"\n[{i+1}/{len(datasets)}] {ds}")

        # Need norm stats for at least one model (the strong one, determined later).
        # Skip only if NEITHER model has norm stats for this dataset.
        if ds not in norm_stats[model_a] and ds not in norm_stats[model_b]:
            logger.info(f"  SKIP (missing norm stats for both models)")
            continue

        try:
            result = run_dataset(
                model_a, model_b, ds,
                saes, splits, norm_stats, test_embeddings,
                unmatched, args.device, args.max_K, args.max_steps,
            )
            np.savez_compressed(str(out_path), **result)

            if result["n_strong_wins"] > 0:
                logger.info(f"  -> {out_path.name}: {result['strong_model']}>{result['weak_model']}, "
                            f"{result['n_strong_wins']} rows, "
                            f"mean_k={result['mean_optimal_k']:.1f}, "
                            f"gap_closed={result['mean_gap_closed']:.2f}")
            else:
                logger.info(f"  -> {out_path.name}: models tied (no rows with clear winner)")

        except Exception as e:
            logger.error(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
