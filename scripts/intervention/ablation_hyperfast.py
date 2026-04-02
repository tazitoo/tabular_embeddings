#!/usr/bin/env python3
"""Context-side ablation for HyperFast.

HyperFast generates MLP weights from context (support) samples via a
hypernetwork.  Query samples pass through the generated MLP but never
participate in weight generation.  To test whether a concept (SAE feature)
matters for HyperFast's advantage, we must intervene on the context
representations that feed the hypernetwork.

Approach:
  1. Fit HyperFast on full context → baseline query predictions
  2. For each ensemble member:
     a. Forward stored context through stored MLP layers 0..L-2 to get
        the penultimate context activations (same 784-dim space as SAE)
     b. Encode with SAE, zero target unmatched features, decode → delta
     c. Apply delta to penultimate context activations
     d. Re-run the final hypernetwork module with modified activations
        to generate new classification layer weights
     e. Forward query through original layers 0..L-2 + new final layer
  3. Average across ensemble members → intervened predictions

This is a proper embedding-space intervention: same row count, same class
balance, only the SAE-identified concept is suppressed in the context
representations that generate the model weights.

Output:
    output/ablation_sweep/{hyperfast_vs_MODEL}/{dataset}.npz

Usage:
    python -m scripts.intervention.ablation_hyperfast --models hyperfast tabpfn --device cuda
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import (
    SPLITS_PATH,
    load_sae, load_test_embeddings,
    compute_per_row_loss, compute_importance_metric,
    get_alive_features, MODEL_KEY_TO_LABEL_KEY, DEFAULT_CONCEPT_LABELS,
)
from scripts.intervention.ablation_sweep import get_unmatched_features
from scripts.matching.utils import load_norm_stats as load_norm_stats_matching

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "output" / "ablation_sweep"
IMPORTANCE_DIR = PROJECT_ROOT / "output" / "perrow_importance"


def _fit_hyperfast(X_context, y_context, device, cat_indices=None):
    """Fit HyperFast, return the fitted classifier."""
    from models.hyperfast_embeddings import HyperFastEmbeddingExtractor

    extractor = HyperFastEmbeddingExtractor(device=device)
    extractor.load_model()
    X_ctx_arr = np.asarray(X_context, dtype=np.float32)
    y_ctx_clean = np.asarray(y_context, dtype=np.int64)
    if cat_indices:
        extractor._model.cat_features = cat_indices
    extractor._model.fit(X_ctx_arr, y_ctx_clean)
    return extractor._model


def _forward_through_layers(main_network, x, n_layers):
    """Forward x through the first n_layers of a stored main_network.

    Replicates the forward_main_network logic but stops early.
    Returns the activations after n_layers (before the next layer).
    """
    for n in range(n_layers):
        if n % 2 == 0:
            residual = x.clone()
        weight, bias = main_network[n]
        x = torch.mm(x, weight) + bias
        if n % 2 == 1 and n != len(main_network) - 1:
            x = x + residual
        if n != len(main_network) - 1:
            x = F.relu(x)
    return x


def _compute_pca_output(X_pca, y, n_classes, max_categories):
    """Compute the pca_output tensor used by hypernetworks.

    This is the concatenation of [row_features, global_mean, class_mean]
    per row, computed from the initial PCA-transformed data.
    """
    pca_global_mean = torch.mean(X_pca, dim=0)
    pca_perclass_mean = []
    for c in range(n_classes):
        mask = (y == c)
        if mask.sum() > 0:
            pca_perclass_mean.append(torch.mean(X_pca[mask], dim=0, keepdim=True))
        else:
            pca_perclass_mean.append(torch.mean(X_pca, dim=0, keepdim=True))
    pca_perclass_mean = torch.cat(pca_perclass_mean)

    pca_concat = []
    for i, lab in enumerate(y):
        lab_idx = lab.item() if torch.is_tensor(lab) else lab
        lab_idx = min(lab_idx, pca_perclass_mean.size(0) - 1)
        row = torch.cat((X_pca[i], pca_global_mean, pca_perclass_mean[lab_idx]))
        pca_concat.append(row)
    return torch.vstack(pca_concat)


def _intervened_predict(clf, X_query, sae, ds_mean, ds_std,
                        features_to_zero, device):
    """Predict query with modified context at SAE extraction point.

    For each ensemble member:
    1. Replay context through stored MLP layers 0..L-2 → penultimate out
    2. Apply SAE delta (zero features) to context out
    3. Re-generate final layer via last hypernetwork module
    4. Forward query through original layers 0..L-2 + new final layer
    """
    from hyperfast.hyperfast import (
        forward_main_network, transform_data_for_main_network,
    )
    from hyperfast.utils import get_main_weights

    hf_model = clf._model  # The HyperFast nn.Module with hypernetworks
    n_classes = clf.n_classes_
    max_categories = clf._cfg.max_categories
    n_dims = clf._cfg.n_dims

    X_query_t = clf._preprocess_test_data(X_query).to(device)
    ds_mean_t = torch.tensor(ds_mean, dtype=torch.float32, device=device)
    ds_std_t = torch.tensor(ds_std, dtype=torch.float32, device=device)

    all_preds = []

    for jj in range(len(clf._main_networks)):
        main_network = clf._move_to_device(clf._main_networks[jj])
        rf = clf._move_to_device(clf._rfs[jj])
        pca = clf._move_to_device(clf._pcas[jj])
        X_pred = clf._X_preds[jj].to(device)
        y_pred = clf._y_preds[jj].to(device)

        # Handle feature bagging
        if clf.feature_bagging:
            X_pred_b = X_pred[:, clf.selected_features[jj]]
            X_q_b = X_query_t[:, clf.selected_features[jj]]
        else:
            X_pred_b = X_pred
            X_q_b = X_query_t

        # 1. Transform context through RF+PCA
        X_pca = transform_data_for_main_network(X_pred_b, clf._cfg, rf, pca)

        # Compute pca_output (fixed, from initial PCA output)
        pca_output = _compute_pca_output(X_pca, y_pred, n_classes, max_categories)
        y_onehot = F.one_hot(y_pred, max_categories).float()

        # 2. Forward context through stored MLP layers 0..L-2
        n_total_layers = len(main_network)
        with torch.no_grad():
            ctx_out = _forward_through_layers(main_network, X_pca.clone(),
                                              n_total_layers - 1)

        # 3. Apply SAE delta to context penultimate activations
        with torch.no_grad():
            ctx_normed = (ctx_out - ds_mean_t) / (ds_std_t + 1e-8)
            h = sae.encode(ctx_normed)
            h_modified = h.clone()
            for fi in features_to_zero:
                h_modified[:, fi] = 0.0
            recon_orig = sae.decode(h)
            recon_mod = sae.decode(h_modified)
            delta = (recon_mod - recon_orig) * ds_std_t
            ctx_out_modified = ctx_out + delta

        # 4. Re-generate final layer weights using modified context
        with torch.no_grad():
            # Input to last hypernetwork: cat(out, pca_output, y_onehot)
            # But we need to match what forward() computes. In forward(),
            # by the last layer, data includes out (current), pca_output,
            # and y_onehot. The pca_output also gets out concatenated
            # starting from layer 1+.
            #
            # For the final layer in forward():
            #   data = torch.cat((out, pca_output, y_onehot), dim=1)
            #   weights_per_sample = get_main_weights(data, hypernetworks[-1])
            #
            # pca_output already has initial PCA features, so data is:
            #   [penultimate_out, X_pca_row, global_mean, class_mean, y_onehot]
            data = torch.cat((ctx_out_modified, pca_output, y_onehot), dim=1)

            last_hn = hf_model.hypernetworks[-1].to(device)
            weights_per_sample = last_hn(data)  # No weight_gen for last layer

            # Per-class pooling of weights (replicates forward() logic)
            weight_parts = []
            last_input_mean = []
            for c in range(n_classes):
                mask = (y_pred == c)
                if mask.sum() > 0:
                    w = torch.mean(weights_per_sample[mask], dim=0, keepdim=True)
                    im = torch.mean(ctx_out_modified[mask], dim=0, keepdim=True)
                else:
                    w = torch.mean(weights_per_sample, dim=0, keepdim=True)
                    im = torch.mean(ctx_out_modified, dim=0, keepdim=True)
                weight_parts.append(w)
                last_input_mean.append(im)
            weights_cat = torch.cat(weight_parts)  # (n_classes, n_dims+1)
            last_input_mean_cat = torch.cat(last_input_mean)  # (n_classes, n_dims)

            # Add input mean to weight columns (not bias)
            weights_cat[:, :-1] = weights_cat[:, :-1] + last_input_mean_cat
            weights_final = weights_cat.T  # (n_dims+1, n_classes)

            # Extract weight matrix and bias
            W_final = weights_final[:-1, :]  # (n_dims, n_classes)
            b_final = weights_final[-1, :]   # (n_classes,)

        # 5. Forward query through original layers 0..L-2 + new final layer
        with torch.no_grad():
            X_q_transformed = transform_data_for_main_network(
                X_q_b, clf._cfg, rf, pca)
            q_out = _forward_through_layers(main_network, X_q_transformed,
                                            n_total_layers - 1)
            # Apply modified final layer
            logits = torch.mm(q_out, W_final) + b_final
            preds = F.softmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)

    return np.mean(all_preds, axis=0)


def _baseline_predict(clf, X_query, device):
    """Standard HyperFast prediction (no intervention)."""
    from hyperfast.hyperfast import forward_main_network, transform_data_for_main_network

    X_query_t = clf._preprocess_test_data(X_query).to(device)
    all_preds = []

    for jj in range(len(clf._main_networks)):
        main_network = clf._move_to_device(clf._main_networks[jj])
        rf = clf._move_to_device(clf._rfs[jj])
        pca = clf._move_to_device(clf._pcas[jj])

        if clf.feature_bagging:
            X_b = X_query_t[:, clf.selected_features[jj]]
        else:
            X_b = X_query_t

        X_transformed = transform_data_for_main_network(X_b, clf._cfg, rf, pca)
        with torch.no_grad():
            out, _ = forward_main_network(X_transformed, main_network)
            all_preds.append(F.softmax(out, dim=1).cpu().numpy())

    return np.mean(all_preds, axis=0)


def _get_context_embeddings(clf, device):
    """Get penultimate activations for context rows (same space as SAE)."""
    from hyperfast.hyperfast import transform_data_for_main_network

    all_intermediates = []
    for jj in range(len(clf._main_networks)):
        main_network = clf._move_to_device(clf._main_networks[jj])
        rf = clf._move_to_device(clf._rfs[jj])
        pca = clf._move_to_device(clf._pcas[jj])
        X_pred = clf._X_preds[jj].to(device)

        if clf.feature_bagging:
            X_b = X_pred[:, clf.selected_features[jj]]
        else:
            X_b = X_pred

        X_pca = transform_data_for_main_network(X_b, clf._cfg, rf, pca)
        n_layers = len(main_network)
        with torch.no_grad():
            out = _forward_through_layers(main_network, X_pca, n_layers - 1)
            all_intermediates.append(out.cpu().numpy())

    return np.mean(all_intermediates, axis=0)


def run_dataset(
    model_a: str, model_b: str, dataset: str,
    sae_hf, splits: dict, norm_stats_hf: dict,
    device: str, unmatched_hf: list, max_features: int = 50,
) -> dict:
    """Context-side ablation for one dataset."""
    imp_a = np.load(IMPORTANCE_DIR / model_a / f"{dataset}.npz", allow_pickle=True)
    imp_b = np.load(IMPORTANCE_DIR / model_b / f"{dataset}.npz", allow_pickle=True)
    preds_a = imp_a["baseline_preds"]
    preds_b = imp_b["baseline_preds"]
    y_query = imp_a["y_query"]
    n_query = len(y_query)
    task = "classification" if preds_a.ndim == 2 else "regression"

    metric_a, metric_name = compute_importance_metric(y_query, preds_a, task)
    metric_b, _ = compute_importance_metric(y_query, preds_b, task)

    if metric_name == "degenerate" or metric_a == float("-inf") or metric_b == float("-inf"):
        logger.info(f"  SKIP (degenerate)")
        return {"strong_model": model_a, "weak_model": model_b,
                "n_strong_wins": 0, "n_query": n_query,
                "metric_strong": 0.0, "metric_weak": 0.0, "metric_name": "degenerate"}

    if metric_a >= metric_b:
        strong, weak = model_a, model_b
        metric_strong, metric_weak = metric_a, metric_b
    else:
        strong, weak = model_b, model_a
        metric_strong, metric_weak = metric_b, metric_a

    logger.info(f"  {strong} ({metric_name}={metric_strong:.4f}) > "
                f"{weak} ({metric_name}={metric_weak:.4f})")

    if strong != "hyperfast":
        logger.info(f"  SKIP (HyperFast is weak)")
        return {"strong_model": strong, "weak_model": weak,
                "n_strong_wins": 0, "n_query": n_query,
                "metric_strong": float(metric_strong),
                "metric_weak": float(metric_weak), "metric_name": metric_name}

    preds_strong = preds_a if strong == model_a else preds_b
    preds_weak = preds_a if weak == model_a else preds_b
    loss_strong = compute_per_row_loss(y_query, preds_strong, task)
    loss_weak = compute_per_row_loss(y_query, preds_weak, task)
    strong_wins = loss_strong < loss_weak
    n_strong_wins = int(strong_wins.sum())
    logger.info(f"  Strong wins on {n_strong_wins}/{n_query} rows")

    if n_strong_wins == 0:
        return {"strong_model": strong, "weak_model": weak,
                "n_strong_wins": 0, "n_query": n_query,
                "metric_strong": float(metric_strong),
                "metric_weak": float(metric_weak), "metric_name": metric_name}

    # Fit HyperFast
    from scripts.intervention.intervene_lib import load_dataset_context
    X_train, y_train, X_query_data, _, _, _ = load_dataset_context(
        strong, dataset, splits)
    if y_train.dtype == np.int32:
        y_train = y_train.astype(np.int64)

    cat_indices = None
    from data.preprocessing import load_preprocessed, CACHE_DIR
    try:
        pre = load_preprocessed("hyperfast", dataset, CACHE_DIR)
        cat_indices = pre.cat_indices if pre.cat_indices else None
    except Exception:
        pass

    t0 = time.time()
    clf = _fit_hyperfast(X_train, y_train, device, cat_indices)
    logger.info(f"  Fit in {time.time() - t0:.1f}s")

    # Get norm stats
    if dataset not in norm_stats_hf:
        logger.info(f"  SKIP (no norm stats)")
        return {"strong_model": strong, "weak_model": weak,
                "n_strong_wins": 0, "n_query": n_query,
                "metric_strong": float(metric_strong),
                "metric_weak": float(metric_weak), "metric_name": metric_name}

    ds_mean, ds_std = norm_stats_hf[dataset]

    # Get context embeddings and find which SAE features fire
    ctx_emb = _get_context_embeddings(clf, device)
    ctx_normed = (ctx_emb - ds_mean) / (ds_std + 1e-8)
    with torch.no_grad():
        ctx_acts = sae_hf.encode(
            torch.tensor(ctx_normed, dtype=torch.float32, device=device))
    ctx_firing = (ctx_acts > 0).cpu().numpy()

    unmatched_set = set(unmatched_hf)
    feature_ctx_rows = {}
    for fi in unmatched_hf:
        rows = np.where(ctx_firing[:, fi])[0]
        if len(rows) > 0:
            feature_ctx_rows[fi] = rows

    logger.info(f"  {len(feature_ctx_rows)} unmatched features fire on context")

    # Rank by importance
    imp_hf = np.load(IMPORTANCE_DIR / "hyperfast" / f"{dataset}.npz", allow_pickle=True)
    feature_indices = imp_hf["feature_indices"]
    row_drops = imp_hf["row_feature_drops"]
    mean_imp = row_drops[strong_wins].mean(axis=0) if strong_wins.any() else row_drops.mean(axis=0)

    fi_to_imp = {}
    for i, fi in enumerate(feature_indices):
        fi = int(fi)
        if fi in unmatched_set and fi in feature_ctx_rows and mean_imp[i] > 0:
            fi_to_imp[fi] = mean_imp[i]

    ranked_features = sorted(fi_to_imp.keys(), key=lambda f: -fi_to_imp[f])
    ranked_features = ranked_features[:max_features]
    logger.info(f"  Testing top {len(ranked_features)} features")

    # Per-row distances
    eps = 1e-7
    if preds_strong.ndim == 2:
        def row_dist(preds, r):
            return -np.log(np.clip(preds[r, int(y_query[r])], eps, 1 - eps))
    else:
        def row_dist(preds, r):
            return float((preds[r] - preds_weak.ravel()[r]) ** 2)

    orig_dists = np.array([row_dist(preds_strong, r) for r in range(n_query)])
    target_dists = np.array([row_dist(preds_weak, r) for r in range(n_query)])

    # Greedy feature ablation
    optimal_k = np.zeros(n_query, dtype=np.int32)
    gap_closed = np.full(n_query, np.nan, dtype=np.float32)
    gap_closed[~strong_wins] = 1.0
    preds_intervened = preds_strong.copy()
    accepted_features = []

    t0 = time.time()
    for step, fi in enumerate(ranked_features):
        # Try adding this feature to the ablation set
        test_features = accepted_features + [fi]

        try:
            new_preds = _intervened_predict(
                clf, X_query_data, sae_hf, ds_mean, ds_std,
                test_features, device)
        except Exception as e:
            logger.warning(f"    Feature {fi}: intervention failed: {e}")
            continue

        # Check if gap_closed improved
        improved = False
        for r in range(n_query):
            if not strong_wins[r]:
                continue
            gap = abs(orig_dists[r] - target_dists[r])
            if gap < 1e-8:
                continue
            old_gc = 0.0 if np.isnan(gap_closed[r]) else gap_closed[r]
            new_dist = row_dist(new_preds, r)
            moved = abs(new_dist - orig_dists[r])
            new_gc = min(1.0, moved / gap)
            if new_gc > old_gc:
                improved = True
                break

        if improved:
            accepted_features.append(fi)
            preds_intervened = new_preds.copy()
            for r in range(n_query):
                if not strong_wins[r]:
                    continue
                gap = abs(orig_dists[r] - target_dists[r])
                if gap < 1e-8:
                    gap_closed[r] = 1.0
                    continue
                cur_dist = row_dist(preds_intervened, r)
                moved = abs(cur_dist - orig_dists[r])
                gap_closed[r] = min(1.0, moved / gap)
            optimal_k[strong_wins] = len(accepted_features)

            mean_gc = np.nanmean(gap_closed[strong_wins])
            logger.info(f"    Step {step+1}: accepted f{fi}, "
                        f"total={len(accepted_features)}, gc={mean_gc:.3f}")

    elapsed = time.time() - t0
    gap_closed[np.isnan(gap_closed) & strong_wins] = 0.0
    valid_gc = gap_closed[strong_wins]
    valid_k = optimal_k[strong_wins]

    logger.info(f"  Done in {elapsed:.1f}s, "
                f"accepted {len(accepted_features)} features, "
                f"gc={np.nanmean(valid_gc):.3f}")

    sel_arr = (np.array(accepted_features, dtype=np.int32)
               if accepted_features else np.array([], dtype=np.int32))

    return {
        "strong_model": strong, "weak_model": weak,
        "optimal_k": optimal_k, "gap_closed": gap_closed,
        "strong_wins": strong_wins,
        "preds_strong": preds_strong, "preds_weak": preds_weak,
        "preds_intervened": preds_intervened,
        "baseline_loss_strong": loss_strong, "baseline_loss_weak": loss_weak,
        "n_query": n_query, "n_strong_wins": n_strong_wins,
        "mean_optimal_k": float(valid_k.mean()) if len(valid_k) else 0.0,
        "median_optimal_k": float(np.median(valid_k)) if len(valid_k) else 0.0,
        "mean_gap_closed": float(np.nanmean(valid_gc)) if len(valid_gc) else 0.0,
        "metric_strong": float(metric_strong),
        "metric_weak": float(metric_weak), "metric_name": metric_name,
        "selected_features": sel_arr,
        "y_query": y_query.astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser(
        description="HyperFast context-side ablation")
    parser.add_argument("--models", nargs=2, required=True, metavar="MODEL",
                        help="hyperfast + one other model")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-features", type=int, default=50)
    args = parser.parse_args()

    model_a, model_b = sorted(args.models)
    if "hyperfast" not in (model_a, model_b):
        raise ValueError("One model must be 'hyperfast'")
    pair_name = f"{model_a}_vs_{model_b}"
    other = model_b if model_a == "hyperfast" else model_a

    splits = json.loads(SPLITS_PATH.read_text())

    sae_hf, _ = load_sae("hyperfast", device=args.device)
    sae_hf.eval()
    norm_stats_hf = load_norm_stats_matching("hyperfast")

    unmatched_hf = get_unmatched_features("hyperfast", other)
    logger.info(f"HyperFast unmatched vs {other}: {len(unmatched_hf)} features")

    ds_a = set(d.stem for d in (IMPORTANCE_DIR / model_a).glob("*.npz"))
    ds_b = set(d.stem for d in (IMPORTANCE_DIR / model_b).glob("*.npz"))
    available = sorted(ds_a & ds_b)

    if args.datasets:
        datasets = [d for d in available if d in args.datasets]
    else:
        datasets = available

    out_dir = OUTPUT_DIR / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"HyperFast context-side ablation: {pair_name}")
    logger.info(f"  Datasets: {len(datasets)}")

    for i, ds in enumerate(datasets):
        out_path = out_dir / f"{ds}.npz"
        if args.resume and out_path.exists():
            logger.info(f"[{i+1}/{len(datasets)}] {ds}: SKIP (exists)")
            continue

        logger.info(f"\n[{i+1}/{len(datasets)}] {ds}")

        try:
            result = run_dataset(
                model_a, model_b, ds, sae_hf, splits, norm_stats_hf,
                args.device, unmatched_hf, args.max_features,
            )
            np.savez_compressed(str(out_path), **result)

            if result["n_strong_wins"] > 0:
                logger.info(f"  -> {out_path.name}: "
                            f"{result['strong_model']}>{result['weak_model']}, "
                            f"{result['n_strong_wins']} rows, "
                            f"gc={result['mean_gap_closed']:.2f}")
            else:
                logger.info(f"  -> {out_path.name}: HyperFast weak or tied")

        except Exception as e:
            logger.error(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
