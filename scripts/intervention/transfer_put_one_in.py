#!/usr/bin/env python3
"""Put-one-in transfer: single-feature injection to validate importance.

For each unmatched feature that fires on each row, injects ONLY that
feature (mapped via adaptive local concept map) and measures the
prediction improvement. No greedy search, no combinations.

Produces a per-feature, per-row "transfer importance" that can be
correlated against the ablation importance to validate whether
leave-one-out (removal) and put-one-in (injection) agree.

Usage:
    python -m scripts.intervention.transfer_put_one_in --models mitra tabpfn --device cuda
    python -m scripts.intervention.transfer_put_one_in --models mitra tabpfn --datasets credit-g
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
    batched_intervention, batched_intervention_sequential,
    MitraTail, SEQUENTIAL_MODELS,
    MODEL_KEY_TO_LABEL_KEY,
)
from scripts.matching.utils import load_norm_stats as load_norm_stats_matching
from scripts.intervention.transfer_virtual_nodes import (
    extract_decoder_atoms,
    filter_landmarks,
)
from scripts.intervention.ablation_sweep import get_unmatched_features

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

IMPORTANCE_DIR = PROJECT_ROOT / "output" / "perrow_importance"
OUTPUT_DIR = PROJECT_ROOT / "output" / "transfer_put_one_in"


def run_dataset(
    model_a: str, model_b: str, dataset: str,
    saes: dict, splits: dict, norm_stats: dict, test_embeddings: dict,
    matched_pairs: dict, unmatched_features: dict, device: str,
) -> dict:
    """Put-one-in transfer for one dataset."""

    # Load cached predictions
    imp_a = np.load(IMPORTANCE_DIR / model_a / f"{dataset}.npz", allow_pickle=True)
    imp_b = np.load(IMPORTANCE_DIR / model_b / f"{dataset}.npz", allow_pickle=True)
    preds_a = imp_a["baseline_preds"]
    preds_b = imp_b["baseline_preds"]
    y_query = imp_a["y_query"]
    n_query = len(y_query)
    task = "classification" if preds_a.ndim == 2 else "regression"

    metric_a, metric_name = compute_importance_metric(y_query, preds_a, task)
    metric_b, _ = compute_importance_metric(y_query, preds_b, task)

    if metric_name == "degenerate":
        return None

    if metric_a >= metric_b:
        strong, weak = model_a, model_b
        metric_strong, metric_weak = metric_a, metric_b
    else:
        strong, weak = model_b, model_a
        metric_strong, metric_weak = metric_b, metric_a

    preds_strong = preds_a if strong == model_a else preds_b
    preds_weak = preds_a if weak == model_a else preds_b
    loss_strong = compute_per_row_loss(y_query, preds_strong, task)
    loss_weak = compute_per_row_loss(y_query, preds_weak, task)
    strong_wins = loss_strong < loss_weak
    n_strong_wins = int(strong_wins.sum())

    logger.info(f"  {strong} > {weak}, {n_strong_wins}/{n_query} strong wins")

    if n_strong_wins == 0:
        return None

    # Build weak model tail
    t0 = time.time()
    X_train_w, y_train_w, X_query_w, _, _, task_w = load_dataset_context(
        weak, dataset, splits)
    if y_train_w.dtype == np.int32:
        y_train_w = y_train_w.astype(np.int64)
    layer_w = get_extraction_layer_taskaware(weak, dataset=dataset)
    cat_indices = None
    if weak in ("hyperfast", "tabpfn"):
        from data.preprocessing import load_preprocessed, CACHE_DIR
        try:
            pre = load_preprocessed(weak, dataset, CACHE_DIR)
            cat_indices = pre.cat_indices if pre.cat_indices else None
        except Exception:
            pass
    target_name = splits.get(dataset, {}).get("target", "target")
    tail_w = build_tail(weak, X_train_w, y_train_w, X_query_w, layer_w, task_w,
                        device, cat_indices=cat_indices, target_name=target_name)
    logger.info(f"  Weak tail built in {time.time() - t0:.1f}s")

    # Decoder atoms and SAE activations
    atoms_strong = extract_decoder_atoms(saes[strong]).numpy()
    atoms_weak = extract_decoder_atoms(saes[weak]).numpy()

    emb_s = test_embeddings[strong][dataset]
    with torch.no_grad():
        h_strong = saes[strong].encode(
            torch.tensor(emb_s, dtype=torch.float32, device=device)
        ).cpu().numpy()

    ds_mean_w, ds_std_w = norm_stats[weak][dataset]
    ds_std_w = np.array(ds_std_w)

    # Build adaptive local virtual atoms
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.linear_model import Ridge

    pair_key = f"{strong}_to_{weak}"
    m_pairs = matched_pairs.get(pair_key, [])
    unmatched = unmatched_features.get(pair_key, [])
    unmatched_set = set(unmatched)

    if len(m_pairs) < 5:
        return None

    matched_src_atoms = atoms_strong[[si for si, _ in m_pairs]]
    matched_tgt_atoms = atoms_weak[[ti for _, ti in m_pairs]]

    filt_src, filt_tgt, filt_pairs, quality = filter_landmarks(
        matched_src_atoms, matched_tgt_atoms, m_pairs,
        min_cosine=0.0, alpha=1.0,
    )

    filt_src_indices = [si for si, _ in filt_pairs]
    filt_tgt_indices = [ti for _, ti in filt_pairs]
    filt_src_atoms = atoms_strong[filt_src_indices]
    filt_tgt_atoms = atoms_weak[filt_tgt_indices]

    src_norms = np.linalg.norm(filt_src_atoms, axis=1, keepdims=True)
    tgt_norms = np.linalg.norm(filt_tgt_atoms, axis=1, keepdims=True)
    src_norms[src_norms < 1e-8] = 1.0
    tgt_norms[tgt_norms < 1e-8] = 1.0
    filt_src_unit = filt_src_atoms / src_norms
    filt_tgt_unit = filt_tgt_atoms / tgt_norms

    noise_floor = 0.20
    sim_threshold = 2.0 * noise_floor

    # Pre-compute virtual atoms for all unmatched features
    virtual_atoms = {}
    for fi in unmatched:
        atom_s = atoms_strong[fi]
        atom_norm = np.linalg.norm(atom_s)
        if atom_norm < 1e-8:
            continue
        query_unit = (atom_s / atom_norm).reshape(1, -1)
        sims = cosine_similarity(query_unit, filt_src_unit)[0]
        mask = np.abs(sims) >= sim_threshold
        K = int(mask.sum())
        if K < 3:
            continue
        idx = np.where(mask)[0]
        reg = Ridge(alpha=1.0, fit_intercept=False)
        reg.fit(filt_src_unit[idx], filt_tgt_unit[idx])
        direction = reg.predict(query_unit)[0]
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-8:
            continue
        local_ratios = tgt_norms[idx].ravel() / src_norms[idx].ravel()
        scale = float(np.median(local_ratios)) * atom_norm
        virtual_atoms[fi] = (direction / dir_norm) * scale

    logger.info(f"  Virtual atoms: {len(virtual_atoms)}/{len(unmatched)}")

    # Load ablation importance for comparison
    imp_strong = np.load(IMPORTANCE_DIR / strong / f"{dataset}.npz", allow_pickle=True)
    feature_indices = imp_strong["feature_indices"]
    row_feature_drops = imp_strong["row_feature_drops"]

    d_target = atoms_weak.shape[1]
    use_sequential = isinstance(tail_w, SEQUENTIAL_MODELS)
    use_mitra = isinstance(tail_w, MitraTail)

    # Per-row, per-feature: inject one feature at a time, measure improvement
    # Store as sparse: list of (row, feature_idx, ablation_imp, transfer_imp, sign)
    results = []

    t0 = time.time()
    for r in range(n_query):
        if not strong_wins[r]:
            continue

        y_r = int(y_query[r])
        eps = 1e-7

        # Baseline weak prediction loss
        if preds_weak.ndim == 2:
            baseline_loss = -np.log(np.clip(preds_weak[r, y_r], eps, 1 - eps))
            target_loss = -np.log(np.clip(preds_strong[r, y_r], eps, 1 - eps))
        else:
            baseline_loss = (float(preds_weak[r]) - float(preds_strong[r])) ** 2
            target_loss = 0.0

        gap = abs(baseline_loss - target_loss)
        if gap < 1e-8:
            continue

        # Find unmatched features that fire on this row
        firing = []
        for i, fi in enumerate(feature_indices):
            fi = int(fi)
            if fi in unmatched_set and h_strong[r, fi] > 0 and fi in virtual_atoms:
                ablation_imp = float(row_feature_drops[r, i])
                firing.append((fi, ablation_imp))

        if not firing:
            continue

        # Inject each feature individually (both +/-), measure improvement
        X_row = X_query_w[r:r + 1]

        for fi, ablation_imp in firing:
            a_s = float(h_strong[r, fi])
            va = virtual_atoms[fi]
            delta_raw = a_s * va * ds_std_w

            delta_pos = torch.tensor(delta_raw, dtype=torch.float32, device=device).unsqueeze(0)
            delta_neg = torch.tensor(-delta_raw, dtype=torch.float32, device=device).unsqueeze(0)
            deltas = torch.cat([delta_pos, delta_neg], dim=0)

            if use_mitra:
                cands = batched_intervention(tail_w, X_row, deltas, inject_context=False)
            elif use_sequential:
                cands = batched_intervention_sequential(tail_w, X_row, deltas, query_idx=r)
            else:
                cands = batched_intervention(tail_w, X_row, deltas, inject_context=False)

            # Pick the better direction (closer to strong, no overshoot)
            best_improvement = 0.0
            best_sign = 0
            for c, sign in enumerate([+1, -1]):
                if preds_weak.ndim == 2:
                    p_c = cands[c][y_r]
                    c_loss = -np.log(np.clip(p_c, eps, 1 - eps))
                    p_s = preds_strong[r, y_r]
                    p_w = preds_weak[r, y_r]
                    # Overshoot check
                    if p_w < p_s and p_c > p_s:
                        continue
                    if p_w > p_s and p_c < p_s:
                        continue
                    improvement = baseline_loss - c_loss  # positive = better
                else:
                    p_c = float(cands[c])
                    p_s = float(preds_strong[r])
                    p_w = float(preds_weak[r])
                    if p_w < p_s and p_c > p_s:
                        continue
                    if p_w > p_s and p_c < p_s:
                        continue
                    c_loss = (p_c - p_s) ** 2
                    improvement = baseline_loss - c_loss

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_sign = sign

            # Normalize to gap fraction
            transfer_imp = best_improvement / gap if gap > 1e-8 else 0.0

            results.append((r, fi, ablation_imp, transfer_imp, best_sign))

        if (r + 1) % 50 == 0:
            elapsed = time.time() - t0
            logger.info(f"    row {r+1}/{n_query}: {len(results)} measurements "
                        f"({(r+1)/elapsed:.1f} rows/s)")

    logger.info(f"  Done: {len(results)} measurements in {time.time() - t0:.1f}s")

    if not results:
        return None

    # Convert to arrays
    arr = np.array(results, dtype=[
        ('row', np.int32), ('feature_idx', np.int32),
        ('ablation_imp', np.float32), ('transfer_imp', np.float32),
        ('sign', np.int32),
    ])

    # Compute correlation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(arr['ablation_imp'], arr['transfer_imp'])
    logger.info(f"  Spearman(ablation_imp, transfer_imp): rho={rho:.3f}, p={pval:.2e}")

    return {
        "strong_model": strong,
        "weak_model": weak,
        "measurements": arr,
        "n_query": n_query,
        "n_strong_wins": n_strong_wins,
        "n_measurements": len(results),
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "metric_strong": float(metric_strong),
        "metric_weak": float(metric_weak),
        "metric_name": metric_name,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Put-one-in transfer: single-feature injection validation")
    parser.add_argument("--models", nargs=2, required=True)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    model_a, model_b = sorted(args.models)
    pair_name = f"{model_a}_vs_{model_b}"

    splits = json.loads(SPLITS_PATH.read_text())

    saes = {}
    norm_stats = {}
    test_embeddings = {}
    for m in (model_a, model_b):
        sae, _ = load_sae(m, device=args.device)
        sae.eval()
        saes[m] = sae
        norm_stats[m] = load_norm_stats_matching(m)
        test_embeddings[m] = load_test_embeddings(m)

    # Build matched pairs and unmatched for both directions
    labels_data = json.loads(Path(
        PROJECT_ROOT / "output" / "cross_model_concept_labels_round10.json"
    ).read_text())

    matched_pairs = {}
    unmatched_features = {}
    for source, target in [(model_a, model_b), (model_b, model_a)]:
        src_key = MODEL_KEY_TO_LABEL_KEY[source]
        tgt_key = MODEL_KEY_TO_LABEL_KEY[target]
        pairs = []
        for g in labels_data["concept_groups"].values():
            members = g.get("members", [])
            src_feats = [f for m, f in members if m == src_key]
            tgt_feats = [f for m, f in members if m == tgt_key]
            if src_feats and tgt_feats:
                pairs.append((src_feats[0], tgt_feats[0]))
        matched_pairs[f"{source}_to_{target}"] = pairs
        unmatched_features[f"{source}_to_{target}"] = get_unmatched_features(source, target)

    # Find datasets
    ds_a = set(d.stem for d in (IMPORTANCE_DIR / model_a).glob("*.npz"))
    ds_b = set(d.stem for d in (IMPORTANCE_DIR / model_b).glob("*.npz"))
    available = sorted(ds_a & ds_b)
    if args.datasets:
        datasets = [d for d in available if d in args.datasets]
    else:
        datasets = available

    out_dir = (args.output_dir if args.output_dir else OUTPUT_DIR) / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Put-one-in transfer: {pair_name}")
    logger.info(f"  Datasets: {len(datasets)}")

    all_rhos = []
    for i, ds in enumerate(datasets):
        out_path = out_dir / f"{ds}.npz"
        logger.info(f"\n[{i+1}/{len(datasets)}] {ds}")

        try:
            result = run_dataset(
                model_a, model_b, ds,
                saes, splits, norm_stats, test_embeddings,
                matched_pairs, unmatched_features, args.device,
            )
            if result is not None:
                np.savez_compressed(str(out_path), **result)
                all_rhos.append(result["spearman_rho"])
                logger.info(f"  -> {ds}: rho={result['spearman_rho']:.3f}, "
                            f"n={result['n_measurements']}")
        except Exception as e:
            logger.error(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()

    if all_rhos:
        logger.info(f"\nOverall mean Spearman rho: {np.mean(all_rhos):.3f} "
                    f"(n={len(all_rhos)} datasets)")


if __name__ == "__main__":
    main()
