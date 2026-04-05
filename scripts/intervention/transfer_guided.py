#!/usr/bin/env python3
"""Guided transfer: greedy search restricted to ablation-identified concepts.

Combines the best of both approaches:
- Ablation identifies WHICH concepts matter (the candidate pool)
- Greedy search validates WHICH of those transfer successfully

This tests whether ablation correctly identified the relevant concepts:
if guided gc > full-search gc, ablation narrows the search to better
candidates. If guided gc < full-search gc, the greedy search was
finding different concepts than ablation.

Usage:
    python -m scripts.intervention.transfer_guided --models mitra tabpfn --device cuda
    python -m scripts.intervention.transfer_guided --models mitra tabpfn --datasets credit-g
"""
import argparse
import json
import logging
import time
from collections import Counter
from itertools import combinations as combos
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

ABLATION_DIR = PROJECT_ROOT / "output" / "ablation_sweep"
IMPORTANCE_DIR = PROJECT_ROOT / "output" / "perrow_importance"
OUTPUT_DIR = PROJECT_ROOT / "output" / "transfer_guided"


def run_dataset(
    model_a: str, model_b: str, dataset: str,
    saes: dict, splits: dict, norm_stats: dict, test_embeddings: dict,
    matched_pairs: dict, device: str, max_steps: int = 64,
) -> dict:
    """Guided transfer for one dataset."""

    # Load ablation results
    pair_name = f"{min(model_a, model_b)}_vs_{max(model_a, model_b)}"
    ablation_path = ABLATION_DIR / pair_name / f"{dataset}.npz"
    if not ablation_path.exists():
        logger.info(f"  SKIP (no ablation results)")
        return None

    abl = np.load(ablation_path, allow_pickle=True)
    strong = str(abl["strong_model"])
    weak = str(abl["weak_model"])
    strong_wins = abl["strong_wins"]
    n_query = int(abl["n_query"])
    n_strong_wins = int(strong_wins.sum())

    if n_strong_wins == 0:
        return None

    selected_features_abl = abl["selected_features"]
    if selected_features_abl.size == 0:
        return None

    preds_strong = abl["preds_strong"]
    preds_weak = abl["preds_weak"]
    y_query = abl["y_query"]
    task = "classification" if preds_strong.ndim == 2 else "regression"

    metric_strong = float(abl["metric_strong"])
    metric_weak = float(abl["metric_weak"])
    metric_name = str(abl["metric_name"])

    logger.info(f"  {strong} > {weak}, {n_strong_wins}/{n_query} strong wins")

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

    # Build adaptive local virtual atoms for ablation-selected features only
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.linear_model import Ridge

    pair_key = f"{strong}_to_{weak}"
    m_pairs = matched_pairs.get(pair_key, [])
    if len(m_pairs) < 5:
        logger.info(f"  SKIP (too few matched pairs)")
        return None

    matched_src_atoms = atoms_strong[[si for si, _ in m_pairs]]
    matched_tgt_atoms = atoms_weak[[ti for _, ti in m_pairs]]

    filt_src, filt_tgt, filt_pairs, quality = filter_landmarks(
        matched_src_atoms, matched_tgt_atoms, m_pairs,
        min_cosine=0.0, alpha=1.0,
    )
    logger.info(f"  Landmarks: {quality['n_kept']}/{quality['n_input']}")

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

    # Get per-row ablation concepts and build virtual atoms for only those
    # Pre-compute virtual atoms for all unique ablation features
    all_abl_features = set()
    for r in range(n_query):
        if strong_wins[r] and selected_features_abl.ndim == 2:
            for fi in selected_features_abl[r]:
                if fi >= 0:
                    all_abl_features.add(int(fi))

    virtual_atoms = {}
    for fi in all_abl_features:
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

    logger.info(f"  Ablation features: {len(all_abl_features)}, "
                f"virtual atoms: {len(virtual_atoms)}")

    d_target = atoms_weak.shape[1]
    use_sequential = isinstance(tail_w, SEQUENTIAL_MODELS)
    use_mitra = isinstance(tail_w, MitraTail)

    # Per-row greedy search over ONLY the ablation-selected features
    optimal_k = np.zeros(n_query, dtype=np.int32)
    gap_closed = np.full(n_query, np.nan, dtype=np.float32)
    baseline_preds_w = preds_weak.copy()
    preds_intervened = preds_weak.copy()

    feature_tried = Counter()
    feature_accepted = Counter()

    t0 = time.time()
    for r in range(n_query):
        if not strong_wins[r]:
            optimal_k[r] = 0
            gap_closed[r] = 1.0
            continue

        # This row's ablation-selected features (the guided candidate pool)
        if selected_features_abl.ndim == 2:
            row_candidates = [int(fi) for fi in selected_features_abl[r]
                              if fi >= 0 and int(fi) in virtual_atoms]
        else:
            row_candidates = []

        if not row_candidates:
            optimal_k[r] = 0
            gap_closed[r] = 0.0
            continue

        # Build +/- deltas for each candidate
        per_feature_deltas = []
        delta_to_feature = []
        for fi in row_candidates:
            a_s = float(h_strong[r, fi])
            if a_s <= 0:
                continue
            va = virtual_atoms[fi]
            delta_raw = a_s * va * ds_std_w
            per_feature_deltas.append(
                torch.tensor(delta_raw, dtype=torch.float32, device=device))
            delta_to_feature.append((fi, +1))
            per_feature_deltas.append(
                torch.tensor(-delta_raw, dtype=torch.float32, device=device))
            delta_to_feature.append((fi, -1))
            feature_tried[fi] += 1

        K = len(per_feature_deltas)
        if K == 0:
            optimal_k[r] = 0
            gap_closed[r] = 0.0
            continue

        # Distance metric
        y_r = int(y_query[r])
        eps = 1e-7
        if baseline_preds_w.ndim == 2:
            target_dist = -np.log(np.clip(preds_strong[r, y_r], eps, 1 - eps))
            def dist_to_strong(p):
                p_loss = -np.log(np.clip(p[y_r], eps, 1 - eps))
                return (p_loss - target_dist) ** 2
        else:
            def dist_to_strong(p):
                return float((p - preds_strong[r]) ** 2)

        if abs(dist_to_strong(baseline_preds_w[r])) < 1e-12:
            optimal_k[r] = 0
            gap_closed[r] = 1.0
            continue

        # Greedy combinatorial search (same as transfer_sweep_v2)
        batch_size = min(12, K)
        current_dist = dist_to_strong(baseline_preds_w[r])
        best_pred = baseline_preds_w[r]
        accepted_combo = ()
        offset = 0
        X_row = X_query_w[r:r + 1]

        while offset < K:
            batch_end = min(offset + batch_size, K)
            batch_indices = list(range(offset, batch_end))
            n_batch = len(batch_indices)

            all_subsets = []
            for order in range(1, min(4, n_batch + 1)):
                all_subsets.extend(combos(batch_indices, order))

            n_combos = len(all_subsets)
            if n_combos == 0:
                break

            with torch.no_grad():
                combo_deltas = []
                for subset in all_subsets:
                    d = torch.zeros(d_target, dtype=torch.float32, device=device)
                    for j in accepted_combo:
                        d += per_feature_deltas[j]
                    for j in subset:
                        d += per_feature_deltas[j]
                    combo_deltas.append(d)
                deltas_batch = torch.stack(combo_deltas)

            if use_mitra:
                cand_preds = batched_intervention(
                    tail_w, X_row, deltas_batch, inject_context=False)
            elif use_sequential:
                cand_preds = batched_intervention_sequential(
                    tail_w, X_row, deltas_batch, query_idx=r)
            else:
                cand_preds = batched_intervention(
                    tail_w, X_row, deltas_batch, inject_context=False)

            # Find best (with overshoot protection)
            best_c = None
            best_c_dist = current_dist
            for c in range(n_combos):
                d = dist_to_strong(cand_preds[c])
                if d < best_c_dist:
                    if baseline_preds_w.ndim == 2:
                        p_i = cand_preds[c][y_r]
                        p_s = preds_strong[r, y_r]
                        p_w = baseline_preds_w[r, y_r]
                        if p_w < p_s and p_i > p_s:
                            continue
                        if p_w > p_s and p_i < p_s:
                            continue
                    else:
                        p_i = float(cand_preds[c])
                        p_s = float(preds_strong[r])
                        p_w = float(baseline_preds_w[r])
                        if p_w < p_s and p_i > p_s:
                            continue
                        if p_w > p_s and p_i < p_s:
                            continue
                    best_c = c
                    best_c_dist = d

            if best_c is not None:
                accepted_combo = tuple(accepted_combo) + all_subsets[best_c]
                current_dist = best_c_dist
                best_pred = cand_preds[best_c]

            offset = batch_end

        # Track accepted features
        accepted_features_r = set(
            delta_to_feature[j][0] for j in accepted_combo
        ) if accepted_combo else set()
        for fi in accepted_features_r:
            feature_accepted[fi] += 1

        accepted_k = len(accepted_combo)
        if accepted_k > 0:
            optimal_k[r] = accepted_k
            if baseline_preds_w.ndim == 2:
                orig_loss = -np.log(np.clip(baseline_preds_w[r, y_r], eps, 1 - eps))
                best_loss = -np.log(np.clip(best_pred[y_r], eps, 1 - eps))
                gap = orig_loss - target_dist
                moved = orig_loss - best_loss
                gap_closed[r] = min(1.0, max(0.0, moved / gap)) if gap > 1e-8 else 1.0
            else:
                orig_sq = (float(baseline_preds_w[r]) - float(preds_strong[r])) ** 2
                best_sq = (float(best_pred) - float(preds_strong[r])) ** 2
                gap_closed[r] = min(1.0, max(0.0, 1.0 - best_sq / orig_sq)) if orig_sq > 1e-12 else 1.0
            preds_intervened[r] = best_pred
        else:
            optimal_k[r] = 0
            gap_closed[r] = 0.0

        if (r + 1) % 50 == 0 or r == n_query - 1:
            elapsed = time.time() - t0
            rate = (r + 1) / elapsed if elapsed > 0 else 0
            valid_gc = gap_closed[:r+1][strong_wins[:r+1]]
            valid_gc = valid_gc[~np.isnan(valid_gc)]
            logger.info(f"    row {r+1}/{n_query}: mean_gc={valid_gc.mean():.3f} "
                        f"({rate:.1f} rows/s)")

    logger.info(f"  Done in {time.time() - t0:.1f}s")

    valid_gc = gap_closed[strong_wins]
    valid_gc = valid_gc[~np.isnan(valid_gc)]
    valid_k = optimal_k[strong_wins]

    n_accepted = int((valid_k > 0).sum())
    logger.info(f"  Acceptance: {n_accepted}/{n_strong_wins}, "
                f"gc={valid_gc.mean():.3f}")

    return {
        "strong_model": strong,
        "weak_model": weak,
        "optimal_k": optimal_k,
        "gap_closed": gap_closed,
        "strong_wins": strong_wins,
        "preds_strong": preds_strong,
        "preds_weak": preds_weak,
        "preds_transferred": preds_intervened,
        "n_query": n_query,
        "n_strong_wins": n_strong_wins,
        "mean_gap_closed": float(valid_gc.mean()) if len(valid_gc) else 0.0,
        "mean_optimal_k": float(valid_k.mean()) if len(valid_k) else 0.0,
        "metric_strong": metric_strong,
        "metric_weak": metric_weak,
        "metric_name": metric_name,
        "y_query": y_query.astype(np.float32),
        "acceptance_rate": float(n_accepted / max(n_strong_wins, 1)),
        "feature_acceptance": np.array(
            [(fi, feature_tried[fi], feature_accepted[fi])
             for fi in sorted(feature_tried.keys())],
            dtype=np.int32) if feature_tried else np.array([], dtype=np.int32),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Guided transfer: greedy search over ablation concepts")
    parser.add_argument("--models", nargs=2, required=True)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=64)
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

    # Build matched pairs for both directions
    labels_data = json.loads(Path(
        PROJECT_ROOT / "output" / "cross_model_concept_labels_round10.json"
    ).read_text())

    matched_pairs = {}
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
        logger.info(f"  {source}_to_{target}: {len(pairs)} matched pairs")

    # Find datasets with ablation results
    ablation_dir = ABLATION_DIR / pair_name
    if not ablation_dir.exists():
        print(f"No ablation results for {pair_name}")
        return

    available = sorted(f.stem for f in ablation_dir.glob("*.npz"))
    if args.datasets:
        datasets = [d for d in available if d in args.datasets]
    else:
        datasets = available

    out_dir = (args.output_dir if args.output_dir else OUTPUT_DIR) / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Guided transfer: {pair_name}")
    logger.info(f"  Datasets: {len(datasets)}")

    for i, ds in enumerate(datasets):
        out_path = out_dir / f"{ds}.npz"
        logger.info(f"\n[{i+1}/{len(datasets)}] {ds}")

        try:
            result = run_dataset(
                model_a, model_b, ds,
                saes, splits, norm_stats, test_embeddings,
                matched_pairs, args.device, args.max_steps,
            )
            if result is not None:
                np.savez_compressed(str(out_path), **result)
                logger.info(f"  -> {ds}: gc={result['mean_gap_closed']:.2f}, "
                            f"k={result['mean_optimal_k']:.1f}")
        except Exception as e:
            logger.error(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
