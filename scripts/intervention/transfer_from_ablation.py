#!/usr/bin/env python3
"""Transfer concepts identified by ablation into the weak model.

Instead of running a greedy search to find which concepts to transfer,
this script takes the concepts already identified by ablation_sweep.py
(the features whose removal degrades the strong model) and injects them
into the weak model. If ablation found that features {f3, f7, f12}
explain the strong model's advantage on a row, we map those same features
into the weak model's embedding space and inject them.

This is a direct test: do the concepts that explain the strong model's
advantage also improve the weak model when transferred?

Usage:
    python -m scripts.intervention.transfer_from_ablation --models mitra tabpfn --device cuda
    python -m scripts.intervention.transfer_from_ablation --models mitra tabpfn --datasets credit-g
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
    compute_local_virtual_atoms,
    filter_landmarks,
)
from scripts.intervention.ablation_sweep import get_unmatched_features

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ABLATION_DIR = PROJECT_ROOT / "output" / "ablation_sweep"
OUTPUT_DIR = PROJECT_ROOT / "output" / "transfer_from_ablation"
IMPORTANCE_DIR = PROJECT_ROOT / "output" / "perrow_importance"


def run_dataset(
    model_a: str, model_b: str, dataset: str,
    saes: dict, splits: dict, norm_stats: dict, test_embeddings: dict,
    device: str,
) -> dict:
    """Transfer ablation-identified concepts into the weak model."""

    # Load ablation results to get selected features per row
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
        logger.info(f"  SKIP (no strong wins)")
        return None

    selected_features_abl = abl["selected_features"]  # (n_query, max_k) or empty
    if selected_features_abl.size == 0:
        logger.info(f"  SKIP (no selected features in ablation)")
        return None

    preds_strong = abl["preds_strong"]
    preds_weak = abl["preds_weak"]
    y_query = abl["y_query"]
    task = "classification" if preds_strong.ndim == 2 else "regression"

    metric_strong = float(abl["metric_strong"])
    metric_weak = float(abl["metric_weak"])
    metric_name = str(abl["metric_name"])

    logger.info(f"  {strong} ({metric_name}={metric_strong:.4f}) > "
                f"{weak} ({metric_name}={metric_weak:.4f})")
    logger.info(f"  Strong wins on {n_strong_wins}/{n_query} rows")

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
    logger.info(f"  Weak tail ({weak}) built in {time.time() - t0:.1f}s")

    # Get decoder atoms for both models
    atoms_strong = extract_decoder_atoms(saes[strong]).numpy()
    atoms_weak = extract_decoder_atoms(saes[weak]).numpy()

    # Get matched pairs for concept map
    labels_data = json.loads(Path(
        PROJECT_ROOT / "output" / f"cross_model_concept_labels_round10.json"
    ).read_text())

    src_key = MODEL_KEY_TO_LABEL_KEY[strong]
    tgt_key = MODEL_KEY_TO_LABEL_KEY[weak]
    matched_pairs = []
    for g in labels_data["concept_groups"].values():
        members = g.get("members", [])
        src_feats = [f for m, f in members if m == src_key]
        tgt_feats = [f for m, f in members if m == tgt_key]
        if src_feats and tgt_feats:
            matched_pairs.append((src_feats[0], tgt_feats[0]))

    if len(matched_pairs) < 5:
        logger.info(f"  SKIP (too few matched pairs: {len(matched_pairs)})")
        return None

    # Build adaptive local virtual atoms for the strong model's features
    t1 = time.time()
    matched_src_atoms = atoms_strong[[si for si, _ in matched_pairs]]
    matched_tgt_atoms = atoms_weak[[ti for _, ti in matched_pairs]]

    filt_src, filt_tgt, filt_pairs, quality = filter_landmarks(
        matched_src_atoms, matched_tgt_atoms, matched_pairs,
        min_cosine=0.0, alpha=1.0,
    )
    logger.info(f"  Landmarks: {quality['n_kept']}/{quality['n_input']} kept "
                f"(LOO cosine={quality.get('mean_cosine', 0):.3f}) "
                f"in {time.time() - t1:.2f}s")

    # Get all unique features selected by ablation
    all_selected = set()
    for r in range(n_query):
        if strong_wins[r] and selected_features_abl.ndim == 2:
            for fi in selected_features_abl[r]:
                if fi >= 0:
                    all_selected.add(int(fi))
    logger.info(f"  Ablation selected {len(all_selected)} unique features")

    # Build virtual atoms using adaptive local maps
    t1 = time.time()
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.linear_model import Ridge

    noise_floor = 0.20
    sim_threshold = 2.0 * noise_floor

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

    virtual_atoms = {}
    for fi in all_selected:
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

    logger.info(f"  Virtual atoms: {len(virtual_atoms)}/{len(all_selected)} "
                f"in {time.time() - t1:.2f}s")

    # Norm stats for denormalization
    ds_mean_w, ds_std_w = norm_stats[weak][dataset]
    ds_std_w = np.array(ds_std_w)

    # SAE activations for the strong model
    emb_s = test_embeddings[strong][dataset]
    with torch.no_grad():
        h_strong = saes[strong].encode(
            torch.tensor(emb_s, dtype=torch.float32, device=device)
        ).cpu().numpy()

    use_sequential = isinstance(tail_w, SEQUENTIAL_MODELS)
    use_mitra = isinstance(tail_w, MitraTail)

    # Per-row: inject the ablation-selected features
    optimal_k = np.zeros(n_query, dtype=np.int32)
    gap_closed = np.full(n_query, np.nan, dtype=np.float32)
    baseline_preds_w = preds_weak.copy()
    preds_intervened = preds_weak.copy()

    t0 = time.time()
    for r in range(n_query):
        if not strong_wins[r]:
            optimal_k[r] = 0
            gap_closed[r] = 1.0
            continue

        # Get this row's ablation-selected features
        if selected_features_abl.ndim == 2:
            row_selected = [int(fi) for fi in selected_features_abl[r] if fi >= 0]
        else:
            row_selected = []

        if not row_selected:
            optimal_k[r] = 0
            gap_closed[r] = 0.0
            continue

        # Build cumulative delta from all selected features
        d_target = atoms_weak.shape[1]
        delta_pos = torch.zeros(d_target, dtype=torch.float32, device=device)
        delta_neg = torch.zeros(d_target, dtype=torch.float32, device=device)
        n_mapped = 0

        for fi in row_selected:
            if fi not in virtual_atoms:
                continue
            a_s = float(h_strong[r, fi])
            if a_s <= 0:
                continue
            va = virtual_atoms[fi]
            raw = a_s * va * ds_std_w
            delta_pos += torch.tensor(raw, dtype=torch.float32, device=device)
            delta_neg -= torch.tensor(raw, dtype=torch.float32, device=device)
            n_mapped += 1

        if n_mapped == 0:
            optimal_k[r] = 0
            gap_closed[r] = 0.0
            continue

        # Test both +/- delta, pick the one that moves toward strong
        X_row = X_query_w[r:r + 1]
        deltas = torch.stack([delta_pos, delta_neg])

        if use_mitra:
            cand = batched_intervention(tail_w, X_row, deltas, inject_context=False)
        elif use_sequential:
            cand = batched_intervention_sequential(tail_w, X_row, deltas, query_idx=r)
        else:
            cand = batched_intervention(tail_w, X_row, deltas, inject_context=False)

        # Pick the candidate closer to strong (with overshoot protection)
        y_r = int(y_query[r])
        eps = 1e-7
        best_pred = baseline_preds_w[r]

        if preds_strong.ndim == 2:
            p_strong = preds_strong[r, y_r]
            p_weak = baseline_preds_w[r, y_r]
            target_loss = -np.log(np.clip(p_strong, eps, 1 - eps))
            baseline_loss = -np.log(np.clip(p_weak, eps, 1 - eps))
            best_dist = (baseline_loss - target_loss) ** 2

            for c in range(2):
                p_c = cand[c][y_r]
                c_loss = -np.log(np.clip(p_c, eps, 1 - eps))
                c_dist = (c_loss - target_loss) ** 2
                # Overshoot check
                if p_weak < p_strong and p_c > p_strong:
                    continue
                if p_weak > p_strong and p_c < p_strong:
                    continue
                if c_dist < best_dist:
                    best_dist = c_dist
                    best_pred = cand[c]
        else:
            p_strong = float(preds_strong[r])
            p_weak = float(baseline_preds_w[r])
            best_dist = (p_weak - p_strong) ** 2

            for c in range(2):
                p_c = float(cand[c])
                c_dist = (p_c - p_strong) ** 2
                if p_weak < p_strong and p_c > p_strong:
                    continue
                if p_weak > p_strong and p_c < p_strong:
                    continue
                if c_dist < best_dist:
                    best_dist = c_dist
                    best_pred = cand[c]

        optimal_k[r] = n_mapped
        preds_intervened[r] = best_pred

        # Compute gap_closed
        if preds_strong.ndim == 2:
            orig_loss = -np.log(np.clip(baseline_preds_w[r, y_r], eps, 1 - eps))
            best_loss = -np.log(np.clip(best_pred[y_r], eps, 1 - eps))
            gap = orig_loss - target_loss
            moved = orig_loss - best_loss
            gap_closed[r] = min(1.0, max(0.0, moved / gap)) if gap > 1e-8 else 1.0
        else:
            orig_sq = (float(baseline_preds_w[r]) - float(preds_strong[r])) ** 2
            best_sq = (float(best_pred) - float(preds_strong[r])) ** 2
            gap_closed[r] = min(1.0, max(0.0, 1.0 - best_sq / orig_sq)) if orig_sq > 1e-12 else 1.0

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
    }


def main():
    parser = argparse.ArgumentParser(
        description="Transfer ablation-identified concepts into weak model")
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

    logger.info(f"Transfer from ablation: {pair_name}")
    logger.info(f"  Datasets: {len(datasets)}")

    for i, ds in enumerate(datasets):
        out_path = out_dir / f"{ds}.npz"
        logger.info(f"\n[{i+1}/{len(datasets)}] {ds}")

        try:
            result = run_dataset(
                model_a, model_b, ds,
                saes, splits, norm_stats, test_embeddings,
                args.device,
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
