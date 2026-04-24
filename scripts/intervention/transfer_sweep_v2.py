#!/usr/bin/env python3
"""Cross-model concept transfer: improve a model with another model's concepts.

Uses the same backbone as ablation_sweep.py: cached perrow_importance for
baseline predictions, build_tail for model inference, batched_intervention
for delta injection. The only difference is:

  - Delta source: virtual atoms mapped from the source model's unmatched
    features into the target model's embedding space (not SAE feature zeroing).
  - Injection mode: query-only (inject_context=False). The train context is
    left unmodified; deltas are added to query positions at the extraction
    layer only. Attention in subsequent layers still lets the modified query
    attend to the (unchanged) context.
  - Direction: forward only (strong→weak). For each row where the strong model
    outperforms the weak, transfer strong's unique concepts into the weak model.

Output:
    output/transfer_sweep_v2/{model_a}_vs_{model_b}/{dataset}.npz

    Symmetric naming (sorted). NPZ records which model was strong/weak.

Usage:
    python -m scripts.intervention.transfer_sweep_v2 --models tabpfn tabicl --device cuda
    python -m scripts.intervention.transfer_sweep_v2 --models tabpfn tabicl --datasets credit-g
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
)
from scripts.matching.utils import load_norm_stats as load_norm_stats_matching
from scripts.intervention.transfer_virtual_nodes import (
    extract_decoder_atoms,
    load_cross_correlations,
    fit_concept_map,
    filter_landmarks,
    compute_local_virtual_atoms,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "output" / "transfer_sweep_v2"
IMPORTANCE_DIR = PROJECT_ROOT / "output" / "perrow_importance"
DEFAULT_MATCHING_FILE = PROJECT_ROOT / "output" / "sae_feature_matching_mnn_floor_p90.json"

SUPPORTED_MODELS = [
    "tabpfn", "tabicl", "tabicl_v2", "mitra",
    "tabdpt", "hyperfast", "carte", "tabula8b",
]


def get_matched_pairs(source_model: str, target_model: str,
                      concept_labels_path=None, matching_file=None):
    """Get MNN-matched (source_idx, target_idx) pairs.

    Two modes:
      1. matching_file (default): read raw MNN edges directly from the pairwise
         matching JSON. Every returned pair is a noise-floor-verified MNN edge.
      2. concept_labels_path: legacy path that reads concept groups from the
         labeling pipeline and picks one representative (first-src × first-tgt)
         per group. Representative pairs can be non-MNN if community detection
         grouped features transitively. Kept for backward compatibility.

    The default (matching_file=DEFAULT_MATCHING_FILE, a noise-floor-filtered
    MNN JSON) produces cleaner landmarks than concept_groups. Callers that
    explicitly want the old behavior must pass matching_file=None AND
    concept_labels_path=... .
    """
    from scripts.intervention.intervene_lib import (
        MODEL_KEY_TO_LABEL_KEY, DEFAULT_CONCEPT_LABELS,
    )

    # When neither override is set, use the default MNN matching file.
    if matching_file is None and concept_labels_path is None:
        matching_file = DEFAULT_MATCHING_FILE

    if matching_file is not None:
        with open(matching_file) as f:
            data = json.load(f)
        src_key = MODEL_KEY_TO_LABEL_KEY.get(source_model, source_model)
        tgt_key = MODEL_KEY_TO_LABEL_KEY.get(target_model, target_model)
        pairs_dict = data["pairs"]
        fwd_key = f"{src_key}__{tgt_key}"
        rev_key = f"{tgt_key}__{src_key}"
        if fwd_key in pairs_dict:
            return [(m["idx_a"], m["idx_b"]) for m in pairs_dict[fwd_key]["matches"]]
        elif rev_key in pairs_dict:
            # File stores target first; flip so the returned tuples are
            # (source_idx, target_idx) as callers expect.
            return [(m["idx_b"], m["idx_a"]) for m in pairs_dict[rev_key]["matches"]]
        else:
            return []

    # Legacy concept-groups path.
    with open(concept_labels_path) as f:
        data = json.load(f)

    src_key = MODEL_KEY_TO_LABEL_KEY.get(source_model, source_model)
    tgt_key = MODEL_KEY_TO_LABEL_KEY.get(target_model, target_model)

    pairs = []
    for group in data.get("concept_groups", {}).values():
        members = group.get("members", [])
        src_feats = [f for m, f in members if m == src_key]
        tgt_feats = [f for m, f in members if m == tgt_key]
        if src_feats and tgt_feats:
            pairs.append((src_feats[0], tgt_feats[0]))

    return pairs


from scripts.intervention.ablation_sweep import (  # noqa: E402
    get_unmatched_features as _get_unmatched_from_ablation,
)


def get_unmatched_features(source_model: str, target_model: str,
                           matching_file=None, **kwargs):
    """Get source features NOT matched to the target model.

    Delegates to ablation_sweep.get_unmatched_features (MNN matching).
    """
    return _get_unmatched_from_ablation(
        source_model, target_model,
        matching_file=matching_file,
    )
    return sorted(all_source - matched_source)


def run_dataset(
    model_a: str,
    model_b: str,
    dataset: str,
    saes: dict,
    splits: dict,
    norm_stats: dict,
    test_embeddings: dict,
    matched_pairs: dict,
    unmatched_features: dict,
    device: str,
    max_steps: int,
    min_cosine: float = 0.0,
    use_local_map: bool = False,
    use_adaptive_local: bool = False,
    importance_dir: Path = None,
    virtual_atoms_cache_dir: Path = None,
    gc_tolerance: float = 0.99,
    min_gap: float = 0.01,
) -> dict:
    """Transfer strong model's unique concepts into weak model for one dataset.

    Mirrors ablation_sweep.run_dataset structure:
    1. Load cached baseline predictions from perrow_importance
    2. Determine strong/weak
    3. Build tail for the WEAK model (target of transfer)
    4. Compute per-row local transfer deltas
    5. Greedy accept/reject per row
    """
    imp_dir = importance_dir if importance_dir else IMPORTANCE_DIR
    # Load cached baseline predictions from perrow_importance
    imp_a = np.load(imp_dir / model_a / f"{dataset}.npz", allow_pickle=True)
    imp_b = np.load(imp_dir / model_b / f"{dataset}.npz", allow_pickle=True)
    preds_a = imp_a["baseline_preds"]
    preds_b = imp_b["baseline_preds"]
    row_indices_a = imp_a["row_indices"]
    row_indices_b = imp_b["row_indices"]
    assert np.array_equal(row_indices_a, row_indices_b), (
        f"Row index mismatch between {model_a} and {model_b} on {dataset}"
    )
    y_query = imp_a["y_query"]
    n_query = len(y_query)
    task = "classification" if preds_a.ndim == 2 else "regression"

    preds = {model_a: preds_a, model_b: preds_b}
    losses = {
        model_a: compute_per_row_loss(y_query, preds_a, task),
        model_b: compute_per_row_loss(y_query, preds_b, task),
    }

    # Determine strong/weak
    metric_a, metric_name = compute_importance_metric(y_query, preds[model_a], task)
    metric_b, _ = compute_importance_metric(y_query, preds[model_b], task)

    if metric_name == "degenerate" or metric_a == float("-inf") or metric_b == float("-inf"):
        logger.info(f"  SKIP (degenerate predictions)")
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

    # Norm stats for target (weak) model
    if dataset not in norm_stats[weak]:
        logger.info(f"  SKIP (weak model {weak} has no norm stats for {dataset})")
        return {
            "strong_model": strong, "weak_model": weak,
            "n_strong_wins": 0, "n_query": n_query,
            "metric_strong": float(metric_strong),
            "metric_weak": float(metric_weak), "metric_name": metric_name,
        }

    weak_loss = losses[weak]
    strong_loss = losses[strong]
    baseline_preds_w = preds[weak]
    strong_preds = preds[strong]

    # Filter to rows where strong outperforms weak
    strong_wins = strong_loss < weak_loss
    n_strong_wins = int(strong_wins.sum())
    logger.info(f"  Strong wins on {n_strong_wins}/{n_query} rows")

    if n_strong_wins == 0:
        return {
            "strong_model": strong, "weak_model": weak,
            "n_strong_wins": 0, "n_query": n_query,
            "metric_strong": float(metric_strong),
            "metric_weak": float(metric_weak), "metric_name": metric_name,
        }

    # Build tail for WEAK model (target of transfer)
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
    torch.manual_seed(13)
    np.random.seed(13)
    tail_w = build_tail(weak, X_train_w, y_train_w, X_query_w, layer_w, task_w, device,
                        cat_indices=cat_indices, target_name=target_name)
    logger.info(f"  Weak tail ({weak}) built in {time.time() - t0:.1f}s")

    # Get decoder atoms and SAE activations for both models
    t1 = time.time()
    atoms_strong = extract_decoder_atoms(saes[strong]).numpy()
    atoms_weak = extract_decoder_atoms(saes[weak]).numpy()
    logger.info(f"  Decoder atoms extracted in {time.time() - t1:.2f}s")

    t1 = time.time()
    with torch.no_grad():
        emb_s = torch.tensor(test_embeddings[strong][dataset],
                             dtype=torch.float32, device=device)
        h_strong = saes[strong].encode(emb_s).cpu().numpy()
        emb_w = torch.tensor(test_embeddings[weak][dataset],
                             dtype=torch.float32, device=device)
        h_weak = saes[weak].encode(emb_w).cpu().numpy()
    logger.info(f"  SAE encode in {time.time() - t1:.2f}s")

    ds_mean_w, ds_std_w = norm_stats[weak][dataset]
    data_std_t_w = torch.tensor(ds_std_w, dtype=torch.float32, device=device)

    # Get matched/unmatched for this direction
    pair_key = f"{strong}_to_{weak}"
    m_pairs = matched_pairs.get(pair_key, [])
    unmatched = unmatched_features.get(pair_key, [])

    if not unmatched:
        logger.info(f"  SKIP (no unmatched features from {strong})")
        return {
            "strong_model": strong, "weak_model": weak,
            "n_strong_wins": n_strong_wins, "n_query": n_query,
            "metric_strong": float(metric_strong),
            "metric_weak": float(metric_weak), "metric_name": metric_name,
        }

    use_sequential = isinstance(tail_w, SEQUENTIAL_MODELS)
    use_mitra = isinstance(tail_w, MitraTail)

    # Load strong model's perrow_importance for feature ranking
    imp_strong = np.load(imp_dir / strong / f"{dataset}.npz", allow_pickle=True)
    row_feature_drops = imp_strong["row_feature_drops"]
    feature_indices = imp_strong["feature_indices"]
    unmatched_set = set(unmatched)

    # Build virtual_atoms_cache either from disk (cache-loading path) or by
    # fitting a landmark map at runtime (the original three branches).
    if virtual_atoms_cache_dir is not None:
        # Cache-loading fast path: skip filter_landmarks and all map branches.
        t1 = time.time()
        cache_file = Path(virtual_atoms_cache_dir) / f"{strong}_to_{weak}.npz"
        if not cache_file.exists():
            logger.info(f"  SKIP (cache file missing: {cache_file})")
            return {
                "strong_model": strong, "weak_model": weak,
                "n_strong_wins": n_strong_wins, "n_query": n_query,
                "metric_strong": float(metric_strong),
                "metric_weak": float(metric_weak), "metric_name": metric_name,
            }
        cache = np.load(cache_file, allow_pickle=True)
        vatoms = cache["virtual_atoms"]
        feature_ids = cache["feature_ids"]
        computed_mask = cache["computed_mask"]
        virtual_atoms_cache = {
            int(feature_ids[i]): vatoms[i].astype(np.float32)
            for i in range(len(feature_ids))
            if computed_mask[i]
        }
        r2_global = 0.0
        d_target = atoms_weak.shape[1]
        # Stand-in so the result dict's len(filt_pairs) reports the cache's
        # landmark count. Not used for anything else in the cache-loading path.
        filt_pairs = [()] * int(cache["n_landmarks"])
        logger.info(
            f"  Loaded {len(virtual_atoms_cache)} virtual atoms from "
            f"{cache_file.name} in {time.time() - t1:.2f}s"
        )
        if len(virtual_atoms_cache) == 0:
            logger.info(f"  SKIP (empty virtual atoms cache)")
            return {
                "strong_model": strong, "weak_model": weak,
                "n_strong_wins": n_strong_wins, "n_query": n_query,
                "metric_strong": float(metric_strong),
                "metric_weak": float(metric_weak), "metric_name": metric_name,
            }
    else:
        # Fit GLOBAL concept map on all matched decoder atom pairs (well-determined)
        matched_src_indices = [si for si, _ in m_pairs]
        matched_tgt_indices = [ti for _, ti in m_pairs]
        matched_src_atoms = atoms_strong[matched_src_indices]
        matched_tgt_atoms = atoms_weak[matched_tgt_indices]

        # Filter landmarks: sign-correct and remove low-quality pairs
        t1 = time.time()
        filt_src, filt_tgt, filt_pairs, quality = filter_landmarks(
            matched_src_atoms, matched_tgt_atoms, m_pairs,
            min_cosine=min_cosine, alpha=1.0,
        )
        logger.info(f"  Landmark filtering: {quality['n_kept']}/{quality['n_input']} kept"
                    f" (mean LOO cosine={quality.get('mean_cosine', 0):.3f})"
                    f" in {time.time() - t1:.2f}s")

        if len(filt_pairs) < 5:
            logger.info(f"  SKIP (too few landmarks after filtering: {len(filt_pairs)})")
            return {
                "strong_model": strong, "weak_model": weak,
                "n_strong_wins": n_strong_wins, "n_query": n_query,
                "metric_strong": float(metric_strong),
                "metric_weak": float(metric_weak), "metric_name": metric_name,
            }

        d_target = atoms_weak.shape[1]

        if use_local_map:
            # Local K-nearest-neighbor maps per unmatched feature (fixed K=8)
            t1 = time.time()
            virtual_dirs, virtual_scales = compute_local_virtual_atoms(
                atoms_strong, atoms_weak, filt_pairs, unmatched,
                n_neighbors=8, alpha=1.0,
            )
            virtual_atoms_cache = {}
            for u, fi in enumerate(unmatched):
                if np.linalg.norm(virtual_dirs[u]) > 1e-8:
                    virtual_atoms_cache[fi] = virtual_dirs[u] * virtual_scales[u]
            r2_global = 0.0
            logger.info(f"  Local virtual atoms (K=8): "
                        f"{len(virtual_atoms_cache)}/{len(unmatched)} computed"
                        f" in {time.time() - t1:.2f}s")
        elif use_adaptive_local:
            # Adaptive local maps: include all neighbors with |cosine| > threshold
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.linear_model import Ridge as _Ridge

            t1 = time.time()
            noise_floor = 0.20  # mean random SAE cosine
            sim_threshold = 2.0 * noise_floor  # 0.40

            matched_src_indices = [si for si, _ in filt_pairs]
            matched_tgt_indices = [ti for _, ti in filt_pairs]
            matched_src_atoms = atoms_strong[matched_src_indices]
            matched_tgt_atoms = atoms_weak[matched_tgt_indices]

            src_norms = np.linalg.norm(matched_src_atoms, axis=1, keepdims=True)
            tgt_norms = np.linalg.norm(matched_tgt_atoms, axis=1, keepdims=True)
            src_norms[src_norms < 1e-8] = 1.0
            tgt_norms[tgt_norms < 1e-8] = 1.0
            matched_src_unit = matched_src_atoms / src_norms
            matched_tgt_unit = matched_tgt_atoms / tgt_norms

            virtual_atoms_cache = {}
            neighbor_counts = []
            for fi in unmatched:
                atom_s = atoms_strong[fi]
                atom_norm = np.linalg.norm(atom_s)
                if atom_norm < 1e-8:
                    continue
                query_unit = (atom_s / atom_norm).reshape(1, -1)

                sims = cosine_similarity(query_unit, matched_src_unit)[0]
                mask = np.abs(sims) >= sim_threshold
                K = int(mask.sum())
                if K < 3:
                    continue
                neighbor_counts.append(K)

                idx = np.where(mask)[0]
                reg = _Ridge(alpha=1.0, fit_intercept=False)
                reg.fit(matched_src_unit[idx], matched_tgt_unit[idx])
                direction = reg.predict(query_unit)[0]
                dir_norm = np.linalg.norm(direction)
                if dir_norm < 1e-8:
                    continue

                local_ratios = tgt_norms[idx].ravel() / src_norms[idx].ravel()
                scale = float(np.median(local_ratios)) * atom_norm
                virtual_atoms_cache[fi] = (direction / dir_norm) * scale

            r2_global = 0.0
            mean_k = np.mean(neighbor_counts) if neighbor_counts else 0
            logger.info(f"  Adaptive local atoms (threshold={sim_threshold:.2f}): "
                        f"{len(virtual_atoms_cache)}/{len(unmatched)} computed, "
                        f"mean K={mean_k:.1f} in {time.time() - t1:.2f}s")
        else:
            # Global concept map
            t1 = time.time()
            M_global, r2_global = fit_concept_map(filt_src, filt_tgt, alpha=1.0)
            logger.info(f"  Global concept map: {filt_src.shape[0]} pairs, "
                        f"R²={r2_global:.3f}, M shape={M_global.shape}"
                        f" in {time.time() - t1:.2f}s")

            # Pre-compute magnitude correction: median target/source norm ratio
            src_norms_matched = np.linalg.norm(filt_src, axis=1)
            tgt_norms_matched = np.linalg.norm(filt_tgt, axis=1)
            valid_norms = (src_norms_matched > 1e-8) & (tgt_norms_matched > 1e-8)
            if valid_norms.sum() > 0:
                norm_ratios = tgt_norms_matched[valid_norms] / src_norms_matched[valid_norms]
                median_norm_ratio = float(np.median(norm_ratios))
            else:
                median_norm_ratio = 1.0

            # Pre-compute virtual atoms using the global map
            virtual_atoms_cache = {}
            for fi in unmatched:
                atom_s = atoms_strong[fi]
                atom_norm = np.linalg.norm(atom_s)
                if atom_norm < 1e-8:
                    continue
                unit_atom_s = atom_s / atom_norm
                virtual_dir = unit_atom_s @ M_global.T
                vdir_norm = np.linalg.norm(virtual_dir)
                if vdir_norm < 1e-8:
                    continue
                virtual_atom = (virtual_dir / vdir_norm) * atom_norm * median_norm_ratio
                virtual_atoms_cache[fi] = virtual_atom

            logger.info(f"  Pre-computed {len(virtual_atoms_cache)}/{len(unmatched)} virtual atoms")

    # Per-row combinatorial greedy search (mirrors ablation_sweep)
    from itertools import combinations as combos

    optimal_k = np.zeros(n_query, dtype=np.int32)
    gap_closed = np.full(n_query, np.nan, dtype=np.float32)
    preds_intervened = baseline_preds_w.copy()

    # Per-step diagnostics: for every row, record the prediction after
    # every accepted greedy step and which features were added at that step.
    # Each step may add 1-3 features (the subset-size upper bound in the
    # combinatorial search). Unused positions are NaN / -1.
    MAX_STEPS = 20
    pred_shape_suffix = baseline_preds_w.shape[1:]  # () for regression, (C,) for cls
    step_preds = np.full((n_query, MAX_STEPS) + pred_shape_suffix,
                         np.nan, dtype=np.float32)
    step_features = np.full((n_query, MAX_STEPS, 3), -1, dtype=np.int32)
    step_sizes = np.zeros((n_query, MAX_STEPS), dtype=np.int8)
    selected_features = np.full((n_query, MAX_STEPS * 3), -1, dtype=np.int32)

    # Per-feature predictions: cumulative injection one feature at a time
    MAX_FEAT_STEPS = 20
    feature_preds = np.full((n_query, MAX_FEAT_STEPS) + pred_shape_suffix,
                            np.nan, dtype=np.float32)

    # Track per-feature acceptance across all rows
    from collections import Counter
    feature_tried = Counter()    # fi -> n_rows where it was a candidate
    feature_accepted = Counter() # fi -> n_rows where it was accepted

    t0 = time.time()
    for r in range(n_query):
        if not strong_wins[r]:
            optimal_k[r] = 0
            gap_closed[r] = 1.0
            continue

        # Skip rows where models effectively agree on this prediction
        if baseline_preds_w.ndim == 2:
            y_r_gap = int(y_query[r])
            pred_gap = abs(float(strong_preds[r, y_r_gap] - baseline_preds_w[r, y_r_gap]))
        else:
            denom = abs(float(strong_preds[r]))
            pred_gap = abs(float(strong_preds[r] - baseline_preds_w[r])) / denom if denom > 1e-8 else 0.0
        if pred_gap < min_gap:
            optimal_k[r] = 0
            gap_closed[r] = 1.0
            continue

        # Find unmatched features that fire on this row, rank by abs(importance)
        row_drops = row_feature_drops[r]
        firing_unmatched = []
        for i, fi in enumerate(feature_indices):
            if fi in unmatched_set and h_strong[r, fi] > 0 and fi in virtual_atoms_cache:
                firing_unmatched.append((i, fi, abs(row_drops[i])))

        if not firing_unmatched:
            optimal_k[r] = 0
            gap_closed[r] = 0.0
            continue

        # Sort by importance magnitude (descending)
        firing_unmatched.sort(key=lambda x: -x[2])

        # Compute individual transfer deltas using pre-computed virtual atoms
        # Try both +delta and -delta (cross-correlation uses |r|, sign may flip)
        per_feature_deltas = []
        delta_to_feature = []  # maps delta index -> (feature_idx, sign)
        for _, fi, _ in firing_unmatched:
            a_s = float(h_strong[r, fi])
            virtual_atom = virtual_atoms_cache[fi]
            delta_raw = a_s * virtual_atom * ds_std_w
            per_feature_deltas.append(
                torch.tensor(delta_raw, dtype=torch.float32, device=device))
            delta_to_feature.append((fi, +1))
            per_feature_deltas.append(
                torch.tensor(-delta_raw, dtype=torch.float32, device=device))
            delta_to_feature.append((fi, -1))

        K = len(per_feature_deltas)

        # Distance metric
        y_r = int(y_query[r])
        if baseline_preds_w.ndim == 2:
            eps = 1e-7
            target_dist = -np.log(np.clip(strong_preds[r, y_r], eps, 1 - eps))
            orig_dist = -np.log(np.clip(baseline_preds_w[r, y_r], eps, 1 - eps))
            def dist_to_strong(p):
                p_loss = -np.log(np.clip(p[y_r], eps, 1 - eps))
                return (p_loss - target_dist) ** 2
        else:
            def dist_to_strong(p):
                return float((p - strong_preds[r]) ** 2)

        if abs(dist_to_strong(baseline_preds_w[r])) < 1e-12:
            optimal_k[r] = 0
            gap_closed[r] = 1.0
            continue

        # Combinatorial batching (same as ablation_sweep)
        batch_size = min(12, K)
        current_dist = dist_to_strong(baseline_preds_w[r])
        best_pred = baseline_preds_w[r]
        accepted_combo = ()
        offset = 0
        X_row = X_query_w[r:r + 1]
        step_idx = 0  # per-row step counter for step_preds/step_features

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

            # Build cumulative deltas: accepted so far + each candidate subset
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

            # Find best subset (reject overshoots past the strong model)
            best_c = None
            best_c_dist = current_dist
            for c in range(n_combos):
                d = dist_to_strong(cand_preds[c])
                if d < best_c_dist:
                    # Overshoot check: prediction should stay between
                    # baseline weak and strong, not go past strong
                    if baseline_preds_w.ndim == 2:
                        p_intervened = cand_preds[c][y_r]
                        p_strong = strong_preds[r, y_r]
                        p_weak = baseline_preds_w[r, y_r]
                        # Reject if intervened overshot past strong
                        if p_weak < p_strong and p_intervened > p_strong:
                            continue
                        if p_weak > p_strong and p_intervened < p_strong:
                            continue
                    else:
                        p_intervened = float(cand_preds[c])
                        p_strong = float(strong_preds[r])
                        p_weak = float(baseline_preds_w[r])
                        if p_weak < p_strong and p_intervened > p_strong:
                            continue
                        if p_weak > p_strong and p_intervened < p_strong:
                            continue
                    best_c = c
                    best_c_dist = d

            if best_c is not None:
                accepted_subset = all_subsets[best_c]
                accepted_combo = tuple(accepted_combo) + accepted_subset
                current_dist = best_c_dist
                best_pred = cand_preds[best_c]

                # Record per-step state (cap at MAX_STEPS)
                if step_idx < MAX_STEPS:
                    step_preds[r, step_idx] = best_pred
                    step_sizes[r, step_idx] = len(accepted_subset)
                    for si, j in enumerate(accepted_subset[:3]):
                        step_features[r, step_idx, si] = delta_to_feature[j][0]
                step_idx += 1

                # Early stop: check if gc crossed tolerance
                if baseline_preds_w.ndim == 2:
                    best_loss = -np.log(np.clip(best_pred[y_r], eps, 1 - eps))
                    gap = orig_dist - target_dist
                    moved = orig_dist - best_loss
                    gc_now = min(1.0, max(0.0, moved / gap)) if gap > 1e-8 else 1.0
                else:
                    orig_dist_sq = float((baseline_preds_w[r] - strong_preds[r]) ** 2)
                    best_dist_sq = float((best_pred - strong_preds[r]) ** 2)
                    gc_now = min(1.0, max(0.0, 1.0 - best_dist_sq / orig_dist_sq)) if orig_dist_sq > 1e-12 else 1.0
                if gc_now >= gc_tolerance:
                    break

            offset = batch_end

        # Flat ordered list of accepted feature indices (LOO-ranked, filtered
        # to accepted, with duplicates possible if both +/- directions of the
        # same feature were accepted).
        flat = [delta_to_feature[j][0] for j in accepted_combo]
        for i, fi in enumerate(flat[:selected_features.shape[1]]):
            selected_features[r, i] = fi

        # Track which features were tried and accepted
        tried_features = set(fi for _, fi, _ in firing_unmatched)
        accepted_features_r = set(
            delta_to_feature[j][0] for j in accepted_combo
        ) if accepted_combo else set()
        for fi in tried_features:
            feature_tried[fi] += 1
        for fi in accepted_features_r:
            feature_accepted[fi] += 1

        accepted_k = len(accepted_combo)
        if accepted_k > 0:
            optimal_k[r] = accepted_k
            if baseline_preds_w.ndim == 2:
                orig_loss = -np.log(np.clip(baseline_preds_w[r, y_r], eps, 1 - eps))
                best_loss = -np.log(np.clip(best_pred[y_r], eps, 1 - eps))
                # gap is positive: weak loss (high) - strong loss (low)
                gap = orig_loss - target_dist
                # moved is positive: weak loss - intervened loss (improved)
                moved = orig_loss - best_loss
                gap_closed[r] = min(1.0, max(0.0, moved / gap)) if gap > 1e-8 else 1.0
            else:
                orig_dist_sq = float((baseline_preds_w[r] - strong_preds[r]) ** 2)
                best_dist_sq = float((best_pred - strong_preds[r]) ** 2)
                gap_closed[r] = min(1.0, max(0.0, 1.0 - best_dist_sq / orig_dist_sq)) if orig_dist_sq > 1e-12 else 1.0
            preds_intervened[r] = best_pred

            # Per-feature replay: inject virtual atoms one at a time
            cum_delta = torch.zeros(1, d_target, dtype=torch.float32, device=device)
            seen_fi = set()
            fi_idx = 0
            for j in accepted_combo:
                fi = delta_to_feature[j][0]
                if fi in seen_fi:
                    continue
                seen_fi.add(fi)
                if fi_idx >= MAX_FEAT_STEPS:
                    break
                cum_delta += per_feature_deltas[j].unsqueeze(0)
                if use_mitra:
                    fp = batched_intervention(tail_w, X_row, cum_delta)
                elif use_sequential:
                    fp = batched_intervention_sequential(tail_w, X_row, cum_delta, query_idx=r)
                else:
                    fp = batched_intervention(tail_w, X_row, cum_delta)
                feature_preds[r, fi_idx] = fp[0]
                fi_idx += 1
        else:
            optimal_k[r] = 0
            gap_closed[r] = 0.0

        if (r + 1) % 50 == 0 or r == n_query - 1:
            elapsed = time.time() - t0
            rate = (r + 1) / elapsed if elapsed > 0 else 0
            eta = (n_query - r - 1) / rate if rate > 0 else 0
            valid = optimal_k[:r+1][strong_wins[:r+1]]
            mean_k = valid.mean() if len(valid) else 0
            logger.info(f"    row {r+1}/{n_query}: mean_k={mean_k:.1f} "
                        f"({rate:.1f} rows/s, ETA {eta:.0f}s)")

    logger.info(f"  Done in {time.time() - t0:.1f}s")

    valid_k = optimal_k[strong_wins]
    valid_gc = gap_closed[strong_wins]
    valid_gc = valid_gc[~np.isnan(valid_gc)]

    # Acceptance rate: rows with k>0 among strong-wins
    n_accepted = int((optimal_k[strong_wins] > 0).sum())
    acceptance_rate = n_accepted / max(n_strong_wins, 1)
    logger.info(f"  Acceptance rate: {n_accepted}/{n_strong_wins} "
                f"({acceptance_rate:.0%}), mean_gc={float(valid_gc.mean()) if len(valid_gc) else 0:.3f}")

    return {
        "strong_model": strong,
        "weak_model": weak,
        "optimal_k": optimal_k,
        "gap_closed": gap_closed,
        "strong_wins": strong_wins,
        "preds_strong": strong_preds,
        "preds_weak": baseline_preds_w,
        "preds_intervened": preds_intervened,
        "baseline_loss_strong": strong_loss,
        "baseline_loss_weak": weak_loss,
        "n_query": n_query,
        "n_strong_wins": n_strong_wins,
        "mean_optimal_k": float(valid_k.mean()) if len(valid_k) else 0.0,
        "median_optimal_k": float(np.median(valid_k)) if len(valid_k) else 0.0,
        "mean_gap_closed": float(valid_gc.mean()) if len(valid_gc) else 0.0,
        "metric_strong": float(metric_strong),
        "metric_weak": float(metric_weak),
        "metric_name": metric_name,
        "y_query": y_query.astype(np.float32),
        "row_indices": row_indices_a.astype(np.int32),
        # Transfer diagnostics
        "concept_map_r2": float(r2_global),
        "n_landmarks": int(len(filt_pairs)),
        "n_virtual_atoms": int(len(virtual_atoms_cache)),
        "acceptance_rate": float(acceptance_rate),
        # Per-step diagnostics: predictions and accepted features after each
        # greedy batch step. Enables post-hoc computation of true per-step
        # |Δpred| decay. See comment above the loop for layout.
        "step_preds": step_preds,
        "step_features": step_features,
        "step_sizes": step_sizes,
        "feature_preds": feature_preds,
        "selected_features": selected_features,
        # Per-feature acceptance: which concepts transferred successfully
        # Arrays of (feature_idx, n_tried, n_accepted) for post-analysis
        "feature_acceptance": np.array(
            [(fi, feature_tried[fi], feature_accepted[fi])
             for fi in sorted(feature_tried.keys())],
            dtype=np.int32) if feature_tried else np.array([], dtype=np.int32),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model concept transfer: improve weak model with strong model's concepts")
    parser.add_argument("--models", nargs=2, required=True, metavar="MODEL",
                        help="Two models to compare (order does not matter)")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--min-cosine", type=float, default=0.0,
                        help="Min LOO cosine to keep landmark pairs (default: 0.0)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: output/transfer_sweep_v2)")
    parser.add_argument("--local-map", action="store_true",
                        help="Use per-feature local K-NN maps instead of global ridge")
    parser.add_argument("--local-map-adaptive", action="store_true",
                        help="Use adaptive local maps (neighbors with cosine > 2x noise floor)")
    parser.add_argument("--sae-dir", type=Path, default=None,
                        help="SAE checkpoint directory (default: sweep round10)")
    parser.add_argument("--importance-dir", type=Path, default=None,
                        help="Per-row importance directory (default: output/perrow_importance)")
    parser.add_argument("--matching-file", type=Path, default=DEFAULT_MATCHING_FILE,
                        help="Raw MNN matching JSON used for BOTH matched landmark pairs "
                             "and unmatched features. Default is the noise-floor-filtered "
                             "mnn_floor_p90 file.")
    parser.add_argument("--virtual-atoms-cache-dir", type=Path, default=None,
                        help="If set, load pre-built virtual atoms from this directory "
                             "(one npz per pair direction, named {source}_to_{target}.npz) "
                             "and skip the filter_landmarks + map-build steps entirely. "
                             "Used by the R²-to-gc translation test against caches built "
                             "by scripts.analysis.build_transfer_caches.")
    parser.add_argument("--gc-tolerance", type=float, default=0.99,
                        help="Stop greedy search per row once gap_closed "
                             "reaches this threshold (default: 0.99)")
    parser.add_argument("--min-gap", type=float, default=0.01,
                        help="Skip rows where models agree within this threshold. "
                             "Classification: |P_strong(y) - P_weak(y)|. "
                             "Regression: |pred_strong - pred_weak| / |pred_strong|. "
                             "(default: 0.01)")
    args = parser.parse_args()

    model_a, model_b = sorted(args.models)
    pair_name = f"{model_a}_vs_{model_b}"

    splits = json.loads(SPLITS_PATH.read_text())

    # Load SAEs, norm stats, test embeddings for both models
    sae_dir = args.sae_dir if args.sae_dir else None
    saes = {}
    norm_stats = {}
    test_embeddings = {}
    for m in (model_a, model_b):
        sae, _ = load_sae(m, device=args.device,
                          **({"sae_dir": sae_dir} if sae_dir else {}))
        sae.eval()
        saes[m] = sae
        norm_stats[m] = load_norm_stats_matching(m)
        test_embeddings[m] = load_test_embeddings(m)

    # Build matched pairs and unmatched features for both directions
    matched_pairs = {}
    unmatched_features = {}
    for source, target in [(model_a, model_b), (model_b, model_a)]:
        key = f"{source}_to_{target}"
        m_pairs = get_matched_pairs(source, target, matching_file=args.matching_file)
        unmatched = get_unmatched_features(
            source, target, matching_file=args.matching_file)
        matched_pairs[key] = m_pairs
        unmatched_features[key] = unmatched
        logger.info(f"  {key}: {len(m_pairs)} matched pairs, {len(unmatched)} unmatched")

    # Find datasets where both models have importance data
    imp_dir = args.importance_dir if args.importance_dir else IMPORTANCE_DIR
    ds_a = set(d.stem for d in (imp_dir / model_a).glob("*.npz"))
    ds_b = set(d.stem for d in (imp_dir / model_b).glob("*.npz"))
    available = sorted(ds_a & ds_b)

    if args.datasets:
        datasets = [d for d in available if d in args.datasets]
    else:
        datasets = available

    out_dir = (args.output_dir if args.output_dir else OUTPUT_DIR) / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Transfer sweep: {model_a} vs {model_b}")
    logger.info(f"  Datasets: {len(datasets)} (of {len(ds_a)} / {len(ds_b)} with importance)")

    for i, ds in enumerate(datasets):
        out_path = out_dir / f"{ds}.npz"
        if args.resume and out_path.exists():
            logger.info(f"[{i+1}/{len(datasets)}] {ds}: SKIP (exists)")
            continue

        logger.info(f"\n[{i+1}/{len(datasets)}] {ds}")

        if ds not in norm_stats[model_a] and ds not in norm_stats[model_b]:
            logger.info(f"  SKIP (missing norm stats for both models)")
            continue

        try:
            result = run_dataset(
                model_a, model_b, ds,
                saes, splits, norm_stats, test_embeddings,
                matched_pairs, unmatched_features,
                args.device, args.max_steps, args.min_cosine,
                args.local_map, args.local_map_adaptive,
                importance_dir=imp_dir,
                virtual_atoms_cache_dir=args.virtual_atoms_cache_dir,
                gc_tolerance=args.gc_tolerance,
                min_gap=args.min_gap,
            )
            np.savez_compressed(str(out_path), **result)

            if result["n_strong_wins"] > 0:
                logger.info(f"  -> {out_path.name}: {result['strong_model']}>{result['weak_model']}, "
                            f"{result['n_strong_wins']} rows, "
                            f"gap_closed={result['mean_gap_closed']:.2f}")
            else:
                logger.info(f"  -> {out_path.name}: models tied")

        except Exception as e:
            logger.error(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
