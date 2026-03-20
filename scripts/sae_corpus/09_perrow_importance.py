#!/usr/bin/env python3
"""Compute per-row, per-feature importance via batched single-row ablation.

For each query row:
  1. Build tail model with K copies of that row as query (K = firing features)
  2. Compute K deltas (one feature zeroed each)
  3. Inject all K deltas into the cached hidden state
  4. One tail.predict call → K ablated predictions

Output:
    output/sae_training_round10/perrow_importance_{variant}_{dataset}.npz

Usage:
    python scripts/sae_corpus/09_perrow_importance.py --device cuda
    python scripts/sae_corpus/09_perrow_importance.py --device cuda --datasets diabetes
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.sparse_autoencoder import SAEConfig, SparseAutoencoder
from data.preprocessing import CACHE_DIR, load_preprocessed
from scripts._project_root import PROJECT_ROOT
from scripts.intervention.concept_importance import compute_per_row_loss
from scripts.intervention.intervene_sae import compute_ablation_delta

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_training_round10"

EVAL_DATASETS = {
    "airfoil_self_noise": {"task": "regression"},
    "diabetes": {"task": "classification"},
}

VARIANTS = {
    "task_aware": {
        "sae_path": OUTPUT_DIR / "tabpfn_taskaware_sae.pt",
        "stats_path": OUTPUT_DIR / "tabpfn_taskaware_norm_stats.npz",
    },
    "per_dataset": {
        "sae_path": OUTPUT_DIR / "tabpfn_perds_sae.pt",
        "stats_path": OUTPUT_DIR / "tabpfn_perds_norm_stats.npz",
    },
}


def load_sae(sae_path, device):
    ckpt = torch.load(str(sae_path), map_location=device, weights_only=False)
    config = SAEConfig(**ckpt["config"])
    sae = SparseAutoencoder(config)
    state = ckpt["state_dict"]
    if "reference_data" in state and state["reference_data"] is not None:
        sae.register_buffer("reference_data", state["reference_data"])
        if "archetype_logits" in state:
            sae.archetype_logits = torch.nn.Parameter(state["archetype_logits"])
        if "archetype_deviation" in state:
            sae.archetype_deviation = torch.nn.Parameter(state["archetype_deviation"])
    sae.load_state_dict(state, strict=False)
    sae.to(device)
    sae.eval()
    return sae, config


def load_norm_stats(stats_path, dataset, device):
    stats = np.load(str(stats_path), allow_pickle=True)
    datasets = list(stats["datasets"])
    idx = datasets.index(dataset)
    mean = torch.tensor(stats["means"][idx], dtype=torch.float32, device=device)
    std = torch.tensor(stats["stds"][idx], dtype=torch.float32, device=device)
    layer = int(stats["layers"][idx])
    return mean, std, layer


def compute_perrow_importance(
    sae, data_mean, data_std, extraction_layer,
    X_ctx, y_ctx, X_q, y_q, task, device,
):
    """Batched per-row LOO importance.

    For each query row:
      1. Build TabPFN tail with K copies as query (K = firing features)
      2. SAE encode → K copies with one feature zeroed each → decode → K deltas
      3. Inject into cached hidden state → one predict → K ablated predictions
      4. Loss change = ablated - baseline
    """
    from models.tabpfn_utils import load_tabpfn
    from scripts.intervention.intervene_sae import TabPFNTail

    n_query = len(X_q)

    # Fit once to get baseline predictions and determine alive features
    clf = load_tabpfn(task=task, device=device, n_estimators=1)
    clf.fit(X_ctx, y_ctx)
    layers = clf.model_.transformer_encoder.layers

    # Get baseline predictions one row at a time (causal)
    logger.info("  Phase 1: baseline predictions...")
    baseline_preds_list = []
    baseline_hidden = []  # store mean-pooled hidden per row

    for row_idx in range(n_query):
        x_row = X_q[row_idx:row_idx + 1]
        captured = {}

        def capture_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                captured["hidden"] = output.detach()

        handle = layers[extraction_layer].register_forward_hook(capture_hook)
        try:
            with torch.no_grad():
                if task == "regression":
                    preds = clf.predict(x_row)
                else:
                    preds = clf.predict_proba(x_row)
        finally:
            handle.remove()

        baseline_preds_list.append(np.asarray(preds))

        # Mean-pool for SAE encoding
        h = captured["hidden"]
        if h.ndim == 4:
            h_pooled = h[0].mean(dim=1)  # (seq, H)
        elif h.ndim == 3:
            h_pooled = h[0] if h.shape[0] == 1 else h.mean(dim=0)
        else:
            h_pooled = h
        baseline_hidden.append(h_pooled)

    baseline_preds = np.concatenate(baseline_preds_list, axis=0)
    baseline_row_loss = compute_per_row_loss(y_q, baseline_preds, task)
    logger.info("    %d rows done", n_query)

    # SAE encode each row's query position to find alive features
    logger.info("  Phase 2: SAE encode → alive features...")
    all_h_encoded = []
    for row_idx in range(n_query):
        query_emb = baseline_hidden[row_idx][-1:]  # last position = query
        with torch.no_grad():
            x_norm = (query_emb - data_mean) / data_std
            encoded = sae.encode(x_norm)
        all_h_encoded.append(encoded[0])

    h_encoded = torch.stack(all_h_encoded)  # (n_query, hidden_dim)
    firing_mask = (h_encoded > 0).cpu().numpy()
    alive_mask = firing_mask.any(axis=0)
    alive_features = np.where(alive_mask)[0].tolist()
    n_alive = len(alive_features)
    feature_n_firing = np.array([firing_mask[:, fi].sum() for fi in alive_features])
    logger.info("    %d alive features", n_alive)

    # Phase 3: batched LOO per row
    # For each row, build a tail with K query copies, inject K deltas at once
    logger.info("  Phase 3: batched LOO ablation...")
    row_feature_drops = np.zeros((n_query, n_alive))

    del clf  # free memory, we'll reload per row with K copies
    torch.cuda.empty_cache()

    t0 = time.time()
    for row_idx in range(n_query):
        x_row = X_q[row_idx:row_idx + 1]

        # Which features fire?
        row_firing = [i for i, fi in enumerate(alive_features) if firing_mask[row_idx, fi]]
        if not row_firing:
            continue

        K = len(row_firing)
        h = baseline_hidden[row_idx]  # (seq, H) mean-pooled

        # Compute K deltas in SAE space
        with torch.no_grad():
            x_norm = (h - data_mean) / data_std
            h_full = sae.encode(x_norm)
            recon_full = sae.decode(h_full)

        # Build tail with K copies of this query row
        X_batch = np.tile(x_row, (K, 1))
        tail = TabPFNTail.from_data(
            X_ctx, y_ctx, X_batch, extraction_layer, task, device,
        )

        # Compute combined delta: for each of K copies, ablate one feature
        # tail.hidden_state is (1, ctx+K, n_structure, H)
        # We need delta of shape (ctx+K, H) — same context delta, different per-query
        with torch.no_grad():
            # Context portion: use first feature's delta (context is same for all)
            fi0 = alive_features[row_firing[0]]
            h_abl0 = h_full.clone()
            h_abl0[:, fi0] = 0.0
            recon_abl0 = sae.decode(h_abl0)
            full_delta0 = (recon_abl0 - recon_full) * data_std  # (seq, H)

            ctx_len = tail.single_eval_pos
            ctx_delta = full_delta0[:ctx_len]  # (ctx, H)

            # Per-query deltas
            query_deltas = []
            for col_idx in row_firing:
                fi = alive_features[col_idx]
                h_abl = h_full.clone()
                h_abl[:, fi] = 0.0
                recon_abl = sae.decode(h_abl)
                d = ((recon_abl - recon_full) * data_std)[-1]  # (H,) query position
                query_deltas.append(d)

            query_delta_stack = torch.stack(query_deltas)  # (K, H)
            combined_delta = torch.cat([ctx_delta, query_delta_stack], dim=0)  # (ctx+K, H)

        # Inject into tail's cached state and predict
        state = tail.hidden_state.clone()
        state[0] += combined_delta.unsqueeze(1)  # broadcast across n_structure
        preds = tail._predict_with_modified_state(state)

        # Compute per-ablation loss
        baseline_loss = baseline_row_loss[row_idx]
        y_tiled = np.full(K, y_q[row_idx])
        ablated_losses = compute_per_row_loss(y_tiled, preds, task)

        for j, col_idx in enumerate(row_firing):
            row_feature_drops[row_idx, col_idx] = ablated_losses[j] - baseline_loss

        del tail
        torch.cuda.empty_cache()

        if (row_idx + 1) % 50 == 0 or row_idx == n_query - 1:
            elapsed = time.time() - t0
            rate = (row_idx + 1) / elapsed
            eta = (n_query - row_idx - 1) / rate
            n_pos = (row_feature_drops[row_idx] > 0).sum()
            logger.info("    row %d/%d: %d firing, %d helpful (%.1f rows/s, ETA %.0fs)",
                        row_idx + 1, n_query, K, n_pos, rate, eta)

    logger.info("  Done in %.1fs", time.time() - t0)

    return {
        "row_feature_drops": row_feature_drops,
        "feature_indices": np.array(alive_features),
        "feature_n_firing": feature_n_firing,
        "baseline_preds": baseline_preds,
        "y_query": y_q,
        "extraction_layer": extraction_layer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()

    datasets = {k: v for k, v in EVAL_DATASETS.items()
                if args.datasets is None or k in args.datasets}

    for ds_name, ds_info in datasets.items():
        task = ds_info["task"]
        print(f"\n{'=' * 70}")
        print(f"  {ds_name} ({task})")
        print("=" * 70)

        tabpfn_data = load_preprocessed("tabpfn", ds_name, CACHE_DIR)
        X_ctx, y_ctx = tabpfn_data.X_train[:600], tabpfn_data.y_train[:600]
        X_q, y_q = tabpfn_data.X_test[:500], tabpfn_data.y_test[:500]

        for var_name, var_info in VARIANTS.items():
            print(f"\n  --- {var_name} ---")

            sae, config = load_sae(var_info["sae_path"], args.device)
            data_mean, data_std, layer = load_norm_stats(
                var_info["stats_path"], ds_name, args.device)
            logger.info("  Layer: L%d", layer)

            result = compute_perrow_importance(
                sae, data_mean, data_std, layer,
                X_ctx, y_ctx, X_q, y_q, task, args.device,
            )

            out_path = OUTPUT_DIR / f"perrow_importance_{var_name}_{ds_name}.npz"
            np.savez_compressed(
                str(out_path),
                row_feature_drops=result["row_feature_drops"],
                feature_indices=result["feature_indices"],
                feature_n_firing=result["feature_n_firing"],
                baseline_preds=result["baseline_preds"],
                y_query=result["y_query"],
                extraction_layer=np.array(result["extraction_layer"]),
            )
            print(f"  → {out_path.name}")

            rd = result["row_feature_drops"]
            mean_drops = rd.mean(axis=0)
            n_helpful = (mean_drops > 0).sum()
            print(f"  Alive: {len(result['feature_indices'])}, "
                  f"Helpful: {n_helpful}, "
                  f"Max importance: {mean_drops.max():.4f}")


if __name__ == "__main__":
    main()
