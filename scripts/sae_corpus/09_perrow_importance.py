#!/usr/bin/env python3
"""Compute per-row, per-feature importance via batched single-row ablation.

For each query row, creates K copies (one per firing feature), each with
one feature ablated, and runs a single forward pass to get all K ablated
predictions at once.  This is causal: each row sees only its own context,
not other query rows.

Output:
    output/sae_training_round10/perrow_importance_{variant}_{dataset}.npz
        row_feature_drops:  (n_query, n_alive) per-row loss change
        feature_indices:    (n_alive,) SAE feature indices
        feature_n_firing:   (n_alive,) count of rows where feature fires
        baseline_preds:     (n_query,) or (n_query, n_classes)
        y_query:            (n_query,)
        extraction_layer:   int

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
    """Batched single-row LOO importance.

    For each query row:
      1. Forward pass with 1 query row → baseline prediction + hidden state
      2. SAE-encode hidden → find K firing features
      3. Create K copies of the query, each with 1 feature ablated
      4. Single batched forward pass → K ablated predictions
      5. Per-row loss change = ablated_loss - baseline_loss
    """
    from models.tabpfn_utils import load_tabpfn

    n_query = len(X_q)

    # Fit once
    clf = load_tabpfn(task=task, device=device, n_estimators=1)
    clf.fit(X_ctx, y_ctx)
    model = clf.model_ if hasattr(clf, "model_") else clf.transformer_
    layers = model.transformer_encoder.layers

    # First pass: get all baseline predictions + hidden states per row
    # We do this one row at a time to get the correct single-row hidden state
    logger.info("  Phase 1: baseline predictions (1 row at a time)...")
    baseline_preds_list = []
    hidden_states = []  # (n_query,) list of (seq, H) tensors

    t0 = time.time()
    for row_idx in range(n_query):
        x_row = X_q[row_idx:row_idx + 1]
        captured = {}

        def capture_hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            if isinstance(out, torch.Tensor):
                captured["hidden"] = out.detach()

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

        h = captured["hidden"]
        if h.ndim == 4:
            h = h[0].mean(dim=1)
        elif h.ndim == 3:
            h = h[0] if h.shape[0] == 1 else h.mean(dim=0)
        hidden_states.append(h)

    baseline_time = time.time() - t0
    logger.info("    %d rows in %.1fs (%.1f rows/s)",
                n_query, baseline_time, n_query / baseline_time)

    # Stack baseline predictions
    baseline_preds = np.concatenate(baseline_preds_list, axis=0)
    baseline_row_loss = compute_per_row_loss(y_q, baseline_preds, task)

    # Determine alive features across all rows
    logger.info("  Phase 2: SAE encode + find alive features...")
    # Encode each row's hidden state through SAE
    all_h_encoded = []
    for row_idx in range(n_query):
        h = hidden_states[row_idx]
        query_h = h[-1:]  # last position = query row
        with torch.no_grad():
            x_norm = (query_h - data_mean) / data_std
            encoded = sae.encode(x_norm)
        all_h_encoded.append(encoded[0])  # (hidden_dim,)

    h_encoded = torch.stack(all_h_encoded)  # (n_query, hidden_dim)
    firing_mask = (h_encoded > 0).cpu().numpy()  # (n_query, hidden_dim)

    # Alive features: fire on at least one row
    alive_mask = firing_mask.any(axis=0)
    alive_features = np.where(alive_mask)[0].tolist()
    n_alive = len(alive_features)
    logger.info("    %d alive features (of %d)", n_alive, sae.config.hidden_dim)

    # Per-row firing counts
    feature_n_firing = np.array([firing_mask[:, fi].sum() for fi in alive_features])

    # Phase 3: batched LOO ablation per row
    logger.info("  Phase 3: batched LOO ablation (%d rows × up to %d features)...",
                n_query, n_alive)
    row_feature_drops = np.zeros((n_query, n_alive))

    t0 = time.time()
    for row_idx in range(n_query):
        h = hidden_states[row_idx]  # (seq, H) — context + 1 query
        seq_len = h.shape[0]

        # Which features fire on this row?
        row_firing = [i for i, fi in enumerate(alive_features) if firing_mask[row_idx, fi]]
        if not row_firing:
            continue

        K = len(row_firing)

        # Compute K deltas: one per firing feature
        # Each delta zeroes one feature in the SAE encoding of the full sequence
        with torch.no_grad():
            x_norm = (h - data_mean) / data_std
            h_full = sae.encode(x_norm)  # (seq, hidden_dim)
            recon_full = sae.decode(h_full)  # (seq, emb_dim)

            # Stack K deltas
            deltas = []
            for col_idx in row_firing:
                fi = alive_features[col_idx]
                h_abl = h_full.clone()
                h_abl[:, fi] = 0.0
                recon_abl = sae.decode(h_abl)
                delta = (recon_abl - recon_full) * data_std  # (seq, emb_dim)
                deltas.append(delta)

        # Batch forward pass: K copies of this query row, each with different delta
        # Stack query rows
        x_row = X_q[row_idx:row_idx + 1]
        X_batch = np.tile(x_row, (K, 1))

        # The hook needs to add a different delta per batch element
        # But TabPFN processes all queries together with shared context
        # So we pass K query rows and add per-row deltas
        delta_stack = torch.stack(deltas)  # (K, seq, emb_dim)

        def make_hook(d_stack, seq_l):
            def hook(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                if isinstance(out, torch.Tensor):
                    out = out.clone()
                    if out.ndim == 4:
                        # (1, ctx+K, n_feat+1, H) — context is shared, K query rows
                        # delta_stack is (K, ctx+1, H) — we need to map to (K, H)
                        # Context delta: use first delta (they're nearly identical for context)
                        ctx_len = out.shape[1] - d_stack.shape[0]
                        # Add context delta from first ablation
                        out[0, :ctx_len] += d_stack[0, :ctx_len].unsqueeze(1)
                        # Add per-query deltas (last position of each delta)
                        for k in range(d_stack.shape[0]):
                            out[0, ctx_len + k] += d_stack[k, -1:].unsqueeze(0)
                    if isinstance(output, tuple):
                        return (out,) + output[1:]
                    return out
                return output
            return hook

        handle = layers[extraction_layer].register_forward_hook(
            make_hook(delta_stack, seq_len))
        try:
            with torch.no_grad():
                if task == "regression":
                    preds = clf.predict(X_batch)
                else:
                    preds = clf.predict_proba(X_batch)
        finally:
            handle.remove()

        preds_np = np.asarray(preds)

        # Compute per-ablation loss
        y_tiled = np.tile(y_q[row_idx], K)
        ablated_losses = compute_per_row_loss(y_tiled, preds_np, task)
        baseline_loss = baseline_row_loss[row_idx]

        for j, col_idx in enumerate(row_firing):
            row_feature_drops[row_idx, col_idx] = ablated_losses[j] - baseline_loss

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

            # Save
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

            # Summary
            rd = result["row_feature_drops"]
            mean_drops = rd.mean(axis=0)
            n_helpful = (mean_drops > 0).sum()
            print(f"  Alive: {len(result['feature_indices'])}, "
                  f"Helpful: {n_helpful}, "
                  f"Max importance: {mean_drops.max():.4f}")


if __name__ == "__main__":
    main()
