#!/usr/bin/env python3
"""Measure CKA shift after ablation: does ablating strong model's unmatched
concepts make its embeddings more similar to the weak model?

For each (weak, strong, dataset) in the ablation sweep:
  1. Load strong + weak test embeddings (already saved from SAE training)
  2. Reconstruct ablated strong embeddings using per-row concept selections
     from the saved NPZ (pure SAE encode/decode, no model inference needed)
  3. Compute linear CKA before and after ablation

Output: output/cka_post_ablation.json

Usage:
    python -m scripts.analysis.cka_post_ablation
"""
import json
import logging
from pathlib import Path

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import (
    load_test_embeddings,
    load_sae,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SWEEP_DIR = PROJECT_ROOT / "output" / "ablation_sweep"
OUTPUT_PATH = PROJECT_ROOT / "output" / "cka_post_ablation.json"
DEVICE = "cpu"  # pure matrix ops, no GPU needed

_NORM_SEARCH_DIRS = [
    PROJECT_ROOT / "output" / "sae_training_round10",
    PROJECT_ROOT / "output" / "sae_training_round6",
]


def load_norm_stats_with_fallback(model_key: str, dataset: str) -> tuple:
    """Load per-dataset norm stats, falling back round10 → round6."""
    for search_dir in _NORM_SEARCH_DIRS:
        candidates = sorted(search_dir.glob(f"{model_key}_*_norm_stats.npz"))
        if not candidates:
            continue
        data = np.load(candidates[0], allow_pickle=True)
        datasets = list(data["datasets"])
        if dataset not in datasets:
            continue
        idx = datasets.index(dataset)
        mean = torch.tensor(data["means"][idx], dtype=torch.float32)
        std = torch.tensor(data["stds"][idx], dtype=torch.float32)
        return mean, std
    raise FileNotFoundError(f"No norm stats found for {model_key}/{dataset}")


# ---------------------------------------------------------------------------
# Linear CKA
# ---------------------------------------------------------------------------

def _gram(X: np.ndarray) -> np.ndarray:
    """Centred Gram matrix."""
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ (X @ X.T) @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two (n, d) matrices."""
    if X.shape[0] != Y.shape[0] or X.shape[0] < 4:
        return float("nan")
    Kx = _gram(X)
    Ky = _gram(Y)
    num = np.sum(Kx * Ky)
    denom = np.sqrt(np.sum(Kx * Kx) * np.sum(Ky * Ky))
    return float(num / denom) if denom > 1e-12 else float("nan")


# ---------------------------------------------------------------------------
# Ablation delta (no model inference)
# ---------------------------------------------------------------------------

def compute_ablated_embeddings(
    emb: np.ndarray,
    selected_features: np.ndarray,
    sae,
) -> np.ndarray:
    """Return ablated embeddings for each row.

    test_embeddings are already normalized (per-dataset StandardScaler applied
    during SAE training data construction). The SAE encodes/decodes in that
    normalized space, so the delta stays in normalized space — no std scaling.

    Args:
        emb: (n, d) already-normalized test embeddings for the strong model
        selected_features: (n, max_k) per-row selected feature indices
            (rows may be padded with -1 or 0 beyond the per-row k)
        sae: loaded SAE model

    Returns:
        emb_ablated: (n, d) ablated normalized embeddings
    """
    x = torch.tensor(emb, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        h = sae.encode(x)               # (n, dict_size) — x already normalized

        # Zero selected features per row
        h_ablated = h.clone()
        for i, feats in enumerate(selected_features):
            feats_valid = [int(f) for f in feats if f >= 0]
            if feats_valid:
                h_ablated[i, feats_valid] = 0.0

        # Delta in normalized space
        recon_full = sae.decode(h)
        recon_ablated = sae.decode(h_ablated)
        delta = recon_ablated - recon_full

    emb_ablated = emb + delta.cpu().numpy()
    return emb_ablated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_pair(pair_dir: Path, all_embs: dict) -> list[dict]:
    """Process one (model_a vs model_b) directory, return per-dataset records."""
    records = []
    npz_files = sorted(pair_dir.glob("*.npz"))
    if not npz_files:
        return records

    # Load SAEs lazily per model
    sae_cache: dict = {}

    for npz_path in npz_files:
        dataset = npz_path.stem
        try:
            d = np.load(npz_path, allow_pickle=True)
            strong_model = str(d["strong_model"])
            weak_model = str(d["weak_model"])
            selected_features = d["selected_features"]   # (n, max_k)
            row_indices = d["row_indices"]               # (n,) absolute indices

            # Get embeddings for this dataset
            strong_embs = all_embs.get(strong_model, {}).get(dataset)
            weak_embs = all_embs.get(weak_model, {}).get(dataset)
            if strong_embs is None or weak_embs is None:
                logger.warning(f"  Missing embeddings for {dataset} ({strong_model}/{weak_model})")
                continue

            # Align to the rows used in ablation
            n = len(row_indices)
            if n < 4:
                continue

            # row_indices are absolute — map to positions in the saved emb array.
            # Embeddings are stored in test-set order (same order as row_indices
            # were collected), so we just take sequential slices.
            emb_s = strong_embs[:n]  # (n, d_strong)
            emb_w = weak_embs[:n]    # (n, d_weak)

            # Load SAE for strong model
            if strong_model not in sae_cache:
                try:
                    sae, _ = load_sae(strong_model, device=DEVICE)
                    sae.eval()
                    sae_cache[strong_model] = sae
                    logger.info(f"  Loaded SAE for {strong_model}")
                except Exception as e:
                    logger.warning(f"  Cannot load SAE for {strong_model}: {e}")
                    sae_cache[strong_model] = None

            sae = sae_cache[strong_model]
            if sae is None:
                continue

            # Compute ablated embeddings
            emb_s_ablated = compute_ablated_embeddings(emb_s, selected_features, sae)

            # CKA before and after
            cka_before = linear_cka(emb_s, emb_w)
            cka_after = linear_cka(emb_s_ablated, emb_w)

            records.append({
                "strong_model": strong_model,
                "weak_model": weak_model,
                "dataset": dataset,
                "cka_before": cka_before,
                "cka_after": cka_after,
                "delta_cka": cka_after - cka_before,
                "n_rows": n,
            })
            logger.info(
                f"  {dataset}: CKA {cka_before:.3f} → {cka_after:.3f} "
                f"(Δ={cka_after - cka_before:+.3f})"
            )

        except Exception as e:
            logger.warning(f"  Error processing {dataset}: {e}")

    return records


def main():
    # Pre-load all model embeddings (avoids re-loading for every pair)
    models = set()
    for pair_dir in SWEEP_DIR.iterdir():
        if not pair_dir.is_dir():
            continue
        for npz_path in pair_dir.glob("*.npz"):
            d = np.load(npz_path, allow_pickle=True)
            models.add(str(d["strong_model"]))
            models.add(str(d["weak_model"]))
            break  # one file is enough to get model names

    logger.info(f"Loading embeddings for {len(models)} models: {sorted(models)}")
    all_embs: dict = {}
    for m in sorted(models):
        try:
            all_embs[m] = load_test_embeddings(m)
            logger.info(f"  {m}: {len(all_embs[m])} datasets")
        except Exception as e:
            logger.warning(f"  Cannot load embeddings for {m}: {e}")
            all_embs[m] = {}

    # Process each pair
    all_records = []
    pair_dirs = sorted(d for d in SWEEP_DIR.iterdir() if d.is_dir())
    for pair_dir in pair_dirs:
        logger.info(f"Processing {pair_dir.name}")
        records = process_pair(pair_dir, all_embs)
        all_records.extend(records)
        logger.info(f"  → {len(records)} datasets processed")

    # Aggregate by model pair
    from collections import defaultdict
    pair_stats: dict = defaultdict(list)
    for r in all_records:
        key = f"{r['weak_model']} vs {r['strong_model']}"
        pair_stats[key].append(r)

    summary = {}
    for pair, recs in sorted(pair_stats.items()):
        deltas = [r["delta_cka"] for r in recs if not np.isnan(r["delta_cka"])]
        befores = [r["cka_before"] for r in recs if not np.isnan(r["cka_before"])]
        afters = [r["cka_after"] for r in recs if not np.isnan(r["cka_after"])]
        summary[pair] = {
            "n_datasets": len(recs),
            "mean_cka_before": float(np.mean(befores)) if befores else None,
            "mean_cka_after": float(np.mean(afters)) if afters else None,
            "mean_delta_cka": float(np.mean(deltas)) if deltas else None,
            "pct_improved": float(np.mean([d > 0 for d in deltas])) if deltas else None,
        }
        logger.info(
            f"{pair}: CKA {summary[pair]['mean_cka_before']:.3f} → "
            f"{summary[pair]['mean_cka_after']:.3f} "
            f"(Δ={summary[pair]['mean_delta_cka']:+.3f}, "
            f"{summary[pair]['pct_improved']*100:.0f}% improved, "
            f"n={len(recs)})"
        )

    output = {"summary": summary, "per_dataset": all_records}
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    logger.info(f"\nSaved {len(all_records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
