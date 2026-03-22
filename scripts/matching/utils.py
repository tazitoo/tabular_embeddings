"""Shared utilities for SAE concept matching pipeline.

Provides embedding loading, normalization, activation computation,
and alive-mask helpers used across matching and concept scripts.
"""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.sae.compare_sae_cross_model import DEFAULT_SAE_ROUND
from analysis.sparse_autoencoder import SparseAutoencoder

EMB_DIR = PROJECT_ROOT / "output" / "embeddings" / "tabarena"
SAE_DATA_DIR = PROJECT_ROOT / "output" / f"sae_training_round{DEFAULT_SAE_ROUND}"


def load_norm_stats(model_key: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load per-dataset normalization stats for a model.

    Returns:
        Dict mapping dataset name → (mean, std), each shape (emb_dim,).
    """
    candidates = sorted(SAE_DATA_DIR.glob(f"{model_key}_*_norm_stats.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No norm stats for '{model_key}' in {SAE_DATA_DIR}"
        )
    data = np.load(candidates[0], allow_pickle=True)
    datasets = list(data["datasets"])
    means = data["means"]  # (n_datasets, emb_dim)
    stds = data["stds"]
    return {ds: (means[i], stds[i]) for i, ds in enumerate(datasets)}


def _unpool_split(path: Path) -> Dict[str, np.ndarray]:
    """Unpool a concatenated split NPZ into per-dataset arrays."""
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"]
    samples_per_dataset = data["samples_per_dataset"]

    result = {}
    offset = 0
    for ds_name, count in samples_per_dataset:
        ds_name = str(ds_name)
        count = int(count)
        result[ds_name] = embeddings[offset:offset + count]
        offset += count
    return result


def load_test_embeddings(model_key: str) -> Dict[str, np.ndarray]:
    """Load per-dataset test-split embeddings (already normalized).

    The SAE training pipeline saves a 70/30 row-level split with per-dataset
    StandardScaler normalization applied using train-split stats. This loads
    the 30% held-out test split and unpools it into per-dataset arrays.

    Returns:
        Dict mapping dataset name → embeddings array (n_test_rows, emb_dim).
    """
    candidates = sorted(SAE_DATA_DIR.glob(f"{model_key}_*_sae_test.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No test data for '{model_key}' in {SAE_DATA_DIR}"
        )
    return _unpool_split(candidates[0])


def load_train_embeddings(model_key: str) -> Dict[str, np.ndarray]:
    """Load per-dataset train-split embeddings (already normalized).

    Used to determine alive masks — the SAE was trained on this data,
    so it's the authoritative source for which features are alive.

    Returns:
        Dict mapping dataset name → embeddings array (n_train_rows, emb_dim).
    """
    candidates = sorted(SAE_DATA_DIR.glob(f"{model_key}_*_sae_training.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No training data for '{model_key}' in {SAE_DATA_DIR}"
        )
    return _unpool_split(candidates[0])


def load_embeddings(
    emb_dir: Path, dataset: str, max_per_dataset: int = 500,
    norm_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """Load embeddings for a dataset, subsampled and optionally normalized.

    Note: Prefer load_test_embeddings() for matching — it uses the held-out
    30% test split that the SAE never saw during training.
    """
    path = emb_dir / f"tabarena_{dataset}.npz"
    data = np.load(path, allow_pickle=True)
    emb = data["embeddings"].astype(np.float32)
    if len(emb) > max_per_dataset:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(emb), max_per_dataset, replace=False)
        emb = emb[idx]
    if norm_stats is not None and dataset in norm_stats:
        mean, std = norm_stats[dataset]
        std = std.copy()
        std[std < 1e-8] = 1.0
        emb = (emb - mean) / std
    return emb


def compute_sae_activations(
    model: SparseAutoencoder, embeddings: np.ndarray
) -> np.ndarray:
    """Encode normalized embeddings through SAE, return activations (n_samples, hidden_dim)."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(embeddings, dtype=torch.float32)
        h = model.encode(x).numpy()
    return h


def get_alive_mask(activations: np.ndarray, threshold: float = 0.001) -> np.ndarray:
    """Boolean mask of features whose max activation exceeds threshold."""
    return activations.max(axis=0) > threshold


def compute_alive_mask(
    sae: SparseAutoencoder,
    train_embs: Dict[str, np.ndarray],
    threshold: float = 0.001,
) -> np.ndarray:
    """Compute alive mask from training data (authoritative source).

    A feature is alive if it activates above threshold on any training row.
    Using training data (which the SAE was trained on) ensures we capture
    all features the SAE learned, independent of test-set sampling.

    Returns:
        Boolean mask of shape (hidden_dim,).
    """
    all_acts = []
    for ds in sorted(train_embs.keys()):
        acts = compute_sae_activations(sae, train_embs[ds])
        all_acts.append(acts)
    pooled = np.concatenate(all_acts, axis=0)
    return get_alive_mask(pooled, threshold)
