"""
Common utilities for TabArena dataset loading and preprocessing.

Provides standardized preprocessing that should be used across ALL analysis scripts.
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from data.extended_loader import TABARENA_DATASETS


# Embedding directory name mapping
# Maps sweep model names to actual embedding directory names
EMB_DIR_MAP = {
    'tabpfn': 'tabpfn',  # Sweep uses 'tabpfn', maps to 'tabpfn' directory
    'tabpfn_layer16': 'tabpfn',
    'tabicl': 'tabicl',  # Sweep uses 'tabicl', maps to 'tabicl' directory
    'tabicl_layer10': 'tabicl',
    'mitra': 'mitra',  # Sweep uses 'mitra', maps to 'mitra' directory
    'mitra_layer12': 'mitra',
    'tabdpt': 'tabdpt',  # Sweep uses 'tabdpt', maps to 'tabdpt' directory
    'tabdpt_layer14': 'tabdpt',
    'carte': 'carte',  # Sweep uses 'carte', maps to 'carte' directory
    'carte_layer1': 'carte',
    'hyperfast': 'hyperfast',  # Sweep uses 'hyperfast', maps to 'hyperfast' directory
    'hyperfast_layer2': 'hyperfast',
    'tabula8b': 'tabula8b',  # Sweep uses 'tabula8b', maps to 'tabula8b' directory
    'tabula8b_layer21': 'tabula8b_layer21_ctx600',  # Exception: uses ctx600 variant
}


def get_embedding_dir(model_name: str) -> str:
    """Get the standard embedding directory name for a model."""
    return EMB_DIR_MAP.get(model_name, model_name)


def get_tabarena_splits() -> Tuple[List[str], List[str]]:
    """Get standard train/test split for TabArena classification datasets.

    Returns:
        train_datasets: First 34 classification datasets (sorted)
        test_datasets: Remaining classification datasets (sorted)
    """
    all_datasets = sorted([k for k, v in TABARENA_DATASETS.items()
                          if v['task'] == 'classification'])
    train_datasets = all_datasets[:34]
    test_datasets = all_datasets[34:]
    return train_datasets, test_datasets


def load_embeddings_raw(
    model_name: str,
    datasets: List[str],
    max_per_dataset: int = 500,
    base_dir: Path = None,
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """Load raw (unnormalized) embeddings from multiple datasets.

    Args:
        model_name: Model identifier (e.g., 'tabpfn', 'mitra_layer12')
        datasets: List of dataset names to load
        max_per_dataset: Maximum samples per dataset (for memory)
        base_dir: Base directory (defaults to output/embeddings/tabarena)

    Returns:
        pooled: (n_total, dim) concatenated embeddings
        offsets: {dataset_name: (start_row, end_row)} row offsets
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / 'output' / 'embeddings' / 'tabarena'

    emb_dir_name = get_embedding_dir(model_name)
    emb_dir = base_dir / emb_dir_name

    all_embs = []
    offsets = {}
    cursor = 0

    for ds_name in datasets:
        emb_path = emb_dir / f"tabarena_{ds_name}.npz"
        if not emb_path.exists():
            continue

        data = np.load(emb_path, allow_pickle=True)
        emb = data['embeddings'].astype(np.float32)

        if len(emb) > max_per_dataset:
            np.random.seed(42)
            idx = np.random.choice(len(emb), max_per_dataset, replace=False)
            emb = emb[idx]

        offsets[ds_name] = (cursor, cursor + len(emb))
        cursor += len(emb)
        all_embs.append(emb)

    if not all_embs:
        raise ValueError(f"No embeddings found for {model_name} in {emb_dir}")

    pooled = np.concatenate(all_embs, axis=0)
    return pooled, offsets


def compute_normalization_stats(embeddings: np.ndarray) -> np.ndarray:
    """Compute TabArena standard normalization statistics.

    CANONICAL PREPROCESSING (used by SAE sweeps):
    - Per-dimension standard deviation with floor at 1e-8
    - NO mean centering (embeddings are NOT zero-centered)

    This matches what our SAE models were trained on.

    Args:
        embeddings: (n_samples, dim) raw embeddings

    Returns:
        std: (1, dim) per-dimension standard deviation
    """
    std = embeddings.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return std


def normalize_embeddings(
    embeddings: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply TabArena standard normalization.

    CANONICAL PREPROCESSING (used by SAE sweeps):
    - Divide by std
    - NO mean subtraction

    Args:
        embeddings: (n_samples, dim) raw embeddings
        std: (1, dim) standard deviation from compute_normalization_stats

    Returns:
        normalized: (n_samples, dim) std-normalized embeddings (NOT mean-centered)
    """
    return embeddings / std


def load_and_normalize_embeddings(
    model_name: str,
    train_datasets: List[str] = None,
    test_datasets: List[str] = None,
    max_per_dataset: int = 500,
) -> Dict[str, np.ndarray]:
    """Load and normalize embeddings with CANONICAL TabArena preprocessing.

    CANONICAL: std normalization ONLY, NO mean centering (matches SAE training).
    Normalization stats computed from train data, applied to both train/test.

    Args:
        model_name: Model identifier
        train_datasets: Training dataset names (defaults to standard split)
        test_datasets: Test dataset names (defaults to standard split)
        max_per_dataset: Max samples per dataset

    Returns:
        Dictionary with keys:
            'train': Normalized train embeddings
            'test': Normalized test embeddings (if test_datasets provided)
            'train_std': Normalization std (for reference)
    """
    if train_datasets is None:
        train_datasets, test_datasets_default = get_tabarena_splits()
        if test_datasets is None:
            test_datasets = test_datasets_default

    # Load train data
    train_raw, _ = load_embeddings_raw(model_name, train_datasets, max_per_dataset)

    # Compute normalization from train
    train_std = compute_normalization_stats(train_raw)

    # Normalize train
    train_normalized = normalize_embeddings(train_raw, train_std)

    result = {
        'train': train_normalized,
        'train_std': train_std,
    }

    # Load and normalize test if requested
    if test_datasets:
        test_raw, _ = load_embeddings_raw(model_name, test_datasets, max_per_dataset)
        test_normalized = normalize_embeddings(test_raw, train_std)
        result['test'] = test_normalized

    return result
