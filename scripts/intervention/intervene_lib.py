"""Shared intervention library for importance, ablation, and transfer.

Owns the core utilities that all intervention scripts share:
- SAE loading, norm stats, extraction layer config
- Loss metrics (per-row and dataset-level)
- Feature metadata (alive features, labels, bands)
- Data loading and row alignment
- Delta computation and batched ablation

Tail classes remain in intervene_sae.py (1000+ lines, 16+ importers).

Usage:
    from scripts.intervention.intervene_lib import (
        load_sae, load_norm_stats, get_extraction_layer, build_tail,
        compute_per_row_loss, compute_importance_metric,
        get_alive_features, get_feature_labels,
        load_dataset_context, align_test_rows,
        compute_feature_deltas, batched_ablation,
    )
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.sae.compare_sae_cross_model import DEFAULT_SAE_ROUND, SAE_FILENAME, sae_sweep_dir

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

DEFAULT_SAE_DIR = sae_sweep_dir()
DEFAULT_TRAINING_DIR = PROJECT_ROOT / "output" / f"sae_training_round{DEFAULT_SAE_ROUND}"
DEFAULT_LAYERS_PATH = PROJECT_ROOT / "config" / "optimal_extraction_layers.json"
DEFAULT_CONCEPT_LABELS = PROJECT_ROOT / "output" / f"cross_model_concept_labels_round{DEFAULT_SAE_ROUND}.json"
SPLITS_PATH = PROJECT_ROOT / "output" / "sae_training_round9" / "tabarena_splits.json"
SAE_DATA_DIR = PROJECT_ROOT / "output" / f"sae_training_round{DEFAULT_SAE_ROUND}"

MODEL_KEYS = {
    "tabpfn": "tabpfn", "mitra": "mitra", "tabicl": "tabicl",
    "tabicl_v2": "tabicl_v2", "tabdpt": "tabdpt", "hyperfast": "hyperfast",
    "carte": "carte", "tabula8b": "tabula8b",
}

MODEL_KEY_TO_LABEL_KEY = {
    "tabpfn": "TabPFN", "mitra": "Mitra", "tabicl": "TabICL",
    "tabicl_v2": "TabICL-v2", "tabdpt": "TabDPT", "hyperfast": "HyperFast",
    "carte": "CARTE", "tabula8b": "Tabula-8B",
}


# ── SAE loading ──────────────────────────────────────────────────────────────


def load_sae(model_key: str, sae_dir: Path = DEFAULT_SAE_DIR, device: str = "cuda"):
    """Load a trained Matryoshka-Archetypal SAE for the given model.

    Handles archetypal SAE extra parameters (archetype_logits, archetype_deviation,
    reference_data) that must be registered before load_state_dict.

    Returns:
        (sae_model, sae_config) tuple with model in eval mode on device.
    """
    from analysis.sparse_autoencoder import SparseAutoencoder, SAEConfig

    ckpt_path = sae_dir / model_key / SAE_FILENAME
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    if not isinstance(config, SAEConfig):
        config = SAEConfig(**config)

    sae = SparseAutoencoder(config)

    state_dict = ckpt["model_state_dict"]
    if "reference_data" in state_dict and state_dict["reference_data"] is not None:
        sae.register_buffer("reference_data", state_dict["reference_data"])
        if "archetype_logits" in state_dict:
            sae.archetype_logits = torch.nn.Parameter(state_dict["archetype_logits"])
        if "archetype_deviation" in state_dict:
            sae.archetype_deviation = torch.nn.Parameter(state_dict["archetype_deviation"])

    sae.load_state_dict(state_dict, strict=False)
    sae.to(device)
    sae.eval()
    return sae, config


def get_extraction_layer(model_key: str, layers_path: Path = DEFAULT_LAYERS_PATH) -> int:
    """Get the optimal extraction layer index for a model."""
    with open(layers_path, encoding="utf-8") as f:
        layers_config = json.load(f)
    return layers_config[model_key]["optimal_layer"]


def load_norm_stats(
    model_key: str,
    dataset_name: str,
    training_dir: Path = DEFAULT_TRAINING_DIR,
    device: str = "cuda",
) -> tuple:
    """Load per-dataset normalization stats (mean, std) as tensors on device."""
    candidates = sorted(training_dir.glob(f"{model_key}_*_norm_stats.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No norm stats for '{model_key}' in {training_dir}."
        )
    stats = np.load(candidates[0])
    datasets = list(stats["datasets"])
    if dataset_name not in datasets:
        raise ValueError(
            f"Dataset '{dataset_name}' not in norm_stats ({len(datasets)} datasets)."
        )
    idx = datasets.index(dataset_name)
    mean = torch.tensor(stats["means"][idx], dtype=torch.float32, device=device)
    std = torch.tensor(stats["stds"][idx], dtype=torch.float32, device=device)
    return mean, std


def load_training_mean(
    model_key: str,
    training_dir: Path = DEFAULT_TRAINING_DIR,
    device: str = "cuda",
) -> torch.Tensor:
    """DEPRECATED: Use load_norm_stats() instead."""
    import warnings
    warnings.warn(
        "load_training_mean() returns ~0 for StandardScaler-normalized data. "
        "Use load_norm_stats(model_key, dataset_name) instead.",
        DeprecationWarning, stacklevel=2,
    )
    candidates = sorted(training_dir.glob(f"{model_key}_*_sae_training.npz"))
    if not candidates:
        raise FileNotFoundError(f"No training data for '{model_key}' in {training_dir}.")
    data = np.load(candidates[0])
    mean = data["embeddings"].mean(axis=0)
    return torch.tensor(mean, dtype=torch.float32, device=device)


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_importance_metric(y_true: np.ndarray, preds: np.ndarray, task: str) -> tuple:
    """Dataset-level metric: AUC (binary), neg_logloss (multiclass), neg_RMSE (regression).

    Returns (metric_value, metric_name) where higher is always better.
    """
    from sklearn.metrics import roc_auc_score, log_loss

    if task == "regression":
        rmse = float(np.sqrt(np.mean((preds - y_true) ** 2)))
        return -rmse, "neg_rmse"

    n_classes = preds.shape[1] if preds.ndim == 2 else len(np.unique(y_true))
    if n_classes == 2:
        proba = preds[:, 1] if preds.ndim == 2 else preds
        try:
            return float(roc_auc_score(y_true, proba)), "auc"
        except ValueError:
            return float(-log_loss(y_true, preds, labels=np.arange(n_classes))), "neg_logloss"
    else:
        return float(-log_loss(y_true, preds, labels=np.arange(n_classes))), "neg_logloss"


def compute_per_row_loss(y_true: np.ndarray, preds: np.ndarray, task: str) -> np.ndarray:
    """Per-row loss: cross-entropy (classification) or squared error (regression).

    Returns (n_samples,) array. Higher = worse predictions.
    Importance = ablated_loss - baseline_loss: positive means the feature helped.
    """
    eps = 1e-7
    if task == "regression":
        return (preds.ravel() - y_true.ravel()) ** 2

    y_int = y_true.astype(int)
    if preds.ndim == 2:
        p_correct = preds[np.arange(len(y_int)), y_int]
    else:
        p = preds.ravel()
        p_correct = np.where(y_int == 1, p, 1 - p)

    p_correct = np.clip(p_correct, eps, 1 - eps)
    return -np.log(p_correct)


# ── Feature metadata ────────────────────────────────────────────────────────


def get_alive_features(model_key: str, labels_path: Path = DEFAULT_CONCEPT_LABELS) -> List[int]:
    """Get sorted list of alive feature indices for a model."""
    with open(labels_path) as f:
        data = json.load(f)
    label_key = MODEL_KEY_TO_LABEL_KEY.get(model_key, model_key)
    features = data["feature_lookup"][label_key]
    return sorted(int(k) for k in features.keys())


def get_feature_labels(model_key: str, labels_path: Path = DEFAULT_CONCEPT_LABELS) -> Dict[int, str]:
    """Get feature_idx -> label mapping for a model."""
    with open(labels_path) as f:
        data = json.load(f)
    label_key = MODEL_KEY_TO_LABEL_KEY.get(model_key, model_key)
    features = data["feature_lookup"][label_key]
    return {int(k): v.get("label", "unknown") for k, v in features.items()}


def get_matryoshka_bands(model_key: str, sae_dir: Path = None) -> Dict[str, int]:
    """Get Matryoshka band boundaries: {band_name: upper_boundary}."""
    if sae_dir is None:
        sae_dir = DEFAULT_SAE_DIR
    ckpt_path = sae_dir / model_key / SAE_FILENAME
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    if hasattr(config, "__dict__"):
        config = config.__dict__
    hidden_dim = config.get("hidden_dim", 0)
    bands = {"S1": hidden_dim // 16, "S2": hidden_dim // 8,
             "S3": hidden_dim // 4, "S4": hidden_dim // 2, "S5": hidden_dim}
    return bands


# ── Tail management ──────────────────────────────────────────────────────────

# Tail classes stay in intervene_sae.py (1000+ lines, 16+ importers).
# Import them for isinstance checks in _inject_query_deltas.
from scripts.intervention.intervene_sae import (  # noqa: E402
    TabPFNTail, TabICLTail, TabICLV2Tail,
    TabDPTTail, MitraTail, HyperFastTail,
    CARTETail, Tabula8BTail,
    build_tail,
)

# Models that must use sequential fallback (no batched K-copy trick)
SEQUENTIAL_MODELS = (MitraTail, HyperFastTail, CARTETail, Tabula8BTail)


# ── Row alignment ───────────────────────────────────────────────────────────


def align_test_rows(
    holdout_indices: np.ndarray,
    test_row_indices: np.ndarray,
) -> np.ndarray:
    """Map absolute dataset row indices to positions in X_test.

    Args:
        holdout_indices: splits[ds]["test_indices"], defines X_test ordering
        test_row_indices: absolute indices for this dataset's test embeddings

    Returns:
        positions: (n_test,) array of indices into X_test

    Raises:
        KeyError: if a test_row_index is not in holdout_indices
    """
    abs_to_pos = {int(idx): pos for pos, idx in enumerate(holdout_indices)}
    return np.array([abs_to_pos[int(ri)] for ri in test_row_indices])


def _unpool_dataset(npz_data, dataset: str):
    """Extract one dataset's offset and count from a concatenated NPZ."""
    spd = npz_data["samples_per_dataset"]
    offset = 0
    for ds_name, count in spd:
        ds_name, count = str(ds_name), int(count)
        if ds_name == dataset:
            return offset, count
        offset += count
    return None, None


# Models that need raw DataFrames (not preprocessed numpy cache)
DATAFRAME_MODELS = {"tabula8b", "carte"}


def load_dataset_context(
    model_key: str,
    dataset: str,
    splits: Optional[dict] = None,
    max_context: int = 1024,
) -> Tuple:
    """Load preprocessed data and resolve row alignment for one dataset.

    For most models, loads from the preprocessing cache (numpy arrays).
    For CARTE and Tabula-8B, loads raw DataFrames from the TabArena cache.

    Returns:
        X_train, y_train, X_query, y_query, row_indices, task
        (X_train/X_query are DataFrames for CARTE/Tabula-8B, numpy otherwise)
    """
    if splits is None:
        splits = json.loads(SPLITS_PATH.read_text())

    split_info = splits[dataset]
    task = split_info["task_type"]
    train_idx = np.array(split_info["train_indices"])
    holdout_indices = np.array(split_info["test_indices"])

    # Load row_indices from test NPZ
    test_npz = sorted(SAE_DATA_DIR.glob(f"{model_key}_taskaware_sae_test.npz"))
    if not test_npz:
        test_npz = sorted(SAE_DATA_DIR.glob(f"{model_key}_*_sae_test.npz"))
    if not test_npz:
        raise FileNotFoundError(f"No test embeddings for {model_key} in {SAE_DATA_DIR}")
    npz_data = np.load(test_npz[0], allow_pickle=True)

    offset, count = _unpool_dataset(npz_data, dataset)
    if offset is None:
        raise ValueError(f"Dataset {dataset} not in test embeddings")
    row_indices = npz_data["row_indices"][offset:offset + count]

    if model_key in DATAFRAME_MODELS:
        # DataFrame models: load raw data, split by absolute indices
        from data.extended_loader import _load_tabarena_cached_v2

        cached = _load_tabarena_cached_v2(dataset)
        if cached is None:
            raise FileNotFoundError(
                f"No raw TabArena cache for {dataset}. "
                f"Run load_tabarena_dataset('{dataset}') first to populate cache."
            )
        X_df, y = cached

        X_train = X_df.iloc[train_idx].reset_index(drop=True)
        y_train = y[train_idx]

        # Subsample context if needed (matches 04_extract_all_layers.py)
        if len(X_train) > max_context:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_train), max_context, replace=False)
            X_train = X_train.iloc[idx].reset_index(drop=True)
            y_train = y_train[idx]

        # Query rows: absolute row_indices into original DataFrame
        X_query = X_df.iloc[row_indices].reset_index(drop=True)
        y_query = y[row_indices]
    else:
        # Standard path: load from preprocessing cache
        from data.preprocessing import load_preprocessed, CACHE_DIR

        data = load_preprocessed(model_key, dataset, CACHE_DIR)
        positions = align_test_rows(holdout_indices, row_indices)
        X_train = data.X_train
        y_train = data.y_train
        X_query = data.X_test[positions]
        y_query = data.y_test[positions]

    return X_train, y_train, X_query, y_query, row_indices, task


def load_test_embeddings(model_key: str) -> Dict[str, np.ndarray]:
    """Load per-dataset test embeddings from the canonical taskaware NPZ.

    Returns:
        Dict mapping dataset name -> (n_rows, emb_dim) normalized embeddings.
    """
    candidates = sorted(SAE_DATA_DIR.glob(f"{model_key}_taskaware_sae_test.npz"))
    if not candidates:
        candidates = sorted(SAE_DATA_DIR.glob(f"{model_key}_*_sae_test.npz"))
    if not candidates:
        raise FileNotFoundError(f"No test embeddings for {model_key} in {SAE_DATA_DIR}")

    data = np.load(candidates[0], allow_pickle=True)
    embeddings = data["embeddings"]
    spd = data["samples_per_dataset"]

    result = {}
    offset = 0
    for ds_name, count in spd:
        ds_name, count = str(ds_name), int(count)
        result[ds_name] = embeddings[offset:offset + count]
        offset += count
    return result


# ── Delta computation ────────────────────────────────────────────────────────


def compute_feature_deltas(
    sae: torch.nn.Module,
    h_row: torch.Tensor,
    feature_indices: List[int],
    data_std: torch.Tensor,
) -> torch.Tensor:
    """Compute per-feature ablation deltas for one row.

    For each feature, zeros it in SAE latent space, decodes, computes delta
    in raw embedding space (denormalized with data_std).

    Args:
        sae: Trained SAE in eval mode
        h_row: (hidden_dim,) SAE activations for this row
        feature_indices: which features to ablate (the firing features)
        data_std: (emb_dim,) per-dataset std for denormalization

    Returns:
        deltas: (K, emb_dim) tensor in raw embedding space
    """
    K = len(feature_indices)
    with torch.no_grad():
        recon_full = sae.decode(h_row.unsqueeze(0))  # (1, emb_dim)

        h_batch = h_row.unsqueeze(0).expand(K, -1).clone()  # (K, hidden_dim)
        for k, fi in enumerate(feature_indices):
            h_batch[k, fi] = 0.0

        recon_ablated = sae.decode(h_batch)  # (K, emb_dim)
        delta_norm = recon_ablated - recon_full  # (K, emb_dim)
        delta_raw = delta_norm * data_std.unsqueeze(0)

    return delta_raw


# ── Batched ablation ─────────────────────────────────────────────────────────


def batched_ablation(
    tail,
    X_row: np.ndarray,
    deltas: torch.Tensor,
    max_K: int = 512,
) -> np.ndarray:
    """Batched ablation: recapture with K query copies, inject deltas, predict.

    Creates K copies of X_row as the query batch, recaptures hidden state
    (1 full forward, no re-fit), injects K different deltas at query positions
    (zero context delta), runs 1 tail pass → K predictions.

    Args:
        tail: fitted tail model with recapture() method
        X_row: (1, n_features) single query row
        deltas: (K, emb_dim) per-feature deltas in raw embedding space
        max_K: chunk size for VRAM safety

    Returns:
        preds: (K, ...) predictions, one per ablation
    """
    K = len(deltas)
    if K == 0:
        return np.array([])

    all_preds = []
    for chunk_start in range(0, K, max_K):
        chunk_deltas = deltas[chunk_start:min(chunk_start + max_K, K)]
        chunk_K = len(chunk_deltas)

        X_batch = np.tile(X_row, (chunk_K, 1))
        tail.recapture(X_batch)

        state = tail.hidden_state.clone()
        _inject_query_deltas(tail, state, chunk_deltas)
        preds = tail._predict_with_modified_state(state)
        all_preds.append(preds)

    return np.concatenate(all_preds, axis=0) if len(all_preds) > 1 else all_preds[0]


def _inject_query_deltas(tail, state: torch.Tensor, deltas: torch.Tensor):
    """Inject per-query deltas into cached hidden state. Zero context delta."""
    K = len(deltas)

    if isinstance(tail, TabPFNTail):
        ctx = tail.single_eval_pos
        for k in range(K):
            state[0, ctx + k, :, :] += deltas[k].unsqueeze(0)

    elif isinstance(tail, (TabICLTail, TabICLV2Tail)):
        ctx = tail.train_size
        for k in range(K):
            state[:, ctx + k, :] += deltas[k].unsqueeze(0)

    elif isinstance(tail, TabDPTTail):
        ctx = tail.n_ctx
        for k in range(K):
            if state.ndim == 3:
                state[ctx + k] += deltas[k].unsqueeze(0)
            else:
                state[ctx + k] += deltas[k]

    elif isinstance(tail, SEQUENTIAL_MODELS):
        raise NotImplementedError(
            f"{type(tail).__name__}: use batched_ablation_sequential() instead"
        )

    else:
        raise TypeError(f"Unknown tail type: {type(tail)}")


def batched_ablation_sequential(
    tail,
    X_row: np.ndarray,
    deltas: torch.Tensor,
) -> np.ndarray:
    """Fallback for Mitra/HyperFast: K sequential predict_row() calls.

    Still benefits from fit-once — just no K-batching speedup.
    """
    K = len(deltas)
    if K == 0:
        return np.array([])

    tail.recapture(X_row)

    preds_list = []
    for k in range(K):
        preds = tail.predict_row(0, deltas[k])
        preds_list.append(preds[0:1])

    return np.concatenate(preds_list, axis=0)
