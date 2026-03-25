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
    """Get the optimal extraction layer index for a model (fixed config)."""
    with open(layers_path, encoding="utf-8") as f:
        layers_config = json.load(f)
    return layers_config[model_key]["optimal_layer"]


def get_extraction_layer_taskaware(model_key: str, dataset: str = None) -> int:
    """Get the task-aware extraction layer from the SAE training norm stats.

    Round 10 SAEs use task-aware layer selection (round(mean_critical_layer)
    per architecture variant). This reads the actual layer used from the
    norm stats NPZ, which is authoritative.

    Args:
        model_key: model name
        dataset: if given, return the layer for this specific dataset

    Returns:
        extraction layer index
    """
    candidates = sorted(SAE_DATA_DIR.glob(f"{model_key}_taskaware_norm_stats.npz"))
    if not candidates:
        candidates = sorted(SAE_DATA_DIR.glob(f"{model_key}_*_norm_stats.npz"))
    if not candidates:
        # Fall back to fixed config
        return get_extraction_layer(model_key)

    stats = np.load(candidates[0], allow_pickle=True)
    if "layers" not in stats:
        return get_extraction_layer(model_key)

    layers = stats["layers"]
    if dataset is not None:
        datasets = list(stats["datasets"])
        if dataset in datasets:
            return int(layers[datasets.index(dataset)])

    # All datasets use the same layer in task-aware fixed mode
    return int(layers[0])


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

    n_true_classes = len(np.unique(y_true))
    n_pred_classes = preds.shape[1] if preds.ndim == 2 else n_true_classes

    # Degenerate: 1D predictions (class labels instead of probabilities)
    # or single-class output — can't compute meaningful metric
    if preds.ndim == 1 or n_pred_classes <= 1:
        return float("-inf"), "degenerate"

    # Class count mismatch: model returns fewer columns than true labels
    if n_pred_classes < n_true_classes:
        return float("-inf"), "degenerate"

    if n_true_classes == 2 and n_pred_classes == 2:
        proba = preds[:, 1] if preds.ndim == 2 else preds
        try:
            return float(roc_auc_score(y_true, proba)), "auc"
        except ValueError:
            return float(-log_loss(y_true, preds, labels=np.arange(n_true_classes))), "neg_logloss"
    else:
        return float(-log_loss(y_true, preds, labels=np.arange(n_true_classes))), "neg_logloss"


def compute_per_row_loss(y_true: np.ndarray, preds: np.ndarray, task: str) -> np.ndarray:
    """Per-row loss: cross-entropy (classification) or squared error (regression).

    Returns (n_samples,) array. Higher = worse predictions.
    Importance = ablated_loss - baseline_loss: positive means the feature helped.
    """
    eps = 1e-7
    if task == "regression":
        return (preds.ravel() - y_true.ravel()) ** 2

    y_int = y_true.astype(int)
    if preds.ndim == 2 and preds.shape[1] == 1:
        # Single-column output (e.g. Mitra binary): treat as P(y=1)
        p = preds.ravel()
        p_correct = np.where(y_int == 1, p, 1 - p)
    elif preds.ndim == 2 and preds.shape[1] > y_int.max():
        p_correct = preds[np.arange(len(y_int)), y_int]
    elif preds.ndim == 2:
        # Fewer columns than expected — remap classes to 0..n_cols-1
        classes = np.unique(y_int)
        remap = {c: i for i, c in enumerate(classes)}
        y_remapped = np.array([remap[c] for c in y_int])
        p_correct = preds[np.arange(len(y_remapped)), y_remapped]
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
SEQUENTIAL_MODELS = (HyperFastTail, CARTETail, Tabula8BTail)


# ── Row alignment ───────────────────────────────────────────────────────────


def _cap_context(X_train, y_train, max_context: int, task: str, seed: int = 42):
    """Subsample context if it exceeds max_context.

    Replicates the exact logic from 04_extract_all_layers.py so the tail
    model sees the same context rows that produced the embeddings.

    - DataFrame models (CARTE, Tabula-8B): simple rng.choice (line 261)
    - Numpy models: stratified for classification, simple for regression (line 55-88)
    """
    if len(X_train) <= max_context:
        return X_train, y_train

    rng = np.random.RandomState(seed)

    if hasattr(X_train, 'iloc'):
        # DataFrame path — simple random (matches line 261-263)
        idx = rng.choice(len(X_train), max_context, replace=False)
        return X_train.iloc[idx].reset_index(drop=True), y_train[idx]

    # Numpy path — stratified for classification (matches sample_context())
    if task == "classification":
        classes, counts = np.unique(y_train, return_counts=True)
        indices = []
        for cls, count in zip(classes, counts):
            cls_idx = np.where(y_train == cls)[0]
            n_take = max(1, int(max_context * count / len(y_train)))
            indices.append(rng.choice(cls_idx, size=min(n_take, len(cls_idx)), replace=False))
        indices = np.concatenate(indices)
        if len(indices) > max_context:
            indices = rng.choice(indices, size=max_context, replace=False)
        elif len(indices) < max_context:
            remaining = np.setdiff1d(np.arange(len(y_train)), indices)
            extra = rng.choice(remaining, size=max_context - len(indices), replace=False)
            indices = np.concatenate([indices, extra])
    else:
        indices = rng.choice(len(X_train), size=max_context, replace=False)

    return X_train[indices], y_train[indices]


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
        X_train, y_train = _cap_context(X_train, y_train, max_context, task)

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
        X_train, y_train = _cap_context(X_train, y_train, max_context, task)
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


def compute_feature_reconstructions(
    sae: torch.nn.Module,
    h_row: torch.Tensor,
    feature_indices: List[int],
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
) -> torch.Tensor:
    """Compute full SAE reconstructions with each feature ablated.

    For models (like Mitra) where we need to REPLACE the activation rather
    than add a delta — the reconstruction IS the intervention.

    Args:
        sae: Trained SAE in eval mode
        h_row: (hidden_dim,) SAE activations for this row (already encoded)
        feature_indices: which features to ablate
        data_mean: (emb_dim,) per-dataset mean for denormalization
        data_std: (emb_dim,) per-dataset std for denormalization

    Returns:
        recons: (K, emb_dim) reconstructed activations in raw embedding space
    """
    K = len(feature_indices)
    with torch.no_grad():
        h_batch = h_row.unsqueeze(0).expand(K, -1).clone()
        for k, fi in enumerate(feature_indices):
            h_batch[k, fi] = 0.0
        recon_norm = sae.decode(h_batch)  # (K, emb_dim) in normalized space
        recon_raw = recon_norm * data_std.unsqueeze(0) + data_mean.unsqueeze(0)
    return recon_raw


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

        if isinstance(tail, MitraTail):
            # Mitra uses hook-based delta injection (checkpoint-safe).
            # chunk_deltas are (K, emb_dim) in raw embedding space.
            # We need to expand each delta to all features (unsqueeze to 4D).
            preds_list = []
            for k in range(chunk_K):
                delta_query = chunk_deltas[k:k+1].unsqueeze(1)  # (1, 1, emb_dim)
                # Zero support delta (context unchanged)
                n_sup = tail.captured_support[0].shape[1] if tail.captured_support else 0
                delta_support = torch.zeros(n_sup, chunk_deltas.shape[-1],
                                            device=chunk_deltas.device).unsqueeze(1)
                p = tail._predict_with_delta(delta_support, delta_query)
                preds_list.append(p)
            preds = np.array(preds_list)
        else:
            state = tail.hidden_state.clone()
            _inject_query_deltas(tail, state, chunk_deltas)
            preds = tail._predict_with_modified_state(state)

        all_preds.append(preds)

    return np.concatenate(all_preds, axis=0) if len(all_preds) > 1 else all_preds[0]


def _mitra_patched_predict(tail, recons: torch.Tensor) -> np.ndarray:
    """Mitra-specific: call tab2d directly with checkpoint-free forward.

    predict_proba() goes through trainer.predict() which uses
    torch.utils.checkpoint.checkpoint(layer, ...) — this discards any
    modifications to layer.forward. We must:
    1. Build batch tensors via trainer's DatasetFinetune
    2. Monkey-patch tab2d.forward to skip checkpoint
    3. Call tab2d() directly
    4. Patch y-token at extraction layer with SAE reconstructions
    5. Convert logits → probabilities
    """
    import types
    import einops
    from autogluon.tabular.models.mitra._internal.data.dataset_finetune import DatasetFinetune

    trainer = tail.clf.trainers[0]
    tab2d = trainer.model
    tab2d.eval()

    # Build batch tensors (same as trainer.predict)
    x_s_raw = trainer.preprocessor.transform_X(tail.X_context)
    x_q_raw = trainer.preprocessor.transform_X(tail.X_query)
    y_s_raw = trainer.preprocessor.transform_y(tail.y_context)

    trainer.rng.set_state(tail.rng_state)
    ds = DatasetFinetune(
        trainer.cfg,
        x_support=x_s_raw, y_support=y_s_raw, x_query=x_q_raw, y_query=None,
        max_samples_support=trainer.cfg.hyperparams['max_samples_support'],
        max_samples_query=trainer.cfg.hyperparams['max_samples_query'],
        rng=trainer.rng,
    )
    batch = next(iter(trainer.make_loader(ds, training=False)))
    device = tail.device
    x_s = batch['x_support'].float().to(device)
    y_s = batch['y_support'].to(device)
    x_q = batch['x_query'].float().to(device)
    pf = batch['padding_features'].to(device)
    pos = batch['padding_obs_support'].to(device)
    poq = batch['padding_obs_query'].to(device)

    # Monkey-patch forward to skip checkpoint AND replace y-token at layer L
    extraction_layer = tail.extraction_layer
    original_forward = tab2d.forward

    def forward_no_ckpt_patched(self, x_support, y_support, x_query,
                                 padding_features, padding_obs_support, padding_obs_query__):
        x_query__ = x_query
        batch_size = x_support.shape[0]
        n_obs_query__ = x_query__.shape[1]

        x_support, x_query__ = self.x_quantile(x_support, x_query__, padding_obs_support, padding_features)
        x_support = self.x_embedding(x_support)
        x_query__ = self.x_embedding(x_query__)
        y_support, y_query__ = self.y_embedding(y_support, padding_obs_support, n_obs_query__)

        support, pack_support = einops.pack((y_support, x_support), 'b s * d')
        query__, pack_query__ = einops.pack((y_query__, x_query__), 'b s * d')

        padding_features_y = torch.zeros((batch_size, 1), device=padding_features.device, dtype=torch.bool)
        padding_features, _ = einops.pack((padding_features_y, padding_features), 'b *')

        # Direct layer calls — no checkpoint
        n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            support, query__ = layer(support, query__, None, None,
                                     batch_size, padding_obs_support, padding_obs_query__, padding_features)
            if i == extraction_layer and query__.ndim == 4:
                # Replace y-token (position 0) with SAE reconstructions
                query__ = query__.clone()
                query__[0, :, 0, :] = recons

        query__ = self.final_layer_norm(query__)
        # If extraction_layer == n_layers, patch after final_layer_norm
        if extraction_layer >= n_layers and query__.ndim == 4:
            query__ = query__.clone()
            query__[0, :, 0, :] = recons
        query__ = self.final_layer(query__)

        query__, _ = einops.unpack(query__, pack_query__, 'b s * c')
        y_query__ = query__[:, :, 0, :]
        return y_query__

    tab2d.forward = types.MethodType(forward_no_ckpt_patched, tab2d)
    try:
        with torch.no_grad():
            logits = tab2d(x_s, y_s, x_q, pf, pos, poq)
    finally:
        tab2d.forward = original_forward

    # Convert logits → predictions
    if tail.task == "regression":
        # Regression: raw logits, single output dim
        raw = logits[0, :, 0].float().cpu().numpy()
        preds = trainer.preprocessor.inverse_transform_y(raw)
    else:
        # Classification: softmax, truncate to actual class count
        n_classes = len(np.unique(tail.y_context))
        probs = torch.softmax(logits[0, :, :n_classes], dim=-1).float().cpu().numpy()
        preds = trainer.preprocessor.inverse_transform_y(probs)
    return np.asarray(preds)


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

    elif isinstance(tail, MitraTail):
        # Mitra handled in batched_ablation() directly — not via state injection
        raise NotImplementedError("Mitra: handled in batched_ablation()")

    elif isinstance(tail, (HyperFastTail, CARTETail, Tabula8BTail)):
        raise NotImplementedError(
            f"{type(tail).__name__}: use batched_ablation_sequential() instead"
        )

    else:
        raise TypeError(f"Unknown tail type: {type(tail)}")


def batched_ablation_sequential(
    tail,
    X_row: np.ndarray,
    deltas: torch.Tensor,
    query_idx: int = 0,
) -> np.ndarray:
    """Fallback for models without batched K-copy support.

    K sequential predict_row() calls. Still benefits from fit-once.

    Args:
        tail: fitted tail model
        X_row: (1, n_features) single query row (unused if tail already has it)
        deltas: (K, emb_dim) per-feature deltas
        query_idx: index of this row in the tail's query set

    Returns:
        preds: (K, ...) predictions, one per ablation
    """
    K = len(deltas)
    if K == 0:
        return np.array([])

    # Recapture if available (Mitra/HyperFast), otherwise use existing state
    if hasattr(tail, "recapture"):
        tail.recapture(X_row)
        query_idx = 0

    preds_list = []
    for k in range(K):
        preds = tail.predict_row(query_idx, deltas[k])
        preds_list.append(preds[query_idx:query_idx + 1])

    return np.concatenate(preds_list, axis=0)
