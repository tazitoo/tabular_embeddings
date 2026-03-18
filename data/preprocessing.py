"""Preprocess tabular datasets once per model and cache to disk.

Each model has a canonical preprocessing pipeline. This module runs that
pipeline on the train split, transforms both train and test, and saves the
result as a float32 numpy .npz file alongside a JSON metadata sidecar.
Downstream scripts load from cache and never re-preprocess.

Usage:
    from data.preprocessing import preprocess_for_model, save_preprocessed, load_preprocessed, is_cached, CACHE_DIR

    data = preprocess_for_model("tabpfn", "my_dataset", X_train_df, y_train, X_test_df, y_test, "classification")
    save_preprocessed(data, CACHE_DIR)

    if is_cached("tabpfn", "my_dataset", CACHE_DIR):
        data = load_preprocessed("tabpfn", "my_dataset", CACHE_DIR)
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT

# Default cache location — import this in downstream scripts so the path is consistent.
CACHE_DIR = PROJECT_ROOT / "output" / "sae_training_round9" / "preprocessed"

# Models whose preprocessed arrays may contain NaN (the model handles it natively).
NAN_SAFE_MODELS = {"tabpfn", "tabdpt"}

# Models using AutoMLPipelineFeatureGenerator for outer preprocessing.
AUTOGLUON_MODELS = {"tabpfn", "tabicl", "tabdpt", "mitra"}


@dataclass
class PreprocessedDataset:
    X_train: np.ndarray    # float32, NaN preserved or filled depending on model
    X_test: np.ndarray     # float32
    y_train: np.ndarray    # int32 (classification) or float32 (regression)
    y_test: np.ndarray
    cat_indices: list[int] # empty for TabICL/Mitra after imputation
    model_name: str
    dataset_name: str
    task_type: str         # "classification" or "regression"


def _npz_path(model_name: str, dataset_name: str, cache_dir: Path) -> Path:
    return cache_dir / model_name / f"{dataset_name}.npz"


def _json_path(model_name: str, dataset_name: str, cache_dir: Path) -> Path:
    return cache_dir / model_name / f"{dataset_name}.json"


def is_cached(model_name: str, dataset_name: str, cache_dir: Path) -> bool:
    """Return True only if both the .npz and .json sidecar exist."""
    return (
        _npz_path(model_name, dataset_name, cache_dir).exists()
        and _json_path(model_name, dataset_name, cache_dir).exists()
    )


def save_preprocessed(data: PreprocessedDataset, cache_dir: Path) -> None:
    """Write cache. JSON sidecar is written first; a partial write is detectable."""
    npz = _npz_path(data.model_name, data.dataset_name, cache_dir)
    jpath = _json_path(data.model_name, data.dataset_name, cache_dir)
    npz.parent.mkdir(parents=True, exist_ok=True)
    # Write JSON first so a killed process leaves npz missing (detectable by is_cached).
    jpath.write_text(json.dumps({
        "model_name": data.model_name,
        "dataset_name": data.dataset_name,
        "task_type": data.task_type,
    }, indent=2))
    np.savez_compressed(
        npz,
        X_train=data.X_train,
        X_test=data.X_test,
        y_train=data.y_train,
        y_test=data.y_test,
        cat_indices=np.array(data.cat_indices, dtype=np.int32),
    )


def load_preprocessed(model_name: str, dataset_name: str, cache_dir: Path) -> PreprocessedDataset:
    npz = _npz_path(model_name, dataset_name, cache_dir)
    jpath = _json_path(model_name, dataset_name, cache_dir)
    if not npz.exists() or not jpath.exists():
        raise FileNotFoundError(f"Incomplete cache for {model_name}/{dataset_name} in {cache_dir}")
    arrays = np.load(npz, allow_pickle=False)
    meta = json.loads(jpath.read_text())
    return PreprocessedDataset(
        X_train=arrays["X_train"],
        X_test=arrays["X_test"],
        y_train=arrays["y_train"],
        y_test=arrays["y_test"],
        cat_indices=arrays["cat_indices"].tolist(),
        model_name=meta["model_name"],
        dataset_name=meta["dataset_name"],
        task_type=meta["task_type"],
    )
