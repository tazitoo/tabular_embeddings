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
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from scripts._project_root import PROJECT_ROOT

# Default cache location — import this in downstream scripts so the path is consistent.
CACHE_DIR = PROJECT_ROOT / "output" / "sae_training_round9" / "preprocessed"

# Models whose preprocessed arrays may contain NaN (the model handles it natively).
NAN_SAFE_MODELS = {"tabpfn", "tabdpt"}

# Models using AutoMLPipelineFeatureGenerator for outer preprocessing.
# tabicl_v2 is the regression-capable successor to tabicl; same NaN-unsafe pipeline.
AUTOGLUON_MODELS = {"tabpfn", "tabicl", "tabicl_v2", "tabdpt", "mitra"}


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


def preprocess_for_model(
    model_name: str,
    dataset_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    task_type: str,
) -> PreprocessedDataset:
    """Preprocess features and labels for the given model.

    Args:
        model_name: One of "tabpfn", "tabicl", "tabdpt", "mitra", "hyperfast".
        dataset_name: Used to populate PreprocessedDataset.dataset_name.
        X_train: Raw training features as a DataFrame.
        y_train: Training labels (any dtype — will be encoded).
        X_test: Raw test features as a DataFrame.
        y_test: Test labels.
        task_type: "classification" or "regression".

    Returns:
        PreprocessedDataset with float32 numpy arrays ready for model input.

    Raises:
        NotImplementedError: For "carte" and "tabula-8b".
        ValueError: For unknown model names.
    """
    model_key = model_name.lower()

    if model_key in AUTOGLUON_MODELS:
        nan_safe = model_key in NAN_SAFE_MODELS
        X_train_np, X_test_np, cat_indices = _preprocess_autogluon(
            X_train, X_test, nan_safe=nan_safe
        )
    elif model_key == "hyperfast":
        X_train_np, X_test_np, cat_indices = _preprocess_hyperfast(X_train, X_test)
    elif model_key in ("carte", "tabula-8b"):
        raise NotImplementedError(f"Bespoke preprocessing for {model_name} not yet implemented")
    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    y_train_enc, y_test_enc = _encode_y(y_train, y_test, task_type)

    return PreprocessedDataset(
        X_train=X_train_np,
        X_test=X_test_np,
        y_train=y_train_enc,
        y_test=y_test_enc,
        cat_indices=cat_indices,
        model_name=model_name,
        dataset_name=dataset_name,
        task_type=task_type,
    )


def _df_to_float32(df: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    """Convert AutoMLPipelineFeatureGenerator output DataFrame to float32 numpy.

    Category columns (.cat.codes): code -1 (NaN) → np.nan. Numeric columns
    cast directly. Column order is preserved.

    Returns:
        (array float32, cat_indices) — indices of columns that were categorical.
    """
    out = np.empty((len(df), len(df.columns)), dtype=np.float32)
    cat_indices = []
    for i, col in enumerate(df.columns):
        s = df[col]
        if hasattr(s, "cat"):
            codes = s.cat.codes.astype(np.float32)
            codes[codes == -1] = np.nan
            out[:, i] = codes
            cat_indices.append(i)
        else:
            out[:, i] = s.values.astype(np.float32)
    return out, cat_indices


def _preprocess_autogluon(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    nan_safe: bool,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Apply AutoMLPipelineFeatureGenerator (fit on train, transform both).

    Preserves NaN when nan_safe=True (TabPFN, TabDPT).
    Applies median imputation when nan_safe=False (TabICL, Mitra).
    Returns empty cat_indices after imputation — category codes are gone.
    """
    from autogluon.features.generators import AutoMLPipelineFeatureGenerator

    fg = AutoMLPipelineFeatureGenerator(verbosity=0)
    X_train_proc = fg.fit_transform(X_train)
    X_test_proc = fg.transform(X_test)

    X_train_np, cat_indices = _df_to_float32(X_train_proc)
    X_test_np, _ = _df_to_float32(X_test_proc)

    if not nan_safe:
        imp = SimpleImputer(strategy="median")
        X_train_np = imp.fit_transform(X_train_np)
        X_test_np = imp.transform(X_test_np)
        cat_indices = []  # imputed values are no longer meaningful category codes

    return X_train_np, X_test_np, cat_indices


def _preprocess_hyperfast(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Ordinal-encode categoricals and impute NaN for HyperFast.

    Per the HyperFast paper (Bonet et al. 2024, §Initial Transformation Layers):
    "We first perform a general data standardization stage by one-hot encoding
    categorical features, mean imputing missing numerical features, mode
    imputing missing categorical features, and feature-wise transforming to
    zero mean and unit variance."

    We handle: ordinal encoding of categoricals, then mean imputation for all
    columns (after encoding, category codes are numeric so mean imputation is
    appropriate). HyperFast's internal _preprocess_test_data() handles the
    StandardScaler and one-hot encoding.

    cat_indices are returned empty after imputation — category codes are
    fractional medians with no categorical meaning (same rationale as
    tabicl/mitra).
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    if cat_cols:
        # OrdinalEncoder only treats np.nan (not None) as a missing value.
        X_train[cat_cols] = X_train[cat_cols].where(X_train[cat_cols].notna(), other=np.nan)
        X_test[cat_cols] = X_test[cat_cols].where(X_test[cat_cols].notna(), other=np.nan)
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
        )
        X_train[cat_cols] = enc.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = enc.transform(X_test[cat_cols])

    X_train_np = X_train.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)

    # Impute NaN — HyperFast expects no missing values at input
    imp = SimpleImputer(strategy="mean")
    X_train_np = imp.fit_transform(X_train_np)
    X_test_np = imp.transform(X_test_np)

    # Preserve cat_indices — HyperFast needs them for internal one-hot encoding.
    # The values are ordinal-encoded integers; HyperFast's OneHotEncoder handles them.
    all_cols = list(X_train.columns)
    cat_indices = [all_cols.index(c) for c in cat_cols]
    return X_train_np, X_test_np, cat_indices


def _encode_y(
    y_train: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode labels: LabelEncoder for classification, float32 for regression.

    Note: LabelEncoder.transform raises ValueError if y_test contains a class
    not seen in y_train. This is intentional — fail loudly on data issues.
    """
    if task_type == "regression":
        return y_train.astype(np.float32), y_test.astype(np.float32)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train).astype(np.int32)
    y_test_enc = le.transform(y_test).astype(np.int32)
    return y_train_enc, y_test_enc
