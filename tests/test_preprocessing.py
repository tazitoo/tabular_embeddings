import numpy as np
import pytest
from pathlib import Path
from data.preprocessing import PreprocessedDataset, save_preprocessed, load_preprocessed, is_cached


def _make_dummy(has_nan: bool = True) -> PreprocessedDataset:
    X = np.array([[1.0, np.nan if has_nan else 2.0], [2.0, 3.0]], dtype=np.float32)
    return PreprocessedDataset(
        X_train=X,
        X_test=np.array([[4.0, 5.0]], dtype=np.float32),
        y_train=np.array([0, 1], dtype=np.int32),
        y_test=np.array([0], dtype=np.int32),
        cat_indices=[],
        model_name="tabpfn",
        dataset_name="test_ds",
        task_type="classification",
    )


def test_roundtrip_preserves_nan(tmp_path):
    data = _make_dummy(has_nan=True)
    save_preprocessed(data, tmp_path)
    loaded = load_preprocessed("tabpfn", "test_ds", tmp_path)
    # NaN must survive the npz roundtrip
    assert np.isnan(loaded.X_train[0, 1])
    np.testing.assert_array_equal(loaded.y_train, data.y_train)
    assert loaded.cat_indices == []
    assert loaded.task_type == "classification"


def test_is_cached_requires_both_files(tmp_path):
    assert not is_cached("tabpfn", "test_ds", tmp_path)
    save_preprocessed(_make_dummy(), tmp_path)
    assert is_cached("tabpfn", "test_ds", tmp_path)
    # Remove .npz — should be treated as not cached
    (tmp_path / "tabpfn" / "test_ds.npz").unlink()
    assert not is_cached("tabpfn", "test_ds", tmp_path)


import pandas as pd
from data.preprocessing import preprocess_for_model


def _make_df_with_nan():
    """DataFrame with numeric NaN and categorical NaN."""
    X_train = pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0],
        "cat": pd.Categorical(["a", "b", None, "b"]),
    })
    X_test = pd.DataFrame({
        "num": [5.0, np.nan],
        "cat": pd.Categorical(["a", None]),
    })
    y_train = np.array([0, 1, 0, 1])
    y_test = np.array([0, 1])
    return X_train, y_train, X_test, y_test


def test_tabpfn_nan_preserved():
    X_train, y_train, X_test, y_test = _make_df_with_nan()
    data = preprocess_for_model("tabpfn", "ds", X_train, y_train, X_test, y_test, "classification")
    assert np.isnan(data.X_train).any(), "tabpfn: NaN must be preserved"
    assert data.X_train.dtype == np.float32
    assert len(data.cat_indices) > 0


def test_tabicl_nan_filled():
    X_train, y_train, X_test, y_test = _make_df_with_nan()
    data = preprocess_for_model("tabicl", "ds", X_train, y_train, X_test, y_test, "classification")
    assert not np.isnan(data.X_train).any(), "tabicl: NaN must be filled"
    assert not np.isnan(data.X_test).any()
    assert data.cat_indices == [], "tabicl: cat_indices empty after imputation"
    assert data.X_train.dtype == np.float32


def test_all_numeric_no_cats():
    """All-numeric dataset — no categoricals, cat_indices must be empty."""
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, np.nan, 6.0]})
    X_test = pd.DataFrame({"a": [7.0], "b": [8.0]})
    y = np.array([0, 1, 0])
    data = preprocess_for_model("tabpfn", "ds", X_train, y, X_test, y[:1], "classification")
    assert data.cat_indices == []
    assert np.isnan(data.X_train).any()  # NaN preserved


def test_y_label_encoded_classification():
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    X_test = pd.DataFrame({"a": [4.0]})
    y_str = np.array(["neg", "pos", "neg"])
    data = preprocess_for_model("tabpfn", "ds", X_train, y_str, X_test, y_str[:1], "classification")
    assert set(data.y_train.tolist()).issubset({0, 1})
    assert data.y_train.dtype == np.int32


def test_y_regression_float32():
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    X_test = pd.DataFrame({"a": [4.0]})
    y = np.array([1.5, 2.5, 3.5])
    data = preprocess_for_model("tabpfn", "ds", X_train, y, X_test, y[:1], "regression")
    assert data.y_train.dtype == np.float32
    np.testing.assert_array_almost_equal(data.y_train, y.astype(np.float32))


def _make_df_hyperfast():
    """Object-dtype categoricals to exercise the OrdinalEncoder path."""
    X_train = pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0],
        "cat": ["a", "b", None, "b"],       # object dtype, not Categorical
    })
    X_test = pd.DataFrame({
        "num": [5.0, np.nan],
        "cat": ["a", None],
    })
    y_train = np.array([0, 1, 0, 1])
    y_test = np.array([0, 1])
    return X_train, y_train, X_test, y_test


def test_hyperfast_nan_imputed():
    """HyperFast paper: mean impute numeric, mode impute categorical."""
    X_train, y_train, X_test, y_test = _make_df_hyperfast()
    data = preprocess_for_model("hyperfast", "ds", X_train, y_train, X_test, y_test, "classification")
    assert not np.isnan(data.X_train).any(), "hyperfast: NaN must be imputed"
    assert not np.isnan(data.X_test).any()
    assert data.X_train.dtype == np.float32


def test_hyperfast_cat_indices_empty_after_imputation():
    """After imputation, category codes are fractional means — not meaningful."""
    X_train, y_train, X_test, y_test = _make_df_hyperfast()
    data = preprocess_for_model("hyperfast", "ds", X_train, y_train, X_test, y_test, "classification")
    assert data.cat_indices == [], f"Expected [], got {data.cat_indices}"


def test_hyperfast_all_numeric():
    X_train = pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, 4.0]})
    X_test = pd.DataFrame({"a": [5.0], "b": [6.0]})
    y = np.array([0, 1])
    data = preprocess_for_model("hyperfast", "ds", X_train, y, X_test, y[:1], "classification")
    assert data.cat_indices == []
    assert not np.isnan(data.X_train).any(), "hyperfast: NaN must be imputed"
