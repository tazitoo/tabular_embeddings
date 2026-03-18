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
