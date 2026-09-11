"""A regenerated model corpus must use the SAME rows as the other models' corpora.

The builder's stratified sampling depends on a live TabPFN difficulty pass, which is
environment-sensitive: rebuilding one model's corpus in a new environment picked
different rows on 32 of 51 datasets (2026-09-11), which breaks the row alignment every
cross-model correlation and sweep assumes. `--rows-from-round R` pins the train/test
rows to a reference corpus of round R instead of resampling.
"""
import importlib.util

import numpy as np

from scripts._project_root import PROJECT_ROOT


def _load_07():
    path = PROJECT_ROOT / "scripts" / "sae_corpus" / "07_build_sae_training_data.py"
    spec = importlib.util.spec_from_file_location("build_sae_training_data_07", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_ref(tmp_path, split, rows_by_ds):
    order = list(rows_by_ds)
    np.savez(tmp_path / f"m_taskaware_sae_{split}.npz",
             row_indices=np.concatenate([rows_by_ds[d] for d in order]).astype(np.int32),
             samples_per_dataset=np.array([(d, len(rows_by_ds[d])) for d in order], dtype=object),
             source_datasets=np.array(order))


def test_reference_rows_are_read_per_dataset(tmp_path):
    m = _load_07()
    _write_ref(tmp_path, "training", {"a": np.array([10, 30, 50]), "b": np.array([7])})
    _write_ref(tmp_path, "test", {"a": np.array([20]), "b": np.array([8, 9])})
    ref = m.load_reference_rows(tmp_path / "m_taskaware_sae_training.npz",
                                tmp_path / "m_taskaware_sae_test.npz")
    assert list(ref["a"][0]) == [10, 30, 50] and list(ref["a"][1]) == [20]
    assert list(ref["b"][0]) == [7] and list(ref["b"][1]) == [8, 9]


def test_reference_rows_map_to_holdout_positions_in_reference_order():
    m = _load_07()
    holdout = np.array([5, 20, 10, 50, 30, 8])          # global row ids of the holdout, in file order
    train_idx, test_idx = m.select_sample_from_reference(
        holdout, ref_train=np.array([10, 30, 50]), ref_test=np.array([20, 8]))
    assert list(holdout[train_idx]) == [10, 30, 50]
    assert list(holdout[test_idx]) == [20, 8]


def test_reference_rows_missing_from_holdout_is_an_error():
    import pytest

    m = _load_07()
    with pytest.raises(ValueError):
        m.select_sample_from_reference(np.array([1, 2, 3]), ref_train=np.array([1, 99]), ref_test=np.array([2]))


def test_reference_training_file_without_row_indices_yields_none_for_train(tmp_path):
    m = _load_07()
    np.savez(tmp_path / "m_taskaware_sae_training.npz",
             samples_per_dataset=np.array([("a", 3)], dtype=object), source_datasets=np.array(["a"]))
    _write_ref(tmp_path, "test", {"a": np.array([20])})
    ref = m.load_reference_rows(tmp_path / "m_taskaware_sae_training.npz",
                                tmp_path / "m_taskaware_sae_test.npz")
    assert ref["a"][0] is None and list(ref["a"][1]) == [20]


def test_pinned_test_rows_train_from_remainder_disjoint_and_full_size():
    m = _load_07()
    rng = np.random.RandomState(0)
    n = 1200
    y = rng.randint(0, 2, n); losses = rng.rand(n)
    pinned = np.arange(0, 1200, 6)[:200]            # 200 test positions
    train_idx, test_idx = m.select_sample(n, y, losses, "classification", pinned_test=pinned)
    assert np.array_equal(test_idx, pinned)
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(train_idx) == 500
    train_r, test_r = m.select_sample(n, y, losses, "regression", pinned_test=pinned)
    assert np.array_equal(test_r, pinned) and len(set(train_r) & set(test_r)) == 0 and len(train_r) == 498


def test_pinned_test_rows_small_dataset_train_is_the_complement():
    m = _load_07()
    pinned = np.array([1, 4, 7])
    train_idx, test_idx = m.select_sample(10, np.zeros(10), None, "classification", pinned_test=pinned)
    assert np.array_equal(test_idx, pinned) and sorted(train_idx) == [0, 2, 3, 5, 6, 8, 9]
