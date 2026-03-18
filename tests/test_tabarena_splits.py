"""Tests for TabArena split download and caching (00_download_tabarena_splits.py)."""

import json
from pathlib import Path

import pytest

from data.extended_loader import TABARENA_DATASETS

SPLITS_PATH = Path("output/sae_training_round9/tabarena_splits.json")


@pytest.fixture(scope="module")
def splits():
    if not SPLITS_PATH.exists():
        pytest.skip("Run 00_download_tabarena_splits.py first")
    return json.loads(SPLITS_PATH.read_text())


def test_all_51_datasets_present(splits):
    assert len(splits) == 51, f"Expected 51 datasets, got {len(splits)}"


def test_all_catalog_datasets_in_splits(splits):
    missing = [k for k in TABARENA_DATASETS if k not in splits]
    assert not missing, f"Missing from splits: {missing}"


def test_split_fields(splits):
    required = {"task_id", "dataset_id", "task_type", "target", "n_samples",
                "train_indices", "test_indices"}
    for name, entry in splits.items():
        missing = required - set(entry)
        assert not missing, f"{name} missing fields: {missing}"


def test_train_test_indices_non_overlapping(splits):
    for name, entry in splits.items():
        train_set = set(entry["train_indices"])
        test_set = set(entry["test_indices"])
        overlap = train_set & test_set
        assert not overlap, f"{name}: {len(overlap)} overlapping indices"


def test_train_test_cover_all_rows(splits):
    for name, entry in splits.items():
        total = len(entry["train_indices"]) + len(entry["test_indices"])
        assert total == entry["n_samples"], (
            f"{name}: train+test={total} != n_samples={entry['n_samples']}"
        )


def test_train_fraction_approx_two_thirds(splits):
    for name, entry in splits.items():
        frac = len(entry["train_indices"]) / entry["n_samples"]
        assert 0.60 <= frac <= 0.75, (
            f"{name}: train fraction {frac:.2f} outside [0.60, 0.75]"
        )


def test_task_types_match_catalog(splits):
    for name, entry in splits.items():
        expected = TABARENA_DATASETS[name]["task"]
        assert entry["task_type"] == expected, (
            f"{name}: splits task_type={entry['task_type']!r} "
            f"but catalog says {expected!r}"
        )


def test_minimum_train_size(splits):
    """Smallest datasets should still have at least 400 train rows."""
    for name, entry in splits.items():
        assert len(entry["train_indices"]) >= 400, (
            f"{name}: only {len(entry['train_indices'])} train rows"
        )
