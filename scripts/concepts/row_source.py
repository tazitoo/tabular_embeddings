#!/usr/bin/env python3
"""Helpers for loading concept-labeling row sources.

This keeps agents, contrastive CSVs, validator CSVs, and dataset-quality
selection aligned on the same underlying row source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import (
    SAE_DATA_DIR,
    get_extraction_layer_taskaware,
    load_test_embeddings,
)


SAE_TEST_SOURCE = "sae_test"
BACKUP_SOURCE = "outer_context_backup"
AUTO_SOURCE = "auto"
VALID_ROW_SOURCES = {SAE_TEST_SOURCE, BACKUP_SOURCE}
VALID_ROW_SOURCE_MODES = {SAE_TEST_SOURCE, BACKUP_SOURCE, AUTO_SOURCE}
DEFAULT_ROW_SOURCE_MODE = SAE_TEST_SOURCE

BACKUP_ROOT = PROJECT_ROOT / "output" / "row_sources" / "outer_context_backup"
BACKUP_OVERRIDE_ROOT = PROJECT_ROOT / "output" / "row_sources" / "outer_context_backup_ctx256"
BACKUP_TRAINALL_ROOT = PROJECT_ROOT / "output" / "row_sources" / "outer_context_backup_trainall"
BACKUP_EMBEDDINGS_DIR = BACKUP_ROOT / "embeddings"
BACKUP_PREDICTIONS_DIR = BACKUP_ROOT / "baseline_predictions"
DEFAULT_BASELINE_PRED_DIR = PROJECT_ROOT / "output" / "baseline_predictions"

_ROW_SOURCE_EMBED_CACHE: dict[tuple[str, str], dict[str, np.ndarray]] = {}
_ROW_SOURCE_INDICES_CACHE: dict[tuple[str, str], dict[str, np.ndarray]] = {}


def _backup_embedding_dirs() -> list[Path]:
    return [
        BACKUP_TRAINALL_ROOT / "embeddings",
        BACKUP_OVERRIDE_ROOT / "embeddings",
        BACKUP_EMBEDDINGS_DIR,
    ]


def _backup_prediction_dirs() -> list[Path]:
    return [
        BACKUP_TRAINALL_ROOT / "baseline_predictions",
        BACKUP_OVERRIDE_ROOT / "baseline_predictions",
        BACKUP_PREDICTIONS_DIR,
    ]


def _load_sae_test_row_indices(model: str) -> dict[str, np.ndarray]:
    key = (model, SAE_TEST_SOURCE)
    if key in _ROW_SOURCE_INDICES_CACHE:
        return _ROW_SOURCE_INDICES_CACHE[key]

    candidates = sorted(SAE_DATA_DIR.glob(f"{model}_taskaware_sae_test.npz"))
    if not candidates:
        candidates = sorted(SAE_DATA_DIR.glob(f"{model}_*_sae_test.npz"))
    if not candidates:
        raise FileNotFoundError(f"No SAE test row-index cache for {model} in {SAE_DATA_DIR}")

    d = np.load(candidates[0], allow_pickle=True)
    row_indices = d["row_indices"]
    samples_per_dataset = d["samples_per_dataset"]

    per_ds: dict[str, np.ndarray] = {}
    offset = 0
    for ds_name, count in samples_per_dataset:
        ds_name, count = str(ds_name), int(count)
        per_ds[ds_name] = row_indices[offset:offset + count]
        offset += count

    _ROW_SOURCE_INDICES_CACHE[key] = per_ds
    return per_ds


def _load_backup_embeddings(model: str) -> dict[str, np.ndarray]:
    key = (model, BACKUP_SOURCE)
    if key in _ROW_SOURCE_EMBED_CACHE:
        return _ROW_SOURCE_EMBED_CACHE[key]

    result: dict[str, np.ndarray] = {}
    per_ds_indices: dict[str, np.ndarray] = {}
    seen_dirs = 0
    for embeddings_dir in _backup_embedding_dirs():
        model_dir = embeddings_dir / model
        if not model_dir.exists():
            continue
        seen_dirs += 1
        for path in sorted(model_dir.glob("*.npz")):
            ds_name = path.stem
            if ds_name in result:
                continue
            d = np.load(path, allow_pickle=True)
            layer_idx = int(get_extraction_layer_taskaware(model, dataset=ds_name))
            layer_names = list(d["layer_names"]) if "layer_names" in d else []
            if layer_names:
                layer_idx = min(layer_idx, len(layer_names) - 1)
            layer_key = f"layer_{layer_idx}"
            if layer_key not in d:
                available = sorted(
                    key for key in d.files
                    if key.startswith("layer_") and key != "layer_names"
                )
                if not available:
                    continue
                layer_key = available[-1]
            result[ds_name] = np.asarray(d[layer_key])
            per_ds_indices[ds_name] = np.asarray(d["row_indices"], dtype=np.int64)

    if seen_dirs == 0:
        raise FileNotFoundError(f"No backup embeddings directories found for {model}")

    _ROW_SOURCE_EMBED_CACHE[key] = result
    _ROW_SOURCE_INDICES_CACHE[key] = per_ds_indices
    return result


def load_row_source_embeddings(model: str, row_source: str) -> dict[str, np.ndarray]:
    """Load per-dataset embeddings for one explicit row source."""
    if row_source == SAE_TEST_SOURCE:
        key = (model, SAE_TEST_SOURCE)
        if key not in _ROW_SOURCE_EMBED_CACHE:
            _ROW_SOURCE_EMBED_CACHE[key] = load_test_embeddings(model)
        return _ROW_SOURCE_EMBED_CACHE[key]
    if row_source == BACKUP_SOURCE:
        return _load_backup_embeddings(model)
    raise ValueError(f"row_source must be one of {sorted(VALID_ROW_SOURCES)}, got {row_source!r}")


def load_row_source_row_indices(model: str, dataset: str, row_source: str) -> Optional[np.ndarray]:
    """Load absolute dataset row indices aligned with one row source."""
    if row_source == SAE_TEST_SOURCE:
        return _load_sae_test_row_indices(model).get(dataset)
    if row_source == BACKUP_SOURCE:
        if (model, BACKUP_SOURCE) not in _ROW_SOURCE_INDICES_CACHE:
            _load_backup_embeddings(model)
        return _ROW_SOURCE_INDICES_CACHE[(model, BACKUP_SOURCE)].get(dataset)
    raise ValueError(f"row_source must be one of {sorted(VALID_ROW_SOURCES)}, got {row_source!r}")


def load_row_source_baseline_predictions(model: str, dataset: str, row_source: str) -> Optional[dict]:
    """Load baseline predictions aligned to one row source."""
    if row_source == SAE_TEST_SOURCE:
        path = DEFAULT_BASELINE_PRED_DIR / model / f"{dataset}.npz"
    elif row_source == BACKUP_SOURCE:
        path = None
        for pred_dir in _backup_prediction_dirs():
            candidate = pred_dir / model / f"{dataset}.npz"
            if candidate.exists():
                path = candidate
                break
        if path is None:
            return None
    else:
        raise ValueError(f"row_source must be one of {sorted(VALID_ROW_SOURCES)}, got {row_source!r}")

    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return {
        "pred_probs": d["pred_probs"],
        "pred_class": d["pred_class"],
        "y_true": d["y_true"],
        "task_type": str(d["task_type"]),
    }


def explicit_row_source_or_default(requested: str, default: str = SAE_TEST_SOURCE) -> str:
    """Resolve a row-source mode to an explicit source."""
    if requested == AUTO_SOURCE:
        return default
    if requested in VALID_ROW_SOURCES:
        return requested
    raise ValueError(f"row_source must be one of {sorted(VALID_ROW_SOURCE_MODES)}, got {requested!r}")


def feature_block_row_source(feature_block: Optional[dict], fallback: str = SAE_TEST_SOURCE) -> str:
    """Return the selected source encoded in a quality-cache feature block."""
    if not feature_block:
        return fallback
    row_source = feature_block.get("selected_row_source")
    if row_source in VALID_ROW_SOURCES:
        return row_source
    return fallback
