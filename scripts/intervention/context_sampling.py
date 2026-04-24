#!/usr/bin/env python3
"""Helpers for deterministic context-row selection."""

from __future__ import annotations

import numpy as np


def select_context_indices(
    *,
    n_rows: int,
    y_train: np.ndarray,
    max_context: int,
    task: str,
    dataframe_style: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """Return deterministic local indices used as context rows.

    For dataframe-style models, this matches the simple random sampling used by
    layer extraction. For numpy models, this matches the stratified sampling
    used for classification and random sampling for regression.
    """
    if n_rows <= max_context:
        return np.arange(n_rows, dtype=int)

    rng = np.random.RandomState(seed)

    if dataframe_style:
        return np.sort(rng.choice(n_rows, max_context, replace=False).astype(int))

    if task == "classification":
        classes, counts = np.unique(y_train, return_counts=True)
        indices = []
        for cls, count in zip(classes, counts):
            cls_idx = np.where(y_train == cls)[0]
            n_take = max(1, int(max_context * count / len(y_train)))
            indices.append(
                rng.choice(cls_idx, size=min(n_take, len(cls_idx)), replace=False)
            )
        indices = np.concatenate(indices)
        if len(indices) > max_context:
            indices = rng.choice(indices, size=max_context, replace=False)
        elif len(indices) < max_context:
            remaining = np.setdiff1d(np.arange(n_rows), indices)
            extra = rng.choice(remaining, size=max_context - len(indices), replace=False)
            indices = np.concatenate([indices, extra])
        return np.sort(indices.astype(int))

    return np.sort(rng.choice(n_rows, size=max_context, replace=False).astype(int))
