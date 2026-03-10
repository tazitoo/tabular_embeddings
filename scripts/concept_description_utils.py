"""Utilities for generating concept descriptions.

Extracts activating/non-activating row samples and formats prompts
for LLM-based concept labeling.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_activating_samples(
    activations: np.ndarray,
    feat_idx: int,
    top_k: int = 5,
    bottom_k: int = 5,
    active_threshold: float = 0.001,
) -> Tuple[List[int], List[int]]:
    """Get indices of top-K activating and bottom-K non-activating rows.

    Args:
        activations: (n_samples, hidden_dim) SAE activation matrix.
        feat_idx: Feature index to analyze.
        top_k: Number of highest-activating rows to return.
        bottom_k: Number of non-activating rows to return.
        active_threshold: Below this, a row is considered non-activating.

    Returns:
        (high_indices, low_indices) — sorted by activation descending / ascending.
    """
    feat_acts = activations[:, feat_idx]

    # High: top-k by activation value (only rows that actually activate)
    active_mask = feat_acts > active_threshold
    active_indices = np.where(active_mask)[0]
    if len(active_indices) == 0:
        return [], list(np.argsort(feat_acts)[:bottom_k])

    n_high = min(top_k, len(active_indices))
    high_indices = active_indices[np.argsort(feat_acts[active_indices])[-n_high:][::-1]]

    # Low: bottom-k from non-activating rows
    inactive_indices = np.where(~active_mask)[0]
    n_low = min(bottom_k, len(inactive_indices))
    low_indices = inactive_indices[np.argsort(feat_acts[inactive_indices])[:n_low]]

    return list(high_indices), list(low_indices)
