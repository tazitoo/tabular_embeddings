"""Tests for concept description utilities."""
import numpy as np
import pytest


def test_get_activating_samples_returns_correct_counts():
    """Top-k activating and bottom-k non-activating rows."""
    from scripts.concept_description_utils import get_activating_samples

    rng = np.random.RandomState(42)
    activations = rng.rand(20, 4)
    activations[:5, 1] = 10.0
    activations[10:, 1] = 0.0

    high, low = get_activating_samples(activations, feat_idx=1, top_k=3, bottom_k=3)

    assert len(high) == 3
    assert len(low) == 3
    assert all(i < 5 for i in high)
    assert all(i >= 10 for i in low)


def test_get_activating_samples_handles_sparse_feature():
    """Feature with very few activating rows returns what's available."""
    from scripts.concept_description_utils import get_activating_samples

    activations = np.zeros((20, 4))
    activations[0, 2] = 1.0

    high, low = get_activating_samples(activations, feat_idx=2, top_k=5, bottom_k=5)

    assert len(high) == 1
    assert len(low) == 5
    assert high[0] == 0
