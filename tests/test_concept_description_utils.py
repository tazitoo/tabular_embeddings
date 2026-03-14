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


def test_format_haiku_prompt_includes_probes():
    """Haiku prompt includes probe consensus and asks for 2-5 words."""
    from scripts.concept_description_utils import format_haiku_prompt

    probes = [("frac_zeros", 5, -1.2), ("numeric_skewness", 3, 0.8)]
    prompt = format_haiku_prompt(group_id=0, probes=probes, n_models=4, n_members=12)

    assert "frac_zeros" in prompt
    assert "2-5 words" in prompt
    assert "4 models" in prompt


def test_format_sonnet_group_prompt_includes_samples():
    """Sonnet group prompt includes contrastive examples before probes."""
    from scripts.concept_description_utils import format_sonnet_group_prompt

    probes = [("frac_zeros", 5, -1.2)]
    high_rows = [{"col_a": 0.0, "col_b": 1.5}, {"col_a": 0.0, "col_b": 2.3}]
    low_rows = [{"col_a": 0.9, "col_b": 0.1}, {"col_a": 0.7, "col_b": 0.4}]

    prompt = format_sonnet_group_prompt(
        group_id=0, probes=probes, n_models=3, n_members=8,
        high_rows=high_rows, low_rows=low_rows,
    )

    assert "frac_zeros" in prompt
    assert "TOP-ACTIVATING" in prompt
    assert "1-2 sentences" in prompt
    assert "MONOSEMANTIC" in prompt
    # Contrastive examples should appear BEFORE probe statistics
    activating_pos = prompt.index("TOP-ACTIVATING")
    probe_pos = prompt.index("STATISTICAL GUIDANCE")
    assert activating_pos < probe_pos, "Contrastive examples must come before probes"


def test_format_sonnet_unexplained_prompt_includes_landmarks():
    """Unexplained feature prompt includes landmark descriptions."""
    from scripts.concept_description_utils import format_sonnet_unexplained_prompt

    high_rows = [{"x": 1.0}]
    low_rows = [{"x": 0.0}]
    landmarks = [
        ("sparse rows with many zero-valued numerics", 0.34),
        ("extreme outlier rows", 0.28),
    ]

    prompt = format_sonnet_unexplained_prompt(
        model="TabPFN", feat_idx=789,
        high_rows=high_rows, low_rows=low_rows, landmarks=landmarks,
    )

    assert "sparse rows" in prompt
    assert "0.340" in prompt
    assert "landmark" in prompt.lower() or "neighbor" in prompt.lower()
