"""Tests for scripts/detect_concept_splitting.py — concept splitting detection."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.detect_concept_splitting import (
    evaluate_pair_splitting,
    evaluate_single_group,
    group_matches_by_target,
)


# ── TestGroupMatchesByTarget ──────────────────────────────────────────────


class TestGroupMatchesByTarget:
    def test_groups_only_2plus(self):
        """Single-member assignments excluded from groups."""
        matches = [
            {"idx_a": 0, "idx_b": 100, "r": 0.5, "tier": 1},
            {"idx_a": 1, "idx_b": 100, "r": 0.3, "tier": 2},
            {"idx_a": 2, "idx_b": 200, "r": 0.4, "tier": 1},  # singleton
        ]
        groups = group_matches_by_target(matches)
        assert 100 in groups
        assert 200 not in groups
        assert len(groups[100]) == 2

    def test_preserves_metadata(self):
        """Group entries retain original match fields."""
        matches = [
            {"idx_a": 10, "idx_b": 50, "r": 0.6, "tier": 1},
            {"idx_a": 20, "idx_b": 50, "r": 0.2, "tier": 2},
        ]
        groups = group_matches_by_target(matches)
        idx_a_vals = {m["idx_a"] for m in groups[50]}
        assert idx_a_vals == {10, 20}
        assert groups[50][0]["r"] == 0.6 or groups[50][1]["r"] == 0.6

    def test_empty_matches(self):
        """No matches produces empty dict."""
        assert group_matches_by_target([]) == {}

    def test_all_singletons(self):
        """If every A maps to a unique B, no groups returned."""
        matches = [
            {"idx_a": i, "idx_b": i * 10, "r": 0.5, "tier": 1}
            for i in range(10)
        ]
        assert group_matches_by_target(matches) == {}


# ── TestTestSingleGroup ──────────────────────────────────────────────────


class TestTestSingleGroup:
    def test_genuine_split(self):
        """Multiple features jointly predict target better than any individual."""
        rng = np.random.RandomState(42)
        n = 5000
        # Target is sum of 3 independent signals
        s1, s2, s3 = rng.randn(n), rng.randn(n), rng.randn(n)
        y = s1 + s2 + s3 + 0.1 * rng.randn(n)

        pooled_A = np.zeros((n, 10))
        pooled_A[:, 0] = s1 + 0.1 * rng.randn(n)
        pooled_A[:, 1] = s2 + 0.1 * rng.randn(n)
        pooled_A[:, 2] = s3 + 0.1 * rng.randn(n)
        for i in range(3, 10):
            pooled_A[:, i] = rng.randn(n) * 0.01

        pooled_B = np.zeros((n, 5))
        pooled_B[:, 3] = y

        result = evaluate_single_group(
            pooled_A, pooled_B, [0, 1, 2], 3,
            match_r_by_idx={0: 0.5, 1: 0.5, 2: 0.5},
        )
        assert result["classification"] == "split"
        assert result["group_r2"] > result["best_individual_r2"] + 0.05
        assert result["delta_r2"] > 0.05
        assert result["n_members"] == 3

    def test_single_match_with_hangers_on(self):
        """One strong predictor + noise features → single_match."""
        rng = np.random.RandomState(42)
        n = 5000
        signal = rng.randn(n)
        y = signal + 0.1 * rng.randn(n)

        pooled_A = np.zeros((n, 5))
        pooled_A[:, 0] = signal + 0.05 * rng.randn(n)
        for i in range(1, 5):
            pooled_A[:, i] = rng.randn(n) * 0.01

        pooled_B = np.zeros((n, 3))
        pooled_B[:, 1] = y

        result = evaluate_single_group(
            pooled_A, pooled_B, [0, 1, 2, 3, 4], 1,
            match_r_by_idx={0: 0.9, 1: 0.05, 2: 0.04, 3: 0.03, 4: 0.02},
        )
        assert result["classification"] == "single_match"
        assert result["best_individual_idx"] == 0
        assert result["delta_r2"] < 0.05

    def test_noise_group(self):
        """All features independent of target → noise."""
        rng = np.random.RandomState(42)
        n = 5000
        pooled_A = rng.randn(n, 5)
        pooled_B = rng.randn(n, 3)

        result = evaluate_single_group(
            pooled_A, pooled_B, [0, 1, 2, 3, 4], 1,
            match_r_by_idx={i: 0.05 for i in range(5)},
        )
        assert result["classification"] == "noise"
        assert result["group_r2"] < 0.3

    def test_two_member_group(self):
        """Smallest possible group (2 members) handled correctly."""
        rng = np.random.RandomState(42)
        n = 5000
        s = rng.randn(n)
        pooled_A = np.zeros((n, 3))
        pooled_A[:, 0] = s + 0.1 * rng.randn(n)
        pooled_A[:, 1] = rng.randn(n) * 0.01
        pooled_B = np.zeros((n, 2))
        pooled_B[:, 1] = s + 0.1 * rng.randn(n)

        result = evaluate_single_group(
            pooled_A, pooled_B, [0, 1], 1,
            match_r_by_idx={0: 0.9, 1: 0.05},
        )
        assert result["n_members"] == 2
        assert result["classification"] in ("split", "single_match", "noise")

    def test_constant_target_safe(self):
        """Constant y → noise, no crash."""
        rng = np.random.RandomState(42)
        n = 100
        pooled_A = rng.randn(n, 3)
        pooled_B = np.zeros((n, 2))  # all zeros

        result = evaluate_single_group(
            pooled_A, pooled_B, [0, 1, 2], 0,
            match_r_by_idx={0: 0.1, 1: 0.05, 2: 0.03},
        )
        assert result["classification"] == "noise"
        assert result["group_r2"] == 0.0


# ── TestTestPairSplitting ─────────────────────────────────────────────────


class TestTestPairSplitting:
    def test_mixed_groups(self):
        """Pair with a genuine split group and noise group."""
        rng = np.random.RandomState(42)
        n = 5000

        # Build pooled_A (10 features) and pooled_B (5 features)
        s1, s2, s3 = rng.randn(n), rng.randn(n), rng.randn(n)

        pooled_A = np.zeros((n, 10))
        pooled_A[:, 0] = s1 + 0.1 * rng.randn(n)
        pooled_A[:, 1] = s2 + 0.1 * rng.randn(n)
        pooled_A[:, 2] = s3 + 0.1 * rng.randn(n)
        for i in range(3, 10):
            pooled_A[:, i] = rng.randn(n) * 0.01

        pooled_B = np.zeros((n, 5))
        pooled_B[:, 0] = s1 + s2 + s3 + 0.1 * rng.randn(n)  # split target
        pooled_B[:, 1] = rng.randn(n)  # noise target

        # Group 1: features 0,1,2 → B:0 (should be split)
        # Group 2: features 3,4 → B:1 (should be noise)
        matches = [
            {"idx_a": 0, "idx_b": 0, "r": 0.5, "tier": 1},
            {"idx_a": 1, "idx_b": 0, "r": 0.4, "tier": 2},
            {"idx_a": 2, "idx_b": 0, "r": 0.3, "tier": 2},
            {"idx_a": 3, "idx_b": 1, "r": 0.05, "tier": 2},
            {"idx_a": 4, "idx_b": 1, "r": 0.04, "tier": 2},
        ]

        result = evaluate_pair_splitting(pooled_A, pooled_B, matches)
        assert result["n_groups_tested"] == 2
        assert result["n_split"] >= 1
        assert result["n_noise"] >= 1

    def test_no_groups(self):
        """All 1-to-1 matches → 0 groups tested."""
        matches = [
            {"idx_a": i, "idx_b": i + 100, "r": 0.5, "tier": 1}
            for i in range(10)
        ]
        result = evaluate_pair_splitting(
            np.random.randn(100, 10),
            np.random.randn(100, 200),
            matches,
        )
        assert result["n_groups_tested"] == 0
        assert result["groups"] == []
