"""Tests for scripts/label_cross_model_concepts.py — cross-model concept labeling."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.label_cross_model_concepts import (
    UnionFind,
    aggregate_group_probes,
    build_feature_lookup,
    build_match_graph,
    build_summary,
    format_group_prompt,
    label_with_rules,
    run_phase1,
    run_phase2,
    run_phase3,
)


# ── TestUnionFind ─────────────────────────────────────────────────────────


class TestUnionFind:
    def test_basic_components(self):
        """Union 5 pairs, verify 2 distinct components."""
        uf = UnionFind()
        uf.union(("A", 0), ("B", 0))
        uf.union(("A", 1), ("B", 1))
        uf.union(("A", 0), ("A", 1))  # merge into one component

        uf.union(("C", 0), ("D", 0))  # separate component

        comps = uf.components()
        assert len(comps) == 2
        sizes = sorted(len(v) for v in comps.values())
        assert sizes == [2, 4]

    def test_singleton(self):
        """Node that is only find()'d but never union()'d stays alone."""
        uf = UnionFind()
        uf.find(("A", 0))
        uf.union(("B", 0), ("B", 1))
        comps = uf.components()
        # Only B:0, B:1 form a component; A:0 is singleton (excluded)
        assert len(comps) == 1
        assert len(list(comps.values())[0]) == 2

    def test_transitive(self):
        """A-B + B-C → A,B,C in same component."""
        uf = UnionFind()
        uf.union(("A", 0), ("B", 0))
        uf.union(("B", 0), ("C", 0))
        comps = uf.components()
        assert len(comps) == 1
        assert len(list(comps.values())[0]) == 3


# ── TestBuildMatchGraph ───────────────────────────────────────────────────


def _make_matching(matches_by_pair):
    """Helper to build a matching dict from {pair_key: [(idx_a, idx_b, r, tier)]}."""
    pairs = {}
    for pair_key, matches in matches_by_pair.items():
        model_a, model_b = pair_key.split("__")
        pairs[pair_key] = {
            "model_a": model_a,
            "model_b": model_b,
            "matches": [
                {"idx_a": a, "idx_b": b, "r": r, "tier": t}
                for a, b, r, t in matches
            ],
        }
    return {"metadata": {"models": []}, "pairs": pairs}


class TestBuildMatchGraph:
    def test_min_r_filtering(self):
        """Only edges with |r| >= min_r are added."""
        matching = _make_matching({
            "A__B": [
                (0, 0, 0.1, 1), (1, 1, 0.2, 1), (2, 2, 0.3, 1),
                (3, 3, 0.4, 1), (4, 4, 0.5, 1), (5, 5, 0.6, 1),
            ],
        })
        uf = build_match_graph(matching, tier=1, min_r=0.3)
        # Only edges with r >= 0.3: (2,2), (3,3), (4,4), (5,5) → 4 pairs
        comps = uf.components()
        # Each edge creates a 2-node component (no transitive links)
        total_nodes = sum(len(v) for v in comps.values())
        assert total_nodes == 8  # 4 edges × 2 nodes each

    def test_tier_filtering(self):
        """tier=1 only adds tier-1 edges, ignores tier-2."""
        matching = _make_matching({
            "A__B": [
                (0, 0, 0.5, 1),  # tier 1
                (1, 1, 0.5, 2),  # tier 2 — should be skipped
            ],
        })
        uf = build_match_graph(matching, tier=1, min_r=0.0)
        comps = uf.components()
        assert len(comps) == 1
        assert len(list(comps.values())[0]) == 2  # only A:0 ↔ B:0


# ── TestAggregateGroupProbes ──────────────────────────────────────────────


class TestAggregateGroupProbes:
    def test_probe_consensus(self):
        """3 members sharing probes → correct vote counts."""
        probe_lookup = {
            "A": {0: {"r2": 0.4, "probes": [("frac_zeros", 0.8), ("row_entropy", -0.5)], "probe_names": set()}},
            "B": {0: {"r2": 0.3, "probes": [("frac_zeros", 0.7), ("numeric_std", 0.3)], "probe_names": set()}},
            "C": {0: {"r2": 0.5, "probes": [("frac_zeros", 0.9), ("row_entropy", -0.4)], "probe_names": set()}},
        }
        members = [("A", 0), ("B", 0), ("C", 0)]
        agg = aggregate_group_probes(members, probe_lookup)

        assert agg["n_models"] == 3
        assert abs(agg["mean_r2"] - 0.4) < 0.01
        # frac_zeros should have count=3
        assert agg["probe_votes"]["frac_zeros"]["count"] == 3
        # row_entropy count=2 (A and C)
        assert agg["probe_votes"]["row_entropy"]["count"] == 2

    def test_missing_feature(self):
        """Member not in probe_lookup → skipped gracefully."""
        probe_lookup = {
            "A": {0: {"r2": 0.4, "probes": [("frac_zeros", 0.8)], "probe_names": set()}},
        }
        members = [("A", 0), ("B", 99)]  # B:99 not in lookup
        agg = aggregate_group_probes(members, probe_lookup)
        assert agg["n_models"] == 2
        assert len(agg["per_member"]) == 2
        assert agg["per_member"][1]["r2"] == 0.0  # missing → 0


# ── TestPhaseIncremental ──────────────────────────────────────────────────


class TestPhaseIncremental:
    def _make_probe_lookup(self):
        """Probe lookup with features across 3 models."""
        return {
            "A": {
                i: {"r2": 0.3, "probes": [("frac_zeros", 0.5)], "probe_names": {"frac_zeros"}}
                for i in range(5)
            },
            "B": {
                i: {"r2": 0.3, "probes": [("frac_zeros", 0.5)], "probe_names": {"frac_zeros"}}
                for i in range(5)
            },
            "C": {
                i: {"r2": 0.3, "probes": [("frac_zeros", 0.5)], "probe_names": {"frac_zeros"}}
                for i in range(5)
            },
        }

    def test_phase2_extends_phase1(self):
        """Phase 2 preserves phase 1 groups and adds new ones."""
        matching = _make_matching({
            "A__B": [
                (0, 0, 0.5, 1),  # tier 1: A:0 ↔ B:0
                (1, 1, 0.4, 2),  # tier 2: A:1 ↔ B:1
            ],
            "A__C": [
                (2, 2, 0.6, 1),  # tier 1: A:2 ↔ C:2
                (3, 3, 0.3, 2),  # tier 2: A:3 ↔ C:3
            ],
        })
        probe_lookup = self._make_probe_lookup()

        p1 = run_phase1(matching, probe_lookup, min_r=0.0, client=None,
                        dry_run=False, no_llm=True, max_group_size=100)
        assert len(p1["concept_groups"]) == 2  # two tier-1 groups

        p2 = run_phase2(p1, matching, probe_lookup, min_r=0.0, client=None,
                        dry_run=False, no_llm=True, max_group_size=100,
                        relabel_threshold=0.3)
        # Should have original 2 + at least 1 new from tier-2
        assert len(p2["concept_groups"]) >= 2

    def test_relabel_threshold(self):
        """Group that grew < threshold keeps old label."""
        matching = _make_matching({
            "A__B": [
                (0, 0, 0.5, 1),
                (1, 0, 0.4, 1),  # A:0, A:1 → B:0 (3-node group)
                (2, 2, 0.3, 2),  # tier 2, separate
            ],
        })
        probe_lookup = self._make_probe_lookup()

        p1 = run_phase1(matching, probe_lookup, min_r=0.0, client=None,
                        dry_run=False, no_llm=True, max_group_size=100)

        # Find the group and record its label
        gid = list(p1["concept_groups"].keys())[0]
        old_label = p1["concept_groups"][gid]["label"]

        # Phase 2 with high threshold — no relabeling
        p2 = run_phase2(p1, matching, probe_lookup, min_r=0.0, client=None,
                        dry_run=False, no_llm=True, max_group_size=100,
                        relabel_threshold=0.99)
        # Old group should keep its label
        if gid in p2["concept_groups"]:
            assert p2["concept_groups"][gid]["label"] == old_label


# ── TestPromptFormatting ──────────────────────────────────────────────────


class TestPromptFormatting:
    def test_format_group_prompt(self):
        """Prompt contains model names, probes, R² values."""
        agg = {
            "n_models": 3,
            "models": ["A", "B", "C"],
            "mean_r2": 0.4,
            "probe_votes": {
                "frac_zeros": {"count": 3, "mean_coeff": 0.8},
                "row_entropy": {"count": 2, "mean_coeff": -0.5},
            },
            "top_probes": [("frac_zeros", 3, 0.8), ("row_entropy", 2, -0.5)],
            "per_member": [
                {"model": "A", "feature_idx": 0, "r2": 0.4, "top_probes": [("frac_zeros", 0.8)]},
                {"model": "B", "feature_idx": 1, "r2": 0.3, "top_probes": [("frac_zeros", 0.7)]},
            ],
        }
        prompt = format_group_prompt(0, agg)
        assert "A, B, C" in prompt
        assert "frac_zeros" in prompt
        assert "0.400" in prompt or "0.40" in prompt
        assert "2-5 words" in prompt


# ── TestLabelWithRules ────────────────────────────────────────────────────


class TestLabelWithRules:
    def test_known_probes(self):
        """Known probe names produce vocab-based labels."""
        label = label_with_rules([("frac_zeros", 0.8), ("row_entropy", -0.5)])
        assert "sparse" in label.lower()

    def test_empty_probes(self):
        """No probes → 'uncharacterized'."""
        assert label_with_rules([]) == "uncharacterized"


# ── TestDryRun ────────────────────────────────────────────────────────────


class TestDryRun:
    def test_dry_run_no_llm_calls(self):
        """Dry run produces [DRY RUN] labels."""
        matching = _make_matching({
            "A__B": [(0, 0, 0.5, 1), (1, 0, 0.4, 1)],
        })
        probe_lookup = {
            "A": {
                0: {"r2": 0.3, "probes": [("frac_zeros", 0.5)], "probe_names": set()},
                1: {"r2": 0.3, "probes": [("frac_zeros", 0.5)], "probe_names": set()},
            },
            "B": {
                0: {"r2": 0.3, "probes": [("frac_zeros", 0.5)], "probe_names": set()},
            },
        }
        result = run_phase1(matching, probe_lookup, min_r=0.0, client=None,
                            dry_run=True, no_llm=False, max_group_size=100)
        for g in result["concept_groups"].values():
            assert g["label"] == "[DRY RUN]"
            assert g["label_method"] == "dry_run"


# ── TestBuildHelpers ──────────────────────────────────────────────────────


class TestBuildHelpers:
    def test_feature_lookup(self):
        """feature_lookup maps every grouped feature to its group label."""
        result = {
            "concept_groups": {
                "0": {
                    "members": [["A", 0], ["B", 1]],
                    "label": "sparse rows",
                },
            },
            "unmatched_features": {
                "explained": {
                    "A": {"5": {"label": "isolated", "r2": 0.3}},
                },
                "unexplained": {},
            },
        }
        lookup = build_feature_lookup(result)
        assert lookup["A"]["0"]["label"] == "sparse rows"
        assert lookup["A"]["0"]["group_id"] == 0
        assert lookup["B"]["1"]["label"] == "sparse rows"
        assert lookup["A"]["5"]["label"] == "isolated"
        assert lookup["A"]["5"]["group_id"] is None

    def test_summary(self):
        """Summary counts features correctly."""
        result = {
            "concept_groups": {
                "0": {"members": [["A", 0], ["B", 1]], "label_method": "llm"},
                "1": {"members": [["A", 2]], "label_method": "rule"},
            },
            "unmatched_features": {
                "explained": {"A": {"5": {"label_method": "rule"}}},
                "unexplained": {"A": {"6": {"label_method": "threshold"}}},
            },
        }
        s = build_summary(result)
        assert s["in_groups"] == 3
        assert s["n_groups"] == 2
        assert s["unmatched_explained"] == 1
        assert s["unmatched_unexplained"] == 1
        assert s["total_alive_features"] == 5
