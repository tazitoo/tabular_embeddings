"""Tests for concept hierarchy builder (Epic 1).

Tests cover:
1. Every alive feature appears exactly once in the hierarchy
2. Band assignment matches Matryoshka dims for each model
3. Category assignment agrees with pymfe_taxonomy.json
4. unique_to[A][vs_B] + shared[A__B] = all groups containing A
5. traverse() with filters returns correct subsets
6. Hierarchy is valid JSON, round-trips correctly
"""

import json
import tempfile
from pathlib import Path

import pytest

from scripts.build_concept_hierarchy import (
    BAND_NAMES,
    UNEXPLAINED_CATEGORY,
    ROW_LEVEL_CATEGORY,
    assign_band,
    assign_category,
    build_feature_index,
    build_hierarchy,
    build_probe_to_category,
    traverse,
    _build_model_comparison,
    _normalize_model_name,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_taxonomy():
    """Minimal PyMFE taxonomy with 3 categories."""
    return {
        "categories": {
            "Statistical": {
                "description": "Descriptive statistics",
                "features": ["kurtosis.sd", "skewness.mean", "cor.mean"],
            },
            "Complexity": {
                "description": "Data complexity",
                "features": ["hubs.mean", "n2.mean", "c1"],
            },
            "Model-Based": {
                "description": "Decision tree properties",
                "features": ["best_node.sd", "leaves"],
            },
        }
    }


@pytest.fixture
def sample_sae_configs():
    """SAE configs for 3 test models with different hidden dims."""
    return {
        "tabpfn": {
            "input_dim": 192,
            "hidden_dim": 1536,
            "matryoshka_dims": [96, 192, 384, 768, 1536],
            "topk": 64,
        },
        "mitra": {
            "input_dim": 512,
            "hidden_dim": 4096,
            "matryoshka_dims": [256, 512, 1024, 2048, 4096],
            "topk": 256,
        },
        "hyperfast": {
            "input_dim": 784,
            "hidden_dim": 6272,
            "matryoshka_dims": [392, 784, 1568, 3136, 6272],
            "topk": 256,
        },
    }


@pytest.fixture
def sample_labels():
    """Minimal cross-model concept labels with matched and unmatched features."""
    return {
        "metadata": {
            "n_models": 3,
            "n_llm_calls": 5,
        },
        "concept_groups": {
            "10": {
                "members": [
                    ["TabPFN", 42],
                    ["TabPFN", 80],
                    ["Mitra", 300],
                ],
                "n_models": 2,
                "tier": 1,
                "mean_r2": 0.15,
                "label": "high kurtosis features",
                "top_probes": [
                    ["kurtosis.sd", 2, 0.12],
                    ["hubs.mean", 2, -0.05],
                ],
                "phase_added": 1,
            },
            "20": {
                "members": [
                    ["Mitra", 500],
                    ["HyperFast", 100],
                ],
                "n_models": 2,
                "tier": 2,
                "mean_r2": 0.08,
                "label": "sparse uniform rows",
                "top_probes": [
                    ["frac_zeros", 2, 0.09],  # row-level probe
                ],
                "phase_added": 1,
            },
            "30": {
                "members": [
                    ["TabPFN", 10],
                    ["Mitra", 50],
                    ["HyperFast", 200],
                ],
                "n_models": 3,
                "tier": 1,
                "mean_r2": 0.25,
                "label": "correlated structure",
                "top_probes": [
                    ["cor.mean", 3, 0.20],
                    ["skewness.mean", 3, 0.05],
                ],
                "phase_added": 1,
            },
        },
        "feature_lookup": {
            "TabPFN": {
                "10": {"group_id": 30, "label": "correlated structure", "category": "matched"},
                "42": {"group_id": 10, "label": "high kurtosis features", "category": "matched"},
                "80": {"group_id": 10, "label": "high kurtosis features", "category": "matched"},
                "900": {"group_id": None, "label": "unexplained", "category": "unmatched"},
            },
            "Mitra": {
                "50": {"group_id": 30, "label": "correlated structure", "category": "matched"},
                "300": {"group_id": 10, "label": "high kurtosis features", "category": "matched"},
                "500": {"group_id": 20, "label": "sparse uniform rows", "category": "matched"},
            },
            "HyperFast": {
                "100": {"group_id": 20, "label": "sparse uniform rows", "category": "matched"},
                "200": {"group_id": 30, "label": "correlated structure", "category": "matched"},
                "5000": {"group_id": None, "label": "unexplained", "category": "unmatched"},
            },
        },
        "unmatched_features": {
            "explained": {
                "TabPFN": {
                    "900": {
                        "r2": 0.30,
                        "label": "decision tree depth signal",
                        "top_probes": [["best_node.sd", -0.15], ["leaves", 0.10]],
                    }
                },
            },
            "unexplained": {
                "HyperFast": {
                    "5000": {
                        "r2": 0.002,
                        "label": "weak unknown pattern",
                        "top_probes": [["frac_zeros", 0.001]],
                    }
                },
            },
        },
        "summary": {
            "total_alive_features": 10,
            "in_groups": 8,
            "n_groups": 3,
            "unmatched_explained": 1,
            "unmatched_unexplained": 1,
        },
    }


@pytest.fixture
def built_hierarchy(sample_labels, sample_taxonomy, sample_sae_configs):
    """Pre-built hierarchy for testing."""
    return build_hierarchy(sample_labels, sample_taxonomy, sample_sae_configs)


# ── Test assign_band ──────────────────────────────────────────────────────────


class TestAssignBand:
    def test_s1_band(self):
        """Feature 0 should be in S1."""
        dims = [96, 192, 384, 768, 1536]
        assert assign_band(0, dims) == "S1"
        assert assign_band(95, dims) == "S1"

    def test_s2_band(self):
        dims = [96, 192, 384, 768, 1536]
        assert assign_band(96, dims) == "S2"
        assert assign_band(191, dims) == "S2"

    def test_s5_band(self):
        dims = [96, 192, 384, 768, 1536]
        assert assign_band(768, dims) == "S5"
        assert assign_band(1535, dims) == "S5"

    def test_boundary_exact(self):
        """Feature at exact boundary goes to next band."""
        dims = [96, 192, 384, 768, 1536]
        assert assign_band(96, dims) == "S2"
        assert assign_band(192, dims) == "S3"

    def test_different_model_dims(self):
        """Different models have different band boundaries."""
        tabpfn_dims = [96, 192, 384, 768, 1536]
        mitra_dims = [256, 512, 1024, 2048, 4096]

        # Feature 100 is S2 for TabPFN but S1 for Mitra
        assert assign_band(100, tabpfn_dims) == "S2"
        assert assign_band(100, mitra_dims) == "S1"


# ── Test assign_category ─────────────────────────────────────────────────────


class TestAssignCategory:
    def test_statistical_category(self, sample_taxonomy):
        probe_to_cat = build_probe_to_category(sample_taxonomy)
        probes = [["kurtosis.sd", 2, 0.12], ["hubs.mean", 2, -0.05]]
        cat = assign_category(probes, probe_to_cat, r2=0.15)
        assert cat == "Statistical"  # kurtosis.sd has higher abs coeff

    def test_complexity_wins_by_coefficient(self, sample_taxonomy):
        probe_to_cat = build_probe_to_category(sample_taxonomy)
        probes = [["hubs.mean", 2, 0.20], ["kurtosis.sd", 2, 0.05]]
        cat = assign_category(probes, probe_to_cat, r2=0.15)
        assert cat == "Complexity"  # hubs.mean has higher abs coeff

    def test_unexplained_low_r2(self, sample_taxonomy):
        probe_to_cat = build_probe_to_category(sample_taxonomy)
        probes = [["kurtosis.sd", 2, 0.12]]
        cat = assign_category(probes, probe_to_cat, r2=0.005)
        assert cat == UNEXPLAINED_CATEGORY

    def test_unexplained_no_probes(self, sample_taxonomy):
        probe_to_cat = build_probe_to_category(sample_taxonomy)
        cat = assign_category([], probe_to_cat, r2=0.5)
        assert cat == UNEXPLAINED_CATEGORY

    def test_row_level_probe(self, sample_taxonomy):
        probe_to_cat = build_probe_to_category(sample_taxonomy)
        probes = [["frac_zeros", 2, 0.15]]  # Not in taxonomy
        cat = assign_category(probes, probe_to_cat, r2=0.10)
        assert cat == ROW_LEVEL_CATEGORY


# ── Test build_hierarchy ─────────────────────────────────────────────────────


class TestBuildHierarchy:
    def test_all_features_present(self, built_hierarchy, sample_labels):
        """Every alive feature appears exactly once in the hierarchy."""
        expected = sample_labels["summary"]["total_alive_features"]
        actual = built_hierarchy["metadata"]["n_features"]
        assert actual == expected

    def test_feature_index_complete(self, built_hierarchy, sample_labels):
        """Feature index has entries for all features in the labels."""
        fi = built_hierarchy["feature_index"]
        total = sum(len(feats) for feats in fi.values())
        assert total == sample_labels["summary"]["total_alive_features"]

    def test_feature_index_unique(self, built_hierarchy):
        """No duplicate entries in feature index."""
        fi = built_hierarchy["feature_index"]
        seen = set()
        for model, feats in fi.items():
            for fid in feats:
                key = (model, fid)
                assert key not in seen, f"Duplicate: {key}"
                seen.add(key)

    def test_matched_features_have_group_id(self, built_hierarchy):
        """Matched features have a non-None group_id."""
        fi = built_hierarchy["feature_index"]
        for model, feats in fi.items():
            for fid, info in feats.items():
                if info["matched"]:
                    assert info["group_id"] is not None

    def test_unmatched_features_have_no_group(self, built_hierarchy):
        """Unmatched features have group_id=None."""
        fi = built_hierarchy["feature_index"]
        for model, feats in fi.items():
            for fid, info in feats.items():
                if not info["matched"]:
                    assert info["group_id"] is None

    def test_metadata_models(self, built_hierarchy):
        """Metadata contains correct model information."""
        models = built_hierarchy["metadata"]["models"]
        assert "TabPFN" in models
        assert models["TabPFN"]["hidden_dim"] == 1536
        assert models["TabPFN"]["bands"] == [96, 192, 384, 768, 1536]

    def test_hierarchy_bands_present(self, built_hierarchy):
        """All 5 bands are present in hierarchy."""
        for band in BAND_NAMES:
            assert band in built_hierarchy["hierarchy"]

    def test_group_label_preserved(self, built_hierarchy):
        """Group labels from input are preserved in hierarchy."""
        hierarchy = built_hierarchy["hierarchy"]
        found_labels = set()
        for band_data in hierarchy.values():
            for cat_data in band_data.values():
                for gid, group in cat_data.get("groups", {}).items():
                    found_labels.add(group["label"])
        assert "high kurtosis features" in found_labels
        assert "correlated structure" in found_labels

    def test_band_assignment_correct(self, built_hierarchy):
        """Features are assigned to correct bands based on their model's dims."""
        fi = built_hierarchy["feature_index"]
        # TabPFN feature 42: dims=[96,192,...] → 42 < 96 → S1
        assert fi["TabPFN"]["42"]["band"] == "S1"
        # TabPFN feature 900: 768 <= 900 < 1536 → S5
        assert fi["TabPFN"]["900"]["band"] == "S5"
        # HyperFast feature 5000: 3136 <= 5000 < 6272 → S5
        assert fi["HyperFast"]["5000"]["band"] == "S5"


# ── Test model_comparison ────────────────────────────────────────────────────


class TestModelComparison:
    def test_unique_plus_shared_equals_all(self, built_hierarchy):
        """unique_to[A][vs_B] + shared[A__B] = all groups containing A."""
        comp = built_hierarchy["model_comparison"]
        fi = built_hierarchy["feature_index"]

        # Get all group_ids per model from feature index
        model_group_ids = {}
        for model, feats in fi.items():
            gids = {info["group_id"] for info in feats.values() if info["group_id"] is not None}
            model_group_ids[model] = gids

        for model_a, groups_a in model_group_ids.items():
            for model_b, groups_b in model_group_ids.items():
                if model_a == model_b:
                    continue

                vs_key = f"vs_{model_b}"
                unique_groups = set(comp["unique_to"].get(model_a, {}).get(vs_key, {}).get("groups", []))

                # Find shared key (either order)
                shared_key_1 = f"{model_a}__{model_b}"
                shared_key_2 = f"{model_b}__{model_a}"
                shared_groups = set()
                if shared_key_1 in comp["shared"]:
                    shared_groups = set(comp["shared"][shared_key_1]["groups"])
                elif shared_key_2 in comp["shared"]:
                    shared_groups = set(comp["shared"][shared_key_2]["groups"])

                # unique + shared should equal all groups for A
                assert unique_groups | shared_groups == groups_a, (
                    f"Mismatch for {model_a} vs {model_b}: "
                    f"unique={unique_groups}, shared={shared_groups}, all={groups_a}"
                )

    def test_unique_groups_are_disjoint(self, built_hierarchy):
        """Unique groups for A vs B should not overlap with shared A__B groups."""
        comp = built_hierarchy["model_comparison"]
        for model_a, vs_data in comp["unique_to"].items():
            for vs_key, info in vs_data.items():
                model_b = vs_key.replace("vs_", "")
                unique_set = set(info["groups"])

                shared_key_1 = f"{model_a}__{model_b}"
                shared_key_2 = f"{model_b}__{model_a}"
                shared_set = set()
                if shared_key_1 in comp["shared"]:
                    shared_set = set(comp["shared"][shared_key_1]["groups"])
                elif shared_key_2 in comp["shared"]:
                    shared_set = set(comp["shared"][shared_key_2]["groups"])

                assert not unique_set & shared_set, (
                    f"Overlap between unique and shared for {model_a} vs {model_b}"
                )


# ── Test traverse ─────────────────────────────────────────────────────────────


class TestTraverse:
    def test_traverse_all(self, built_hierarchy):
        """Unfiltered traverse returns everything."""
        result = traverse(built_hierarchy["hierarchy"])
        # Should have at least some data
        has_data = False
        for band_data in result.values():
            for cat_data in band_data.values():
                if cat_data.get("groups") or cat_data.get("unmatched"):
                    has_data = True
        assert has_data

    def test_traverse_by_band(self, built_hierarchy):
        """Filter by band returns only that band."""
        result = traverse(built_hierarchy["hierarchy"], band="S1")
        assert set(result.keys()) <= {"S1"}

    def test_traverse_by_category(self, built_hierarchy):
        """Filter by category returns only that category."""
        result = traverse(built_hierarchy["hierarchy"], category="Statistical")
        for band_data in result.values():
            assert set(band_data.keys()) <= {"Statistical"}

    def test_traverse_by_model(self, built_hierarchy):
        """Filter by model returns only groups containing that model."""
        result = traverse(built_hierarchy["hierarchy"], model="TabPFN")
        for band_data in result.values():
            for cat_data in band_data.values():
                for gid, group in cat_data.get("groups", {}).items():
                    assert "TabPFN" in group["features"]

    def test_traverse_matched_only(self, built_hierarchy):
        """matched_only=True excludes unmatched features."""
        result = traverse(built_hierarchy["hierarchy"], matched_only=True)
        for band_data in result.values():
            for cat_data in band_data.values():
                assert not cat_data.get("unmatched", {}), "Should have no unmatched"

    def test_traverse_nonexistent_band(self, built_hierarchy):
        """Nonexistent band returns empty result."""
        result = traverse(built_hierarchy["hierarchy"], band="S99")
        assert result == {}


# ── Test JSON round-trip ─────────────────────────────────────────────────────


class TestSerialization:
    def test_json_roundtrip(self, built_hierarchy):
        """Hierarchy survives JSON serialization/deserialization."""
        json_str = json.dumps(built_hierarchy)
        reloaded = json.loads(json_str)
        assert reloaded["metadata"] == built_hierarchy["metadata"]
        assert reloaded["feature_index"] == built_hierarchy["feature_index"]

    def test_json_file_roundtrip(self, built_hierarchy):
        """Hierarchy survives write-to-file and read-back."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(built_hierarchy, f, indent=2)
            tmp_path = f.name

        with open(tmp_path) as f:
            reloaded = json.load(f)

        assert reloaded["metadata"]["n_features"] == built_hierarchy["metadata"]["n_features"]
        Path(tmp_path).unlink()

    def test_build_feature_index_from_hierarchy(self, built_hierarchy):
        """build_feature_index reconstructs the same index from hierarchy."""
        original_index = built_hierarchy["feature_index"]
        rebuilt_index = build_feature_index(built_hierarchy["hierarchy"])

        # Check all models present
        assert set(rebuilt_index.keys()) == set(original_index.keys())

        # Check all features present
        for model in original_index:
            assert set(rebuilt_index[model].keys()) == set(original_index[model].keys())
            for fid in original_index[model]:
                assert rebuilt_index[model][fid]["band"] == original_index[model][fid]["band"]
                assert rebuilt_index[model][fid]["group_id"] == original_index[model][fid]["group_id"]


# ── Test normalize_model_name ─────────────────────────────────────────────────


class TestNormalizeModelName:
    def test_display_to_key(self):
        assert _normalize_model_name("TabPFN") == "tabpfn"
        assert _normalize_model_name("Tabula-8B") == "tabula8b"
        assert _normalize_model_name("HyperFast") == "hyperfast"
        assert _normalize_model_name("CARTE") == "carte"

    def test_already_normalized(self):
        assert _normalize_model_name("tabpfn") == "tabpfn"


# ── Integration test with real data (skipped if files missing) ───────────────


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# In worktrees, output/ is gitignored. Resolve to the main repo for real data.
MAIN_REPO = Path("/Volumes/Samsung2TB/src/tabular_embeddings")
DATA_ROOT = MAIN_REPO if (MAIN_REPO / "output" / "cross_model_concept_labels.json").exists() else PROJECT_ROOT


@pytest.mark.skipif(
    not (DATA_ROOT / "output" / "cross_model_concept_labels.json").exists(),
    reason="Real data files not available",
)
class TestIntegration:
    @pytest.fixture(autouse=True)
    def load_real_data(self):
        import torch

        with open(DATA_ROOT / "output" / "cross_model_concept_labels.json") as f:
            self.labels = json.load(f)
        with open(DATA_ROOT / "config" / "pymfe_taxonomy.json") as f:
            self.taxonomy = json.load(f)

        sae_dir = DATA_ROOT / "output" / "sae_tabarena_sweep_round5"
        self.sae_configs = {}
        for ckpt_path in sae_dir.glob("*/sae_matryoshka_archetypal_validated.pt"):
            model_key = ckpt_path.parent.name
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            config = ckpt.get("config", {})
            if hasattr(config, "__dict__"):
                config = config.__dict__
            self.sae_configs[model_key] = {
                "input_dim": config.get("input_dim"),
                "hidden_dim": config.get("hidden_dim"),
                "matryoshka_dims": config.get("matryoshka_dims", []),
                "topk": config.get("topk"),
            }

        self.result = build_hierarchy(self.labels, self.taxonomy, self.sae_configs)

    def test_all_9376_features_present(self):
        """All 9376 alive features from cross-model labels are in hierarchy."""
        expected = self.labels["summary"]["total_alive_features"]
        actual = self.result["metadata"]["n_features"]
        assert actual == expected, f"Expected {expected}, got {actual}"

    def test_7_models_present(self):
        assert self.result["metadata"]["n_models"] == 7

    def test_725_groups_present(self):
        assert self.result["metadata"]["n_groups"] == 725

    def test_feature_index_covers_all_models(self):
        fi = self.result["feature_index"]
        expected_models = {"TabPFN", "CARTE", "TabICL", "TabDPT", "Mitra", "HyperFast", "Tabula-8B"}
        assert set(fi.keys()) == expected_models

    def test_no_duplicate_features(self):
        fi = self.result["feature_index"]
        seen = set()
        for model, feats in fi.items():
            for fid in feats:
                key = (model, fid)
                assert key not in seen, f"Duplicate: {key}"
                seen.add(key)

    def test_unique_shared_covers_all_groups(self):
        """For each model pair, unique + shared = all groups for A."""
        comp = self.result["model_comparison"]
        fi = self.result["feature_index"]

        model_gids = {}
        for model, feats in fi.items():
            model_gids[model] = {
                info["group_id"] for info in feats.values()
                if info["group_id"] is not None
            }

        for model_a in model_gids:
            for model_b in model_gids:
                if model_a == model_b:
                    continue

                unique = set(
                    comp["unique_to"].get(model_a, {})
                    .get(f"vs_{model_b}", {})
                    .get("groups", [])
                )
                shared_key = f"{model_a}__{model_b}"
                shared_key_rev = f"{model_b}__{model_a}"
                shared = set()
                if shared_key in comp["shared"]:
                    shared = set(comp["shared"][shared_key]["groups"])
                elif shared_key_rev in comp["shared"]:
                    shared = set(comp["shared"][shared_key_rev]["groups"])

                assert unique | shared == model_gids[model_a], (
                    f"{model_a} vs {model_b}: union={len(unique | shared)}, "
                    f"expected={len(model_gids[model_a])}"
                )
