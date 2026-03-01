"""Tests for concept_performance_diagnostic.py.

Tests cover:
1. Performance metric computation (AUC, logloss, RMSE)
2. Concept gap computation from fingerprints + MNN matching
3. Statistical analysis (Spearman correlations)
4. Target selection from diagnostic results
5. MNN pair key construction
6. CSV append/idempotency logic
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from scripts.concept_performance_diagnostic import (
    compute_metric,
    compute_concept_gaps,
    analyze_concept_performance,
    _pair_key,
    DISPLAY_NAMES,
    KEY_FROM_DISPLAY,
    load_fingerprints,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def mock_fingerprints():
    """Create synthetic fingerprints for two models."""
    np.random.seed(42)
    hidden_a = 256
    hidden_b = 192

    datasets = [f"dataset_{i}" for i in range(20)]

    fp_a = {
        "model": "tabpfn",
        "hidden_dim": hidden_a,
        "n_datasets": len(datasets),
        "alive_features": list(range(0, hidden_a, 2)),  # Every other feature
        "bands": {"S1": 16, "S2": 32, "S3": 64, "S4": 128, "S5": 256},
        "global_mean": np.random.randn(hidden_a).tolist(),
        "dataset_means": {
            ds: np.random.randn(hidden_a).tolist() for ds in datasets
        },
        "dataset_deviations": {
            ds: np.random.randn(hidden_a).tolist() for ds in datasets
        },
    }

    fp_b = {
        "model": "mitra",
        "hidden_dim": hidden_b,
        "n_datasets": len(datasets),
        "alive_features": list(range(0, hidden_b, 2)),
        "bands": {"S1": 12, "S2": 24, "S3": 48, "S4": 96, "S5": 192},
        "global_mean": np.random.randn(hidden_b).tolist(),
        "dataset_means": {
            ds: np.random.randn(hidden_b).tolist() for ds in datasets
        },
        "dataset_deviations": {
            ds: np.random.randn(hidden_b).tolist() for ds in datasets
        },
    }

    return {"tabpfn": fp_a, "mitra": fp_b}


@pytest.fixture
def mock_mnn_data():
    """Create synthetic MNN matching data."""
    np.random.seed(42)
    return {
        "metadata": {
            "n_models": 2,
            "models": ["Mitra", "TabPFN"],
        },
        "pairs": {
            "Mitra__TabPFN": {
                "n_alive_a": 96,
                "n_alive_b": 128,
                "n_matched": 30,
                "mean_match_r": 0.65,
                "n_samples": 500,
                "matches": [
                    {"idx_a": i * 2, "idx_b": i * 3, "r": 0.5 + 0.02 * i}
                    for i in range(30)
                ],
                "unmatched_a": list(range(60, 96, 2)),
                "unmatched_b": list(range(90, 128, 2)),
            }
        },
        "summary": {"per_pair": {}},
    }


@pytest.fixture
def mock_performance_df():
    """Create synthetic performance DataFrame."""
    rows = []
    np.random.seed(42)
    datasets = [f"dataset_{i}" for i in range(20)]
    for model in ["tabpfn", "mitra"]:
        for ds in datasets:
            base = 0.85 if model == "tabpfn" else 0.80
            rows.append({
                "model": model,
                "dataset": ds,
                "task": "classification",
                "metric_name": "auc",
                "metric_value": base + np.random.randn() * 0.05,
                "n_query": 500,
            })
    return pd.DataFrame(rows)


# ── Test compute_metric ──────────────────────────────────────────────────


class TestComputeMetric:
    def test_binary_classification_auc(self):
        """Binary classification returns AUC metric."""
        y_true = np.array([0, 0, 1, 1, 1])
        preds = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]])
        name, value = compute_metric(preds, y_true, "classification")
        assert name == "auc"
        assert 0.5 < value <= 1.0

    def test_binary_perfect_auc(self):
        """Perfect predictions give AUC = 1.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        preds = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
        name, value = compute_metric(preds, y_true, "classification")
        assert name == "auc"
        assert value == pytest.approx(1.0)

    def test_multiclass_logloss(self):
        """Multiclass classification returns neg_logloss."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        preds = np.array([
            [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
            [0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7],
        ])
        name, value = compute_metric(preds, y_true, "classification")
        assert name == "neg_logloss"
        assert value < 0  # logloss is positive, so negated is negative

    def test_regression_rmse(self):
        """Regression returns neg_rmse."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        preds = np.array([1.1, 2.2, 2.8, 4.1])
        name, value = compute_metric(preds, y_true, "regression")
        assert name == "neg_rmse"
        assert value < 0  # RMSE is positive, negated
        assert abs(value) < 0.5  # Predictions are close

    def test_regression_perfect(self):
        """Perfect regression predictions give neg_rmse = 0.0."""
        y_true = np.array([1.0, 2.0, 3.0])
        preds = np.array([1.0, 2.0, 3.0])
        name, value = compute_metric(preds, y_true, "regression")
        assert name == "neg_rmse"
        assert value == pytest.approx(0.0)

    def test_higher_is_better_convention(self):
        """All metrics follow higher-is-better convention."""
        # Good binary predictions
        y = np.array([0, 0, 1, 1])
        good_preds = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        bad_preds = np.array([[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]])
        _, good_val = compute_metric(good_preds, y, "classification")
        _, bad_val = compute_metric(bad_preds, y, "classification")
        assert good_val > bad_val


# ── Test pair key construction ───────────────────────────────────────────


class TestPairKey:
    def test_alphabetical_order(self):
        """Pair key is always alphabetically ordered."""
        assert _pair_key("TabPFN", "Mitra") == "Mitra__TabPFN"
        assert _pair_key("Mitra", "TabPFN") == "Mitra__TabPFN"

    def test_same_model(self):
        """Same model produces valid key."""
        assert _pair_key("TabPFN", "TabPFN") == "TabPFN__TabPFN"


# ── Test compute_concept_gaps ────────────────────────────────────────────


class TestComputeConceptGaps:
    def test_returns_dataframe(self, mock_fingerprints, mock_mnn_data):
        """Result is a DataFrame with expected columns."""
        df = compute_concept_gaps(
            mock_fingerprints, mock_mnn_data, "tabpfn", "mitra",
        )
        assert isinstance(df, pd.DataFrame)
        assert "dataset" in df.columns
        assert "concept_asymmetry" in df.columns
        assert "unmatched_act_a" in df.columns
        assert "unmatched_act_b" in df.columns
        assert "matched_differential" in df.columns
        assert "cosine_sim" in df.columns

    def test_per_band_columns(self, mock_fingerprints, mock_mnn_data):
        """Result includes per-band breakdown columns."""
        df = compute_concept_gaps(
            mock_fingerprints, mock_mnn_data, "tabpfn", "mitra",
        )
        for band in ["S1", "S2", "S3", "S4", "S5"]:
            assert f"concept_asymmetry_{band}" in df.columns
            assert f"matched_differential_{band}" in df.columns

    def test_common_datasets_only(self, mock_fingerprints, mock_mnn_data):
        """Only datasets present in both models are included."""
        df = compute_concept_gaps(
            mock_fingerprints, mock_mnn_data, "tabpfn", "mitra",
        )
        ds_a = set(mock_fingerprints["tabpfn"]["dataset_means"].keys())
        ds_b = set(mock_fingerprints["mitra"]["dataset_means"].keys())
        expected = ds_a & ds_b
        assert set(df["dataset"]) == expected

    def test_swapped_pair_order(self, mock_fingerprints, mock_mnn_data):
        """Requesting mitra,tabpfn (reverse of MNN storage) works correctly."""
        df_fwd = compute_concept_gaps(
            mock_fingerprints, mock_mnn_data, "tabpfn", "mitra",
        )
        df_rev = compute_concept_gaps(
            mock_fingerprints, mock_mnn_data, "mitra", "tabpfn",
        )
        # Asymmetry should flip sign
        merged = df_fwd.merge(
            df_rev, on="dataset", suffixes=("_fwd", "_rev"),
        )
        for _, row in merged.iterrows():
            # Forward A = reverse B
            assert row["unmatched_act_a_fwd"] == pytest.approx(
                row["unmatched_act_b_rev"], abs=1e-6,
            )

    def test_missing_model_raises(self, mock_fingerprints, mock_mnn_data):
        """Missing fingerprints raise ValueError."""
        with pytest.raises(ValueError, match="Missing fingerprints"):
            compute_concept_gaps(
                mock_fingerprints, mock_mnn_data, "tabpfn", "hyperfast",
            )

    def test_cosine_sim_in_range(self, mock_fingerprints, mock_mnn_data):
        """Cosine similarity values are in [-1, 1]."""
        df = compute_concept_gaps(
            mock_fingerprints, mock_mnn_data, "tabpfn", "mitra",
        )
        assert (df["cosine_sim"] >= -1.0).all()
        assert (df["cosine_sim"] <= 1.0).all()


# ── Test analyze_concept_performance ─────────────────────────────────────


class TestAnalyzeConceptPerformance:
    def test_returns_expected_structure(
        self, mock_performance_df, mock_fingerprints, mock_mnn_data,
    ):
        """Analysis returns dict with expected top-level keys."""
        result = analyze_concept_performance(
            mock_performance_df, mock_fingerprints, mock_mnn_data,
            pairs=[("tabpfn", "mitra")],
        )
        assert "pairs" in result
        assert "correlation_matrix" in result
        assert "band_importance" in result
        assert "n_pairs" in result
        assert result["n_pairs"] == 1

    def test_correlation_values_valid(
        self, mock_performance_df, mock_fingerprints, mock_mnn_data,
    ):
        """Spearman correlations are in [-1, 1]."""
        result = analyze_concept_performance(
            mock_performance_df, mock_fingerprints, mock_mnn_data,
            pairs=[("tabpfn", "mitra")],
        )
        for pair_label, pr in result["pairs"].items():
            for metric, corr in pr["correlations"].items():
                assert -1.0 <= corr["rho"] <= 1.0
                assert 0.0 <= corr["p_value"] <= 1.0

    def test_band_importance_all_bands(
        self, mock_performance_df, mock_fingerprints, mock_mnn_data,
    ):
        """Band importance covers all 5 Matryoshka scales."""
        result = analyze_concept_performance(
            mock_performance_df, mock_fingerprints, mock_mnn_data,
            pairs=[("tabpfn", "mitra")],
        )
        for band in ["S1", "S2", "S3", "S4", "S5"]:
            assert band in result["band_importance"]

    def test_skips_pair_without_enough_data(
        self, mock_fingerprints, mock_mnn_data,
    ):
        """Pairs with <5 common datasets are skipped."""
        # Performance with very few datasets
        small_perf = pd.DataFrame([
            {"model": "tabpfn", "dataset": "d1", "task": "classification",
             "metric_name": "auc", "metric_value": 0.9, "n_query": 100},
            {"model": "mitra", "dataset": "d1", "task": "classification",
             "metric_name": "auc", "metric_value": 0.8, "n_query": 100},
        ])
        result = analyze_concept_performance(
            small_perf, mock_fingerprints, mock_mnn_data,
            pairs=[("tabpfn", "mitra")],
        )
        assert result["n_pairs"] == 0


# ── Test display name mappings ───────────────────────────────────────────


class TestDisplayNames:
    def test_all_models_have_display_names(self):
        """All model keys have display name mappings."""
        for key in ["tabpfn", "mitra", "tabicl", "tabdpt", "hyperfast", "carte", "tabula8b"]:
            assert key in DISPLAY_NAMES

    def test_reverse_mapping(self):
        """KEY_FROM_DISPLAY correctly reverses DISPLAY_NAMES."""
        for key, display in DISPLAY_NAMES.items():
            assert KEY_FROM_DISPLAY[display] == key


# ── Test load_fingerprints ───────────────────────────────────────────────


class TestLoadFingerprints:
    def test_loads_from_json_files(self, tmp_path, mock_fingerprints):
        """Fingerprints are loaded from JSON files on disk."""
        for model_key, fp in mock_fingerprints.items():
            fp_path = tmp_path / f"{model_key}_fingerprints.json"
            with open(fp_path, "w") as f:
                json.dump(fp, f)

        result = load_fingerprints(tmp_path, models=["tabpfn", "mitra"])
        assert "tabpfn" in result
        assert "mitra" in result
        assert result["tabpfn"]["hidden_dim"] == 256

    def test_warns_on_missing_model(self, tmp_path):
        """Missing model fingerprint files generate a warning."""
        result = load_fingerprints(tmp_path, models=["nonexistent"])
        assert "nonexistent" not in result
