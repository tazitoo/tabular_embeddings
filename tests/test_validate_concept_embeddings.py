"""Tests for concept embedding validation."""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def embedding_model_available():
    """Check if sentence-transformers is available."""
    try:
        from sentence_transformers import SentenceTransformer
        return True
    except ImportError:
        pytest.skip("sentence-transformers not installed")


def test_matryoshka_consistency_high_for_stable_embeddings(embedding_model_available):
    """Matryoshka dimensions should agree on rankings for stable embeddings."""
    from scripts.validate_concept_embeddings import matryoshka_consistency

    descriptions = [
        "Rows with many zero-valued features.",
        "Sparse numeric rows with low entropy.",
        "Dense rows with all features populated.",
        "Extreme outliers with very high z-scores.",
        "Rows with strongly correlated feature pairs.",
    ]

    result = matryoshka_consistency(descriptions)
    assert result["768v256"] > 0.85


def test_bootstrap_stability_high_for_deterministic_model(embedding_model_available):
    """Bootstrap stability should be high for deterministic embeddings."""
    from scripts.validate_concept_embeddings import bootstrap_stability

    descriptions = [
        "Sparse rows with zeros.",
        "Dense rows with values.",
        "Outlier rows with extremes.",
        "Correlated feature pairs.",
        "Skewed distributions.",
    ]

    result = bootstrap_stability(descriptions, n_bootstrap=3)
    assert result > 0.9


def test_validation_report_structure():
    """Validation report has required keys."""
    from scripts.validate_concept_embeddings import build_validation_report

    nomic_checks = {
        "matryoshka_spearman": {"768v256": 0.97},
        "bootstrap_stability": 0.95,
        "passed": True,
    }
    report = build_validation_report(nomic_checks)

    assert "nomic_self_checks" in report
    assert "api_validation" in report
    assert report["api_validation"]["ran"] is False


def test_validation_report_with_api():
    """Validation report includes API results when provided."""
    from scripts.validate_concept_embeddings import build_validation_report

    nomic_checks = {"passed": False}
    api_result = {"ran": True, "provider": "voyage", "spearman_rho": 0.91}

    report = build_validation_report(nomic_checks, api_result)
    assert report["api_validation"]["ran"] is True
    assert report["api_validation"]["provider"] == "voyage"
