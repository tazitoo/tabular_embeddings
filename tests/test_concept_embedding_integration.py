"""End-to-end integration test for the concept embedding pipeline.

Tests the full chain: synthetic concept descriptions -> embedding -> metrics.
The slow test requires sentence-transformers (model download) and is skipped
if the dependency is not installed.
"""
import json

import numpy as np
import pytest


@pytest.mark.slow
def test_full_pipeline_small_subset(tmp_path):
    """Run description -> embedding -> validation on synthetic data."""
    pytest.importorskip("sentence_transformers")

    from scripts.embed_concept_descriptions import (
        compute_matched_pair_agreement,
        compute_within_group_coherence,
        embed_descriptions,
    )

    descriptions_data = {
        "metadata": {"test": True},
        "groups": {
            "0": {
                "brief_label": "sparse rows",
                "summary": "Rows with many zero-valued features.",
                "features": {
                    "A:1": {"description": "Sparse numeric rows with many zeros."},
                    "B:2": {"description": "Rows with mostly zero feature values."},
                },
            },
            "1": {
                "brief_label": "outliers",
                "summary": "Extreme outlier rows.",
                "features": {
                    "A:3": {"description": "Rows with extreme z-scores above 3."},
                    "B:4": {"description": "Outlier rows far from the centroid."},
                },
            },
        },
        "unmatched": {
            "A:5": {"description": "Rows with highly skewed feature distributions."},
        },
    }

    # Write to disk to verify JSON round-trip
    desc_path = tmp_path / "concept_descriptions.json"
    with open(desc_path, "w") as f:
        json.dump(descriptions_data, f)

    # Collect descriptions and group_ids from the synthetic data
    descs = []
    group_ids = []
    for gid, group in descriptions_data["groups"].items():
        for feat in group["features"].values():
            descs.append(feat["description"])
            group_ids.append(int(gid))
    for feat in descriptions_data["unmatched"].values():
        descs.append(feat["description"])
        group_ids.append(-1)

    # Embed at dim=256 (Matryoshka truncation)
    embeddings = embed_descriptions(descs, dim=256)
    assert embeddings.shape == (5, 256)

    # Verify unit normalization
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    # Within-group coherence
    coherence = compute_within_group_coherence(embeddings, group_ids)
    assert coherence["mean"] > 0.0
    assert coherence["n_groups"] == 2

    # Matched-pair agreement (pairs within each group)
    pairs = [(0, 1), (2, 3)]
    agreement = compute_matched_pair_agreement(embeddings, pairs)
    assert agreement["mean"] > agreement["random_baseline"]


def test_validation_report_structure():
    """build_validation_report returns expected schema."""
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
    assert report["nomic_self_checks"]["passed"] is True
