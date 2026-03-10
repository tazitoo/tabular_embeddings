"""Tests for concept description embedding."""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def embedding_model_available():
    """Check if sentence-transformers and nomic model are available."""
    try:
        from sentence_transformers import SentenceTransformer
        return True
    except ImportError:
        pytest.skip("sentence-transformers not installed")


def test_embed_descriptions_returns_correct_shape(embedding_model_available):
    """Embedding N descriptions produces (N, dim) array."""
    from scripts.embed_concept_descriptions import embed_descriptions

    descriptions = [
        "Rows with many zero-valued features.",
        "Extreme outlier rows with high z-scores.",
        "Dense numeric rows with low sparsity.",
    ]
    embeddings = embed_descriptions(descriptions, dim=768)

    assert embeddings.shape == (3, 768)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=0.01)


def test_embed_descriptions_matryoshka_truncation(embedding_model_available):
    """Matryoshka truncation produces correct dimensionality."""
    from scripts.embed_concept_descriptions import embed_descriptions

    descriptions = ["test sentence one", "test sentence two"]
    emb_256 = embed_descriptions(descriptions, dim=256)
    emb_768 = embed_descriptions(descriptions, dim=768)

    assert emb_256.shape == (2, 256)
    assert emb_768.shape == (2, 768)
    # Truncated should match prefix of full before re-normalization
    # (Matryoshka property: prefix is meaningful)
    cos_sim = np.dot(emb_256[0], emb_768[0, :256]) / (
        np.linalg.norm(emb_256[0]) * np.linalg.norm(emb_768[0, :256])
    )
    assert cos_sim > 0.99


def test_within_group_coherence():
    """Within-group cosine sim is computed correctly."""
    from scripts.embed_concept_descriptions import compute_within_group_coherence

    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.1, 0.9],
    ])
    group_ids = [0, 0, 1, 1]

    result = compute_within_group_coherence(embeddings, group_ids)
    assert result["mean"] > 0.8
    assert result["n_groups"] == 2


def test_matched_pair_agreement():
    """Matched pairs have higher sim than random baseline."""
    from scripts.embed_concept_descriptions import compute_matched_pair_agreement

    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.9, 0.1],
        [0.1, 0.9],
    ])
    matched_pairs = [(0, 2), (1, 3)]

    result = compute_matched_pair_agreement(embeddings, matched_pairs)
    assert result["mean"] > result["random_baseline"]
    assert result["n_pairs"] == 2


def test_interpolation_coherence():
    """Interpolation coherence computes similarity to landmarks."""
    from scripts.embed_concept_descriptions import compute_interpolation_coherence

    embeddings = np.array([
        [1.0, 0.0],  # A:1
        [0.9, 0.1],  # A:2 (landmark)
        [0.8, 0.2],  # A:3 (unmatched, near landmark)
    ])
    feature_ids = ["A:1", "A:2", "A:3"]
    unmatched_landmarks = {"A:3": [("A:2", 0.5)]}

    result = compute_interpolation_coherence(embeddings, feature_ids, unmatched_landmarks)
    assert result["mean"] > 0.9
    assert result["n_features"] == 1
