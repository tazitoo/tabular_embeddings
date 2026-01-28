"""Embedding geometry analysis tools."""

from .similarity import (
    compute_pairwise_similarity,
    compute_cka_matrix,
    compute_procrustes_alignment,
    SimilarityResult,
)

from .sparse_autoencoder import (
    SAEConfig,
    SAEResult,
    SparseAutoencoder,
    train_sae,
    compare_dictionaries,
    measure_dictionary_richness,
    analyze_feature_geometry,
)

__all__ = [
    # Similarity metrics
    "compute_pairwise_similarity",
    "compute_cka_matrix",
    "compute_procrustes_alignment",
    "SimilarityResult",
    # Sparse Autoencoder
    "SAEConfig",
    "SAEResult",
    "SparseAutoencoder",
    "train_sae",
    "compare_dictionaries",
    "measure_dictionary_richness",
    "analyze_feature_geometry",
]
