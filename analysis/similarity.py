"""
Embedding similarity and geometry analysis.

Implements metrics from "Harnessing the Universal Geometry of Embeddings":
- Cosine similarity (pairwise and batch)
- CKA (Centered Kernel Alignment)
- Procrustes distance and alignment
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SimilarityResult:
    """Container for pairwise similarity results between two models."""

    model_a: str
    model_b: str

    # Core metrics
    mean_cosine_sim: float
    cka_score: float
    procrustes_distance: float

    # Per-sample similarities
    cosine_similarities: np.ndarray  # (n_samples,)

    # Alignment info
    rotation_matrix: Optional[np.ndarray] = None

    # Additional stats
    embedding_dim_a: int = 0
    embedding_dim_b: int = 0
    n_samples: int = 0

    def summary(self) -> str:
        """Return formatted summary string."""
        return (
            f"{self.model_a} vs {self.model_b}:\n"
            f"  CKA Score: {self.cka_score:.4f}\n"
            f"  Mean Cosine Sim: {self.mean_cosine_sim:.4f} (+/- {self.cosine_similarities.std():.4f})\n"
            f"  Procrustes Distance: {self.procrustes_distance:.4f}\n"
            f"  Dims: {self.embedding_dim_a} vs {self.embedding_dim_b}\n"
        )


def cosine_similarity_paired(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity for corresponding sample pairs.

    Args:
        emb_a: (n_samples, dim_a) embeddings from model A
        emb_b: (n_samples, dim_b) embeddings from model B

    Returns:
        (n_samples,) cosine similarities for each pair

    Note: If dimensions differ, we project to common space first.
    """
    assert emb_a.shape[0] == emb_b.shape[0], "Must have same number of samples"

    # Handle dimension mismatch via SVD projection to common space
    if emb_a.shape[1] != emb_b.shape[1]:
        n_samples = emb_a.shape[0]
        common_dim = min(emb_a.shape[1], emb_b.shape[1], n_samples, 128)
        emb_a = project_to_dim(emb_a, common_dim)
        emb_b = project_to_dim(emb_b, common_dim)

    # Normalize
    emb_a_norm = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-8)
    emb_b_norm = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-8)

    # Paired cosine sim
    return np.sum(emb_a_norm * emb_b_norm, axis=1)


def project_to_dim(X: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Project embeddings to target dimension using truncated SVD.

    Args:
        X: (n_samples, dim) embeddings
        target_dim: Target dimension

    Returns:
        (n_samples, target_dim) projected embeddings
    """
    if X.shape[1] == target_dim:
        return X

    if X.shape[1] < target_dim:
        # Pad with zeros
        padding = np.zeros((X.shape[0], target_dim - X.shape[1]))
        return np.hstack([X, padding])

    # Truncated SVD — target_dim can't exceed rank (min of n_samples, dim)
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    effective_dim = min(target_dim, U.shape[1])
    return U[:, :effective_dim] * S[:effective_dim]


def centered_kernel_alignment(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Compute CKA (Centered Kernel Alignment) between two representations.

    CKA measures structural similarity between representation spaces,
    independent of dimensionality and orthogonal transforms.

    Args:
        emb_a: (n_samples, dim_a) embeddings from model A
        emb_b: (n_samples, dim_b) embeddings from model B

    Returns:
        CKA score in [0, 1], where 1 means identical structure.
        Returns NaN if sample counts differ.
    """
    if emb_a.shape[0] != emb_b.shape[0]:
        return float("nan")

    # Center the embeddings
    emb_a = emb_a - emb_a.mean(axis=0)
    emb_b = emb_b - emb_b.mean(axis=0)

    # Compute Gram matrices (linear kernel)
    K_a = emb_a @ emb_a.T
    K_b = emb_b @ emb_b.T

    # HSIC (Hilbert-Schmidt Independence Criterion)
    def hsic(K1, K2):
        n = K1.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return np.trace(K1 @ H @ K2 @ H) / ((n - 1) ** 2)

    hsic_ab = hsic(K_a, K_b)
    hsic_aa = hsic(K_a, K_a)
    hsic_bb = hsic(K_b, K_b)

    return hsic_ab / (np.sqrt(hsic_aa * hsic_bb) + 1e-8)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Feature-space linear CKA between two representation matrices.

    Equivalent to centered_kernel_alignment() but avoids forming n×n Gram
    matrices. Complexity is O(n·p·q) instead of O(n²·(p+q)).

    Formula: ||Y^T X||_F^2 / (||X^T X||_F · ||Y^T Y||_F)
    computed on column-centered inputs.

    Args:
        X: (n_samples, p) activations from model A
        Y: (n_samples, q) activations from model B

    Returns:
        CKA score in [0, 1]. Returns NaN if sample counts differ.
    """
    if X.shape[0] != Y.shape[0]:
        return float("nan")

    # Center columns
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Cross-covariance and self-covariance Frobenius norms
    YtX = Y.T @ X  # (q, p)
    XtX = X.T @ X  # (p, p)
    YtY = Y.T @ Y  # (q, q)

    numerator = np.sum(YtX ** 2)
    denominator = np.sqrt(np.sum(XtX ** 2) * np.sum(YtY ** 2))

    return float(numerator / (denominator + 1e-8))


def procrustes_align(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Procrustes distance and optimal orthogonal alignment.

    Finds the orthogonal matrix R that minimizes ||emb_a @ R - emb_b||.

    Args:
        emb_a: (n_samples, dim) embeddings from model A
        emb_b: (n_samples, dim) embeddings from model B

    Returns:
        (distance, rotation_matrix, aligned_emb_a)
    """
    # Match dimensions: project both to common dim via PCA
    # When n_samples < dim, PCA can only return n_samples components
    n_samples = emb_a.shape[0]
    common_dim = min(emb_a.shape[1], emb_b.shape[1], n_samples)
    emb_a = project_to_dim(emb_a, common_dim)
    emb_b = project_to_dim(emb_b, common_dim)

    # Center
    emb_a_centered = emb_a - emb_a.mean(axis=0)
    emb_b_centered = emb_b - emb_b.mean(axis=0)

    # Normalize scale
    scale_a = np.linalg.norm(emb_a_centered, 'fro')
    scale_b = np.linalg.norm(emb_b_centered, 'fro')
    emb_a_normalized = emb_a_centered / (scale_a + 1e-8)
    emb_b_normalized = emb_b_centered / (scale_b + 1e-8)

    # SVD of cross-covariance
    M = emb_a_normalized.T @ emb_b_normalized
    U, _, Vt = np.linalg.svd(M)

    # Optimal rotation
    R = U @ Vt

    # Distance after alignment (normalized)
    aligned = emb_a_normalized @ R
    distance = np.linalg.norm(aligned - emb_b_normalized, 'fro')

    return distance, R, emb_a_centered @ R


def compute_pairwise_similarity(
    embeddings: Dict[str, np.ndarray],
    use_procrustes: bool = True,
) -> Dict[Tuple[str, str], SimilarityResult]:
    """
    Compute pairwise similarity between all model embeddings.

    Args:
        embeddings: Dict mapping model_name -> (n_samples, dim) embeddings
        use_procrustes: Whether to compute Procrustes alignment

    Returns:
        Dict mapping (model_a, model_b) -> SimilarityResult
    """
    results = {}
    model_names = list(embeddings.keys())

    for i, model_a in enumerate(model_names):
        for model_b in model_names[i + 1:]:
            emb_a = embeddings[model_a]
            emb_b = embeddings[model_b]

            # Skip if sample counts don't match
            if emb_a.shape[0] != emb_b.shape[0]:
                continue

            # CKA
            cka = centered_kernel_alignment(emb_a, emb_b)

            # Cosine similarity (after projection to common dim)
            cos_sims = cosine_similarity_paired(emb_a, emb_b)

            # Procrustes
            if use_procrustes:
                proc_dist, rotation, _ = procrustes_align(emb_a, emb_b)
            else:
                proc_dist = 0.0
                rotation = None

            results[(model_a, model_b)] = SimilarityResult(
                model_a=model_a,
                model_b=model_b,
                mean_cosine_sim=float(cos_sims.mean()),
                cka_score=float(cka),
                procrustes_distance=float(proc_dist),
                cosine_similarities=cos_sims,
                rotation_matrix=rotation,
                embedding_dim_a=emb_a.shape[1],
                embedding_dim_b=emb_b.shape[1],
                n_samples=len(emb_a),
            )

    return results


def compute_cka_matrix(
    embeddings: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute CKA matrix for all model pairs.

    Args:
        embeddings: Dict mapping model_name -> embeddings

    Returns:
        (cka_matrix, model_names) where cka_matrix[i,j] is CKA(model_i, model_j)
    """
    model_names = list(embeddings.keys())
    n = len(model_names)
    cka_matrix = np.eye(n)  # Diagonal is 1

    for i, model_a in enumerate(model_names):
        for j, model_b in enumerate(model_names):
            if i < j:
                cka = centered_kernel_alignment(
                    embeddings[model_a],
                    embeddings[model_b]
                )
                cka_matrix[i, j] = cka
                cka_matrix[j, i] = cka

    return cka_matrix, model_names


def compute_procrustes_alignment(
    embeddings: Dict[str, np.ndarray],
    reference_model: str,
) -> Dict[str, Tuple[np.ndarray, float]]:
    """
    Align all embeddings to a reference model's space.

    Args:
        embeddings: Dict mapping model_name -> embeddings
        reference_model: Name of reference model to align to

    Returns:
        Dict mapping model_name -> (aligned_embeddings, distance)
    """
    if reference_model not in embeddings:
        raise ValueError(f"Reference model '{reference_model}' not in embeddings")

    ref_emb = embeddings[reference_model]
    results = {reference_model: (ref_emb.copy(), 0.0)}

    for model_name, emb in embeddings.items():
        if model_name == reference_model:
            continue

        distance, _, aligned = procrustes_align(emb, ref_emb)
        results[model_name] = (aligned, distance)

    return results


def intrinsic_dimensionality(emb: np.ndarray, method: str = "pca_90") -> int:
    """
    Estimate intrinsic dimensionality of embeddings.

    Args:
        emb: (n_samples, dim) embeddings
        method: "pca_90" (dims for 90% variance), "mle" (maximum likelihood)

    Returns:
        Estimated intrinsic dimension
    """
    if method == "pca_90":
        # PCA-based: dimensions needed for 90% variance
        emb_centered = emb - emb.mean(axis=0)
        _, S, _ = np.linalg.svd(emb_centered, full_matrices=False)
        var_explained = np.cumsum(S ** 2) / np.sum(S ** 2)
        return int(np.searchsorted(var_explained, 0.90) + 1)

    elif method == "mle":
        # Maximum likelihood estimation (Levina & Bickel)
        from sklearn.neighbors import NearestNeighbors

        k = min(20, len(emb) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(emb)
        distances, _ = nbrs.kneighbors(emb)
        distances = distances[:, 1:]  # Exclude self

        # MLE estimate
        log_ratios = np.log(distances[:, -1:] / (distances[:, :-1] + 1e-10))
        return int(np.round(1 / np.mean(log_ratios)))

    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    # Test similarity metrics
    print("Testing similarity metrics...\n")

    np.random.seed(42)

    # Create test embeddings with known relationships
    n_samples = 200
    dim = 50

    # Model A: base embeddings
    emb_a = np.random.randn(n_samples, dim).astype(np.float32)

    # Model B: rotated version of A (should have high CKA)
    rotation = np.linalg.qr(np.random.randn(dim, dim))[0]
    emb_b = emb_a @ rotation

    # Model C: independent embeddings (should have low CKA)
    emb_c = np.random.randn(n_samples, dim).astype(np.float32)

    # Model D: different dimension
    emb_d = np.random.randn(n_samples, 30).astype(np.float32)

    embeddings = {
        "model_a": emb_a,
        "model_b_rotated": emb_b,
        "model_c_independent": emb_c,
        "model_d_diff_dim": emb_d,
    }

    # Compute pairwise similarities
    results = compute_pairwise_similarity(embeddings)

    print("Pairwise Similarities:")
    print("=" * 60)
    for (m1, m2), result in results.items():
        print(result.summary())

    # CKA matrix
    cka_matrix, names = compute_cka_matrix(embeddings)
    print("\nCKA Matrix:")
    print("=" * 60)
    print(f"{'':25s}", end="")
    for name in names:
        print(f"{name[:12]:>12s}", end=" ")
    print()
    for i, name in enumerate(names):
        print(f"{name:25s}", end="")
        for j in range(len(names)):
            print(f"{cka_matrix[i, j]:12.3f}", end=" ")
        print()

    # Intrinsic dimensionality
    print("\nIntrinsic Dimensionality (PCA 90%):")
    for name, emb in embeddings.items():
        intrinsic_dim = intrinsic_dimensionality(emb, method="pca_90")
        print(f"  {name}: {intrinsic_dim} (original: {emb.shape[1]})")
