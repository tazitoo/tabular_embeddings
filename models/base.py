"""Base class for embedding extraction from tabular foundation models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class EmbeddingResult:
    """Container for extracted embeddings with metadata."""

    # Core embeddings - shape (n_samples, embedding_dim) or dict of layer -> embeddings
    embeddings: np.ndarray | Dict[str, np.ndarray]

    # Metadata
    model_name: str
    extraction_point: str  # e.g., "final", "layer_3", "context_encoding"
    embedding_dim: int
    n_samples: int

    # Optional: multiple extraction points
    layer_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)

    # Optional: attention weights if available
    attention_weights: Optional[np.ndarray] = None

    def __post_init__(self):
        if isinstance(self.embeddings, np.ndarray):
            self.n_samples = self.embeddings.shape[0]
            self.embedding_dim = self.embeddings.shape[1] if self.embeddings.ndim > 1 else 1


class EmbeddingExtractor(ABC):
    """
    Abstract base class for extracting embeddings from tabular foundation models.

    Each model has different internal architectures, so extraction methods vary:
    - TabPFN: Transformer hidden states from encoder
    - HyperFast: Hypernetwork context encoding + predicted weights
    - TabICL: In-context learned representations
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        self._hooks = []
        self._activations = {}

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model identifier."""
        pass

    @property
    @abstractmethod
    def available_layers(self) -> List[str]:
        """Return list of layer names where embeddings can be extracted."""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and prepare for embedding extraction."""
        pass

    @abstractmethod
    def extract_embeddings(
        self,
        X_context: Union[np.ndarray, pd.DataFrame],
        y_context: np.ndarray,
        X_query: Union[np.ndarray, pd.DataFrame],
        layers: Optional[List[str]] = None,
        task: str = "classification",
        cat_feature_indices: Optional[List[int]] = None,
    ) -> EmbeddingResult:
        """
        Extract embeddings for query samples given context.

        Args:
            X_context: Training/context features (n_context, n_features)
            y_context: Training/context labels (n_context,)
            X_query: Query features to get embeddings for (n_query, n_features)
            layers: Which layers to extract from (default: final layer only)
            task: "classification" or "regression"
            cat_feature_indices: Indices of categorical columns in X.
                When X is a DataFrame, categoricals are auto-detected from
                object/category dtype. When X is a numpy array, this list
                tells the model which columns are categorical.

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        pass

    @staticmethod
    def _detect_cat_features(
        X: Union[np.ndarray, pd.DataFrame],
        cat_feature_indices: Optional[List[int]] = None,
    ) -> List[int]:
        """Detect categorical feature indices from DataFrame dtypes or explicit list."""
        if cat_feature_indices is not None:
            return cat_feature_indices
        if isinstance(X, pd.DataFrame):
            cat_cols = X.select_dtypes(include=["object", "category"]).columns
            return [X.columns.get_loc(c) for c in cat_cols]
        return []

    @staticmethod
    def _to_numpy_with_label_encoding(
        X: Union[np.ndarray, pd.DataFrame],
        cat_feature_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """Convert DataFrame to numpy with label-encoded categoricals.

        Returns (X_array, cat_indices) where categorical columns are
        integer-encoded but their indices are tracked.
        """
        if isinstance(X, np.ndarray):
            return X.astype(np.float32), cat_feature_indices or []

        X_df = X.copy()
        cat_cols = X_df.select_dtypes(include=["object", "category"]).columns
        cat_indices = [X_df.columns.get_loc(c) for c in cat_cols]

        for col in cat_cols:
            X_df[col] = X_df[col].astype("category").cat.codes.astype(np.float32)
            # Replace -1 (NaN in cat.codes) with NaN
            X_df[col] = X_df[col].replace(-1, np.nan)

        X_array = X_df.values.astype(np.float32)
        return X_array, cat_indices

    def _register_hook(self, module, name: str):
        """Register a forward hook to capture activations."""
        def hook(module, input, output):
            tensor = output[0] if isinstance(output, tuple) else output
            # Convert BFloat16/Float16 to Float32 for numpy compatibility
            tensor = tensor.detach().float().cpu().numpy()
            self._activations[name] = tensor

        handle = module.register_forward_hook(hook)
        self._hooks.append(handle)
        return handle

    def _clear_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._activations = {}

    def verify_loaded(self) -> bool:
        """Verify model is properly loaded and can extract embeddings."""
        try:
            # Create minimal test data
            X_ctx = np.random.randn(50, 10).astype(np.float32)
            y_ctx = (np.random.rand(50) > 0.5).astype(int)
            X_query = np.random.randn(5, 10).astype(np.float32)

            result = self.extract_embeddings(X_ctx, y_ctx, X_query)

            return (
                result.embeddings is not None and
                len(result.embeddings) == len(X_query) and
                result.embedding_dim > 0
            )
        except Exception as e:
            print(f"Verification failed: {e}")
            return False


def cosine_similarity_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity between two embedding matrices.

    Args:
        emb_a: (n_samples, dim_a) embeddings from model A
        emb_b: (n_samples, dim_b) embeddings from model B

    Returns:
        (n_samples, n_samples) similarity matrix if shapes differ,
        (n_samples,) diagonal similarities if same samples
    """
    # Normalize
    emb_a_norm = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-8)
    emb_b_norm = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-8)

    # If same number of samples, return diagonal (corresponding pairs)
    if emb_a.shape[0] == emb_b.shape[0]:
        return np.sum(emb_a_norm * emb_b_norm, axis=1)

    # Otherwise return full matrix
    return emb_a_norm @ emb_b_norm.T


def centered_kernel_alignment(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Compute CKA (Centered Kernel Alignment) between two representations.

    CKA measures structural similarity between representation spaces,
    independent of dimensionality and orthogonal transforms.

    Args:
        emb_a: (n_samples, dim_a) embeddings from model A
        emb_b: (n_samples, dim_b) embeddings from model B

    Returns:
        CKA score in [0, 1], where 1 means identical structure
    """
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


def procrustes_distance(emb_a: np.ndarray, emb_b: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute Procrustes distance after optimal orthogonal alignment.

    Finds the orthogonal matrix R that minimizes ||emb_a @ R - emb_b||.

    Args:
        emb_a: (n_samples, dim) embeddings from model A
        emb_b: (n_samples, dim) embeddings from model B (must have same dim)

    Returns:
        (distance, rotation_matrix) where distance is Frobenius norm after alignment
    """
    assert emb_a.shape == emb_b.shape, "Embeddings must have same shape for Procrustes"

    # Center
    emb_a = emb_a - emb_a.mean(axis=0)
    emb_b = emb_b - emb_b.mean(axis=0)

    # SVD of cross-covariance
    M = emb_a.T @ emb_b
    U, _, Vt = np.linalg.svd(M)

    # Optimal rotation
    R = U @ Vt

    # Distance after alignment
    aligned = emb_a @ R
    distance = np.linalg.norm(aligned - emb_b, 'fro')

    return distance, R
