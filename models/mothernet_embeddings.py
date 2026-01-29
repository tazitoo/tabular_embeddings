"""
MotherNet embedding extraction.

MotherNet (Microsoft TICL) is a hypernetwork that generates MLP weights from
context data in a single transformer forward pass. After fit(), the generated
MLP layers are stored as numpy (bias, weight) tuples. We replay the MLP forward
pass and capture the penultimate hidden activation — the last ReLU'd hidden
state before the output projection.

Architecture: context → transformer encoder → decoder → generated MLP weights → apply to query
Extraction point: penultimate hidden layer of generated MLP
"""

from typing import Dict, List, Optional

import numpy as np

from .base import EmbeddingExtractor, EmbeddingResult


class MotherNetEmbeddingExtractor(EmbeddingExtractor):
    """Extract embeddings from MotherNet's generated MLP network."""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)

    @property
    def model_name(self) -> str:
        return "MotherNet"

    @property
    def available_layers(self) -> List[str]:
        return [
            "penultimate_hidden",  # Last ReLU'd hidden state before output projection
            "final_probs",
        ]

    def load_model(self) -> None:
        """Load MotherNet classifier from Microsoft TICL package.

        The ticl package eagerly imports all models (gamformer, tabpfn, etc.) in
        its __init__.py, pulling in optional deps like 'interpret'. We bypass
        this by injecting a stub module before importing.
        """
        import importlib.util
        import sys
        import types

        # Find the ticl package on disk
        import os
        pkg_path = None
        for p in sys.path:
            candidate = os.path.join(p, "ticl", "prediction", "mothernet.py")
            if os.path.exists(candidate):
                pkg_path = os.path.join(p, "ticl")
                break

        if pkg_path is None:
            raise ImportError(
                "MotherNet (ticl) not found. Install with: "
                "pip install 'ticl @ git+https://github.com/microsoft/ticl.git'"
            )

        # Inject stub for ticl to prevent eager __init__ imports
        needs_cleanup = "ticl" not in sys.modules
        if needs_cleanup:
            stub = types.ModuleType("ticl")
            stub.__path__ = [pkg_path]
            sys.modules["ticl"] = stub

        try:
            spec = importlib.util.spec_from_file_location(
                "ticl.prediction.mothernet",
                f"{pkg_path}/prediction/mothernet.py",
                submodule_search_locations=[],
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            MotherNetClassifier = mod.MotherNetClassifier
        except Exception as e:
            raise ImportError(f"Failed to load MotherNet from ticl: {e}")
        finally:
            if needs_cleanup:
                sys.modules.pop("ticl", None)

        self._model = MotherNetClassifier(device=self.device)

    def extract_embeddings(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_query: np.ndarray,
        layers: Optional[List[str]] = None,
    ) -> EmbeddingResult:
        """
        Extract penultimate hidden activations from MotherNet's generated MLP.

        After fit(), self._model.parameters_ holds the generated MLP as
        [(b1, w1), (b2, w2), ..., (b_out, w_out)] numpy arrays. We replay the
        forward pass (scale → matmul+bias → ReLU) stopping before the last
        layer to capture the penultimate hidden state.

        Args:
            X_context: Training features (n_context, n_features)
            y_context: Training labels (n_context,)
            X_query: Query features (n_query, n_features)
            layers: Unused (kept for API compatibility)

        Returns:
            EmbeddingResult with penultimate hidden activations
        """
        if self._model is None:
            self.load_model()

        X_context = np.asarray(X_context, dtype=np.float32)
        y_context = np.asarray(y_context, dtype=np.int64)
        X_query = np.asarray(X_query, dtype=np.float32)
        X_context = np.nan_to_num(X_context, nan=0.0, posinf=0.0, neginf=0.0)
        X_query = np.nan_to_num(X_query, nan=0.0, posinf=0.0, neginf=0.0)

        n_query = len(X_query)

        # Fit generates MLP weights from context via transformer + decoder
        self._model.fit(X_context, y_context)

        layer_embeddings = {}

        # Get probabilities via normal path
        probs = self._model.predict_proba(X_query)
        layer_embeddings["final_probs"] = probs

        # Replay MLP forward pass, stopping before the last layer
        mlp_layers = self._model.parameters_  # [(b, w), ...]
        mean = self._model.mean_
        std = self._model.std_

        X_test = np.nan_to_num(np.array(X_query, dtype=float), 0)
        out = (X_test - mean) / std
        out = np.clip(out, a_min=-100, a_max=100)

        # Forward through all layers except the last (output projection)
        for i, (b, w) in enumerate(mlp_layers[:-1]):
            out = np.dot(out, w) + b
            out = np.maximum(out, 0)  # ReLU

        # out is now the penultimate hidden state: (n_query, hidden_dim)
        primary_embedding = out

        layer_embeddings["penultimate_hidden"] = primary_embedding

        return EmbeddingResult(
            embeddings=primary_embedding,
            model_name=self.model_name,
            extraction_point="penultimate_hidden",
            embedding_dim=primary_embedding.shape[1] if primary_embedding.ndim > 1 else 1,
            n_samples=n_query,
            layer_embeddings=layer_embeddings,
        )


if __name__ == "__main__":
    print("Testing MotherNet embedding extraction...")

    extractor = MotherNetEmbeddingExtractor(device="cpu")
    extractor.load_model()

    print(f"Model: {extractor.model_name}")
    print(f"Available layers: {extractor.available_layers}")

    X_ctx = np.random.randn(100, 20).astype(np.float32)
    y_ctx = (np.random.rand(100) > 0.7).astype(int)
    X_query = np.random.randn(10, 20).astype(np.float32)

    result = extractor.extract_embeddings(X_ctx, y_ctx, X_query)

    print(f"\nExtraction result:")
    print(f"  Embedding shape: {result.embeddings.shape}")
    print(f"  Embedding dim: {result.embedding_dim}")
    print(f"  Extraction point: {result.extraction_point}")
    for layer_name, emb in result.layer_embeddings.items():
        print(f"  {layer_name}: {emb.shape}")
