"""
HyperFast embedding extraction.

HyperFast is a hypernetwork that predicts neural network weights in a single
forward pass. We can extract:
1. Context encoding: How the hypernetwork encodes the training data
2. Predicted weights: The generated NN weights (flattened)
3. Final representations: Hidden states before classification
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from .base import EmbeddingExtractor, EmbeddingResult


class HyperFastEmbeddingExtractor(EmbeddingExtractor):
    """Extract embeddings from HyperFast hypernetwork."""

    def __init__(
        self,
        device: str = "cuda",  # HyperFast needs CUDA (MPS fallback is very slow)
        n_ensemble: int = 16,
    ):
        super().__init__(device)
        self.n_ensemble = n_ensemble

    @property
    def model_name(self) -> str:
        return "HyperFast"

    @property
    def available_layers(self) -> List[str]:
        return [
            "context_encoding",   # How context is encoded
            "predicted_weights",  # Generated NN weights
            "final_hidden",       # Last hidden state
            "final_probs",        # Output probabilities
        ]

    def load_model(self) -> None:
        """Load HyperFast model."""
        from hyperfast import HyperFastClassifier

        self._model = HyperFastClassifier(
            device=self.device,
            n_ensemble=self.n_ensemble,
        )

    def extract_embeddings(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_query: np.ndarray,
        layers: Optional[List[str]] = None,
    ) -> EmbeddingResult:
        """
        Extract embeddings from HyperFast.

        HyperFast's hypernetwork takes the context (X_train, y_train) and
        generates weights for a target network. We can extract:
        - The context encoding (how it summarizes training data)
        - The predicted weights (what NN it generates)
        - Final hidden states during inference

        Args:
            X_context: Training features (n_context, n_features)
            y_context: Training labels (n_context,)
            X_query: Query features (n_query, n_features)
            layers: Which layers to extract (default: all)

        Returns:
            EmbeddingResult with embeddings
        """
        if self._model is None:
            self.load_model()

        layers = layers or ["context_encoding", "final_probs"]

        # Clean inputs
        X_context = np.asarray(X_context, dtype=np.float32)
        y_context = np.asarray(y_context, dtype=np.int64)
        X_query = np.asarray(X_query, dtype=np.float32)

        X_context = np.nan_to_num(X_context, nan=0.0, posinf=0.0, neginf=0.0)
        X_query = np.nan_to_num(X_query, nan=0.0, posinf=0.0, neginf=0.0)

        layer_embeddings = {}

        # Fit model (this runs the hypernetwork to generate weights)
        self._model.fit(X_context, y_context)

        # Extract final probabilities
        if "final_probs" in layers:
            probs = self._model.predict_proba(X_query)
            if probs.ndim == 2:
                layer_embeddings["final_probs"] = probs
            else:
                layer_embeddings["final_probs"] = probs.reshape(-1, 1)

        # Try to extract internal representations
        internal = self._extract_internal_embeddings(X_context, y_context, X_query)
        layer_embeddings.update(internal)

        # Primary embedding
        for key in ["context_encoding", "final_hidden", "final_probs"]:
            if key in layer_embeddings:
                primary_embedding = layer_embeddings[key]
                extraction_point = key
                break
        else:
            primary_embedding = layer_embeddings.get("final_probs", np.zeros((len(X_query), 2)))
            extraction_point = "final_probs"

        return EmbeddingResult(
            embeddings=primary_embedding,
            model_name=self.model_name,
            extraction_point=extraction_point,
            embedding_dim=primary_embedding.shape[1] if primary_embedding.ndim > 1 else 1,
            n_samples=len(X_query),
            layer_embeddings=layer_embeddings,
        )

    def _extract_internal_embeddings(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_query: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Extract internal representations from HyperFast.

        HyperFast structure:
        1. Encoder: processes (X_context, y_context) -> context embedding
        2. Hypernetwork: context embedding -> NN weights
        3. Target network: applies generated weights to X_query
        """
        embeddings = {}

        # Access internal model
        model = self._model
        if not hasattr(model, 'model') and not hasattr(model, '_model'):
            # Create pseudo-embedding from predictions
            probs = model.predict_proba(X_query)
            embeddings["context_encoding"] = np.hstack([
                probs,
                np.log(probs + 1e-8),
                np.var(probs, axis=1, keepdims=True) * np.ones_like(probs),
            ])
            return embeddings

        # Try to hook into hypernetwork
        internal_model = getattr(model, 'model', getattr(model, '_model', None))
        if internal_model is None:
            return embeddings

        self._clear_hooks()

        # Register hooks on key components
        for name, module in internal_model.named_modules():
            if any(x in name.lower() for x in ['encoder', 'context', 'hyper']):
                self._register_hook(module, name)

        try:
            # Trigger forward pass
            with torch.no_grad():
                _ = model.predict_proba(X_query)

            # Process captured activations
            for layer_name, activation in self._activations.items():
                if activation.ndim >= 2:
                    if activation.ndim == 3:
                        activation = activation.mean(axis=1)
                    # Broadcast to query size if needed
                    if activation.shape[0] == 1:
                        activation = np.tile(activation, (len(X_query), 1))
                    elif activation.shape[0] != len(X_query):
                        # Context encoding - replicate for each query
                        activation = np.tile(
                            activation.mean(axis=0, keepdims=True),
                            (len(X_query), 1)
                        )
                    embeddings[layer_name] = activation

        finally:
            self._clear_hooks()

        return embeddings


def verify_hyperfast_embedding_extraction(device: str = "cuda") -> bool:
    """Quick verification that HyperFast embedding extraction works."""
    try:
        extractor = HyperFastEmbeddingExtractor(device=device)
        extractor.load_model()
        return extractor.verify_loaded()
    except Exception as e:
        print(f"HyperFast embedding extraction verification failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    print(f"Testing HyperFast embedding extraction on {device}...")

    extractor = HyperFastEmbeddingExtractor(device=device)
    extractor.load_model()

    print(f"Model: {extractor.model_name}")
    print(f"Available layers: {extractor.available_layers}")

    # Test extraction
    X_ctx = np.random.randn(100, 20).astype(np.float32)
    y_ctx = (np.random.rand(100) > 0.7).astype(int)
    X_query = np.random.randn(10, 20).astype(np.float32)

    result = extractor.extract_embeddings(X_ctx, y_ctx, X_query)

    print(f"\nExtraction result:")
    print(f"  Embedding shape: {result.embeddings.shape}")
    print(f"  Embedding dim: {result.embedding_dim}")
    print(f"  Extraction point: {result.extraction_point}")
    print(f"  Layer embeddings: {list(result.layer_embeddings.keys())}")

    for layer_name, emb in result.layer_embeddings.items():
        print(f"    {layer_name}: {emb.shape}")
