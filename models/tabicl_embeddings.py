"""
TabICL embedding extraction.

TabICL uses a column-then-row transformer architecture:
1. Stage 1: Column attention processes features, then row attention captures sample relations
2. Stage 2: ICL prediction head

We extract embeddings from the row transformer output (between stage 1 and stage 2),
which captures learned sample-level representations after cross-feature attention.
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from .base import EmbeddingExtractor, EmbeddingResult


class TabICLEmbeddingExtractor(EmbeddingExtractor):
    """Extract embeddings from TabICL column-then-row transformer."""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self._layer_names = []

    @property
    def model_name(self) -> str:
        return "TabICL"

    @property
    def available_layers(self) -> List[str]:
        return self._layer_names if self._layer_names else [
            "row_transformer_output",
            "final_probs",
        ]

    def load_model(self) -> None:
        """Load TabICL classifier."""
        from tabicl import TabICLClassifier

        self._model = TabICLClassifier(device=self.device)
        self._discover_layers()

    def _discover_layers(self):
        """Inspect model to find hookable layers."""
        self._layer_names = ["final_probs"]

        if not hasattr(self._model, 'model'):
            return

        model = self._model.model if hasattr(self._model, 'model') else self._model
        for name, module in model.named_modules():
            name_lower = name.lower()
            if any(x in name_lower for x in ['row', 'encoder', 'transformer']):
                if 'layer' in name_lower or 'block' in name_lower:
                    self._layer_names.append(name)
            if 'column' in name_lower and ('layer' in name_lower or 'block' in name_lower):
                self._layer_names.append(name)

        if len(self._layer_names) == 1:
            self._layer_names.insert(0, "row_transformer_output")

    def extract_embeddings(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_query: np.ndarray,
        layers: Optional[List[str]] = None,
    ) -> EmbeddingResult:
        """
        Extract embeddings from TabICL.

        Primary extraction: row transformer output (between column/row processing
        and the ICL prediction head). Falls back to predict_proba pseudo-embedding.
        """
        if self._model is None:
            self.load_model()

        layers = layers or ["row_transformer_output", "final_probs"]

        X_context = np.asarray(X_context, dtype=np.float32)
        y_context = np.asarray(y_context, dtype=np.int64)
        X_query = np.asarray(X_query, dtype=np.float32)
        X_context = np.nan_to_num(X_context, nan=0.0, posinf=0.0, neginf=0.0)
        X_query = np.nan_to_num(X_query, nan=0.0, posinf=0.0, neginf=0.0)

        self._model.fit(X_context, y_context)
        layer_embeddings = {}

        # Try internal hook-based extraction
        internal = self._extract_internal_embeddings(X_context, y_context, X_query)
        layer_embeddings.update(internal)

        # Always get final probs
        if "final_probs" in layers:
            probs = self._model.predict_proba(X_query)
            layer_embeddings["final_probs"] = probs

        # Build pseudo-embedding if no internal layers captured
        if not any(k != "final_probs" for k in layer_embeddings):
            probs = layer_embeddings.get("final_probs", self._model.predict_proba(X_query))
            layer_embeddings["row_transformer_output"] = np.hstack([
                probs,
                np.log(probs + 1e-8),
                probs * (1 - probs),
            ])

        # Select primary embedding
        for key in ["row_transformer_output", "final_probs"]:
            if key in layer_embeddings:
                primary_embedding = layer_embeddings[key]
                extraction_point = key
                break
        else:
            primary_embedding = list(layer_embeddings.values())[0]
            extraction_point = list(layer_embeddings.keys())[0]

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
        """Hook into TabICL row transformer to capture intermediate representations."""
        embeddings = {}

        # Try to access the underlying model
        model = None
        for attr in ('model', '_model', 'net', 'network'):
            if hasattr(self._model, attr):
                candidate = getattr(self._model, attr)
                if hasattr(candidate, 'named_modules'):
                    model = candidate
                    break

        if model is None:
            return embeddings

        self._clear_hooks()

        # Hook row transformer / encoder layers
        for name, module in model.named_modules():
            name_lower = name.lower()
            if any(x in name_lower for x in ['row', 'encoder', 'transformer', 'attention']):
                if hasattr(module, 'forward'):
                    self._register_hook(module, name)

        try:
            with torch.no_grad():
                _ = self._model.predict_proba(X_query)

            for layer_name, activation in self._activations.items():
                if activation.ndim >= 2:
                    if activation.ndim == 3:
                        activation = activation.mean(axis=1)
                    if activation.shape[0] != len(X_query):
                        # May include context samples - take last n_query
                        if activation.shape[0] > len(X_query):
                            activation = activation[-len(X_query):]
                        else:
                            activation = np.tile(
                                activation.mean(axis=0, keepdims=True),
                                (len(X_query), 1),
                            )
                    embeddings[layer_name] = activation
        finally:
            self._clear_hooks()

        return embeddings


if __name__ == "__main__":
    print("Testing TabICL embedding extraction...")

    extractor = TabICLEmbeddingExtractor(device="cpu")
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
    print(f"  Layer embeddings: {list(result.layer_embeddings.keys())}")

    for layer_name, emb in result.layer_embeddings.items():
        print(f"    {layer_name}: {emb.shape}")
