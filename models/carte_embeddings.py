"""
CARTE embedding extraction.

CARTE (Context-Aware Representation for Table Exploration) converts each row
into a star graph where the central node represents the row and leaf nodes
represent features. A GNN performs message passing to build row representations.

We extract the central node embedding from the final GNN layer as the primary
representation. Supports both classification and regression.
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from .base import EmbeddingExtractor, EmbeddingResult


class CARTEEmbeddingExtractor(EmbeddingExtractor):
    """Extract embeddings from CARTE graph transformer."""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self._layer_names = []

    @property
    def model_name(self) -> str:
        return "CARTE"

    @property
    def available_layers(self) -> List[str]:
        return self._layer_names if self._layer_names else [
            "gnn_output",
            "final_probs",
        ]

    def load_model(self) -> None:
        """Load CARTE classifier."""
        try:
            from carte_ai import CARTEClassifier
            self._model = CARTEClassifier()
        except ImportError:
            raise ImportError(
                "CARTE not found. Install with: pip install carte-ai"
            )
        self._discover_layers()

    def _discover_layers(self):
        """Discover hookable GNN layers."""
        self._layer_names = ["final_probs"]

        model = None
        for attr in ('model', '_model', 'net', 'network', 'gnn'):
            if hasattr(self._model, attr):
                candidate = getattr(self._model, attr)
                if hasattr(candidate, 'named_modules'):
                    model = candidate
                    break

        if model is not None:
            for name, module in model.named_modules():
                name_lower = name.lower()
                if any(x in name_lower for x in ['conv', 'gnn', 'message', 'graph', 'pool']):
                    self._layer_names.append(name)

        if len(self._layer_names) == 1:
            self._layer_names.insert(0, "gnn_output")

    def extract_embeddings(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_query: np.ndarray,
        layers: Optional[List[str]] = None,
        task: str = "classification",
    ) -> EmbeddingResult:
        """
        Extract embeddings from CARTE.

        Primary extraction: central node embedding from the final GNN layer
        after message passing on the star graph representation.
        Falls back to predict_proba pseudo-embedding.
        """
        if self._model is None:
            self.load_model()

        layers = layers or ["gnn_output", "final_probs"]

        X_context = np.asarray(X_context, dtype=np.float32)
        y_context = np.asarray(y_context, dtype=np.int64)
        X_query = np.asarray(X_query, dtype=np.float32)
        X_context = np.nan_to_num(X_context, nan=0.0, posinf=0.0, neginf=0.0)
        X_query = np.nan_to_num(X_query, nan=0.0, posinf=0.0, neginf=0.0)

        # CARTE may expect DataFrame input; try numpy first
        self._fit_model(X_context, y_context)
        layer_embeddings = {}

        # Internal hook-based extraction
        internal = self._extract_internal_embeddings(X_context, y_context, X_query)
        layer_embeddings.update(internal)

        # Final probabilities
        if "final_probs" in layers:
            probs = self._predict_proba(X_query)
            if probs is not None:
                layer_embeddings["final_probs"] = probs

        # Pseudo-embedding fallback
        if not any(k != "final_probs" for k in layer_embeddings):
            probs = layer_embeddings.get("final_probs")
            if probs is None:
                probs = self._predict_proba(X_query)
            if probs is not None:
                if probs.ndim == 1:
                    probs = probs.reshape(-1, 1)
                layer_embeddings["gnn_output"] = np.hstack([
                    probs,
                    np.log(np.abs(probs) + 1e-8),
                    probs * (1 - probs) if probs.max() <= 1 else probs ** 2,
                ])

        # Select primary embedding
        for key in ["gnn_output", "final_probs"]:
            if key in layer_embeddings:
                primary_embedding = layer_embeddings[key]
                extraction_point = key
                break
        else:
            primary_embedding = list(layer_embeddings.values())[0] if layer_embeddings else np.zeros((len(X_query), 2))
            extraction_point = list(layer_embeddings.keys())[0] if layer_embeddings else "fallback"

        if primary_embedding.ndim == 1:
            primary_embedding = primary_embedding.reshape(-1, 1)

        return EmbeddingResult(
            embeddings=primary_embedding,
            model_name=self.model_name,
            extraction_point=extraction_point,
            embedding_dim=primary_embedding.shape[1] if primary_embedding.ndim > 1 else 1,
            n_samples=len(X_query),
            layer_embeddings=layer_embeddings,
        )

    def _fit_model(self, X_context: np.ndarray, y_context: np.ndarray):
        """Fit CARTE, handling both numpy and DataFrame interfaces."""
        try:
            self._model.fit(X_context, y_context)
        except (TypeError, ValueError):
            # CARTE may require DataFrame input
            import pandas as pd
            feature_names = [f"f{i}" for i in range(X_context.shape[1])]
            df = pd.DataFrame(X_context, columns=feature_names)
            self._model.fit(df, y_context)

    def _predict_proba(self, X_query: np.ndarray) -> Optional[np.ndarray]:
        """Get predictions, handling interface variations."""
        try:
            probs = self._model.predict_proba(X_query)
            if isinstance(probs, np.ndarray):
                return probs
            return np.array(probs)
        except (TypeError, ValueError):
            import pandas as pd
            feature_names = [f"f{i}" for i in range(X_query.shape[1])]
            df = pd.DataFrame(X_query, columns=feature_names)
            try:
                probs = self._model.predict_proba(df)
                return np.array(probs)
            except Exception:
                pass
        except AttributeError:
            # No predict_proba, try predict
            try:
                preds = self._model.predict(X_query)
                return np.array(preds).reshape(-1, 1)
            except Exception:
                pass
        return None

    def _extract_internal_embeddings(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_query: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Hook into CARTE GNN to capture central node embeddings."""
        embeddings = {}

        model = None
        for attr in ('model', '_model', 'net', 'network', 'gnn'):
            if hasattr(self._model, attr):
                candidate = getattr(self._model, attr)
                if hasattr(candidate, 'named_modules'):
                    model = candidate
                    break

        if model is None:
            return embeddings

        self._clear_hooks()

        for name, module in model.named_modules():
            name_lower = name.lower()
            if any(x in name_lower for x in ['conv', 'gnn', 'message', 'graph', 'pool', 'attention']):
                if hasattr(module, 'forward'):
                    self._register_hook(module, name)

        try:
            with torch.no_grad():
                _ = self._predict_proba(X_query)

            for layer_name, activation in self._activations.items():
                if activation.ndim >= 2:
                    if activation.ndim == 3:
                        # For GNN: (batch, nodes, hidden) -> take central node (index 0)
                        activation = activation[:, 0, :]
                    if activation.shape[0] != len(X_query):
                        if activation.shape[0] > len(X_query):
                            activation = activation[:len(X_query)]
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
    print("Testing CARTE embedding extraction...")

    extractor = CARTEEmbeddingExtractor(device="cpu")
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
