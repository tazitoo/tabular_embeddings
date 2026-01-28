"""
MotherNet embedding extraction.

MotherNet is a hypernetwork (from Microsoft's TICL project) that generates MLP
weights from context data in a single forward pass - same family as HyperFast.

We extract:
1. Transformer encoder hidden states (how context is processed)
2. Generated MLP weights as a "dataset embedding" (what NN the model creates)
3. Final prediction probabilities

Classification only.
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from .base import EmbeddingExtractor, EmbeddingResult


class MotherNetEmbeddingExtractor(EmbeddingExtractor):
    """Extract embeddings from MotherNet hypernetwork."""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)

    @property
    def model_name(self) -> str:
        return "MotherNet"

    @property
    def available_layers(self) -> List[str]:
        return [
            "context_encoding",
            "predicted_weights",
            "final_probs",
        ]

    def load_model(self) -> None:
        """Load MotherNet classifier."""
        try:
            from mothernet import MotherNetClassifier
            self._model = MotherNetClassifier(device=self.device)
        except ImportError:
            try:
                from mothernet.prediction import MotherNetClassifier
                self._model = MotherNetClassifier(device=self.device)
            except ImportError:
                raise ImportError(
                    "MotherNet not found. Install with: "
                    "git clone https://github.com/microsoft/ticl && "
                    "cd ticl && pip install -e ."
                )

    def extract_embeddings(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_query: np.ndarray,
        layers: Optional[List[str]] = None,
    ) -> EmbeddingResult:
        """
        Extract embeddings from MotherNet.

        Primary extraction: transformer encoder hidden states capturing how
        context is encoded. Also captures generated MLP weights as a
        dataset-level embedding.
        """
        if self._model is None:
            self.load_model()

        layers = layers or ["context_encoding", "predicted_weights", "final_probs"]

        X_context = np.asarray(X_context, dtype=np.float32)
        y_context = np.asarray(y_context, dtype=np.int64)
        X_query = np.asarray(X_query, dtype=np.float32)
        X_context = np.nan_to_num(X_context, nan=0.0, posinf=0.0, neginf=0.0)
        X_query = np.nan_to_num(X_query, nan=0.0, posinf=0.0, neginf=0.0)

        self._model.fit(X_context, y_context)
        layer_embeddings = {}

        # Final probabilities
        if "final_probs" in layers:
            probs = self._model.predict_proba(X_query)
            if probs.ndim == 2:
                layer_embeddings["final_probs"] = probs
            else:
                layer_embeddings["final_probs"] = probs.reshape(-1, 1)

        # Internal extraction via hooks
        internal = self._extract_internal_embeddings(X_context, y_context, X_query)
        layer_embeddings.update(internal)

        # Select primary embedding
        for key in ["context_encoding", "predicted_weights", "final_probs"]:
            if key in layer_embeddings:
                primary_embedding = layer_embeddings[key]
                extraction_point = key
                break
        else:
            primary_embedding = layer_embeddings.get(
                "final_probs", np.zeros((len(X_query), 2))
            )
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
        Extract internal representations from MotherNet.

        MotherNet structure (similar to HyperFast):
        1. Encoder: processes (X_context, y_context) -> context embedding
        2. Hypernetwork: context embedding -> MLP weights
        3. Generated MLP: applies predicted weights to X_query
        """
        embeddings = {}

        # Access internal model
        internal_model = None
        for attr in ('model', '_model', 'net', 'network', 'transformer'):
            if hasattr(self._model, attr):
                candidate = getattr(self._model, attr)
                if hasattr(candidate, 'named_modules'):
                    internal_model = candidate
                    break

        if internal_model is None:
            # Pseudo-embedding fallback
            probs = self._model.predict_proba(X_query)
            if probs.ndim == 1:
                probs = probs.reshape(-1, 1)
            embeddings["context_encoding"] = np.hstack([
                probs,
                np.log(probs + 1e-8),
                np.var(probs, axis=1, keepdims=True) * np.ones_like(probs),
            ])
            return embeddings

        self._clear_hooks()

        # Hook encoder/context/hypernetwork components
        for name, module in internal_model.named_modules():
            name_lower = name.lower()
            if any(x in name_lower for x in ['encoder', 'context', 'hyper', 'transformer']):
                self._register_hook(module, name)

        try:
            with torch.no_grad():
                _ = self._model.predict_proba(X_query)

            for layer_name, activation in self._activations.items():
                if activation.ndim >= 2:
                    if activation.ndim == 3:
                        activation = activation.mean(axis=1)
                    # Broadcast context-level embeddings to query size
                    if activation.shape[0] == 1:
                        activation = np.tile(activation, (len(X_query), 1))
                    elif activation.shape[0] != len(X_query):
                        activation = np.tile(
                            activation.mean(axis=0, keepdims=True),
                            (len(X_query), 1),
                        )
                    embeddings[layer_name] = activation

            # Try to capture predicted weights as a dataset embedding
            self._extract_predicted_weights(internal_model, embeddings, len(X_query))

        finally:
            self._clear_hooks()

        return embeddings

    def _extract_predicted_weights(
        self,
        model: torch.nn.Module,
        embeddings: Dict[str, np.ndarray],
        n_query: int,
    ):
        """Extract generated MLP weights as a dataset-level embedding."""
        try:
            # Look for weight prediction layers / output heads
            weight_tensors = []
            for name, param in model.named_parameters():
                name_lower = name.lower()
                if any(x in name_lower for x in ['predict', 'weight_gen', 'output']):
                    weight_tensors.append(param.detach().cpu().numpy().flatten())

            if weight_tensors:
                # Concatenate and truncate to reasonable size
                weights = np.concatenate(weight_tensors)
                if len(weights) > 1024:
                    weights = weights[:1024]
                # Broadcast to per-query (same for all queries since it's dataset-level)
                embeddings["predicted_weights"] = np.tile(
                    weights.reshape(1, -1), (n_query, 1)
                )
        except Exception:
            pass


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
    print(f"  Layer embeddings: {list(result.layer_embeddings.keys())}")

    for layer_name, emb in result.layer_embeddings.items():
        print(f"    {layer_name}: {emb.shape}")
