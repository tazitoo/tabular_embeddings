"""
TabDPT embedding extraction.

TabDPT (Tabular Discriminative Pre-Trained Transformer) uses a transformer
with retrieval augmentation. It processes tabular data through transformer layers
and supports both classification and regression.

We extract hidden states from transformer layers before the prediction head.
Note: flash attention must be disabled (compile=False) for hook-based extraction.
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from .base import EmbeddingExtractor, EmbeddingResult


class TabDPTEmbeddingExtractor(EmbeddingExtractor):
    """Extract embeddings from TabDPT transformer + retrieval model."""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self._layer_names = []

    @property
    def model_name(self) -> str:
        return "TabDPT"

    @property
    def available_layers(self) -> List[str]:
        return self._layer_names if self._layer_names else [
            "transformer_hidden",
            "final_probs",
        ]

    def load_model(self, task: str = "classification") -> None:
        """Load TabDPT model from HuggingFace (Layer6/TabDPT).

        Uses compile=False to allow hook-based embedding extraction.
        """
        try:
            if task == "regression":
                from tabdpt import TabDPTRegressor
                self._model = TabDPTRegressor(device=self.device, compile=False)
            else:
                from tabdpt import TabDPTClassifier
                self._model = TabDPTClassifier(device=self.device, compile=False)
        except ImportError:
            try:
                if task == "regression":
                    from tabdpt.model import TabDPTRegressor
                    self._model = TabDPTRegressor(device=self.device, compile=False)
                else:
                    from tabdpt.model import TabDPTClassifier
                    self._model = TabDPTClassifier(device=self.device, compile=False)
            except ImportError:
                raise ImportError(
                    "TabDPT not found. Install with: "
                    "git clone https://github.com/layer6ai-labs/TabDPT && "
                    "cd TabDPT && pip install -e ."
                )
        self._current_task = task
        self._discover_layers()

    def _discover_layers(self):
        """Discover hookable transformer layers."""
        self._layer_names = ["final_probs"]

        model = None
        for attr in ('model', '_model', 'net', 'network', 'transformer'):
            if hasattr(self._model, attr):
                candidate = getattr(self._model, attr)
                if hasattr(candidate, 'named_modules'):
                    model = candidate
                    break

        if model is None:
            self._layer_names.insert(0, "transformer_hidden")
            return

        for name, module in model.named_modules():
            name_lower = name.lower()
            if any(x in name_lower for x in ['encoder', 'transformer', 'block', 'layer']):
                if hasattr(module, 'forward') and ('layer' in name_lower or 'block' in name_lower):
                    self._layer_names.append(name)

        if len(self._layer_names) == 1:
            self._layer_names.insert(0, "transformer_hidden")

    def extract_embeddings(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_query: np.ndarray,
        layers: Optional[List[str]] = None,
        task: str = "classification",
    ) -> EmbeddingResult:
        """
        Extract embeddings from TabDPT.

        Primary extraction: transformer hidden states before prediction head.
        Falls back to predict_proba pseudo-embedding.
        """
        # Load or reload model if task type changed
        if self._model is None or getattr(self, "_current_task", None) != task:
            self.load_model(task=task)

        layers = layers or ["transformer_hidden", "final_probs"]

        X_context = np.asarray(X_context, dtype=np.float32)
        y_dtype = np.float32 if task == "regression" else np.int64
        y_context = np.asarray(y_context, dtype=y_dtype)
        X_query = np.asarray(X_query, dtype=np.float32)
        X_context = np.nan_to_num(X_context, nan=0.0, posinf=0.0, neginf=0.0)
        X_query = np.nan_to_num(X_query, nan=0.0, posinf=0.0, neginf=0.0)

        self._model.fit(X_context, y_context)
        layer_embeddings = {}

        # Internal hook-based extraction
        internal = self._extract_internal_embeddings(X_context, y_context, X_query, task=task)
        layer_embeddings.update(internal)

        # Final predictions (probabilities for classification, values for regression)
        if "final_probs" in layers:
            if task == "regression":
                preds = self._model.predict(X_query)
                layer_embeddings["final_preds"] = preds.reshape(-1, 1) if preds.ndim == 1 else preds
            else:
                probs = self._model.predict_proba(X_query)
                layer_embeddings["final_probs"] = probs

        # Pseudo-embedding fallback
        if not any(k != "final_probs" for k in layer_embeddings):
            probs = layer_embeddings.get("final_probs", self._model.predict_proba(X_query))
            layer_embeddings["transformer_hidden"] = np.hstack([
                probs,
                np.log(probs + 1e-8),
                probs * (1 - probs),
            ])

        # Select primary embedding
        for key in ["transformer_hidden", "final_probs"]:
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
        task: str = "classification",
    ) -> Dict[str, np.ndarray]:
        """Hook into TabDPT transformer layers for hidden state extraction."""
        embeddings = {}

        model = None
        for attr in ('model', '_model', 'net', 'network', 'transformer'):
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
            if any(x in name_lower for x in ['encoder', 'transformer', 'block', 'attention']):
                if hasattr(module, 'forward'):
                    self._register_hook(module, name)

        try:
            with torch.no_grad():
                if task == "regression":
                    _ = self._model.predict(X_query)
                else:
                    _ = self._model.predict_proba(X_query)

            for layer_name, activation in self._activations.items():
                if activation.ndim >= 2:
                    if activation.ndim == 3:
                        activation = activation.mean(axis=1)
                    if activation.shape[0] != len(X_query):
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
    print("Testing TabDPT embedding extraction...")

    extractor = TabDPTEmbeddingExtractor(device="cpu")
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
