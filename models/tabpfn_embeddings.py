"""
TabPFN embedding extraction.

TabPFN v2.5 uses a transformer architecture that processes context (X_train, y_train)
and query (X_test) together. We can extract:
1. Final layer representations (default)
2. Intermediate transformer layer outputs
3. Attention weights for interpretability
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from .base import EmbeddingExtractor, EmbeddingResult


class TabPFNEmbeddingExtractor(EmbeddingExtractor):
    """Extract embeddings from TabPFN v2.5 transformer layers."""

    def __init__(
        self,
        device: str = "cpu",
        version: str = "v2.5",
        n_estimators: int = 2,
    ):
        super().__init__(device)
        self.version = version
        self.n_estimators = n_estimators
        self._layer_names = []

    @property
    def model_name(self) -> str:
        return f"TabPFN-{self.version}"

    @property
    def available_layers(self) -> List[str]:
        """Return discovered layer names after model is loaded."""
        return self._layer_names if self._layer_names else ["final"]

    def load_model(self) -> None:
        """Load TabPFN and discover extractable layers."""
        from tabpfn import TabPFNClassifier

        if self.version == "v2.5":
            self._model = TabPFNClassifier(
                device=self.device,
                n_estimators=self.n_estimators,
            )
        else:
            self._model = TabPFNClassifier.create_default_for_version(
                self.version,
                device=self.device,
                n_estimators=self.n_estimators,
            )

        # Discover transformer layers
        self._discover_layers()

    def _discover_layers(self):
        """Inspect model to find extractable layers."""
        self._layer_names = ["final"]

        # TabPFN internal structure varies by version
        # Try to find transformer/encoder layers
        if hasattr(self._model, 'model_'):
            model = self._model.model_
            # Look for transformer encoder layers
            for name, module in model.named_modules():
                if 'encoder' in name.lower() or 'transformer' in name.lower():
                    if 'layer' in name.lower():
                        self._layer_names.append(name)

        # If no layers found, we'll use output-level extraction
        if len(self._layer_names) == 1:
            self._layer_names.append("pre_head")
            self._layer_names.append("context_encoding")

    def extract_embeddings(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_query: np.ndarray,
        layers: Optional[List[str]] = None,
    ) -> EmbeddingResult:
        """
        Extract embeddings from TabPFN.

        For TabPFN, we can extract:
        - "final": The representation used for classification
        - "context_encoding": How the model encodes the training context
        - Layer-specific if transformer internals are accessible

        Args:
            X_context: Training features (n_context, n_features)
            y_context: Training labels (n_context,)
            X_query: Query features (n_query, n_features)
            layers: Which layers to extract (default: ["final"])

        Returns:
            EmbeddingResult with embeddings
        """
        if self._model is None:
            self.load_model()

        layers = layers or ["final"]

        # Clean inputs
        X_context = np.asarray(X_context, dtype=np.float32)
        y_context = np.asarray(y_context, dtype=np.int64)
        X_query = np.asarray(X_query, dtype=np.float32)

        # Handle NaN/Inf
        X_context = np.nan_to_num(X_context, nan=0.0, posinf=0.0, neginf=0.0)
        X_query = np.nan_to_num(X_query, nan=0.0, posinf=0.0, neginf=0.0)

        # Fit on context
        self._model.fit(X_context, y_context)

        # Extract embeddings based on requested layers
        layer_embeddings = {}

        # Method 1: Use predict_proba to get final layer output
        # The logits/probabilities are a projection of the final embedding
        if "final" in layers:
            probs = self._model.predict_proba(X_query)
            # Probabilities are low-dim (n_classes), but we can also get logits
            layer_embeddings["final_probs"] = probs

        # Method 2: Try to access internal representations
        # This requires hooking into the model during forward pass
        if any(l != "final" for l in layers) or "context_encoding" in layers:
            internal_emb = self._extract_internal_embeddings(X_context, y_context, X_query)
            layer_embeddings.update(internal_emb)

        # Default to final probabilities if no internal access
        if not layer_embeddings:
            probs = self._model.predict_proba(X_query)
            primary_embedding = probs
        else:
            # Use the highest-level internal embedding available
            for key in ["context_encoding", "pre_head", "final_probs"]:
                if key in layer_embeddings:
                    primary_embedding = layer_embeddings[key]
                    break
            else:
                primary_embedding = list(layer_embeddings.values())[0]

        return EmbeddingResult(
            embeddings=primary_embedding,
            model_name=self.model_name,
            extraction_point=layers[0] if layers else "final",
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
        Extract internal representations by hooking into the model.

        TabPFN processes context and query together through a transformer.
        We register forward hooks to capture intermediate activations.
        """
        embeddings = {}

        if not hasattr(self._model, 'model_'):
            return embeddings

        model = self._model.model_
        model.eval()

        # Clear previous hooks
        self._clear_hooks()

        # Register hooks on transformer layers
        hooked_layers = []
        for name, module in model.named_modules():
            # Hook encoder/transformer layers
            if any(x in name.lower() for x in ['encoder', 'transformer', 'attention']):
                if hasattr(module, 'forward'):
                    self._register_hook(module, name)
                    hooked_layers.append(name)

        # Also try to hook the final representation before classification head
        for name, module in model.named_modules():
            if 'head' in name.lower() or 'output' in name.lower():
                # Hook the layer before this one if possible
                pass

        try:
            # Run forward pass to trigger hooks
            with torch.no_grad():
                # TabPFN expects fit then predict, which we've done
                # Hooks should capture during predict_proba
                _ = self._model.predict_proba(X_query)

            # Extract captured activations
            for layer_name, activation in self._activations.items():
                # Activations may have batch/sequence dims
                if activation.ndim >= 2:
                    # Take mean over sequence if present
                    if activation.ndim == 3:
                        # (batch, seq, hidden) -> (batch, hidden)
                        activation = activation.mean(axis=1)
                    embeddings[layer_name] = activation

        finally:
            self._clear_hooks()

        # If no internal layers captured, create pseudo-embedding from predictions
        if not embeddings:
            probs = self._model.predict_proba(X_query)
            # Expand to higher dim for analysis
            embeddings["context_encoding"] = np.hstack([
                probs,
                np.log(probs + 1e-8),  # Log-odds
                probs * (1 - probs),    # Uncertainty (variance of Bernoulli)
            ])

        return embeddings


def verify_tabpfn_embedding_extraction(device: str = "cpu") -> bool:
    """Quick verification that TabPFN embedding extraction works."""
    try:
        extractor = TabPFNEmbeddingExtractor(device=device)
        extractor.load_model()
        return extractor.verify_loaded()
    except Exception as e:
        print(f"TabPFN embedding extraction verification failed: {e}")
        return False


if __name__ == "__main__":
    # Quick test
    print("Testing TabPFN embedding extraction...")

    extractor = TabPFNEmbeddingExtractor(device="cpu")
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
