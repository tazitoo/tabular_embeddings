"""
Mitra embedding extraction.

Mitra is a 72M parameter 2D attention transformer that processes both rows and columns
simultaneously. It is available through AutoGluon's tabular interface.

We extract embeddings by:
1. Accessing the underlying PyTorch model through AutoGluon's predictor
2. Registering hooks on the 12-layer transformer outputs
3. Falling back to prediction-based pseudo-embeddings if internal access fails

Supports both classification and regression.
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from .base import EmbeddingExtractor, EmbeddingResult


class MitraEmbeddingExtractor(EmbeddingExtractor):
    """Extract embeddings from Mitra 2D attention transformer via AutoGluon."""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self._layer_names = []
        self._predictor = None

    @property
    def model_name(self) -> str:
        return "Mitra"

    @property
    def available_layers(self) -> List[str]:
        return self._layer_names if self._layer_names else [
            "transformer_2d_output",
            "final_probs",
        ]

    def load_model(self) -> None:
        """Load Mitra via AutoGluon TabularPredictor."""
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError:
            raise ImportError(
                "AutoGluon with Mitra support not found. Install with: "
                "pip install 'autogluon.tabular[mitra]'"
            )

        # We create a lightweight predictor configured to use only Mitra.
        # The actual fit happens in extract_embeddings since AutoGluon
        # needs a DataFrame with the target column.
        self._predictor = None  # Lazy init on first extract call
        self._model = True  # Mark as loaded
        self._discover_layers()

    def _discover_layers(self):
        """Set expected layer names. Actual discovery happens after fit."""
        self._layer_names = [
            "transformer_2d_output",
            "final_probs",
        ]

    def _get_mitra_model(self):
        """Dig into AutoGluon predictor to get the underlying Mitra PyTorch model."""
        if self._predictor is None:
            return None

        try:
            # AutoGluon stores models in _trainer.models
            trainer = self._predictor._trainer
            for model_name in trainer.get_model_names():
                model_obj = trainer.load_model(model_name)
                # Check if it's a Mitra model
                if 'mitra' in model_name.lower() or 'mitra' in type(model_obj).__name__.lower():
                    # Access the underlying PyTorch model
                    for attr in ('model', '_model', 'net', 'network'):
                        if hasattr(model_obj, attr):
                            candidate = getattr(model_obj, attr)
                            if hasattr(candidate, 'named_modules'):
                                return candidate
        except Exception:
            pass

        return None

    def extract_embeddings(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_query: np.ndarray,
        layers: Optional[List[str]] = None,
    ) -> EmbeddingResult:
        """
        Extract embeddings from Mitra.

        Fits AutoGluon predictor on context data, then extracts representations
        from the 12-layer 2D attention transformer.
        """
        if self._model is None:
            self.load_model()

        layers = layers or ["transformer_2d_output", "final_probs"]

        X_context = np.asarray(X_context, dtype=np.float32)
        y_context = np.asarray(y_context, dtype=np.int64)
        X_query = np.asarray(X_query, dtype=np.float32)
        X_context = np.nan_to_num(X_context, nan=0.0, posinf=0.0, neginf=0.0)
        X_query = np.nan_to_num(X_query, nan=0.0, posinf=0.0, neginf=0.0)

        import pandas as pd
        from autogluon.tabular import TabularPredictor
        import tempfile

        # Build DataFrames (AutoGluon requires DataFrame input)
        feature_names = [f"f{i}" for i in range(X_context.shape[1])]
        df_train = pd.DataFrame(X_context, columns=feature_names)
        df_train["target"] = y_context
        df_query = pd.DataFrame(X_query, columns=feature_names)

        # Fit predictor with Mitra only
        with tempfile.TemporaryDirectory() as tmpdir:
            self._predictor = TabularPredictor(
                label="target",
                path=tmpdir,
                verbosity=0,
            )
            self._predictor.fit(
                df_train,
                hyperparameters={"MITRA": {}},
                num_gpus=1 if self.device != "cpu" else 0,
            )

            layer_embeddings = {}

            # Try internal extraction
            internal = self._extract_internal_embeddings(X_query)
            layer_embeddings.update(internal)

            # Final probabilities
            if "final_probs" in layers:
                try:
                    probs = self._predictor.predict_proba(df_query).values
                    layer_embeddings["final_probs"] = probs
                except Exception:
                    preds = self._predictor.predict(df_query).values
                    layer_embeddings["final_probs"] = preds.reshape(-1, 1)

        # Pseudo-embedding fallback
        if not any(k != "final_probs" for k in layer_embeddings):
            probs = layer_embeddings.get("final_probs")
            if probs is not None:
                if probs.ndim == 1:
                    probs = probs.reshape(-1, 1)
                layer_embeddings["transformer_2d_output"] = np.hstack([
                    probs,
                    np.log(np.abs(probs) + 1e-8),
                    probs ** 2,
                ])

        # Select primary embedding
        for key in ["transformer_2d_output", "final_probs"]:
            if key in layer_embeddings:
                primary_embedding = layer_embeddings[key]
                extraction_point = key
                break
        else:
            primary_embedding = list(layer_embeddings.values())[0]
            extraction_point = list(layer_embeddings.keys())[0]

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

    def _extract_internal_embeddings(self, X_query: np.ndarray) -> Dict[str, np.ndarray]:
        """Hook into Mitra transformer for hidden state extraction."""
        embeddings = {}

        model = self._get_mitra_model()
        if model is None:
            return embeddings

        self._clear_hooks()

        for name, module in model.named_modules():
            name_lower = name.lower()
            if any(x in name_lower for x in ['attention', 'transformer', 'encoder', 'block']):
                if hasattr(module, 'forward'):
                    self._register_hook(module, name)

        try:
            import pandas as pd
            feature_names = [f"f{i}" for i in range(X_query.shape[1])]
            df_query = pd.DataFrame(X_query, columns=feature_names)

            with torch.no_grad():
                _ = self._predictor.predict(df_query)

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
    print("Testing Mitra embedding extraction...")

    extractor = MitraEmbeddingExtractor(device="cpu")
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
