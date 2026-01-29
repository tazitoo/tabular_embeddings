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

    # Local checkpoint paths on GPU workers (avoids HuggingFace download)
    WORKER_CLF_PATH = "/data/models/tabular_fm/tabpfn/tabpfn-v2.5-classifier-v2.5_real.ckpt"
    WORKER_REG_PATH = "/data/models/tabular_fm/tabpfn/tabpfn-v2.5-regressor-v2.5_default.ckpt"

    def __init__(
        self,
        device: str = "cpu",
        version: str = "v2.5",
        n_estimators: int = 2,
        model_path: Optional[str] = None,
    ):
        super().__init__(device)
        self.version = version
        self.n_estimators = n_estimators
        self.model_path = model_path
        self._layer_names = []

    @property
    def model_name(self) -> str:
        return f"TabPFN-{self.version}"

    @property
    def available_layers(self) -> List[str]:
        """Return discovered layer names after model is loaded."""
        return self._layer_names if self._layer_names else ["final"]

    def load_model(self, task: str = "classification") -> None:
        """Load TabPFN classifier or regressor based on task type."""
        import os

        is_regression = task == "regression"

        if is_regression:
            from tabpfn import TabPFNRegressor as TabPFNModel
            worker_path = self.WORKER_REG_PATH
        else:
            from tabpfn import TabPFNClassifier as TabPFNModel
            worker_path = self.WORKER_CLF_PATH

        # Resolve model path: explicit > worker checkpoint > auto-download
        model_path = self.model_path
        if model_path is None and os.path.exists(worker_path):
            model_path = worker_path

        if self.version == "v2.5":
            kwargs = dict(device=self.device, n_estimators=self.n_estimators)
            if model_path is not None:
                kwargs["model_path"] = model_path
            self._model = TabPFNModel(**kwargs)
        else:
            self._model = TabPFNModel.create_default_for_version(
                self.version,
                device=self.device,
                n_estimators=self.n_estimators,
            )
        self._current_task = task

    def extract_embeddings(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_query: np.ndarray,
        layers: Optional[List[str]] = None,
        task: str = "classification",
    ) -> EmbeddingResult:
        """
        Extract embeddings from TabPFN's final transformer layer.

        TabPFN v2.5 is a PerFeatureTransformer with 14 layers. Activation shape
        is (1, n_ctx+n_query+64_thinking, n_structure, 192). We hook the last
        transformer layer, slice out the query samples, and mean-pool over the
        structure dimension to get (n_query, 192).

        Supports both classification (TabPFNClassifier) and regression
        (TabPFNRegressor) — automatically reloads the correct model variant.

        Args:
            X_context: Training features (n_context, n_features)
            y_context: Training labels (n_context,)
            X_query: Query features (n_query, n_features)
            layers: Unused (kept for API compatibility)
            task: "classification" or "regression"

        Returns:
            EmbeddingResult with (n_query, 192) embeddings
        """
        # Load or reload model if task type changed
        if self._model is None or getattr(self, "_current_task", None) != task:
            self.load_model(task=task)

        X_context = np.asarray(X_context, dtype=np.float32)
        y_dtype = np.float32 if task == "regression" else np.int64
        y_context = np.asarray(y_context, dtype=y_dtype)
        X_query = np.asarray(X_query, dtype=np.float32)
        X_context = np.nan_to_num(X_context, nan=0.0, posinf=0.0, neginf=0.0)
        X_query = np.nan_to_num(X_query, nan=0.0, posinf=0.0, neginf=0.0)

        n_query = len(X_query)
        self._model.fit(X_context, y_context)

        layer_embeddings = {}

        # Hook the last transformer layer
        model = self._model.model_
        captured = {}

        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                captured["last_layer"] = output.detach().cpu().numpy()

        last_layer = model.transformer_encoder.layers[-1]
        handle = last_layer.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                if task == "regression":
                    preds = self._model.predict(X_query)
                    layer_embeddings["final_preds"] = preds
                else:
                    probs = self._model.predict_proba(X_query)
                    layer_embeddings["final_probs"] = probs
        finally:
            handle.remove()

        if "last_layer" in captured:
            # Shape: (1, n_ctx+n_query+thinking, n_structure, hidden_dim)
            act = captured["last_layer"]
            # Query samples are the last n_query along dim 1
            query_act = act[0, -n_query:, :, :]  # (n_query, n_structure, hidden)
            # Mean-pool over structure dimension
            primary_embedding = query_act.mean(axis=1)  # (n_query, hidden)
            layer_embeddings["last_transformer_layer"] = primary_embedding
        else:
            # Fallback: use probabilities (should not happen)
            primary_embedding = probs

        return EmbeddingResult(
            embeddings=primary_embedding,
            model_name=self.model_name,
            extraction_point="last_transformer_layer",
            embedding_dim=primary_embedding.shape[1] if primary_embedding.ndim > 1 else 1,
            n_samples=n_query,
            layer_embeddings=layer_embeddings,
        )


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
