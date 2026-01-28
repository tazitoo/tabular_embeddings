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
        Extract embeddings from TabICL's ICL predictor.

        TabICL architecture: col_embedder (128d) -> row_interactor (128d) ->
        icl_predictor (512d). We hook the last ICL block. Activation shape is
        (n_ensemble, n_ctx+n_query, 512). We slice query samples and mean-pool
        over the ensemble to get (n_query, 512).

        Args:
            X_context: Training features (n_context, n_features)
            y_context: Training labels (n_context,)
            X_query: Query features (n_query, n_features)
            layers: Unused (kept for API compatibility)

        Returns:
            EmbeddingResult with (n_query, 512) embeddings
        """
        if self._model is None:
            self.load_model()

        X_context = np.asarray(X_context, dtype=np.float32)
        y_context = np.asarray(y_context, dtype=np.int64)
        X_query = np.asarray(X_query, dtype=np.float32)
        X_context = np.nan_to_num(X_context, nan=0.0, posinf=0.0, neginf=0.0)
        X_query = np.nan_to_num(X_query, nan=0.0, posinf=0.0, neginf=0.0)

        n_query = len(X_query)
        self._model.fit(X_context, y_context)

        layer_embeddings = {}
        captured = {}

        # Access the underlying TabICL nn.Module
        model = self._model.model_

        def make_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    captured[name] = output.detach().cpu().numpy()
            return hook_fn

        # Hook last ICL predictor block (512-dim) and row interactor output (128-dim)
        handles = []
        handles.append(
            model.icl_predictor.tf_icl.blocks[-1].register_forward_hook(
                make_hook("icl_last_block")
            )
        )
        handles.append(
            model.row_interactor.out_ln.register_forward_hook(
                make_hook("row_output")
            )
        )

        try:
            with torch.no_grad():
                probs = self._model.predict_proba(X_query)
            layer_embeddings["final_probs"] = probs
        finally:
            for h in handles:
                h.remove()

        # Process ICL last block: (n_ensemble, n_ctx+n_query, 512)
        if "icl_last_block" in captured:
            act = captured["icl_last_block"]  # (ensemble, total_samples, 512)
            query_act = act[:, -n_query:, :]  # (ensemble, n_query, 512)
            primary_embedding = query_act.mean(axis=0)  # (n_query, 512)
            layer_embeddings["icl_last_block"] = primary_embedding
        else:
            primary_embedding = probs

        # Process row output: (n_ensemble, n_ctx+n_query, n_struct, 128)
        if "row_output" in captured:
            act = captured["row_output"]
            query_act = act[:, -n_query:, :, :]  # (ensemble, n_query, struct, 128)
            row_emb = query_act.mean(axis=(0, 2))  # (n_query, 128)
            layer_embeddings["row_output"] = row_emb

        return EmbeddingResult(
            embeddings=primary_embedding,
            model_name=self.model_name,
            extraction_point="icl_last_block",
            embedding_dim=primary_embedding.shape[1] if primary_embedding.ndim > 1 else 1,
            n_samples=n_query,
            layer_embeddings=layer_embeddings,
        )


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
