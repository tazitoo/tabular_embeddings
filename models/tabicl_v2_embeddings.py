"""
TabICL v2 embedding extraction.

TabICL v2 (tabicl>=2.0) supports both classification and regression with 12 ICL
transformer blocks at 512-dim. Same column-then-row architecture as v1 but with
different weights and a regression head.

Registered separately from v1 to preserve both in cross-model comparisons.
"""

from typing import List, Optional

import numpy as np
import torch

from .base import EmbeddingExtractor, EmbeddingResult


class TabICLV2EmbeddingExtractor(EmbeddingExtractor):
    """Extract embeddings from TabICL v2 column-then-row transformer."""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self._layer_names = []
        self._task = "classification"

    @property
    def model_name(self) -> str:
        return "TabICL-v2"

    @property
    def available_layers(self) -> List[str]:
        return self._layer_names if self._layer_names else [
            "row_transformer_output",
            "final_probs",
        ]

    def load_model(self, task: str = "classification") -> None:
        """Load TabICL v2 classifier or regressor."""
        self._task = task
        if task == "regression":
            from tabicl import TabICLRegressor
            self._model = TabICLRegressor(device=self.device)
        else:
            from tabicl import TabICLClassifier
            self._model = TabICLClassifier(device=self.device)

    def extract_embeddings(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_query: np.ndarray,
        layers: Optional[List[str]] = None,
        task: str = "classification",
    ) -> EmbeddingResult:
        """Extract embeddings from TabICL v2's ICL predictor.

        Architecture: col_embedder (128d) -> row_interactor (128d) ->
        icl_predictor (512d, 12 blocks). We hook the last ICL block.
        Activation shape is (n_ensemble, n_ctx+n_query, 512). We slice
        query samples and mean-pool over the ensemble to get (n_query, 512).
        """
        if self._model is None or self._task != task:
            self.load_model(task=task)

        X_context = np.asarray(X_context, dtype=np.float32)
        y_context = np.asarray(y_context, dtype=np.float32 if task == "regression" else np.int64)
        X_query = np.asarray(X_query, dtype=np.float32)
        X_context = np.nan_to_num(X_context, nan=0.0, posinf=0.0, neginf=0.0)
        X_query = np.nan_to_num(X_query, nan=0.0, posinf=0.0, neginf=0.0)

        n_query = len(X_query)
        self._model.fit(X_context, y_context)

        layer_embeddings = {}
        captured = {}

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
                if task == "regression":
                    preds = self._model.predict(X_query)
                    layer_embeddings["final_preds"] = preds.reshape(-1, 1)
                else:
                    probs = self._model.predict_proba(X_query)
                    layer_embeddings["final_probs"] = probs
        finally:
            for h in handles:
                h.remove()

        # Process ICL last block: (n_ensemble, n_ctx+n_query, 512)
        if "icl_last_block" in captured:
            act = captured["icl_last_block"]
            query_act = act[:, -n_query:, :]
            primary_embedding = query_act.mean(axis=0)  # (n_query, 512)
            layer_embeddings["icl_last_block"] = primary_embedding
        else:
            primary_embedding = layer_embeddings.get(
                "final_probs", layer_embeddings.get("final_preds")
            )

        # Process row output: (n_ensemble, n_ctx+n_query, n_struct, 128)
        if "row_output" in captured:
            act = captured["row_output"]
            query_act = act[:, -n_query:, :, :]
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
