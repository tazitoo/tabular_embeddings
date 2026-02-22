"""
Mitra embedding extraction.

Mitra is a 72M parameter 2D attention transformer that processes rows and columns
simultaneously. Available through AutoGluon's sklearn-compatible interface.

Architecture: x_embedding → 12 Tab2D transformer layers → final_layer_norm → final_layer
Extraction point: final_layer_norm output (last hidden state before output projection)

Note: By default uses ICL mode (fine_tune=False) for frozen-weight embedding
extraction. Pass fine_tune=True to finetune pretrained weights per dataset.
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from .base import EmbeddingExtractor, EmbeddingResult


class MitraEmbeddingExtractor(EmbeddingExtractor):
    """Extract embeddings from Mitra's 2D attention transformer."""

    def __init__(self, device: str = "cpu", n_estimators: int = 1, fine_tune: bool = False):
        super().__init__(device)
        self.n_estimators = n_estimators
        self.fine_tune = fine_tune
        self._classifier = None

    @property
    def model_name(self) -> str:
        return "Mitra"

    @property
    def available_layers(self) -> List[str]:
        return [
            "final_hidden",   # After final_layer_norm, before output projection
            "final_probs",
        ]

    def load_model(self, task: str = "classification") -> None:
        """Load Mitra classifier or regressor based on task type."""
        try:
            if task == "regression":
                from autogluon.tabular.models.mitra.sklearn_interface import MitraRegressor as MitraModel
            else:
                from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier as MitraModel
        except ImportError:
            raise ImportError(
                "AutoGluon with Mitra support not found. Install with: "
                "pip install 'autogluon.tabular[mitra]'"
            )

        self._classifier = MitraModel(
            device=self.device,
            n_estimators=self.n_estimators,
            fine_tune=self.fine_tune,
        )
        self._model = True  # Mark as loaded
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
        Extract embeddings from Mitra's last hidden state.

        Fits MitraClassifier on context data (ICL mode by default, no weight
        updates), then hooks final_layer_norm during prediction to capture the
        last hidden state before the output projection.

        The Tab2D forward pass produces shape (batch, n_query, n_features+1, dim).
        We take the first feature position (y-token) and average across batches
        to get (n_query, dim).

        Args:
            X_context: Training features (n_context, n_features)
            y_context: Training labels (n_context,)
            X_query: Query features (n_query, n_features)
            layers: Unused (kept for API compatibility)

        Returns:
            EmbeddingResult with last hidden state embeddings
        """
        if self._model is None or getattr(self, "_current_task", None) != task:
            self.load_model(task=task)

        X_context = np.asarray(X_context, dtype=np.float32)
        y_dtype = np.float32 if task == "regression" else np.int64
        y_context = np.asarray(y_context, dtype=y_dtype)
        X_query = np.asarray(X_query, dtype=np.float32)
        X_context = np.nan_to_num(X_context, nan=0.0, posinf=0.0, neginf=0.0)
        X_query = np.nan_to_num(X_query, nan=0.0, posinf=0.0, neginf=0.0)

        n_query = len(X_query)

        # Fit (finetunes pretrained weights on context data)
        self._classifier.fit(X_context, y_context)

        layer_embeddings = {}

        # Hook final_layer_norm on each trainer's Tab2D model
        # Use lists to accumulate across internal batches (Mitra may batch queries)
        all_hidden_states = []
        captured_per_trainer = []

        handles = []
        for trainer in self._classifier.trainers:
            tab2d_model = trainer.model
            captured = {"hidden": []}

            def make_hook(capture_dict):
                def hook_fn(module, input, output):
                    if isinstance(output, torch.Tensor):
                        capture_dict["hidden"].append(output.detach().cpu().numpy())
                return hook_fn

            h = tab2d_model.final_layer_norm.register_forward_hook(make_hook(captured))
            handles.append(h)
            captured_per_trainer.append(captured)

        try:
            if task == "regression":
                preds = self._classifier.predict(X_query)
                layer_embeddings["final_preds"] = preds
            else:
                probs = self._classifier.predict_proba(X_query)
                layer_embeddings["final_probs"] = probs
        finally:
            for h in handles:
                h.remove()

        # Process captured hidden states — reduce each batch to 2D before concatenating.
        # Internal batches may have different query counts (Mitra batches for GPU memory).
        for captured in captured_per_trainer:
            if not captured["hidden"]:
                continue
            batch_embs = []
            for hidden in captured["hidden"]:
                if hidden.ndim == 2:
                    # (n_valid_tokens, dim) — flash_attn path
                    batch_embs.append(hidden)
                elif hidden.ndim == 4:
                    # (1, n_query_batch, n_features+1, dim) — take y-token
                    y_token = hidden[:, :, 0, :]  # (1, n_query_batch, dim)
                    batch_embs.append(y_token.mean(axis=0))  # (n_query_batch, dim)
                elif hidden.ndim == 3:
                    # (1, n_query_batch, dim)
                    batch_embs.append(hidden.mean(axis=0))  # (n_query_batch, dim)
            if not batch_embs:
                continue
            emb_all = np.concatenate(batch_embs, axis=0)
            if emb_all.shape[0] >= n_query:
                all_hidden_states.append(emb_all[-n_query:])
            else:
                all_hidden_states.append(emb_all)

        if all_hidden_states:
            # Average across ensemble trainers
            primary_embedding = np.stack(all_hidden_states, axis=0).mean(axis=0)
            layer_embeddings["final_hidden"] = primary_embedding
        else:
            # Fallback to probs
            primary_embedding = probs

        return EmbeddingResult(
            embeddings=primary_embedding,
            model_name=self.model_name,
            extraction_point="final_hidden" if all_hidden_states else "final_probs",
            embedding_dim=primary_embedding.shape[1] if primary_embedding.ndim > 1 else 1,
            n_samples=n_query,
            layer_embeddings=layer_embeddings,
        )


if __name__ == "__main__":
    import sys

    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    print(f"Testing Mitra embedding extraction on {device}...")

    extractor = MitraEmbeddingExtractor(device=device, n_estimators=1)
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
    for layer_name, emb in result.layer_embeddings.items():
        print(f"  {layer_name}: {emb.shape}")
