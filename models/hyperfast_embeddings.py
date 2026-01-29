"""
HyperFast embedding extraction.

HyperFast is a hypernetwork that generates task-specific neural network weights
in a single forward pass. The generated network processes query samples through
residual linear layers. We extract the penultimate hidden activations from the
generated network — these are per-sample representations shaped by the
hypernetwork's encoding of the training context.

Architecture: context → hypernetwork → generated NN weights → apply to query
Extraction point: penultimate layer of generated NN (intermediate_activations)
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .base import EmbeddingExtractor, EmbeddingResult


# Default weight path (standard install location)
DEFAULT_WEIGHT_PATH = "/Users/brian/.hyperfast/hyperfast.ckpt"
WORKER_WEIGHT_PATH = "/data/models/tabular_fm/hyperfast/hyperfast.ckpt"


class HyperFastEmbeddingExtractor(EmbeddingExtractor):
    """Extract embeddings from HyperFast's generated target network."""

    def __init__(
        self,
        device: str = "cuda",
        n_ensemble: int = 16,
        weight_path: Optional[str] = None,
    ):
        super().__init__(device)
        self.n_ensemble = n_ensemble
        self.weight_path = weight_path

    @property
    def model_name(self) -> str:
        return "HyperFast"

    @property
    def available_layers(self) -> List[str]:
        return [
            "penultimate_hidden",  # Per-sample hidden state from generated NN
            "final_probs",
        ]

    def _resolve_weight_path(self) -> Optional[str]:
        """Find HyperFast weights on disk."""
        import os
        import socket

        if self.weight_path:
            return self.weight_path

        hostname = socket.gethostname()
        # GPU workers (hostname may or may not have numeric suffix)
        worker_names = ("surfer", "terrax", "octo", "firelord",
                        "surfer4", "terrax4", "octo4", "firelord4")
        if hostname in worker_names:
            if os.path.exists(WORKER_WEIGHT_PATH):
                return WORKER_WEIGHT_PATH
        # Orchestrator (galactus)
        if os.path.exists(DEFAULT_WEIGHT_PATH):
            return DEFAULT_WEIGHT_PATH

        return None

    def load_model(self) -> None:
        """Load HyperFast model with explicit weight path."""
        from hyperfast import HyperFastClassifier

        path = self._resolve_weight_path()
        self._model = HyperFastClassifier(
            device=self.device,
            n_ensemble=self.n_ensemble,
            custom_path=path,
        )

    def extract_embeddings(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_query: np.ndarray,
        layers: Optional[List[str]] = None,
    ) -> EmbeddingResult:
        """
        Extract penultimate hidden activations from HyperFast's generated network.

        HyperFast generates a task-specific NN from context data, then applies it
        to query samples. forward_main_network() returns (logits, intermediate_activations)
        where intermediate_activations is the penultimate hidden state. We monkey-patch
        the predict path to capture these per-ensemble-member, then average across
        the ensemble.

        Args:
            X_context: Training features (n_context, n_features)
            y_context: Training labels (n_context,)
            X_query: Query features (n_query, n_features)
            layers: Unused (kept for API compatibility)

        Returns:
            EmbeddingResult with penultimate hidden activations
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

        # Capture intermediate activations during predict
        from hyperfast.hyperfast import forward_main_network, transform_data_for_main_network

        all_activations = []
        X_tensor = torch.from_numpy(X_query).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self._model.batch_size, shuffle=False
        )

        all_probs = []
        for (X_batch,) in loader:
            X_batch = X_batch.to(self.device)
            batch_activations = []

            with torch.no_grad():
                for jj in range(len(self._model._main_networks)):
                    main_network = self._model._move_to_device(self._model._main_networks[jj])
                    rf = self._model._move_to_device(self._model._rfs[jj])
                    pca = self._model._move_to_device(self._model._pcas[jj])

                    if self._model.feature_bagging:
                        X_b = X_batch[:, self._model.selected_features[jj]]
                    else:
                        X_b = X_batch

                    X_transformed = transform_data_for_main_network(
                        X=X_b, cfg=self._model._cfg, rf=rf, pca=pca
                    )
                    outputs, intermediate_acts = forward_main_network(
                        X_transformed, main_network
                    )
                    batch_activations.append(intermediate_acts.cpu().numpy())

                    # Also collect probs for first batch
                    predicted = F.softmax(outputs, dim=1)
                    if jj == 0:
                        all_probs_batch = predicted.cpu().numpy()
                    else:
                        all_probs_batch = all_probs_batch + predicted.cpu().numpy()

            # Average across ensemble members
            stacked = np.stack(batch_activations, axis=0)  # (n_ensemble, batch, hidden)
            batch_mean = stacked.mean(axis=0)  # (batch, hidden)
            all_activations.append(batch_mean)
            all_probs.append(all_probs_batch / len(self._model._main_networks))

        primary_embedding = np.concatenate(all_activations, axis=0)  # (n_query, hidden)
        probs = np.concatenate(all_probs, axis=0)

        layer_embeddings["penultimate_hidden"] = primary_embedding
        layer_embeddings["final_probs"] = probs

        return EmbeddingResult(
            embeddings=primary_embedding,
            model_name=self.model_name,
            extraction_point="penultimate_hidden",
            embedding_dim=primary_embedding.shape[1] if primary_embedding.ndim > 1 else 1,
            n_samples=n_query,
            layer_embeddings=layer_embeddings,
        )


if __name__ == "__main__":
    import sys

    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    print(f"Testing HyperFast embedding extraction on {device}...")

    extractor = HyperFastEmbeddingExtractor(device=device)
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
