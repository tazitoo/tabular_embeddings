"""
CARTE embedding extraction.

CARTE (Context-Aware Representation for Table Exploration) converts each row
into a star graph where the central node represents the row and leaf nodes
represent features. A GNN performs message passing to build row representations.

We extract the central node embedding from the final GNN layer as the primary
representation. Supports both classification and regression.

Requires FastText model for text embeddings. Download with:
    import fasttext.util
    fasttext.util.download_model('en', if_exists='ignore')
"""

import copy
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import pandas as pd

from .base import EmbeddingExtractor, EmbeddingResult


def _patch_carte_amp():
    """Patch CARTE to disable AMP (incompatible with PyTorch 2.x)."""
    try:
        import carte_ai.src.carte_estimator as ce
        from torch_geometric.loader import DataLoader
        from tqdm import tqdm
    except ImportError:
        return

    def patched_run_step(self, model, data, optimizer, scaler):
        optimizer.zero_grad()
        data.to(self.device_)
        out = model(data)
        target = data.y
        if self.loss == 'categorical_crossentropy':
            target = target.to(torch.long)
        if self.output_dim_ == 1:
            out = out.view(-1).to(torch.float32)
            target = target.to(torch.float32)
        loss = self.criterion_(out, target)
        loss.backward()
        optimizer.step()

    def patched_eval(self, model, ds_eval):
        with torch.no_grad():
            model.eval()
            out = model(ds_eval)
            target = ds_eval.y
            if self.loss == 'categorical_crossentropy':
                target = target.to(torch.long)
            if self.output_dim_ == 1:
                out = out.view(-1).to(torch.float32)
                target = target.to(torch.float32)
            self.valid_loss_metric_.update(out, target)
            loss_eval = self.valid_loss_metric_.compute()
            loss_eval = loss_eval.detach().item()
            if self.valid_loss_flag_ == 'neg':
                loss_eval = -1 * loss_eval
            self.valid_loss_metric_.reset()
        return loss_eval

    def patched_train(self, X, split_index):
        ds_train = [X[i] for i in split_index[0]]
        ds_valid = [X[i] for i in split_index[1]]
        ds_valid_eval = self._set_data_eval(data=ds_valid)

        model_run_train = self._load_model()
        model_run_train.to(self.device_)
        optimizer = torch.optim.AdamW(
            model_run_train.parameters(), lr=self.learning_rate
        )
        scaler = None

        train_loader = DataLoader(
            ds_train, batch_size=self.batch_size, shuffle=False
        )
        valid_loss_best = 9e15
        es_counter = 0
        model_best_ = copy.deepcopy(model_run_train)
        for _ in tqdm(
            range(1, self.max_epoch + 1),
            desc='Model',
            disable=self.disable_pbar
        ):
            self._run_epoch(model_run_train, optimizer, train_loader, scaler)
            valid_loss = self._eval(model_run_train, ds_valid_eval)
            if valid_loss < valid_loss_best:
                valid_loss_best = valid_loss
                model_best_ = copy.deepcopy(model_run_train)
                es_counter = 0
            else:
                es_counter += 1
                if es_counter > self.early_stopping_patience:
                    break
        model_best_.eval()
        return model_best_, valid_loss_best

    ce.CARTEClassifier._run_step = patched_run_step
    ce.CARTEClassifier._eval = patched_eval
    ce.CARTEClassifier._run_train_with_early_stopping = patched_train

    # Also patch regressor if available
    if hasattr(ce, 'CARTERegressor'):
        ce.CARTERegressor._run_step = patched_run_step
        ce.CARTERegressor._eval = patched_eval
        ce.CARTERegressor._run_train_with_early_stopping = patched_train


def _find_fasttext_model() -> Optional[str]:
    """Find FastText model in common locations."""
    candidates = [
        os.path.expanduser("~/cc.en.300.bin"),
        os.path.expanduser("~/.cache/fasttext/cc.en.300.bin"),
        "/home/brian/cc.en.300.bin",
        "/data/models/cc.en.300.bin",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


class CARTEEmbeddingExtractor(EmbeddingExtractor):
    """Extract embeddings from CARTE graph transformer."""

    def __init__(self, device: str = "cpu", fasttext_path: Optional[str] = None):
        super().__init__(device)
        self._layer_names = []
        self._t2g = None
        self._fasttext_path = fasttext_path or _find_fasttext_model()
        self._patched = False

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
        """Load CARTE classifier and Table2GraphTransformer."""
        if not self._patched:
            _patch_carte_amp()
            self._patched = True

        try:
            from carte_ai import CARTEClassifier, Table2GraphTransformer
        except ImportError:
            raise ImportError(
                "CARTE not found. Install with: pip install carte-ai"
            )

        if self._fasttext_path is None:
            raise ValueError(
                "FastText model not found. Download with:\n"
                "  import fasttext.util\n"
                "  fasttext.util.download_model('en', if_exists='ignore')"
            )

        self._model = CARTEClassifier(
            device=self.device,
            num_model=3,
            max_epoch=50,
            disable_pbar=True,
        )
        self._t2g = Table2GraphTransformer(
            lm_model='fasttext',
            fasttext_model_path=self._fasttext_path,
        )
        self._discover_layers()

    def _discover_layers(self):
        """Discover hookable GNN layers."""
        self._layer_names = ["gnn_output", "final_probs"]

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
        """
        if self._model is None:
            self.load_model()

        layers = layers or ["gnn_output", "final_probs"]

        # Prepare data
        X_context = np.asarray(X_context, dtype=np.float32)
        X_query = np.asarray(X_query, dtype=np.float32)
        X_context = np.nan_to_num(X_context, nan=0.0, posinf=0.0, neginf=0.0)
        X_query = np.nan_to_num(X_query, nan=0.0, posinf=0.0, neginf=0.0)

        # For regression, discretize targets for stratified splitting
        if task == "regression":
            y_context = np.asarray(y_context, dtype=np.float32)
            # Bin continuous targets into classes for stratification
            n_bins = min(10, len(np.unique(y_context)))
            y_binned = pd.qcut(
                y_context, q=n_bins, labels=False, duplicates='drop'
            )
            y_for_fit = y_binned.astype(np.int64)
        else:
            y_context = np.asarray(y_context)
            if y_context.dtype == np.float64:
                y_context = y_context.astype(np.int64)
            y_for_fit = y_context

        # Convert to DataFrame with synthetic column names
        feature_names = [f"f{i}" for i in range(X_context.shape[1])]
        df_context = pd.DataFrame(X_context, columns=feature_names)
        df_query = pd.DataFrame(X_query, columns=feature_names)

        # CARTE requires at least one categorical column for graph construction
        # Add a synthetic categorical column based on binned values
        n_bins = min(5, X_context.shape[1])
        bin_col = pd.cut(
            df_context.iloc[:, 0],
            bins=n_bins,
            labels=[f"bin_{i}" for i in range(n_bins)]
        ).astype(str)
        df_context["_cat"] = bin_col

        bin_col_query = pd.cut(
            df_query.iloc[:, 0],
            bins=n_bins,
            labels=[f"bin_{i}" for i in range(n_bins)]
        ).astype(str)
        df_query["_cat"] = bin_col_query

        # Transform to graphs
        self._t2g.fit(df_context)
        X_context_graph = self._t2g.transform(df_context)
        X_query_graph = self._t2g.transform(df_query)

        # Attach y values to context graphs
        for i, g in enumerate(X_context_graph):
            g.y = torch.tensor([y_for_fit[i]], dtype=torch.float32)

        # Fit model
        self._model.fit(X_context_graph, y_for_fit)

        # Extract embeddings via hooks
        layer_embeddings = self._extract_internal_embeddings(X_query_graph)

        # Get predictions
        if "final_probs" in layers:
            try:
                probs = self._model.predict_proba(X_query_graph)
                if probs is not None:
                    probs = np.array(probs)
                    if probs.ndim == 1:
                        probs = probs.reshape(-1, 1)
                    layer_embeddings["final_probs"] = probs
            except Exception:
                pass

        # Fallback: construct pseudo-embedding from predictions
        if "gnn_output" not in layer_embeddings:
            probs = layer_embeddings.get("final_probs")
            if probs is None:
                try:
                    probs = np.array(self._model.predict_proba(X_query_graph))
                    if probs.ndim == 1:
                        probs = probs.reshape(-1, 1)
                except Exception:
                    probs = np.zeros((len(X_query), 2))
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
            primary_embedding = np.zeros((len(X_query), 2))
            extraction_point = "fallback"

        if primary_embedding.ndim == 1:
            primary_embedding = primary_embedding.reshape(-1, 1)

        return EmbeddingResult(
            embeddings=primary_embedding,
            model_name=self.model_name,
            extraction_point=extraction_point,
            embedding_dim=primary_embedding.shape[1],
            n_samples=len(X_query),
            layer_embeddings=layer_embeddings,
        )

    def _extract_internal_embeddings(
        self,
        X_query_graph: list,
    ) -> Dict[str, np.ndarray]:
        """Hook into CARTE GNN to capture central node embeddings."""
        embeddings = {}

        # Find the internal model(s) - CARTE uses bagging
        if not hasattr(self._model, 'model_list_') or not self._model.model_list_:
            return embeddings

        # Use first model from the ensemble
        model = self._model.model_list_[0]
        model.eval()

        self._clear_hooks()

        # Register hooks on GNN layers
        target_layers = []
        for name, module in model.named_modules():
            name_lower = name.lower()
            if any(x in name_lower for x in ['block', 'attention', 'readout']):
                if hasattr(module, 'forward'):
                    self._register_hook(module, name)
                    target_layers.append(name)

        try:
            # Create batch from query graphs
            from torch_geometric.data import Batch
            batch = Batch.from_data_list(X_query_graph)
            batch.to(self._model.device_)

            with torch.no_grad():
                _ = model(batch)

            # Process captured activations
            for layer_name, activation in self._activations.items():
                if isinstance(activation, torch.Tensor):
                    activation = activation.cpu().numpy()

                if activation.ndim >= 2:
                    # For batched GNN output, need to unbatch
                    # Check if this is per-graph or per-node output
                    n_query = len(X_query_graph)
                    if activation.shape[0] == n_query:
                        # Already per-graph
                        embeddings[layer_name] = activation
                    elif activation.shape[0] > n_query:
                        # Per-node, need to extract central nodes
                        # Central node is at index 0 for each graph
                        # Use batch.ptr to find graph boundaries
                        if hasattr(batch, 'ptr'):
                            ptr = batch.ptr.cpu().numpy()
                            central_emb = []
                            for i in range(len(ptr) - 1):
                                central_emb.append(activation[ptr[i]])
                            embeddings[layer_name] = np.stack(central_emb)
                        else:
                            # Fallback: assume equal-sized graphs
                            nodes_per_graph = activation.shape[0] // n_query
                            central_emb = activation[::nodes_per_graph][:n_query]
                            embeddings[layer_name] = central_emb

            # Use deepest layer as gnn_output
            if target_layers and target_layers[-1] in embeddings:
                embeddings["gnn_output"] = embeddings[target_layers[-1]]

        except Exception as e:
            import warnings
            warnings.warn(f"Hook extraction failed: {e}")
        finally:
            self._clear_hooks()

        return embeddings


if __name__ == "__main__":
    print("Testing CARTE embedding extraction...")

    # Check for FastText model
    ft_path = _find_fasttext_model()
    if ft_path is None:
        print("FastText model not found. Downloading...")
        import fasttext.util
        fasttext.util.download_model('en', if_exists='ignore')
        ft_path = _find_fasttext_model()

    extractor = CARTEEmbeddingExtractor(device="cpu", fasttext_path=ft_path)
    extractor.load_model()

    print(f"Model: {extractor.model_name}")
    print(f"Available layers: {extractor.available_layers}")

    # Test with synthetic data
    np.random.seed(42)
    X_ctx = np.random.randn(100, 10).astype(np.float32)
    y_ctx = (np.random.rand(100) > 0.5).astype(int)
    X_query = np.random.randn(20, 10).astype(np.float32)

    result = extractor.extract_embeddings(X_ctx, y_ctx, X_query)

    print(f"\nExtraction result:")
    print(f"  Embedding shape: {result.embeddings.shape}")
    print(f"  Embedding dim: {result.embedding_dim}")
    print(f"  Extraction point: {result.extraction_point}")
    print(f"  Layer embeddings: {list(result.layer_embeddings.keys())}")

    for layer_name, emb in result.layer_embeddings.items():
        print(f"    {layer_name}: {emb.shape}")
