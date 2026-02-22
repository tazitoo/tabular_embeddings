#!/usr/bin/env python3
"""
Layer-wise CKA analysis within a single model.

Extracts embeddings from each transformer layer and computes pairwise CKA
to visualize how representations evolve through the network.

Usage:
    python scripts/4_results/layerwise_cka_analysis.py --model tabpfn --device cuda
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.similarity import centered_kernel_alignment


def load_dataset(dataset_name: str, max_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a dataset from TabArena or OpenML."""
    import openml
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

    # TabArena suite ID
    TABARENA_SUITE_ID = 457

    # Get dataset from OpenML
    try:
        dataset = openml.datasets.get_dataset(dataset_name, download_data=True)
    except:
        # Try by ID if name doesn't work
        suite = openml.study.get_suite(TABARENA_SUITE_ID)
        # Find dataset in suite
        for did in suite.data:
            d = openml.datasets.get_dataset(did, download_data=False)
            if d.name == dataset_name:
                dataset = openml.datasets.get_dataset(did, download_data=True)
                break
        else:
            raise ValueError(f"Dataset {dataset_name} not found")

    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    # Encode categorical features
    import pandas as pd
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit_transform(X[[col]])

    # Convert to numpy
    X = X.values.astype(np.float32)
    y = y.values

    # Handle NaNs
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Encode labels if needed
    if y.dtype == object or (hasattr(y.dtype, 'name') and y.dtype.name == 'category'):
        y = LabelEncoder().fit_transform(y.astype(str))

    # Limit samples
    if len(X) > max_samples * 2:
        indices = np.random.permutation(len(X))[:max_samples * 2]
        X = X[indices]
        y = y[indices]

    # Split into context and query
    n = len(X)
    split = n // 2
    X_context, X_query = X[:split], X[split:]
    y_context, y_query = y[:split], y[split:]

    return X_context, y_context, X_query, y_query


def extract_tabpfn_all_layers(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    device: str = "cuda",
    task: str = "classification",
) -> Dict[str, np.ndarray]:
    """Extract embeddings from all TabPFN transformer layers.

    Args:
        task: "classification" or "regression" — selects the correct model variant.
    """
    from models.tabpfn_utils import load_tabpfn

    clf = load_tabpfn(task=task, device=device)
    clf.fit(X_context, y_context)

    model = clf.model_
    n_layers = len(model.transformer_encoder.layers)
    print(f"TabPFN has {n_layers} transformer layers")

    # Register hooks for all layers
    # Use lists to accumulate across potential internal batches
    captured = defaultdict(list)
    handles = []
    n_query = len(X_query)

    for i, layer in enumerate(model.transformer_encoder.layers):
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    captured[f"layer_{layer_idx}"].append(output.detach().cpu().numpy())
            return hook_fn
        handle = layer.register_forward_hook(make_hook(i))
        handles.append(handle)

    # Forward pass
    try:
        with torch.no_grad():
            if task == "regression":
                _ = clf.predict(X_query)
            else:
                _ = clf.predict_proba(X_query)
    finally:
        for handle in handles:
            handle.remove()

    # Process captured activations — concatenate across potential batches
    layer_embeddings = {}
    for key, act_list in captured.items():
        act = np.concatenate(act_list, axis=0)
        # Shape: (1, n_ctx+n_query+thinking, n_structure, hidden_dim)
        # Query samples are the last n_query along dim 1
        query_act = act[0, -n_query:, :, :]  # (n_query, n_structure, hidden)
        # Mean-pool over structure dimension
        emb = query_act.mean(axis=1)  # (n_query, hidden)
        layer_embeddings[key] = emb

    return layer_embeddings


def extract_mitra_all_layers(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    device: str = "cuda",
    task: str = "classification",
) -> Dict[str, np.ndarray]:
    """Extract embeddings from all Mitra Tab2D transformer layers (12 layers)."""
    if task == "regression":
        from autogluon.tabular.models.mitra.sklearn_interface import MitraRegressor
        clf = MitraRegressor(device=device, n_estimators=1, fine_tune=False)
    else:
        from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier
        clf = MitraClassifier(device=device, n_estimators=1, fine_tune=False)

    clf.fit(X_context, y_context)

    n_query = len(X_query)

    # Access the Tab2D model through trainers
    trainer = clf.trainers[0]
    tab2d_model = trainer.model

    # Find all transformer layers
    # Tab2D has: x_embedding -> layers (ModuleList of Tab2DLayer) -> final_layer_norm -> final_layer
    layers = tab2d_model.layers
    n_layers = len(layers)
    print(f"Mitra has {n_layers} Tab2D transformer layers")

    # Register hooks for all layers
    # Use lists to accumulate across internal batches (Mitra may batch queries)
    captured = defaultdict(list)
    handles = []

    for i, layer in enumerate(layers):
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # Mitra layers return tuples (hidden_state, ...)
                if isinstance(output, tuple):
                    out = output[0]  # First element is hidden state
                else:
                    out = output
                if isinstance(out, torch.Tensor):
                    captured[f"layer_{layer_idx}"].append(out.detach().float().cpu().numpy())
            return hook_fn
        handle = layer.register_forward_hook(make_hook(i))
        handles.append(handle)

    # Also hook final_layer_norm
    def final_norm_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured["final_norm"].append(output.detach().float().cpu().numpy())
    handles.append(tab2d_model.final_layer_norm.register_forward_hook(final_norm_hook))

    # Forward pass
    try:
        with torch.no_grad():
            if task == "regression":
                _ = clf.predict(X_query)
            else:
                _ = clf.predict_proba(X_query)
    finally:
        for handle in handles:
            handle.remove()

    # Process captured activations — reduce each batch to 2D before concatenating.
    # Internal batches may have different query counts, so we can't concatenate raw
    # 4D tensors. Process each to (n_batch_queries, dim) first.
    layer_embeddings = {}
    for key, act_list in captured.items():
        batch_embs = []
        for act in act_list:
            if act.ndim == 2:
                # (n_valid_tokens, dim) — flash_attn path, already 2D
                batch_embs.append(act)
            elif act.ndim == 4:
                # (1, n_query_batch, n_features+1, dim) — take y-token, squeeze batch
                y_token = act[:, :, 0, :]  # (1, n_query_batch, dim)
                batch_embs.append(y_token.mean(axis=0))  # (n_query_batch, dim)
            elif act.ndim == 3:
                # (1, n_query_batch, dim)
                batch_embs.append(act.mean(axis=0))  # (n_query_batch, dim)

        if not batch_embs:
            continue

        emb_all = np.concatenate(batch_embs, axis=0)
        if emb_all.shape[0] >= n_query:
            layer_embeddings[key] = emb_all[-n_query:]
        else:
            print(f"  Warning: {key} has {emb_all.shape[0]} samples < {n_query} queries")
            layer_embeddings[key] = emb_all

    return layer_embeddings


def extract_tabicl_all_layers(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Extract embeddings from all TabICL transformer layers."""
    from tabicl import TabICLClassifier

    clf = TabICLClassifier(device=device)
    clf.fit(X_context, y_context)

    n_query = len(X_query)
    model = clf.model_

    # TabICL architecture: col_embedder -> row_interactor -> icl_predictor
    # ICL predictor has tf_icl.blocks (transformer blocks)
    icl_blocks = model.icl_predictor.tf_icl.blocks
    n_blocks = len(icl_blocks)
    print(f"TabICL has {n_blocks} ICL transformer blocks")

    # Register hooks for all ICL blocks
    # Use lists to accumulate across potential internal batches
    captured = defaultdict(list)
    handles = []

    for i, block in enumerate(icl_blocks):
        def make_hook(block_idx):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    captured[f"layer_{block_idx}"].append(output.detach().cpu().numpy())
            return hook_fn
        handle = block.register_forward_hook(make_hook(i))
        handles.append(handle)

    # Also hook row_interactor output
    def row_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured["row_output"].append(output.detach().cpu().numpy())
    handles.append(model.row_interactor.out_ln.register_forward_hook(row_hook))

    # Forward pass
    try:
        with torch.no_grad():
            _ = clf.predict_proba(X_query)
    finally:
        for handle in handles:
            handle.remove()

    # Process captured activations — concatenate across potential batches
    layer_embeddings = {}
    for key, act_list in captured.items():
        act = np.concatenate(act_list, axis=0)
        # Shape: (n_ensemble, n_ctx+n_query, dim) or (n_ensemble, n_ctx+n_query, n_struct, dim)
        if act.ndim == 3:
            query_act = act[:, -n_query:, :]  # (ensemble, n_query, dim)
            emb = query_act.mean(axis=0)  # (n_query, dim)
        elif act.ndim == 4:
            query_act = act[:, -n_query:, :, :]  # (ensemble, n_query, struct, dim)
            emb = query_act.mean(axis=(0, 2))  # (n_query, dim)
        else:
            continue
        layer_embeddings[key] = emb

    return layer_embeddings


def extract_tabdpt_all_layers(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings from all TabDPT transformer encoder layers (16 layers).

    TabDPT architecture: encoder → transformer_encoder (16 layers) → head
    """
    from tabdpt import TabDPTClassifier

    clf = TabDPTClassifier(device=device, compile=False)
    clf.fit(X_context, y_context)

    n_query = len(X_query)

    # Access the underlying model
    model = clf.model

    # Get transformer encoder layers
    encoder_layers = model.transformer_encoder
    n_layers = len(encoder_layers)
    print(f"TabDPT has {n_layers} transformer encoder layers")

    # Register hooks for all encoder layers
    # Use lists to accumulate across potential internal batches
    captured = defaultdict(list)
    handles = []

    for i, layer in enumerate(encoder_layers):
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                if isinstance(out, torch.Tensor):
                    captured[f"layer_{layer_idx}"].append(out.detach().float().cpu().numpy())
            return hook_fn
        handle = layer.register_forward_hook(make_hook(i))
        handles.append(handle)

    # Also hook the input encoder and head
    def encoder_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            out = output.detach().float().cpu().numpy()  # Convert bfloat16 to float32
            captured["input_encoder"].append(out)
    handles.append(model.encoder.register_forward_hook(encoder_hook))

    # Forward pass
    try:
        with torch.no_grad():
            _ = clf.predict_proba(X_query)
    finally:
        for handle in handles:
            handle.remove()

    # Process captured activations — concatenate across potential batches
    layer_embeddings = {}
    for key, act_list in captured.items():
        act = np.concatenate(act_list, axis=0)
        # TabDPT shapes: (batch, seq_len, hidden_dim) or (batch, hidden_dim)
        if act.ndim == 3:
            # Take mean over sequence dimension, then slice query samples
            emb = act.mean(axis=1)
            if emb.shape[0] > n_query:
                emb = emb[-n_query:]
        elif act.ndim == 2:
            emb = act[-n_query:] if act.shape[0] > n_query else act
        else:
            continue
        layer_embeddings[key] = emb

    return layer_embeddings


def extract_carte_all_layers(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    device: str = "cuda",
    task: str = "classification",
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings from CARTE GNN layers.

    CARTE architecture: initial_x → read_out_block (attention + MLP) → ft_classifier
    Since CARTE is shallow, we hook:
    - initial_x: Node embedding layer
    - read_out_block.g_attn: Graph attention output
    - read_out_block: Full block output (after MLP)
    - ft_classifier intermediate layers
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from models.carte_embeddings import _patch_carte_amp, _find_fasttext_model
    _patch_carte_amp()

    from carte_ai import CARTEClassifier, Table2GraphTransformer
    import pandas as pd

    ft_path = _find_fasttext_model()
    if not ft_path:
        raise ValueError("FastText model not found")

    clf = CARTEClassifier(device=device, num_model=3, max_epoch=50, disable_pbar=True)
    t2g = Table2GraphTransformer(lm_model="fasttext", fasttext_model_path=ft_path)

    # Robust preprocessing to prevent PowerTransformer bracket errors.
    # CARTE's Table2GraphTransformer applies Yeo-Johnson internally, which
    # diverges on constant columns or extreme value ranges (e.g. APSFailure
    # has columns spanning 0 to 2e9).
    from sklearn.preprocessing import RobustScaler
    X_context = np.nan_to_num(
        np.asarray(X_context, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    X_query = np.nan_to_num(
        np.asarray(X_query, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    # Drop constant columns (Yeo-Johnson can't optimize on zero-variance)
    col_std = X_context.std(axis=0)
    nonconstant = col_std > 0
    if not nonconstant.all():
        X_context = X_context[:, nonconstant]
        X_query = X_query[:, nonconstant]
    # RobustScaler to tame extreme ranges before Yeo-Johnson
    scaler = RobustScaler()
    X_context = scaler.fit_transform(X_context)
    X_query = scaler.transform(X_query)
    # Clip remaining outliers to ±10 IQR
    X_context = np.clip(X_context, -10, 10)
    X_query = np.clip(X_query, -10, 10)

    # For regression, discretize targets for CARTE's classifier interface
    if task == "regression":
        y_context = np.asarray(y_context, dtype=np.float32)
        n_bins = min(10, len(np.unique(y_context)))
        y_for_fit = pd.qcut(y_context, q=n_bins, labels=False, duplicates='drop').astype(np.int64)
    else:
        y_context = np.asarray(y_context)
        if y_context.dtype == np.float64:
            y_context = y_context.astype(np.int64)
        y_for_fit = y_context

    # Prepare data
    feature_names = [f"f{i}" for i in range(X_context.shape[1])]
    df_context = pd.DataFrame(X_context, columns=feature_names)
    df_query = pd.DataFrame(X_query, columns=feature_names)

    # Add synthetic categorical column
    n_bins = min(5, X_context.shape[1])
    df_context["_cat"] = pd.cut(df_context["f0"], bins=n_bins,
                                 labels=[f"bin_{i}" for i in range(n_bins)]).astype(str)
    df_query["_cat"] = pd.cut(df_query["f0"], bins=n_bins,
                               labels=[f"bin_{i}" for i in range(n_bins)]).astype(str)

    # Transform to graphs
    t2g.fit(df_context)
    X_context_graph = t2g.transform(df_context)
    X_query_graph = t2g.transform(df_query)

    # Attach y values
    for i, g in enumerate(X_context_graph):
        g.y = torch.tensor([y_for_fit[i]], dtype=torch.float32)

    # Fit
    clf.fit(X_context_graph, y_for_fit)

    n_query = len(X_query)
    model = clf.model_list_[0]
    model.eval()
    base = model.ft_base

    print(f"CARTE GNN: initial_x → read_out_block → classifier")

    # Register hooks
    # Use lists to accumulate across potential internal batches
    captured = defaultdict(list)
    handles = []

    # Hook initial_x
    def init_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured["layer_0_initial"].append(output.detach().cpu().numpy())
    handles.append(base.initial_x.register_forward_hook(init_hook))

    # Hook read_out_block attention
    def attn_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured["layer_1_attention"].append(output.detach().cpu().numpy())
    handles.append(base.read_out_block.g_attn.register_forward_hook(attn_hook))

    # Hook read_out_block full output
    def block_hook(module, input, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        if isinstance(out, torch.Tensor):
            captured["layer_2_readout"].append(out.detach().cpu().numpy())
    handles.append(base.read_out_block.register_forward_hook(block_hook))

    # Hook classifier layers
    for i, layer in enumerate(model.ft_classifier):
        if isinstance(layer, torch.nn.Linear):
            def make_clf_hook(idx):
                def hook_fn(module, input, output):
                    if isinstance(output, torch.Tensor):
                        captured[f"layer_{3+idx}_classifier"].append(output.detach().cpu().numpy())
                return hook_fn
            handles.append(layer.register_forward_hook(make_clf_hook(i)))

    # Forward pass
    try:
        from torch_geometric.data import Batch
        batch = Batch.from_data_list(X_query_graph)
        batch.to(clf.device_)

        with torch.no_grad():
            _ = model(batch)
    finally:
        for handle in handles:
            handle.remove()

    # Process captured activations — concatenate across potential batches,
    # then extract central node for per-node outputs
    layer_embeddings = {}
    for key, act_list in captured.items():
        act = np.concatenate(act_list, axis=0)
        if act.shape[0] == n_query:
            # Already per-graph
            layer_embeddings[key] = act
        elif act.shape[0] > n_query and hasattr(batch, 'ptr'):
            # Per-node output - extract central nodes
            ptr = batch.ptr.cpu().numpy()
            central_emb = []
            for i in range(len(ptr) - 1):
                central_emb.append(act[ptr[i]])
            layer_embeddings[key] = np.stack(central_emb)
        elif act.ndim >= 2:
            layer_embeddings[key] = act[:n_query]

    return layer_embeddings


# Global cache for Tabula-8B model (expensive to load, ~16GB)
_tabula8b_cache: Dict[str, object] = {}


def _get_tabula8b_model(device: str = "cuda"):
    """Load and cache the Tabula-8B model."""
    if "model" not in _tabula8b_cache:
        import transformers

        MODEL_ID = "mlfoundations/tabula-8b"
        print(f"Loading Tabula-8B from {MODEL_ID} (fp16)...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        _tabula8b_cache["model"] = model
        _tabula8b_cache["tokenizer"] = tokenizer
    return _tabula8b_cache["model"], _tabula8b_cache["tokenizer"]


def extract_tabula8b_all_layers(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    device: str = "cuda",
    task: str = "classification",
    col_names: Optional[List[str]] = None,
    target_name: str = "target",
    max_context_rows: int = 16,
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings from all Tabula-8B (Llama-3 8B) transformer layers.

    Tabula-8B is a causal LM fine-tuned on serialized tabular data. Each row is
    converted to text like "The col_name is value." and fed through the model.
    We extract the hidden state at the LAST token position for each query row,
    which captures the model's representation of that row given the few-shot context.

    This requires one forward pass per query row (causal LM limitation), so it's
    slower than ICL models that process all queries in one pass.
    """
    model, tokenizer = _get_tabula8b_model(device)

    n_query = len(X_query)
    n_features = X_context.shape[1]

    # Generate column names if not provided
    if col_names is None:
        col_names = [f"feature_{i}" for i in range(n_features)]

    # Find transformer layers
    # Llama-3 architecture: model.model.layers[0..31]
    llama_model = model.model
    layers = llama_model.layers
    n_layers = len(layers)
    print(f"Tabula-8B has {n_layers} transformer layers (dim={llama_model.config.hidden_size})")

    # Serialize a single row to text
    def serialize_row(X_row, y_val=None):
        parts = []
        for j, col in enumerate(col_names):
            val = X_row[j]
            if float(val) == int(val):
                parts.append(f"The {col} is {int(val)}.")
            else:
                parts.append(f"The {col} is {val:.4g}.")
        text = " ".join(parts)
        if y_val is not None:
            text += f" The {target_name} is {y_val}."
        return text

    # Build few-shot context text (subsample if too many context rows)
    n_ctx = min(len(X_context), max_context_rows)
    if n_ctx < len(X_context):
        rng = np.random.RandomState(42)
        ctx_idx = rng.choice(len(X_context), n_ctx, replace=False)
    else:
        ctx_idx = np.arange(n_ctx)

    context_parts = []
    for i in ctx_idx:
        context_parts.append(serialize_row(X_context[i], y_context[i]))
    context_text = "\n".join(context_parts) + "\n"

    # Tokenize context once (shared prefix), respecting model's context window
    max_len = min(getattr(model.config, "max_position_embeddings", 4096), 4096)
    context_tokens = tokenizer.encode(context_text, add_special_tokens=True)

    # Iteratively reduce context rows if tokens exceed budget (leave 200 for query)
    while len(context_tokens) > max_len - 200 and n_ctx > 2:
        n_ctx = n_ctx // 2
        ctx_idx = ctx_idx[:n_ctx]
        context_parts = [serialize_row(X_context[i], y_context[i]) for i in ctx_idx]
        context_text = "\n".join(context_parts) + "\n"
        context_tokens = tokenizer.encode(context_text, add_special_tokens=True)

    print(f"  Context: {n_ctx} rows, {len(context_tokens)} tokens (max {max_len})")

    # Register hooks for all layers + final norm.
    # IMPORTANT: Extract only the last-token hidden state in each hook to avoid
    # retaining full (1, seq_len, 4096) tensors for all 33 layers simultaneously,
    # which causes OOM on 24GB GPUs (model=16GB + KV cache=4GB + hooks=1GB > 24GB).
    captured = {}
    handles = []

    for i, layer in enumerate(layers):
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                captured[f"layer_{layer_idx}"] = out[0, -1, :].float().cpu()
            return hook_fn
        handles.append(layer.register_forward_hook(make_hook(i)))

    def final_norm_hook(module, input, output):
        captured["final_norm"] = output[0, -1, :].float().cpu()
    handles.append(llama_model.norm.register_forward_hook(final_norm_hook))

    # Extract embeddings for each query row
    all_layer_embs = {f"layer_{i}": [] for i in range(n_layers)}
    all_layer_embs["final_norm"] = []

    try:
        with torch.no_grad():
            for qi in range(n_query):
                if qi % 50 == 0 or qi == n_query - 1:
                    print(f"  Query {qi+1}/{n_query}")

                # Serialize query row (no label)
                query_text = serialize_row(X_query[qi])
                query_tokens = tokenizer.encode(
                    query_text, add_special_tokens=False,
                )

                # Combine context + query
                input_ids = context_tokens + query_tokens
                if len(input_ids) > max_len:
                    input_ids = input_ids[-max_len:]

                # With device_map="auto", input goes to the model's first device
                input_device = next(model.parameters()).device
                input_tensor = torch.tensor([input_ids], device=input_device)

                # Forward pass — hooks extract last-token hidden states
                _ = model(input_ids=input_tensor)

                # Hooks already extracted last-token vectors to CPU
                for key, vec in captured.items():
                    all_layer_embs[key].append(vec.numpy())

                captured.clear()

                # Free KV cache between query rows
                if qi % 10 == 0:
                    torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()

    # Stack into arrays
    layer_embeddings = {}
    for key, emb_list in all_layer_embs.items():
        if emb_list:
            layer_embeddings[key] = np.stack(emb_list, axis=0)  # (n_query, hidden_dim)

    return layer_embeddings


def extract_hyperfast_all_layers(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings from all layers of HyperFast's GENERATED network.

    HyperFast generates a task-specific MLP from context data. The generated
    network is a list of (weight, bias) tuples representing linear layers.
    We manually forward through each layer and capture activations.
    """
    from hyperfast import HyperFastClassifier
    from hyperfast.hyperfast import transform_data_for_main_network
    import os

    # Load model
    worker_path = "/data/models/tabular_fm/hyperfast/hyperfast.ckpt"
    custom_path = worker_path if os.path.exists(worker_path) else None

    clf = HyperFastClassifier(device=device, n_ensemble=16, custom_path=custom_path)
    clf.fit(X_context, y_context)

    n_query = len(X_query)

    # Get generated network structure - it's a list of (weight, bias) tuples
    main_network = clf._main_networks[0]
    n_layers = len(main_network)
    print(f"HyperFast generated network has {n_layers} layers (including output)")

    # Custom forward pass through generated network, capturing each layer
    X_tensor = torch.from_numpy(X_query.astype(np.float32)).to(device)

    # Initialize storage for each layer (input + all hidden layers, skip output)
    all_layer_activations = {f"layer_{i}": [] for i in range(n_layers)}

    with torch.no_grad():
        for jj in range(len(clf._main_networks)):
            main_net = clf._main_networks[jj]
            rf = clf._move_to_device(clf._rfs[jj])
            pca = clf._move_to_device(clf._pcas[jj])

            if clf.feature_bagging:
                X_b = X_tensor[:, clf.selected_features[jj]]
            else:
                X_b = X_tensor

            X_transformed = transform_data_for_main_network(
                X=X_b, cfg=clf._cfg, rf=rf, pca=pca
            )

            # Forward through generated network layers
            x = X_transformed
            all_layer_activations["layer_0"].append(x.cpu().numpy())

            for layer_idx, (weight, bias) in enumerate(main_net[:-1]):  # Skip output layer
                weight = clf._move_to_device(weight)
                bias = clf._move_to_device(bias)
                x_new = torch.nn.functional.linear(x, weight, bias)
                x_new = torch.nn.functional.relu(x_new)

                # Residual connection if dimensions match
                if x_new.shape[-1] == x.shape[-1]:
                    x = x + x_new
                else:
                    x = x_new

                all_layer_activations[f"layer_{layer_idx + 1}"].append(x.cpu().numpy())

    # Average across ensemble
    layer_embeddings = {}
    for key, acts in all_layer_activations.items():
        if acts:
            stacked = np.stack(acts, axis=0)  # (n_ensemble, n_query, dim)
            layer_embeddings[key] = stacked.mean(axis=0)  # (n_query, dim)

    return layer_embeddings


def sort_layer_names(names: List[str]) -> List[str]:
    """Sort layer names, handling both 'layer_N' and special names like 'row_output'."""
    def sort_key(name):
        if name.startswith('layer_'):
            try:
                return (0, int(name.split('_')[1]))
            except (ValueError, IndexError):
                return (1, name)
        elif name == 'row_output':
            return (-1, 0)  # Before layer_0
        elif name == 'final_norm':
            return (2, 0)  # After all layers
        else:
            return (1, name)
    return sorted(names, key=sort_key)


def compute_layerwise_cka(layer_embeddings: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """Compute pairwise CKA between all layers."""
    layer_names = sort_layer_names(list(layer_embeddings.keys()))
    n_layers = len(layer_names)

    cka_matrix = np.zeros((n_layers, n_layers))

    for i, name_i in enumerate(layer_names):
        for j, name_j in enumerate(layer_names):
            if i <= j:
                cka = centered_kernel_alignment(
                    layer_embeddings[name_i],
                    layer_embeddings[name_j]
                )
                cka_matrix[i, j] = cka
                cka_matrix[j, i] = cka

    return cka_matrix, layer_names


def plot_layerwise_cka(cka_matrix: np.ndarray, layer_names: List[str], output_path: Path, model_name: str):
    """Create heatmap of layer-wise CKA."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create labels like "L0", "L1", etc.
    labels = [f"L{i}" for i in range(len(layer_names))]

    sns.heatmap(
        cka_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'CKA Similarity', 'shrink': 0.8},
        ax=ax,
        annot_kws={'size': 8}
    )

    ax.set_title(f'{model_name} Layer-wise CKA Similarity', fontsize=14, pad=15)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    plt.tight_layout()

    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    print(f"Saved: {output_path.with_suffix('.png')}")
    plt.close()


def plot_cka_by_distance(cka_matrix: np.ndarray, output_path: Path, model_name: str):
    """Plot CKA vs layer distance to show representation drift."""
    import matplotlib.pyplot as plt

    n_layers = cka_matrix.shape[0]
    distances = []
    cka_values = []

    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            distances.append(j - i)
            cka_values.append(cka_matrix[i, j])

    # Average CKA by distance
    max_dist = n_layers - 1
    avg_cka = []
    for d in range(1, max_dist + 1):
        vals = [cka_values[k] for k, dist in enumerate(distances) if dist == d]
        avg_cka.append(np.mean(vals))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, max_dist + 1), avg_cka, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Layer Distance', fontsize=12)
    ax.set_ylabel('Average CKA Similarity', fontsize=12)
    ax.set_title(f'{model_name}: CKA vs Layer Distance', fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Mark the 2/3 point
    two_thirds_layer = int(n_layers * 2 / 3)
    ax.axvline(x=two_thirds_layer, color='red', linestyle='--', alpha=0.5,
               label=f'2/3 depth ({two_thirds_layer} layers)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Layer-wise CKA analysis")
    parser.add_argument("--model", type=str, default="tabpfn",
                        choices=["tabpfn", "mitra", "tabicl", "hyperfast", "tabdpt", "carte", "tabula8b"],
                        help="Model to analyze")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of samples for analysis")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name from TabArena/OpenML (use synthetic if not specified)")
    parser.add_argument("--task", type=str, default="classification",
                        choices=["classification", "regression"],
                        help="Task type (selects correct TabPFN variant)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    if args.dataset:
        # Load real dataset
        print(f"Loading dataset: {args.dataset}")
        X_context, y_context, X_query, _ = load_dataset(args.dataset, max_samples=args.n_samples)
        dataset_name = args.dataset
    else:
        # Generate synthetic data for analysis
        n_features = 20
        n_context = args.n_samples
        n_query = args.n_samples

        X_context = np.random.randn(n_context, n_features).astype(np.float32)
        y_context = (np.random.rand(n_context) > 0.5).astype(int)
        X_query = np.random.randn(n_query, n_features).astype(np.float32)
        dataset_name = "synthetic"

    print(f"Extracting layer-wise embeddings from {args.model}...")
    print(f"  Dataset: {dataset_name}")
    print(f"  Context: {X_context.shape}, Query: {X_query.shape}")

    task = getattr(args, "task", "classification")

    if args.model == "tabpfn":
        layer_embeddings = extract_tabpfn_all_layers(
            X_context, y_context, X_query, device=args.device, task=task
        )
    elif args.model == "mitra":
        layer_embeddings = extract_mitra_all_layers(
            X_context, y_context, X_query, device=args.device
        )
    elif args.model == "tabicl":
        layer_embeddings = extract_tabicl_all_layers(
            X_context, y_context, X_query, device=args.device
        )
    elif args.model == "hyperfast":
        layer_embeddings = extract_hyperfast_all_layers(
            X_context, y_context, X_query, device=args.device
        )
    elif args.model == "tabdpt":
        layer_embeddings = extract_tabdpt_all_layers(
            X_context, y_context, X_query, device=args.device
        )
    elif args.model == "carte":
        layer_embeddings = extract_carte_all_layers(
            X_context, y_context, X_query, device=args.device
        )
    elif args.model == "tabula8b":
        layer_embeddings = extract_tabula8b_all_layers(
            X_context, y_context, X_query, device=args.device, task=task
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if not layer_embeddings:
        print("No layer embeddings extracted!")
        return

    print(f"\nExtracted embeddings from {len(layer_embeddings)} layers:")
    for name in sort_layer_names(list(layer_embeddings.keys())):
        print(f"  {name}: {layer_embeddings[name].shape}")

    # Compute layer-wise CKA
    print("\nComputing layer-wise CKA...")
    cka_matrix, layer_names = compute_layerwise_cka(layer_embeddings)

    # Build output filename suffix
    suffix = f"{args.model}_{dataset_name}"

    # Save results
    np.savez(
        output_dir / f"layerwise_cka_{suffix}.npz",
        cka_matrix=cka_matrix,
        layer_names=layer_names,
        dataset=dataset_name,
    )
    print(f"Saved: {output_dir / f'layerwise_cka_{suffix}.npz'}")

    # Plot heatmap
    title = f"{args.model.upper()} Layer-wise CKA ({dataset_name})"
    plot_layerwise_cka(
        cka_matrix,
        layer_names,
        output_dir / f"layerwise_cka_heatmap_{suffix}",
        title
    )

    # Plot CKA by distance
    plot_cka_by_distance(
        cka_matrix,
        output_dir / f"layerwise_cka_distance_{suffix}.png",
        title
    )

    # Print summary
    print("\n" + "=" * 60)
    print("LAYER-WISE CKA SUMMARY")
    print("=" * 60)

    n_layers = len(layer_names)
    print(f"\nFirst layer vs others:")
    for i in range(1, n_layers):
        print(f"  L0 vs L{i}: {cka_matrix[0, i]:.3f}")

    print(f"\nLast layer vs others:")
    for i in range(n_layers - 1):
        print(f"  L{i} vs L{n_layers-1}: {cka_matrix[i, n_layers-1]:.3f}")

    # Find where representation is most stable (highest CKA with neighbors)
    neighbor_cka = []
    for i in range(1, n_layers - 1):
        avg = (cka_matrix[i, i-1] + cka_matrix[i, i+1]) / 2
        neighbor_cka.append((i, avg))

    if neighbor_cka:
        most_stable = max(neighbor_cka, key=lambda x: x[1])
        print(f"\nMost stable layer (highest neighbor CKA): L{most_stable[0]} (avg CKA={most_stable[1]:.3f})")
        print(f"2/3 depth would be layer: L{int(n_layers * 2 / 3)}")


def compute_critical_depth(cka_matrix: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute metrics about layer depth from CKA matrix.

    Returns:
        - critical_layer: First layer where CKA with L0 drops below threshold
        - critical_depth_frac: critical_layer / total_layers
        - final_cka: CKA between first and last layer
        - half_cka_layer: Layer where CKA with L0 reaches ~0.5 of initial
    """
    n_layers = cka_matrix.shape[0]
    l0_cka = cka_matrix[0, :]  # CKA of each layer with layer 0

    # Find first layer below threshold
    critical_layer = n_layers - 1  # Default to last
    for i in range(1, n_layers):
        if l0_cka[i] < threshold:
            critical_layer = i
            break

    # Find layer where CKA drops to ~half
    initial_cka = l0_cka[1] if n_layers > 1 else 1.0
    half_target = initial_cka * 0.5
    half_cka_layer = n_layers - 1
    for i in range(1, n_layers):
        if l0_cka[i] < half_target:
            half_cka_layer = i
            break

    return {
        'n_layers': n_layers,
        'critical_layer': critical_layer,
        'critical_depth_frac': critical_layer / n_layers,
        'half_cka_layer': half_cka_layer,
        'half_cka_depth_frac': half_cka_layer / n_layers,
        'final_cka': cka_matrix[0, -1],
        'l0_cka_profile': l0_cka.tolist(),
    }


def batch_analyze(model: str, datasets: List[str], device: str = "cuda",
                  n_samples: int = 500, output_dir: Path = None,
                  task: str = "classification") -> dict:
    """Run layer-wise CKA analysis across multiple datasets."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "output"

    results = {}

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {model} on {dataset_name}")
        print('='*60)

        try:
            # Load dataset
            X_context, y_context, X_query, _ = load_dataset(dataset_name, max_samples=n_samples)

            # Extract embeddings
            if model == "tabpfn":
                layer_embeddings = extract_tabpfn_all_layers(
                    X_context, y_context, X_query, device=device, task=task
                )
            elif model == "tabicl":
                layer_embeddings = extract_tabicl_all_layers(
                    X_context, y_context, X_query, device=device
                )
            elif model == "mitra":
                layer_embeddings = extract_mitra_all_layers(
                    X_context, y_context, X_query, device=device
                )
            elif model == "tabdpt":
                layer_embeddings = extract_tabdpt_all_layers(
                    X_context, y_context, X_query, device=device
                )
            elif model == "hyperfast":
                layer_embeddings = extract_hyperfast_all_layers(
                    X_context, y_context, X_query, device=device
                )
            elif model == "carte":
                layer_embeddings = extract_carte_all_layers(
                    X_context, y_context, X_query, device=device
                )
            elif model == "tabula8b":
                layer_embeddings = extract_tabula8b_all_layers(
                    X_context, y_context, X_query, device=device, task=task
                )
            else:
                raise ValueError(f"Unknown model: {model}")

            if not layer_embeddings:
                print(f"  No embeddings extracted for {dataset_name}")
                continue

            # Compute CKA
            cka_matrix, layer_names = compute_layerwise_cka(layer_embeddings)

            # Compute depth metrics
            depth_metrics = compute_critical_depth(cka_matrix)
            depth_metrics['dataset'] = dataset_name
            depth_metrics['model'] = model
            depth_metrics['layer_names'] = layer_names

            results[dataset_name] = depth_metrics

            # Save individual result
            suffix = f"{model}_{dataset_name}"
            np.savez(
                output_dir / f"layerwise_cka_{suffix}.npz",
                cka_matrix=cka_matrix,
                layer_names=layer_names,
                dataset=dataset_name,
            )

            print(f"  Layers: {depth_metrics['n_layers']}")
            print(f"  Critical layer (CKA<0.5): L{depth_metrics['critical_layer']} ({depth_metrics['critical_depth_frac']:.1%})")
            print(f"  Final CKA with L0: {depth_metrics['final_cka']:.3f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


def plot_depth_distribution(results: dict, output_path: Path, model_name: str):
    """Plot distribution of critical depth across datasets."""
    import matplotlib.pyplot as plt

    depths = [r['critical_depth_frac'] for r in results.values()]
    half_depths = [r['half_cka_depth_frac'] for r in results.values()]
    n_layers_list = [r['n_layers'] for r in results.values()]
    critical_layers = [r['critical_layer'] for r in results.values()]

    # Get representative n_layers (most common) and compute optimal layer
    n_layers = max(set(n_layers_list), key=n_layers_list.count)
    mean_depth = np.mean(depths)
    optimal_layer = int(round(mean_depth * (n_layers - 1)))
    mean_critical = np.mean(critical_layers)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Distribution of critical depth
    ax = axes[0]
    ax.hist(depths, bins=10, edgecolor='black', alpha=0.7)
    ax.axvline(x=2/3, color='red', linestyle='--', linewidth=2, label='2/3 depth')
    ax.axvline(x=mean_depth, color='blue', linestyle='-', linewidth=2,
               label=f'Mean: {mean_depth:.2f}')
    ax.set_xlabel('Critical Depth (fraction)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{model_name}: Critical Depth Distribution\n(layer where CKA with L0 < 0.5)', fontsize=12)
    ax.legend()
    ax.set_xlim(0, 1)

    # Panel B: Individual dataset profiles
    ax = axes[1]
    for i, (dataset, r) in enumerate(results.items()):
        profile = r['l0_cka_profile']
        n_lay = len(profile)
        x_norm = np.arange(n_lay) / (n_lay - 1) if n_lay > 1 else [0]
        ax.plot(x_norm, profile, alpha=0.5, linewidth=1)

    ax.axvline(x=2/3, color='red', linestyle='--', linewidth=2, label='2/3 depth')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='CKA=0.5')

    # Add optimal layer annotation
    ax.axvline(x=mean_depth, color='blue', linestyle='-', linewidth=2,
               label=f'Optimal: L{optimal_layer}/{n_layers} ({mean_depth:.0%})')

    ax.set_xlabel('Normalized Depth (0=input, 1=output)', fontsize=12)
    ax.set_ylabel('CKA with Layer 0', fontsize=12)
    ax.set_title(f'{model_name}: CKA Drift Profiles\n({len(results)} datasets)', fontsize=12)
    ax.legend(loc='lower left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"{model_name} DEPTH ANALYSIS SUMMARY ({len(results)} datasets)")
    print('='*60)
    print(f"Critical depth (CKA<0.5 with L0):")
    print(f"  Mean: {np.mean(depths):.3f}")
    print(f"  Std:  {np.std(depths):.3f}")
    print(f"  Min:  {np.min(depths):.3f}")
    print(f"  Max:  {np.max(depths):.3f}")
    print(f"  2/3 reference: 0.667")
    print(f"\nHalf-CKA depth (where CKA drops to 50% of L1):")
    print(f"  Mean: {np.mean(half_depths):.3f}")
    print(f"  Std:  {np.std(half_depths):.3f}")


def batch_main():
    """Entry point for batch analysis."""
    parser = argparse.ArgumentParser(description="Batch layer-wise CKA analysis across TabArena")
    parser.add_argument("--model", type=str, default="tabpfn",
                        choices=["tabpfn", "mitra", "tabicl", "hyperfast", "tabdpt", "carte", "tabula8b"],
                        help="Model to analyze")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of samples per dataset")
    parser.add_argument("--max-datasets", type=int, default=15,
                        help="Maximum number of datasets to process")
    parser.add_argument("--task", type=str, default="classification",
                        choices=["classification", "regression"],
                        help="Task type (selects correct TabPFN variant)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific dataset names (overrides OpenML suite discovery)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # Get datasets
    if args.datasets:
        datasets = args.datasets
    else:
        import openml
        suite = openml.study.get_suite(457)
        datasets = []
        for did in list(suite.data)[:args.max_datasets * 2]:  # Get extra in case some fail
            try:
                d = openml.datasets.get_dataset(did, download_data=False)
                datasets.append(d.name)
            except:
                pass
            if len(datasets) >= args.max_datasets:
                break

    print(f"Running batch analysis for {args.model} ({args.task}) on {len(datasets)} datasets")
    print(f"Datasets: {datasets}")

    # Run batch analysis
    results = batch_analyze(
        model=args.model,
        datasets=datasets,
        device=args.device,
        n_samples=args.n_samples,
        output_dir=output_dir,
        task=args.task,
    )

    # Save aggregated results
    import json
    suffix = f"{args.model}_{args.task}" if args.task != "classification" else args.model
    results_path = output_dir / f"layerwise_depth_analysis_{suffix}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved aggregated results: {results_path}")

    # Plot distribution
    if results:
        plot_depth_distribution(
            results,
            output_dir / f"layerwise_depth_distribution_{suffix}.png",
            f"{args.model.upper()} ({args.task})"
        )


if __name__ == "__main__":
    import sys
    if "--batch" in sys.argv:
        sys.argv.remove("--batch")
        batch_main()
    else:
        main()
