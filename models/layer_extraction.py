"""Reusable model loading, hook registration, and layer extraction.

Centralizes the model-specific logic for interacting with tabular foundation
model internals. Most models accept preprocessed float32 numpy arrays.
Tabula-8B is the exception — it takes raw DataFrames and serializes to text.

Three levels of abstraction:
    load_and_fit     — load model, fit on context data (or load LLM)
    get_layer_modules — return hookable nn.Modules keyed by layer name
    predict          — run forward pass (predict / predict_proba)
    extract_all_layers — convenience: hook all layers, forward, return embeddings

Usage (extraction):
    from models.layer_extraction import load_and_fit, extract_all_layers

    clf = load_and_fit("tabpfn", X_ctx, y_ctx, task="classification", device="cuda")
    layer_embs = extract_all_layers("tabpfn", clf, X_query, task="classification")
    # layer_embs["layer_18"] → (n_query, hidden_dim)

Usage (tabula-8b — DataFrames, not numpy):
    from models.layer_extraction import load_and_fit, extract_all_layers

    handle = load_and_fit("tabula8b", X_ctx_df, y_ctx, task="classification", device="cuda")
    layer_embs = extract_all_layers("tabula8b", handle, X_query_df, task="classification")

Usage (intervention — custom hooks):
    from models.layer_extraction import load_and_fit, get_layer_modules, predict

    clf = load_and_fit("tabpfn", X_ctx, y_ctx, task="classification", device="cuda")
    modules = get_layer_modules("tabpfn", clf)
    handle = modules["layer_18"].register_forward_hook(my_modify_hook)
    preds = predict(clf, X_query, task="classification")
    handle.remove()
"""

from collections import OrderedDict, defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Model loading + fit
# ---------------------------------------------------------------------------

def load_and_fit(
    model_name: str,
    X_context: np.ndarray,
    y_context: np.ndarray,
    task: str = "classification",
    device: str = "cuda",
    **kwargs,
) -> Any:
    """Load a tabular FM and fit on context data.

    Args:
        model_name: One of tabpfn, tabicl, tabicl_v2, tabdpt, mitra, hyperfast.
        X_context: Preprocessed context features, float32 numpy.
        y_context: Context labels (int32 for clf, float32 for reg).
        task: "classification" or "regression".
        device: Torch device.
        **kwargs: Forwarded to model constructor.

    Returns:
        Fitted classifier/regressor object.
    """
    key = model_name.lower()

    if key == "tabpfn":
        from models.tabpfn_utils import load_tabpfn
        clf = load_tabpfn(task=task, device=device, n_estimators=1)
        cat_indices = kwargs.get("cat_indices", [])
        if cat_indices:
            clf.categorical_features_indices = cat_indices
        clf.fit(X_context, y_context)

    elif key in ("tabicl", "tabicl_v2"):
        if task == "regression":
            from tabicl import TabICLRegressor
            clf = TabICLRegressor(device=device, n_estimators=1)
        else:
            from tabicl import TabICLClassifier
            clf = TabICLClassifier(device=device, n_estimators=1)
        clf.fit(X_context, y_context)

    elif key == "tabdpt":
        if task == "regression":
            from tabdpt import TabDPTRegressor
            clf = TabDPTRegressor(device=device, compile=False)
        else:
            from tabdpt import TabDPTClassifier
            clf = TabDPTClassifier(device=device, compile=False)
        clf.fit(X_context, y_context)

    elif key == "mitra":
        if task == "regression":
            from autogluon.tabular.models.mitra.sklearn_interface import MitraRegressor
            clf = MitraRegressor(device=device, n_estimators=1, fine_tune=False)
        else:
            from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier
            clf = MitraClassifier(device=device, n_estimators=1, fine_tune=False)
        # Mitra expects int64 (Long) labels, not int32
        y_ctx = y_context.astype(np.int64) if task == "classification" else y_context
        clf.fit(X_context, y_ctx)
        torch.cuda.empty_cache()

    elif key == "hyperfast":
        if task == "regression":
            raise ValueError("HyperFast is classification-only")
        import os
        from hyperfast import HyperFastClassifier
        worker_path = "/data/models/tabular_fm/hyperfast/hyperfast.ckpt"
        custom_path = worker_path if os.path.exists(worker_path) else None
        cat_indices = kwargs.get("cat_indices", [])
        n_features = X_context.shape[1]
        clf = HyperFastClassifier(
            device=device, n_ensemble=1, optimization=None,
            custom_path=custom_path,
            cat_features=cat_indices if cat_indices else None,
            feature_bagging=n_features > 3000,
        )
        clf.fit(X_context, y_context)

    elif key == "tabula8b":
        clf = _load_tabula8b(device)
        # Store context for later serialization during extraction
        clf._tabula8b_context = (X_context, y_context)
        clf._tabula8b_target_name = kwargs.get("target_name", "target")

    elif key == "carte":
        clf = _load_and_fit_carte(X_context, y_context, task, device)

    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    return clf


# ---------------------------------------------------------------------------
# Layer module access
# ---------------------------------------------------------------------------

def get_layer_modules(model_name: str, clf: Any) -> OrderedDict:
    """Return hookable layer modules keyed by canonical layer name.

    The returned OrderedDict preserves depth order (layer_0, layer_1, ...,
    plus any special layers like row_output or final_norm).

    HyperFast uses a generated network with no persistent nn.Modules — it is
    not supported by this function. Use extract_all_layers() instead.
    """
    key = model_name.lower()

    if key == "tabpfn":
        model = clf.model_
        modules = OrderedDict()
        for i, layer in enumerate(model.transformer_encoder.layers):
            modules[f"layer_{i}"] = layer
        return modules

    elif key in ("tabicl", "tabicl_v2"):
        model = clf.model_
        modules = OrderedDict()
        # row_output comes before ICL blocks in the forward pass
        modules["row_output"] = model.row_interactor.out_ln
        for i, block in enumerate(model.icl_predictor.tf_icl.blocks):
            modules[f"layer_{i}"] = block
        return modules

    elif key == "tabdpt":
        model = clf.model
        modules = OrderedDict()
        modules["input_encoder"] = model.encoder
        for i, layer in enumerate(model.transformer_encoder):
            modules[f"layer_{i}"] = layer
        return modules

    elif key == "mitra":
        trainer = clf.trainers[0]
        tab2d = trainer.model
        modules = OrderedDict()
        for i, layer in enumerate(tab2d.layers):
            modules[f"layer_{i}"] = layer
        modules["final_norm"] = tab2d.final_layer_norm
        return modules

    elif key == "hyperfast":
        raise NotImplementedError(
            "HyperFast uses a generated network — no persistent nn.Modules to hook. "
            "Use extract_all_layers() which handles the manual forward pass."
        )

    elif key == "tabula8b":
        llama_model = clf.model
        modules = OrderedDict()
        for i, layer in enumerate(llama_model.layers):
            modules[f"layer_{i}"] = layer
        modules["final_norm"] = llama_model.norm
        return modules

    elif key == "carte":
        # CARTE has custom graph-based hooks — use extract_all_layers()
        raise NotImplementedError(
            "CARTE uses graph-based hooks with per-node extraction. "
            "Use extract_all_layers() which handles the graph batching."
        )

    else:
        raise ValueError(f"Unknown model: {model_name!r}")


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def predict(clf: Any, X_query: np.ndarray, task: str = "classification"):
    """Run forward pass on query data. Returns predictions."""
    with torch.no_grad():
        if task == "regression":
            return clf.predict(X_query)
        else:
            return clf.predict_proba(X_query)


# ---------------------------------------------------------------------------
# All-layer extraction
# ---------------------------------------------------------------------------

def extract_all_layers(
    model_name: str,
    clf: Any,
    X_query: np.ndarray,
    task: str = "classification",
    batch_size: int = 1024,
) -> dict[str, np.ndarray]:
    """Extract embeddings from all layers for query samples.

    Registers read hooks on every layer, runs the forward pass (batched if
    needed), and returns {layer_name: (n_query, hidden_dim)} embeddings.

    The input X_query must be preprocessed float32 numpy — this function does
    NOT modify the data in any way.

    Args:
        model_name: Model identifier.
        clf: Fitted classifier (from load_and_fit).
        X_query: Query features, preprocessed float32 numpy.
        task: "classification" or "regression".
        batch_size: Max query rows per forward pass (prevents OOM).

    Returns:
        Dict mapping layer names to (n_query, hidden_dim) float32 arrays.
    """
    key = model_name.lower()

    if key == "hyperfast":
        return _extract_hyperfast(clf, X_query, batch_size=batch_size)

    if key == "tabula8b":
        return _extract_tabula8b(clf, X_query)

    if key == "carte":
        return _extract_carte(clf, X_query)

    # Generic path: hook all layers, forward pass, process activations
    modules = get_layer_modules(model_name, clf)
    n_query = len(X_query)

    # Model-specific batch size adjustment — Mitra's 2D attention is
    # O(n_obs * n_features * dim) per layer.  After fit() (which has its own
    # OOM retry loop that shrinks max_samples_support), the query forward pass
    # shares VRAM with the fitted model.  Scale query batch by feature count.
    if key == "mitra":
        n_features = X_query.shape[1]
        batch_size = min(batch_size, max(32, 100_000 // max(n_features, 1)))

    # Process each batch independently — raw hook outputs have different
    # sequence lengths per batch (context + query_batch + thinking tokens),
    # so they can't be concatenated before processing.
    batch_results = defaultdict(list)  # layer_name → list of (n_batch_query, dim)

    for start in range(0, n_query, batch_size):
        X_batch = X_query[start:start + batch_size]
        n_batch = len(X_batch)

        captured = defaultdict(list)
        handles = []

        for name, module in modules.items():
            def make_hook(layer_name):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        out = output[0]
                    else:
                        out = output
                    if isinstance(out, torch.Tensor):
                        captured[layer_name].append(out.detach().float().cpu().numpy())
                return hook_fn
            handles.append(module.register_forward_hook(make_hook(name)))

        try:
            predict(clf, X_batch, task)
        finally:
            for handle in handles:
                handle.remove()

        # Process this batch's activations to (n_batch_query, dim)
        batch_embs = _process_activations(model_name, captured, n_batch)
        for layer_name, emb in batch_embs.items():
            batch_results[layer_name].append(emb)

    # Concatenate processed embeddings across batches
    result = {}
    for layer_name, emb_list in batch_results.items():
        result[layer_name] = np.concatenate(emb_list, axis=0)

    return result


# ---------------------------------------------------------------------------
# Model-specific activation processing
# ---------------------------------------------------------------------------

def _process_activations(
    model_name: str,
    captured: dict[str, list],
    n_query: int,
) -> dict[str, np.ndarray]:
    """Convert raw hooked tensors to (n_query, hidden_dim) embeddings.

    Each model produces activations with different tensor layouts. This
    function handles mean-pooling over structure dimensions and slicing
    to extract only the query samples.
    """
    key = model_name.lower()

    if key == "tabpfn":
        return _process_tabpfn(captured, n_query)
    elif key in ("tabicl", "tabicl_v2"):
        return _process_tabicl(captured, n_query)
    elif key == "tabdpt":
        return _process_tabdpt(captured, n_query)
    elif key == "mitra":
        return _process_mitra(captured, n_query)
    else:
        raise ValueError(f"No activation processor for {model_name!r}")


def _process_tabpfn(captured, n_query):
    """TabPFN: (1, n_ctx+n_query+thinking, n_structure, hidden) → mean-pool structure."""
    result = {}
    for key, act_list in captured.items():
        act = np.concatenate(act_list, axis=0)
        # (1, seq_len, n_structure, hidden_dim)
        query_act = act[0, -n_query:, :, :]  # (n_query, n_structure, hidden)
        result[key] = query_act.mean(axis=1)  # (n_query, hidden)
    return result


def _process_tabicl(captured, n_query):
    """TabICL: batched internally — reduce each chunk to 2D, concatenate, slice."""
    result = {}
    for key, act_list in captured.items():
        batch_embs = []
        for act in act_list:
            if act.ndim == 3:
                batch_embs.append(act[0])  # (n_batch, dim)
            elif act.ndim == 4:
                batch_embs.append(act[0].mean(axis=1))
            elif act.ndim == 2:
                batch_embs.append(act)
        if not batch_embs:
            continue
        emb_all = np.concatenate(batch_embs, axis=0)
        result[key] = emb_all[-n_query:]
    return result


def _process_tabdpt(captured, n_query):
    """TabDPT: (batch, seq_len, hidden) → mean over seq, slice query."""
    result = {}
    for key, act_list in captured.items():
        act = np.concatenate(act_list, axis=0)
        if act.ndim == 3:
            emb = act.mean(axis=1)
            result[key] = emb[-n_query:] if emb.shape[0] > n_query else emb
        elif act.ndim == 2:
            result[key] = act[-n_query:] if act.shape[0] > n_query else act
    return result


def _process_mitra(captured, n_query):
    """Mitra: handles both flash_attn (2D) and standard (4D y-token) paths."""
    result = {}
    for key, act_list in captured.items():
        batch_embs = []
        for act in act_list:
            if act.ndim == 2:
                # flash_attn path: (n_valid_tokens, dim)
                batch_embs.append(act)
            elif act.ndim == 4:
                # (1, n_batch, n_features+1, dim) — take y-token
                y_token = act[:, :, 0, :]  # (1, n_batch, dim)
                batch_embs.append(y_token.mean(axis=0))
            elif act.ndim == 3:
                batch_embs.append(act.mean(axis=0))
        if not batch_embs:
            continue
        emb_all = np.concatenate(batch_embs, axis=0)
        result[key] = emb_all[-n_query:] if emb_all.shape[0] >= n_query else emb_all
    return result


# ---------------------------------------------------------------------------
# HyperFast (manual forward — no standard hooks)
# ---------------------------------------------------------------------------

def _extract_hyperfast(clf, X_query: np.ndarray, batch_size: int = 512) -> dict[str, np.ndarray]:
    """Extract all layers from HyperFast's generated network.

    HyperFast generates a task-specific MLP from context. We manually forward
    through each layer and capture activations, averaged across the ensemble.
    Batches queries to avoid OOM on large holdout sets.
    """
    from hyperfast.hyperfast import transform_data_for_main_network

    device = clf.device if hasattr(clf, 'device') else 'cuda'
    n_layers = len(clf._main_networks[0])
    n_query = len(X_query)

    batch_results = defaultdict(list)  # layer_name → list of (n_batch, dim)

    for start in range(0, n_query, batch_size):
        X_batch = X_query[start:start + batch_size]
        X_batch_preprocessed = clf._preprocess_test_data(X_batch).to(device)

        all_layer_acts = {f"layer_{i}": [] for i in range(n_layers)}

        with torch.no_grad():
            for jj in range(len(clf._main_networks)):
                main_net = clf._main_networks[jj]
                rf = clf._move_to_device(clf._rfs[jj])
                pca = clf._move_to_device(clf._pcas[jj])

                if clf.feature_bagging:
                    X_b = X_batch_preprocessed[:, clf.selected_features[jj]]
                else:
                    X_b = X_batch_preprocessed

                X_transformed = transform_data_for_main_network(
                    X=X_b, cfg=clf._cfg, rf=rf, pca=pca
                )

                x = X_transformed
                all_layer_acts["layer_0"].append(x.cpu().numpy())

                for layer_idx, (weight, bias) in enumerate(main_net[:-1]):
                    weight = clf._move_to_device(weight)
                    bias = clf._move_to_device(bias)
                    x_new = torch.nn.functional.linear(x, weight, bias)
                    x_new = torch.nn.functional.relu(x_new)

                    if x_new.shape[-1] == x.shape[-1]:
                        x = x + x_new
                    else:
                        x = x_new

                    all_layer_acts[f"layer_{layer_idx + 1}"].append(x.cpu().numpy())

        # Average across ensemble for this batch
        for key, acts in all_layer_acts.items():
            if acts:
                stacked = np.stack(acts, axis=0)  # (n_ensemble, n_batch, dim)
                batch_results[key].append(stacked.mean(axis=0))  # (n_batch, dim)

        torch.cuda.empty_cache()

    # Concatenate batches
    result = {}
    for key, emb_list in batch_results.items():
        result[key] = np.concatenate(emb_list, axis=0)

    return result


# ---------------------------------------------------------------------------
# CARTE (GNN — graph construction from DataFrames)
# ---------------------------------------------------------------------------

def _load_and_fit_carte(X_context: pd.DataFrame, y_context: np.ndarray,
                        task: str, device: str) -> Any:
    """Load CARTE, build graphs from DataFrame, and fit.

    CARTE embeds column names and categorical values via FastText, so the
    raw DataFrame with real column names is required (not numpy).

    Returns a dict with the fitted classifier, graph transformer, and
    query-ready metadata.
    """
    from models.carte_embeddings import _patch_carte_amp, _find_fasttext_model
    _patch_carte_amp()
    from carte_ai import CARTEClassifier, Table2GraphTransformer
    from sklearn.preprocessing import RobustScaler

    ft_path = _find_fasttext_model()
    if not ft_path:
        raise ValueError("FastText model not found — see models/carte_embeddings.py")

    if not isinstance(X_context, pd.DataFrame):
        raise TypeError("CARTE requires DataFrame input with real column names")

    df_context = X_context.copy()
    # Category dtype → object for CARTE
    for col in df_context.select_dtypes(include=["category"]).columns:
        df_context[col] = df_context[col].astype("object")

    # Robust preprocessing for numeric columns (prevents PowerTransformer errors)
    num_cols = df_context.select_dtypes(include=["number"]).columns.tolist()
    scaler = None
    dropped_cols = []
    if num_cols:
        col_std = df_context[num_cols].std()
        constant_cols = col_std[col_std < 1e-6].index.tolist()
        if constant_cols:
            df_context = df_context.drop(columns=constant_cols)
            dropped_cols.extend(constant_cols)
            num_cols = [c for c in num_cols if c not in constant_cols]

        if num_cols:
            scaler = RobustScaler()
            df_context[num_cols] = scaler.fit_transform(df_context[num_cols].values)
            df_context[num_cols] = df_context[num_cols].clip(-10, 10)
            post_std = df_context[num_cols].std()
            bad_post = post_std[post_std.isna() | (post_std < 1e-6)].index.tolist()
            if bad_post:
                df_context = df_context.drop(columns=bad_post)
                dropped_cols.extend(bad_post)

    # CARTE needs at least one object-dtype column for graph construction
    if len(df_context.select_dtypes(include=["object"]).columns) == 0:
        n_bins = min(5, max(2, len(df_context.columns)))
        first_num = df_context.select_dtypes(include=["number"]).columns[0]
        df_context["_cat"] = pd.cut(df_context[first_num], bins=n_bins,
                                     labels=[f"bin_{i}" for i in range(n_bins)]).astype(str)

    # For regression, discretize targets for CARTE's classifier interface
    if task == "regression":
        y_ctx = np.asarray(y_context, dtype=np.float32)
        n_bins = min(10, len(np.unique(y_ctx)))
        y_for_fit = pd.qcut(y_ctx, q=n_bins, labels=False, duplicates='drop').astype(np.int64)
    else:
        y_ctx = np.asarray(y_context)
        if y_ctx.dtype == np.float64:
            y_ctx = y_ctx.astype(np.int64)
        y_for_fit = y_ctx

    t2g = Table2GraphTransformer(lm_model="fasttext", fasttext_model_path=ft_path)
    t2g.fit(df_context)
    X_context_graph = t2g.transform(df_context)

    for i, g in enumerate(X_context_graph):
        g.y = torch.tensor([y_for_fit[i]], dtype=torch.float32)

    clf = CARTEClassifier(device=device, num_model=1, max_epoch=50, early_stopping_patience=10, disable_pbar=True)
    clf.fit(X_context_graph, y_for_fit)
    torch.cuda.empty_cache()

    # Bundle everything needed for extraction
    clf._carte_t2g = t2g
    clf._carte_scaler = scaler
    clf._carte_dropped_cols = dropped_cols
    clf._carte_num_cols = [c for c in num_cols if c not in dropped_cols]
    clf._carte_had_cat = "_cat" not in df_context.columns  # original had object cols

    return clf


def _extract_carte(clf, X_query: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extract all-layer embeddings from CARTE GNN.

    CARTE is shallow (5 layers): initial_x → attention → readout → classifier.
    Hooks capture per-node and per-graph representations. Central node is
    extracted from per-node outputs.
    """
    if not isinstance(X_query, pd.DataFrame):
        raise TypeError("CARTE requires DataFrame input")

    t2g = clf._carte_t2g
    scaler = clf._carte_scaler
    dropped_cols = clf._carte_dropped_cols
    num_cols = clf._carte_num_cols

    df_query = X_query.copy()
    for col in df_query.select_dtypes(include=["category"]).columns:
        df_query[col] = df_query[col].astype("object")

    if dropped_cols:
        df_query = df_query.drop(columns=[c for c in dropped_cols if c in df_query.columns])
    if scaler is not None and num_cols:
        valid_cols = [c for c in num_cols if c in df_query.columns]
        if valid_cols:
            df_query[valid_cols] = scaler.transform(df_query[valid_cols].values)
            df_query[valid_cols] = df_query[valid_cols].clip(-10, 10)

    if not clf._carte_had_cat and "_cat" not in df_query.columns:
        first_num = df_query.select_dtypes(include=["number"]).columns[0]
        n_bins = min(5, max(2, len(df_query.columns)))
        df_query["_cat"] = pd.cut(df_query[first_num], bins=n_bins,
                                   labels=[f"bin_{i}" for i in range(n_bins)]).astype(str)

    X_query_graph = t2g.transform(df_query)
    n_query = len(X_query)

    model = clf.model_list_[0]
    model.eval()
    base = model.ft_base

    # Register hooks
    captured = defaultdict(list)
    handles = []

    def init_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured["layer_0"].append(output.detach().cpu().numpy())
    handles.append(base.initial_x.register_forward_hook(init_hook))

    def attn_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured["layer_1"].append(output.detach().cpu().numpy())
    handles.append(base.read_out_block.g_attn.register_forward_hook(attn_hook))

    def block_hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            captured["layer_2"].append(out.detach().cpu().numpy())
    handles.append(base.read_out_block.register_forward_hook(block_hook))

    for i, layer in enumerate(model.ft_classifier):
        if isinstance(layer, torch.nn.Linear):
            def make_clf_hook(idx):
                def hook_fn(module, input, output):
                    if isinstance(output, torch.Tensor):
                        captured[f"layer_{3+idx}"].append(output.detach().cpu().numpy())
                return hook_fn
            handles.append(layer.register_forward_hook(make_clf_hook(i)))

    # Forward pass in batches
    from torch_geometric.data import Batch
    batch_size = 100
    all_ptrs = []
    try:
        for start in range(0, n_query, batch_size):
            end = min(start + batch_size, n_query)
            batch = Batch.from_data_list(X_query_graph[start:end])
            batch.to(clf.device_)
            all_ptrs.append(batch.ptr.cpu().numpy())
            with torch.no_grad():
                _ = model(batch)
            del batch
            torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()

    # Process: extract central node for per-node outputs
    result = {}
    for key, act_list in captured.items():
        act = np.concatenate(act_list, axis=0)
        if act.shape[0] == n_query:
            result[key] = act
        elif act.shape[0] > n_query:
            central_emb = []
            node_offset = 0
            for ptr in all_ptrs:
                for i in range(len(ptr) - 1):
                    central_emb.append(act[node_offset + ptr[i]])
                node_offset += ptr[-1]
            result[key] = np.stack(central_emb)
        elif act.ndim >= 2:
            result[key] = act[:n_query]

    return result


# ---------------------------------------------------------------------------
# Tabula-8B (LLM — text serialization, not numpy)
# ---------------------------------------------------------------------------

# Global cache for Tabula-8B model (expensive to load, ~16GB)
_tabula8b_cache: dict[str, object] = {}


def _load_tabula8b(device: str = "cuda"):
    """Load and cache the Tabula-8B model. Returns the underlying LlamaModel."""
    if "model" not in _tabula8b_cache:
        import os
        import transformers

        LOCAL_PATH = "/data/models/tabula-8b"
        MODEL_ID = LOCAL_PATH if os.path.isdir(LOCAL_PATH) else "mlfoundations/tabula-8b"
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

    return _tabula8b_cache["model"]


def _serialize_row(row: pd.Series, target_name: str = "target",
                   y_val=None) -> str:
    """Serialize a DataFrame row to text: 'The col_name is value.' per column."""
    parts = []
    for col_name, val in row.items():
        if pd.isna(val):
            continue
        parts.append(f"The {col_name} is {val}.")
    text = " ".join(parts)
    if y_val is not None:
        text += f" The {target_name} is {y_val}."
    return text


def _extract_tabula8b(clf, X_query, max_context_rows: int = 16) -> dict[str, np.ndarray]:
    """Extract all-layer embeddings from Tabula-8B.

    Serializes each query row to text, prepends few-shot context, and extracts
    the last-token hidden state at every transformer layer. Processes one query
    row at a time to stay within token budget.

    Args:
        clf: The loaded CausalLM model (from load_and_fit).
        X_query: Raw DataFrame with real column names and string categoricals.
        max_context_rows: Max few-shot examples (reduced if token budget exceeded).
    """
    model = clf
    tokenizer = _tabula8b_cache["tokenizer"]
    llama_model = model.model

    X_context, y_context = model._tabula8b_context
    target_name = getattr(model, '_tabula8b_target_name', 'target')

    # X_query and X_context must be DataFrames
    if not isinstance(X_query, pd.DataFrame):
        raise TypeError("Tabula-8B requires DataFrame input with real column names")
    if not isinstance(X_context, pd.DataFrame):
        raise TypeError("Tabula-8B context must be a DataFrame")

    n_query = len(X_query)
    layers = llama_model.layers
    n_layers = len(layers)

    max_len = min(getattr(model.config, "max_position_embeddings", 4096), 4096)

    # Build context text — subsample if needed
    n_ctx = min(len(X_context), max_context_rows)
    rng = np.random.RandomState(42)
    ctx_idx = rng.choice(len(X_context), n_ctx, replace=False) if n_ctx < len(X_context) else np.arange(n_ctx)

    context_parts = [_serialize_row(X_context.iloc[i], target_name, y_context[i]) for i in ctx_idx]
    context_text = "\n".join(context_parts) + "\n"
    context_tokens = tokenizer.encode(context_text, add_special_tokens=True)

    # Iteratively reduce context if token budget exceeded (leave 200 for query)
    while len(context_tokens) > max_len - 200 and n_ctx > 2:
        n_ctx = n_ctx // 2
        ctx_idx = ctx_idx[:n_ctx]
        context_parts = [_serialize_row(X_context.iloc[i], target_name, y_context[i]) for i in ctx_idx]
        context_text = "\n".join(context_parts) + "\n"
        context_tokens = tokenizer.encode(context_text, add_special_tokens=True)

    print(f"  Tabula-8B context: {n_ctx} rows, {len(context_tokens)} tokens (max {max_len})")

    # Register hooks — capture only last-token hidden state to save VRAM
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

    # Extract one query row at a time
    all_layer_embs = defaultdict(list)

    try:
        with torch.no_grad():
            for qi in range(n_query):
                if qi % 50 == 0 or qi == n_query - 1:
                    print(f"  Query {qi+1}/{n_query}")

                query_text = _serialize_row(X_query.iloc[qi])
                query_tokens = tokenizer.encode(query_text, add_special_tokens=False)

                input_ids = context_tokens + query_tokens
                if len(input_ids) > max_len:
                    input_ids = input_ids[-max_len:]

                input_device = next(model.parameters()).device
                input_tensor = torch.tensor([input_ids], device=input_device)

                # Forward through base LlamaModel (skip lm_head to save VRAM)
                _ = llama_model(input_ids=input_tensor)

                for key, vec in captured.items():
                    all_layer_embs[key].append(vec.numpy())
                captured.clear()

                if qi % 10 == 0:
                    torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()

    result = {}
    for key, emb_list in all_layer_embs.items():
        if emb_list:
            result[key] = np.stack(emb_list, axis=0)  # (n_query, hidden_dim)

    return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sort_layer_names(names: list[str]) -> list[str]:
    """Sort layer names in canonical depth order.

    Ordering: row_output < layer_0 < layer_1 < ... < final_norm < others
    """
    def sort_key(name):
        if name.startswith("layer_"):
            try:
                return (0, int(name.split("_")[1]))
            except (ValueError, IndexError):
                return (1, name)
        elif name == "row_output":
            return (-1, 0)
        elif name == "final_norm":
            return (2, 0)
        elif name == "input_encoder":
            return (-2, 0)
        else:
            return (3, name)
    return sorted(names, key=sort_key)
