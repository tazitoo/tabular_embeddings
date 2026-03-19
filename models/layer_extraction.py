"""Reusable model loading, hook registration, and layer extraction.

Centralizes the model-specific logic for interacting with tabular foundation
model internals. All functions accept preprocessed float32 numpy arrays —
no DataFrame conversion, no nan_to_num, no preprocessing of any kind.

Three levels of abstraction:
    load_and_fit     — load model, fit on context data
    get_layer_modules — return hookable nn.Modules keyed by layer name
    predict          — run forward pass (predict / predict_proba)
    extract_all_layers — convenience: hook all layers, forward, return embeddings

Usage (extraction):
    from models.layer_extraction import load_and_fit, extract_all_layers

    clf = load_and_fit("tabpfn", X_ctx, y_ctx, task="classification", device="cuda")
    layer_embs = extract_all_layers("tabpfn", clf, X_query, task="classification")
    # layer_embs["layer_18"] → (n_query, hidden_dim)

Usage (intervention — custom hooks):
    from models.layer_extraction import load_and_fit, get_layer_modules, predict

    clf = load_and_fit("tabpfn", X_ctx, y_ctx, task="classification", device="cuda")
    modules = get_layer_modules("tabpfn", clf)
    handle = modules["layer_18"].register_forward_hook(my_modify_hook)
    preds = predict(clf, X_query, task="classification")
    handle.remove()
"""

from collections import OrderedDict, defaultdict
from typing import Any

import numpy as np
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
        clf.fit(X_context, y_context)
        torch.cuda.empty_cache()

    elif key == "hyperfast":
        if task == "regression":
            raise ValueError("HyperFast is classification-only")
        import os
        from hyperfast import HyperFastClassifier
        worker_path = "/data/models/tabular_fm/hyperfast/hyperfast.ckpt"
        custom_path = worker_path if os.path.exists(worker_path) else None
        clf = HyperFastClassifier(device=device, n_ensemble=16, custom_path=custom_path)
        clf.fit(X_context, y_context)

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
        return _extract_hyperfast(clf, X_query)

    # Generic path: hook all layers, forward pass, process activations
    modules = get_layer_modules(model_name, clf)
    n_query = len(X_query)

    # Model-specific batch size adjustment
    if key == "mitra":
        n_features = X_query.shape[1]
        n_context = len(clf.trainers[0].X_train) if hasattr(clf.trainers[0], 'X_train') else 0
        max_total = max(n_context + 50, 150_000 // max(n_features, 1))
        batch_size = min(batch_size, max(50, max_total - n_context))

    captured = defaultdict(list)
    handles = []

    # Register hooks
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

    # Batched forward pass
    try:
        for start in range(0, n_query, batch_size):
            X_batch = X_query[start:start + batch_size]
            predict(clf, X_batch, task)
    finally:
        for handle in handles:
            handle.remove()

    return _process_activations(model_name, captured, n_query)


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

def _extract_hyperfast(clf, X_query: np.ndarray) -> dict[str, np.ndarray]:
    """Extract all layers from HyperFast's generated network.

    HyperFast generates a task-specific MLP from context. We manually forward
    through each layer and capture activations, averaged across the ensemble.
    """
    from hyperfast.hyperfast import transform_data_for_main_network

    X_query_preprocessed = clf._preprocess_test_data(X_query)

    n_layers = len(clf._main_networks[0])
    all_layer_acts = {f"layer_{i}": [] for i in range(n_layers)}

    with torch.no_grad():
        for jj in range(len(clf._main_networks)):
            main_net = clf._main_networks[jj]
            rf = clf._move_to_device(clf._rfs[jj])
            pca = clf._move_to_device(clf._pcas[jj])

            if clf.feature_bagging:
                X_b = X_query_preprocessed[:, clf.selected_features[jj]]
            else:
                X_b = X_query_preprocessed

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

    # Average across ensemble
    result = {}
    for key, acts in all_layer_acts.items():
        if acts:
            stacked = np.stack(acts, axis=0)  # (n_ensemble, n_query, dim)
            result[key] = stacked.mean(axis=0)  # (n_query, dim)

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
