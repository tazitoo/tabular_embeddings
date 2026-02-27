#!/usr/bin/env python3
"""SAE feature ablation via pre-hook intervention on tabular foundation models.

For ICL transformers (TabPFN, Mitra, TabICL, TabDPT), we cannot simply feed ablated
embeddings back — the model processes context+query in one forward pass. Instead:

1. Forward pass 1: Capture hidden state at layer L, get baseline predictions.
2. SAE encode the query portion (mean-pooled over structure dims).
3. Zero out specified features in the SAE latent space.
4. Compute delta = decode(ablated) - decode(original).
5. Forward pass 2: Pre-hook on layer L+1 adds delta to query hidden states.
6. Return (baseline_preds, ablated_preds, y_query).

For HyperFast (generated MLP), we manually forward through the network weights
with the ablated intermediate activations.

Usage:
    # Verify identity intervention (ablate nothing)
    python scripts/intervene_sae.py --model tabpfn --dataset adult \\
        --ablate-features "" --verify-identity --device cuda

    # Ablate specific features
    python scripts/intervene_sae.py --model tabpfn --dataset adult \\
        --ablate-features 42,108,305 --device cuda

    # Ablate all features (should degrade predictions)
    python scripts/intervene_sae.py --model tabpfn --dataset adult \\
        --ablate-all --device cuda
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "4_results"))

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_SAE_DIR = PROJECT_ROOT / "output" / "sae_tabarena_sweep_round5"
DEFAULT_LAYERS_PATH = PROJECT_ROOT / "config" / "optimal_extraction_layers.json"

# Model display name -> checkpoint key
MODEL_KEYS = {
    "tabpfn": "tabpfn",
    "mitra": "mitra",
    "tabicl": "tabicl",
    "tabdpt": "tabdpt",
    "hyperfast": "hyperfast",
}


def load_sae(model_key: str, sae_dir: Path = DEFAULT_SAE_DIR, device: str = "cuda"):
    """Load a trained Matryoshka-Archetypal SAE for the given model.

    Handles archetypal SAE extra parameters (archetype_logits, archetype_deviation,
    reference_data) that must be registered before load_state_dict.

    Returns:
        (sae_model, sae_config) tuple with model in eval mode on device.
    """
    from analysis.sparse_autoencoder import SparseAutoencoder, SAEConfig

    ckpt_path = sae_dir / model_key / "sae_matryoshka_archetypal_validated.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    if not isinstance(config, SAEConfig):
        config = SAEConfig(**config)

    sae = SparseAutoencoder(config)

    # Register archetypal parameters before loading state dict
    state_dict = ckpt["model_state_dict"]
    if "reference_data" in state_dict and state_dict["reference_data"] is not None:
        sae.register_buffer("reference_data", state_dict["reference_data"])
        if "archetype_logits" in state_dict:
            sae.archetype_logits = torch.nn.Parameter(state_dict["archetype_logits"])
        if "archetype_deviation" in state_dict:
            sae.archetype_deviation = torch.nn.Parameter(state_dict["archetype_deviation"])

    sae.load_state_dict(state_dict, strict=False)
    sae.to(device)
    sae.eval()

    return sae, config


def get_extraction_layer(model_key: str, layers_path: Path = DEFAULT_LAYERS_PATH) -> int:
    """Get the optimal extraction layer index for a model."""
    with open(layers_path) as f:
        layers_config = json.load(f)
    return layers_config[model_key]["optimal_layer"]


def compute_ablation_delta(
    sae: torch.nn.Module,
    embeddings: torch.Tensor,
    ablate_features: List[int],
) -> torch.Tensor:
    """Compute the delta from ablating SAE features.

    Args:
        sae: Trained SAE in eval mode
        embeddings: (n_query, emb_dim) mean-pooled query embeddings
        ablate_features: Feature indices to zero out

    Returns:
        delta: (n_query, emb_dim) to add to hidden states
    """
    with torch.no_grad():
        h = sae.encode(embeddings)
        original_recon = sae.decode(h)

        h_ablated = h.clone()
        if ablate_features:
            h_ablated[:, ablate_features] = 0.0
        ablated_recon = sae.decode(h_ablated)

        delta = ablated_recon - original_recon

    return delta


# ── TabPFN Intervention ─────────────────────────────────────────────────────


def intervene_tabpfn(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    sae: torch.nn.Module,
    ablate_features: List[int],
    extraction_layer: int,
    device: str = "cuda",
    task: str = "classification",
) -> Dict[str, np.ndarray]:
    """Run TabPFN with SAE feature ablation at extraction layer.

    Returns dict with: baseline_preds, ablated_preds, y_query
    """
    from models.tabpfn_utils import load_tabpfn

    clf = load_tabpfn(task=task, device=device, n_estimators=1)
    clf.fit(X_context, y_context)
    model = clf.model_
    n_query = len(X_query)
    layers = model.transformer_encoder.layers

    # --- Pass 1: Capture hidden state + baseline predictions ---
    captured = {}

    def capture_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured["hidden"] = output.detach()

    handle = layers[extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            if task == "regression":
                baseline_preds = clf.predict(X_query)
            else:
                baseline_preds = clf.predict_proba(X_query)
    finally:
        handle.remove()

    hidden_state = captured["hidden"]
    # Shape: (1, seq_len, n_structure, hidden_dim)
    # Query positions are last n_query along dim 1
    query_hidden = hidden_state[0, -n_query:, :, :]  # (n_query, n_struct, hidden)
    query_emb = query_hidden.mean(dim=1)  # (n_query, hidden) — mean-pool structure

    # --- Compute delta ---
    delta = compute_ablation_delta(sae, query_emb, ablate_features)
    # Broadcast delta back to structure dimension
    # delta shape: (n_query, hidden) → (n_query, 1, hidden)
    delta_broadcast = delta.unsqueeze(1)

    # --- Pass 2: Pre-hook on layer L+1 to inject delta ---
    next_layer = extraction_layer + 1
    if next_layer >= len(layers):
        raise ValueError(
            f"Extraction layer {extraction_layer} is the last layer "
            f"({len(layers)} total). Cannot place pre-hook on next layer."
        )

    def modify_hook(module, args):
        """Pre-forward hook: modify input to layer L+1."""
        inp = args[0] if isinstance(args, tuple) else args
        if isinstance(inp, torch.Tensor) and inp.ndim == 4:
            # inp shape: (1, seq_len, n_structure, hidden_dim)
            modified = inp.clone()
            modified[0, -n_query:, :, :] += delta_broadcast
            return (modified,) + args[1:] if isinstance(args, tuple) else modified
        return args

    handle = layers[next_layer].register_forward_pre_hook(modify_hook)
    try:
        with torch.no_grad():
            if task == "regression":
                ablated_preds = clf.predict(X_query)
            else:
                ablated_preds = clf.predict_proba(X_query)
    finally:
        handle.remove()

    return {
        "baseline_preds": np.asarray(baseline_preds),
        "ablated_preds": np.asarray(ablated_preds),
        "y_query": np.asarray(y_query),
    }


# ── Mitra Intervention ───────────────────────────────────────────────────────


def intervene_mitra(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    sae: torch.nn.Module,
    ablate_features: List[int],
    extraction_layer: int,
    device: str = "cuda",
    task: str = "classification",
) -> Dict[str, np.ndarray]:
    """Run Mitra with SAE feature ablation at extraction layer.

    Mitra has 12 Tab2D layers. Hidden state is 4D: (1, n_batch, n_feat+1, dim).
    The y-token at position 0 on dim 2 is what becomes the embedding.
    """
    n_features = X_query.shape[1]
    max_context = max(100, 200_000 // max(n_features, 1))
    if len(X_context) > max_context:
        X_context = X_context[:max_context]
        y_context = y_context[:max_context]

    if task == "regression":
        from autogluon.tabular.models.mitra.sklearn_interface import MitraRegressor
        clf = MitraRegressor(device=device, n_estimators=1, fine_tune=False)
    else:
        from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier
        clf = MitraClassifier(device=device, n_estimators=1, fine_tune=False)

    clf.fit(X_context, y_context)
    torch.cuda.empty_cache()

    n_query = len(X_query)
    trainer = clf.trainers[0]
    tab2d_model = trainer.model
    layers = tab2d_model.layers

    # --- Pass 1: Capture hidden state + baseline predictions ---
    captured_hidden = []

    def capture_hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            captured_hidden.append(out.detach())

    handle = layers[extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            if task == "regression":
                baseline_preds = clf.predict(X_query)
            else:
                baseline_preds = clf.predict_proba(X_query)
    finally:
        handle.remove()

    # Process hidden states: extract y-token embeddings from batched outputs
    query_embs = []
    for hidden in captured_hidden:
        if hidden.ndim == 4:
            # (1, n_batch, n_feat+1, dim) → y-token at position 0
            y_tokens = hidden[:, :, 0, :]  # (1, n_batch, dim)
            query_embs.append(y_tokens.squeeze(0))  # (n_batch, dim)
        elif hidden.ndim == 3:
            query_embs.append(hidden.squeeze(0))  # (n_batch, dim)
        elif hidden.ndim == 2:
            query_embs.append(hidden)

    all_emb = torch.cat(query_embs, dim=0)
    query_emb = all_emb[-n_query:]  # (n_query, dim)

    # --- Compute delta ---
    delta = compute_ablation_delta(sae, query_emb, ablate_features)

    # --- Pass 2: Pre-hook on layer L+1 ---
    next_layer = extraction_layer + 1
    if next_layer >= len(layers):
        raise ValueError(
            f"Extraction layer {extraction_layer} is the last layer "
            f"({len(layers)} total)."
        )

    # Track which batch we're on for delta injection
    batch_offset = [0]

    def modify_hook(module, args):
        inp = args[0] if isinstance(args, tuple) else args
        if isinstance(inp, torch.Tensor):
            modified = inp.clone()
            if modified.ndim == 4:
                # (1, n_batch, n_feat+1, dim)
                n_batch = modified.shape[1]
                start = batch_offset[0]
                end = min(start + n_batch, n_query)
                actual = end - start
                if actual > 0:
                    # Add delta to y-token (position 0) and broadcast to all feature positions
                    modified[0, :actual, :, :] += delta[start:end].unsqueeze(1)
                    batch_offset[0] = end
            return (modified,) + args[1:] if isinstance(args, tuple) else modified
        return args

    handle = layers[next_layer].register_forward_pre_hook(modify_hook)
    try:
        with torch.no_grad():
            batch_offset[0] = 0  # Reset
            if task == "regression":
                ablated_preds = clf.predict(X_query)
            else:
                ablated_preds = clf.predict_proba(X_query)
    finally:
        handle.remove()

    return {
        "baseline_preds": np.asarray(baseline_preds),
        "ablated_preds": np.asarray(ablated_preds),
        "y_query": np.asarray(y_query),
    }


# ── TabICL Intervention ──────────────────────────────────────────────────────


def intervene_tabicl(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    sae: torch.nn.Module,
    ablate_features: List[int],
    extraction_layer: int,
    device: str = "cuda",
    task: str = "classification",
) -> Dict[str, np.ndarray]:
    """Run TabICL with SAE feature ablation at extraction layer.

    TabICL has 8 ICL predictor blocks. Hidden state is 3D: (n_ensemble, seq, 512).
    Query positions are last n_query along dim 1.
    """
    from tabicl import TabICLClassifier

    clf = TabICLClassifier(device=device, n_estimators=1)
    clf.fit(X_context, y_context)

    n_query = len(X_query)
    model = clf.model_
    blocks = model.icl_predictor.tf_icl.blocks

    # --- Pass 1: Capture hidden state + baseline predictions ---
    captured = {}

    def capture_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured["hidden"] = output.detach()

    handle = blocks[extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            baseline_preds = clf.predict_proba(X_query)
    finally:
        handle.remove()

    hidden_state = captured["hidden"]
    # Shape: (n_ensemble, n_ctx+n_query, 512)
    query_hidden = hidden_state[:, -n_query:, :]  # (n_ens, n_query, 512)
    # Mean-pool ensemble dimension
    query_emb = query_hidden.mean(dim=0)  # (n_query, 512)

    # --- Compute delta ---
    delta = compute_ablation_delta(sae, query_emb, ablate_features)
    # Broadcast to ensemble dimension: (1, n_query, 512)
    delta_broadcast = delta.unsqueeze(0)

    # --- Pass 2: Pre-hook on block L+1 ---
    next_block = extraction_layer + 1
    if next_block >= len(blocks):
        raise ValueError(
            f"Extraction block {extraction_layer} is the last block "
            f"({len(blocks)} total)."
        )

    def modify_hook(module, args):
        inp = args[0] if isinstance(args, tuple) else args
        if isinstance(inp, torch.Tensor) and inp.ndim == 3:
            modified = inp.clone()
            modified[:, -n_query:, :] += delta_broadcast
            return (modified,) + args[1:] if isinstance(args, tuple) else modified
        return args

    handle = blocks[next_block].register_forward_pre_hook(modify_hook)
    try:
        with torch.no_grad():
            ablated_preds = clf.predict_proba(X_query)
    finally:
        handle.remove()

    return {
        "baseline_preds": np.asarray(baseline_preds),
        "ablated_preds": np.asarray(ablated_preds),
        "y_query": np.asarray(y_query),
    }


# ── TabDPT Intervention ──────────────────────────────────────────────────────


def intervene_tabdpt(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    sae: torch.nn.Module,
    ablate_features: List[int],
    extraction_layer: int,
    device: str = "cuda",
    task: str = "classification",
) -> Dict[str, np.ndarray]:
    """Run TabDPT with SAE feature ablation at extraction layer.

    TabDPT has 16 transformer encoder layers. Hidden state is 3D: (batch, seq, H).
    """
    from tabdpt import TabDPTClassifier, TabDPTRegressor

    if task == "regression":
        clf = TabDPTRegressor(device=device, compile=False)
    else:
        clf = TabDPTClassifier(device=device, compile=False)
    clf.fit(X_context, y_context)

    n_query = len(X_query)
    model = clf.model
    encoder_layers = model.transformer_encoder

    # --- Pass 1: Capture hidden state + baseline predictions ---
    captured = {}

    def capture_hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            captured["hidden"] = out.detach()

    handle = encoder_layers[extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            if task == "regression":
                baseline_preds = clf.predict(X_query)
            else:
                baseline_preds = clf.predict_proba(X_query)
    finally:
        handle.remove()

    hidden_state = captured["hidden"]
    if hidden_state.ndim == 3:
        # (batch, seq, H) → mean over seq, take last n_query
        query_emb = hidden_state.mean(dim=1)[-n_query:]
    elif hidden_state.ndim == 2:
        query_emb = hidden_state[-n_query:]
    else:
        raise ValueError(f"Unexpected hidden state shape: {hidden_state.shape}")

    # --- Compute delta ---
    delta = compute_ablation_delta(sae, query_emb, ablate_features)

    # --- Pass 2: Pre-hook on layer L+1 ---
    next_layer = extraction_layer + 1
    if next_layer >= len(encoder_layers):
        raise ValueError(
            f"Extraction layer {extraction_layer} is the last layer "
            f"({len(encoder_layers)} total)."
        )

    def modify_hook(module, args):
        inp = args[0] if isinstance(args, tuple) else args
        if isinstance(inp, torch.Tensor):
            modified = inp.clone()
            if modified.ndim == 3:
                # (batch, seq, H) — broadcast delta across seq dim
                modified[-n_query:] += delta.unsqueeze(1)
            elif modified.ndim == 2:
                modified[-n_query:] += delta
            return (modified,) + args[1:] if isinstance(args, tuple) else modified
        return args

    handle = encoder_layers[next_layer].register_forward_pre_hook(modify_hook)
    try:
        with torch.no_grad():
            if task == "regression":
                ablated_preds = clf.predict(X_query)
            else:
                ablated_preds = clf.predict_proba(X_query)
    finally:
        handle.remove()

    return {
        "baseline_preds": np.asarray(baseline_preds),
        "ablated_preds": np.asarray(ablated_preds),
        "y_query": np.asarray(y_query),
    }


# ── HyperFast Intervention ──────────────────────────────────────────────────


def intervene_hyperfast(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    sae: torch.nn.Module,
    ablate_features: List[int],
    extraction_layer: int,
    device: str = "cuda",
    task: str = "classification",
) -> Dict[str, np.ndarray]:
    """Run HyperFast with SAE feature ablation.

    HyperFast generates a task-specific MLP from context data.
    We manually forward through the generated weights, applying the SAE
    delta at the extraction layer.
    """
    from hyperfast.hyperfast import forward_main_network, transform_data_for_main_network
    from models.hyperfast_embeddings import HyperFastExtractor

    extractor = HyperFastExtractor(device=device)
    extractor.fit(X_context, y_context)
    clf = extractor._model

    n_query = len(X_query)
    X_query_t = torch.tensor(X_query, dtype=torch.float32).to(device)

    baseline_outputs = []
    ablated_outputs = []

    for jj in range(len(clf._main_networks)):
        main_network = clf._move_to_device(clf._main_networks[jj])
        rf = clf._move_to_device(clf._rfs[jj])
        pca = clf._move_to_device(clf._pcas[jj])

        if clf.feature_bagging:
            X_b = X_query_t[:, clf.selected_features[jj]]
        else:
            X_b = X_query_t

        X_transformed = transform_data_for_main_network(
            X=X_b, cfg=clf._cfg, rf=rf, pca=pca,
        )

        # --- Baseline forward (full network) ---
        with torch.no_grad():
            outputs_base, intermediate = forward_main_network(
                X_transformed, main_network,
            )
        baseline_outputs.append(F.softmax(outputs_base, dim=1).cpu().numpy())

        # --- Ablated forward: apply delta at extraction layer ---
        # Manual forward to the extraction layer
        with torch.no_grad():
            x = X_transformed
            for layer_idx, (weight, bias) in enumerate(main_network):
                weight = clf._move_to_device(weight)
                bias = clf._move_to_device(bias)
                x_new = F.linear(x, weight, bias)
                if layer_idx < len(main_network) - 1:
                    x_new = F.relu(x_new)
                    if x_new.shape[-1] == x.shape[-1]:
                        x = x + x_new
                    else:
                        x = x_new
                else:
                    # Last layer → output logits
                    x = x_new

                if layer_idx == extraction_layer:
                    # Apply SAE delta here
                    delta = compute_ablation_delta(sae, x, ablate_features)
                    x = x + delta

            ablated_outputs.append(F.softmax(x, dim=1).cpu().numpy())

    # Ensemble average
    baseline_avg = np.mean(baseline_outputs, axis=0)
    ablated_avg = np.mean(ablated_outputs, axis=0)

    return {
        "baseline_preds": baseline_avg,
        "ablated_preds": ablated_avg,
        "y_query": np.asarray(y_query),
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────

INTERVENE_FN = {
    "tabpfn": intervene_tabpfn,
    "mitra": intervene_mitra,
    "tabicl": intervene_tabicl,
    "tabdpt": intervene_tabdpt,
    "hyperfast": intervene_hyperfast,
}


def intervene(
    model_key: str,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    ablate_features: List[int],
    device: str = "cuda",
    task: str = "classification",
    sae_dir: Path = DEFAULT_SAE_DIR,
    layers_path: Path = DEFAULT_LAYERS_PATH,
) -> Dict[str, np.ndarray]:
    """Run a model with SAE feature ablation at its optimal extraction layer.

    Args:
        model_key: One of 'tabpfn', 'mitra', 'tabicl', 'tabdpt', 'hyperfast'
        X_context, y_context: ICL context data
        X_query, y_query: Query data (y_query for evaluation only)
        ablate_features: SAE feature indices to zero out
        device: Torch device
        task: 'classification' or 'regression'
        sae_dir: Path to SAE checkpoints
        layers_path: Path to optimal_extraction_layers.json

    Returns:
        Dict with 'baseline_preds', 'ablated_preds', 'y_query'
    """
    if model_key not in INTERVENE_FN:
        raise ValueError(f"Unsupported model: {model_key}. Choose from {list(INTERVENE_FN.keys())}")

    sae, _ = load_sae(model_key, sae_dir=sae_dir, device=device)
    extraction_layer = get_extraction_layer(model_key, layers_path=layers_path)

    return INTERVENE_FN[model_key](
        X_context=X_context,
        y_context=y_context,
        X_query=X_query,
        y_query=y_query,
        sae=sae,
        ablate_features=ablate_features,
        extraction_layer=extraction_layer,
        device=device,
        task=task,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="SAE feature ablation via intervention")
    parser.add_argument("--model", type=str, required=True, choices=list(INTERVENE_FN.keys()))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ablate-features", type=str, default="",
                        help="Comma-separated feature indices to ablate (empty=none)")
    parser.add_argument("--ablate-all", action="store_true",
                        help="Ablate all SAE features")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--context-size", type=int, default=600)
    parser.add_argument("--query-size", type=int, default=500)
    parser.add_argument("--verify-identity", action="store_true",
                        help="Verify that ablating nothing gives identical predictions")
    parser.add_argument("--sae-dir", type=Path, default=DEFAULT_SAE_DIR)
    parser.add_argument("--layers-config", type=Path, default=DEFAULT_LAYERS_PATH)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    from scripts.extract_layer_embeddings import load_context_query, get_dataset_task

    task = get_dataset_task(args.dataset)
    X_context, y_context, X_query = load_context_query(
        args.dataset, context_size=args.context_size, query_size=args.query_size,
    )
    # For evaluation, we need y_query. Re-load with y split.
    from data.extended_loader import TABARENA_DATASETS, load_tabarena_dataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    result = load_tabarena_dataset(args.dataset, max_samples=args.context_size + args.query_size)
    X, y, _ = result
    n = len(X)
    ctx_size = min(args.context_size, int(n * 0.7))
    q_size = min(args.query_size, n - ctx_size)

    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)
        query_frac = q_size / (ctx_size + q_size)
        try:
            X_ctx, X_q, y_ctx, y_q = train_test_split(
                X, y, test_size=query_frac, random_state=42, stratify=y,
            )
        except ValueError:
            X_ctx, X_q, y_ctx, y_q = train_test_split(
                X, y, test_size=query_frac, random_state=42,
            )
    else:
        X_ctx = X[:ctx_size]
        y_ctx = y[:ctx_size]
        X_q = X[ctx_size:ctx_size + q_size]
        y_q = y[ctx_size:ctx_size + q_size]

    X_context = X_ctx[:ctx_size]
    y_context = y_ctx[:ctx_size]
    X_query = X_q[:q_size]
    y_query = y_q[:q_size]

    # Parse ablation features
    if args.ablate_all:
        sae, config = load_sae(args.model, sae_dir=args.sae_dir, device=args.device)
        ablate_features = list(range(config.hidden_dim))
        del sae
    elif args.ablate_features.strip():
        ablate_features = [int(x.strip()) for x in args.ablate_features.split(",")]
    else:
        ablate_features = []

    print(f"Model: {args.model}, Dataset: {args.dataset}, Task: {task}")
    print(f"Context: {len(X_context)}, Query: {len(X_query)}")
    print(f"Ablating {len(ablate_features)} features")

    results = intervene(
        model_key=args.model,
        X_context=X_context,
        y_context=y_context,
        X_query=X_query,
        y_query=y_query,
        ablate_features=ablate_features,
        device=args.device,
        task=task,
        sae_dir=args.sae_dir,
        layers_path=args.layers_config,
    )

    # Evaluate
    if task == "classification":
        baseline_acc = np.mean(results["baseline_preds"].argmax(axis=1) == results["y_query"])
        ablated_acc = np.mean(results["ablated_preds"].argmax(axis=1) == results["y_query"])
        print(f"\nBaseline accuracy: {baseline_acc:.4f}")
        print(f"Ablated accuracy:  {ablated_acc:.4f}")
        print(f"Drop:              {baseline_acc - ablated_acc:.4f}")
    else:
        from sklearn.metrics import r2_score
        baseline_r2 = r2_score(results["y_query"], results["baseline_preds"])
        ablated_r2 = r2_score(results["y_query"], results["ablated_preds"])
        print(f"\nBaseline R²: {baseline_r2:.4f}")
        print(f"Ablated R²:  {ablated_r2:.4f}")
        print(f"Drop:        {baseline_r2 - ablated_r2:.4f}")

    if args.verify_identity and not ablate_features:
        diff = np.abs(results["baseline_preds"] - results["ablated_preds"]).max()
        if diff < 1e-5:
            print(f"\nIdentity check PASSED (max diff: {diff:.2e})")
        else:
            print(f"\nIdentity check FAILED (max diff: {diff:.2e})")
            sys.exit(1)


if __name__ == "__main__":
    main()
