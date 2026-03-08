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
DEFAULT_TRAINING_DIR = PROJECT_ROOT / "output" / "sae_training_round5"
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


def load_training_mean(
    model_key: str,
    training_dir: Path = DEFAULT_TRAINING_DIR,
    layers_path: Path = DEFAULT_LAYERS_PATH,
    device: str = "cuda",
) -> torch.Tensor:
    """Load the training pool mean used to center data before SAE training.

    train_sae() centers data via X_centered = X - X.mean(dim=0), but doesn't
    save the mean. We recompute it from the training .npz file.

    Returns:
        (emb_dim,) tensor on device
    """
    layer = get_extraction_layer(model_key, layers_path)
    training_path = training_dir / f"{model_key}_layer{layer}_sae_training.npz"
    if not training_path.exists():
        raise FileNotFoundError(
            f"SAE training data not found: {training_path}. "
            f"Cannot compute centering mean for intervention."
        )
    data = np.load(training_path)
    mean = data["embeddings"].mean(axis=0)
    return torch.tensor(mean, dtype=torch.float32, device=device)


def compute_ablation_delta(
    sae: torch.nn.Module,
    embeddings: torch.Tensor,
    ablate_features: List[int],
    data_mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the delta from ablating SAE features.

    The SAE was trained on mean-centered embeddings. We must center before
    encoding, then the delta (decode(ablated) - decode(original)) is applied
    in the original (uncentered) space — centering cancels in the subtraction.

    Args:
        sae: Trained SAE in eval mode
        embeddings: (n_query, emb_dim) mean-pooled query embeddings (raw, uncentered)
        ablate_features: Feature indices to zero out
        data_mean: (emb_dim,) training pool mean for centering. If None, no centering.

    Returns:
        delta: (n_query, emb_dim) to add to hidden states
    """
    with torch.no_grad():
        x = embeddings
        if data_mean is not None:
            x = x - data_mean

        h = sae.encode(x)
        original_recon = sae.decode(h)

        h_ablated = h.clone()
        if ablate_features:
            h_ablated[:, ablate_features] = 0.0
        ablated_recon = sae.decode(h_ablated)

        delta = ablated_recon - original_recon

    return delta


def compute_boost_delta(
    sae: torch.nn.Module,
    embeddings: torch.Tensor,
    boost_features: List[int],
    target_activations: List[float],
    data_mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the delta from boosting SAE features to target activation levels.

    Like compute_ablation_delta but sets features to specified target values
    instead of zeroing them. Boosting to 0.0 is equivalent to ablation.

    Args:
        sae: Trained SAE in eval mode
        embeddings: (n_query, emb_dim) mean-pooled query embeddings (raw, uncentered)
        boost_features: Feature indices to modify
        target_activations: Target activation value for each feature
        data_mean: (emb_dim,) training pool mean for centering. If None, no centering.

    Returns:
        delta: (n_query, emb_dim) to add to hidden states
    """
    if len(boost_features) != len(target_activations):
        raise ValueError(
            f"boost_features ({len(boost_features)}) and "
            f"target_activations ({len(target_activations)}) must have same length"
        )

    with torch.no_grad():
        x = embeddings
        if data_mean is not None:
            x = x - data_mean

        h = sae.encode(x)
        original_recon = sae.decode(h)

        h_boosted = h.clone()
        for feat, target in zip(boost_features, target_activations):
            h_boosted[:, feat] = target
        boosted_recon = sae.decode(h_boosted)

        delta = boosted_recon - original_recon

    return delta


# ── Tail Models ─────────────────────────────────────────────────────────────
# Cache the forward pass up to layer L once, then re-run only layers L+1..end
# + prediction head for each intervention. Avoids redundant computation of the
# identical prefix layers on every forward pass.


class TabPFNTail:
    """Cached TabPFN tail: injects delta at layer L via hook, runs predict.

    Caches the hidden state at layer L from the first forward pass.
    On predict(), installs a hook that replaces the layer L output with the
    cached state + delta, so layers 0..L run but their output is discarded.
    The full predict_proba pipeline (BarDistribution etc.) handles logit→prob.

    Usage:
        tail = TabPFNTail.from_data(X_ctx, y_ctx, X_query, layer=17, ...)
        preds = tail.predict(delta)  # delta: (seq_len, hidden_dim)
        preds = tail.predict_row(row_idx, delta_row)  # delta_row: (hidden_dim,)
    """

    def __init__(self, clf, layers, hidden_state, single_eval_pos,
                 extraction_layer, n_query, X_query, task, device):
        self.clf = clf
        self.layers = layers
        self.hidden_state = hidden_state  # (1, seq_len, n_structure, hidden_dim)
        self.single_eval_pos = single_eval_pos
        self.extraction_layer = extraction_layer
        self.n_query = n_query
        self.X_query = X_query
        self.task = task
        self.device = device

    @classmethod
    def from_data(cls, X_context, y_context, X_query, extraction_layer,
                  task="classification", device="cuda"):
        """One-time setup: fit model, capture hidden state at layer L, get baseline."""
        from models.tabpfn_utils import load_tabpfn

        clf = load_tabpfn(task=task, device=device, n_estimators=1)
        clf.fit(X_context, y_context)
        model = clf.model_
        layers = model.transformer_encoder.layers

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

        n_query = len(X_query)
        single_eval_pos = captured["hidden"].shape[1] - n_query

        tail = cls(
            clf=clf, layers=layers,
            hidden_state=captured["hidden"],
            single_eval_pos=single_eval_pos,
            extraction_layer=extraction_layer,
            n_query=n_query, X_query=X_query,
            task=task, device=device,
        )
        tail.baseline_preds = np.asarray(baseline_preds)
        return tail

    def _predict_with_modified_state(self, modified_state):
        """Install hook that replaces layer L output, run predict."""
        def replace_hook(module, input, output):
            return modified_state

        handle = self.layers[self.extraction_layer].register_forward_hook(replace_hook)
        try:
            with torch.no_grad():
                if self.task == "regression":
                    preds = self.clf.predict(self.X_query)
                else:
                    preds = self.clf.predict_proba(self.X_query)
        finally:
            handle.remove()
        return np.asarray(preds)

    def predict(self, delta):
        """Full-batch intervention: delta shape (seq_len, hidden_dim)."""
        state = self.hidden_state.clone()
        delta_broadcast = delta.unsqueeze(1)  # (seq_len, 1, hidden_dim)
        state[0] += delta_broadcast
        return self._predict_with_modified_state(state)

    def predict_row(self, row_idx, delta_row):
        """Single-row intervention: delta_row shape (hidden_dim,).

        Modifies only query row `row_idx` (0-indexed within query set).
        """
        state = self.hidden_state.clone()
        seq_idx = self.single_eval_pos + row_idx
        state[0, seq_idx, :, :] += delta_row.unsqueeze(0)
        return self._predict_with_modified_state(state)


class TabICLTail:
    """Cached TabICL tail: injects delta at block L via hook, runs predict.

    Usage:
        tail = TabICLTail.from_data(X_ctx, y_ctx, X_query, layer=8, device="cuda")
        preds = tail.predict(delta)
        preds = tail.predict_row(row_idx, delta_row)
    """

    def __init__(self, clf, blocks, hidden_state, train_size,
                 extraction_layer, n_query, X_query, device):
        self.clf = clf
        self.blocks = blocks
        self.hidden_state = hidden_state  # (n_ensemble, seq_len, 512)
        self.train_size = train_size
        self.extraction_layer = extraction_layer
        self.n_query = n_query
        self.X_query = X_query
        self.device = device

    @classmethod
    def from_data(cls, X_context, y_context, X_query, extraction_layer,
                  device="cuda"):
        """One-time setup: fit model, capture hidden state at layer L."""
        from tabicl import TabICLClassifier

        clf = TabICLClassifier(device=device, n_estimators=1)
        clf.fit(X_context, y_context)

        model = clf.model_
        blocks = model.icl_predictor.tf_icl.blocks

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

        n_query = len(X_query)
        train_size = captured["hidden"].shape[1] - n_query

        tail = cls(
            clf=clf, blocks=blocks,
            hidden_state=captured["hidden"],
            train_size=train_size,
            extraction_layer=extraction_layer,
            n_query=n_query, X_query=X_query, device=device,
        )
        tail.baseline_preds = np.asarray(baseline_preds)
        return tail

    def _predict_with_modified_state(self, modified_state):
        """Install hook that replaces block L output, run predict."""
        def replace_hook(module, input, output):
            return modified_state

        handle = self.blocks[self.extraction_layer].register_forward_hook(replace_hook)
        try:
            with torch.no_grad():
                preds = self.clf.predict_proba(self.X_query)
        finally:
            handle.remove()
        return np.asarray(preds)

    def predict(self, delta):
        """Full-batch intervention: delta shape (seq_len, hidden_dim)."""
        state = self.hidden_state.clone()
        delta_broadcast = delta.unsqueeze(0)  # (1, seq_len, hidden_dim)
        state += delta_broadcast
        return self._predict_with_modified_state(state)

    def predict_row(self, row_idx, delta_row):
        """Single-row intervention: delta_row shape (hidden_dim,).

        Modifies only query row `row_idx`.
        """
        state = self.hidden_state.clone()
        seq_idx = self.train_size + row_idx
        state[:, seq_idx, :] += delta_row.unsqueeze(0)
        return self._predict_with_modified_state(state)


def build_tail(model_key, X_context, y_context, X_query, extraction_layer,
               task="classification", device="cuda"):
    """Factory: build the appropriate tail model for the given model key."""
    if model_key == "tabpfn":
        return TabPFNTail.from_data(
            X_context, y_context, X_query, extraction_layer, task, device,
        )
    elif model_key == "tabicl":
        return TabICLTail.from_data(
            X_context, y_context, X_query, extraction_layer, device,
        )
    else:
        raise ValueError(f"Tail model not implemented for {model_key}")


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
    data_mean: Optional[torch.Tensor] = None,
    external_delta: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Run TabPFN with SAE feature ablation at extraction layer.

    Ablates ALL positions (context + query) at layer L. If a concept matters
    for predicting query samples, it matters equally for representing context
    samples the model attends to. Ablating only query lets the model recover
    via attention to intact context representations.

    If external_delta is provided, skip SAE delta computation and inject it
    directly (used by concept transfer from another model's SAE space).

    Returns dict with: baseline_preds, ablated_preds, y_query
    """
    from models.tabpfn_utils import load_tabpfn

    clf = load_tabpfn(task=task, device=device, n_estimators=1)
    clf.fit(X_context, y_context)
    model = clf.model_
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
    # Shape: (1, seq_len, n_structure, hidden_dim) where seq_len = n_ctx + n_query
    # Mean-pool structure dim for ALL positions (context + query)
    all_emb = hidden_state[0].mean(dim=1)  # (seq_len, hidden)

    # --- Compute delta for all positions ---
    if external_delta is not None:
        delta = external_delta
    else:
        delta = compute_ablation_delta(sae, all_emb, ablate_features, data_mean=data_mean)
    delta_broadcast = delta.unsqueeze(1)  # (seq_len, 1, hidden)

    # --- Pass 2: Inject delta at layer L output for all positions ---
    def modify_output_hook(module, input, output):
        if isinstance(output, torch.Tensor) and output.ndim == 4:
            output = output.clone()
            output[0] += delta_broadcast
            return output
        return output

    handle = layers[extraction_layer].register_forward_hook(modify_output_hook)
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
    data_mean: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Run Mitra with SAE feature ablation at extraction layer.

    Mitra's Tab2D layers return a (support, query) tuple — support is the context
    tensor, query is the prediction tensor. Both must be captured and modified.
    Each tensor is 4D: (1, n_samples, n_feat+1, dim) with y-token at position 0.
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

    # Save RNG state — Mitra's DatasetFinetune uses trainer.rng for batching,
    # so the RNG must be at the same state for both passes to get identical batches.
    rng_state = trainer.rng.get_state()

    # --- Pass 1: Capture hidden state + baseline predictions ---
    # Mitra layers return (support_tensor, query_tensor) tuples.
    captured_support = []
    captured_query = []

    def capture_hook(module, input, output):
        if isinstance(output, tuple) and len(output) >= 2:
            sup, qry = output[0], output[1]
            if isinstance(sup, torch.Tensor):
                captured_support.append(sup.detach())
            if isinstance(qry, torch.Tensor):
                captured_query.append(qry.detach())

    handle = layers[extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            if task == "regression":
                baseline_preds = clf.predict(X_query)
            else:
                baseline_preds = clf.predict_proba(X_query)
    finally:
        handle.remove()

    # Extract y-token embeddings from both support and query tensors
    def extract_y_tokens(tensor_list):
        embs = []
        for h in tensor_list:
            if h.ndim == 4:
                embs.append(h[0, :, 0, :])  # (n_batch, dim)
            elif h.ndim == 3:
                embs.append(h.squeeze(0))
            elif h.ndim == 2:
                embs.append(h)
        return torch.cat(embs, dim=0) if embs else None

    support_emb = extract_y_tokens(captured_support)
    query_emb = extract_y_tokens(captured_query)

    # Compute delta for support and query separately
    delta_support = compute_ablation_delta(sae, support_emb, ablate_features, data_mean=data_mean)
    delta_query = compute_ablation_delta(sae, query_emb, ablate_features, data_mean=data_mean)

    # --- Pass 2: Inject delta at layer L output for both support and query ---
    # Restore RNG state so batching is identical to pass 1
    trainer.rng.set_state(rng_state)
    sup_offset = [0]
    qry_offset = [0]

    def modify_output_hook(module, input, output):
        if not (isinstance(output, tuple) and len(output) >= 2):
            return output

        sup, qry = output[0], output[1]
        modified = list(output)

        # Modify support tensor
        if isinstance(sup, torch.Tensor) and sup.ndim == 4:
            sup = sup.clone()
            n_sup = sup.shape[1]
            s = sup_offset[0]
            sup[0] += delta_support[s:s + n_sup].unsqueeze(1)
            sup_offset[0] = s + n_sup
            modified[0] = sup

        # Modify query tensor
        if isinstance(qry, torch.Tensor) and qry.ndim == 4:
            qry = qry.clone()
            n_qry = qry.shape[1]
            s = qry_offset[0]
            qry[0] += delta_query[s:s + n_qry].unsqueeze(1)
            qry_offset[0] = s + n_qry
            modified[1] = qry

        return tuple(modified)

    handle = layers[extraction_layer].register_forward_hook(modify_output_hook)
    try:
        with torch.no_grad():
            sup_offset[0] = 0
            qry_offset[0] = 0
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
    data_mean: Optional[torch.Tensor] = None,
    external_delta: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Run TabICL with SAE feature ablation at extraction layer.

    TabICL has 8 ICL predictor blocks. Hidden state is 3D: (n_ensemble, seq, 512).
    Ablates ALL positions (context + query).

    Uses BATCH-MEAN centering instead of training-mean centering because TabICL's
    column-then-row architecture creates highly dataset-specific representations.
    Per-dataset means are orthogonal to the pooled training mean (cosine~0.02),
    making the pooled SAE unable to reconstruct with training-mean centering
    (R²=-1.2). Batch-mean centering gives R²=0.35 and genuine (though weaker)
    ablation effects that consistently outperform random noise controls.

    If external_delta is provided, skip SAE delta computation and inject it
    directly (used by concept transfer from another model's SAE space).
    """
    from tabicl import TabICLClassifier

    clf = TabICLClassifier(device=device, n_estimators=1)
    clf.fit(X_context, y_context)

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
    # Mean-pool ensemble dim for ALL positions (context + query)
    all_emb = hidden_state.mean(dim=0)  # (seq_len, 512)

    # Use batch-mean centering: TabICL per-dataset means are orthogonal to the
    # pooled training mean, so training-mean centering produces garbage R²=-1.2.
    # Batch-mean gives R²=0.35 with genuine (small) ablation signal.
    batch_mean = all_emb.mean(dim=0)  # (512,)

    # --- Compute delta for all positions ---
    if external_delta is not None:
        delta = external_delta
    else:
        delta = compute_ablation_delta(sae, all_emb, ablate_features, data_mean=batch_mean)
    delta_broadcast = delta.unsqueeze(0)  # (1, seq_len, 512)

    # --- Pass 2: Inject delta at block L output for all positions ---
    def modify_output_hook(module, input, output):
        if isinstance(output, torch.Tensor) and output.ndim == 3:
            output = output.clone()
            output += delta_broadcast
            return output
        return output

    handle = blocks[extraction_layer].register_forward_hook(modify_output_hook)
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
    data_mean: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Run TabDPT with SAE feature ablation at extraction layer.

    TabDPT has 16 transformer encoder layers. Hidden state is 3D: (batch, seq, H).
    Ablates ALL positions (context + query).
    """
    from tabdpt import TabDPTClassifier, TabDPTRegressor

    if task == "regression":
        clf = TabDPTRegressor(device=device, compile=False)
    else:
        clf = TabDPTClassifier(device=device, compile=False)
    clf.fit(X_context, y_context)

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
        # (n_samples, seq, H) → mean over seq for ALL positions
        all_emb = hidden_state.mean(dim=1)  # (n_samples, H)
    elif hidden_state.ndim == 2:
        all_emb = hidden_state
    else:
        raise ValueError(f"Unexpected hidden state shape: {hidden_state.shape}")

    # --- Compute delta for all positions ---
    delta = compute_ablation_delta(sae, all_emb, ablate_features, data_mean=data_mean)

    # --- Pass 2: Inject delta at layer L output for all positions ---
    def modify_output_hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            out = out.clone()
            if out.ndim == 3:
                out += delta.unsqueeze(1)
            elif out.ndim == 2:
                out += delta
            if isinstance(output, tuple):
                return (out,) + output[1:]
            return out
        return output

    handle = encoder_layers[extraction_layer].register_forward_hook(modify_output_hook)
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
    data_mean: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Run HyperFast with SAE feature ablation.

    HyperFast generates a task-specific MLP from context data.
    We manually forward through the generated weights, applying the SAE
    delta at the extraction layer.
    """
    from hyperfast.hyperfast import forward_main_network, transform_data_for_main_network
    from models.hyperfast_embeddings import HyperFastEmbeddingExtractor

    extractor = HyperFastEmbeddingExtractor(device=device)
    extractor.load_model()
    X_ctx_clean = np.nan_to_num(np.asarray(X_context, dtype=np.float32), nan=0.0)
    y_ctx_clean = np.asarray(y_context, dtype=np.int64)
    extractor._model.fit(X_ctx_clean, y_ctx_clean)
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
                    delta = compute_ablation_delta(sae, x, ablate_features, data_mean=data_mean)
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


# ── Sweep (one model load, many ablation levels) ─────────────────────────────


def sweep_intervene_tabpfn(
    X_context, y_context, X_query, y_query, sae,
    feature_lists, extraction_layer, device, task, data_mean,
):
    """Sweep multiple ablation levels for TabPFN (one load, one capture)."""
    from models.tabpfn_utils import load_tabpfn

    clf = load_tabpfn(task=task, device=device, n_estimators=1)
    clf.fit(X_context, y_context)
    layers = clf.model_.transformer_encoder.layers

    captured = {}

    def capture_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured["hidden"] = output.detach()

    handle = layers[extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            baseline_preds = clf.predict(X_query) if task == "regression" else clf.predict_proba(X_query)
    finally:
        handle.remove()

    all_emb = captured["hidden"][0].mean(dim=1)  # (seq_len, hidden)

    results = []
    for features in feature_lists:
        delta = compute_ablation_delta(sae, all_emb, features, data_mean=data_mean)
        delta_broadcast = delta.unsqueeze(1)

        def make_hook(db):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor) and output.ndim == 4:
                    output = output.clone()
                    output[0] += db
                    return output
                return output
            return hook

        handle = layers[extraction_layer].register_forward_hook(make_hook(delta_broadcast))
        try:
            with torch.no_grad():
                preds = clf.predict(X_query) if task == "regression" else clf.predict_proba(X_query)
        finally:
            handle.remove()
        results.append(np.asarray(preds))

    return np.asarray(baseline_preds), results


def sweep_intervene_tabicl(
    X_context, y_context, X_query, y_query, sae,
    feature_lists, extraction_layer, device, task, data_mean,
):
    """Sweep multiple ablation levels for TabICL (one load, one capture)."""
    from tabicl import TabICLClassifier

    clf = TabICLClassifier(device=device, n_estimators=1)
    clf.fit(X_context, y_context)
    blocks = clf.model_.icl_predictor.tf_icl.blocks

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

    all_emb = captured["hidden"].mean(dim=0)  # (seq_len, 512)
    batch_mean = all_emb.mean(dim=0)

    results = []
    for features in feature_lists:
        delta = compute_ablation_delta(sae, all_emb, features, data_mean=batch_mean)
        delta_broadcast = delta.unsqueeze(0)

        def make_hook(db):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor) and output.ndim == 3:
                    output = output.clone()
                    output += db
                    return output
                return output
            return hook

        handle = blocks[extraction_layer].register_forward_hook(make_hook(delta_broadcast))
        try:
            with torch.no_grad():
                preds = clf.predict_proba(X_query)
        finally:
            handle.remove()
        results.append(np.asarray(preds))

    return np.asarray(baseline_preds), results


def sweep_intervene_tabdpt(
    X_context, y_context, X_query, y_query, sae,
    feature_lists, extraction_layer, device, task, data_mean,
):
    """Sweep multiple ablation levels for TabDPT (one load, one capture)."""
    from tabdpt import TabDPTClassifier, TabDPTRegressor

    clf = TabDPTRegressor(device=device, compile=False) if task == "regression" \
        else TabDPTClassifier(device=device, compile=False)
    clf.fit(X_context, y_context)
    encoder_layers = clf.model.transformer_encoder

    captured = {}

    def capture_hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            captured["hidden"] = out.detach()

    handle = encoder_layers[extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            baseline_preds = clf.predict(X_query) if task == "regression" else clf.predict_proba(X_query)
    finally:
        handle.remove()

    hidden = captured["hidden"]
    all_emb = hidden.mean(dim=1) if hidden.ndim == 3 else hidden

    results = []
    for features in feature_lists:
        delta = compute_ablation_delta(sae, all_emb, features, data_mean=data_mean)

        def make_hook(d):
            def hook(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                if isinstance(out, torch.Tensor):
                    out = out.clone()
                    out += d.unsqueeze(1) if out.ndim == 3 else d
                    return (out,) + output[1:] if isinstance(output, tuple) else out
                return output
            return hook

        handle = encoder_layers[extraction_layer].register_forward_hook(make_hook(delta))
        try:
            with torch.no_grad():
                preds = clf.predict(X_query) if task == "regression" else clf.predict_proba(X_query)
        finally:
            handle.remove()
        results.append(np.asarray(preds))

    return np.asarray(baseline_preds), results


SWEEP_FN = {
    "tabpfn": sweep_intervene_tabpfn,
    "tabicl": sweep_intervene_tabicl,
    "tabdpt": sweep_intervene_tabdpt,
}


# ── Per-row heterogeneous sweep ───────────────────────────────────────────────


def compute_ablation_delta_perrow(
    sae: torch.nn.Module,
    embeddings: torch.Tensor,
    feature_masks: torch.Tensor,
    data_mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-row ablation deltas with different features zeroed per row.

    Args:
        sae: Trained SAE in eval mode
        embeddings: (n_rows, emb_dim) raw embeddings
        feature_masks: (n_rows, sae_hidden) boolean, True = ABLATE this feature
        data_mean: (emb_dim,) centering mean

    Returns:
        delta: (n_rows, emb_dim) per-row deltas
    """
    with torch.no_grad():
        x = embeddings - data_mean if data_mean is not None else embeddings
        h = sae.encode(x)
        original_recon = sae.decode(h)
        h_ablated = h.clone()
        h_ablated[feature_masks] = 0.0
        ablated_recon = sae.decode(h_ablated)
        return ablated_recon - original_recon


def compute_perrow_logloss(
    preds: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """Compute per-row binary logloss.

    Args:
        preds: P(class=1) predictions, shape (n,) or (n, 2).
        y_true: Ground-truth labels, shape (n,).

    Returns:
        Per-row logloss, shape (n,).
    """
    eps = 1e-7
    p1 = preds[:, 1] if preds.ndim == 2 else preds
    p = np.clip(p1, eps, 1 - eps)
    y = y_true.astype(float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def get_improvable_rows(
    preds_strong: np.ndarray,
    preds_weak: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """Identify rows where the strong model outperforms the weak model.

    Uses per-row logloss comparison: strong model has lower logloss on this row.
    Shared by both ablation and transfer pipelines to ensure consistent row
    selection across all plots and analyses.

    Args:
        preds_strong: Strong model's P(class=1), shape (n,) or (n, 2).
        preds_weak: Weak model's P(class=1), shape (n,) or (n, 2).
        y_true: Ground-truth labels, shape (n,).

    Returns:
        Boolean mask, shape (n,). True where strong model has lower logloss.
    """
    ll_strong = compute_perrow_logloss(preds_strong, y_true)
    ll_weak = compute_perrow_logloss(preds_weak, y_true)
    return ll_strong < ll_weak


def _perrow_importance(
    baseline_preds: np.ndarray,
    individual_preds: List[np.ndarray],
    y_query: np.ndarray,
) -> np.ndarray:
    """Compute per-row importance matrix from individual feature ablation predictions.

    Returns:
        (n_features, n_query) array. Positive = ablation increased logloss (feature helpful).
    """
    y = y_query.astype(float)
    eps = 1e-7
    bp1 = baseline_preds[:, 1] if baseline_preds.ndim == 2 else baseline_preds
    bp = np.clip(bp1, eps, 1 - eps)
    base_ll = -(y * np.log(bp) + (1 - y) * np.log(1 - bp))

    n_feat = len(individual_preds)
    n_query = len(y_query)
    importance = np.zeros((n_feat, n_query))

    for i, preds in enumerate(individual_preds):
        ap1 = preds[:, 1] if preds.ndim == 2 else preds
        ap = np.clip(ap1, eps, 1 - eps)
        abl_ll = -(y * np.log(ap) + (1 - y) * np.log(1 - ap))
        importance[i] = abl_ll - base_ll

    return importance


def _perrow_rankings(
    importance: np.ndarray,
    feature_indices: List[int],
    query_activations: np.ndarray,
) -> List[List[int]]:
    """Build per-row feature rankings: only firing features, sorted by per-row importance.

    Returns:
        List of n_query lists, each containing feature indices sorted by importance desc.
    """
    _, n_query = importance.shape
    rankings = []
    for row in range(n_query):
        row_feats = []
        for i, feat_idx in enumerate(feature_indices):
            if query_activations[row, feat_idx] > 0:
                row_feats.append((feat_idx, importance[i, row]))
        row_feats.sort(key=lambda x: -x[1])
        rankings.append([f for f, _ in row_feats])
    return rankings


def perrow_sweep_intervene_tabpfn(
    X_context, y_context, X_query, y_query, sae,
    unmatched_features, extraction_layer, device, task, data_mean,
):
    """Per-row heterogeneous ablation sweep for TabPFN.

    Phase 1: individual feature importance (N forward passes)
    Phase 2: per-row ranking (sort by per-row importance, only firing features)
    Phase 3: heterogeneous sweep (max_k forward passes, per-row delta at each k)
    """
    from models.tabpfn_utils import load_tabpfn

    clf = load_tabpfn(task=task, device=device, n_estimators=1)
    clf.fit(X_context, y_context)
    layers = clf.model_.transformer_encoder.layers
    n_ctx = len(X_context)
    n_query = len(X_query)

    # --- Capture hidden state + baseline ---
    captured = {}

    def capture_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured["hidden"] = output.detach()

    handle = layers[extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            baseline_preds = clf.predict(X_query) if task == "regression" \
                else clf.predict_proba(X_query)
    finally:
        handle.remove()

    all_emb = captured["hidden"][0].mean(dim=1)  # (seq_len, hidden)
    # Use -n_query indexing: TabPFN seq_len may differ from len(X_ctx)+len(X_query)
    query_emb = all_emb[-n_query:]   # last n_query positions = query rows
    ctx_emb = all_emb[:-n_query]     # everything before = context

    # SAE encode query rows for activation detection
    with torch.no_grad():
        x_centered = query_emb - data_mean if data_mean is not None else query_emb
        h_encoded = sae.encode(x_centered)
    query_acts = h_encoded.cpu().numpy()  # (n_query, sae_hidden)

    # --- Phase 1: per-feature importance ---
    logger.info("Phase 1: per-row importance for %d features (%d forward passes)...",
                len(unmatched_features), len(unmatched_features))
    individual_preds = []
    for feat_idx in unmatched_features:
        delta = compute_ablation_delta(sae, all_emb, [feat_idx], data_mean=data_mean)
        db = delta.unsqueeze(1)

        def make_hook(d):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor) and output.ndim == 4:
                    out = output.clone()
                    out[0] += d
                    return out
                return output
            return hook

        h = layers[extraction_layer].register_forward_hook(make_hook(db))
        try:
            with torch.no_grad():
                preds = clf.predict(X_query) if task == "regression" \
                    else clf.predict_proba(X_query)
        finally:
            h.remove()
        individual_preds.append(np.asarray(preds))

    baseline_np = np.asarray(baseline_preds)
    importance = _perrow_importance(baseline_np, individual_preds, y_query)
    rankings = _perrow_rankings(importance, unmatched_features, query_acts)

    max_k = max((len(r) for r in rankings), default=0)
    logger.info("Phase 2: rankings built. Max firing features/row: %d", max_k)

    # --- Phase 3: heterogeneous sweep ---
    logger.info("Phase 3: heterogeneous sweep k=1..%d...", max_k)
    sae_hidden = h_encoded.shape[1]
    sweep_preds = []

    for k in range(1, max_k + 1):
        # Context: ablate union of all features at this k
        all_feats_k = set()
        for r in rankings:
            all_feats_k.update(r[:k])
        ctx_delta = compute_ablation_delta(
            sae, ctx_emb, list(all_feats_k), data_mean=data_mean,
        )

        # Query: per-row ablation via feature masks
        masks = torch.zeros(n_query, sae_hidden, dtype=torch.bool, device=device)
        for row_idx in range(n_query):
            feats = rankings[row_idx][:k]
            if feats:
                masks[row_idx, feats] = True
        query_delta = compute_ablation_delta_perrow(
            sae, query_emb, masks, data_mean=data_mean,
        )

        combined = torch.cat([ctx_delta, query_delta], dim=0).unsqueeze(1)

        def make_hook(d):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor) and output.ndim == 4:
                    out = output.clone()
                    out[0] += d
                    return out
                return output
            return hook

        h = layers[extraction_layer].register_forward_hook(make_hook(combined))
        try:
            with torch.no_grad():
                preds = clf.predict(X_query) if task == "regression" \
                    else clf.predict_proba(X_query)
        finally:
            h.remove()
        sweep_preds.append(np.asarray(preds))

    return {
        "baseline_preds": baseline_np,
        "perrow_importance": importance,
        "perrow_rankings": rankings,
        "sweep_preds": sweep_preds,
        "max_k_per_row": np.array([len(r) for r in rankings]),
        "query_activations": query_acts,
        "unmatched_features": list(unmatched_features),
    }


def perrow_sweep_intervene_tabicl(
    X_context, y_context, X_query, y_query, sae,
    unmatched_features, extraction_layer, device, task, data_mean,
):
    """Per-row heterogeneous ablation sweep for TabICL.

    Same structure as TabPFN version but with TabICL-specific hook patterns
    and batch-mean centering.
    """
    from tabicl import TabICLClassifier

    clf = TabICLClassifier(device=device, n_estimators=1)
    clf.fit(X_context, y_context)
    blocks = clf.model_.icl_predictor.tf_icl.blocks
    n_ctx = len(X_context)
    n_query = len(X_query)

    # --- Capture hidden state + baseline ---
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

    all_emb = captured["hidden"].mean(dim=0)  # (seq_len, 512)
    batch_mean = all_emb.mean(dim=0)  # batch-mean centering for TabICL
    # Use -n_query indexing: seq_len may differ from len(X_ctx)+len(X_query)
    query_emb = all_emb[-n_query:]
    ctx_emb = all_emb[:-n_query]

    # SAE encode query rows for activation detection
    with torch.no_grad():
        x_centered = query_emb - batch_mean
        h_encoded = sae.encode(x_centered)
    query_acts = h_encoded.cpu().numpy()  # (n_query, sae_hidden)

    # --- Phase 1: per-feature importance ---
    logger.info("Phase 1: per-row importance for %d features (%d forward passes)...",
                len(unmatched_features), len(unmatched_features))
    individual_preds = []
    for feat_idx in unmatched_features:
        delta = compute_ablation_delta(sae, all_emb, [feat_idx], data_mean=batch_mean)
        db = delta.unsqueeze(0)  # (1, seq_len, 512) broadcast over ensemble

        def make_hook(d):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor) and output.ndim == 3:
                    out = output.clone()
                    out += d
                    return out
                return output
            return hook

        h = blocks[extraction_layer].register_forward_hook(make_hook(db))
        try:
            with torch.no_grad():
                preds = clf.predict_proba(X_query)
        finally:
            h.remove()
        individual_preds.append(np.asarray(preds))

    baseline_np = np.asarray(baseline_preds)
    importance = _perrow_importance(baseline_np, individual_preds, y_query)
    rankings = _perrow_rankings(importance, unmatched_features, query_acts)

    max_k = max((len(r) for r in rankings), default=0)
    logger.info("Phase 2: rankings built. Max firing features/row: %d", max_k)

    # --- Phase 3: heterogeneous sweep ---
    logger.info("Phase 3: heterogeneous sweep k=1..%d...", max_k)
    sae_hidden = h_encoded.shape[1]
    sweep_preds = []

    for k in range(1, max_k + 1):
        # Context: ablate union of all features at this k
        all_feats_k = set()
        for r in rankings:
            all_feats_k.update(r[:k])
        ctx_delta = compute_ablation_delta(
            sae, ctx_emb, list(all_feats_k), data_mean=batch_mean,
        )

        # Query: per-row ablation via feature masks
        masks = torch.zeros(n_query, sae_hidden, dtype=torch.bool, device=device)
        for row_idx in range(n_query):
            feats = rankings[row_idx][:k]
            if feats:
                masks[row_idx, feats] = True
        query_delta = compute_ablation_delta_perrow(
            sae, query_emb, masks, data_mean=batch_mean,
        )

        combined = torch.cat([ctx_delta, query_delta], dim=0).unsqueeze(0)

        def make_hook(d):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor) and output.ndim == 3:
                    out = output.clone()
                    out += d
                    return out
                return output
            return hook

        h = blocks[extraction_layer].register_forward_hook(make_hook(combined))
        try:
            with torch.no_grad():
                preds = clf.predict_proba(X_query)
        finally:
            h.remove()
        sweep_preds.append(np.asarray(preds))

    return {
        "baseline_preds": baseline_np,
        "perrow_importance": importance,
        "perrow_rankings": rankings,
        "sweep_preds": sweep_preds,
        "max_k_per_row": np.array([len(r) for r in rankings]),
        "query_activations": query_acts,
        "unmatched_features": list(unmatched_features),
    }


PERROW_SWEEP_FN = {
    "tabpfn": perrow_sweep_intervene_tabpfn,
    "tabicl": perrow_sweep_intervene_tabicl,
}


def perrow_sweep_intervene(
    model_key: str,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    unmatched_features: List[int],
    device: str = "cuda",
    task: str = "classification",
    sae_dir: Path = DEFAULT_SAE_DIR,
    layers_path: Path = DEFAULT_LAYERS_PATH,
    training_dir: Path = DEFAULT_TRAINING_DIR,
) -> dict:
    """Per-row heterogeneous ablation sweep (one model load, per-row feature ranking).

    Args:
        unmatched_features: Feature indices unique to this model (not in the other).

    Returns:
        Dict with baseline_preds, perrow_importance, perrow_rankings,
        sweep_preds, max_k_per_row, query_activations, unmatched_features.
    """
    if model_key not in PERROW_SWEEP_FN:
        raise ValueError(
            f"Per-row sweep not supported for {model_key}. "
            f"Choose from {list(PERROW_SWEEP_FN.keys())}"
        )

    # Deterministic forward passes (match intervene() seeding)
    torch.manual_seed(42)
    np.random.seed(42)

    sae, _ = load_sae(model_key, sae_dir=sae_dir, device=device)
    extraction_layer = get_extraction_layer(model_key, layers_path=layers_path)
    data_mean = load_training_mean(
        model_key, training_dir=training_dir, layers_path=layers_path, device=device,
    )

    return PERROW_SWEEP_FN[model_key](
        X_context, y_context, X_query, y_query, sae,
        unmatched_features, extraction_layer, device, task, data_mean,
    )


def sweep_intervene(
    model_key: str,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    feature_lists: List[List[int]],
    device: str = "cuda",
    task: str = "classification",
    sae_dir: Path = DEFAULT_SAE_DIR,
    layers_path: Path = DEFAULT_LAYERS_PATH,
    training_dir: Path = DEFAULT_TRAINING_DIR,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Sweep multiple ablation levels efficiently (one model load).

    Args:
        feature_lists: List of feature index lists, e.g. [[0], [0,1], [0,1,2], ...]

    Returns:
        (baseline_preds, list_of_ablated_preds)
    """
    if model_key not in SWEEP_FN:
        raise ValueError(f"Sweep not supported for {model_key}. Choose from {list(SWEEP_FN.keys())}")

    sae, _ = load_sae(model_key, sae_dir=sae_dir, device=device)
    extraction_layer = get_extraction_layer(model_key, layers_path=layers_path)
    data_mean = load_training_mean(
        model_key, training_dir=training_dir, layers_path=layers_path, device=device,
    )

    return SWEEP_FN[model_key](
        X_context, y_context, X_query, y_query, sae,
        feature_lists, extraction_layer, device, task, data_mean,
    )


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
    ablate_features: Optional[List[int]] = None,
    device: str = "cuda",
    task: str = "classification",
    sae_dir: Path = DEFAULT_SAE_DIR,
    layers_path: Path = DEFAULT_LAYERS_PATH,
    training_dir: Path = DEFAULT_TRAINING_DIR,
    external_delta: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Run a model with SAE feature ablation at its optimal extraction layer.

    Args:
        model_key: One of 'tabpfn', 'mitra', 'tabicl', 'tabdpt', 'hyperfast'
        X_context, y_context: ICL context data
        X_query, y_query: Query data (y_query for evaluation only)
        ablate_features: SAE feature indices to zero out (optional when external_delta given)
        device: Torch device
        task: 'classification' or 'regression'
        sae_dir: Path to SAE checkpoints
        layers_path: Path to optimal_extraction_layers.json
        training_dir: Path to SAE training data (for centering mean)
        external_delta: Pre-computed delta to inject directly, skipping SAE computation.
            Used by concept transfer to inject deltas translated from another model's space.

    Returns:
        Dict with 'baseline_preds', 'ablated_preds', 'y_query'
    """
    if model_key not in INTERVENE_FN:
        raise ValueError(f"Unsupported model: {model_key}. Choose from {list(INTERVENE_FN.keys())}")

    # Deterministic forward passes (TabPFN resamples context internally)
    torch.manual_seed(42)
    np.random.seed(42)

    extraction_layer = get_extraction_layer(model_key, layers_path=layers_path)

    if external_delta is not None:
        # External delta provided — SAE not needed for delta computation.
        # Still need a dummy sae/features for model-specific fn signature.
        sae = None
        if ablate_features is None:
            ablate_features = []
        data_mean = None
    else:
        sae, _ = load_sae(model_key, sae_dir=sae_dir, device=device)
        if ablate_features is None:
            ablate_features = []
        data_mean = load_training_mean(
            model_key, training_dir=training_dir, layers_path=layers_path, device=device,
        )

    kwargs = dict(
        X_context=X_context,
        y_context=y_context,
        X_query=X_query,
        y_query=y_query,
        sae=sae,
        ablate_features=ablate_features,
        extraction_layer=extraction_layer,
        device=device,
        task=task,
        data_mean=data_mean,
    )

    # Only pass external_delta to functions that support it
    import inspect
    fn = INTERVENE_FN[model_key]
    if "external_delta" in inspect.signature(fn).parameters:
        kwargs["external_delta"] = external_delta

    return fn(**kwargs)


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
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
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
        training_dir=args.training_dir,
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
