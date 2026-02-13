"""
Sparse Autoencoder for Mechanistic Interpretability of Tabular Foundation Models.

Applies dictionary learning to extract interpretable "concepts" from model embeddings.
Based on:
- "Towards Monosemanticity" (Anthropic, 2023)
- "The Geometry of Concepts" (Park et al., 2024)
- "Sparse Autoencoders Find Highly Interpretable Features" (ICLR 2025)

For tabular FMs, concepts might represent:
- Feature interactions (price * volume)
- Temporal patterns (momentum, mean reversion)
- Nonlinear transforms (log, sqrt, polynomial)
- Categorical encodings
- Domain-specific patterns (volatility regimes, seasonality)

Hypothesis: A more universal/capable tabular FM will learn a richer, more complete
dictionary of concepts that transfer across domains.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Optional wandb for training visualization
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# Global cache for K-means centroids (key: data hash, value: centroids tensor)
_KMEANS_CACHE = {}


def get_cached_kmeans(data: torch.Tensor, n_clusters: int, seed: int = 42) -> torch.Tensor:
    """
    Get K-means centroids with caching to avoid recomputation.

    Uses spherical K-means (cosine similarity) for high-dimensional embeddings.
    Assumes input data is already normalized. Centroids are constrained to unit sphere.

    Cache key is based on data hash, n_clusters, and seed.
    Centroids are computed once and reused across trials.

    Args:
        data: (n_samples, n_features) normalized embeddings on GPU/CPU
        n_clusters: Number of clusters
        seed: Random seed

    Returns:
        centroids: (n_clusters, n_features) cluster centers (unit-normalized)
    """
    from sklearn.cluster import KMeans as SklearnKMeans

    # Compute cache key from data hash
    data_hash = hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest()[:16]
    cache_key = f"{data_hash}_{n_clusters}_{seed}"

    if cache_key in _KMEANS_CACHE:
        # Cache hit: return cached centroids on same device as input data
        cached = _KMEANS_CACHE[cache_key]
        return cached.to(device=data.device, dtype=data.dtype)

    # Cache miss: compute centroids using sklearn K-means (on CPU)
    # torch_kmeans is memory-inefficient for this data size (tries to allocate 52GB)
    # sklearn is slower but works, and we cache the result anyway
    data_cpu = data.cpu().numpy()

    # Spherical K-means: normalize data before clustering (cosine distance)
    from sklearn.preprocessing import normalize
    data_normalized = normalize(data_cpu, norm='l2', axis=1)

    kmeans = SklearnKMeans(n_clusters=n_clusters, random_state=seed, n_init=10, max_iter=100, n_jobs=-1)
    kmeans.fit(data_normalized)
    centroids = kmeans.cluster_centers_  # (n_clusters, n_features)

    # Convert to tensor and normalize (should already be normalized, but be safe)
    centroids = torch.tensor(centroids, dtype=data.dtype, device='cpu')
    centroids = F.normalize(centroids, p=2, dim=1)

    # Store in cache (keep on CPU to save GPU memory)
    _KMEANS_CACHE[cache_key] = centroids.cpu()

    # Return on same device as input data
    return centroids.to(device=data.device)


@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder training."""

    # Architecture
    input_dim: int
    hidden_dim: int  # Dictionary size (typically 4-16x input_dim)

    # Sparsity
    sparsity_penalty: float = 1e-3  # L1 coefficient
    sparsity_type: str = "l1"  # "l1", "topk", "batchtopk", "matryoshka", "archetypal", or "matryoshka_archetypal"
    topk: int = 32  # For topk/batchtopk sparsity (also used with archetypal/matryoshka_archetypal)

    # Matryoshka settings (nested representation learning)
    # Used by "matryoshka" and "matryoshka_archetypal" types
    matryoshka_dims: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    matryoshka_weights: List[float] = None  # Weights for each scale (default: equal)

    # Archetypal settings (convex hull constraints)
    archetypal_n_archetypes: int = None  # If None, uses hidden_dim
    archetypal_simplex_temp: float = 1.0  # Temperature for softmax projection
    archetypal_relaxation: float = 0.0  # δ for RA-SAE: 0 = strict A-SAE, >0 = relaxed
    archetypal_use_centroids: bool = True  # Use K-means centroids (paper) vs raw data

    # Dead neuron revival - auxiliary loss
    aux_loss_type: str = "none"  # "none", "ghost_grads", "auxk", or "residual_targeting"
    aux_loss_alpha: float = 0.03125  # Coefficient α for auxiliary loss (1/32 default for AuxK)
    aux_loss_warmup_epochs: int = 3  # Epochs to wait before enabling aux loss (allows initial stabilization)
    dead_threshold: int = 500  # Steps without activation to consider dead (legacy)
    dead_freq_threshold: float = 1e-3  # EMA activation frequency below which a feature is "dead"
    ema_decay: float = 0.999  # Decay rate for activation frequency EMA

    # Dead neuron revival - resampling
    resample_dead_neurons: bool = False  # Enable Anthropic-style neuron resampling
    resample_interval: int = 25000  # Resample every N steps (Anthropic uses 25000)
    resample_samples: int = 1024  # Number of samples to use for resampling

    # Legacy ghost grads compatibility
    use_ghost_grads: bool = None  # Deprecated: use aux_loss_type="ghost_grads" instead
    ghost_grad_coef: float = None  # Deprecated: use aux_loss_alpha instead

    # Legacy aliases (map to ghost grads)
    use_aux_loss: bool = None  # Deprecated: use use_ghost_grads instead
    aux_loss_coef: float = None  # Deprecated: use ghost_grad_coef instead
    aux_topk: int = 512  # Unused, kept for checkpoint compat

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 256
    n_epochs: int = 100
    use_lr_schedule: bool = True  # Three-phase LR schedule: warmup(5%) + stable(75%) + decay(20%)
    warmup_epochs: int = 3  # DEPRECATED: use_lr_schedule now auto-computes warmup as 5% of n_epochs
    strip_fraction: float = 0.0  # Fraction of most-negative encoder weights to zero per dead neuron (0=disabled)

    # Normalization
    normalize_encoder: bool = True  # Unit norm encoder columns
    tied_weights: bool = False  # Decoder = Encoder.T
    use_batchnorm: bool = True  # Use BatchNorm for input normalization (recommended)

    def __post_init__(self):
        """Resolve legacy aux_loss fields to new aux_loss_type system."""
        # Handle legacy use_ghost_grads -> aux_loss_type migration
        if self.use_ghost_grads is not None:
            if self.use_ghost_grads and self.aux_loss_type == "none":
                self.aux_loss_type = "ghost_grads"
            # Keep use_ghost_grads in sync for compatibility
            self.use_ghost_grads = (self.aux_loss_type == "ghost_grads")

        # Handle legacy ghost_grad_coef -> aux_loss_alpha migration
        if self.ghost_grad_coef is not None:
            self.aux_loss_alpha = self.ghost_grad_coef

        # Handle legacy use_aux_loss alias
        if self.use_aux_loss is not None:
            if self.use_aux_loss and self.aux_loss_type == "none":
                self.aux_loss_type = "ghost_grads"
        if self.aux_loss_coef is not None and self.ghost_grad_coef == 0.5:
            self.ghost_grad_coef = self.aux_loss_coef
        # Normalize: ensure booleans are set
        if self.use_ghost_grads is None:
            self.use_ghost_grads = True
        if self.use_aux_loss is None:
            self.use_aux_loss = self.use_ghost_grads
        if self.aux_loss_coef is None:
            self.aux_loss_coef = self.ghost_grad_coef


@dataclass
class SAEResult:
    """Results from SAE training and analysis."""

    # Learned dictionary
    dictionary: np.ndarray  # (hidden_dim, input_dim) - the concepts

    # Training metrics
    reconstruction_loss: float
    sparsity_loss: float
    aux_loss: float  # Auxiliary loss component (e.g., AuxK for dead neurons)
    total_loss: float

    # Feature statistics
    feature_activations: np.ndarray  # (n_samples, hidden_dim)
    feature_frequencies: np.ndarray  # (hidden_dim,) - how often each fires
    mean_active_features: float  # Average features active per sample

    # Interpretability metrics
    dead_features: int  # Features that never activate
    alive_features: int

    # Training history (per-epoch)
    training_history: dict = None  # {"recon_loss": [...], "sparsity_loss": [...], "aux_loss": [...], "total_loss": [...], "lr": [...]}

    # Config used
    config: SAEConfig = None


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for dictionary learning on embeddings.

    Architecture:
        encoder: x -> ReLU(W_enc @ x + b_enc)  [sparse activations]
        decoder: h -> W_dec @ h + b_dec        [reconstruction]

    The encoder weights W_enc columns are the "concepts" or dictionary elements.

    Supported sparsity types:
        - "l1": Standard L1 penalty on activations
        - "topk": Keep only top-k activations per sample
        - "batchtopk": Keep top (batch_size × k) activations across batch (adaptive)
        - "gated": Learnable gates for sparsity (Anthropic style)
        - "matryoshka": Nested representations valid at multiple scales
        - "archetypal": Dictionary elements as convex hull vertices
        - "batchtopk_archetypal": BatchTopK + archetypal decoder
        - "matryoshka_batchtopk_archetypal": All three combined
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config

        # BatchNorm for input normalization (learns mean/std during training)
        # Ensures consistent normalization in train/eval without manual stats tracking
        if config.use_batchnorm:
            self.bn = nn.BatchNorm1d(config.input_dim)
        else:
            self.bn = None

        # Encoder
        self.W_enc = nn.Parameter(torch.randn(config.hidden_dim, config.input_dim) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(config.hidden_dim))

        # Decoder
        if config.tied_weights:
            self.W_dec = None  # Will use W_enc.T
        else:
            self.W_dec = nn.Parameter(torch.randn(config.input_dim, config.hidden_dim) * 0.01)
        self.b_dec = nn.Parameter(torch.zeros(config.input_dim))

        # For gated SAE
        if config.sparsity_type == "gated":
            self.gate = nn.Parameter(torch.randn(config.hidden_dim, config.input_dim) * 0.01)
            self.gate_bias = nn.Parameter(torch.zeros(config.hidden_dim))

        # For archetypal SAE - dictionary atoms are convex combos of reference data
        # This anchors the dictionary to data geometry for training stability
        if config.sparsity_type in ("archetypal", "matryoshka_archetypal"):
            # Reference data (or centroids) will be set via set_reference_data()
            self.register_buffer('reference_data', None)
            # Archetypal coefficients: softmax gives convex combination
            # Shape: (n_reference, hidden_dim) -> each dictionary atom is combo of refs
            self.archetype_logits = None  # Initialized when reference data is set
            # RA-SAE: deviation matrix Λ for relaxed archetypal
            # D = WC + Λ, where ||Λ||₂² ≤ δ
            self.archetype_deviation = None  # Initialized when reference data is set

        # Dead neuron tracking for ghost grads
        if config.use_ghost_grads:
            self.register_buffer('steps_since_active', torch.zeros(config.hidden_dim, dtype=torch.long))
            self.register_buffer('total_steps', torch.tensor(0, dtype=torch.long))
            # EMA activation frequency: tracks per-sample firing rate (not batch-level)
            # Solves the TopK-marginal problem where batch-level any() misses rare features
            self.register_buffer('activation_freq', torch.zeros(config.hidden_dim))

        # BatchTopK inference threshold (EMA of minimum positive activation during training)
        if config.sparsity_type in ("batchtopk", "batchtopk_archetypal", "matryoshka_batchtopk_archetypal"):
            self.register_buffer('inference_threshold', torch.tensor(0.0))
            self.register_buffer('batchtopk_n_updates', torch.tensor(0, dtype=torch.long))

    def encode(self, x: torch.Tensor, return_pre_act: bool = False) -> torch.Tensor:
        """
        Encode input to sparse hidden representation.

        Args:
            x: Input tensor (batch_size, input_dim)
            return_pre_act: If True, also return pre-activation for aux loss

        Returns:
            h: Sparse activations (batch_size, hidden_dim)
            pre_act: (optional) Pre-activation values
        """
        # Apply BatchNorm for consistent normalization in train/eval (if enabled)
        # During training: learns mean/std from data
        # During eval: uses running mean/std (no manual stats needed)
        if self.bn is not None:
            x = self.bn(x)

        # For Archetypal SAE, the constraint is on the DECODER only
        # Encoder uses standard learned weights W_enc
        # (Decoder uses dictionary = convex combo of reference data)
        pre_act = F.linear(x, self.W_enc, self.b_enc)

        if self.config.sparsity_type == "topk":
            # TopK sparsity: keep only top k activations per sample
            h = F.relu(pre_act)
            topk_vals, topk_idx = torch.topk(h, self.config.topk, dim=-1)
            mask = torch.zeros_like(h)
            mask.scatter_(-1, topk_idx, 1.0)
            h = h * mask

        elif self.config.sparsity_type == "batchtopk":
            # BatchTopK sparsity: keep top (batch_size × k) activations across entire batch
            # Allows variable sparsity per sample (adaptive to sample complexity)
            # Paper: Bussmann (2024), arXiv:2412.06410
            h = F.relu(pre_act)

            if self.training:
                # Training: batch-level selection
                h_flat = h.flatten()  # (batch_size × hidden_dim)
                n_keep = h.shape[0] * self.config.topk  # batch_size × k
                # Find threshold: kth largest value where k = total_activations - n_keep
                threshold_val = torch.kthvalue(h_flat, len(h_flat) - n_keep + 1).values
                h = h * (h >= threshold_val).float()

                # Update inference threshold: EMA of minimum positive activation
                with torch.no_grad():
                    if (h > 0).any():
                        min_positive = h[h > 0].min()
                        # EMA update with decay=0.99 (slow-moving average)
                        if self.batchtopk_n_updates == 0:
                            self.inference_threshold = min_positive
                        else:
                            self.inference_threshold = 0.99 * self.inference_threshold + 0.01 * min_positive
                        self.batchtopk_n_updates += 1
            else:
                # Inference: use learned threshold (removes batch dependency)
                h = h * (h >= self.inference_threshold).float()

        elif self.config.sparsity_type == "gated":
            # Gated SAE: separate gate for sparsity
            gate_pre = F.linear(x, self.gate, self.gate_bias)
            gate = torch.sigmoid(gate_pre)
            h = F.relu(pre_act) * gate

        elif self.config.sparsity_type == "matryoshka":
            # Matryoshka: standard ReLU, loss handles nesting
            h = F.relu(pre_act)

        elif self.config.sparsity_type == "archetypal":
            # Archetypal SAE: dictionary atoms are constrained to convex hull of data
            # Apply ReLU, then optionally TopK/BatchTopK for sparsity
            h = F.relu(pre_act)
            # Note: Archetypal doesn't have its own sparsity_type variants
            # Instead, use "batchtopk_archetypal" or "matryoshka_batchtopk_archetypal"

        elif self.config.sparsity_type == "matryoshka_archetypal":
            # Matryoshka-Archetypal: combines nested loss + archetypal decoder
            # - Encoder: standard learned weights (like archetypal)
            # - Decoder: convex combo of K-means centroids (archetypal constraint)
            # - Activations: ReLU + TopK sparsity (or BatchTopK)
            # - Loss: multi-scale reconstruction at Matryoshka dims (handled in train_sae)
            h = F.relu(pre_act)
            # Note: Use "matryoshka_batchtopk_archetypal" for BatchTopK variant

        elif self.config.sparsity_type in ("batchtopk_archetypal", "matryoshka_batchtopk_archetypal"):
            # Archetypal/Matryoshka-Archetypal + BatchTopK sparsity
            h = F.relu(pre_act)

            if self.training:
                # Training: batch-level selection (adaptive sparsity)
                h_flat = h.flatten()
                n_keep = h.shape[0] * self.config.topk
                threshold_val = torch.kthvalue(h_flat, len(h_flat) - n_keep + 1).values
                h = h * (h >= threshold_val).float()

                # Update inference threshold
                with torch.no_grad():
                    if (h > 0).any():
                        min_positive = h[h > 0].min()
                        if self.batchtopk_n_updates == 0:
                            self.inference_threshold = min_positive
                        else:
                            self.inference_threshold = 0.99 * self.inference_threshold + 0.01 * min_positive
                        self.batchtopk_n_updates += 1
            else:
                # Inference: learned threshold
                h = h * (h >= self.inference_threshold).float()

        else:
            # Standard L1: just ReLU
            h = F.relu(pre_act)

        if return_pre_act:
            return h, pre_act
        return h

    def decode(self, h: torch.Tensor, max_dim: int = None) -> torch.Tensor:
        """
        Decode sparse representation back to input space.

        Args:
            h: Hidden activations (batch_size, hidden_dim)
            max_dim: For Matryoshka, only use first max_dim features

        Returns:
            x_hat: Reconstruction (batch_size, input_dim)
        """
        if max_dim is not None:
            h = h[:, :max_dim]

        # For Archetypal/Matryoshka-Archetypal SAE, use dictionary derived from reference data
        if self.config.sparsity_type in ("archetypal", "matryoshka_archetypal") and self.archetype_logits is not None:
            # Dictionary is convex combo of reference points: (hidden_dim, input_dim)
            W_dec = self.get_archetypal_dictionary()  # (hidden_dim, input_dim)
            if max_dim is not None:
                W_dec = W_dec[:max_dim]
            # Decode: h @ W_dec -> (batch, hidden) @ (hidden, input) = (batch, input)
            return F.linear(h, W_dec.T, self.b_dec)  # W_dec.T is (input, hidden)
        elif self.config.tied_weights:
            W = self.W_enc[:max_dim] if max_dim else self.W_enc
            return F.linear(h, W.T, self.b_dec)
        else:
            W = self.W_dec[:, :max_dim] if max_dim else self.W_dec
            return F.linear(h, W, self.b_dec)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (reconstruction, hidden_activations)."""
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h

    def forward_matryoshka(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass for Matryoshka SAE returning reconstructions at multiple scales.

        Returns:
            reconstructions: List of reconstructions at each Matryoshka dimension
            h: Full hidden activations
        """
        h = self.encode(x)
        reconstructions = []
        for dim in self.config.matryoshka_dims:
            if dim <= self.config.hidden_dim:
                x_hat = self.decode(h, max_dim=dim)
                reconstructions.append(x_hat)
        return reconstructions, h

    def compute_ghost_grad_loss(
        self, x: torch.Tensor, h: torch.Tensor, pre_act: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ghost gradient loss for dead neuron revival.

        For dead features, computes a secondary forward pass using exp()
        activation (instead of ReLU) to guarantee gradient flow, then
        decodes using only dead features and minimizes residual error.
        This reorients dead features toward the current reconstruction error.

        Reference: Anthropic, "Scaling Monosemanticity" (2024), Appendix B.

        Args:
            x: Original input (batch_size, input_dim)
            h: Current (alive) activations (batch_size, hidden_dim)
            pre_act: Pre-activation values (batch_size, hidden_dim)

        Returns:
            Ghost grad loss term (scalar)
        """
        if self.config.aux_loss_type != "ghost_grads":
            return torch.tensor(0.0, device=x.device)

        # Update dead neuron tracking with EMA of per-sample activation frequency.
        # Using per-sample mean (not batch-level any()) to properly detect TopK-marginal
        # features that fire rarely enough to be effectively dead but often enough to
        # reset a binary step counter.
        with torch.no_grad():
            batch_freq = (h > 0).float().mean(dim=0)  # Per-sample frequency in this batch
            decay = self.config.ema_decay
            self.activation_freq = decay * self.activation_freq + (1 - decay) * batch_freq
            # Also update legacy step counter for checkpoint compat
            active_mask = (h > 0).any(dim=0)
            self.steps_since_active[active_mask] = 0
            self.steps_since_active[~active_mask] += 1
            self.total_steps += 1

        # Use EMA frequency for dead detection (not step counter)
        dead_mask = self.activation_freq < self.config.dead_freq_threshold
        n_dead = dead_mask.sum().item()

        if n_dead == 0:
            return torch.tensor(0.0, device=x.device)

        # Compute residual error from alive features
        with torch.no_grad():
            x_hat = self.decode(h)
            residual = x - x_hat  # (batch_size, input_dim)

        # Ghost activations: use exp() on dead features' pre-activations
        # exp() ensures non-zero gradients even for very negative pre-activations,
        # unlike ReLU which has zero gradient for negative inputs
        ghost_pre_act = pre_act[:, dead_mask]  # (batch_size, n_dead)
        ghost_act = torch.exp(ghost_pre_act)

        # Normalize ghost activations to prevent explosion
        # Scale to have similar magnitude to alive activations
        with torch.no_grad():
            alive_mean_act = h[h > 0].mean() if (h > 0).any() else torch.tensor(1.0, device=x.device)
            ghost_scale = alive_mean_act / (ghost_act.mean() + 1e-8)
        ghost_act = ghost_act * ghost_scale

        # Decode using only dead features to reconstruct the residual
        if self.config.sparsity_type in ("archetypal", "matryoshka_archetypal") and self.archetype_logits is not None:
            W_dec = self.get_archetypal_dictionary()  # (hidden_dim, input_dim)
            W_dec_dead = W_dec[dead_mask]  # (n_dead, input_dim)
        elif self.config.tied_weights:
            W_dec_dead = self.W_enc[dead_mask]  # (n_dead, input_dim)
        else:
            W_dec_dead = self.W_dec[:, dead_mask]  # (input_dim, n_dead)
            # ghost_recon = ghost_act @ W_dec_dead.T for untied
            ghost_recon = F.linear(ghost_act, W_dec_dead)
            ghost_loss = F.mse_loss(ghost_recon, residual)
            return self.config.aux_loss_alpha * ghost_loss

        # For archetypal/tied: W_dec_dead is (n_dead, input_dim)
        ghost_recon = ghost_act @ W_dec_dead  # (batch, n_dead) @ (n_dead, input) = (batch, input)
        ghost_loss = F.mse_loss(ghost_recon, residual)

        return self.config.aux_loss_alpha * ghost_loss

    def compute_auxk_loss(
        self, x: torch.Tensor, h: torch.Tensor, x_hat: torch.Tensor,
    ) -> torch.Tensor:
        """
        AuxK auxiliary loss for TopK SAEs (Gao et al., 2024).

        Computes reconstruction using ONLY dead latents to ensure they
        remain useful for feature extraction. The loss encourages dead
        features to contribute to reducing the reconstruction error.

        ℒ_aux = ‖e − ê‖²₂
        where:
            e = x - x̂  (main reconstruction error)
            ê = x - x̂_aux  (reconstruction using only dead latents)

        Reference: "Scaling and Evaluating Sparse Autoencoders",
                   Gao et al. (2024), arXiv:2406.04093

        Args:
            x: Original input (batch_size, input_dim)
            h: Current activations (batch_size, hidden_dim)
            x_hat: Main reconstruction (batch_size, input_dim)

        Returns:
            AuxK loss term (scalar)
        """
        if self.config.aux_loss_type != "auxk":
            return torch.tensor(0.0, device=x.device)

        # Update dead neuron tracking with EMA
        with torch.no_grad():
            batch_freq = (h > 0).float().mean(dim=0)
            decay = self.config.ema_decay
            self.activation_freq = decay * self.activation_freq + (1 - decay) * batch_freq

        # Identify dead features
        dead_mask = self.activation_freq < self.config.dead_freq_threshold
        n_dead = dead_mask.sum().item()

        if n_dead == 0:
            return torch.tensor(0.0, device=x.device)

        # Main reconstruction error
        e = x - x_hat  # (batch_size, input_dim)

        # Create auxiliary activations using ONLY dead features
        h_aux = h.clone()
        h_aux[:, ~dead_mask] = 0.0  # Zero out alive features

        # Reconstruct using only dead features
        x_hat_aux = self.decode(h_aux)

        # Auxiliary reconstruction error
        e_aux = x - x_hat_aux  # (batch_size, input_dim)

        # AuxK loss: MSE between main and auxiliary reconstruction errors
        auxk_loss = F.mse_loss(e, e_aux)

        return self.config.aux_loss_alpha * auxk_loss

    def compute_residual_targeting_loss(
        self, x: torch.Tensor, h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Residual targeting auxiliary loss for dead neuron revival.

        Directly trains dead neurons to reconstruct what alive neurons miss:
        ℒ_aux = MSE(x - x̂_alive, x̂_dead)

        This is more effective than reconstructing the entire input because
        it explicitly drives dead neurons to reduce the overall reconstruction
        error by focusing on the residual.

        Args:
            x: Original input (batch_size, input_dim)
            h: Current activations (batch_size, hidden_dim)

        Returns:
            Residual targeting loss term (scalar)
        """
        if self.config.aux_loss_type != "residual_targeting":
            return torch.tensor(0.0, device=x.device)

        # Update dead neuron tracking with EMA
        with torch.no_grad():
            batch_freq = (h > 0).float().mean(dim=0)
            decay = self.config.ema_decay
            self.activation_freq = decay * self.activation_freq + (1 - decay) * batch_freq

        # Identify dead features
        dead_mask = self.activation_freq < self.config.dead_freq_threshold
        n_dead = dead_mask.sum().item()

        if n_dead == 0:
            return torch.tensor(0.0, device=x.device)

        # Reconstruct using ONLY alive features
        h_alive = h.clone()
        h_alive[:, dead_mask] = 0.0  # Zero out dead features
        x_hat_alive = self.decode(h_alive)

        # Residual from alive features
        residual = x - x_hat_alive  # (batch_size, input_dim)

        # Reconstruct using ONLY dead features
        h_dead = h.clone()
        h_dead[:, ~dead_mask] = 0.0  # Zero out alive features
        x_hat_dead = self.decode(h_dead)

        # Loss: train dead neurons to reconstruct the residual
        res_loss = F.mse_loss(x_hat_dead, residual)

        return self.config.aux_loss_alpha * res_loss

    def synaptic_strip(self, dead_mask: torch.Tensor, strip_fraction: float = 0.1):
        """
        Revive dead neurons by zeroing their most negative encoder weights.

        For each dead neuron, zeros the most negative fraction of incoming
        weights (W_enc row) and resets the bias to a small positive value.
        This shifts the pre-activation distribution positive, allowing
        the neuron to fire again after ReLU.

        Args:
            dead_mask: Boolean tensor (hidden_dim,) indicating dead neurons
            strip_fraction: Fraction of most-negative weights to zero (default 0.1)
        """
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0

        with torch.no_grad():
            for idx in torch.where(dead_mask)[0]:
                w = self.W_enc.data[idx]  # (input_dim,)
                n_strip = max(1, int(len(w) * strip_fraction))
                # Find the most negative weights and zero them
                _, neg_indices = torch.topk(-w, n_strip)  # indices of most negative
                w[neg_indices] = 0.0
                # Reset bias to small positive to encourage firing
                self.b_enc.data[idx] = 0.01

        return n_dead

    def resample_dead_neurons(
        self, x: torch.Tensor, dead_mask: torch.Tensor, n_samples: int = 1024
    ) -> int:
        """
        Resample dead neurons using Anthropic's approach.

        Hard reset of dead neuron weights by reinitializing encoder to point
        toward high-error samples and decoder to random directions. This is
        more aggressive than gradient-based revival and can escape local minima.

        Procedure:
        1. Compute reconstruction error for each sample
        2. Sample high-error examples (proportional to squared error)
        3. For each dead neuron:
           - Reinit encoder weights to (sample - mean) direction, normalized
           - Reinit encoder bias to activate at ~5% of samples
           - Reinit decoder weights to random unit vector (or tied to encoder)

        Reference: Anthropic, "Towards Monosemanticity" (2023)

        Args:
            x: Training data (n_samples, input_dim)
            dead_mask: Boolean tensor (hidden_dim,) indicating dead neurons
            n_samples: Number of samples to use for resampling (default 1024)

        Returns:
            Number of neurons resampled
        """
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0

        with torch.no_grad():
            # Compute reconstruction error for each sample (batched to avoid OOM)
            batch_size = 256
            errors = []
            for i in range(0, len(x), batch_size):
                batch = x[i:i+batch_size]
                _, h = self.forward(batch)
                x_hat = self.decode(h)
                batch_errors = (batch - x_hat).pow(2).sum(dim=1)  # (batch_size,)
                errors.append(batch_errors)
            errors = torch.cat(errors, dim=0)  # (n_samples,)

            # Sample high-error examples (proportional to squared error)
            # Use up to n_samples examples, weighted by reconstruction error
            n_available = min(len(x), n_samples)
            probs = errors / (errors.sum() + 1e-8)
            sample_indices = torch.multinomial(
                probs, num_samples=min(n_available, n_dead), replacement=True
            )

            # Get samples and compute mean for centering
            samples = x[sample_indices]  # (n_resampled, input_dim)
            x_mean = x.mean(dim=0, keepdim=True)  # (1, input_dim)

            # Resample each dead neuron
            dead_indices = torch.where(dead_mask)[0]
            for i, neuron_idx in enumerate(dead_indices):
                # Sample index (cycle if more dead neurons than samples)
                sample_idx = i % len(samples)
                sample = samples[sample_idx]

                # Encoder: point toward (sample - mean), normalized
                direction = sample - x_mean.squeeze()
                direction = F.normalize(direction, p=2, dim=0)
                self.W_enc.data[neuron_idx] = direction * 0.1  # Small init

                # Encoder bias: set to activate at ~5% of samples (negative offset)
                # Target: pre_act = W_enc @ x + b > 0 for top 5% of samples
                # Heuristic: b ≈ -1.5 * ||W_enc|| to get sparse firing
                self.b_enc.data[neuron_idx] = -0.15

                # Decoder: random unit vector (or tied to encoder if using tied weights)
                if self.config.tied_weights:
                    # Decoder is automatically W_enc.T, no need to update
                    pass
                elif self.config.sparsity_type in ("archetypal", "matryoshka_archetypal",
                                                     "batchtopk_archetypal", "matryoshka_batchtopk_archetypal"):
                    # For archetypal SAE, reinitialize archetype logits
                    # Sample random convex combination (softmax of random logits)
                    # archetype_logits shape: (n_ref, hidden_dim) - each COLUMN is one atom
                    if self.archetype_logits is not None:
                        n_ref = self.archetype_logits.shape[0]  # Number of reference points
                        device = self.archetype_logits.device  # Use same device as parameter
                        # Peaked initialization: set 2-3 archetypes to high values
                        new_logits = torch.zeros(n_ref, device=device)
                        n_peaks = min(3, n_ref)
                        peak_indices = torch.randperm(n_ref, device=device)[:n_peaks]
                        new_logits[peak_indices] = 5.0
                        self.archetype_logits.data[:, neuron_idx] = new_logits  # Set COLUMN
                else:
                    # Standard decoder: random unit vector
                    random_dir = torch.randn(self.config.input_dim, device=x.device)
                    random_dir = F.normalize(random_dir, p=2, dim=0)
                    self.W_dec.data[:, neuron_idx] = random_dir * 0.1

            # Reset activation frequency for resampled neurons
            if hasattr(self, 'activation_freq'):
                self.activation_freq[dead_mask] = 0.0

        return n_dead

    # Legacy alias
    def compute_aux_loss(self, x, h, pre_act):
        """Deprecated: use compute_ghost_grad_loss instead."""
        return self.compute_ghost_grad_loss(x, h, pre_act)

    def get_dictionary(self) -> np.ndarray:
        """Get learned dictionary (decoder weights for reconstruction)."""
        # For Archetypal/Matryoshka-Archetypal SAE, dictionary is convex combo of reference data
        if self.config.sparsity_type in ("archetypal", "matryoshka_archetypal") and self.archetype_logits is not None:
            W = self.get_archetypal_dictionary().detach().cpu().numpy()
        else:
            W = self.W_enc.detach().cpu().numpy()

        if self.config.normalize_encoder:
            # Normalize to unit vectors
            norms = np.linalg.norm(W, axis=1, keepdims=True)
            W = W / (norms + 1e-8)
        return W

    def normalize_decoder(self):
        """Normalize decoder columns to unit norm (helps training stability)."""
        if self.W_dec is not None:
            with torch.no_grad():
                norms = self.W_dec.norm(dim=0, keepdim=True)
                self.W_dec.data = self.W_dec.data / (norms + 1e-8)

    def set_reference_data(self, reference_data: torch.Tensor, n_ref: int = None):
        """
        Set reference data for Archetypal SAE.

        For A-SAE: Dictionary atoms are strict convex combinations of reference points.
        For RA-SAE: Dictionary = convex_combo + bounded_deviation (controlled by δ).

        Args:
            reference_data: (n_samples, input_dim) training data
            n_ref: Number of reference points/centroids (default: min(1000, n_samples))
        """
        if n_ref is None:
            n_ref = min(1000, len(reference_data))

        device = reference_data.device

        # Use K-means centroids (paper's approach) or raw data subsample
        if self.config.archetypal_use_centroids and len(reference_data) > n_ref:
            # K-means clustering to find centroids (GPU-accelerated with caching)
            centroids = get_cached_kmeans(reference_data, n_clusters=n_ref, seed=42)
            self.reference_data = centroids  # (n_ref, input_dim)
        else:
            # Subsample raw data
            if len(reference_data) > n_ref:
                idx = torch.randperm(len(reference_data))[:n_ref]
                reference_data = reference_data[idx]
            self.reference_data = reference_data  # (n_ref, input_dim)

        n_ref = len(self.reference_data)

        # Initialize archetypal coefficients: (n_ref, hidden_dim)
        # Each column will be softmaxed to give convex weights for one dictionary atom
        # Use peaked initialization: each atom starts by selecting 1-3 centroids
        # This ensures the initial dictionary spans the data space
        init_logits = torch.zeros(n_ref, self.config.hidden_dim, device=device)
        for i in range(self.config.hidden_dim):
            # Each dictionary atom initially points to a random subset of centroids
            n_active = np.random.randint(1, min(4, n_ref + 1))
            active_idx = np.random.choice(n_ref, n_active, replace=False)
            init_logits[active_idx, i] = 5.0  # High value -> after softmax, these dominate
        self.archetype_logits = nn.Parameter(init_logits)

        # RA-SAE: deviation matrix Λ (initialized to zero)
        # D = WC + Λ, where ||Λ||₂² ≤ δ
        if self.config.archetypal_relaxation > 0:
            self.archetype_deviation = nn.Parameter(
                torch.zeros(self.config.hidden_dim, self.config.input_dim, device=device)
            )

    def get_archetypal_dictionary(self) -> torch.Tensor:
        """
        Compute dictionary as convex combinations of reference data.

        For A-SAE (δ=0): D = WC (strict convex hull)
        For RA-SAE (δ>0): D = WC + Λ (relaxed with bounded deviation)

        Returns:
            Dictionary (hidden_dim, input_dim) where each row is a convex combo
            of reference points (plus optional deviation for RA-SAE).
        """
        if self.reference_data is None or self.archetype_logits is None:
            raise RuntimeError("Must call set_reference_data() first for Archetypal SAE")

        # Softmax over reference points gives convex weights
        # Shape: (n_ref, hidden_dim)
        weights = F.softmax(self.archetype_logits / self.config.archetypal_simplex_temp, dim=0)

        # Dictionary = reference_data.T @ weights
        # (input_dim, n_ref) @ (n_ref, hidden_dim) = (input_dim, hidden_dim)
        dictionary = self.reference_data.T @ weights
        dictionary = dictionary.T  # (hidden_dim, input_dim)

        # RA-SAE: add bounded deviation
        if self.config.archetypal_relaxation > 0 and self.archetype_deviation is not None:
            # Project deviation to satisfy ||Λ||₂² ≤ δ
            delta = self.config.archetypal_relaxation
            deviation_norm = self.archetype_deviation.norm()
            if deviation_norm > delta:
                # Scale down to satisfy constraint
                deviation = self.archetype_deviation * (delta / deviation_norm)
            else:
                deviation = self.archetype_deviation
            dictionary = dictionary + deviation

        return dictionary


def train_sae(
    embeddings: np.ndarray,
    config: SAEConfig,
    device: str = "cpu",
    verbose: bool = True,
    use_wandb: bool = False,
) -> Tuple[SparseAutoencoder, SAEResult]:
    """
    Train a Sparse Autoencoder on embeddings.

    Supports multiple SAE variants:
    - Standard L1/TopK/Gated sparsity
    - Matryoshka: nested loss at multiple scales
    - Archetypal: convex hull constraints
    - Auxiliary loss: dead neuron revival

    Args:
        embeddings: (n_samples, embedding_dim) input embeddings
        config: SAE configuration
        device: torch device
        verbose: print training progress

    Returns:
        (trained_model, results)
    """
    # Prepare data
    X = torch.tensor(embeddings, dtype=torch.float32)

    # Center the data (important for SAE)
    X_mean = X.mean(dim=0, keepdim=True)
    X_centered = X - X_mean

    dataset = TensorDataset(X_centered)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize model
    model = SparseAutoencoder(config).to(device)

    # For Archetypal/Matryoshka-Archetypal SAE, set reference data (dictionary atoms = convex combos of refs)
    if config.sparsity_type in ("archetypal", "matryoshka_archetypal", "batchtopk_archetypal", "matryoshka_batchtopk_archetypal"):
        # Use centered training data as reference points
        n_ref = config.archetypal_n_archetypes or min(1000, len(X_centered))
        model.set_reference_data(X_centered.to(device), n_ref=n_ref)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Three-phase LR schedule (SAE best practice):
    # - Warmup (first 5%): linear 0 → peak_lr
    # - Stable (middle 75%): constant at peak_lr
    # - Decay (final 20%): linear peak_lr → 0
    use_lr_schedule = getattr(config, 'use_lr_schedule', True)
    if use_lr_schedule:
        n_epochs = config.n_epochs
        warmup_epochs = int(0.05 * n_epochs)  # First 5%
        stable_end = int(0.80 * n_epochs)     # Until 80%

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Warmup phase: 0 → 1.0
                return (epoch + 1) / warmup_epochs
            elif epoch < stable_end:
                # Stable phase: constant at 1.0
                return 1.0
            else:
                # Decay phase: 1.0 → 0
                decay_progress = (epoch - stable_end) / (n_epochs - stable_end)
                return 1.0 - decay_progress

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # For Matryoshka and Matryoshka-Archetypal, setup dimension weights
    if config.sparsity_type in ("matryoshka", "matryoshka_archetypal", "matryoshka_batchtopk_archetypal"):
        mat_dims = [d for d in config.matryoshka_dims if d <= config.hidden_dim]
        if config.matryoshka_weights is None:
            mat_weights = [1.0 / len(mat_dims)] * len(mat_dims)
        else:
            mat_weights = config.matryoshka_weights[:len(mat_dims)]
            mat_weights = [w / sum(mat_weights) for w in mat_weights]  # Normalize

    # Training loop
    history = {"recon_loss": [], "sparsity_loss": [], "aux_loss": [], "total_loss": [], "lr": []}

    for epoch in range(config.n_epochs):
        epoch_recon = 0.0
        epoch_sparse = 0.0
        epoch_aux = 0.0
        n_batches = 0

        for (batch,) in loader:
            batch = batch.to(device)

            # Forward pass depends on SAE type
            pre_act = None  # Initialize for aux loss check

            if config.sparsity_type == "matryoshka":
                # Matryoshka: compute loss at multiple scales
                aux_type = config.aux_loss_type
                if aux_type == "none" and config.use_ghost_grads:
                    aux_type = "ghost_grads"

                if aux_type == "ghost_grads":
                    h, pre_act = model.encode(batch, return_pre_act=True)
                else:
                    h = model.encode(batch)

                reconstructions = []
                for dim in mat_dims:
                    x_hat_scale = model.decode(h, max_dim=dim)
                    reconstructions.append(x_hat_scale)

                recon_loss = torch.tensor(0.0, device=device)
                for x_hat_scale, weight in zip(reconstructions, mat_weights):
                    recon_loss = recon_loss + weight * F.mse_loss(x_hat_scale, batch)

                # Save final scale reconstruction for aux loss
                x_hat = reconstructions[-1]

                # Sparsity on full activations
                sparsity_loss = config.sparsity_penalty * h.abs().mean()

            elif config.sparsity_type in ("archetypal", "batchtopk_archetypal"):
                # Archetypal SAE: dictionary atoms are constrained to convex hull of data
                # (via archetype_logits -> softmax -> weighted sum of reference points)
                # TopK/BatchTopK sparsity is handled in encode()
                h, pre_act = model.encode(batch, return_pre_act=True)
                x_hat = model.decode(h)
                recon_loss = F.mse_loss(x_hat, batch)
                # Small L1 for regularization
                sparsity_loss = config.sparsity_penalty * 0.1 * h.abs().mean()

            elif config.sparsity_type in ("matryoshka_archetypal", "matryoshka_batchtopk_archetypal"):
                # Matryoshka-Archetypal: combines two key ideas
                # 1. Archetypal decoder: dictionary atoms = convex combos of K-means centroids (stability)
                # 2. Matryoshka loss: multi-scale reconstruction at nested truncation points (hierarchy)
                # 3. TopK sparsity: handled in encode() for interpretable sparse activations
                # Note: Ghost grads destabilize training here (R²→-12804 on CARTE, NaN on TabDPT).
                #       AuxK is compatible and may help with dead features.
                h = model.encode(batch)

                # Multi-scale reconstruction loss (Matryoshka)
                # Loss at each scale ensures features are ordered by importance
                recon_loss = torch.tensor(0.0, device=device)
                for dim, weight in zip(mat_dims, mat_weights):
                    x_hat = model.decode(h, max_dim=dim)
                    recon_loss = recon_loss + weight * F.mse_loss(x_hat, batch)
                # x_hat now holds the final scale reconstruction

                # Small L1 regularization (TopK sparsity is main constraint from encode())
                sparsity_loss = config.sparsity_penalty * 0.1 * h.abs().mean()

            else:
                # Standard forward
                if config.use_ghost_grads:
                    h, pre_act = model.encode(batch, return_pre_act=True)
                    x_hat = model.decode(h)
                else:
                    x_hat, h = model(batch)
                    pre_act = None
                recon_loss = F.mse_loss(x_hat, batch)

                # Sparsity loss
                if config.sparsity_type == "l1":
                    sparsity_loss = config.sparsity_penalty * h.abs().mean()
                elif config.sparsity_type in ("topk", "batchtopk"):
                    # TopK/BatchTopK has implicit sparsity, small L1 for dead feature prevention
                    sparsity_loss = config.sparsity_penalty * 0.1 * h.abs().mean()
                elif config.sparsity_type == "gated":
                    sparsity_loss = config.sparsity_penalty * h.abs().mean()
                else:
                    sparsity_loss = config.sparsity_penalty * h.abs().mean()

            # Auxiliary loss for dead neuron revival
            # Handle legacy use_ghost_grads config
            aux_type = config.aux_loss_type
            if aux_type == "none" and config.use_ghost_grads:
                aux_type = "ghost_grads"  # Legacy compatibility

            # Delayed startup: skip aux loss during warmup to allow initial stabilization
            warmup_epochs = getattr(config, 'aux_loss_warmup_epochs', 0)
            if epoch < warmup_epochs:
                aux_loss = torch.tensor(0.0, device=device)
            elif aux_type == "ghost_grads" and pre_act is not None:
                aux_loss = model.compute_ghost_grad_loss(batch, h, pre_act)
            elif aux_type == "auxk":
                aux_loss = model.compute_auxk_loss(batch, h, x_hat)
            elif aux_type == "residual_targeting":
                aux_loss = model.compute_residual_targeting_loss(batch, h)
            else:
                aux_loss = torch.tensor(0.0, device=device)

            loss = recon_loss + sparsity_loss + aux_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalize decoder periodically
            if n_batches % 10 == 0:
                model.normalize_decoder()

            epoch_recon += recon_loss.item()
            epoch_sparse += sparsity_loss.item()
            epoch_aux += aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss
            n_batches += 1

        epoch_recon /= n_batches
        epoch_sparse /= n_batches
        epoch_aux /= n_batches
        history["recon_loss"].append(epoch_recon)
        history["sparsity_loss"].append(epoch_sparse)
        history["aux_loss"].append(epoch_aux)
        history["total_loss"].append(epoch_recon + epoch_sparse + epoch_aux)
        history["lr"].append(scheduler.get_last_lr()[0] if scheduler else config.learning_rate)

        # LR warmup step
        if scheduler is not None:
            scheduler.step()

        # Dead neuron detection and optional synaptic stripping (after each epoch)
        strip_fraction = getattr(config, 'strip_fraction', 0.0)
        n_dead = 0
        n_stripped = 0
        n_resampled = 0
        n_alive = config.hidden_dim
        if hasattr(model, 'activation_freq'):
            dead_mask = model.activation_freq < config.dead_freq_threshold
            n_dead = dead_mask.sum().item()
            n_alive = config.hidden_dim - n_dead
            if n_dead > 0 and strip_fraction > 0:
                n_stripped = model.synaptic_strip(dead_mask, strip_fraction)

        # Dead neuron resampling (Anthropic-style hard reset)
        resample_enabled = getattr(config, 'resample_dead_neurons', False)
        resample_interval = getattr(config, 'resample_interval', 25000)
        resample_samples = getattr(config, 'resample_samples', 1024)

        if resample_enabled and hasattr(model, 'total_steps'):
            # Check if we should resample this epoch (based on total training steps)
            steps_per_epoch = len(loader)
            total_steps = (epoch + 1) * steps_per_epoch

            # Resample if we've crossed a resample_interval boundary
            prev_steps = epoch * steps_per_epoch
            if (total_steps // resample_interval) > (prev_steps // resample_interval):
                if hasattr(model, 'activation_freq'):
                    dead_mask = model.activation_freq < config.dead_freq_threshold
                    n_dead = dead_mask.sum().item()
                    if n_dead > 0:
                        n_resampled = model.resample_dead_neurons(
                            X_centered.to(device), dead_mask, n_samples=resample_samples
                        )

        # Log to wandb
        if use_wandb and HAS_WANDB:
            wandb.log({
                "epoch": epoch,
                "loss/reconstruction": epoch_recon,
                "loss/sparsity": epoch_sparse,
                "loss/auxiliary": epoch_aux,
                "loss/total": epoch_recon + epoch_sparse + epoch_aux,
                "features/dead": n_dead,
                "features/alive": n_alive,
                "features/dead_fraction": n_dead / config.hidden_dim,
                "features/resampled": n_resampled,
                "lr": scheduler.get_last_lr()[0] if scheduler else config.learning_rate,
            })

        if verbose and (epoch + 1) % 20 == 0:
            msg = f"  Epoch {epoch+1}/{config.n_epochs}: recon={epoch_recon:.6f}, sparsity={epoch_sparse:.6f}"
            if config.use_ghost_grads or config.aux_loss_type != "none":
                msg += f", aux={epoch_aux:.6f}"
            msg += f", dead={n_dead}"
            if n_stripped > 0:
                msg += f", stripped={n_stripped}"
            if n_resampled > 0:
                msg += f", resampled={n_resampled}"
            print(msg)

    # Compute final statistics
    model.eval()
    with torch.no_grad():
        X_centered = X_centered.to(device)
        _, all_activations = model(X_centered)
        all_activations = all_activations.cpu().numpy()

    # Feature statistics
    feature_frequencies = (all_activations > 0).mean(axis=0)
    mean_active = (all_activations > 0).sum(axis=1).mean()
    dead_features = (feature_frequencies < 1e-4).sum()

    result = SAEResult(
        dictionary=model.get_dictionary(),
        reconstruction_loss=history["recon_loss"][-1],
        sparsity_loss=history["sparsity_loss"][-1],
        aux_loss=history["aux_loss"][-1],
        total_loss=history["total_loss"][-1],
        feature_activations=all_activations,
        feature_frequencies=feature_frequencies,
        mean_active_features=float(mean_active),
        dead_features=int(dead_features),
        alive_features=int(config.hidden_dim - dead_features),
        training_history=history,
        config=config,
    )

    return model, result


def compute_c_dec(decoder: np.ndarray) -> float:
    """
    Compute decoder pairwise cosine similarity (c_dec).

    From Chanin & Garriga-Alonso (arXiv:2508.16560).
    Measures feature redundancy/interference in the decoder.

    c_dec = (1 / (h choose 2)) Σ Σ |cos(W_dec,i, W_dec,j)|

    Lower values = better (features more distinct/orthogonal)
    Minimized at optimal sparsity level

    Args:
        decoder: (hidden_dim, input_dim) decoder matrix

    Returns:
        Mean absolute cosine similarity between all decoder feature pairs
    """
    # Normalize decoder rows to unit vectors
    decoder_norm = decoder / (np.linalg.norm(decoder, axis=1, keepdims=True) + 1e-8)

    # Compute all pairwise cosine similarities
    similarity_matrix = decoder_norm @ decoder_norm.T  # (h, h)

    # Take absolute value (we care about magnitude of similarity, not sign)
    abs_similarity = np.abs(similarity_matrix)

    # Sum over upper triangle (distinct pairs i < j, excluding diagonal)
    h = decoder.shape[0]
    upper_triangle = np.triu_indices(h, k=1)  # Indices where i < j
    pairwise_sims = abs_similarity[upper_triangle]

    # Average over all pairs
    c_dec = pairwise_sims.mean()

    return float(c_dec)


def compare_dictionaries(
    dict_a: np.ndarray,
    dict_b: np.ndarray,
    top_k: int = 10,
) -> Dict:
    """
    Compare two learned dictionaries (concept sets).

    Measures how similar the learned concepts are across models.
    High similarity suggests models learned similar "vocabulary" of features.

    Args:
        dict_a: (hidden_dim_a, input_dim) dictionary from model A
        dict_b: (hidden_dim_b, input_dim) dictionary from model B
        top_k: Number of top matches to report

    Returns:
        Dict with comparison metrics
    """
    # Normalize dictionaries
    dict_a = dict_a / (np.linalg.norm(dict_a, axis=1, keepdims=True) + 1e-8)
    dict_b = dict_b / (np.linalg.norm(dict_b, axis=1, keepdims=True) + 1e-8)

    # Compute all pairwise similarities
    similarity_matrix = dict_a @ dict_b.T  # (hidden_a, hidden_b)

    # Best match for each feature in A
    best_match_a = similarity_matrix.max(axis=1)
    best_match_idx_a = similarity_matrix.argmax(axis=1)

    # Best match for each feature in B
    best_match_b = similarity_matrix.max(axis=0)
    best_match_idx_b = similarity_matrix.argmax(axis=0)

    # Bidirectional matches (same feature is best match in both directions)
    bidirectional_matches = 0
    for i, j in enumerate(best_match_idx_a):
        if best_match_idx_b[j] == i:
            bidirectional_matches += 1

    # Coverage: fraction of features with a good match (>0.7 similarity)
    coverage_a = (best_match_a > 0.7).mean()
    coverage_b = (best_match_b > 0.7).mean()

    # Top matches
    flat_idx = np.argsort(similarity_matrix.flatten())[::-1][:top_k]
    top_matches = []
    for idx in flat_idx:
        i, j = idx // similarity_matrix.shape[1], idx % similarity_matrix.shape[1]
        top_matches.append({
            "feature_a": int(i),
            "feature_b": int(j),
            "similarity": float(similarity_matrix[i, j]),
        })

    return {
        "mean_best_match_a": float(best_match_a.mean()),
        "mean_best_match_b": float(best_match_b.mean()),
        "bidirectional_matches": bidirectional_matches,
        "bidirectional_rate": bidirectional_matches / min(len(dict_a), len(dict_b)),
        "coverage_a_at_0.7": float(coverage_a),
        "coverage_b_at_0.7": float(coverage_b),
        "similarity_matrix": similarity_matrix,
        "top_matches": top_matches,
    }


def measure_dictionary_richness(
    sae_result: SAEResult,
    input_features: Optional[np.ndarray] = None,
    sae_model: Optional["SparseAutoencoder"] = None,
) -> Dict:
    """
    Measure the "richness" of a learned dictionary.

    A richer dictionary suggests the model has learned more concepts.

    Metrics:
    - Alive features: Non-dead dictionary elements
    - Effective dimensions: Entropy-based count of independent features (perplexity)
    - L0 sparsity: Mean number of active features per sample
    - Explained variance: R² between input and reconstruction
    - Dictionary diversity: Mean pairwise orthogonality of features

    References:
    - Effective dimensions via entropy/perplexity: Shannon (1948), "A Mathematical
      Theory of Communication". Perplexity = exp(entropy) gives the effective number
      of equiprobable outcomes.
    - Dictionary diversity/coherence: Newman & Mimno (2010), "Automatic Evaluation
      of Topic Coherence". Low mean pairwise similarity indicates diverse features.
    - L0 sparsity: Anthropic (2024), "Scaling Monosemanticity". Standard metric
      for SAE interpretability - fewer active features = more monosemantic.
    - Explained variance (R²): Standard regression metric. 1 - MSE/Var(X).

    Args:
        sae_result: SAE training result
        input_features: Original input features for explained variance computation

    Returns:
        Dict with richness metrics
    """
    freqs = sae_result.feature_frequencies
    activations = sae_result.feature_activations
    dictionary = sae_result.dictionary

    # 1. Alive features (non-dead)
    alive = sae_result.alive_features
    total = len(freqs)

    # 2. Effective dimensions via entropy (perplexity)
    # Normalize frequencies to probabilities
    freq_norm = freqs / (freqs.sum() + 1e-8)
    entropy = -np.sum(freq_norm * np.log(freq_norm + 1e-8))
    effective_dims = np.exp(entropy)  # Perplexity

    # 3. L0 sparsity: mean number of non-zero activations per sample
    # Following Anthropic's "Scaling Monosemanticity" convention
    l0_per_sample = (activations != 0).sum(axis=1)  # (n_samples,)
    l0_sparsity = float(l0_per_sample.mean())
    l0_sparsity_frac = l0_sparsity / total  # As fraction of dictionary size

    # 4. Activation sparsity: fraction of zero activations overall
    activation_sparsity = (activations == 0).mean()

    # 5. Dictionary diversity: average pairwise distance (1 - cosine similarity)
    dict_norm = dictionary / (np.linalg.norm(dictionary, axis=1, keepdims=True) + 1e-8)
    pairwise_sim = dict_norm @ dict_norm.T
    # Exclude diagonal
    np.fill_diagonal(pairwise_sim, 0)
    mean_pairwise_sim = pairwise_sim.sum() / (len(dictionary) * (len(dictionary) - 1))
    diversity = 1 - mean_pairwise_sim  # Higher = more diverse

    # 6. Reconstruction quality (implicit from loss)
    recon_quality = 1 / (1 + sae_result.reconstruction_loss)

    # 7. Explained variance ratio (R²) - requires original inputs
    # Note: SAE is trained on centered data (X - mean), so we must center before computing R²
    explained_variance = None
    if input_features is not None and sae_model is not None:
        import torch
        device = next(sae_model.parameters()).device
        sae_model.eval()
        # Center data (same as train_sae does)
        x_mean = input_features.mean(axis=0, keepdims=True)
        x_centered = input_features - x_mean
        with torch.no_grad():
            x = torch.tensor(x_centered, dtype=torch.float32, device=device)
            x_hat, _ = sae_model(x)  # Forward pass: encode then decode
            x_hat = x_hat.cpu().numpy()
        # R² = 1 - SS_res / SS_tot (on centered data)
        ss_res = np.sum((x_centered - x_hat) ** 2)
        ss_tot = np.sum(x_centered ** 2)  # Centered data, so mean is 0
        explained_variance = float(1 - ss_res / (ss_tot + 1e-8))
    elif input_features is not None:
        # Fallback: assume tied weights (dictionary is decoder)
        # This may be inaccurate for untied weight SAEs
        x_centered = input_features - input_features.mean(axis=0, keepdims=True)
        reconstructions = activations @ dictionary
        ss_res = np.sum((x_centered - reconstructions) ** 2)
        ss_tot = np.sum(x_centered ** 2)
        explained_variance = float(1 - ss_res / (ss_tot + 1e-8))

    # 8. Composite richness score
    # Normalize components to [0, 1] and combine
    alive_score = alive / total
    diversity_score = diversity
    sparsity_score = activation_sparsity  # Higher sparsity = more interpretable

    richness_score = (
        0.3 * alive_score +
        0.3 * diversity_score +
        0.2 * sparsity_score +
        0.2 * recon_quality
    )

    result = {
        "alive_features": alive,
        "dead_features": total - alive,
        "alive_ratio": alive_score,
        "effective_dimensions": float(effective_dims),
        "sparsity": float(activation_sparsity),
        "dictionary_diversity": float(diversity),
        "reconstruction_quality": float(recon_quality),
        "richness_score": float(richness_score),
        "mean_active_per_sample": sae_result.mean_active_features,
        # New metrics
        "l0_sparsity": l0_sparsity,  # Mean active features per sample (count)
        "l0_sparsity_frac": l0_sparsity_frac,  # As fraction of dictionary
    }

    if explained_variance is not None:
        result["explained_variance"] = explained_variance

    return result


def analyze_feature_geometry(
    dictionary: np.ndarray,
    feature_activations: np.ndarray,
) -> Dict:
    """
    Analyze the geometry of learned features (inspired by "Geometry of Concepts").

    Looks for:
    - Clusters of related features
    - Parallelogram structures (analogies)
    - Power law in eigenvalues

    Args:
        dictionary: (hidden_dim, input_dim) learned features
        feature_activations: (n_samples, hidden_dim) activation patterns

    Returns:
        Dict with geometric analysis
    """
    # Normalize dictionary
    dict_norm = dictionary / (np.linalg.norm(dictionary, axis=1, keepdims=True) + 1e-8)

    # 1. Eigenvalue spectrum of dictionary gram matrix
    gram = dict_norm @ dict_norm.T
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

    # Fit power law: eigenvalue ~ rank^(-alpha)
    ranks = np.arange(1, len(eigenvalues) + 1)
    # Log-log fit
    valid = eigenvalues > 1e-6
    if valid.sum() > 10:
        log_ranks = np.log(ranks[valid])
        log_eigs = np.log(eigenvalues[valid])
        # Linear regression in log space
        slope, _ = np.polyfit(log_ranks, log_eigs, 1)
        power_law_alpha = -slope
    else:
        power_law_alpha = 0.0

    # 2. Clustering coefficient
    # Build adjacency from high similarity pairs
    adj = (gram > 0.5).astype(float)
    np.fill_diagonal(adj, 0)
    degrees = adj.sum(axis=1)

    # Clustering: fraction of connected triads
    clustering_coeffs = []
    for i in range(len(dictionary)):
        neighbors = np.where(adj[i] > 0)[0]
        if len(neighbors) < 2:
            continue
        # Count edges among neighbors
        n_edges = adj[neighbors][:, neighbors].sum() / 2
        max_edges = len(neighbors) * (len(neighbors) - 1) / 2
        if max_edges > 0:
            clustering_coeffs.append(n_edges / max_edges)

    mean_clustering = np.mean(clustering_coeffs) if clustering_coeffs else 0.0

    # 3. Co-activation patterns
    # Features that tend to activate together
    coactivation = (feature_activations > 0).T @ (feature_activations > 0)
    coactivation = coactivation / (len(feature_activations) + 1e-8)
    np.fill_diagonal(coactivation, 0)

    # Modularity: do features form groups that co-activate?
    mean_coactivation = coactivation.mean()

    return {
        "eigenvalues": eigenvalues[:20].tolist(),  # Top 20
        "power_law_alpha": float(power_law_alpha),
        "mean_clustering": float(mean_clustering),
        "mean_coactivation": float(mean_coactivation),
        "n_high_similarity_pairs": int((gram > 0.7).sum() / 2),
    }


if __name__ == "__main__":
    # Demo with random embeddings - test all SAE variants
    print("=" * 60)
    print("Testing Sparse Autoencoder Module - All Variants")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    # Generate test embeddings
    n_samples = 1000
    embedding_dim = 64
    embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)

    # Test configurations for each SAE type
    sae_configs = {
        "L1 + Aux Loss": SAEConfig(
            input_dim=embedding_dim,
            hidden_dim=256,
            sparsity_penalty=1e-3,
            sparsity_type="l1",
            use_aux_loss=True,
            n_epochs=50,
        ),
        "TopK": SAEConfig(
            input_dim=embedding_dim,
            hidden_dim=256,
            sparsity_penalty=1e-3,
            sparsity_type="topk",
            topk=32,
            use_aux_loss=True,
            n_epochs=50,
        ),
        "Matryoshka": SAEConfig(
            input_dim=embedding_dim,
            hidden_dim=256,
            sparsity_penalty=1e-3,
            sparsity_type="matryoshka",
            matryoshka_dims=[32, 64, 128, 256],
            use_aux_loss=True,
            n_epochs=50,
        ),
        "Archetypal": SAEConfig(
            input_dim=embedding_dim,
            hidden_dim=64,  # Fewer archetypes
            sparsity_penalty=1e-2,
            sparsity_type="archetypal",
            archetypal_simplex_temp=0.5,
            use_aux_loss=False,  # Not applicable for archetypal
            n_epochs=50,
        ),
    }

    results_summary = {}

    for name, config in sae_configs.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"  hidden_dim={config.hidden_dim}, sparsity={config.sparsity_type}")
        print("=" * 60)

        model, result = train_sae(embeddings, config, verbose=True)

        print(f"\nResults for {name}:")
        print(f"  Reconstruction loss: {result.reconstruction_loss:.6f}")
        print(f"  Sparsity loss: {result.sparsity_loss:.6f}")
        print(f"  Alive features: {result.alive_features}/{config.hidden_dim}")
        print(f"  Mean active per sample: {result.mean_active_features:.1f}")

        # Richness analysis
        richness = measure_dictionary_richness(result)
        print(f"  Richness score: {richness['richness_score']:.4f}")
        print(f"  Dictionary diversity: {richness['dictionary_diversity']:.4f}")

        results_summary[name] = {
            "recon_loss": result.reconstruction_loss,
            "alive_ratio": result.alive_features / config.hidden_dim,
            "richness": richness['richness_score'],
        }

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'SAE Type':<20} {'Recon Loss':>12} {'Alive Ratio':>12} {'Richness':>10}")
    print("-" * 60)
    for name, metrics in results_summary.items():
        print(f"{name:<20} {metrics['recon_loss']:>12.6f} {metrics['alive_ratio']:>12.2%} {metrics['richness']:>10.4f}")

    print("\nSAE module test complete!")
