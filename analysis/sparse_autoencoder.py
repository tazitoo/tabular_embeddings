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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder training."""

    # Architecture
    input_dim: int
    hidden_dim: int  # Dictionary size (typically 4-16x input_dim)

    # Sparsity
    sparsity_penalty: float = 1e-3  # L1 coefficient
    sparsity_type: str = "l1"  # "l1", "topk", "gated", "matryoshka", or "archetypal"
    topk: int = 32  # For topk sparsity

    # Matryoshka settings (nested representation learning)
    matryoshka_dims: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    matryoshka_weights: List[float] = None  # Weights for each scale (default: equal)

    # Archetypal settings (convex hull constraints)
    archetypal_n_archetypes: int = None  # If None, uses hidden_dim
    archetypal_simplex_temp: float = 1.0  # Temperature for softmax projection

    # Dead neuron revival (auxiliary loss)
    use_aux_loss: bool = True  # Enable dead neuron revival
    aux_loss_coef: float = 1e-2  # Coefficient for auxiliary loss
    dead_threshold: int = 10000  # Steps without activation to consider dead
    aux_topk: int = 512  # Number of dead neurons to revive per batch

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 256
    n_epochs: int = 100

    # Normalization
    normalize_encoder: bool = True  # Unit norm encoder columns
    tied_weights: bool = False  # Decoder = Encoder.T


@dataclass
class SAEResult:
    """Results from SAE training and analysis."""

    # Learned dictionary
    dictionary: np.ndarray  # (hidden_dim, input_dim) - the concepts

    # Training metrics
    reconstruction_loss: float
    sparsity_loss: float
    total_loss: float

    # Feature statistics
    feature_activations: np.ndarray  # (n_samples, hidden_dim)
    feature_frequencies: np.ndarray  # (hidden_dim,) - how often each fires
    mean_active_features: float  # Average features active per sample

    # Interpretability metrics
    dead_features: int  # Features that never activate
    alive_features: int

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
        - "gated": Learnable gates for sparsity (Anthropic style)
        - "matryoshka": Nested representations valid at multiple scales
        - "archetypal": Dictionary elements as convex hull vertices
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config

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

        # For archetypal SAE - learnable archetype coefficients
        if config.sparsity_type == "archetypal":
            n_arch = config.archetypal_n_archetypes or config.hidden_dim
            # Archetype mixing weights (will be projected to simplex)
            self.archetype_logits = nn.Parameter(torch.randn(config.hidden_dim, n_arch) * 0.01)

        # Dead neuron tracking for auxiliary loss
        if config.use_aux_loss:
            # Register buffer to track steps since last activation
            self.register_buffer('steps_since_active', torch.zeros(config.hidden_dim, dtype=torch.long))
            self.register_buffer('total_steps', torch.tensor(0, dtype=torch.long))

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
        # Pre-activation
        pre_act = F.linear(x, self.W_enc, self.b_enc)

        if self.config.sparsity_type == "topk":
            # TopK sparsity: keep only top k activations
            h = F.relu(pre_act)
            topk_vals, topk_idx = torch.topk(h, self.config.topk, dim=-1)
            mask = torch.zeros_like(h)
            mask.scatter_(-1, topk_idx, 1.0)
            h = h * mask

        elif self.config.sparsity_type == "gated":
            # Gated SAE: separate gate for sparsity
            gate_pre = F.linear(x, self.gate, self.gate_bias)
            gate = torch.sigmoid(gate_pre)
            h = F.relu(pre_act) * gate

        elif self.config.sparsity_type == "matryoshka":
            # Matryoshka: standard ReLU, loss handles nesting
            h = F.relu(pre_act)

        elif self.config.sparsity_type == "archetypal":
            # Archetypal: activations represent archetype memberships
            # Project to simplex (each sample is convex combination of archetypes)
            h = F.softmax(pre_act / self.config.archetypal_simplex_temp, dim=-1)

        elif self.config.sparsity_type == "matryoshka_archetypal":
            # Combined Matryoshka-Archetypal: nested SPARSE simplex structure
            # Uses JumpReLU + normalization for sparse simplex activations
            # This gives granularity (can truncate) + interpretability (sparse simplex)
            h = torch.zeros_like(pre_act)
            mat_dims = self.config.matryoshka_dims

            # JumpReLU threshold (learnable or fixed)
            threshold = self.config.archetypal_simplex_temp  # Repurpose temp as threshold

            # Apply JumpReLU + normalize within each nested scale
            prev_dim = 0
            for dim in mat_dims:
                if dim <= self.config.hidden_dim:
                    scale_act = pre_act[:, prev_dim:dim]
                    # JumpReLU: zero out values below threshold
                    sparse_act = F.relu(scale_act - threshold)
                    # Normalize to simplex (sum to 1 per sample within this scale)
                    scale_sum = sparse_act.sum(dim=-1, keepdim=True) + 1e-8
                    h[:, prev_dim:dim] = sparse_act / scale_sum * (dim - prev_dim)
                    prev_dim = dim

            # Handle remaining dimensions if any
            if prev_dim < self.config.hidden_dim:
                scale_act = pre_act[:, prev_dim:]
                sparse_act = F.relu(scale_act - threshold)
                scale_sum = sparse_act.sum(dim=-1, keepdim=True) + 1e-8
                h[:, prev_dim:] = sparse_act / scale_sum * (self.config.hidden_dim - prev_dim)

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

        if self.config.tied_weights:
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

    def compute_aux_loss(self, x: torch.Tensor, h: torch.Tensor, pre_act: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary loss to revive dead neurons.

        Based on Anthropic's approach: encourage dead neurons to activate on
        samples with high reconstruction error.

        Args:
            x: Original input
            h: Current activations
            pre_act: Pre-activation values

        Returns:
            aux_loss: Auxiliary loss term
        """
        if not self.config.use_aux_loss:
            return torch.tensor(0.0, device=x.device)

        # Update dead neuron tracking
        with torch.no_grad():
            active_mask = (h > 0).any(dim=0)  # Which neurons fired this batch
            self.steps_since_active[active_mask] = 0
            self.steps_since_active[~active_mask] += 1
            self.total_steps += 1

        # Find dead neurons (haven't fired in dead_threshold steps)
        dead_mask = self.steps_since_active > self.config.dead_threshold

        if not dead_mask.any():
            return torch.tensor(0.0, device=x.device)

        # Compute reconstruction error per sample
        x_hat = self.decode(h)
        recon_error = (x - x_hat).pow(2).sum(dim=-1)  # (batch_size,)

        # Get top-k samples with highest reconstruction error
        n_samples = min(self.config.aux_topk, len(x))
        _, high_error_idx = torch.topk(recon_error, n_samples)

        # Get pre-activations of dead neurons on high-error samples
        dead_pre_act = pre_act[high_error_idx][:, dead_mask]  # (n_samples, n_dead)

        # Auxiliary loss: encourage dead neurons to have positive pre-activation
        # on high-error samples (so they'll fire after ReLU)
        aux_loss = F.relu(-dead_pre_act + 0.1).mean()  # Encourage pre_act > 0.1

        return self.config.aux_loss_coef * aux_loss

    def get_dictionary(self) -> np.ndarray:
        """Get learned dictionary (encoder weights)."""
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


def train_sae(
    embeddings: np.ndarray,
    config: SAEConfig,
    device: str = "cpu",
    verbose: bool = True,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # For Matryoshka and Matryoshka-Archetypal, setup dimension weights
    if config.sparsity_type in ("matryoshka", "matryoshka_archetypal"):
        mat_dims = [d for d in config.matryoshka_dims if d <= config.hidden_dim]
        if config.matryoshka_weights is None:
            mat_weights = [1.0 / len(mat_dims)] * len(mat_dims)
        else:
            mat_weights = config.matryoshka_weights[:len(mat_dims)]
            mat_weights = [w / sum(mat_weights) for w in mat_weights]  # Normalize

    # Training loop
    history = {"recon_loss": [], "sparsity_loss": [], "aux_loss": [], "total_loss": []}

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
                # Get pre_act for aux loss if needed
                if config.use_aux_loss:
                    h, pre_act = model.encode(batch, return_pre_act=True)
                else:
                    h = model.encode(batch)
                reconstructions = []
                for dim in mat_dims:
                    x_hat = model.decode(h, max_dim=dim)
                    reconstructions.append(x_hat)
                recon_loss = torch.tensor(0.0, device=device)
                for x_hat, weight in zip(reconstructions, mat_weights):
                    recon_loss = recon_loss + weight * F.mse_loss(x_hat, batch)
                # Sparsity on full activations
                sparsity_loss = config.sparsity_penalty * h.abs().mean()

            elif config.sparsity_type == "archetypal":
                # Archetypal: activations are simplex-constrained
                h, pre_act = model.encode(batch, return_pre_act=True)
                x_hat = model.decode(h)
                recon_loss = F.mse_loss(x_hat, batch)
                # Entropy regularization to encourage sparse archetype usage
                entropy = -(h * torch.log(h + 1e-8)).sum(dim=-1).mean()
                sparsity_loss = -config.sparsity_penalty * entropy  # Minimize entropy = more peaked

            elif config.sparsity_type == "matryoshka_archetypal":
                # Combined: Matryoshka multi-scale loss + Archetypal simplex structure
                h, pre_act = model.encode(batch, return_pre_act=True)

                # Multi-scale reconstruction loss (Matryoshka)
                reconstructions = []
                for dim in mat_dims:
                    x_hat = model.decode(h, max_dim=dim)
                    reconstructions.append(x_hat)
                recon_loss = torch.tensor(0.0, device=device)
                for x_hat, weight in zip(reconstructions, mat_weights):
                    recon_loss = recon_loss + weight * F.mse_loss(x_hat, batch)

                # Entropy regularization within each scale (Archetypal)
                # Encourages peaked/sparse archetype usage at each granularity
                total_entropy = torch.tensor(0.0, device=device)
                prev_dim = 0
                for dim in mat_dims:
                    if dim <= config.hidden_dim:
                        h_scale = h[:, prev_dim:dim]
                        # Normalize to get probabilities for this scale
                        h_norm = h_scale / (h_scale.sum(dim=-1, keepdim=True) + 1e-8)
                        entropy = -(h_norm * torch.log(h_norm + 1e-8)).sum(dim=-1).mean()
                        total_entropy = total_entropy + entropy
                        prev_dim = dim
                sparsity_loss = -config.sparsity_penalty * total_entropy / len(mat_dims)

            else:
                # Standard forward
                if config.use_aux_loss:
                    h, pre_act = model.encode(batch, return_pre_act=True)
                    x_hat = model.decode(h)
                else:
                    x_hat, h = model(batch)
                    pre_act = None
                recon_loss = F.mse_loss(x_hat, batch)

                # Sparsity loss
                if config.sparsity_type == "l1":
                    sparsity_loss = config.sparsity_penalty * h.abs().mean()
                elif config.sparsity_type == "topk":
                    # TopK has implicit sparsity, small L1 for dead feature prevention
                    sparsity_loss = config.sparsity_penalty * 0.1 * h.abs().mean()
                elif config.sparsity_type == "gated":
                    sparsity_loss = config.sparsity_penalty * h.abs().mean()
                else:
                    sparsity_loss = config.sparsity_penalty * h.abs().mean()

            # Auxiliary loss for dead neuron revival
            if config.use_aux_loss and pre_act is not None:
                aux_loss = model.compute_aux_loss(batch, h, pre_act)
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

        # Log dead neuron count if using aux loss
        if config.use_aux_loss and hasattr(model, 'steps_since_active'):
            n_dead = (model.steps_since_active > config.dead_threshold).sum().item()
        else:
            n_dead = 0

        if verbose and (epoch + 1) % 20 == 0:
            msg = f"  Epoch {epoch+1}/{config.n_epochs}: recon={epoch_recon:.6f}, sparsity={epoch_sparse:.6f}"
            if config.use_aux_loss:
                msg += f", aux={epoch_aux:.6f}, dead={n_dead}"
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
        total_loss=history["total_loss"][-1],
        feature_activations=all_activations,
        feature_frequencies=feature_frequencies,
        mean_active_features=float(mean_active),
        dead_features=int(dead_features),
        alive_features=int(config.hidden_dim - dead_features),
        config=config,
    )

    return model, result


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
