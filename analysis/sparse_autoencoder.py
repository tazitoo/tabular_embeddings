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
    sparsity_type: str = "l1"  # "l1", "topk", or "gated"
    topk: int = 32  # For topk sparsity

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse hidden representation."""
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

        else:
            # Standard L1: just ReLU
            h = F.relu(pre_act)

        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode sparse representation back to input space."""
        if self.config.tied_weights:
            # W_enc is (hidden_dim, input_dim), so W_enc.T is (input_dim, hidden_dim)
            # F.linear computes h @ W.T, so we need W of shape (input_dim, hidden_dim)
            return F.linear(h, self.W_enc.T, self.b_dec)
        else:
            # W_dec is (input_dim, hidden_dim)
            # F.linear computes h @ W_dec.T = h @ (hidden_dim, input_dim) = correct
            return F.linear(h, self.W_dec, self.b_dec)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (reconstruction, hidden_activations)."""
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h

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

    # Training loop
    history = {"recon_loss": [], "sparsity_loss": [], "total_loss": []}

    for epoch in range(config.n_epochs):
        epoch_recon = 0.0
        epoch_sparse = 0.0
        n_batches = 0

        for (batch,) in loader:
            batch = batch.to(device)

            # Forward
            x_hat, h = model(batch)

            # Reconstruction loss
            recon_loss = F.mse_loss(x_hat, batch)

            # Sparsity loss
            if config.sparsity_type == "l1":
                sparsity_loss = config.sparsity_penalty * h.abs().mean()
            elif config.sparsity_type == "topk":
                # TopK has implicit sparsity, small L1 for dead feature prevention
                sparsity_loss = config.sparsity_penalty * 0.1 * h.abs().mean()
            else:
                sparsity_loss = config.sparsity_penalty * h.abs().mean()

            loss = recon_loss + sparsity_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalize decoder periodically
            if n_batches % 10 == 0:
                model.normalize_decoder()

            epoch_recon += recon_loss.item()
            epoch_sparse += sparsity_loss.item()
            n_batches += 1

        epoch_recon /= n_batches
        epoch_sparse /= n_batches
        history["recon_loss"].append(epoch_recon)
        history["sparsity_loss"].append(epoch_sparse)
        history["total_loss"].append(epoch_recon + epoch_sparse)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{config.n_epochs}: "
                  f"recon={epoch_recon:.6f}, sparsity={epoch_sparse:.6f}")

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
) -> Dict:
    """
    Measure the "richness" of a learned dictionary.

    A richer dictionary suggests the model has learned more concepts.

    Metrics:
    - Alive features: Non-dead dictionary elements
    - Effective dimensions: Entropy-based count of independent features
    - Specialization: How focused features are (vs uniform)
    - Coverage: How well dictionary spans input space

    Args:
        sae_result: SAE training result
        input_features: Optional original input features for coverage analysis

    Returns:
        Dict with richness metrics
    """
    freqs = sae_result.feature_frequencies
    activations = sae_result.feature_activations
    dictionary = sae_result.dictionary

    # 1. Alive features (non-dead)
    alive = sae_result.alive_features
    total = len(freqs)

    # 2. Effective dimensions via entropy
    # Normalize frequencies to probabilities
    freq_norm = freqs / (freqs.sum() + 1e-8)
    entropy = -np.sum(freq_norm * np.log(freq_norm + 1e-8))
    effective_dims = np.exp(entropy)  # Perplexity

    # 3. Specialization: average sparsity of activations
    sparsity = (activations == 0).mean()

    # 4. Dictionary diversity: average pairwise distance
    dict_norm = dictionary / (np.linalg.norm(dictionary, axis=1, keepdims=True) + 1e-8)
    pairwise_sim = dict_norm @ dict_norm.T
    # Exclude diagonal
    np.fill_diagonal(pairwise_sim, 0)
    mean_pairwise_sim = pairwise_sim.sum() / (len(dictionary) * (len(dictionary) - 1))
    diversity = 1 - mean_pairwise_sim  # Higher = more diverse

    # 5. Reconstruction quality (implicit from loss)
    recon_quality = 1 / (1 + sae_result.reconstruction_loss)

    # 6. Composite richness score
    # Normalize components to [0, 1] and combine
    alive_score = alive / total
    diversity_score = diversity
    sparsity_score = sparsity  # Higher sparsity = more interpretable

    richness_score = (
        0.3 * alive_score +
        0.3 * diversity_score +
        0.2 * sparsity_score +
        0.2 * recon_quality
    )

    return {
        "alive_features": alive,
        "dead_features": total - alive,
        "alive_ratio": alive_score,
        "effective_dimensions": float(effective_dims),
        "sparsity": float(sparsity),
        "dictionary_diversity": float(diversity),
        "reconstruction_quality": float(recon_quality),
        "richness_score": float(richness_score),
        "mean_active_per_sample": sae_result.mean_active_features,
    }


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
    # Demo with random embeddings
    print("Testing Sparse Autoencoder module...\n")

    np.random.seed(42)
    torch.manual_seed(42)

    # Generate test embeddings
    n_samples = 1000
    embedding_dim = 64
    embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)

    # Configure SAE
    config = SAEConfig(
        input_dim=embedding_dim,
        hidden_dim=256,  # 4x overcomplete
        sparsity_penalty=1e-3,
        sparsity_type="l1",
        n_epochs=50,
    )

    print(f"Training SAE: {embedding_dim}D -> {config.hidden_dim}D")
    print(f"Sparsity: {config.sparsity_type}, penalty={config.sparsity_penalty}")

    model, result = train_sae(embeddings, config, verbose=True)

    print(f"\nResults:")
    print(f"  Reconstruction loss: {result.reconstruction_loss:.6f}")
    print(f"  Sparsity loss: {result.sparsity_loss:.6f}")
    print(f"  Alive features: {result.alive_features}/{config.hidden_dim}")
    print(f"  Mean active per sample: {result.mean_active_features:.1f}")

    # Richness analysis
    richness = measure_dictionary_richness(result)
    print(f"\nRichness Metrics:")
    for k, v in richness.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Geometry analysis
    geometry = analyze_feature_geometry(result.dictionary, result.feature_activations)
    print(f"\nGeometry Metrics:")
    print(f"  Power law alpha: {geometry['power_law_alpha']:.3f}")
    print(f"  Mean clustering: {geometry['mean_clustering']:.3f}")
    print(f"  Mean co-activation: {geometry['mean_coactivation']:.3f}")

    print("\nSAE module test complete!")
