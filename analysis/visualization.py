"""
Visualization tools for embedding geometry analysis.

Provides:
- t-SNE / UMAP plots of embeddings
- CKA heatmaps
- Similarity distribution plots
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def plot_cka_heatmap(
    cka_matrix: np.ndarray,
    model_names: List[str],
    title: str = "CKA Similarity Matrix",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot CKA similarity matrix as a heatmap.

    Args:
        cka_matrix: (n_models, n_models) CKA scores
        model_names: List of model names
        title: Plot title
        output_path: Optional path to save figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cka_matrix, cmap="RdYlGn", vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(len(model_names)))
    ax.set_yticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_yticklabels(model_names)

    # Annotations
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            text = ax.text(j, i, f"{cka_matrix[i, j]:.3f}",
                          ha="center", va="center",
                          color="black" if cka_matrix[i, j] > 0.5 else "white")

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="CKA Score")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_embedding_comparison(
    embeddings: Dict[str, np.ndarray],
    labels: Optional[np.ndarray] = None,
    method: str = "tsne",
    title: str = "Embedding Comparison",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> None:
    """
    Plot 2D projections of embeddings from multiple models side by side.

    Args:
        embeddings: Dict mapping model_name -> (n_samples, dim) embeddings
        labels: Optional sample labels for coloring
        method: "tsne" or "umap"
        title: Overall title
        output_path: Optional path to save figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    n_models = len(embeddings)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, emb) in zip(axes, embeddings.items()):
        # Compute 2D projection
        if method == "tsne":
            from sklearn.manifold import TSNE
            proj = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb)-1))
            emb_2d = proj.fit_transform(emb)
        elif method == "umap":
            try:
                import umap
                proj = umap.UMAP(n_components=2, random_state=42)
                emb_2d = proj.fit_transform(emb)
            except ImportError:
                from sklearn.manifold import TSNE
                proj = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb)-1))
                emb_2d = proj.fit_transform(emb)
        else:
            # PCA fallback
            from sklearn.decomposition import PCA
            proj = PCA(n_components=2)
            emb_2d = proj.fit_transform(emb)

        # Plot
        if labels is not None:
            scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1],
                               c=labels, cmap="tab10", alpha=0.6, s=20)
        else:
            ax.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.6, s=20)

        ax.set_title(f"{model_name}\n(dim={emb.shape[1]})")
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")

    plt.suptitle(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_similarity_distribution(
    cosine_similarities: Dict[Tuple[str, str], np.ndarray],
    title: str = "Pairwise Cosine Similarity Distributions",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot distribution of cosine similarities for each model pair.

    Args:
        cosine_similarities: Dict mapping (model_a, model_b) -> similarities
        title: Plot title
        output_path: Optional path to save figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    for (m1, m2), sims in cosine_similarities.items():
        label = f"{m1[:10]} vs {m2[:10]}"
        ax.hist(sims, bins=30, alpha=0.5, label=label, density=True)

    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(-1, 1)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_intrinsic_dimensionality(
    embeddings: Dict[str, np.ndarray],
    title: str = "Intrinsic Dimensionality Analysis",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """
    Plot PCA variance explained curves for each model's embeddings.

    Args:
        embeddings: Dict mapping model_name -> embeddings
        title: Plot title
        output_path: Optional path to save figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(embeddings)))

    for (model_name, emb), color in zip(embeddings.items(), colors):
        # Compute PCA
        emb_centered = emb - emb.mean(axis=0)
        _, S, _ = np.linalg.svd(emb_centered, full_matrices=False)

        var_explained = S ** 2
        var_ratio = var_explained / var_explained.sum()
        cumsum = np.cumsum(var_ratio)

        # Left: cumulative variance
        ax1.plot(range(1, len(cumsum) + 1), cumsum, label=model_name[:15], color=color)
        ax1.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)

        # Find 90% threshold
        n_90 = np.searchsorted(cumsum, 0.9) + 1
        ax1.axvline(x=n_90, color=color, linestyle=':', alpha=0.5)

        # Right: bar chart of intrinsic dims
        ax2.bar(model_name[:15], n_90, color=color, alpha=0.7)

    ax1.set_xlabel("Number of Components")
    ax1.set_ylabel("Cumulative Variance Explained")
    ax1.set_title("PCA Variance Curves")
    ax1.legend(loc="lower right")
    ax1.set_xlim(0, min(100, max(emb.shape[1] for emb in embeddings.values())))

    ax2.set_xlabel("Model")
    ax2.set_ylabel("Components for 90% Variance")
    ax2.set_title("Intrinsic Dimensionality")
    ax2.tick_params(axis='x', rotation=45)

    plt.suptitle(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_procrustes_alignment(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    model_a: str,
    model_b: str,
    labels: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> None:
    """
    Visualize Procrustes alignment between two embedding spaces.

    Shows: original A, original B, aligned A overlaid on B.

    Args:
        emb_a: Embeddings from model A
        emb_b: Embeddings from model B
        model_a: Name of model A
        model_b: Name of model B
        labels: Optional labels for coloring
        output_path: Optional path to save figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Project both to 2D using PCA on combined data
    combined = np.vstack([emb_a, emb_b])
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)

    emb_a_2d = combined_2d[:len(emb_a)]
    emb_b_2d = combined_2d[len(emb_a):]

    # Compute Procrustes alignment in 2D
    from analysis.similarity import procrustes_align
    _, _, aligned_a_2d = procrustes_align(emb_a_2d, emb_b_2d)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot 1: Model A
    if labels is not None:
        axes[0].scatter(emb_a_2d[:, 0], emb_a_2d[:, 1], c=labels, cmap="tab10", alpha=0.6)
    else:
        axes[0].scatter(emb_a_2d[:, 0], emb_a_2d[:, 1], alpha=0.6)
    axes[0].set_title(f"{model_a} (Original)")

    # Plot 2: Model B
    if labels is not None:
        axes[1].scatter(emb_b_2d[:, 0], emb_b_2d[:, 1], c=labels, cmap="tab10", alpha=0.6)
    else:
        axes[1].scatter(emb_b_2d[:, 0], emb_b_2d[:, 1], alpha=0.6)
    axes[1].set_title(f"{model_b} (Target)")

    # Plot 3: Aligned A on B
    if labels is not None:
        axes[2].scatter(emb_b_2d[:, 0], emb_b_2d[:, 1], c=labels, cmap="tab10", alpha=0.3, label=model_b)
        axes[2].scatter(aligned_a_2d[:, 0], aligned_a_2d[:, 1], c=labels, cmap="tab10", alpha=0.6, marker='x', label=f"{model_a} (aligned)")
    else:
        axes[2].scatter(emb_b_2d[:, 0], emb_b_2d[:, 1], alpha=0.3, label=model_b)
        axes[2].scatter(aligned_a_2d[:, 0], aligned_a_2d[:, 1], alpha=0.6, marker='x', label=f"{model_a} (aligned)")
    axes[2].set_title("Procrustes Alignment")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    plt.suptitle(f"Embedding Space Alignment: {model_a} -> {model_b}")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Demo with random data
    print("Testing visualization tools...")

    np.random.seed(42)

    # Create test embeddings
    n_samples = 200
    labels = np.random.randint(0, 3, n_samples)

    embeddings = {
        "Model_A": np.random.randn(n_samples, 50).astype(np.float32),
        "Model_B": np.random.randn(n_samples, 30).astype(np.float32),
        "Model_C": np.random.randn(n_samples, 100).astype(np.float32),
    }

    # CKA matrix
    from analysis.similarity import compute_cka_matrix
    cka_matrix, names = compute_cka_matrix(embeddings)

    print("Plotting CKA heatmap...")
    plot_cka_heatmap(cka_matrix, names, title="Test CKA Matrix")

    print("Plotting embedding comparison...")
    plot_embedding_comparison(embeddings, labels=labels, method="pca")

    print("Plotting intrinsic dimensionality...")
    plot_intrinsic_dimensionality(embeddings)

    print("\nVisualization tests complete!")
