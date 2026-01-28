"""Embedding extraction wrappers for tabular foundation models."""

from .base import EmbeddingExtractor

__all__ = ["EmbeddingExtractor", "get_extractor", "MODEL_REGISTRY"]

# Registry: prefix -> (module_path, class_name, default_device)
MODEL_REGISTRY = {
    "tabpfn": ("models.tabpfn_embeddings", "TabPFNEmbeddingExtractor", "cpu"),
    "hyperfast": ("models.hyperfast_embeddings", "HyperFastEmbeddingExtractor", "cuda"),
    "tabicl": ("models.tabicl_embeddings", "TabICLEmbeddingExtractor", "cpu"),
    "tabdpt": ("models.tabdpt_embeddings", "TabDPTEmbeddingExtractor", "cpu"),
    "mitra": ("models.mitra_embeddings", "MitraEmbeddingExtractor", "cpu"),
    "mothernet": ("models.mothernet_embeddings", "MotherNetEmbeddingExtractor", "cpu"),
    "carte": ("models.carte_embeddings", "CARTEEmbeddingExtractor", "cpu"),
}


def get_extractor(model_name: str, device: str = "cpu") -> EmbeddingExtractor:
    """
    Factory function to create an embedding extractor by model name.

    Args:
        model_name: Model identifier (matched by prefix against registry)
        device: Torch device. For models with a CUDA default (e.g. hyperfast),
                the default is used unless device is explicitly "cpu".

    Returns:
        Instantiated EmbeddingExtractor subclass

    Raises:
        ValueError: If model_name doesn't match any registered model
    """
    import importlib

    name_lower = model_name.lower()

    for prefix, (module_path, class_name, default_device) in MODEL_REGISTRY.items():
        if name_lower.startswith(prefix):
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            # Use model's preferred device unless caller explicitly set cpu
            effective_device = default_device if device != "cpu" else device
            # But if caller requests cuda/mps, always respect that
            if device in ("cuda", "mps"):
                effective_device = device
            return cls(device=effective_device)

    available = ", ".join(sorted(MODEL_REGISTRY.keys()))
    raise ValueError(f"Unknown model: {model_name}. Available: {available}")
