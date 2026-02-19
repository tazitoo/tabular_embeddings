"""Configuration management for optimal extraction layers."""

import json
from pathlib import Path
from typing import Dict, Any

_CONFIG_PATH = Path(__file__).parent / "optimal_extraction_layers.json"


def load_optimal_layers() -> Dict[str, Any]:
    """
    Load optimal extraction layer configuration.

    Returns:
        Dictionary mapping model names to their optimal extraction config:
        {
            'tabpfn': {
                'optimal_layer': 17,
                'n_layers': 24,
                'critical_layer_mean': 18,
                'optimal_depth_fraction': 0.708,
                'architecture': 'Transformer (ICL)',
                'extraction_point': 'layer_17',
                'rationale': '...'
            },
            ...
        }
    """
    with open(_CONFIG_PATH) as f:
        return json.load(f)


def get_optimal_layer(model_name: str) -> int:
    """
    Get the optimal extraction layer for a model.

    Args:
        model_name: Model identifier (e.g., 'tabpfn', 'mitra')

    Returns:
        Optimal layer index for extraction

    Raises:
        KeyError: If model not found in config
    """
    config = load_optimal_layers()
    model_key = model_name.lower().replace('-', '').replace('_', '')

    # Try exact match first
    if model_key in config:
        return config[model_key]['optimal_layer']

    # Try prefix match (for tabula8b, tabula-8b, etc.)
    for key in config:
        if model_key.startswith(key) or key.startswith(model_key):
            return config[key]['optimal_layer']

    raise KeyError(f"Model '{model_name}' not found in optimal layers config. "
                   f"Available: {list(config.keys())}")


def get_extraction_dir(model_name: str, base_dir: str = "embeddings/tabarena") -> str:
    """
    Get the embedding directory path for a model using optimal layer.

    Args:
        model_name: Model identifier
        base_dir: Base directory for embeddings

    Returns:
        Path to embedding directory (e.g., 'embeddings/tabarena/tabpfn_layer17')
    """
    from pathlib import Path

    try:
        layer = get_optimal_layer(model_name)
        return str(Path(base_dir) / f"{model_name}_layer{layer}")
    except KeyError:
        # Fallback to base model directory if not in config
        return str(Path(base_dir) / model_name)
