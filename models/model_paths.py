"""Central registry for model checkpoint paths.

Maps model name → checkpoint path, checking worker locations first,
then local/HuggingFace fallbacks. Import MODEL_PATHS or call
get_model_path(model_name) from anywhere.
"""
import os
import socket
from pathlib import Path

# Worker-side paths (all workers use the same layout)
_WORKER_BASE = Path("/data/models/tabular_fm")
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# Model path registry: (worker_path, local_fallbacks...)
_REGISTRY = {
    "hyperfast": [
        _WORKER_BASE / "hyperfast" / "hyperfast.ckpt",
        Path.home() / ".hyperfast" / "hyperfast.ckpt",
        Path("hyperfast.ckpt"),
    ],
    "tabula8b": [
        Path("/data/models/tabula-8b"),
        _HF_CACHE / "models--mlfoundations--tabula-8b",
        "mlfoundations/tabula-8b",  # HuggingFace model ID (string, not Path)
    ],
    "tabpfn": [
        _WORKER_BASE / "tabpfn",
        # TabPFN uses HuggingFace token-gated download; no fixed path
    ],
    "mitra": [
        # Mitra downloads via autogluon; uses HF cache
    ],
    "tabicl": [
        # TabICL downloads via HF
    ],
    "tabicl_v2": [
        # TabICL-v2 downloads via HF
    ],
    "tabdpt": [
        # TabDPT downloads via pip install
    ],
    "carte": [
        # CARTE uses fasttext model
        _WORKER_BASE / "fasttext" / "cc.en.300.bin",
        Path.home() / ".fasttext" / "cc.en.300.bin",
    ],
}


def get_model_path(model_name: str) -> str:
    """Return the first existing checkpoint path for the given model.

    Checks worker paths first, then local fallbacks. Returns a string
    (file path or HuggingFace model ID).

    Raises FileNotFoundError if no checkpoint is found.
    """
    candidates = _REGISTRY.get(model_name, [])
    for path in candidates:
        if isinstance(path, str):
            # HuggingFace model ID — return as-is
            return path
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        f"No checkpoint found for '{model_name}'. "
        f"Searched: {[str(p) for p in candidates]}"
    )


# Convenience dict for quick access
MODEL_PATHS = {}
for _name in _REGISTRY:
    try:
        MODEL_PATHS[_name] = get_model_path(_name)
    except FileNotFoundError:
        pass
