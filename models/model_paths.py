"""Central registry for model checkpoint paths.

Maps model name → checkpoint path, checking worker locations first,
then HuggingFace cache, then HF model IDs as fallback.

Usage:
    from models.model_paths import get_model_path
    path = get_model_path("hyperfast")  # returns first existing path
"""
import os
from pathlib import Path

_WORKER_BASE = Path("/data/models/tabular_fm")
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# Each entry: list of (path_or_hf_id, ...) checked in order.
# Strings that aren't valid paths are treated as HuggingFace model IDs.
_REGISTRY = {
    "tabpfn": [
        _WORKER_BASE / "tabpfn",
        _HF_CACHE / "models--prior-labs--tabpfn-v2-classifier",
        "prior-labs/tabpfn-v2-classifier",
    ],
    "mitra": [
        _HF_CACHE / "models--autogluon--mitra-classifier",
        "autogluon/mitra-classifier",
    ],
    "tabicl": [
        _HF_CACHE / "models--jingang--TabICL-clf",
        _HF_CACHE / "models--jingang--TabICL",
        "jingang/TabICL",
    ],
    "tabicl_v2": [
        _HF_CACHE / "models--jingang--TabICL",
        "jingang/TabICL",
    ],
    "tabdpt": [
        _HF_CACHE / "models--Layer6--TabDPT",
        "Layer6/TabDPT",
    ],
    "hyperfast": [
        _WORKER_BASE / "hyperfast" / "hyperfast.ckpt",
        Path.home() / ".hyperfast" / "hyperfast.ckpt",
        Path("hyperfast.ckpt"),
    ],
    "carte": [
        # CARTE uses fasttext for graph construction
        _WORKER_BASE / "fasttext" / "cc.en.300.bin",
        Path.home() / ".fasttext" / "cc.en.300.bin",
        Path("/data/models/fasttext/cc.en.300.bin"),
    ],
    "tabula8b": [
        Path("/data/models/tabula-8b"),
        _HF_CACHE / "models--mlfoundations--tabula-8b",
        "mlfoundations/tabula-8b",
    ],
}


def get_model_path(model_name: str) -> str:
    """Return the first existing checkpoint path for the given model.

    Checks local/worker paths first, then returns HF model ID as fallback.
    Raises FileNotFoundError if no path found and no HF fallback exists.
    """
    candidates = _REGISTRY.get(model_name)
    if candidates is None:
        raise KeyError(f"Unknown model: '{model_name}'. Known: {list(_REGISTRY.keys())}")

    last_hf_id = None
    for path in candidates:
        if isinstance(path, str) and not os.path.sep in path:
            # Looks like a HuggingFace model ID (e.g. "autogluon/mitra-classifier")
            last_hf_id = path
            continue
        p = Path(path)
        if p.exists():
            return str(p)

    # No local path found — return HF ID if available
    if last_hf_id is not None:
        return last_hf_id

    raise FileNotFoundError(
        f"No checkpoint found for '{model_name}'. "
        f"Searched: {[str(p) for p in candidates]}"
    )
