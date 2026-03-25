"""Central registry for model checkpoint paths.

Maps model name → checkpoint path, checking worker locations first,
then HuggingFace cache, then HF model IDs as fallback.

Usage:
    from models.model_paths import get_model_path
    path = get_model_path("hyperfast")
    path = get_model_path("tabpfn", task="regression")
"""
import os
from pathlib import Path

_WORKER_BASE = Path("/data/models/tabular_fm")
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# Each entry: list of (path, ...) checked in order.
# Strings without os.sep are treated as HuggingFace model IDs.
_REGISTRY = {
    # TabPFN v2.5 — task-specific checkpoints (real data variant)
    "tabpfn_classifier": [
        _WORKER_BASE / "tabpfn" / "tabpfn-v2.5-classifier-v2.5_real.ckpt",
    ],
    "tabpfn_regressor": [
        _WORKER_BASE / "tabpfn" / "tabpfn-v2.5-regressor-v2.5_real.ckpt",
    ],
    "tabpfn": [
        _WORKER_BASE / "tabpfn",
    ],
    "mitra": [
        _HF_CACHE / "models--autogluon--mitra-classifier",
        "autogluon/mitra-classifier",
    ],
    "mitra_regressor": [
        _HF_CACHE / "models--autogluon--mitra-regressor",
        "autogluon/mitra-regressor",
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
    ],
    "carte": [
        _WORKER_BASE / "fasttext" / "cc.en.300.bin",
        Path.home() / ".fasttext" / "cc.en.300.bin",
    ],
    "tabula8b": [
        Path("/data/models/tabula-8b"),
        _HF_CACHE / "models--mlfoundations--tabula-8b",
        "mlfoundations/tabula-8b",
    ],
}


def get_model_path(model_name: str, task: str = "classification") -> str:
    """Return the first existing checkpoint path for the given model.

    For models with task-specific checkpoints (e.g. TabPFN), pass
    task="regression" to get the regressor variant.

    Returns a string: local file path or HuggingFace model ID.
    """
    # Try task-specific key first (e.g. "tabpfn_regressor")
    if task == "regression":
        task_key = f"{model_name}_regressor"
        if task_key in _REGISTRY:
            try:
                return _resolve(task_key)
            except FileNotFoundError:
                pass

    return _resolve(model_name)


def _resolve(key: str) -> str:
    candidates = _REGISTRY.get(key)
    if candidates is None:
        raise KeyError(f"Unknown model: '{key}'. Known: {list(_REGISTRY.keys())}")

    last_hf_id = None
    for path in candidates:
        if isinstance(path, str) and os.sep not in path:
            last_hf_id = path
            continue
        if Path(path).exists():
            return str(path)

    if last_hf_id is not None:
        return last_hf_id

    raise FileNotFoundError(
        f"No checkpoint found for '{key}'. "
        f"Searched: {[str(p) for p in candidates]}"
    )
