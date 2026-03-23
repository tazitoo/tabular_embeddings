"""
TabPFN loading utilities.

Centralizes checkpoint resolution and telemetry suppression so every script
that uses TabPFN goes through one path.

Usage:
    from models.tabpfn_utils import load_tabpfn

    clf = load_tabpfn(task="classification", device="cuda")
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)
"""

import os

# Suppress TabPFN/posthog telemetry before any tabpfn import
os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
os.environ["POSTHOG_DISABLED"] = "1"

# Worker checkpoint paths (avoids HuggingFace download on GPU nodes)
CHECKPOINT_PATHS = {
    "classification": "/data/models/tabular_fm/tabpfn/tabpfn-v2.5-classifier-v2.5_real.ckpt",
    "regression": "/data/models/tabular_fm/tabpfn/tabpfn-v2.5-regressor-v2.5_default.ckpt",
}

# TabPFN v2 checkpoints — used for TabArena validation only (same version as benchmark)
CHECKPOINT_PATHS_V2 = {
    "classification": "/data/models/tabular_fm/tabpfn/tabpfn-v2-classifier.ckpt",
    "regression": "/data/models/tabular_fm/tabpfn/tabpfn-v2-regressor.ckpt",
}


def load_tabpfn(
    task: str = "classification",
    device: str = "cuda",
    n_estimators: int = 2,
    model_path: str | None = None,
):
    """Load a TabPFN classifier or regressor with telemetry disabled.

    Automatically resolves worker checkpoint paths. Falls back to HuggingFace
    download if no local checkpoint is found.

    Args:
        task: "classification" or "regression"
        device: Torch device
        n_estimators: Number of TabPFN ensemble members
        model_path: Explicit checkpoint path (overrides auto-detection)

    Returns:
        Fitted-ready TabPFNClassifier or TabPFNRegressor instance
    """
    if task == "regression":
        from tabpfn import TabPFNRegressor as TabPFNModel
    else:
        from tabpfn import TabPFNClassifier as TabPFNModel

    # Resolve checkpoint: explicit > worker path > auto-download
    if model_path is None:
        worker_path = CHECKPOINT_PATHS.get(task, "")
        if os.path.exists(worker_path):
            model_path = worker_path

    kwargs = dict(device=device, n_estimators=n_estimators,
                  ignore_pretraining_limits=True)
    if model_path is not None:
        kwargs["model_path"] = model_path

    return TabPFNModel(**kwargs)
