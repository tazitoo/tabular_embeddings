# Intervention Backbone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `intervene_lib.py` (shared intervention library) and `perrow_importance.py` (first consumer) that compute per-row, per-feature importance across all models via batched SAE ablation.

**Architecture:** Add `recapture()` to existing tail classes in `intervene_sae.py` (no import breakage). Create `intervene_lib.py` with new shared routines (data loading, delta computation, batched ablation) that import from existing modules. Create thin `perrow_importance.py` consumer.

**Tech Stack:** PyTorch, NumPy, scikit-learn. Runs on GPU workers (24GB VRAM). Data from preprocessing cache (round 9) and SAE training data (round 10).

**Spec:** `docs/plans/2026-03-22-perrow-importance.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `scripts/intervention/intervene_sae.py` | Modify (add `recapture()` to 6 tail classes) | Tail model classes stay here; 16+ files import from it |
| `scripts/intervention/intervene_lib.py` | Create | Shared routines: data loading, encoding, delta computation, batched ablation, metrics re-exports |
| `scripts/intervention/perrow_importance.py` | Create | Thin CLI consumer: per-dataset loop, resume, output saving |
| `tests/test_intervene_lib.py` | Create | Tests for new library functions |
| `tests/test_intervene_sae.py` | Modify (add recapture tests) | Tests for recapture() |

**Not touched (yet):** `concept_importance.py`, `09_perrow_importance.py` — archived after all 3 consumers (importance, ablation, transfer) are working.

---

### Task 1: Add `recapture()` to TabPFNTail

The key new method. Separates "fit" from "capture hidden state" so the model is fit once per dataset and recapture is called per query row with K copies.

**Files:**
- Modify: `scripts/intervention/intervene_sae.py:280-390` (TabPFNTail class)
- Test: `tests/test_intervene_sae.py`

- [ ] **Step 1: Write the test**

Add to `tests/test_intervene_sae.py`:

```python
class TestTabPFNRecapture:
    """Test that recapture() re-captures hidden state without re-fitting."""

    def test_recapture_updates_hidden_state(self, mock_sae):
        """After recapture with different query, hidden_state shape changes."""
        # This test requires a fitted TabPFNTail — use a mock or skip on CI
        pytest.importorskip("tabpfn")
        from scripts.intervention.intervene_sae import TabPFNTail

        # Minimal synthetic data
        rng = np.random.RandomState(42)
        X_ctx = rng.randn(20, 5).astype(np.float32)
        y_ctx = (rng.rand(20) > 0.5).astype(np.int32)
        X_q = rng.randn(3, 5).astype(np.float32)

        tail = TabPFNTail.from_data(X_ctx, y_ctx, X_q, extraction_layer=10,
                                     task="classification", device="cpu")
        assert tail.n_query == 3
        old_shape = tail.hidden_state.shape

        # Recapture with 5 query copies (simulating K=5 ablation)
        X_q5 = np.tile(X_q[0:1], (5, 1))
        preds = tail.recapture(X_q5)
        assert tail.n_query == 5
        assert tail.hidden_state.shape[1] == old_shape[1] - 3 + 5
        assert preds.shape[0] == 5

    def test_recapture_preserves_fitted_model(self, mock_sae):
        """Model predictions should be consistent before/after recapture."""
        pytest.importorskip("tabpfn")
        from scripts.intervention.intervene_sae import TabPFNTail

        rng = np.random.RandomState(42)
        X_ctx = rng.randn(20, 5).astype(np.float32)
        y_ctx = (rng.rand(20) > 0.5).astype(np.int32)
        X_q = rng.randn(2, 5).astype(np.float32)

        tail = TabPFNTail.from_data(X_ctx, y_ctx, X_q, extraction_layer=10,
                                     task="classification", device="cpu")
        baseline = tail.baseline_preds.copy()

        # Recapture with same query → same predictions
        preds = tail.recapture(X_q)
        np.testing.assert_allclose(preds, baseline, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_intervene_sae.py::TestTabPFNRecapture -v`
Expected: FAIL — `TabPFNTail` has no `recapture` method

- [ ] **Step 3: Implement `recapture()` on TabPFNTail**

In `scripts/intervention/intervene_sae.py`, add after `predict_row()` (around line 389):

```python
def recapture(self, X_query_new):
    """Re-capture hidden state with new query data. Model stays fitted.

    Use this to swap in K copies of a single row for batched ablation
    without the cost of re-fitting the model.

    Returns:
        preds: (n_query_new, ...) baseline predictions for new query
    """
    captured = {}

    def capture_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured["hidden"] = output.detach()

    handle = self.layers[self.extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            if self.task == "regression":
                preds = self.clf.predict(X_query_new)
            else:
                preds = self.clf.predict_proba(X_query_new)
    finally:
        handle.remove()

    self.hidden_state = captured["hidden"]
    self.n_query = len(X_query_new)
    self.X_query = X_query_new
    self.single_eval_pos = captured["hidden"].shape[1] - self.n_query
    self.baseline_preds = np.asarray(preds)
    return self.baseline_preds
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_intervene_sae.py::TestTabPFNRecapture -v`
Expected: PASS (or SKIP if tabpfn not installed locally)

- [ ] **Step 5: Commit**

```bash
git add scripts/intervention/intervene_sae.py tests/test_intervene_sae.py
git commit -m "feat: add recapture() to TabPFNTail for batched ablation"
```

---

### Task 2: Add `recapture()` to remaining 5 tail classes

Same pattern as Task 1 for TabICL, TabICL-v2, TabDPT, Mitra, HyperFast.
CARTE and Tabula-8B deferred (no preprocessing cache).

**Files:**
- Modify: `scripts/intervention/intervene_sae.py` (5 tail classes)
- Test: `tests/test_intervene_sae.py`

- [ ] **Step 1: Add `recapture()` to TabICLTail** (after line ~491)

```python
def recapture(self, X_query_new):
    """Re-capture hidden state with new query data. Model stays fitted."""
    captured = {}

    def capture_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured["hidden"] = output.detach()

    handle = self.blocks[self.extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            preds = self.clf.predict_proba(X_query_new)
    finally:
        handle.remove()

    self.hidden_state = captured["hidden"]
    self.n_query = len(X_query_new)
    self.X_query = X_query_new
    self.train_size = captured["hidden"].shape[1] - self.n_query
    self.baseline_preds = np.asarray(preds)
    return self.baseline_preds
```

- [ ] **Step 2: Add `recapture()` to TabICLV2Tail** (after line ~601)

Same as TabICL but handles regression:

```python
def recapture(self, X_query_new):
    """Re-capture hidden state with new query data. Model stays fitted."""
    captured = {}

    def capture_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured["hidden"] = output.detach()

    handle = self.blocks[self.extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            if self.task == "regression":
                preds = self.clf.predict(X_query_new)
            else:
                preds = self.clf.predict_proba(X_query_new)
    finally:
        handle.remove()

    self.hidden_state = captured["hidden"]
    self.n_query = len(X_query_new)
    self.X_query = X_query_new
    self.train_size = captured["hidden"].shape[1] - self.n_query
    self.baseline_preds = np.asarray(preds)
    return self.baseline_preds
```

- [ ] **Step 3: Add `recapture()` to TabDPTTail** (after line ~1197)

```python
def recapture(self, X_query_new):
    """Re-capture hidden state with new query data. Model stays fitted."""
    captured = {}

    def capture_hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            captured["hidden"] = out.detach()

    handle = self.encoder_layers[self.extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            if self.task == "regression":
                preds = self.clf.predict(X_query_new)
            else:
                preds = self.clf.predict_proba(X_query_new)
    finally:
        handle.remove()

    self.hidden_state = captured["hidden"]
    self.n_query = len(X_query_new)
    self.n_ctx = self.hidden_state.shape[0] - self.n_query if self.hidden_state.ndim == 2 else self.n_ctx
    self.X_query = X_query_new
    self.baseline_preds = np.asarray(preds)
    return self.baseline_preds
```

- [ ] **Step 4: Add `recapture()` to MitraTail** (after line ~1079)

Mitra captures (support, query) tuples — must mirror `from_data()` logic:

```python
def recapture(self, X_query_new):
    """Re-capture hidden state with new query data. Model stays fitted."""
    captured_support = []
    captured_query = []

    def capture_hook(module, input, output):
        if isinstance(output, tuple) and len(output) >= 2:
            sup, qry = output[0], output[1]
            if isinstance(sup, torch.Tensor):
                captured_support.append(sup.detach())
            if isinstance(qry, torch.Tensor):
                captured_query.append(qry.detach())

    trainer = self.clf.trainers[0]
    rng_state = trainer.rng.get_state()

    handle = self.layers[self.extraction_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            if self.task == "regression":
                preds = self.clf.predict(X_query_new)
            else:
                preds = self.clf.predict_proba(X_query_new)
    finally:
        handle.remove()
        trainer.rng.set_state(rng_state)

    self.captured_support = captured_support
    self.captured_query = captured_query
    self.n_query = len(X_query_new)
    self.X_query = X_query_new
    self.baseline_preds = np.asarray(preds)
    return self.baseline_preds
```

- [ ] **Step 5: Add `recapture()` to HyperFastTail** (after line ~1316)

HyperFast caches intermediates per ensemble member:

```python
def recapture(self, X_query_new):
    """Re-capture intermediates with new query data. Model stays fitted."""
    X_t = torch.tensor(X_query_new, dtype=torch.float32).to(self.device)
    intermediates = []
    for jj in range(len(self.main_networks)):
        with torch.no_grad():
            x = X_t
            for layer_idx in range(self.extraction_layer + 1):
                x = self.main_networks[jj][layer_idx](x)
                if layer_idx < len(self.main_networks[jj]) - 1:
                    x = torch.relu(x)
            intermediates.append(x.detach())
    self.intermediates = intermediates
    self.n_query = len(X_query_new)
    self.X_query = X_query_new

    # Get baseline predictions
    preds = self.predict(torch.zeros_like(intermediates[0]))
    self.baseline_preds = np.asarray(preds)
    return self.baseline_preds
```

Note: HyperFast's `recapture` is different — it replays the generated MLP
up to the extraction layer. Read `HyperFastTail.from_data()` carefully during
implementation to get the exact intermediate caching logic right. The code
above is a sketch; the implementer must match `from_data()`'s forward logic
exactly.

- [ ] **Step 6: Commit**

```bash
git add scripts/intervention/intervene_sae.py
git commit -m "feat: add recapture() to 5 remaining tail classes"
```

---

### Task 3: Create `intervene_lib.py` — metrics and data loading

The CPU-only portion of the library. No GPU needed to test.

**Files:**
- Create: `scripts/intervention/intervene_lib.py`
- Create: `tests/test_intervene_lib.py`

- [ ] **Step 1: Write tests for metrics (re-exported from concept_importance)**

```python
"""Tests for intervene_lib shared library."""
import numpy as np
import pytest


class TestComputePerRowLoss:
    def test_binary_classification(self):
        from scripts.intervention.intervene_lib import compute_per_row_loss
        y = np.array([0, 1, 1])
        preds = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
        loss = compute_per_row_loss(y, preds, "classification")
        assert loss.shape == (3,)
        assert loss[0] < loss[2]  # correct pred has lower loss
        assert loss[1] < loss[2]

    def test_regression(self):
        from scripts.intervention.intervene_lib import compute_per_row_loss
        y = np.array([1.0, 2.0, 3.0])
        preds = np.array([1.1, 2.5, 3.0])
        loss = compute_per_row_loss(y, preds, "regression")
        assert loss.shape == (3,)
        assert loss[2] == pytest.approx(0.0)
        assert loss[1] > loss[0]
```

- [ ] **Step 2: Write tests for row index alignment**

```python
class TestRowIndexAlignment:
    def test_abs_to_position_mapping(self):
        from scripts.intervention.intervene_lib import align_test_rows
        # Holdout fold indices (not necessarily sorted)
        holdout_indices = np.array([50, 10, 30, 20, 40])
        # Test embedding rows map to absolute indices 30 and 10
        test_row_indices = np.array([30, 10])
        positions = align_test_rows(holdout_indices, test_row_indices)
        # 30 is at position 2, 10 is at position 1
        np.testing.assert_array_equal(positions, [2, 1])

    def test_missing_index_raises(self):
        from scripts.intervention.intervene_lib import align_test_rows
        holdout_indices = np.array([10, 20, 30])
        test_row_indices = np.array([10, 99])  # 99 not in holdout
        with pytest.raises(KeyError):
            align_test_rows(holdout_indices, test_row_indices)
```

- [ ] **Step 3: Write tests for delta computation**

```python
class TestComputeFeatureDeltas:
    def test_delta_shape_and_sign(self):
        """Ablating a feature produces a non-zero delta of correct shape."""
        import torch
        from scripts.intervention.intervene_lib import compute_feature_deltas

        # Mock SAE: encode returns input, decode returns input
        class MockSAE:
            def encode(self, x): return torch.relu(x)
            def decode(self, h): return h

        sae = MockSAE()
        # Row activation: 3 features, features 0 and 2 are active
        h_row = torch.tensor([1.5, 0.0, 0.8])
        feature_indices = [0, 2]  # firing features
        data_std = torch.ones(3)

        deltas = compute_feature_deltas(sae, h_row, feature_indices, data_std)
        assert deltas.shape == (2, 3)  # K=2 features, emb_dim=3
        # Ablating feature 0: delta should be non-zero at position 0
        assert deltas[0, 0] != 0.0
        # Ablating feature 2: delta should be non-zero at position 2
        assert deltas[1, 2] != 0.0

    def test_data_std_denormalization(self):
        """Delta is multiplied by data_std to get raw-space delta."""
        import torch
        from scripts.intervention.intervene_lib import compute_feature_deltas

        class MockSAE:
            def encode(self, x): return torch.relu(x)
            def decode(self, h): return h

        sae = MockSAE()
        h_row = torch.tensor([1.0, 0.0, 1.0])
        data_std_1 = torch.ones(3)
        data_std_2 = torch.tensor([2.0, 2.0, 2.0])

        d1 = compute_feature_deltas(sae, h_row, [0], data_std_1)
        d2 = compute_feature_deltas(sae, h_row, [0], data_std_2)
        # d2 should be 2x d1 due to std scaling
        torch.testing.assert_close(d2, d1 * 2.0)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_intervene_lib.py -v`
Expected: FAIL — module not found

- [ ] **Step 5: Create `intervene_lib.py` with metrics, alignment, and delta computation**

```python
"""Shared intervention library for importance, ablation, and transfer.

Provides:
- Data loading and row alignment (CPU)
- SAE encoding and delta computation (GPU)
- Batched ablation via tail model recapture (GPU)
- Loss metrics (CPU)

Usage:
    from scripts.intervention.intervene_lib import (
        load_dataset_context, encode_test_embeddings,
        compute_feature_deltas, batched_ablation,
        compute_per_row_loss, compute_importance_metric,
    )
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.sae.compare_sae_cross_model import DEFAULT_SAE_ROUND

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

SPLITS_PATH = PROJECT_ROOT / "output" / "sae_training_round9" / "tabarena_splits.json"
SAE_DATA_DIR = PROJECT_ROOT / "output" / f"sae_training_round{DEFAULT_SAE_ROUND}"

# ── Re-exports from existing modules ────────────────────────────────────────

from scripts.intervention.concept_importance import (  # noqa: E402
    compute_per_row_loss,
    compute_importance_metric,
    get_alive_features,
    get_feature_labels,
    get_matryoshka_bands,
    MODEL_KEY_TO_LABEL_KEY,
)
from scripts.intervention.intervene_sae import (  # noqa: E402
    load_sae,
    load_norm_stats,
    get_extraction_layer,
    build_tail,
)
from scripts.matching.utils import load_norm_stats as load_norm_stats_matching  # noqa: E402
from data.preprocessing import load_preprocessed, CACHE_DIR  # noqa: E402

__all__ = [
    "compute_per_row_loss", "compute_importance_metric",
    "get_alive_features", "get_feature_labels", "get_matryoshka_bands",
    "load_sae", "load_norm_stats", "get_extraction_layer", "build_tail",
    "load_dataset_context", "encode_test_embeddings",
    "compute_feature_deltas", "batched_ablation", "align_test_rows",
]


# ── Row alignment ───────────────────────────────────────────────────────────


def align_test_rows(
    holdout_indices: np.ndarray,
    test_row_indices: np.ndarray,
) -> np.ndarray:
    """Map absolute dataset row indices to positions in X_test.

    The preprocessing cache orders X_test by holdout_indices (from
    tabarena_splits.json). The SAE test embeddings reference absolute
    row indices. This maps between the two.

    Args:
        holdout_indices: splits[ds]["test_indices"], defines X_test ordering
        test_row_indices: absolute indices for this dataset's test embeddings

    Returns:
        positions: (n_test,) array of indices into X_test
    """
    abs_to_pos = {int(idx): pos for pos, idx in enumerate(holdout_indices)}
    return np.array([abs_to_pos[int(ri)] for ri in test_row_indices])


def load_dataset_context(
    model_key: str,
    dataset: str,
    splits: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Load preprocessed data and resolve row alignment for one dataset.

    Returns:
        X_train, y_train: context arrays for tail model
        X_query, y_query: query arrays (aligned to test embeddings)
        row_indices: absolute dataset row indices for the query rows
        task: "classification" or "regression"
    """
    if splits is None:
        splits = json.loads(SPLITS_PATH.read_text())

    split_info = splits[dataset]
    task = split_info["task_type"]
    holdout_indices = np.array(split_info["test_indices"])

    data = load_preprocessed(model_key, dataset, CACHE_DIR)

    # Load test_row_indices from the SAE training data NPZ
    # These are saved by 06_build_sae_training_data.py
    test_npz_candidates = sorted(SAE_DATA_DIR.glob(f"{model_key}_*_sae_test.npz"))
    if not test_npz_candidates:
        raise FileNotFoundError(f"No test embeddings for {model_key} in {SAE_DATA_DIR}")
    test_data = np.load(test_npz_candidates[0], allow_pickle=True)

    # Unpool this dataset's row indices
    spd = test_data["samples_per_dataset"]
    offset = 0
    test_row_indices = None
    for ds_name, count in spd:
        ds_name = str(ds_name)
        count = int(count)
        if ds_name == dataset:
            # Row indices may be stored, or we replay the selection
            if "row_indices" in test_data:
                test_row_indices = test_data["row_indices"][offset:offset + count]
            break
        offset += count

    if test_row_indices is None:
        raise ValueError(
            f"Dataset {dataset} not found in test embeddings, or row_indices "
            f"not saved. Re-run 06_build_sae_training_data.py to save them."
        )

    positions = align_test_rows(holdout_indices, test_row_indices)
    X_query = data.X_test[positions]
    y_query = data.y_test[positions]

    return data.X_train, data.y_train, X_query, y_query, test_row_indices, task


# ── SAE encoding ─────────────────────────────────────────────────────────────


def encode_test_embeddings(
    sae: torch.nn.Module,
    dataset: str,
    device: str = "cpu",
) -> Tuple[torch.Tensor, np.ndarray, List[int]]:
    """Load and encode one dataset's test embeddings through the SAE.

    Test embeddings are already normalized — pass directly to sae.encode().

    Returns:
        activations: (n_query, hidden_dim) tensor
        firing_mask: (n_query, hidden_dim) bool array
        alive_features: sorted list of feature indices firing on any row
    """
    test_npz_candidates = sorted(SAE_DATA_DIR.glob(f"*_sae_test.npz"))
    # Find the right model's test file (caller should have loaded SAE already)
    # For now, load all and unpool
    # TODO: accept pre-unpooled embeddings to avoid re-loading

    from scripts.matching.utils import load_test_embeddings
    # This returns Dict[dataset_name, embeddings_array]
    model_key = None
    for p in sorted(SAE_DATA_DIR.glob("*_sae_test.npz")):
        model_key = p.name.split("_taskaware_")[0] if "_taskaware_" in p.name else p.name.split("_")[0]
        break

    raise NotImplementedError(
        "encode_test_embeddings needs model_key parameter — "
        "see Task 4 for the full implementation that receives pre-loaded data"
    )


def compute_feature_deltas(
    sae: torch.nn.Module,
    h_row: torch.Tensor,
    feature_indices: List[int],
    data_std: torch.Tensor,
) -> torch.Tensor:
    """Compute per-feature ablation deltas for one row.

    For each feature in feature_indices, zeros it in the SAE latent space,
    decodes, and computes the delta in raw embedding space.

    Args:
        sae: Trained SAE in eval mode
        h_row: (hidden_dim,) SAE activations for this row (already encoded)
        feature_indices: which features to ablate (the firing features)
        data_std: (emb_dim,) per-dataset std for denormalization

    Returns:
        deltas: (K, emb_dim) tensor in raw embedding space
    """
    K = len(feature_indices)
    with torch.no_grad():
        recon_full = sae.decode(h_row.unsqueeze(0))  # (1, emb_dim)

        # Batch: create K copies with one feature zeroed each
        h_batch = h_row.unsqueeze(0).expand(K, -1).clone()  # (K, hidden_dim)
        for k, fi in enumerate(feature_indices):
            h_batch[k, fi] = 0.0

        recon_ablated = sae.decode(h_batch)  # (K, emb_dim)
        delta_norm = recon_ablated - recon_full  # (K, emb_dim)
        delta_raw = delta_norm * data_std.unsqueeze(0)  # denormalize

    return delta_raw


# ── Batched ablation ─────────────────────────────────────────────────────────


def batched_ablation(
    tail,
    X_row: np.ndarray,
    deltas: torch.Tensor,
    max_K: int = 512,
) -> np.ndarray:
    """Batched ablation: recapture with K query copies, inject deltas, predict.

    For one query row, creates K copies as the query batch, recaptures the
    hidden state (1 full forward pass, no re-fit), injects K different deltas
    at the K query positions (zero context delta), and runs 1 tail pass.

    Args:
        tail: fitted tail model with recapture() method
        X_row: (1, n_features) single query row
        deltas: (K, emb_dim) per-feature deltas in raw embedding space
        max_K: chunk size for VRAM safety

    Returns:
        preds: (K, ...) predictions, one per ablation
    """
    K = len(deltas)

    if K == 0:
        return np.array([])

    all_preds = []
    for chunk_start in range(0, K, max_K):
        chunk_end = min(chunk_start + max_K, K)
        chunk_deltas = deltas[chunk_start:chunk_end]
        chunk_K = len(chunk_deltas)

        # Create chunk_K copies of the query row
        X_batch = np.tile(X_row, (chunk_K, 1))

        # Recapture: 1 full forward pass with chunk_K query copies
        tail.recapture(X_batch)

        # Inject deltas at query positions (zero context delta)
        state = tail.hidden_state.clone()
        _inject_query_deltas(tail, state, chunk_deltas)

        # 1 tail pass → chunk_K predictions
        preds = tail._predict_with_modified_state(state)
        all_preds.append(preds)

    return np.concatenate(all_preds, axis=0) if len(all_preds) > 1 else all_preds[0]


def _inject_query_deltas(tail, state: torch.Tensor, deltas: torch.Tensor):
    """Inject per-query deltas into cached hidden state. Zero context delta.

    Handles model-specific state shapes:
    - TabPFN: (1, seq, n_struct, H) — inject at state[0, ctx+k, :, :]
    - TabICL/v2: (n_ens, seq, H) — inject at state[:, ctx+k, :]
    - TabDPT: (n_samples, H) or (n_samples, seq, H) — inject at state[ctx+k]
    - Mitra: separate support/query captures — modify query portion
    - HyperFast: (n_query, H) intermediates — add directly
    """
    from scripts.intervention.intervene_sae import (
        TabPFNTail, TabICLTail, TabICLV2Tail, TabDPTTail,
        MitraTail, HyperFastTail,
    )

    K = len(deltas)

    if isinstance(tail, TabPFNTail):
        ctx = tail.single_eval_pos
        for k in range(K):
            state[0, ctx + k, :, :] += deltas[k].unsqueeze(0)

    elif isinstance(tail, (TabICLTail, TabICLV2Tail)):
        ctx = tail.train_size
        for k in range(K):
            state[:, ctx + k, :] += deltas[k].unsqueeze(0)

    elif isinstance(tail, TabDPTTail):
        ctx = tail.n_ctx
        if state.ndim == 3:
            for k in range(K):
                state[ctx + k] += deltas[k].unsqueeze(0)
        else:
            for k in range(K):
                state[ctx + k] += deltas[k]

    elif isinstance(tail, MitraTail):
        # Mitra uses hook-based injection, not direct state manipulation
        # For batched ablation, we use predict() with a crafted delta
        # that has zeros for support and per-query deltas
        raise NotImplementedError(
            "Mitra batched ablation requires hook-based injection. "
            "Use tail.predict() with a combined (n_sup + K, dim) delta."
        )

    elif isinstance(tail, HyperFastTail):
        # HyperFast intermediates: directly add to cached activations
        # This is handled differently — deltas are added in predict()
        raise NotImplementedError(
            "HyperFast batched ablation: modify intermediates directly."
        )

    else:
        raise TypeError(f"Unknown tail type: {type(tail)}")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_intervene_lib.py -v`
Expected: PASS for metrics, alignment, and delta tests

- [ ] **Step 7: Commit**

```bash
git add scripts/intervention/intervene_lib.py tests/test_intervene_lib.py
git commit -m "feat: create intervene_lib.py with metrics, alignment, delta computation"
```

---

### Task 4: Create `perrow_importance.py` consumer script

The thin CLI consumer that runs the per-dataset importance loop.

**Files:**
- Create: `scripts/intervention/perrow_importance.py`

- [ ] **Step 1: Create the script**

```python
#!/usr/bin/env python3
"""Per-row, per-feature importance via batched SAE ablation.

For each model and dataset:
  1. Load preprocessed data, build tail model (fit once)
  2. Load test embeddings, encode through SAE
  3. For each query row: recapture with K copies, ablate each firing
     feature, measure prediction loss change

Output:
    output/perrow_importance/{model}/{dataset}.npz

Usage:
    python -m scripts.intervention.perrow_importance --model tabpfn --device cuda
    python -m scripts.intervention.perrow_importance --model mitra --datasets adult
    python -m scripts.intervention.perrow_importance --model tabpfn --resume
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import (
    SPLITS_PATH, SAE_DATA_DIR,
    load_sae, load_norm_stats, get_extraction_layer,
    build_tail, compute_per_row_loss, compute_feature_deltas,
    batched_ablation, align_test_rows,
)
from scripts.matching.utils import load_test_embeddings, load_norm_stats as load_norm_stats_matching
from data.preprocessing import load_preprocessed, CACHE_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "output" / "perrow_importance"

# Models with preprocessing cache
SUPPORTED_MODELS = ["tabpfn", "tabicl", "tabicl_v2", "mitra", "tabdpt", "hyperfast"]


def run_dataset(
    model_key: str, dataset: str, sae, data_std, extraction_layer: int,
    splits: dict, device: str, max_K: int,
) -> dict:
    """Run per-row importance for one dataset. Returns result dict."""
    split_info = splits[dataset]
    task = split_info["task_type"]
    holdout_indices = np.array(split_info["test_indices"])

    # Load preprocessed data
    data = load_preprocessed(model_key, dataset, CACHE_DIR)
    X_train, y_train = data.X_train, data.y_train

    # Load test embeddings for this dataset (already normalized)
    per_ds = load_test_embeddings(model_key)
    if dataset not in per_ds:
        raise ValueError(f"No test embeddings for {dataset}")
    emb = per_ds[dataset]
    n_query = len(emb)

    # SAE encode
    with torch.no_grad():
        emb_t = torch.tensor(emb, dtype=torch.float32, device=device)
        activations = sae.encode(emb_t)  # (n_query, hidden_dim)

    firing_mask = (activations > 0).cpu().numpy()
    alive_mask = firing_mask.any(axis=0)
    alive_features = np.where(alive_mask)[0].tolist()

    # Map test embedding rows to X_test positions
    # Need test_row_indices — replay select_sample() or load from NPZ
    # For now, use the NPZ's samples_per_dataset to find offset,
    # then rely on holdout_indices ordering
    # TODO: load saved row_indices from the NPZ if available
    #
    # Simplified: assume X_test is ordered by holdout_indices and
    # test embeddings are the first n_query rows of the test split
    # (This is correct when n_holdout <= 700, i.e. all rows used)
    X_query = data.X_test[:n_query]
    y_query = data.y_test[:n_query]

    logger.info(f"  Context: {X_train.shape}, Query: {n_query} rows, "
                f"Alive: {len(alive_features)} features")

    # Build tail ONCE
    t0 = time.time()
    tail = build_tail(model_key, X_train, y_train, X_query,
                      extraction_layer, task, device)
    baseline_preds = tail.baseline_preds
    baseline_loss = compute_per_row_loss(y_query, baseline_preds, task)
    logger.info(f"  Tail built in {time.time() - t0:.1f}s, "
                f"baseline loss: {baseline_loss.mean():.4f}")

    # Per-row importance
    n_alive = len(alive_features)
    row_feature_drops = np.zeros((n_query, n_alive), dtype=np.float32)

    # Load norm stats for denormalization
    norm_stats = load_norm_stats_matching(model_key)
    ds_mean, ds_std = norm_stats[dataset]
    data_std_t = torch.tensor(ds_std, dtype=torch.float32, device=device)

    t0 = time.time()
    for r in range(n_query):
        row_firing = [i for i, fi in enumerate(alive_features)
                      if firing_mask[r, fi]]
        if not row_firing:
            continue

        firing_feat_indices = [alive_features[i] for i in row_firing]

        # Compute K deltas
        h_row = activations[r]
        deltas = compute_feature_deltas(sae, h_row, firing_feat_indices, data_std_t)

        # Batched ablation: K copies, 1 recapture + 1 tail pass
        X_row = X_query[r:r + 1]
        preds = batched_ablation(tail, X_row, deltas, max_K=max_K)

        # Compute importance
        y_tiled = np.full(len(preds), y_query[r])
        ablated_loss = compute_per_row_loss(y_tiled, preds, task)
        for j, col_idx in enumerate(row_firing):
            row_feature_drops[r, col_idx] = ablated_loss[j] - baseline_loss[r]

        if (r + 1) % 50 == 0 or r == n_query - 1:
            elapsed = time.time() - t0
            rate = (r + 1) / elapsed
            eta = (n_query - r - 1) / rate if rate > 0 else 0
            logger.info(f"    row {r+1}/{n_query}: {len(row_firing)} firing "
                        f"({rate:.1f} rows/s, ETA {eta:.0f}s)")

    return {
        "row_feature_drops": row_feature_drops,
        "feature_indices": np.array(alive_features, dtype=np.int32),
        "baseline_preds": baseline_preds,
        "y_query": y_query,
        "extraction_layer": extraction_layer,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Per-row feature importance via batched SAE ablation")
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-K", type=int, default=512)
    args = parser.parse_args()

    splits = json.loads(SPLITS_PATH.read_text())

    # Load SAE once
    sae, config = load_sae(args.model, device=args.device)
    sae.eval()
    extraction_layer = get_extraction_layer(args.model)
    norm_stats = load_norm_stats_matching(args.model)

    # Get list of datasets with test embeddings
    per_ds = load_test_embeddings(args.model)
    datasets = sorted(per_ds.keys())
    if args.datasets:
        datasets = [d for d in datasets if d in args.datasets]

    out_dir = OUTPUT_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Per-row importance: {args.model}")
    logger.info(f"  SAE: {config.input_dim} -> {config.hidden_dim}, L{extraction_layer}")
    logger.info(f"  Datasets: {len(datasets)}")
    logger.info(f"  Device: {args.device}")

    for i, ds in enumerate(datasets):
        out_path = out_dir / f"{ds}.npz"
        if args.resume and out_path.exists():
            logger.info(f"[{i+1}/{len(datasets)}] {ds}: SKIP (exists)")
            continue

        logger.info(f"\n[{i+1}/{len(datasets)}] {ds}")
        try:
            # Check preprocessing cache exists
            if not (CACHE_DIR / args.model / f"{ds}.npz").exists():
                logger.info(f"  SKIP (no preprocessing cache)")
                continue

            data_std = norm_stats.get(ds)
            if data_std is None:
                logger.info(f"  SKIP (no norm stats)")
                continue

            result = run_dataset(
                args.model, ds, sae, data_std, extraction_layer,
                splits, args.device, args.max_K,
            )

            np.savez_compressed(
                str(out_path),
                **result,
            )
            logger.info(f"  -> {out_path.name}")

        except Exception as e:
            logger.error(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses correctly**

Run: `python -m scripts.intervention.perrow_importance --help`
Expected: usage message with --model, --datasets, --device, --resume, --max-K

- [ ] **Step 3: Commit**

```bash
git add scripts/intervention/perrow_importance.py
git commit -m "feat: add perrow_importance.py consumer script"
```

---

### Task 5: Integration test on one dataset (on GPU worker)

Verify the full pipeline works end-to-end on one model + one dataset.

**Files:**
- No new files — run on worker

- [ ] **Step 1: Sync code to a worker**

```bash
python cluster.py --sync
```

- [ ] **Step 2: Run on one small dataset**

```bash
ssh surfer4
cd /home/brian/src/tabular_embeddings
/home/brian/anaconda3/envs/tfm/bin/python -m scripts.intervention.perrow_importance \
    --model tabpfn --datasets diabetes --device cuda
```

Expected: output at `output/perrow_importance/tabpfn/diabetes.npz`

- [ ] **Step 3: Verify output**

```bash
/home/brian/anaconda3/envs/tfm/bin/python -c "
import numpy as np
d = np.load('output/perrow_importance/tabpfn/diabetes.npz', allow_pickle=True)
print('Keys:', list(d.keys()))
print('row_feature_drops:', d['row_feature_drops'].shape)
print('feature_indices:', d['feature_indices'].shape)
print('Non-zero entries:', (d['row_feature_drops'] != 0).sum())
print('Mean importance (firing only):', d['row_feature_drops'][d['row_feature_drops'] != 0].mean())
"
```

- [ ] **Step 4: Fix any issues, re-run, commit fixes**

- [ ] **Step 5: Run on a second dataset to confirm generalization**

```bash
/home/brian/anaconda3/envs/tfm/bin/python -m scripts.intervention.perrow_importance \
    --model tabpfn --datasets adult --device cuda
```

- [ ] **Step 6: Commit any fixes**

```bash
git add -u
git commit -m "fix: integration fixes from GPU worker testing"
```

---

### Task 6: Test with a second model (TabICL or Mitra)

Verify the pipeline generalizes beyond TabPFN.

**Files:**
- No new files — run on worker

- [ ] **Step 1: Run on TabICL with one dataset**

```bash
/home/brian/anaconda3/envs/tfm/bin/python -m scripts.intervention.perrow_importance \
    --model tabicl --datasets diabetes --device cuda
```

- [ ] **Step 2: Run on Mitra with one dataset**

```bash
/home/brian/anaconda3/envs/tfm/bin/python -m scripts.intervention.perrow_importance \
    --model mitra --datasets diabetes --device cuda
```

- [ ] **Step 3: Compare output shapes and value ranges across models**

```bash
/home/brian/anaconda3/envs/tfm/bin/python -c "
import numpy as np
for model in ['tabpfn', 'tabicl', 'mitra']:
    try:
        d = np.load(f'output/perrow_importance/{model}/diabetes.npz')
        drops = d['row_feature_drops']
        nz = drops[drops != 0]
        print(f'{model}: shape={drops.shape}, nonzero={len(nz)}, '
              f'mean={nz.mean():.4f}, max={nz.max():.4f}')
    except FileNotFoundError:
        print(f'{model}: not found')
"
```

- [ ] **Step 4: Fix model-specific issues, commit**

---

### Task 7: Full sweep deployment

Launch full sweeps across workers.

- [ ] **Step 1: Launch one model per worker**

```bash
# On surfer4 (TabPFN)
nohup /home/brian/anaconda3/envs/tfm/bin/python -m scripts.intervention.perrow_importance \
    --model tabpfn --device cuda --resume > perrow_tabpfn.log 2>&1 &

# On terrax4 (TabICL)
nohup /home/brian/anaconda3/envs/tfm/bin/python -m scripts.intervention.perrow_importance \
    --model tabicl --device cuda --resume > perrow_tabicl.log 2>&1 &

# On octo4 (Mitra)
nohup /home/brian/anaconda3/envs/tfm/bin/python -m scripts.intervention.perrow_importance \
    --model mitra --device cuda --resume > perrow_mitra.log 2>&1 &

# On firelord4 (TabDPT)
nohup /home/brian/anaconda3/envs/tfm/bin/python -m scripts.intervention.perrow_importance \
    --model tabdpt --device cuda --resume > perrow_tabdpt.log 2>&1 &
```

- [ ] **Step 2: Monitor progress**

```bash
for host in surfer4 terrax4 octo4 firelord4; do
    echo "=== $host ==="
    ssh $host "tail -3 ~/src/tabular_embeddings/perrow_*.log 2>/dev/null"
done
```

- [ ] **Step 3: After first batch completes, launch remaining models**

TabICL-v2 and HyperFast on freed workers.

- [ ] **Step 4: Verify all outputs**

```bash
/home/brian/anaconda3/envs/tfm/bin/python -c "
import os, numpy as np
base = 'output/perrow_importance'
for model in os.listdir(base):
    n = len([f for f in os.listdir(os.path.join(base, model)) if f.endswith('.npz')])
    print(f'{model}: {n} datasets')
"
```

---

## Implementation Notes

**Key gotchas the implementer must watch for:**

1. **Row index alignment** — The most error-prone part. See spec section
   "Row Index Alignment" for the exact pseudocode. If `row_indices` are not
   saved in the NPZ, you must replay `select_sample()` from
   `06_build_sae_training_data.py` with the same seed. Verify by checking
   counts match.

2. **Double normalization** — Test embeddings are already normalized. Pass
   directly to `sae.encode()`. Only use `data_std` for denormalizing deltas,
   never for normalizing inputs.

3. **Mitra RNG state** — Must save/restore `trainer.rng` state around
   `recapture()` to get deterministic batching. See existing `from_data()`.

4. **HyperFast recapture** — Different from other models. Must replay the
   generated MLP forward through layers 0..L to get new intermediates.
   Read `HyperFastTail.from_data()` carefully.

5. **TabDPT hidden state shape** — Can be 2D `(n_samples, H)` or 3D
   `(n_samples, seq, H)`. Both cases must be handled in `_inject_query_deltas`.

6. **Existing imports** — 16+ files import from `intervene_sae.py` and
   `concept_importance.py`. Do NOT move or rename these files. The new
   `intervene_lib.py` imports from them and adds new functionality.
