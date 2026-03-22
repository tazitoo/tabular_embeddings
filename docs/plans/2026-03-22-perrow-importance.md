# Intervention Backbone: Per-Row SAE Ablation Pipeline

**Date**: 2026-03-22
**Status**: Design approved, ready for implementation

## Goal

Build the shared infrastructure for three downstream tasks — importance,
ablation, and transfer — that all follow the same pattern:

1. Load preprocessed data + build tail model (fit once per dataset)
2. Make baseline predictions
3. Load test embeddings → SAE encode
4. Tweak SAE activations (zero features, swap features, inject foreign features)
5. Recapture hidden state with K modified query copies → tail predict
6. Compare ablated vs baseline predictions

Importance uses step 4 = "zero one feature at a time."
Ablation uses steps 4+6 = "zero features cumulatively, search for performance
cliff."
Transfer uses step 4 = "replace features from model A with model B's, search
for recovery point."

The routines written here — data loading, tail management, delta computation,
batched prediction, loss comparison — are the backbone for all three.

## Key Design Decisions

1. **Causal ablation**: Zero context delta. Only the query position is
   perturbed. This gives a clean causal interpretation — "what happens to this
   row's prediction when this concept is removed."

2. **Fit once, reuse tail**: The foundation model is fit once per dataset. The
   tail model caches the hidden state at the extraction layer. All 200 query
   rows share one tail; per-row ablations use `predict_row(row_idx, delta_row)`
   without rebuilding.

3. **Batched K-copy ablation**: For each query row with K firing features,
   re-capture hidden state with K copies of that row as query (1 full forward
   pass, no re-fit), then inject K different deltas and run 1 tail pass → K
   predictions. Total: O(N_rows) full forward passes + O(N_rows) tail passes.
   This requires a `recapture(X_query_new)` method on tail classes that
   separates "fit model" from "capture hidden state." The same recapture
   pattern is reused by ablation and transfer experiments downstream.

4. **Test embeddings as input**: Pre-computed **already-normalized** embeddings
   from `sae_training_round10/` go directly to `sae.encode()`. No
   re-normalization needed. Deltas are denormalized with `data_std` before
   injection into raw hidden states.

## Data Flow

```
Inputs:
  tabarena_splits.json
    → train_indices  (context rows for tail model)
    → test_indices   (holdout fold)

  sae_training_round10/{model}_taskaware_sae_test.npz
    → 200 normalized embeddings/dataset (stratified subsample of test_indices)
    → samples_per_dataset (dataset, count) pairs

  sae_training_round10/{model}_taskaware_norm_stats.npz
    → per-dataset (mean, std) for denormalizing deltas

  sae_tabarena_sweep_round10/{model}/sae_matryoshka_archetypal_validated.pt
    → SAE checkpoint

  sae_training_round9/preprocessed/{model}/{dataset}.npz
    → X_train, y_train (context for tail model)
    → X_test, y_test   (query rows; test_row_indices selects the 200 subset)

Pipeline per dataset:
  1. Unpool test embeddings for this dataset from concatenated NPZ
  2. Load norm stats → (mean, std) for this dataset
  3. SAE encode: embeddings are already normalized, pass directly to
     sae.encode() → activations (n_query, hidden_dim)
  4. Compute firing mask: activations > 0 → (n_query, hidden_dim) boolean
  5. Compute alive features: features firing on any row
  6. Load preprocessed data → X_train, y_train as context
  7. Map test_row_indices to X_test positions (see Row Index Alignment below)
     → X_query (n_query, n_features), y_query (n_query,)
  8. Build tail ONCE: Tail.from_data(X_train, y_train, X_query, layer, task, device)
     → baseline_preds (n_query, n_classes)
  9. For each query row r (0..n_query-1):
       a. K = number of firing features for row r
       b. tail.recapture(tile(X_query[r], K))  — 1 full forward pass, no re-fit
       c. For each firing feature k, compute delta_k:
            h_abl = h[r].clone(); h_abl[k] = 0
            delta_norm = sae.decode(h_abl) - sae.decode(h[r])
            delta_raw = delta_norm * data_std
       d. Inject K deltas at query positions 0..K-1 (zero context delta)
       e. 1 tail pass via _predict_with_modified_state() → K predictions
       f. importance[r, k] = compute_per_row_loss(...)[k] - baseline_loss[r]
 10. Save output NPZ

Output:
  output/perrow_importance/{model}/{dataset}.npz
    row_feature_drops:  (n_query, n_alive) float32  — 0 for non-firing entries
    feature_indices:    (n_alive,) int32
    baseline_preds:     (n_query, ...) float32
    y_query:            (n_query,)
    row_indices:        (n_query,) int32  — original dataset row indices
    extraction_layer:   int
```

## Row Index Alignment

The test embeddings are a stratified subsample of the holdout fold. Mapping
them back to preprocessed X_test rows requires careful index chaining.

`06_build_sae_training_data.py` saves `test_row_indices[ds]` which are
**absolute indices into the original dataset**. Meanwhile,
`PreprocessedDataset.X_test` is ordered by `splits[ds]["test_indices"]`.

```python
# Get absolute row indices for this dataset's test embeddings
# (from 06_build_sae_training_data.py line 438-439)
holdout_indices = np.array(splits[ds]["test_indices"])

# Build reverse lookup: absolute_index → position in X_test
abs_to_position = {int(idx): pos for pos, idx in enumerate(holdout_indices)}

# Map test embedding rows to X_test positions
positions = np.array([abs_to_position[ri] for ri in test_row_indices_for_ds])

X_query = data.X_test[positions]
y_query = data.y_test[positions]
```

**Verification**: assert `len(X_query) == n_test_embeddings_for_ds` and spot-check
a few rows by re-extracting embeddings and comparing against stored values.

## Normalization: Avoiding Double-Normalize

Test embeddings in the NPZ are **already per-dataset StandardScaler normalized**
(applied during `06_build_sae_training_data.py`). The SAE was trained on
normalized embeddings, so they go directly to `sae.encode()`.

The delta computation is:
```python
# h is from sae.encode(already_normalized_embeddings)
h_abl = h.clone()
h_abl[:, feature_k] = 0.0
delta_norm = sae.decode(h_abl) - sae.decode(h)  # in normalized space
delta_raw = delta_norm * data_std                 # back to raw embedding space
```

Do NOT use `compute_ablation_delta()` which normalizes its input — that would
double-normalize. Compute the delta inline as above.

## Tail API

All 8 tail classes currently expose:
- `from_data(X_ctx, y_ctx, X_query, layer, task, device)` — fit + capture
- `predict_row(row_idx, delta_row)` — single-row intervention
- `_predict_with_modified_state(state)` — internal, runs tail layers

**New method needed**: `recapture(X_query_new)` — re-run forward pass with new
query data, re-capture hidden state at layer L, but **do not re-fit** the model.
This separates the expensive fit (once per dataset) from the cheap hidden-state
capture (once per query row).

```python
# Usage pattern:
tail = TabPFNTail.from_data(X_train, y_train, X_query_200, layer, task, device)
baseline_preds = tail.baseline_preds  # (200, n_classes)

for r in range(n_query):
    X_batch = np.tile(X_query[r:r+1], (K, 1))
    tail.recapture(X_batch)  # 1 full forward pass, no re-fit
    # Now tail.hidden_state has K query slots
    # Inject K deltas at query positions, 1 tail pass → K predictions
    state = tail.hidden_state.clone()
    for k_idx in range(K):
        state[..., ctx + k_idx, ...] += deltas[k_idx]
    preds = tail._predict_with_modified_state(state)  # K predictions
```

The `recapture()` method is the same for all tail groups — it re-runs the
model's predict with a hook at layer L, replaces `self.hidden_state`, updates
`self.n_query` and `self.X_query`. Per-model differences (hook placement,
state tensor shape) are already encapsulated in each tail class.

This pattern is reused by three downstream scripts: importance (this script),
ablation sweeps, and transfer experiments.

## CLI

```bash
python scripts/intervention/perrow_importance.py --model tabpfn --device cuda
python scripts/intervention/perrow_importance.py --model mitra --datasets adult diabetes
python scripts/intervention/perrow_importance.py --model tabpfn --resume
```

Arguments:
- `--model` (required): Model key (tabpfn, tabicl, tabicl_v2, mitra, tabdpt,
  hyperfast, carte, tabula8b)
- `--datasets` (optional): Filter to specific datasets, default=all with test
  embeddings
- `--device` (default=cuda)
- `--resume`: Skip datasets where output NPZ already exists
- `--max-K` (default=512): VRAM safety — chunk features if K exceeds this
  (unlikely needed; ctx+256 fits easily in 24GB)

## Preprocessing Cache Gap

The round 9 preprocessing cache covers 6 models: tabpfn, tabdpt, tabicl,
tabicl_v2, mitra, hyperfast. CARTE and Tabula-8B have bespoke preprocessing
pipelines not yet cached.

Recommendation: Implement for 6 cached models first. Add CARTE/Tabula-8B later
when their preprocessing is cached.

## Code Structure

### Library: `scripts/intervention/intervene_lib.py`

Refactored from `intervene_sae.py` (tail classes, delta computation, hook logic)
plus new shared routines. Replaces the 8 duplicated `sweep_*()` functions in
`concept_importance.py`. After consumers are working, `concept_importance.py`
and `intervene_sae.py` move to `scripts/intervention/archived/`.

Contains:

**Tail classes** (moved from `intervene_sae.py`):
- `TabPFNTail`, `TabICLTail`, `TabICLV2Tail`, `MitraTail`, `TabDPTTail`,
  `CARTETail`, `HyperFastTail`, `Tabula8BTail`
- Each gains a `recapture(X_query_new)` method

**Data loading**:
```python
def load_dataset_context(model_key, dataset, splits)
    """Load preprocessed data, resolve row alignment, return context + query."""
    → X_train, y_train, X_query, y_query, row_indices, task

def encode_test_embeddings(sae, dataset, test_emb_npz)
    """Unpool dataset's embeddings from concatenated NPZ, encode through SAE."""
    → activations, firing_mask, alive_features
```

**Tail management**:
```python
def build_tail(model_key, X_train, y_train, X_query, layer, task, device)
    """Fit model once, build tail, return tail + baseline_preds."""
    → tail, baseline_preds
```

**Delta computation + batched prediction**:
```python
def compute_feature_deltas(sae, activations_row, feature_indices, data_std)
    """For one row, compute deltas for ablating each feature."""
    → deltas: (K, emb_dim) tensor in raw space

def batched_ablation(tail, X_row, deltas, X_context, y_context)
    """Recapture with K query copies, inject K deltas, 1 tail pass."""
    → K predictions
```

**Metrics** (moved from `concept_importance.py`):
```python
def compute_per_row_loss(y_true, preds, task) → (n_samples,) array
def compute_importance_metric(y_true, preds, task) → (value, name)
```

These compose naturally for all three tasks:

- **Importance**: `for row: deltas = zero each feature; preds = batched_ablation(...)`
- **Ablation**: `for row: sort features by importance; cumulatively ablate; search for cliff`
- **Transfer**: `for row: replace features with foreign model's; batched_ablation; search for recovery`

### First consumer: `scripts/intervention/perrow_importance.py`

Thin script that calls the backbone in a per-dataset loop. Handles CLI, resume,
output saving. ~100 lines. Lives in `intervention/` alongside the backbone.

### Code provenance

**`intervene_sae.py`** → tail classes, `load_sae()`, `get_extraction_layer()`,
`load_norm_stats()` move into `intervene_lib.py`. Then archived.

**`concept_importance.py`** (2,472 lines) → `compute_per_row_loss()`,
`compute_importance_metric()`, `get_alive_features()`, `get_feature_labels()`,
`get_matryoshka_bands()`, `MODEL_KEY_TO_LABEL_KEY` move into `intervene_lib.py`.
The 8 duplicate `sweep_*()` functions are replaced by one generic loop. Then
archived.

**`09_perrow_importance.py`** → the batched K-copy pattern (`tile(x_row, K)` →
capture → inject K deltas → 1 tail pass) generalizes into `batched_ablation()`.
The delta computation pattern (`h_abl[:, feat] = 0; delta = (decode(h_abl) -
decode(h)) * std`) moves into `compute_feature_deltas()`.

**`matching/utils.py`** — `load_test_embeddings()`, `load_norm_stats()` reused.

**`data/preprocessing.py`** — `load_preprocessed()`, `CACHE_DIR` reused.

## Performance

Per dataset (TabPFN, 200 rows, ~256 firing features avg):
- Model fit: ~3s (once per dataset)
- Per-row: 1 recapture (~20ms full forward) + 1 tail pass (~10ms) ≈ 30ms/row
- 200 rows × 30ms = ~6s per dataset
- 51 datasets: ~5 min per model on one GPU

The batched K-copy approach is O(N_rows × 2) forward passes (1 full recapture +
1 tail pass per row), compared to O(N_rows × K) for sequential predict_row().
For K=256, that's a ~128x speedup.

Parallelizable across 4 workers (one model per worker), or across workers by
dataset subset. The `--resume` flag allows restarting failed runs.
