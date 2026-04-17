# Baseline predictions cache (2026-04-17)

## Motivation

The contrastive labeling pipeline (`scripts/concepts/label_contrastive_mesh.py`)
lets agents describe SAE features in terms of the raw data of activating vs
contrast rows. Adding the model's prediction for each row — the predicted class,
its probability, and whether that prediction is correct — sharpens labels:

- Feature fires on **confident correct** predictions → tracks a decisive signal.
- Feature fires on **uncertain** predictions → flags ambiguous rows.
- Feature fires on **wrong** predictions → tracks a confounder the model mis-uses.

Predictions are not currently cached on disk:

- `output/perrow_importance/{model}/{dataset}.npz` has a `baseline_preds` field,
  but it is keyed on 200 random test rows per dataset, not the first 200 rows
  the SAE test embeddings use (confirmed 2026-04-17: ~1/6 overlap on f_92 SDSS17).
- Embeddings are cached (`output/embeddings/tabarena/{model}/{dataset}.npz`) but
  not predictions.

## Scope

| | |
|---|---|
| Models | tabpfn, tabicl, tabicl_v2, tabdpt, mitra, carte |
| Excluded | hyperfast (different representation, separate track), tabula-8b (out of main sweep) |
| Datasets | TabArena (~51) |
| Rows per dataset | First 200 of test split — the same indexing as `load_test_embeddings(model)[dataset]` |

## Output layout

```
output/baseline_predictions/{model}/{dataset}.npz
  pred_probs     (200, n_classes) float32   # classification
       or        (200,)            float32   # regression
  pred_class     (200,)           int64      # argmax for classification
       or        (200,)           float32   # same as pred_probs for regression
  y_true         (200,)           float32
  row_indices    (200,)           int64      # global test_indices[:200]
  model_key      str
  task_type      str ("classification" | "regression")
  extraction_layer int               # just for provenance; predictions are full-model
```

One file per (model, dataset). Deterministic — re-running regenerates identical
files.

## Per-model tail dispatch

`scripts/intervention/intervene_sae.py` defines a tail class per model:
`MitraTail`, `TabPFNTail`, `TabICLTail`, `TabICLV2Tail`, `TabDPTTail`, `CARTETail`.
`build_tail(model_key, X_context, y_context, X_query, extraction_layer, task, ...)`
dispatches.

For prediction caching we do NOT need the extraction-layer embedding intercept —
we just want `predict_proba(X_query)` or equivalent. Each tail class already
supports a forward pass that produces the final prediction, so the script can:

1. Load splits + data: `X`, `y` from `load_tabarena_dataset`.
2. Extract train and first-200-test: `X_train = X.iloc[train_idx]`, `X_q = X.iloc[test_idx[:200]]`.
3. Build the tail for this model × dataset × task.
4. Run the tail's predict path on `X_q`. Capture `pred_probs`.
5. Save npz.

Some tails expect **batched** query inputs (CARTE requires graph batching;
Mitra tiles queries into context). The script wraps each tail in a uniform
`predict_batch(tail, X_q)` helper.

## Env constraints

- `tabicl_v2` requires `tfm2` env (see `feedback_tabicl_v2_env.md`).
- Others use `tfm`.
- Script must detect the needed env and skip / warn if misconfigured.

## Compute estimate

- Per (model, dataset): one forward pass on 200 queries + context setup. Seconds
  to low minutes on GPU. CARTE is the slowest (per-dataset fine-tune).
- 6 models × 51 datasets = 306 (model, dataset) pairs.
- Total wall time: ~30-90 min per model on one GPU worker, so ~4-9 hrs
  sequential. Dispatch round-robin across 4-5 workers: ~1-2 hrs.

## Execution plan

1. Write `scripts/intervention/cache_baseline_predictions.py` with:
   - `--model <key>` and `--datasets a b c...` (defaults all TabArena).
   - `--output-dir output/baseline_predictions/`
   - `--device cuda` (default).
   - `--resume` skips files that already exist with valid shapes.
2. Validate on one (model, dataset): `mitra × SDSS17`. Eyeball `pred_probs[:5]`,
   confirm argmax matches expected class distribution.
3. Dispatch across workers. One model per worker (tfm2 for tabicl_v2 on
   terrax4 or nova4; others split by availability).
4. Rsync results back to local.

## Wiring into the labeling pipeline

Post-cache:

- `build_contrastive_examples.py` loads the predictions cache for each dataset.
  For each selected row index `ri` (local 0-199), it looks up
  `pred_probs[ri]`, `pred_class[ri]`, `y_true[ri]` and writes into the CSV as
  new columns `pred_class (conf=X.YY, correct=bool)`.
- `label_contrastive_mesh.py` updates `_dataset_block` to mention that rows carry
  a prediction annotation. The prompt instructs the agent to note whether the
  feature fires on confident-correct, confident-wrong, or uncertain rows.
- Validation: on held-out rows, same annotation enables predicate-checking like
  "feature fires iff predicted y=1 with prob>0.9".

## Deferred / non-goals

- Recomputing `perrow_importance` with first-200 indexing is out of scope — that
  would subsume this cache but re-touches a larger, already-validated pipeline.
- HyperFast predictions (separate track).
- Predictions for non-TabArena suites (PMLB, RelBench, Probing).
