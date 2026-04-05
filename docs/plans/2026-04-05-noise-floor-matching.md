# Noise Floor Matching Filter

**Date:** 2026-04-05
**Status:** Approved

## Problem

29% of MNN matches (2,940/10,178) have Pearson correlation indistinguishable from
random SAE features. The per-pair cross-model random baseline (trained-A vs random-B)
establishes direction-specific noise floors. Matches below these floors are false
positives that contaminate downstream ablation, transfer, and importance results.

Worst affected pairs: Mitra-TabPFN (78% removed at p90), TabICL-v2-TabPFN (84%),
TabICL-TabPFN (77%) — all TabPFN pairs due to small dictionary (768 features).

## Solution

Filter MNN matches using per-pair, direction-specific p90 thresholds from the
trained-vs-random cross-model baseline. Both directions must pass for a match
to survive (since MNN is bidirectional). Also clean up vestigial `--alive-threshold`
/ `_t0.001` naming (TopK SAEs make magnitude thresholds meaningless).

## Design

### 1. `01_match_sae_concepts_mnn.py`

- Add `--baseline-path` (default: `output/sae_cross_model_random_baseline.json`)
- Add `--percentile` (default: `p90`, choices: `p90`, `p95`)
- After MNN matching per pair, load direction-specific thresholds:
  - `threshold_ab = baseline[(A, B)][percentile]`
  - `threshold_ba = baseline[(B, A)][percentile]`
  - Drop match if `r < threshold_ab` or `r < threshold_ba`
- Remove `--alive-threshold` arg entirely (alive mask uses `> 0`, TopK)
- Remove `_t{alive_threshold}` from output filename convention
- New default output: `sae_feature_matching_mnn_floor_p90.json`
- Print filtered/total counts per pair in stdout

### 2. `02_build_concept_graph.py`

- Remove `mnn_below_threshold` tier — three tiers: `mnn`, `threshold`, `unmatched`
- Update default `--mnn-path` to `sae_feature_matching_mnn_floor_p90.json`
- Remove dead code for the dropped tier

### 3. Downstream default path updates

Update default matching file paths in all consumers:
- `scripts/matching/03_*.py` through `05_*.py`
- `scripts/intervention/ablation_sweep.py`
- `scripts/intervention/transfer_sweep_v2.py`
- `scripts/intervention/concept_importance.py`
- `scripts/intervention/concept_causal_intervention.py`
- Any other script referencing `sae_feature_matching_mnn_t0.001.json`

### What doesn't change

- Cross-correlation matrices (same computation in step 01)
- Random baseline file (`sae_cross_model_random_baseline.json`, already computed)
- Alive mask logic (still `activation > 0`)

## Execution Order

1. Code changes (steps 01, 02, downstream paths)
2. Rerun step 01 -> new matching file
3. Rerun steps 02-05 (concept graph, grouping)
4. Rerun ablation, transfer, importance on workers
5. Update figures/tables
6. Defer labeling (step 06) to later sessions

## Expected Impact

At p90 threshold:
- 10,178 -> 7,238 matches (71% survive)
- Unmatched feature sets grow for ablation
- Transfer matched pairs shrink
- TabPFN pairs become very thin (15-49 matches) — may need discussion
