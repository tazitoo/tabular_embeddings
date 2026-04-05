# Noise Floor Matching Filter — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Filter MNN matches below per-pair direction-specific p90 random baseline thresholds and clean up vestigial alive-threshold naming.

**Architecture:** Add a post-filtering step to `01_match_sae_concepts_mnn.py` that loads the random baseline JSON and drops MNN matches below the per-pair p90 threshold. Simplify `02_build_concept_graph.py` to three tiers. Update all downstream default paths.

**Tech Stack:** Python, NumPy, JSON, pytest

---

### Task 1: Add noise floor filtering to step 01

**Files:**
- Modify: `scripts/matching/01_match_sae_concepts_mnn.py`

- [ ] **Step 1: Write filtering helper function**

Add a function after the existing imports (around line 44) that loads the random baseline and filters a match list:

```python
def load_noise_floor_thresholds(
    baseline_path: Path, percentile_key: str = "p90"
) -> Dict[Tuple[str, str], float]:
    """Load per-pair per-direction noise floor thresholds.

    Returns:
        Dict mapping (trained_model, random_model) -> threshold.
    """
    with open(baseline_path) as f:
        data = json.load(f)

    thresholds = {}
    for pair_key, stats in data["pairs"].items():
        parts = pair_key.split("__trained_vs_")
        if len(parts) != 2:
            continue
        model_a = parts[0]
        model_b = parts[1].replace("__random", "")
        thresholds[(model_a, model_b)] = stats[percentile_key]
    return thresholds


def filter_matches_by_noise_floor(
    matches: list,
    name_a: str,
    name_b: str,
    thresholds: Dict[Tuple[str, str], float],
) -> Tuple[list, int]:
    """Drop MNN matches below per-pair direction-specific noise floor.

    Both directions must pass: r >= threshold(A->B) AND r >= threshold(B->A).

    Returns:
        (filtered_matches, n_removed)
    """
    t_ab = thresholds.get((name_a, name_b))
    t_ba = thresholds.get((name_b, name_a))

    if t_ab is None or t_ba is None:
        # No baseline for this pair — keep all matches, warn
        print(f"  WARNING: no noise floor for {name_a}<->{name_b}, keeping all matches")
        return matches, 0

    threshold = max(t_ab, t_ba)  # both directions must pass
    filtered = [m for m in matches if m["r"] >= threshold]
    return filtered, len(matches) - len(filtered)
```

Wait — per the design, direction-specific means both directions must pass. Since a single MNN match has one `r` value (the correlation between the pair), and both `threshold(A->B)` and `threshold(B->A)` must be satisfied, we take the max of both directions as the effective threshold. This is correct.

- [ ] **Step 2: Remove `--alive-threshold` arg, add baseline args**

Replace the CLI args section (lines 427-432) and update the default output path (lines 468-474):

Replace:
```python
    parser.add_argument(
        "--alive-threshold",
        type=float,
        default=0.001,
        help="Min max-activation to consider a feature alive (default: 0.001)",
    )
```

With:
```python
    parser.add_argument(
        "--baseline-path",
        type=str,
        default="output/sae_cross_model_random_baseline.json",
        help="Cross-model random baseline JSON for noise floor thresholds",
    )
    parser.add_argument(
        "--percentile",
        type=str,
        default="p90",
        choices=["p90", "p95"],
        help="Percentile from random baseline to use as noise floor (default: p90)",
    )
```

Replace the default output path block:
```python
    if args.output is None:
        args.output = (
            f"output/sae_feature_matching"
            f"_{args.method}"
            f"_t{args.alive_threshold}"
            f".json"
        )
```

With:
```python
    if args.output is None:
        args.output = (
            f"output/sae_feature_matching"
            f"_{args.method}"
            f"_floor_{args.percentile}"
            f".json"
        )
```

- [ ] **Step 3: Load thresholds and apply filtering in the main matching loop**

After `args = parser.parse_args()` and the output path setup, load the thresholds:

```python
    baseline_file = PROJECT_ROOT / args.baseline_path
    noise_thresholds = None
    if baseline_file.exists():
        noise_thresholds = load_noise_floor_thresholds(baseline_file, args.percentile)
        print(f"Noise floor: {args.percentile} from {baseline_file.name} "
              f"({len(noise_thresholds)} directed pairs)")
    else:
        print(f"WARNING: no baseline at {baseline_file}, skipping noise floor filter")
```

In the main matching loop (after line 746 `pairs[pair_key] = result`), add filtering. Insert after `result` is obtained from `match_model_pair()` but before storing in `pairs`:

```python
        # Apply noise floor filter
        if noise_thresholds is not None and not is_tiered:
            filtered, n_removed = filter_matches_by_noise_floor(
                result["matches"], name_a, name_b, noise_thresholds,
            )
            if n_removed > 0:
                # Recompute unmatched sets
                matched_a = {m["idx_a"] for m in filtered}
                matched_b = {m["idx_b"] for m in filtered}
                all_a = set(result.get("unmatched_a", [])) | {m["idx_a"] for m in result["matches"]}
                all_b = set(result.get("unmatched_b", [])) | {m["idx_b"] for m in result["matches"]}
                result["matches"] = filtered
                result["unmatched_a"] = sorted(all_a - matched_a)
                result["unmatched_b"] = sorted(all_b - matched_b)
                result["n_matched"] = len(filtered)
                result["n_removed_noise_floor"] = n_removed
                result["mean_match_r"] = (
                    float(np.mean([m["r"] for m in filtered])) if filtered else 0.0
                )
                print(f"  noise floor: removed {n_removed}, kept {len(filtered)}")
```

Note: tiered matching is a separate code path — noise floor filtering only applies to the standard MNN loop.

- [ ] **Step 4: Update metadata in output JSON**

Replace `"alive_threshold": args.alive_threshold` (line 780) with:

```python
            "noise_floor_percentile": args.percentile,
            "noise_floor_baseline": args.baseline_path,
```

Also update the random-baseline and cross-model-baseline output paths. In the `--random-baseline` block (around line 587-593), replace:

```python
        if args.output == (
            f"output/sae_feature_matching"
            f"_{args.method}"
            f"_t{args.alive_threshold}"
            f".json"
        ):
            args.output = args.output.replace(".json", "_random_baseline.json")
```

With:
```python
        if args.output == (
            f"output/sae_feature_matching"
            f"_{args.method}"
            f"_floor_{args.percentile}"
            f".json"
        ):
            args.output = args.output.replace(".json", "_random_baseline.json")
```

Remove the `"alive_threshold": args.alive_threshold` from the random baseline metadata block (line 581) too.

Similarly in the cross-model-baseline block (around line 694-700), replace:
```python
        if args.output == (
            f"output/sae_feature_matching"
            f"_{args.method}"
            f"_t{args.alive_threshold}"
            f".json"
        ):
            args.output = "output/sae_cross_model_random_baseline.json"
```

With:
```python
        if args.output == (
            f"output/sae_feature_matching"
            f"_{args.method}"
            f"_floor_{args.percentile}"
            f".json"
        ):
            args.output = "output/sae_cross_model_random_baseline.json"
```

- [ ] **Step 5: Commit**

```bash
git add scripts/matching/01_match_sae_concepts_mnn.py
git commit -m "feat: add per-pair p90 noise floor filter to MNN matching

Loads direction-specific thresholds from random baseline and drops
matches where r < max(threshold(A->B), threshold(B->A)). Removes
vestigial --alive-threshold arg (TopK SAEs make it meaningless)."
```

---

### Task 2: Simplify tier system in step 02

**Files:**
- Modify: `scripts/matching/02_build_concept_graph.py`

- [ ] **Step 1: Update docstring and default path**

Replace the module docstring tier list (lines 6-10):
```python
  - mnn: mutual nearest neighbor match AND above pair-specific threshold
  - threshold: best correlate exceeds pair-specific p90 random baseline
  - mnn_below_threshold: MNN match but below pair threshold
  - unmatched: no correlate above noise floor in any partner
```

With:
```python
  - mnn: mutual nearest neighbor match AND above pair-specific threshold
  - threshold: best correlate exceeds pair-specific p90 random baseline
  - unmatched: no correlate above noise floor in any partner
```

Update default `--mnn-path` (line 238):
```python
        default="output/sae_feature_matching_mnn_floor_p90.json",
```

- [ ] **Step 2: Remove `mnn_below_threshold` from classification logic**

Replace the tier classification block (lines 169-177):
```python
            has_mnn = key in mnn_matches
            if has_mnn and above_any_threshold:
                tier = "mnn"
            elif has_mnn:
                tier = "mnn_below_threshold"
            elif above_any_threshold:
                tier = "threshold"
            else:
                tier = "unmatched"
```

With:
```python
            has_mnn = key in mnn_matches
            if has_mnn and above_any_threshold:
                tier = "mnn"
            elif above_any_threshold:
                tier = "threshold"
            else:
                tier = "unmatched"
```

- [ ] **Step 3: Remove `mnn_below_threshold` from summary output**

Replace the summary table line (line 291):
```python
        mnn_below = tc.get("mnn_below_threshold", 0)
```

With removal of that line, and update the header (line 285-286):
```python
    print(f"{'Model':<12s} {'Alive':>6s} {'MNN':>6s} {'Thresh':>7s} {'Unmatch':>8s} {'Match%':>7s}")
    print("-" * 52)
```

And the per-model print (line 296):
```python
        print(f"{model:<12s} {alive:>6d} {mnn:>6d} {thresh:>7d} {unmatched:>8d} {pct:>6.1f}%")
```

The `mnn_below` variable and its column are removed entirely.

- [ ] **Step 4: Commit**

```bash
git add scripts/matching/02_build_concept_graph.py
git commit -m "refactor: simplify to 3 tiers (mnn/threshold/unmatched)

mnn_below_threshold tier removed — step 01 now filters sub-threshold
MNN matches at source. Update default path to new naming convention."
```

---

### Task 3: Update downstream default paths

**Files:**
- Modify: `scripts/matching/05_label_cross_model_concepts.py:1337`
- Modify: `scripts/intervention/concept_importance.py:1975`
- Modify: `scripts/intervention/concept_causal_intervention.py:48`
- Modify: `scripts/intervention/concept_performance_diagnostic.py:47`
- Modify: `scripts/intervention/ablation_sweep.py` (no hardcoded default, uses CLI `--matching-file`)
- Modify: `scripts/intervention/transfer_sweep_v2.py` (no hardcoded default, uses CLI `--matching-file`)

- [ ] **Step 1: Update `05_label_cross_model_concepts.py`**

Line 1337, replace:
```python
        default="output/sae_feature_matching_mnn_t0.001.json",
```
With:
```python
        default="output/sae_feature_matching_mnn_floor_p90.json",
```

- [ ] **Step 2: Update `concept_importance.py`**

Line 1975, replace:
```python
DEFAULT_MNN_PATH = PROJECT_ROOT / "output" / "sae_feature_matching_mnn_t0.001_n500.json"
```
With:
```python
DEFAULT_MNN_PATH = PROJECT_ROOT / "output" / "sae_feature_matching_mnn_floor_p90.json"
```

- [ ] **Step 3: Update `concept_causal_intervention.py`**

Line 48, replace:
```python
DEFAULT_MNN_PATH = PROJECT_ROOT / "output" / "sae_feature_matching_mnn_t0.001_n500.json"
```
With:
```python
DEFAULT_MNN_PATH = PROJECT_ROOT / "output" / "sae_feature_matching_mnn_floor_p90.json"
```

- [ ] **Step 4: Update `concept_performance_diagnostic.py`**

Line 47, replace:
```python
DEFAULT_MNN_PATH = PROJECT_ROOT / "output" / "sae_feature_matching_mnn_t0.001_n500.json"
```
With:
```python
DEFAULT_MNN_PATH = PROJECT_ROOT / "output" / "sae_feature_matching_mnn_floor_p90.json"
```

- [ ] **Step 5: Commit**

```bash
git add scripts/matching/05_label_cross_model_concepts.py \
      scripts/intervention/concept_importance.py \
      scripts/intervention/concept_causal_intervention.py \
      scripts/intervention/concept_performance_diagnostic.py
git commit -m "chore: update default matching paths to noise-floor-filtered output"
```

---

### Task 4: Update tests

**Files:**
- Modify: `tests/test_match_sae_features.py`

- [ ] **Step 1: Fix stale `get_alive_mask` tests**

The existing tests pass a `threshold` kwarg to `get_alive_mask`, but the current function (in `scripts/matching/utils.py:124`) no longer accepts one — it hardcodes `> 0` for TopK. Update the test class `TestGetAliveMask` (lines 28-50):

```python
class TestGetAliveMask:
    def test_correct_shape(self):
        acts = np.array([[0.5, 0.0, 0.3], [0.0, 0.0, 0.1]])
        mask = get_alive_mask(acts)
        assert mask.shape == (3,)

    def test_dead_features_excluded(self):
        acts = np.array([[0.5, 0.0, 0.3], [0.2, 0.0, 0.0]])
        mask = get_alive_mask(acts)
        assert mask[0] is np.True_
        assert mask[1] is np.False_
        assert mask[2] is np.True_

    def test_any_positive_is_alive(self):
        """With TopK, any activation > 0 means the feature fired."""
        acts = np.array([[0.01, 0.001, 0.0005]])
        mask = get_alive_mask(acts)
        assert mask.sum() == 3  # all > 0

    def test_all_dead(self):
        acts = np.zeros((10, 5))
        mask = get_alive_mask(acts)
        assert mask.sum() == 0
```

- [ ] **Step 2: Add test for noise floor filtering**

Add a new test class after the existing tests:

```python
from scripts.matching.01_match_sae_concepts_mnn import filter_matches_by_noise_floor


class TestFilterMatchesByNoiseFloor:
    def test_filters_below_threshold(self):
        matches = [
            {"idx_a": 0, "idx_b": 0, "r": 0.5},
            {"idx_a": 1, "idx_b": 1, "r": 0.1},  # below noise floor
            {"idx_a": 2, "idx_b": 2, "r": 0.3},
        ]
        thresholds = {("A", "B"): 0.2, ("B", "A"): 0.15}
        filtered, n_removed = filter_matches_by_noise_floor(
            matches, "A", "B", thresholds
        )
        assert n_removed == 1
        assert len(filtered) == 2
        assert all(m["r"] >= 0.2 for m in filtered)

    def test_uses_max_of_both_directions(self):
        """Effective threshold is max(A->B, B->A) since both must pass."""
        matches = [
            {"idx_a": 0, "idx_b": 0, "r": 0.25},  # above A->B but below B->A
        ]
        thresholds = {("A", "B"): 0.2, ("B", "A"): 0.3}
        filtered, n_removed = filter_matches_by_noise_floor(
            matches, "A", "B", thresholds
        )
        assert n_removed == 1
        assert len(filtered) == 0

    def test_missing_baseline_keeps_all(self):
        matches = [{"idx_a": 0, "idx_b": 0, "r": 0.05}]
        filtered, n_removed = filter_matches_by_noise_floor(
            matches, "A", "B", {}
        )
        assert n_removed == 0
        assert len(filtered) == 1

    def test_empty_matches(self):
        filtered, n_removed = filter_matches_by_noise_floor([], "A", "B", {})
        assert n_removed == 0
        assert filtered == []
```

- [ ] **Step 3: Run tests**

Run: `cd /Volumes/Samsung2TB/src/tabular_embeddings && python -m pytest tests/test_match_sae_features.py -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_match_sae_features.py
git commit -m "test: update alive mask tests for TopK, add noise floor filter tests"
```

---

### Task 5: Run step 01 to generate new matching file

**Files:**
- Output: `output/sae_feature_matching_mnn_floor_p90.json`

- [ ] **Step 1: Run matching with noise floor filter**

This runs on the local machine (galactus) since it loads precomputed test embeddings and SAE checkpoints. Takes ~5-10 minutes.

```bash
cd /Volumes/Samsung2TB/src/tabular_embeddings
python scripts/matching/01_match_sae_concepts_mnn.py --save-correlations
```

Expected output: per-pair match counts with noise floor removal stats, new JSON at `output/sae_feature_matching_mnn_floor_p90.json`.

- [ ] **Step 2: Verify output**

```bash
python -c "
import json
with open('output/sae_feature_matching_mnn_floor_p90.json') as f:
    d = json.load(f)
total = sum(r['n_matched'] for r in d['pairs'].values())
removed = sum(r.get('n_removed_noise_floor', 0) for r in d['pairs'].values())
print(f'Total matches: {total}, removed by noise floor: {removed}')
print(f'Percentile: {d[\"metadata\"][\"noise_floor_percentile\"]}')
"
```

Expected: ~7,238 total matches, ~2,940 removed.

- [ ] **Step 3: Commit output**

```bash
git add output/sae_feature_matching_mnn_floor_p90.json
git commit -m "data: MNN matching with per-pair p90 noise floor filter

7,238 matches survive (was 10,178). 2,940 sub-threshold matches removed.
Per-pair direction-specific thresholds from trained-vs-random baseline."
```

---

### Task 6: Rerun steps 02-05

**Files:**
- Output: `output/sae_feature_match_graph_p90.json`
- Output: concept grouping outputs from steps 03-05

- [ ] **Step 1: Run step 02 (concept graph)**

```bash
python scripts/matching/02_build_concept_graph.py
```

Verify the output has no `mnn_below_threshold` tier:
```bash
python -c "
import json
with open('output/sae_feature_match_graph_p90.json') as f:
    d = json.load(f)
print('Tiers:', d['tier_counts'])
assert 'mnn_below_threshold' not in d['tier_counts']
"
```

- [ ] **Step 2: Run steps 03-05**

```bash
python scripts/matching/03_compute_pymfe_cache.py
python scripts/matching/04_analyze_concept_regression.py
python scripts/matching/05_label_cross_model_concepts.py --skip-llm
```

Note: `--skip-llm` on step 05 skips the labeling phase (deferred). The grouping and structural analysis still runs.

- [ ] **Step 3: Commit outputs**

```bash
git add output/sae_feature_match_graph_p90.json
git commit -m "data: updated concept graph and grouping with noise-floor matching"
```

---

### Task 7: Rerun ablation, transfer, importance on workers

These are GPU jobs that run on the worker pool.

- [ ] **Step 1: Sync code to workers**

```bash
python cluster.py --sync
```

- [ ] **Step 2: Launch ablation sweep**

Launch on workers using the new matching file. The `--matching-file` flag points to the new output:

```bash
# One worker at a time — see feedback_launch_jobs_individually memory
ssh surfer4 "cd /home/brian/src/tabular_embeddings && nohup /home/brian/anaconda3/envs/tfm/bin/python scripts/intervention/ablation_sweep.py --matching-file output/sae_feature_matching_mnn_floor_p90.json --worker 0 --n-workers 4 > logs/ablation_noisefloor_surfer.log 2>&1 &"
```

Repeat for terrax4 (worker 1), octo4 (worker 2), firelord4 (worker 3).

- [ ] **Step 3: Launch transfer sweep after ablation completes**

```bash
ssh surfer4 "cd /home/brian/src/tabular_embeddings && nohup /home/brian/anaconda3/envs/tfm/bin/python scripts/intervention/transfer_sweep_v2.py --matching-file output/sae_feature_matching_mnn_floor_p90.json --worker 0 --n-workers 4 > logs/transfer_noisefloor_surfer.log 2>&1 &"
```

- [ ] **Step 4: Launch concept importance after transfer completes**

```bash
ssh surfer4 "cd /home/brian/src/tabular_embeddings && nohup /home/brian/anaconda3/envs/tfm/bin/python scripts/intervention/concept_importance.py > logs/importance_noisefloor_surfer.log 2>&1 &"
```

---

### Task 8: Update figures and tables

- [ ] **Step 1: Identify affected figures**

Run a search for scripts that read matching output or ablation/transfer results:

```bash
grep -rl "sae_feature_matching\|ablation_sweep\|transfer_sweep" scripts/figures/ scripts/tables/
```

Update each script's default paths or verify they accept CLI args pointing to the new outputs.

- [ ] **Step 2: Regenerate figures**

Run the relevant figure scripts. The exact set depends on what Task 7 produces — update this step after results are in.

- [ ] **Step 3: Commit updated figures**

```bash
git add output/*.pdf scripts/figures/*.py scripts/tables/*.py
git commit -m "figures: update with noise-floor-filtered matching results"
```
