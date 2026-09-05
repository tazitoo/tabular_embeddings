# Next session: App F.3 patch summarization (handoff 2026-09-05)

Follows `docs/plans/2026-08-12-patching-handoff.md`. The patching sweep is done
and v31 is canonical; this session merged the branch, closed the population to
335/335, and built the tables that summarize the 320 patched concepts.

## Start here

Most of `reviews/camera_ready_todo.md` (34 open items) is paper prose — §A, §B,
§D and §F want the paper open, not a continuation of table work. Three items in
§C are still warm because the plumbing now exists:

1. **Per-concept gap-closure into Tables 1 and 2** (todo §F). `patch_effect` and
   `interval_readout` are already loaded by `scripts/tables/patch_explanations.py`;
   adding a column to `patch_pipeline_tables.py` is mechanical.
2. **`acc` (acceptance) column on the ablation table** — same shape as the `Rows`
   column settled on in the explanation panels.
3. **Full-test-set numbers replacing the oracle subset** — the big one, needs
   compute, not a table change.

## What shipped

`main` is at `1a77992`, pushed. Two generators, both dual-writing to the paper
repo (their copies there are **untracked and need their own commit**):

- `scripts/tables/patch_pipeline_tables.py` — attrition 335 -> 320, and patch
  character by transfer-rank quartile.
- `scripts/tables/patch_explanations.py` — recipient x donor diagnostic matrix,
  four quartile panels, and `patch_explanations.csv` (all 877 cells).

The fleet is on `c74560e` (one commit behind) and does not need these; sync when
experiments resume.

## Decisions worth not relitigating

- **Prediction units, not gap closure, for single rows.** gc divides by the
  strong-weak gap; a small denominator turns a 0.008 probability move into a gc
  of 0.48. The honest scale of changing one concept of ~256 is 1e-3.
- **Stratify by the ABLATION effect, not the patch's own.** Ordering by our
  result floats successes up and buries failures.
- **Report the row the objective selected**, not the largest suppression — the
  latter reports the acceptance criterion rather than the pipeline.
- **Extinguished rate, not median suppression.** Suppression is bimodal (exactly
  1.0 or stalled, nothing in 0.9-1.0), so the median is a step function of the
  rate.
- **Table 1's population is TWO filters**, not one: off-manifold fraction in
  [0.6, 0.8] AND acceptance count in [200, 499], from 6088 accepted concepts.

## Open, and not written down anywhere else

**The "17 of 335" in `camera_ready_todo.md` does not reproduce.** The line claims
dropping TabDPT-recipient cells costs 17 concepts (335 -> 318). Confirmed: 259
concepts have tabdpt among their recipients, exactly matching the comment in
`patch_search.py`. Not confirmed: the 242/17 split — every one of those 259 has
another recipient, so the count comes out 259/0. The 17 only appears after the
carte/env/wide filter chain runs first, which was not replicated. **Do not ship
that number** without re-deriving it.

**Suppression difficulty is driven by activation magnitude.** `a_start` vs
suppression rho = -0.42 (n=2772, p~1e-119); the number of available columns has
no effect (rho = +0.001). Patches remove a roughly fixed 1-4 units of activation
regardless of the target, so the extinguish rate falls 80% -> 19% across a_start
quartiles. Three candidate mechanisms were tested and contradicted: reach,
collateral (blast is *lower* at high activation, rho = -0.068), and co-activation
density (*fewer* co-actives at high activation, rho = -0.311). Unexplained.

**Concept-vector subtraction was tried as an alternative attribution method and
does not replace patching.** Subtract a_c*d_c, look up the nearest real row, read
the input diff. Findings: works in embedding space only for TabPFN (51% clean-pair
rate vs 0.0-0.2% for every other donor, tracking TabPFN's ~7 intrinsic dims); the
retrieved pairs differ in 50-164 raw columns because matching codes does not
constrain the table; and on the six cases where a null test had any power, it
agreed with the patch pipeline on **zero** columns. Under a size-matched null only
2 of 27 cases survive Bonferroni. Scripts left uncommitted in `scripts/rebuttal/`
(`_sweep_sub_tmp.py`, `_minpair_tmp.py`, `_axis_tmp.py`, `_ctxmatch_tmp.py`,
`_setdisc_tmp.py`, `_null_tmp.py`, `_cmp_tmp.py`, `_coldiff_tmp.py`,
`concept_subtraction_neighbors.py`) in case the method is worth refining.

This bears on todo §D "systematic human/ground-truth concept-validation": two
plausible attribution methods disagree and neither has ground truth. The 16
synthetic probing generators are the route to settling it, with the caveat that
**TabPFN is trained on synthetic priors**, so synthetic validation measures it in
unusually favourable conditions and the donor-stratified result would be the
informative one.

## Smaller notes

- `tabdpt f78` was never dispatched in the sweep; it ran clean on firelord4 and
  is at `output/rebuttal/v31q_f78.json` (loaded by a second glob, not folded into
  `v31q/`). Population is now 334/335 searched, the one loss being `tabicl_v2
  f839`, whose only cells need TabICL v1 and v2 in one env.
- `output/pymfe_tabarena_cache.json` was renamed to
  `..._degraded_2026_08.json` (`c74560e`): an Aug staging error overwrote the
  full 145-feature cache with `--fast` output (73 features). Nothing on the
  patching path reads it; the Feb version is recoverable at
  `git show ab71d38:output/pymfe_tabarena_cache.json`.
- Two stale test modules fail at collection on `main` and predate this work:
  `tests/test_concept_hierarchy_full.py` and
  `tests/test_label_cross_model_concepts.py` import symbols that no longer exist.
