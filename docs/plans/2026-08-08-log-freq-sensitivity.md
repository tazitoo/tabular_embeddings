# Log-frequency sensitivity and raw-space patching (v7)

Status: designed, not implemented. v6 is the current sweep.

## Why

Two branches of the patch search were measuring different things and being compared as
though they were the same:

| | probe generator | edit cost |
|---|---|---|
| continuous | `x0 ± 0.5 x IQR` (local finite difference) | `abs(delta) / IQR` |
| categorical | levels across the whole distribution | `-log p(level)` |

A continuous probe's cost is pinned near `step_frac` = 0.5 by construction; a categorical
probe on Amazon spans 1.50 to 8.54 nats. So categorical columns produce larger responses
and rank above continuous ones for reasons that have nothing to do with which column
controls the concept. The same seam biases the minimal-edit tie-break toward continuous
columns.

It also caused the measured v5 -> v6 tabicl regression: median `drop_frac` 1.000 -> 0.561,
worse on 51.4% of rows, while every other donor held at 1.000. Correcting `column_types`
to match preprocessing left tabicl with nothing categorical (imputation clears
`cat_indices`), so its low-cardinality integer columns moved to the continuous branch and
lost access to their own levels -- a 5-value column went from "try all 5" to "try one
IQR-step either side".

## The design

Everything on one scale: `L(x) = log freq(x)`. High at the mode, low in the tails.

**Step pool from the histogram, not a chosen step size.** Bin the column once (bins for
numeric, levels for categorical). From the row's current bin, every other bin is an
available destination carrying its own `delta L`. There is no requested step size and no
snapping, so `step_frac` is deleted rather than renamed -- it was a parameter that only
existed because we pretended to choose a step when we are choosing from a menu.

**One-sided first-order differences, both types.**

    delta L = abs(L(x') - L(x))        from the histogram
    g       = [a_c(x) - a_c(x')] / delta L

Centred differencing was considered and rejected: it only exists for continuous columns,
so ranking would compare a second-order estimate against a first-order one. One-sided
everywhere means both types use the identical estimator, which is the precondition for
ranking them against each other. It also removes all left/right bookkeeping -- `L` is
non-monotone in `x` (peaks at the mode), so it cannot serve as a signed coordinate, but
each probe standing alone never needs it to.

**Pass 1 ranks on the concept alone.** Currently it ranks by SELECTIVITY -- concept
movement per unit of collateral -- which applies the objective's blast penalty inside the
generator. That discards columns that move the concept hard and others hard, so pass 2
never discovers that a milder step down the same column is selective. The blast that
disqualified the column was a property of the probe's size, not of the column. Rank on
`g` and nothing else; keep measuring the others' response, since pass 2 needs `blast` per
candidate anyway and `selectivity_ratio` stays useful as a diagnostic (it was the only
patch-side quantity that tracked overshoot: 29.1% in its lowest quartile to 13.8% in its
highest).

`rank_columns` is already correct -- rank-correlation with the concept's cached
activation, target only, no forward passes.

**More than one bin per column**, spread across the available `delta L` range rather than
clustered at similar rarity. Not for the ranking's sake -- one probe would nearly do
there -- but because pass 2 inherits pass 1's values, and a single bin per column
hard-wires the maximum-suppression menu that causes overshoot.

## Raw space

Built and gated already (`Space`, `preprocessed_space`, `raw_space`); not yet wired into
the search. `raw_space` refuses to return a space whose transform does not reproduce
`X_query` exactly. `verify_preprocessor_refit` confirms the refit is exact on 204/204
(model, dataset) pairs, and the round-trip through `materialize` is exact on the query
rows in use.

Three reasons the patch belongs in raw space:

1. It becomes model-independent. MIC is 94 categorical columns for every donor in raw
   space; in preprocessed space it is 88 for tabpfn/tabdpt and 0 for tabicl/mitra, so two
   donors search structurally different spaces over identical data. This is the root of
   the tabicl regression, not just a reporting inconvenience.
2. Cross-column consistency is automatic -- a row assembled by editing preprocessed cells
   can be one the pipeline could never emit; a transformed raw row cannot be.
3. Categorical typing comes from the raw dtype, which is what the generator keys on.

Dropped columns (Amazon `ROLE_CODE`) need no filter: the generator never reads them, so
their measured sensitivity is exactly 0.0 and they rank last. That exactness is a free
correctness check on the raw wiring -- a nonzero value means variants are not going
through `materialize`, or rows are misaligned.

## Scope for v7

In: histogram step pool, one-sided `delta a / delta L` slopes, drop-only pass-1 ranking,
`step_frac` deleted, search wired to raw space.

Out (future work): pass 2 picking values from the histograms directly. It keeps pinning
each column to its most-suppressive pass-1 value.

**Consequence to expect:** overshoot will NOT improve in v7. Only 8.8% of chosen patches
land in 0.8-1.2 reversal, because the objective never sees a milder candidate, and that
stays true while pass 2 pins maxima. v7 should move ranking quality, tabicl's
suppression, and cross-type comparability.

## Open

- Reducing several candidate slopes to one number per column for ranking: largest `abs(g)`
  answers "can this column move the concept", which is what pass 1 is for; smallest step
  is the more accurate derivative but answers a question we are not asking.
- The step pool is relative to where the row sits, so a column's sensitivity is per-row,
  not a column constant. Pooling across rows for a per-concept ranking should carry
  `delta L` alongside rather than averaging bare slopes.
- The objective rewards overshoot: `sqrt(reversal)` is unbounded and monotone, so a patch
  overshooting 4x scores twice one that lands. Ruled as a search problem, not an objective
  problem -- the objective picks correctly from a set containing no landing patch.
