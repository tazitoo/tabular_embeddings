# Patching handoff — 2026-08-12

## Read this first

**Use what the transfer/intervention pipeline already has.** Nearly every defect this
session came from building alongside existing, working machinery instead of reusing it.
The parts that worked first time were the borrowed ones: the crossing guard from
`transfer_sweep_v2` (overshoot 26.1% -> 0%), `_gc`, the linearity of `delta_r` that makes
LOO arithmetic, `gc_tolerance`. The parts that needed three or four reworks were invented.
Before adding anything, grep the pipeline for it.

**Docstrings here state intent, not behaviour — verify before trusting.** `recon_excess`
claims to judge against "the real-row range from `sae_insample_null.py`". It does not.

## State

Branch `feature/app-f3-patching`, HEAD around `56cf32a` + the uncommitted `blast + EPS`
change in `objective()`.

Last complete sweep is **v15** (`output/rebuttal/patchv15clf_*.json`), 335 concepts, zero
errors. **v16 was launched and killed** — its blast term was known-broken. Nothing since
v15 is a valid result.

v15 numbers, with the caveat below: landing 71.8%, undershoot 26.1%, overshoot 0%, rows
patched 84.1%, coverage 335/335 with 3 concepts finding no qualifying patch (mitra f2,
mitra f10, tabicl_v2 f839).

## Open defects, in the order I would fix them

### 1. `recon_excess` has no working off-manifold guard  (worst)
`--recon-bar` defaults to `None`, so the hard filter never fires. `rex` divides by the
row's OWN unpatched reconstruction error, which is the wrong reference: a row going
0.10 -> 0.50 scores +400% and one going 0.40 -> 0.50 scores +25%, though both END in the
same place, and it is the ending place that says whether the dictionary can represent the
row.

`sae_insample_null.py` exists to answer exactly this ("what reconstruction error do REAL
rows have? the null a patched row is judged by") and was never computed or wired in — no
`sae_insample_null.json` on disk. **Use it.** The right test is positional against the
distribution of real rows for that (model, dataset), cached per cell.

Consequence: no sweep so far had a working off-manifold guard, so `drop_frac` = 1.000 may
include rows suppressed by leaving the representable region.

### 2. The objective's form is unsettled
Settled by the user: the score is RELATIVE, only ranking matters, and a 10% change in any
term should be equivalent. That means a pure product of ratios, `score = prod(t_i^c_i)`,
where equal exponents give equal sensitivity to relative change. Additive constants break
that property — `1 + x` makes a 10% change in `x` worth different amounts depending on
where `x` sits, and it is a WEIGHT wearing a guard's clothes. A divisor needs `+ EPS`, not
`+ 1`.

Uncommitted: `blast + EPS` is in `objective()`. `recon_excess` still has `1 +` and cannot
lose it while it is an EXCESS (zero-based); it would need to become the ratio
`recon'/recon_0` — but that ratio REWARDS reconstructing better than the original, which
the one-sided form deliberately declines to do. Unresolved.

### 3. Exponents never tested
`EXPONENTS = {"drop": 1.0, "reversal": 0.5, "blast": 1.0, "recon": 1.0}`, settable via
`--exponents`. Only ratios matter (raising all by k gives score^k, same ranking), so fix
drop=1 and read the rest against it.

`reversal: 0.5` is stale. The sqrt was chosen to damp an UNBOUNDED term that was rewarding
overshoot. The crossing guard now bounds it at 1, and on [0,1] a sqrt COMPRESSES
differences between low values — so it currently makes the search less willing to trade
suppression for recipient movement, the opposite of its purpose. Test after 1 and 2.

### 4. `rel_j > 1` on 27.8% of rows
The collateral cost model is linear in `rel_j = |da_j|/|a_j|`, which exceeds 1 when a
concept more than doubles. Whether "grew 300%" should cost 3x "removed entirely" is
unexamined. Visible in the recorded `collateral` detail.

### 5. Pre-existing, unrelated to this session's work
- mitra recipient tails assert `predict_stats is only applicable for regression tasks` on
  regression datasets (task #24). Not hit under `--task classification`.
- Rows where the concept re-extracts to ~0 (task #19).
- `acceptance counts greedy events, not deployed rows` (task #22).

## Hardware — do not attribute these to models or code

**octo4 corrupts computation.** Non-ECC RAM. Three python3.12 segfaults since Aug 8, at
scattered addresses (`76713782ee38`, `1000008`, `1`), mixed read/write errors — the
signature of memory corruption, not a deterministic bug. It is ALSO the only host
producing canary failures: reproducible numerical drift of 0.03-0.89 on specific rows that
every other host computes cleanly. One cause explains both.

I mis-attributed these twice — first to tabdpt having a content-dependent query window,
then to a query-set size effect. **The host rotation settled it**: failures followed the
box, not the arm. Rotate hosts before concluding anything about a model.

Excluded from sweeps since v13. Cleaned and reseated 2026-08-12; memtest was running, no
errors at 52% of pass 1. The decisive workload test is `seismic-bumps row 172`, which
drifted 0.73 reproducibly on octo4 and is clean everywhere else — run it before readmitting
the box.

**firelord4 also segfaults** (3 during v15) but has produced no canary failures. Same class
of risk, unresolved. Sweeps survive it because runs resume from their output file.

**Check the journal, not just the log.** Every mid-sweep death across v8/v10/v12/v15 was a
segfault leaving no traceback. `journalctl -k | grep segfault` had the answer for days
while I hypothesised SIGHUP and OOM.

## What is sound

- Query window never grows; candidates cycle through `len(X_query)`, and the unmodified
  row rides in every batch as a canary. This caught contamination in every sweep v3-v8:
  tabpfn's baseline moved 1.2e-02 in a batch of 238, tabicl's 6.2e-03 in 1577.
- `tests/test_query_window_invariance.py`, 10 tests, passing on a worker.
- Raw space is the default and verified: refit reproduces the cache on 204/204
  (model, dataset) pairs; patches report real column names and values.
- `column_types` is preprocessing's answer verbatim, no second classifier.
- Per-column line search, 192 values on one column per forward.
- Per-concept LOO, `collateral` detail, `row_additivity`, objective logged factor by
  factor, `trajectory` per committed column.
- `DONE <out> concepts=N ok=N errors=N` terminal marker; `job_done` matches a process to
  its log via `/proc/<pid>/fd/1`, so two arms on one host resolve independently.

## Next run

Fix 1, decide 2, then a single sweep with exponents unchanged as the baseline the exponent
sweep is measured against. Four hosts (firelord4, surfer4, terrax4, morg x2 GPUs), octo4
out. Sample stays `--n-datasets 3 --n-rows 10 --top-cols 8`.

Every objective term should get a test before the sweep, the way the window property has.
Each was found wrong by a sweep, days later, after producing numbers. `recon_excess` is
testable in seconds — construct a row far outside the dictionary's range, assert the term
flags it — and would have failed on day one.
