# Next steps: patching + camera-ready (handoff 2026-08-02)

Context for the next session. INCEPT (NeurIPS 19412), post-rebuttal, camera-ready +
patch-coverage commitment. This session was a marathon on the functional decomposition
(on/off-manifold) and concept selection for the patch/label commitment.

## TL;DR status
- **dVDs discussion reply is SENT** — it used trained-99 to show the sensitivity *range*
  and the 90% numbers for trained-vs-random. No parsimony multiplier stated (see below).
- **Full functional-decomposition threshold sweep is DONE**: trained + random ×
  {80,90,95,99}% variance thresholds, all 6 recipients, consolidated on the Mac.
- **Camera-ready TODO is the source of truth**: `reviews/camera_ready_todo.md` (§A–F).
  This doc is the *narrative*; the TODO is the checklist.

## Key results / decisions this session

### On/off-manifold functional decomposition
- Split each deployed transfer delta into on-manifold (projection onto recipient's
  top-eigenvector 90%-variance subspace E) vs off-manifold (complement); inject each,
  measure gap-closure. Population = **strong-wins rows that carry a deployed delta**
  (a subset of below-diagonal). Report **relative** rel_on = gc_on/gc_full,
  rel_off = gc_off/gc_full; `gc_full` stays internal (its value ~0.97 is over the
  acted-on subset and must NOT be placed next to the paper's 0.90/0.883 headlines).
- **Threshold-robust, recombination-dominant.** ALL-row rel_off across 80/90/95/99:
  trained 0.48/0.46/0.45/0.41, random 0.46/0.43/0.42/0.40. On-manifold reproduces the
  bulk (rel_on ~0.82–0.92); off-manifold is real but overlapping/secondary on mean-of-gc.
- **Energy claim DROPPED.** On-manifold energy is confounded — the archetype-random
  baseline keeps the data-derived dictionary, so both arms are on-manifold *by
  construction*; isotropic k_e/d ignores acceptance. Neither is a clean null. The
  truly-random isotropic-SAE control is the fix → deferred to a camera-ready appendix
  (TODO §C).
- **Fixed a real bug**: `functional_decomposition.py` had a blanket `except: continue`
  that silently dropped every carte-recipient dataset (pre-`predict_row_batched`), so the
  old table was a 5-recipient result masquerading as complete. Now fail-loud; carte is a
  recipient (216 rows) and the pooled rel_off rose ~0.36→0.46 as a result.

### Gap-stratified disambiguation (NEWEST, least-settled, most promising)
- **gc (mean-of-ratios) is misleading**: it normalizes out the stakes, so near-tie rows
  (tiny gap, gc≈1 on ~0 nats) get equal weight and drown the high-stakes rows.
- At 99%, mean-of-gc off-manifold looks *tied* (trained 0.40 vs random 0.37). But:
  - **Quartile-stratified by absolute logloss gap** (robust): trained rel_off rises
    0.33→0.38→0.39→0.48; random stays flat 0.33→0.37→0.37→0.42. Off-manifold matters
    MORE on high-stakes rows, trained > random at the top.
  - **Loss-weighted** (Σ gc·gap / Σ gc_full·gap = nats-removed ratio): trained rel_off
    **0.78** vs random **0.50**. Tail-sensitive (large-gap band → ~16 nats) — quote
    winsorized, lead with the quartile bands.
- This is a *functional* comparison (dictionary+acceptance held fixed) → a legit
  learned-vs-random signal gc hid. **ROBUSTNESS GATE**: only 99% is computed; must run
  80/90/95 AND confirm the trend holds across pairs/recipients (not a subset) before it
  earns a claim. Appendix section titled "Disambiguation of apparently tied off-manifold
  gc" (TODO §C).

### Concept selection for patch/label commitment
- **The core tension is unbreakable**: labelability (needs contrast = sparse firing)
  anti-correlates with importance (acceptance/universality). High-acceptance concepts
  fire ≥1000× ⇒ dense (median 99% firing) ⇒ no contrast ⇒ unlabelable. The ≥1000-firing
  universal set (98 concepts) is 43/98 *non-viable* (density ≥99.5%, zero contrast).
- Off-manifold is NOT a labelability shortcut: corr(off-manifold fraction, firing
  density) ≈ 0. Off-manifold concepts are labelable only because *most* concepts are
  sparse — the labelable ones are low-acceptance/low-universality (low-signal).
- **Landing point (the "sweet spot")**: off-manifold band **[0.6,0.8) × acceptance
  200–499 = 335 concepts** — 20.5% of the *off-manifold* contribution (NOT total),
  median firing density 0.76 (~24% contrastive rows = "hard but possible"), median
  universality 4. Meaningful AND patchable, without overclaiming "most of the
  contribution" (that's stuck in the density-~1.0 concepts). Extending to [0.4,0.6) at
  the same acceptance adds 141 → 476. **User flagged 335 is already a lot to
  label+report** — keep an "as time permits" hedge on the tail.
- **K parsimony (Table 2)**: trained 8.9 vs random 17.3 concepts/row. These are the
  *unweighted* per-dataset means from `transfer_global_mnnp90_{trained_tols,random}`.
  Decision: **quote the pair with NO efficiency multiplier** — the corrected random arm
  (`forward_deltas_random`) gives K≈21 → 4.2×, while 17.3 → 3.4×; stating a multiplier
  with 17.3 would be inconsistent. Row-weighted they're 9.7 vs 17.9.

## Committed scripts (all under scripts/rebuttal/)
- `functional_decomposition.py` — the decomposition; `--var-threshold` (default 0.90),
  fail-loud, stores ke/emb_dim/var_threshold. Population = strong-wins-with-delta.
- `functional_queue.sh` — round-robin launcher; `VAR_THRESHOLD` env to sweep.
- `threshold_sweep_table.py` — on/off + energy across thresholds, per arm
  (`--arm trained|random`). Row-pools gc, dataset-averages energy.
- `off_manifold_concept_stratification.py` — per-concept off-manifold fraction vs
  firing density / universality / acceptance.
- `gap_stratified_decomposition.py` — the gc-robust disambiguation (`--thr`).
- `aggregate_functional_clean.py` — per-recipient rel_on/rel_off, both arms (pre-sweep).

## Output dirs (consolidated on the Mac, output/rebuttal/)
- Trained: `functional_decomposition{,_t80,_t95,_t99}` (90% is the unsuffixed dir).
- Random: `functional_decomposition_random{,_t80,_t95,_t99}`.
- Deltas (decomposition inputs, global-cache verified): `forward_deltas`,
  `forward_deltas_random`.
- Stale pre-fix backups: `*_pre6recip`.

## NEXT STEPS — PATCHING (App F.3 input-level suppression patches)
1. Lock the concept set. Current recommendation: the 335 (off [0.6,0.8) × acc 200–499),
   or a tighter cut if 335 is too many to label+patch+report. Pull the enumerable list
   (donor, feat_id, density, universality, #datasets) — `off_manifold_concept_
   stratification.py` has the machinery; add a `--dump` of the cell.
2. For each concept: label it (LLM labeling in Claude Code, no API key — see project
   convention), find a column-value suppression patch, verify it lowers the concept and
   report the effect on the recipient. Report coverage + the "no qualifying patch found"
   cases honestly (dVDs asked for both).
3. Expect ~24 comfortable + a hard tail; the density-0.76 median means ~24% contrast —
   workable but not trivial. Hedge the low-priority tail as "time permits."

## NEXT STEPS — CAMERA-READY (see reviews/camera_ready_todo.md §A–F)
Active analysis threads that need compute/writing:
1. **Gap-stratified disambiguation across 80/90/95** (only 99 done) + the per-pair
   robustness check. This is the highest-value new result — gate before claiming.
2. **Truly-random isotropic-SAE control** (§C) — full pipeline for a new SAE variant
   (isotropic dictionary → fresh matching → caches → transfer → decomposition) to
   properly null the energy question and give a stronger R1 baseline.
3. **§F dVDs main-text relocations** — full-test + weak→strong 0.981 into body, donor
   prediction as required input in Sec 4.2, SAE stability from App D, geometry-matched
   control into Sec 3 ahead of numbers, per-concept gc in Tables 1/2, SD9t's 4 fixes,
   F.8 landmark summary.
4. **Dense-latents F.5 citation** (arXiv:2506.15679) — dense latents are features, and
   explain labeling difficulty.
5. **Conditional retitle** to "INCEPT: Infusing Novel Concepts to Explain Pretrained
   Tabular Model Disparity" — ONLY if a reviewer/AC requests it.
6. **`summarize_symmetric.py` fix** (pre-existing §C/internal) — transfer forward dir
   points at stale `transfer_sweep_v2`; should be `transfer_global_mnnp90_trained_tols`.

## Gotchas / provenance (don't re-learn these the hard way)
- **Monitor no-`cd` bug**: background monitors that `ls output/rebuttal/...` over ssh
  WITHOUT `cd`ing into the repo silently report 0 (land in $HOME). Cost ~2h this session
  (random-99 was done, monitor said not). Base completion checks on the absolute-path
  `/tmp/fq_*.out` "functional_queue complete" markers, not relative `ls`.
- **Env split**: standard-worker `tfm` and `tfm2` differ (tfm2 has newer tabpfn 7.0.1 /
  torch 2.7.1); tabicl_v2 pairs run in tfm2. **morg's tfm2 mismatches** standard tfm2 —
  tabicl_v2 pairs MUST run on a standard worker, not morg. morg's `tfm` differs only in a
  numpy patch (2.2.6 vs 2.3.5) — within tolerance. random-99's 10 non-tfm2 pairs ran on
  morg (this numpy diff, accepted); everything else on standard workers.
- **carte is a SEQUENTIAL_MODEL** — routed via `predict_row_batched`, no `recapture`.
  The CARTETail-recapture TODO item is superseded (carte-recipient runs now).
- **Two random transfer runs exist**: `transfer_global_mnnp90_random` (K=17.3, older)
  vs `forward_deltas_random` (K=21.3, the corrected acceptance-matched arm feeding the
  decomposition). Keep K and any efficiency multiplier from the SAME run.
