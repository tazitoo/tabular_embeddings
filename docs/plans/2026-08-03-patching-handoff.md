# Next session: App F.3 patching (handoff 2026-08-03)

Supersedes the "NEXT STEPS — PATCHING" section of
`docs/plans/2026-08-02-patching-camera-ready-next-steps.md`. That doc's
camera-ready narrative still stands; only the status below is newer.

## Start here

The patching thread is **untouched** — this session never got to it. Pick up at
step 1 of the previous handoff:

1. **Lock the concept set.** Current recommendation from 2026-08-02: the 335
   concepts at off-manifold band [0.6,0.8) x acceptance 200-499 (median firing
   density 0.76, median universality 4). User flagged 335 as already a lot to
   label + patch + report — a tighter cut is acceptable, with the tail hedged as
   "as time permits". Add a `--dump` of the cell to
   `scripts/rebuttal/off_manifold_concept_stratification.py` to get the
   enumerable list (donor, feat_id, density, universality, #datasets).
2. Label each concept (LLM labeling in Claude Code, no API key).
3. Find a column-value suppression patch per concept, verify it lowers the
   concept, report the effect on the recipient. Report coverage AND the
   "no qualifying patch found" cases — dVDs asked for both.

## What changed underneath you (all of it is infrastructure, none of it moves results)

The whole session went into environment/reproducibility work. Net effect on the
numbers: **nothing**. Quantified on `rel_off`, the reported quantity:

| source of variation | effect on rel_off |
|---|---|
| carte nondeterminism (one arbitrary draw -> deterministic) | -0.0003 |
| the 160-package env migration | 0.0000 to +0.0016 |
| GPU architecture (3090 vs 4090) | +0.0114 pooled, ~85% of it carte |

All sub-noise against per-recipient rel_off spanning 0.11-0.94. **The existing
numbers were never at risk.** Do NOT re-litigate prior results on account of any
of this. Reproduce with `scripts/rebuttal/compare_gc_across_runs.py`.

### Fleet is now on one locked environment
- All 5 hosts: python 3.12.12, numpy 2.3.5, scipy 1.16.3, pandas 2.3.3,
  sklearn 1.6.1, torch 2.6.0+cu124. `uv.lock` (176 pkgs) is committed.
- surfer and morg were REBUILT from python 3.10; old envs kept as
  `tfm_py310_backup` on both.
- Rebuild any worker: `conda create -y -n tfm python=3.12.12` then
  `UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX uv sync --extra all --inexact`.
- **tfm2 has NOT been rebuilt or locked on any host.** It has its own spec at
  `envs/tfm2/pyproject.toml` (it pins torch 2.7.1 against root's 2.6.0, so it
  cannot be an extra — `uv lock` proved this by failing). Its
  `autogluon==1.5.0` pin is UNVERIFIED against torch 2.7.1. Any tabicl_v2 work
  needs this sorted first.

### carte determinism is fixed
`functional_decomposition.py` now sets `use_deterministic_algorithms(True)` +
`CUBLAS_WORKSPACE_CONFIG`. carte was the only nondeterministic model (it is the
only recipient whose tail is TRAINED). Verified not to perturb the other five.
Full characterisation in `docs/reproducibility.md`.

### Two spec bugs fixed that would have broken a fresh clone
- `openml` was undeclared despite being imported by step one of the pipeline.
- `matching` (pymfe, imported from `scripts/__init__.py`) was missing from `[all]`.

## Analysis threads CLOSED this session (do not reopen without new evidence)

- **Gap-stratified disambiguation** (the 2026-08-02 handoff's "highest-value new
  result"): **failed its robustness gate.** Per-pair, trained Q4>Q1 in 13/27 and
  random in 13/29 — indistinguishable. The pooled trend was a composition
  effect. Recorded in `reviews/camera_ready_todo.md` SS C with the full table.
- **The pooled on/off-manifold number**: has no coherent reading. E is the
  recipient's OWN eigenbasis per dataset, so pooling averages quantities defined
  against different bases (effective rank ke=1-11/300 for carte vs 43-128/768
  for tabdpt). Report per-recipient or not at all.
- **99% threshold**: settled on **90%** instead. At 99%, 118/672 cells draw more
  eigenvectors than the strong-wins population can support (n < d in every
  cell); at 90% only 2/672 do. Clean 90% back-to-back for both arms is in
  `functional_decomposition{,_random}_t90` with full ke/var_threshold provenance.

## Gotchas

- **The 40GbE VLAN went down mid-session.** `surfer4`/`firelord4`/etc. were
  unreachable; `surfer.local`/`firelord.local` (1GbE) worked. CLAUDE.md still
  says "Always SSH to the 4 variant" — add a fallback caveat at some point.
- **octo is the slowest worker** (2.5x slower than firelord on the same 3 pairs:
  116 min vs 46). Don't put the carte-recipient pair there; carte trains its
  tail per dataset and dominates runtime.
- **`~/.claude/.last-update-result.json`** caches the "Auto-update failed"
  banner. A stale entry keeps complaining long after the cause is gone.
