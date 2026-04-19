# Labeling pipeline — generalization experiment (handoff 2026-04-18)

## Objective

Verify that the validator-score baseline established on mitra/f_11
(`accuracy(micro)=0.660, macro=0.660, f1=0.667`) generalizes across the
other 4 mitra features that have contrastive CSVs (`f_6, f_36, f_86,
f_92`). Result tells us whether the pipeline is reproducibly decent
(worth HP-tuning) or whether f_11's score was a fluke.

Pipeline stages (see `docs/labeling_pipeline.md` for full reference):

```
contrastive CSVs + validator CSVs
    -> round-1 agents (one per dataset)
       -> judge (per-round)
          -> ring rounds 2..5 (agents + judge each round)
             -> judge 'done' (or max rounds)
                -> final label (judge-synthesized)
                   -> held-out validator
                      -> metrics (micro/macro acc, f1, per-dataset)
```

## Current best config

| Knob | Value |
|------|-------|
| `PROMPT_ORDER` | **A** (evidence before instructions; winner over B and C) |
| `MAX_ROUNDS` | 5 (judge may stop earlier with `done`) |
| `PEER_SAMPLE_N_ACT` / `PEER_SAMPLE_N_CON` | 3 / 3 |
| `JUDGE_SAMPLE_N_ACT` / `JUDGE_SAMPLE_N_CON` | 2 / 2 |
| Validator held-out | 5 activating + 5 non-activating per dataset (50 rows total for a 5-dataset feature) |
| Agent model | all opus (round-1, ring, judge, validator) |
| Feature-level CSV regen | requires `_classify_column` fix for pandas string dtype (commit `ed2106e`) |

Prompt / evaluator provenance:
- `PROMPT_ORDER` is implemented in `scripts/concepts/label_contrastive_mesh.py` via env var `PROMPT_ORDER`; prompt-family ordering is assembled in `_assemble(...)`.
- Worker prompts (`round1_prompt`, `ring_prompt`) should stay on **A** for this sweep. Do not mix prompt families across features within this generalization run.
- Validator prompt already includes the anti-bias instruction from `docs/labeling_pipeline.md`: judge each row independently; do not assume any particular number of rows should fire in a dataset.
- Use the first judge `done` verdict as the stopping point. Do **not** continue to round 5 just because `MAX_ROUNDS=5`.

Resume invariants:
- Keep validator CSVs / truth files fixed for a given feature; do not regenerate mid-sweep.
- Keep validator model and wording fixed; otherwise scores are not comparable to `f_11` / `f_6`.
- Snapshot `f{feat}_label.json`, `f{feat}_mesh_state.json`, and the validator metrics immediately after `record-validator`.
- Treat cross-feature mean / spread as the real comparison target; single-feature deltas of ~0.1 can be noise.

## Results so far

| Feature | Rounds to `done` | Per-dataset verdicts at done | micro / macro / f1 | Final label |
|---------|------------------|------------------------------|--------------------|-------------|
| **f_11** | 4 (judge done r4) | 5 accept | **0.660 / 0.660 / 0.667** | "Activating rows concentrate a localized column subset at its modal or rare co-firing signature with confident-correct predictions, while contrasts break the localization or spread into extreme tails." |
| **f_6**  | 4 (judge done r4) | 4 accept, 1 revise | **0.660 / 0.660 / 0.691** | "Activating rows place tail-mass values (rare categorical codes or extreme-percentile numerics) on a localized column subset while contrasts hold baseline-frequency values at those positions." |
| **f_36** | 3 (judge done r3) | 4 accept, 1 revise | **0.680 / 0.680 / 0.680** | "Activating rows densely cover a broad column block at mid-band percentiles or rare-frequency categorical levels; contrasts thin out or push isolated positions to extreme tails." |
| **f_86** | 4 (judge done r4) | 5 accept | **0.200 / 0.200 / 0.130** ⚠️ polarity-inverted | "Activating rows concentrate tail mass on a localized numeric column subset at upper-mid-to-tail percentiles; contrasts lack that subset or invert its polarity." |
| **f_92** | 3 (judge done r3) | 5 revise (but done) | **0.420 / 0.420 / 0.293** ⚠️ partial polarity mismatch | "Activating rows concentrate tail-mass on a narrow recurring column subset with near-saturated prediction confidence, while contrasts disperse comparable tail-mass across broader non-overlapping columns at lower confidence." |

Per-dataset accuracies:

**f_11:** hazelnut 0.80, NATICUSdroid 0.80, students 0.60, Marketing 0.60, splice 0.50.
**f_6:** HR 0.70, NATICUS 0.70, churn 0.70, credit_card 0.70, in_vehicle 0.50.
**f_36:** kddcup09 0.80, SDSS17 0.80, taiwanese 0.70, Marketing 0.60, Diabetes130US 0.50.
**f_86:** wine_quality 0.60, miami 0.20, superconductivity 0.20, diamonds 0.00, physiochemical_protein 0.00. **Every diamonds/protein prediction is exactly inverted from truth — the label's polarity is backwards.**
**f_92:** SDSS17 0.50, NATICUSdroid 0.50, splice 0.50, hiva_agnostic 0.50, APSFailure 0.10. APSFailure's activating pattern is "broad co-firing across wide column set" but the label calls for a "narrow recurring subset" — same class of polarity mismatch as f_86 but only on one dataset.

Running aggregate over completed features (n=5):
- mean micro accuracy = **0.524** (0.660, 0.660, 0.680, 0.200, 0.420)
- mean macro accuracy = **0.524**
- mean f1 = **0.492**
- Excluding f_86 and f_92 as polarity-inversion cases: mean = **0.667** on n=3.

### f_86 note

Initial f_86 label was polarity-inverted for protein + diamonds
(0/10 each). Added `next_round_pairings` to the judge schema (commit
`e0ffb63`): judge dynamically pairs contradicting datasets in the
next ring round, so agents reconcile face-to-face rather than the
synthesizer papering over direction conflicts. Re-ran f_86 on the
pairing-enabled pipeline:

| dataset | pre-pairing | with pairings |
|---|---|---|
| diamonds | 0.00 | **0.70** |
| physiochemical_protein | 0.00 | **0.60** |
| miami_housing | 0.20 | 0.30 |
| wine_quality | 0.60 | 0.40 |
| superconductivity | 0.20 | 0.20 |
| **overall** | **0.20** | **0.44** (+0.24) |

New label (polarity-agnostic): *"Activating rows concentrate most axes
in one coherent percentile band with one or two axes pushed to the
opposite tail, while contrast rows scatter across dispersed mid-range
positions without that bulk-plus-outlier co-firing."*

Judge used pairings as designed: round 1 paired protein↔supercon (the
polarity conflict); round 2 paired diamonds↔protein and
wine↔supercon (struggling datasets with accepted ones); round 3 done
with 4 accepts + 1 revise. Snapshots at
`output/contrastive_examples/mitra/f86_{label,mesh_state}_Apairing.json`.

Next: re-run f_92 with pairings and see if the APSFailure polarity
mismatch resolves similarly.

State + label snapshots on disk:
- `output/contrastive_examples/mitra/f11_label_A.json`
- `output/contrastive_examples/mitra/f11_mesh_state_A.json`
- `output/contrastive_examples/mitra/f6_label_A.json`
- `output/contrastive_examples/mitra/f6_mesh_state_A.json`
- `output/contrastive_examples/mitra/f36_label_A.json`
- `output/contrastive_examples/mitra/f36_mesh_state_A.json`

Paper-draft note:
- These two labels are good enough to use as draft placeholders in the paper while the remaining three features run.
- Once `f_36`, `f_86`, `f_92` finish, prefer a 5-feature writeup with representative examples rather than over-focusing on `f_11`.

## Pending concepts

All 5 features complete.

Each remaining feature needs:

1. `python3 -m scripts.concepts.label_contrastive_mesh --model mitra --feat <N> reset`
2. `init` → dispatch 5 round-1 opus agents in parallel; save agent outputs to a JSON list in dataset order from `datasets_used`
3. Record labels → `show-judge` → dispatch judge (opus) → `record-judge`
4. If judge returns `continue`: `show-round --round k+1` → dispatch → record → judge → repeat until `done` or r5
5. `save-label`
6. `show-validator` → dispatch validator (opus) → `record-validator`
7. Snapshot: copy `f{N}_label.json` to `f{N}_label_A.json` and `f{N}_mesh_state.json` to `f{N}_mesh_state_A.json`

Expected: per previous two features, ~4–5 rounds, done at r4, ~25 agent calls per feature.

Notes on dispatch / files:
- Round prompt dumps are written to `/tmp/f${FEAT}_r{k}.txt`; these are useful for postmortem label-evolution inspection.
- Validator prompt dumps are written to `/tmp/f${FEAT}_val.txt`; validator raw response should be saved to `/tmp/f${FEAT}_val_resp.txt`.
- `record` expects a JSON list of labels in `datasets_used` order, not a dataset→label mapping.
- `record-judge` / `record-validator` expect the raw fenced-JSON model response, not a manually extracted object.

## Next steps (in priority order)

1. **Complete f_36, f_86, f_92 sweeps.** Populate the results table; check whether validator score stays near 0.66 across all 5 features or whether some features are harder (e.g., regression-only f_86 may behave differently).
2. **Write up 5-feature generalization result in `docs/labeling_pipeline.md`** — extend the PROMPT_ORDER sweep section with a "5-feature f_A scores" subsection and pick a canonical baseline mean ± std.
3. **If generalization holds**, start HP-tuning from the config above. Candidate knobs (in order of expected leverage):
   - `JUDGE_SAMPLE_N_ACT/CON` 2/2 → 3/3 or 5/5 (denser evidence per judge call)
   - `MAX_ROUNDS` 5 → 8 (lets judges like f_86's harder ones keep iterating)
   - Agent model mix: haiku round-1 + opus ring + opus judge (cost reduction, minor expected accuracy hit)
   - Round-1 prompt priming: "label will be held-out validated" — may improve or may not
4. **If generalization fails** (e.g., one feature drops to 0.50), investigate what's different about that feature before tuning: is it a regression-only feature, is the firing signal genuinely fuzzier, are contrastive CSVs mis-stratified?

## Known caveats

- f_11's `splice` dataset had pre-fix `_classify_column` bug that emitted garbage `(pXX)` on pandas string dtype. All contrastive + validator CSVs have been regenerated since the fix (commit `ed2106e`), so results from this session are on corrected data.
- Validator is held-out (rows not in contrastive CSVs), but 5+5 per dataset is small; variance is real. Per-dataset acc swings of ±0.1 across orderings on the same feature are within noise; cross-feature means are the comparable quantity.
- Agents occasionally leak domain terms (e.g., splice → "CAG-GG", "pyrimidine"). The judge is instructed to flag these and usually does. We have not yet built an automatic linter.
- `f_86` is regression-only; do not be surprised if its validator behavior differs from the classification-heavy features. Record it faithfully before changing knobs.
- If a future session uses a different shell environment where `python` is absent, use `python3` consistently for the CLI commands below.

## Commands to resume

```bash
# For each remaining feature N in {36, 86, 92}:
FEAT=N
python3 -m scripts.concepts.label_contrastive_mesh --model mitra --feat $FEAT reset
PROMPT_ORDER=A python3 -m scripts.concepts.label_contrastive_mesh --model mitra --feat $FEAT init > /tmp/f${FEAT}_r1.txt

# dispatch 5 opus agents in parallel (one per dataset); collect labels in dataset order; save JSON list to /tmp/f${FEAT}_r1_labels.json
PROMPT_ORDER=A python3 -m scripts.concepts.label_contrastive_mesh --model mitra --feat $FEAT record --labels-file /tmp/f${FEAT}_r1_labels.json
PROMPT_ORDER=A python3 -m scripts.concepts.label_contrastive_mesh --model mitra --feat $FEAT show-judge > /tmp/f${FEAT}_judge_r1.txt
# dispatch judge; save raw fenced JSON to /tmp/f${FEAT}_judge_r1_resp.txt, then:
PROMPT_ORDER=A python3 -m scripts.concepts.label_contrastive_mesh --model mitra --feat $FEAT record-judge --response-file /tmp/f${FEAT}_judge_r1_resp.txt

# if judge says continue, repeat for round k:
PROMPT_ORDER=A python3 -m scripts.concepts.label_contrastive_mesh --model mitra --feat $FEAT show-round --round K > /tmp/f${FEAT}_rK.txt
# dispatch 5 opus agents; save JSON list to /tmp/f${FEAT}_rK_labels.json
PROMPT_ORDER=A python3 -m scripts.concepts.label_contrastive_mesh --model mitra --feat $FEAT record --labels-file /tmp/f${FEAT}_rK_labels.json
PROMPT_ORDER=A python3 -m scripts.concepts.label_contrastive_mesh --model mitra --feat $FEAT show-judge > /tmp/f${FEAT}_judge_rK.txt
# dispatch judge; save raw fenced JSON to /tmp/f${FEAT}_judge_rK_resp.txt, then:
PROMPT_ORDER=A python3 -m scripts.concepts.label_contrastive_mesh --model mitra --feat $FEAT record-judge --response-file /tmp/f${FEAT}_judge_rK_resp.txt

PROMPT_ORDER=A python3 -m scripts.concepts.label_contrastive_mesh --model mitra --feat $FEAT save-label
PROMPT_ORDER=A python3 -m scripts.concepts.label_contrastive_mesh --model mitra --feat $FEAT show-validator > /tmp/f${FEAT}_val.txt
# dispatch validator agent, save response, then:
PROMPT_ORDER=A python3 -m scripts.concepts.label_contrastive_mesh --model mitra --feat $FEAT record-validator --response-file /tmp/f${FEAT}_val_resp.txt

# snapshot
cp output/contrastive_examples/mitra/f${FEAT}_label.json output/contrastive_examples/mitra/f${FEAT}_label_A.json
cp output/contrastive_examples/mitra/f${FEAT}_mesh_state.json output/contrastive_examples/mitra/f${FEAT}_mesh_state_A.json
```
