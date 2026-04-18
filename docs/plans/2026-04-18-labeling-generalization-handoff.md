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

## Results so far

| Feature | Rounds to `done` | Per-dataset verdicts at done | micro / macro / f1 | Final label |
|---------|------------------|------------------------------|--------------------|-------------|
| **f_11** | 4 (judge done r4) | 5 accept | **0.660 / 0.660 / 0.667** | "Activating rows concentrate a localized column subset at its modal or rare co-firing signature with confident-correct predictions, while contrasts break the localization or spread into extreme tails." |
| **f_6**  | 4 (judge done r4) | 4 accept, 1 revise | **0.660 / 0.660 / 0.691** | "Activating rows place tail-mass values (rare categorical codes or extreme-percentile numerics) on a localized column subset while contrasts hold baseline-frequency values at those positions." |

Per-dataset accuracies:

**f_11:** hazelnut 0.80, NATICUSdroid 0.80, students 0.60, Marketing 0.60, splice 0.50.
**f_6:** HR 0.70, NATICUS 0.70, churn 0.70, credit_card 0.70, in_vehicle 0.50.

State + label snapshots on disk:
- `output/contrastive_examples/mitra/f11_{label,mesh_state}_A.json`
- `output/contrastive_examples/mitra/f6_{label,mesh_state}_A.json`

## Pending concepts

| Feature | Datasets | Status |
|---------|----------|--------|
| **f_36** | Diabetes130US, kddcup09_appetency, Marketing_Campaign, SDSS17, taiwanese_bankruptcy_prediction | validator CSVs built (`output/contrastive_examples/mitra/f36_validator_*.csv` + `f36_validator_truth.json`); pipeline NOT yet run |
| **f_86** | diamonds, miami_housing, physiochemical_protein, superconductivity, wine_quality (all regression) | validator CSVs built; pipeline NOT yet run |
| **f_92** | APSFailure, NATICUSdroid, SDSS17, hiva_agnostic, splice | validator CSVs built; pipeline NOT yet run |

Each remaining feature needs:

1. `python -m scripts.concepts.label_contrastive_mesh --model mitra --feat <N> reset`
2. `init` → dispatch 5 round-1 opus agents in parallel (per existing pattern)
3. Record labels → `show-judge` → dispatch judge (opus) → `record-judge`
4. If judge returns `continue`: `show-round --round k+1` → dispatch → record → judge → repeat until `done` or r5
5. `save-label`
6. `show-validator` → dispatch validator (opus) → `record-validator`
7. Snapshot: `cp f{N}_label.json f{N}_label_A.json` and same for state

Expected: per previous two features, ~4–5 rounds, done at r4, ~25 agent calls per feature.

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

## Commands to resume

```bash
# For each remaining feature N in {36, 86, 92}:
FEAT=N
python -m scripts.concepts.label_contrastive_mesh --model mitra --feat $FEAT reset
PROMPT_ORDER=A python -m scripts.concepts.label_contrastive_mesh --model mitra --feat $FEAT init > /tmp/f${FEAT}_r1.txt

# dispatch 5 opus agents in parallel (one per dataset); collect labels; record; judge; iterate.

PROMPT_ORDER=A python -m scripts.concepts.label_contrastive_mesh --model mitra --feat $FEAT save-label
PROMPT_ORDER=A python -m scripts.concepts.label_contrastive_mesh --model mitra --feat $FEAT show-validator > /tmp/f${FEAT}_val.txt
# dispatch validator agent, save response, then:
PROMPT_ORDER=A python -m scripts.concepts.label_contrastive_mesh --model mitra --feat $FEAT record-validator --response-file /tmp/f${FEAT}_val_resp.txt

# snapshot
cp output/contrastive_examples/mitra/f${FEAT}_label.json output/contrastive_examples/mitra/f${FEAT}_label_A.json
cp output/contrastive_examples/mitra/f${FEAT}_mesh_state.json output/contrastive_examples/mitra/f${FEAT}_mesh_state_A.json
```
