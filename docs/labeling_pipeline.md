# SAE Feature Labeling Pipeline

Produces a single-sentence shape-level label for one SAE feature per model,
by running n parallel agents over contrastive evidence across n datasets with
a judge-gated iteration loop and a held-out validator.

## Stages

```
  +------------------+    +----------------+    +---------------+    +---------------+
  | contrastive CSVs | -> | round-1 agents | -> | judge (round) | -> | final label   |
  | + validator CSVs |    | + ring rounds  |    | + calibration |    | (judge-       |
  +------------------+    +----------------+    +---------------+    |  synthesized) |
                                                                     +-------+-------+
                                                                             |
                                                              +--------------+-------------+
                                                              | held-out validator         |
                                                              | + accuracy / precision /   |
                                                              |   recall / f1              |
                                                              +----------------------------+
```

## Prerequisites

- SAE checkpoint for `{model}` at `output/sae_sweep_round10/{model}/...` (or whatever `DEFAULT_SAE_ROUND` points at).
- SAE train/test split NPZ at `output/sae_training_round10/{model}_taskaware_sae_test.npz` (used to identify activating rows).
- Baseline predictions cache at `output/baseline_predictions/{model}/{dataset}.npz`
  (built by `scripts/intervention/cache_baseline_predictions.py`).

## Build contrastive and validator CSVs

```bash
# Contrastive CSVs (activating + nearest-non-activating rows, annotated)
python -m scripts.concepts.build_contrastive_examples \
    --model mitra --features 11 --device cpu

# Held-out validator CSVs (rows NOT in the contrastive set)
python -m scripts.concepts.build_contrastive_examples \
    --validator --model mitra --features 11 --n-act 5 --n-con 5
```

Output under `output/contrastive_examples/{model}/`:

| File                                  | Purpose                                          |
|---------------------------------------|--------------------------------------------------|
| `f{feat}_{dataset}.csv`               | Contrastive rows (label/band/activation/target + annotated data + pred columns) |
| `f{feat}_context.json`                | Per-dataset metadata and the list of datasets the feature fires on |
| `f{feat}_validator_{dataset}.csv`     | Held-out rows (no label/band/activation/row_idx; opaque r000..rNNN row_ids; shuffled) |
| `f{feat}_validator_truth.json`        | `{dataset: {row_id: actual_fires_bool}}` for grading |

## Run the labeling loop

All commands take `--model X --feat N`.

### Round 1

```bash
# Prints n per-dataset prompts (one per dataset the feature fires on)
python -m scripts.concepts.label_contrastive_mesh --model mitra --feat 11 init
```

Dispatch n agents in parallel (one per dataset). Each agent reads its prompt
and returns a single 10-25 word shape-only label. Collect the n labels into a
JSON list in dataset order:

```bash
python -m scripts.concepts.label_contrastive_mesh --model mitra --feat 11 \
    record --labels-file /path/to/round1_labels.json
```

### Judge

```bash
python -m scripts.concepts.label_contrastive_mesh --model mitra --feat 11 show-judge
```

The judge prompt embeds, per dataset, `JUDGE_SAMPLE_N_ACT` + `JUDGE_SAMPLE_N_CON`
sampled rows (deterministic per round+dataset) plus a `PRIOR-ROUND OBSERVATIONS`
block from round 2 onward.

Dispatch a single judge agent (opus is the default we've been using). It must
output a fenced `json` block with per-dataset verdict and `{claim, certainty}`-
tagged claims. When `overall_verdict` is `done` or we're in the final round,
it must also emit `final_label`.

```bash
python -m scripts.concepts.label_contrastive_mesh --model mitra --feat 11 \
    record-judge --response-file /path/to/judge_output.txt
```

### Ring rounds 2..MAX_ROUNDS

If the judge returns `continue`:

```bash
python -m scripts.concepts.label_contrastive_mesh --model mitra --feat 11 \
    show-round --round 2
```

Each round-k agent (for k >= 2) sees:

- Its own full contrastive CSV.
- A peer's ~3+3 sampled rows (selected randomly per round, excluded from the
  judge's sample pool).
- Its own per-dataset judge feedback from round k-1, with certainty tags.
- The peer's per-dataset judge feedback.

Dispatch, record, judge, repeat. Stop when judge says `done` or when `rounds
== MAX_ROUNDS`.

### Save label

```bash
python -m scripts.concepts.label_contrastive_mesh --model mitra --feat 11 save-label
```

Writes `output/contrastive_examples/{model}/f{feat}_label.json`:

```json
{
  "model": "...",
  "feat_idx": 11,
  "label": "<judge's final_label>",
  "label_source": "judge",   // or "synthesis" / "most_central"
  "datasets_used": [...],
  "rounds_completed": 4,
  "final_round": 4,
  "judge_verdicts": [...],
  "similarity_history": [...],
  "labels_per_round": [...],
  "validator_results": null   // filled in after running the validator
}
```

### Validator

```bash
python -m scripts.concepts.label_contrastive_mesh --model mitra --feat 11 show-validator
```

Validator agent sees the final label + n datasets of shuffled held-out rows
(no `label`/`band`/`activation`/`row_idx` leaks). For each `row_id`, it outputs
`prediction: "fires" | "does_not_fire"`. Prompt carries the anti-bias rule
("Judge each row independently. Do not assume any particular number of rows
should fire in a dataset").

```bash
python -m scripts.concepts.label_contrastive_mesh --model mitra --feat 11 \
    record-validator --response-file /path/to/validator_output.txt
```

Writes per-dataset accuracy/precision/recall plus overall micro accuracy,
macro accuracy, and f1 into the state file and into `save-label` output.

## Hyperparameters

Configured in `scripts/concepts/label_contrastive_mesh.py`:

| HP | Value | What it controls |
|----|-------|------------------|
| `MAX_ROUNDS` | 5 | Cap on judge-gated iteration. Judge may stop earlier with `done`. |
| `JUDGE_SAMPLE_N_ACT` | 2 | Activating rows per dataset the judge sees each round |
| `JUDGE_SAMPLE_N_CON` | 2 | Contrast rows per dataset the judge sees each round |
| `PEER_SAMPLE_N_ACT` | 3 | Activating rows a ring peer shares (rounds 2+) |
| `PEER_SAMPLE_N_CON` | 3 | Contrast rows a ring peer shares (rounds 2+) |
| `CONVERGENCE_THRESHOLD` | 0.80 | Cosine threshold — diagnostic only, judge decides stopping |
| `LABEL_WORDS_MIN/MAX` | 10/25 | Per-agent label length |
| `JUDGE_SAMPLE_SEED` | 13 | Deterministic-sampling salt |

Validator defaults (CLI flags on `build_contrastive_examples.py --validator`):

| HP | Default |
|----|---------|
| `--n-act` | 5 (held-out activating rows per dataset) |
| `--n-con` | 5 (held-out non-activating rows per dataset) |

## Baseline result (2026-04-17)

Pipeline run on **mitra / f_11** (features 5 datasets: NATICUSdroid,
hazelnut-spread-contaminant-detection, students_dropout_and_academic_success,
Marketing_Campaign, splice):

| Round | Cosine | Per-dataset verdicts | Overall |
|-------|--------|----------------------|---------|
| 1 | 0.879 | 5 revise | continue |
| 2 | 0.874 | 3 accept, 2 revise | continue |
| 3 | 0.884 | 5 accept (all stable cross-round claims) | continue |
| 4 | 0.875 | 4 accept, 1 revise | **done** |

**Final label** (judge-synthesized at round 4):
> Activating rows concentrate a localized column subset at its modal or rare co-firing signature with confident-correct predictions, while contrasts break the localization or spread into extreme tails.

**Validator (5 datasets × 5 activating + 5 contrast held-out rows = 50 rows):**

```
HEADLINE  accuracy(micro)=0.520  accuracy(macro)=0.520  f1=0.556
          26/50 correct  precision=0.517  recall=0.600

  [hazelnut]            acc=0.70
  [NATICUSdroid]        acc=0.50
  [splice]              acc=0.50
  [students_dropout]    acc=0.50
  [Marketing_Campaign]  acc=0.40
```

**Interpretation.** The pipeline converges on an internally-consistent
shape-level label backed by `stable cross-round pattern` claims per dataset,
but the resulting meta-label is too abstract to classify held-out rows above
chance. The validator catches this gap: without it, the judge's `done` verdict
would silently anchor a 52%-predictive label as the answer.

## HP tuning baseline

Use the validator accuracy triple `(micro, macro, f1) = (0.520, 0.520, 0.556)`
as the comparison point when sweeping knobs. Knobs worth trying next:

| Knob | Current | Candidates |
|------|---------|------------|
| `MAX_ROUNDS` | 5 | 8, 10 |
| `PEER_SAMPLE_N_ACT/CON` | 3/3 | 5/5, 2/2 |
| `JUDGE_SAMPLE_N_ACT/CON` | 2/2 | 3/3, 5/5 |
| Round-1 prompt | shape-only + contrast discipline | add "label will be validated on held-out rows" priming |
| Judge accept criterion | `stable cross-round` backing | also require falsifiable numeric bounds |
| Agent model mix | all opus | haiku round-1 + opus ring + opus judge |

A single-knob sweep is one pipeline re-run (5 parallel round-1 agents + up to
4 rounds × (5 agents + 1 judge) + 1 validator) per setting.
