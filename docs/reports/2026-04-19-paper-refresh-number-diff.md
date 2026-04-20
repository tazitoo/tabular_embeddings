# Paper refresh — number diff (2026-04-19 / 2026-04-20)

Numbers quoted in `sections/4_results.tex` that moved due to the
refreshed ablation/transfer sweeps. Author to integrate into prose at
discretion.

## Methodology note: min_gap row-skip asymmetry

The trained sweep `output/ablation_sweep_tols/` introduces a
`min_gap=0.01` row-skip: on rows where the two models agree within 1%,
the sweep sets `optimal_k=0, gap_closed=1.0` without ablating. The old
random baseline (`output/ablation_sweep_random/`, Apr 3, pre-feature)
did not have this shortcut. Audit showed 26.82% of strong-wins rows
are auto-credited in trained — enough to meaningfully inflate the
trained-minus-random delta (worst-case 16 pp).

Resolution: re-ran the random baseline with the current
`ablation_sweep.py` defaults and random SAE + random MNN matching +
random importance, into `output/ablation_sweep_random_tols/`. Both
sweeps now auto-credit near-identical row sets (26.82% trained vs
26.98% random, 0.16 pp difference from minor `strong_wins` coverage
variance), and the asymmetry is resolved. The refreshed
`\Delta` numbers below use the matched-methodology random baseline.

## Section 4 — Table 2 (ablation summary)

`tables/section4_summary.tex` is regenerated and dual-written. The
prose numbers in `sections/4_results.tex:109` need to change from:

> "Table~\ref{tab:ablation_summary} reports the mean gap closed across
> all 15 pairs and **689** comparisons: unique SAE concepts explain on
> average **89%** of the strong model's per-row advantage. The pairs
> with highest gap closed involve TabICL-v2 and TabPFN as the strong
> model (**0.97** each), while Mitra and CARTE share the lowest gap
> closed when they are strong (**0.68** each, though CARTE is strong
> on only **16** comparisons vs. Mitra's **88**). TabPFN distinguishes
> itself by the largest net learned signal over the random baseline
> (**Δ = 0.68**), meaning its unmatched concepts are both frequently
> used in ablation and substantially more effective than random
> directions at closing the gap."

to:

> "… 15 pairs and **629** comparisons: unique SAE concepts explain on
> average **94%** of the strong model's per-row advantage. The pairs
> with highest gap closed involve TabICL-v2 (**0.99**) and TabPFN
> (**0.98**) as the strong model, while Mitra has the lowest gap
> closed when strong (**0.74**; CARTE is a close second at **0.84**
> but is strong on only **11** comparisons vs. Mitra's **82**).
> TabPFN distinguishes itself by the largest net learned signal over
> the random baseline (**Δ = 0.51**), meaning its unmatched concepts
> are both frequently used in ablation and substantially more
> effective than random directions at closing the gap."

(Note: "Mitra and CARTE share the lowest" is no longer true; Mitra is
alone at the bottom. CARTE moved up substantially — from 0.68 to 0.84.)

### Per-model movement

| Model | Old N | New N | Old gc | New gc | Old Δ | New Δ | Old K | New K |
|-------|-------|-------|--------|--------|-------|-------|-------|-------|
| TabICL-v2 | 188 | 172 | 0.97 | **0.99** | 0.33 | **0.26** | 4.4 | **1.9** |
| TabPFN    | 148 | 133 | 0.97 | **0.98** | 0.68 | **0.51** | 8.5 | **2.8** |
| TabDPT    | 132 | 120 | 0.83 | **0.95** | 0.41 | **0.42** | 2.6 | **3.1** |
| TabICL    | 117 | 111 | 0.91 | **0.97** | 0.41 | **0.33** | 4.3 | **1.6** |
| Mitra     |  88 |  82 | 0.68 | **0.74** | 0.39 | **0.28** | 3.4 | **2.4** |
| CARTE     |  16 |  11 | 0.68 | **0.84** | 0.44 | **0.42** | 10.2 | **3.7** |
| Overall   | 689 | 629 | 0.89 | **0.94** | 0.44 | **0.36** | 4.9 | **2.4** |

Direction of shifts, all expected:
- `gc` up: `gc_tolerance=0.99` early-stop shortens the greedy search,
  and `min_gap=0.01` auto-credits agreed rows. Both biases push
  reported gap-closed higher.
- `K` down: tight `gc_tolerance` means fewer features per row to reach
  threshold. The per-model K numbers in the paper prose don't appear
  verbatim; the table contains them.
- `Δ` down: both trained and random get the same inflation, so the
  trained excess shrinks.
- `N` down by ~60: `min_gap` row-skip drops rows that previously had
  small but nonzero gaps from the "strong wins" count on some
  datasets.

## Section 4 — Transfer numbers (line 184-195)

Current paper prose:
> "Across 660 (pair, dataset) records, transfer achieves a mean gap
> closed of 0.89 (median 0.98). 72% of records close >90% of the gap.
> Within the transformer cluster, transfer is near-perfect:
> Mitra↔TabICL achieves 0.998, TabICL↔TabPFN achieves 0.995, and
> TabDPT↔TabPFN achieves 0.986. CARTE-involved pairs achieve the
> lowest transfer (0.70–0.80), consistent with CARTE's geometric
> distance from the transformer cluster."

These numbers come from `output/transfer_global_mnnp90_trained_tols/`
which is one of the three refreshed dirs. The table in the paper does
not include a tabular summary; the numbers are inline. **No aggregate
regen script exists yet** — the values in prose are presumably from a
prior hand-run that may predate the 2026-04-19 data refresh.

Recommendation: write a small aggregator that reads
`transfer_global_mnnp90_trained_tols/` and produces the per-pair mean
gc + overall mean + median + fraction-above-90% + acceptance rates
(0.6%, 15.1%, 24.9% rejection claims on line 188). Author should
decide whether to commission that now or trust the existing numbers.
Same holds for:

- `r = 0.96` (acceptance vs gap closed at pair level)
- `r = 0.15` (map quality R² vs gap closed)
- `r = 0.03` (concept importance vs transfer acceptance)
- `49.6%` (transfer > ablation fraction)

These all need the aggregated transfer output to recompute; flagging
as open work.

## Section 4 — Figures

Regenerated against refreshed data:

- `figures/4_results/intervention_distributions.pdf` — now sources
  from `ablation_sweep_tols/`, `transfer_global_mnnp90_trained_tols/`,
  `ablation_sweep_random_tols/`, `transfer_random/`. Numbers in panels
  (gap closed distribution, K distribution, acceptance) reflect the
  matched-methodology comparison.
- `figures/4_results/intervention_example_3panel.pdf` — reads the
  refreshed `_tols` dirs directly.
- `figures/4_results/row_intervention_figure.pdf` — regenerated from
  `ablation_figure_data/carte_vs_mitra/credit-g.npz` (Apr 16 22:00,
  already refreshed). Walks: ablation removes f_92→f_6→f_11 closing
  gc=0.90; transfer injects f_92→f_86→f_36 closing gc=0.998.
- `figures/4_results/importance_decay_grid.pdf` — regenerated as PDF
  (was PNG). Data dependencies (perrow_importance, ablation_sweep)
  were already refreshed.
- `figures/4_results/geometric_vs_concept.pdf` — sanity regen from
  unchanged `geometric_sweep_tabarena_7model.csv`; numbers unchanged
  (per-band Pearson r and Jaccard).

## Appendix A — layerwise CKA

Sanity regen of `figures/A_appendix/layerwise_cka_appendix_*.pdf`
produced all 8 models. Note:

- `plot_model_evidence()` in `scripts/paper/appendix_a/fig_per_model_layers.py`
  had a pre-existing crash when TabPFN's classifier (24L) and
  regressor (18L) profiles coexist in batch data. Fixed by filtering
  profiles to match the first dataset's n_layers.
- `layerwise_cka_appendix_mitra_regressor.pdf` was **not regenerated**
  — `output/layerwise_depth_analysis_mitra_regressor.json` doesn't
  exist, so the Feb 19 PDF was copied forward. Stale.

## Labeling-dependent artefacts (BLOCKED)

The following are blocked on task #28 (outside this repo) landing all
five `f{6,11,36,86,92}_label_{A,Apairing}_v10.json` in
`output/contrastive_examples/mitra/`. As of session end, only f_92 has
landed.

- `scripts/tables/concept_labels_table.tex` → paper `tables/concept_labels_table.tex`
- Per-feature validator accuracy table (new or update existing)
- Pairing delta range ("+0.08 to +0.24") in prose

## Cleanup notes

- Paper repo files deleted (redundant PNGs, A_appendix has canonical
  PDFs): `figures/6_appendix/*.png` (whole dir), `figures/4_results/domain_reconstruction.png`, `figures/4_results/feature_selectivity.png`.
- firelord4 worker had stashed sec4 artefacts at
  `/home/brian/src/tabular_embeddings/.stash_2026-04-19/` to unblock
  `git pull`. Review before deleting.
- `scripts/tables/table4/` renamed to `scripts/tables/ablation_summary/`
  (paper calls it Table 2; old name was misleading). Generator CLI is
  `python -m scripts.tables.ablation_summary.ablation_summary`.

## Open items for next session

1. Write transfer aggregator to recompute r-values, acceptance rates,
   mean gc, etc. against `transfer_global_mnnp90_trained_tols/` +
   `transfer_random/`, and diff against paper prose.
2. Generate or source `layerwise_depth_analysis_mitra_regressor.json`
   so the appendix figure can refresh.
3. Pick up concept_labels_table + validator numbers once the v10
   label snapshots land.
4. Author pass on prose integration for all flagged numbers.
