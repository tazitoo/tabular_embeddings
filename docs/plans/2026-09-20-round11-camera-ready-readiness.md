# Round-11 readiness for the NeurIPS camera-ready (checked 2026-09-20)

Every figure/table in the NeurIPS paper (`~/src/tabular_embedding_paper`, main), its
generator, what it reads, and whether the round-11 equivalent is on the Mac.
Round 11 retrained ONLY the TabDPT SAE (corpus fix); the other SAEs are round-10
checkpoints. Everything downstream of matching (matching files, importances,
interventions, decomposition, patching) was rerun for all pairs.

## A. Data on disk, generator only needs re-pointing to `output/round11/...`

| paper item | generator | round-10 input | round-11 equivalent (on disk) |
|---|---|---|---|
| Table 1 ablation summary | `scripts/tables/ablation_summary/ablation_summary.py` | ablation_sweep_tols, ablation_sweep_random_tols, perrow_importance | forward_ablation (698), forward_ablation_random (698), perrow_importance |
| Table 2 transfer summary | `scripts/tables/transfer_summary/transfer_summary.py` | transfer_global_mnnp90_trained_tols, transfer_global_mnnp90_random | forward_deltas (688), forward_deltas_random |
| Fig 1(b) gc distributions | `scripts/figures/plot_intervention_distributions.py` | the four dirs above | same |
| Fig gap_closed_heatmap | `scripts/figures/4_results/plot_gap_closed_heatmap.py` | ablation_sweep_tols, transfer_..._tols | forward_ablation, forward_deltas |
| Fig 1(a) row scatter | `scripts/paper/sec4/row_intervention_scatter.py` | ablation_figure_data, transfer_figure_data (one credit-g row) | forward_ablation + forward_deltas store step_preds/step_features for every row; the special dirs are no longer needed |
| Fig 3-panel example, App E ablation grids (51) | `plot_ablation_example_3panel.py`, `plot_ablation_grid.py` | output/ablation_sweep (pre-tols!) | forward_ablation |
| Fig importance_decay_grid | `scripts/figures/4_results/plot_importance_decay.py` | perrow_importance, sae_feature_matching_mnn_floor_p90.json | round11/perrow_importance, round11/sae_feature_matching_mnn_floor_p90.json |
| App F trained_vs_random_ablation | `scripts/figures/plot_trained_vs_random_ablation.py` | ablation_sweep_tols(_random) | forward_ablation(_random) |
| Table 3 off-manifold split (energy, median K) | `scripts/rebuttal/aggregate_functional_clean.py` | output/rebuttal/functional_decomposition | round11/functional_decomposition (15) + _random (15) |
| Table 3 principal-angle column | `scripts/rebuttal/subspace_analysis.py` -> `subspace_by_recipient.py` | rebuttal/forward_deltas -> subspace_summary.json | round11/forward_deltas has `deployed_delta`; CPU-only, NOT yet run for round 11 |
| App patch tables (patch_explanations, patch_pipeline) | `scripts/tables/patch_explanations.py`, `patch_pipeline_tables.py` | rebuttal/v31q/*.json, rebuttal/patching_burndown.csv, off_manifold_concept_dump_all.csv | round11/patch_runs/r11v31 (245/245 clean, +52 Rule-A extras), round11/patching_burndown.csv, round11/off_manifold_concept_dump_trained.csv |
| Procrustes table (App A) | `scripts/tables/table2/table2.py` | sae_training_round10/*_sae_test.npz | sae_training_round11/*_sae_test.npz (all 8 models) |

Caveats inside A: the 5 hiva_agnostic cells with Mitra as WEAK in forward_ablation are not
baseline-consistent (Mitra fit-time OOM support-reduction) -> exclude or refit before Table 1.
Round-10 -> round-11 headline drift: ablation 0.934 -> 0.955, transfer 0.898 -> 0.906; prose
numbers must be re-derived along with the tables.

## B. Label-derived items: NO round-11 artifact on disk

These read round-10 matching + LLM labels from `output/concept_labeling/` (April) and have no
round-11 counterpart. Group labels for round 11 exist (`output/cross_model_concept_labels_round11.json`)
but the derived files do not. Each needs its producer rerun against round-11 matching, and the
gap-index / individual-concept tables need a labeling pass.

| paper item | producer to rerun | missing round-11 input |
|---|---|---|
| App gap_index_classification / _regression tables | `scripts/concepts/run_gap_index_manifest.py`, `label_individual_concepts.py` (LLM) | weak_gap_index.json, gap_index_classification_frozen_baseline/ |
| App unmatched_firing_table | `scripts/concepts/unmatched_firing_frequency.py` | unmatched_firing_frequency.json |
| App G band_cofiring figure | `scripts/concepts/band_cofiring.py` | band_cofiring.json |
| Fig ablation_concentration | `scripts/concepts/top_feature_ablation_ranking.py` then `persona_drafts/boss/fig_ablation_concentration.py` | top_features_per_model.json |
| App C concept_universality, cross_model_concept_hierarchy | `scripts/paper/appendix_c/fig_concept_*.py` | concept_hierarchy_full.json (round-11 build) |
| Table concept_pair_counts.tex, mitra_feature_labels.tex | static (May 3-4), no script | rebuild by hand from round-11 matching + labels |
| Patch-table label/explanation text | contrastive labeling (round11/contrastive_examples has mitra 497, tabicl_v2 117, tabicl 50, tabpfn 28; round11/concept_labeling has only the quality cache) | per-concept labels for the 245 patched concepts |

## C. Geometry items: unaffected or TabDPT-only

CKA heatmap + regression CKA table (`table1.py` on round-9 embeddings), App A layerwise CKA
figures, App B stability_vs_l2 (round-10 sweep), App F Matryoshka variance (inline; SAE
checkpoints). Only TabDPT changed in round 11 (18 re-extracted datasets in the SAE corpus). If the
camera-ready is to be round-11-consistent, the TabDPT row/column of the CKA figure and the TabDPT
Matryoshka curve should be recomputed from the round-11 TabDPT extraction/SAE; the rest is unchanged.
Fig 1 pipeline figure is static.

## Order of work
1. Run `subspace_analysis.py` on round-11 forward_deltas (CPU, minutes).
2. Re-point the group-A generators (a `--round 11` or RESULTS_DIR switch), regenerate, diff vs
   round-10 tables, re-derive prose numbers.
3. Decide group B: regenerate (needs an LLM labeling pass for gap-index + patch explanations) or
   keep round-10 with an explicit note. Mixed rounds in one paper must be stated.
4. Group C: TabDPT-only refresh, or state that geometry figures are round-9/10.
