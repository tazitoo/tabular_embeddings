# Paper Figure & Table Scripts

Scripts that generate figures and tables for the paper, organized by section.
Script names match LaTeX `\label{}` references for traceability.

## Directory Structure

```
scripts/paper/
  sec1/                            # Section 1: Introduction
    fig_dictionary_comparison.py   # \label{fig:dictionary_comparison}
  sec4/                            # Section 4: Results
    fig_geometric_vs_concept.py    # \label{fig:geometric_vs_concept}
  appendix_a/                      # Appendix A: Representation Extraction
    fig_per_model_layers.py        # \label{fig:{model}_layers}
  appendix_b/                      # Appendix B: SAE Training
    fig_stability_vs_loss.py       # \label{fig:stability_vs_loss}  (TODO)
  appendix_c/                      # Appendix C: Concept Analysis
    fig_concept_hierarchy.py       # \label{fig:concept_hierarchy}
    fig_concept_universality.py    # \label{fig:concept_universality}
    fig_pairwise_concept_overlap.py  # \label{fig:pairwise_concept_overlap}
  appendix_d/                      # Appendix D: Domain Analysis
    fig_domain_reconstruction.py   # \label{fig:domain_reconstruction_detail}
    fig_feature_selectivity.py     # (TODO: split from domain_reconstruction)
```

## Conventions

- **Naming**: `fig_<label>.py` for figures, `tab_<label>.py` for tables
- **Output**: All scripts write to `output/paper_figures/<section>/`
- **Data**: Heavy computation saves intermediate JSON to `output/paper_data/<section>/`
  so plotting can be re-run without recomputation
- **PROJECT_ROOT**: All scripts resolve to repo root via `Path(__file__).parent` chain

## Running

```bash
# From repo root
python scripts/paper/sec1/fig_dictionary_comparison.py
python scripts/paper/sec4/fig_geometric_vs_concept.py
```

## Copying to paper repo

After generating, copy PDFs to the paper repo:
```bash
cp output/paper_figures/sec1/*.pdf /path/to/paper/figures/1_intro/
cp output/paper_figures/appendix_a/*.pdf /path/to/paper/figures/A_appendix/
```
