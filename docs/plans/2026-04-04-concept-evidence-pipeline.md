# Concept → Dataset → Visual Evidence Pipeline

## Goal
For each model pair, produce a complete causal chain: identify the concepts
that differentiate the models, link them to specific datasets, and generate
visual evidence showing the causal effect of ablation (and transfer).

This is the core evidence for the paper's claims. The pipeline produces both
the main-text highlights and the exhaustive appendix material.

## Stage 1: Concept Ranking (`scripts/analysis/rank_pair_concepts.py`)

**Input**: `output/ablation_sweep/{pair}/*.npz`  
**Output**: `output/concept_evidence/{pair}/concept_ranking.json`

For each of the 15 model pairs:
1. Load all ablation results (51 datasets)
2. For each dataset, extract `selected_features` per row
3. Aggregate: count how many rows each feature was selected across all datasets
4. Filter to unmatched features only
5. Output ranked list: `(feature_idx, n_rows_selected, n_datasets_selected, mean_importance)`

Also compute:
- Which Matryoshka band each feature belongs to
- Whether the feature is in a concept group and its label
- Top 3 datasets where it fires most (from concept_analysis_round10.json)

## Stage 2: Concept Categorization (`scripts/analysis/categorize_concepts.py`)

**Input**: concept_ranking.json + concept labels  
**Output**: `output/concept_evidence/{pair}/concept_categories.json`

Group the top-N concepts into interpretable categories by clustering their labels:
- Magnitude patterns (high/low z-scores, log-magnitude)
- Distribution shape (variance, range, skew)
- Outlier detection (centroid distance, isolation, extreme values)
- Categorical patterns (entropy, rarity, modal dominance)  
- Sparsity patterns (zero fraction, binary features)
- Boundary/density (decision boundary proximity, local density)

Method: Use the existing concept labels + a simple keyword/embedding clustering.
Could also dispatch to an LLM to group labels into categories.

For each category:
- Count of concepts in category
- Total row selections (how much of the pair's advantage is in this category)
- Top datasets where this category of concepts fires

## Stage 3: Dataset Association (`scripts/analysis/concept_dataset_matrix.py`)

**Input**: concept_ranking.json + perrow_importance NPZs  
**Output**: `output/concept_evidence/{pair}/concept_dataset_matrix.npz`

Build a (n_concepts × n_datasets) matrix of concept importance:
- Cell = mean importance of concept f on dataset d (across strong-win rows)
- Rows = top-N concepts from Stage 1
- Cols = datasets where the strong model wins

This matrix tells us: "concept f92 is most important on hiva_agnostic and MIC
but irrelevant on credit-g." Enables dataset-specific storytelling.

Also compute:
- For each dataset: which 3 concepts explain most of its gap
- For each concept: which 3 datasets it's most active on

## Stage 4: Single-Concept Scatter Plots (`scripts/figures/plot_concept_scatter.py`)

**Input**: ablation NPZ + concept ranking  
**Output**: `output/figures/concept_evidence/{pair}/{dataset}_{concept}.pdf`

For the top-K concepts × their top datasets, generate a focused scatter plot:
- Gray dots: all test rows (x=strong P(correct), y=weak P(correct))
- Blue dots: rows where this specific concept fires
- Black dots: predictions after ablating ONLY this concept on those rows
- Annotation: concept label, feature index, gc for this concept alone

This is the "smoking gun" figure — a single concept's causal effect on a
single dataset.

Parameters:
- K = 5 concepts per pair (top by row count)
- 3 datasets per concept (top by importance)
- Total: ~225 plots (15 pairs × 5 concepts × 3 datasets)

## Stage 5: Transfer Overlay (`scripts/figures/plot_concept_transfer_scatter.py`)

**Input**: transfer NPZ + ablation NPZ + concept ranking  
**Output**: `output/figures/concept_evidence/{pair}/{dataset}_transfer_{concept}.pdf`

For concepts that appear in both ablation and transfer results:
- Same scatter as Stage 4
- Add green dots: weak model predictions after receiving this concept via transfer
- Shows whether the transferred concept moves the weak model in the right direction

Split concepts into:
- **Transferable**: accepted by greedy search in transfer, gc > 0
- **Non-transferable**: tried but rejected, or gc ≈ 0

## Stage 6: Pair Summary (`scripts/figures/plot_pair_summary.py`)

**Input**: All outputs from Stages 1-5  
**Output**: `output/figures/concept_evidence/{pair}/summary.pdf`

One-page summary per pair:
- Left: concept category bar chart (which types of concepts differentiate these models)
- Center: concept × dataset heatmap (where concepts matter most)
- Right: transferability pie chart (what fraction successfully transfers)
- Bottom: text listing top 3 concepts with their labels

For the main paper: pick the 2-3 most illustrative pairs and include their
summaries. All 15 go in the appendix.

## Stage 7: Cross-Pair Synthesis (`scripts/analysis/cross_pair_synthesis.py`)

**Input**: All pair rankings from Stage 1  
**Output**: `output/concept_evidence/cross_pair_synthesis.json`

Aggregate analysis:
- **Universal differentiators**: concepts that appear in top-10 for many pairs
  (these are the most important concepts overall)
- **Pair-specific concepts**: concepts that only matter for one pair
  (architecture-specific advantages)
- **Transfer success by concept category**: do certain concept types transfer
  better than others? (e.g., magnitude concepts transfer but boundary concepts don't)
- **Model fingerprint**: for each model, what concept categories define its
  unique contribution?

## Execution Order

```
Stage 1 (ranking)       → can run now, all data exists
Stage 2 (categorize)    → depends on Stage 1
Stage 3 (dataset matrix) → depends on Stage 1
Stage 4 (scatter plots)  → depends on Stage 1, can run in parallel with 2-3
Stage 5 (transfer)       → depends on Stage 4 + transfer results (waiting)
Stage 6 (pair summary)   → depends on all above
Stage 7 (cross-pair)     → depends on Stage 1 from all pairs
```

Stages 1-4 can be built and run now. Stage 5 waits for transfer completion.
Stages 6-7 are synthesis that comes last.

## Paper Integration

**Main text (Section 4)**:
- 1-2 pair summaries (Stage 6) for the most illustrative pairs
- 1-2 single-concept scatter plots (Stage 4) as highlighted examples
- Cross-pair synthesis table (Stage 7) showing concept categories

**Appendix**:
- Full concept rankings for all 15 pairs (Stage 1)
- All single-concept scatter plots (Stage 4)
- All pair summaries (Stage 6)
- Transfer overlay plots (Stage 5)

## TabICL v1 vs v2 Case Study

Special attention for this pair since architectural changes are documented:
- v2 gains 1,098 unmatched features vs v1
- v2 wins 25/38 datasets with gc=0.991
- Top new concepts: outlier detection, rare category handling, sparsity patterns
- Case study: which of these transfer back to v1? (data vs architecture gap)
