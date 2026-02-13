# Paper Plan: Diagnosing Tabular Foundation Models via Sparse Concept Probing

## Venue

**AutoML 2026** — Ljubljana, Slovenia, Sep 28–Oct 1
- Deadline: **April 30, 2026** (11 weeks from Feb 8)
- Format: 9 pages + references, LaTeX template
- Requires: reproducibility review, public code
- Tracks: Methods (primary) or ABCD (benchmark angle)

## Narrative Arc

Tabular foundation models (TFMs) are proliferating, but we lack tools to
understand *what* they learn and *where* they fail. We train Matryoshka-Archetypal
SAEs on intermediate representations of 5-6 TFMs and use the learned concepts as
a diagnostic lens:

1. **Geometric alignment** (CKA/Procrustes): TFMs partially converge — a
   transformer ICL cluster exists, but architecture-class boundaries persist.
2. **Concept alignment** (SAE Jaccard): Coarse concepts (S1) are shared across
   the transformer cluster; fine concepts (S5) are model-specific.
3. **Diagnostic gap analysis**: SAE concept activations, correlated with
   per-dataset performance, reveal *where* each model's representation falls
   short — and which concept gaps predict failure.
4. **Synthesis**: Geometric similarity (CKA) predicts concept overlap (r=0.93
   at S1), but concept-level analysis reveals gaps invisible to geometry alone.

The contribution is not just empirical comparison — it's a **diagnostic toolkit**
that can be applied to any new TFM to predict failure modes before benchmarking.

## Key Constraint: Preprocessing Honesty

Our pipeline zero-fills missing values (`nan_to_num(nan=0.0)`) and ordinal-encodes
categoricals (`cat.codes`). Concept gaps related to missingness or categorical
structure are **preprocessing artifacts, not model blind spots**. The paper must
be explicit about this. Diagnosable properties are those that survive preprocessing:
distribution shape, geometric structure, sparsity, feature correlations, scale.

---

## Results Section Burn-Down

### 4.1 Geometric Alignment (CKA / Procrustes)

**Status**: Complete (7 models)

**Deliverables**:
- [x] Table 1: Pairwise CKA (mean ± std) — `scripts/table1/table1.py`
- [x] Table 2: Procrustes disparity d² ∈ [0,1] — `scripts/table2/table2.py`
- [x] Expand to 7 models (all SAEs validated)
- [ ] Narrative: partial convergence (transformer cluster exists, outlier tiers)

**Key findings (36 common datasets, 7 models)**:
- Transformer cluster: Mitra-TabDPT CKA=0.83, Mitra-TabPFN=0.68, TabPFN-TabDPT=0.67, TabPFN-TabICL=0.66, Mitra-TabICL=0.62, TabDPT-TabICL=0.61
- CARTE outlier: CKA=0.39-0.53 vs transformers (GNN architecture)
- HyperFast outlier: CKA=0.13-0.27 vs all (hypernetwork)
- Tabula-8B outlier: CKA=0.24-0.34 vs all (LLM), high variance (std 0.17-0.23)
- Tabula-8B-HyperFast: CKA=0.24 — two outliers are as similar to each other as to the cluster
- Procrustes: Mitra-TabDPT d²=0.30 (closest), Tabula-8B-HyperFast d²=0.47 (two outliers closest after transformer cluster)

---

### 4.2 Cross-Model Concept Alignment

**Status**: Complete (7 models)

**Deliverables**:
- [x] Figure 1: Jaccard heatmaps by Matryoshka scale band — `scripts/figure1/figure1.py`
- [x] All 7 models included (TabPFN, CARTE, TabICL, TabDPT, Mitra, HyperFast, Tabula-8B)
- [ ] Sensitivity analysis on corr_threshold (appendix)

**Key findings (7 models, 36 common datasets)**:
- S1 [0,32): transformer cluster Jaccard 0.35-0.48, CARTE 0.17-0.44, HyperFast 0.24-0.35, Tabula-8B 0.21-0.36
- HyperFast has LOW CKA (0.13-0.27) but moderate S1 Jaccard — coarse concepts shared despite different geometry
- Tabula-8B shows similar pattern: CKA=0.24-0.34 but S1 Jaccard=0.21-0.36
- HyperFast-Tabula-8B S2 Jaccard=0.44 (highest in entire study!) — two outliers share medium-scale concepts
- Tabula-8B S5 near zero (0.01-0.07) — fine-grained concepts are most model-specific for LLM architecture
- S2-S5: rapid decay to 0.02-0.21 (model-specific fine concepts)
- corr_threshold=0.15 gives meaningful cross-model clusters

---

### 4.3 Diagnostic Gap Analysis ← NEW ANGLE

**Status**: Plan drafted, needs implementation

**Goal**: Show that SAE concept activations predict per-dataset performance gaps.

**Approach**:

#### Step 1: Collect per-dataset performance
Get TabArena leaderboard scores (or run inference) for each model on each dataset.
Need: accuracy/AUC per model per dataset.

#### Step 2: Concept activation profiles per dataset
For each model's SAE, compute mean activation vector per dataset (already available
from Figure 1 pipeline). This gives a (n_datasets, n_features) matrix per model.

#### Step 3: Correlate concept gaps with performance
For each model pair (A, B):
- Identify concepts present in A but absent in B (asymmetric Jaccard)
- Compute per-dataset "concept gap score" = activation of A-only concepts
- Correlate concept gap score with performance delta (A minus B)
- If concept gaps predict where A outperforms B → diagnostic value

#### Step 4: Cross-model diagnostic profiles
- Which concepts are universal winners? (present in all high-performing models)
- Which concepts are diagnostic? (presence/absence predicts performance)
- Are there domain-specific diagnostic concepts?

**Deliverables**:
- [ ] Figure: concept gap vs performance delta scatter (per model pair)
- [ ] Table: top diagnostic concepts with interpretable labels
- [ ] Narrative: SAE concepts as a predictive diagnostic tool

**Risk**: Signal may be weak with only 39 datasets (small n for correlation).
Mitigation: focus on rank correlation, bootstrap CIs, and qualitative case studies.

---

### 4.4 Synthesis (CKA × Concepts)

**Status**: Complete (7 models, 21 pairs)

**Deliverables**:
- [x] Figure 7: CKA vs Jaccard scatter — `scripts/figure7/figure7.py`
- [x] All 7 models included (4 architecture categories)
- [ ] 2×2 interpretation framework in text
- [ ] Connect to diagnostic findings from 4.3

**Key findings**:
- CKA predicts S1 concept overlap but with weaker correlation now (21 pairs vs 15)
- Tabula-8B pairs cluster with HyperFast pairs in CKA-Jaccard space (low CKA, moderate S1)
- HyperFast-Tabula-8B is the most interesting outlier: low CKA (0.24) but high S2 Jaccard (0.44)
- Fine concepts (S5) fully decouple from geometry — Tabula-8B S5 near zero despite moderate CKA
- Four architecture tiers: transformer cluster → CARTE → {HyperFast, Tabula-8B} → (gap)

---

## Models Status

| Model | Embeddings | Layer | SAE | In Figures | Notes |
|-------|-----------|-------|-----|------------|-------|
| TabPFN | 51 ds | L16/24 (67%) | Validated | Yes | Classifier |
| CARTE | 39 ds | L1/3 (33%) | Validated | Yes | GNN, classification+regression |
| TabICL | 44 ds | L10/14 (71%) | Validated | Yes | Column-then-row TFM |
| TabDPT | 51 ds | L14/16 (88%) | Validated | Yes | TFM + retrieval |
| Mitra | 48 ds | L12/13 (92%) | Validated | Yes | Score=0.967. L12 required for cross-model CKA (L10→CKA=0.01) |
| HyperFast | 39 ds | L2/3 (67%) | Validated | Yes | Score=0.954, R²=0.941, stability=0.963, 5833/6272 alive |
| Tabula-8B | 51 ds | L21/32 (66%) | Validated | Yes | Score=0.855, R²=0.804, stability=0.893. Llama-3 8B, 4096 dim. CKA=0.24-0.34 vs others |

---

## Figures (9-page budget)

| # | Section | Description | Status | Script |
|---|---------|-------------|--------|--------|
| 1 | 4.2 | **Cross-model concept agreement by scale band.** 2×3 Jaccard heatmaps. | Done (7 models) | `scripts/figure1/figure1.py` |
| 2 | 4.3 | **Concept gap vs performance delta.** Scatter per model pair showing SAE-predicted gaps correlate with actual performance differences. THE DIAGNOSTIC FIGURE. | Not started | — |
| 3 | 4.4 | **CKA vs concept overlap.** Scatter + band-decay panel. | Done (7 models, 21 pairs) | `scripts/figure7/figure7.py` |

Tables:
| # | Section | Description | Status | Script |
|---|---------|-------------|--------|--------|
| 1 | 4.1 | Pairwise CKA (mean ± std) | Done (7 models) | `scripts/table1/table1.py` |
| 2 | 4.1 | Procrustes disparity d² | Done (7 models) | `scripts/table2/table2.py` |
| 3 | 4.3 | Top diagnostic concepts | Not started | — |

Appendix (supplementary):
| # | Description | Status |
|---|-------------|--------|
| A1 | SAE reconstruction quality per model per band | Not started |
| A2 | Domain taxonomy (10 domains × 51 datasets) | Done |
| A3 | corr_threshold sensitivity | Not started |
| A4 | Full concept presence matrix | Prototype exists |

---

## Timeline (11 weeks to Apr 30)

### Phase 1: Foundation (Feb 8–21) — 2 weeks ✓ COMPLETE
- [x] Table 1 (CKA) and Table 2 (Procrustes) — 7 models
- [x] Figure 1 (Jaccard heatmaps) — 7 models
- [x] Figure 7 (CKA vs concept scatter) — 7 models, 21 pairs
- [x] All 7 SAEs trained and validated
- [x] All figures/tables expanded to 7 models

### Phase 2: Diagnostic Angle (Feb 22–Mar 14) — 3 weeks
- [ ] Collect per-dataset performance scores for all models (need to run inference)
- [ ] Implement concept gap → performance correlation pipeline
- [ ] Figure 2: diagnostic scatter (the new key figure)
- [ ] Table 3: top diagnostic concepts
- [ ] Identify 2-3 case study datasets for qualitative analysis

### Phase 3: Writing (Mar 15–Apr 12) — 4 weeks
- [ ] Draft all sections (intro, method, results 4.1-4.4, discussion)
- [ ] Reproducibility checklist (AutoML requirement)
- [ ] Public code repo cleanup
- [ ] Internal review / iteration

### Phase 4: Polish (Apr 13–30) — 2.5 weeks
- [ ] Figure polish, camera-ready quality
- [ ] Supplementary materials
- [ ] Final proofread
- [ ] Submit by Apr 30

---

## Priority Order (what to do next)

1. **Collect TabArena performance data** (run inference — tabarena package is placeholder)
2. **Build diagnostic pipeline** (concept gap × performance correlation)
3. **Start writing** — intro and method sections can begin now
4. **Appendix figures** (SAE quality, corr_threshold sensitivity)
