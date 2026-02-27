# Critique: Matryoshka + Archetypal SAE for Tabular FM Comparison

**Date:** 2026-02-08
**Based on:** Last 30 days of SAE research across arxiv, DeepMind, Goodfire, LessWrong, OpenReview
**Revised after:** Reading the paper draft (main-3.pdf, Feb 8 2026)

---

## What's Working Well

### 1. The use case is exactly where SAEs are defensible right now

The sharpest framing from 2025-2026 research ([arxiv 2506.23845](https://arxiv.org/html/2506.23845v1)) argues SAEs are powerful for *discovering unknown concepts* but weak for *acting on known concepts*. This paper is doing pure discovery — "what features do these models learn?" — not classification or steering. This puts it on the right side of the DeepMind critique, which found negative results specifically when using SAEs for downstream tasks like detecting harmful intent. **The paper should cite this distinction explicitly** (see Recommendations).

### 2. Archetypal SAE solves the stability problem convincingly

Table 9 in the draft shows Matryoshka-Archetypal achieves 0.91 dictionary cosine stability vs. 0.32 for Matryoshka alone — a 2.8x improvement. This directly addresses the biggest criticism in the field (different seeds yield different decompositions). The convex hull constraint anchors features to data centroids, making the learned dictionary far more reproducible. This is a genuine methodological contribution.

### 3. The ICC analysis is the paper's strongest empirical result

Table 14 showing ICC decay from L0 (mean 0.551) to L4 (mean 0.185) is real statistical evidence that the matryoshka hierarchy maps to abstraction level — L0 features are dataset-level descriptors, L4 features are row-level detectors. This partially addresses the concern about "imposed vs discovered" hierarchy (see below for remaining issues). The between-dataset vs. within-dataset variance decomposition is a clean, interpretable analysis.

### 4. The concept examples are compelling and data-centric

Tables 11-13 (outlier detection, row complexity, numeric stability concepts) show that TabPFN encodes meta-properties of rows that generalize across datasets. Feature 30 firing on extreme outliers across Bioresponse, MIC, and hiva_agnostic is exactly the kind of interpretable, cross-domain concept that makes SAE analysis worthwhile for tabular data.

### 5. The 2x2 framework (Section 4.5) is analytically sharp

High/low CKA x high/low concept overlap gives four interpretable quadrants. This goes beyond "do models agree?" to "how do they agree?" — distinguishing geometric alignment from conceptual alignment. This is a novel contribution to the representation comparison literature.

### 6. Layer selection is principled

Section 8 and Table 7 show a systematic layerwise CKA critical depth analysis across 8-15 TabArena datasets per model. This addresses what could have been a major weakness — the extraction points are empirically justified, not arbitrary. The finding that there's no universal critical depth (35%-87% range) is itself an interesting negative result about TFM maturity.

### 7. The paper already has activation-based similarity (Eq. 6)

The methodology defines activation-based correlation and domain profile similarity alongside Jaccard. This is methodologically sound — the paper has the right tools available, even if the current Figure 1 uses binary Jaccard.

---

## Concerns (Revised After Reading Draft)

### Concern 1: Matryoshka hierarchy — partially validated, but baseline still needed

**Previous concern:** The matryoshka loss forces features into a hierarchy; S1 universality could be an artifact.

**Updated assessment:** The ICC analysis (Table 14) provides real evidence that the hierarchy is meaningful — L0 features genuinely capture dataset-level properties while L4 captures row-level ones. This is not trivially imposed by the loss.

**Remaining issue:** The ICC analysis shows the hierarchy *describes* something real, but doesn't prove the *matryoshka loss* is necessary to find it. A standard TopK SAE might also learn features where the top-k-by-frequency have high ICC. The ablation baseline is still needed to establish that matryoshka training is doing something that wouldn't emerge naturally from a simpler approach.

**Suggested test:** Train TopK SAEs (no matryoshka loss), rank features by activation frequency, compute ICC for the top-32 vs bottom-32. If the ICC gradient still exists, the matryoshka loss is providing ordering convenience, not discovering structure. If it doesn't, the matryoshka loss is genuinely needed.

### Concern 2: The paper is TabPFN-heavy — cross-model results are mostly placeholders

The draft's actual empirical results (Tables 8-14, Section 10) are almost entirely from TabPFN. The CKA matrix (Table 3) is filled in, but concept-level results (Tables 2, 5) are empty. The paper's central claim — that models share concepts at coarse levels but diverge at fine levels — currently rests on a single model's SAE analysis plus the Figure 1 Jaccard heatmaps from the codebase (4 models only).

This is the paper's most critical gap. The ICC analysis, concept examples, and HP sweep are all strong — but they're all TabPFN. The story requires at minimum 4-5 models with filled-in concept alignment tables to be publishable.

### Concern 3: Matryoshka-Archetypal is validated but not committed to in the methodology

Table 9 shows Matryoshka-Archetypal wins the HP sweep (score 0.924, stability 0.91). But Section 3.3 ("Matryoshka Sparse Autoencoder") describes generic Matryoshka SAE without mentioning the archetypal decoder constraint. Section 5.5 lists "Archetypal Matryoshka SAEs" as *future work*.

This is confusing. The codebase implements it, the HP sweep validates it, but the paper methodology doesn't use it. Either:
- **Commit to Matryoshka-Archetypal as the paper's method** (strongest option — the stability numbers justify it), or
- **Present the comparison honestly** — show that Matryoshka alone has 0.32 stability, explain why this motivated adding the archetypal constraint, and present Matryoshka-Archetypal as the final method.

Leaving it as "future work" when you already have the results is leaving your best card on the table.

### Concern 4: The missing link — geometry/concepts → downstream performance

**Status: Still the biggest gap.** No figure or table in the draft connects CKA similarity or concept overlap to downstream task performance. DeepMind deprioritized SAE research precisely because features didn't predict outcomes. TabArena has benchmark scores for all 51 datasets.

Without this analysis, a reviewer can reasonably ask: "So what? Models share some concepts — does that predict anything useful?" The 2x2 framework (Section 4.5) is analytically elegant but purely descriptive without a performance dimension.

### Concern 5: ICC values at L0 are moderate (0.551), not high

The paper frames L0 features as "dataset-level descriptors" but the mean ICC is 0.551 — meaning ~45% of L0 feature variance is *within*-dataset. This is a meaningful hierarchy, but it's not a clean dataset-vs-row separation. The paper should acknowledge this nuance rather than presenting it as a binary (dataset-level vs. row-level). The gradient from 0.55 → 0.19 is the real finding, not the absolute values.

### Concern 6: Missing citations that would pre-empt reviewer objections

The paper does not cite or address:

1. **DeepMind's negative results / deprioritization** (March 2025) — The most prominent recent SAE critique. Addressing it head-on ("our use case is discovery, not downstream application") would pre-empt the obvious reviewer question.

2. **"Use SAEs to Discover Unknown Concepts, Not to Act on Known Concepts"** ([arxiv 2506.23845](https://arxiv.org/html/2506.23845v1)) — This paper directly validates your approach. It argues SAEs excel at concept enumeration (what you're doing) but fail at concept detection/steering (what you're not doing). Citing it strengthens your positioning.

3. **SAE feature instability across random seeds** — Your archetypal stability results (Table 9) are a direct response to this known problem, but the paper doesn't frame it that way. Explicitly citing the instability concern and showing your 0.91 stability as the solution would be more compelling.

### Concern 7: Binary Jaccard vs. activation correlation — use both

**Previous concern:** Binary Jaccard discards magnitude information.

**Updated assessment:** The paper already defines activation-based similarity (Eq. 6) and domain profile similarity (Eq. 7) as methods. But Figure 1 and the cross-model analysis in the codebase still use binary Jaccard. The paper should report both metrics. If they agree, it strengthens the finding. If they diverge, that's an important result about what "concept sharing" means.

### Concern 8: The "20 core concepts explain 90% of variance" claim needs support

Section 10.3 states "The SAE decomposes these into 20 core concepts that explain 90% of embedding variance." What variance? R² from SAE reconstruction at which truncation point? Or variance explained by the 20 most-active features? This claim needs a precise definition and supporting figure (e.g., a cumulative variance explained curve).

---

## Prioritized Recommendations (Revised)

### P0 — Do these before submitting

1. **Fill in cross-model concept alignment tables (Tables 2, 5).** The paper's central claim requires concept-level results from at least 4-5 models. Running SAE training on Mitra, TabDPT, CARTE, and TabICL embeddings and computing pairwise concept alignment at each matryoshka level is the single most important task. Without this, the paper is a TabPFN case study, not a cross-model comparison.

2. **Add the CKA/concept-overlap vs. performance analysis.** Correlate CKA similarity and concept Jaccard with downstream performance similarity (rank correlation or absolute error correlation) across the 51 TabArena datasets. This is what separates a descriptive geometry paper from an explanatory one. Plot it as a scatter with the 2x2 quadrants from Section 4.5 overlaid.

3. **Commit to Matryoshka-Archetypal in the methodology.** The HP sweep (Table 9) already validates it. Rewrite Section 3.3 to describe the archetypal decoder constraint as part of the method, not future work. Frame the stability improvement (0.32 → 0.91) as a methodological contribution.

### P1 — Strongly recommended

4. **Add the TopK-without-matryoshka baseline.** Train standard TopK SAEs, rank features by activation frequency, and compute ICC for top-32 vs. rest. This ablation determines whether the matryoshka loss discovers structure or merely imposes convenient ordering. One table in the appendix would suffice.

5. **Report activation-based correlation (Eq. 6) alongside Jaccard.** You already define it in the paper. Use it. If both metrics tell the same story, that's a robustness check. If they diverge, you've learned something important.

6. **Add key citations and position against the DeepMind critique.** Add 2-3 sentences in Related Work or Discussion addressing DeepMind's deprioritization (your use case is discovery, not downstream application) and citing the "Discover, Not Act" paper as supporting evidence.

7. **Run 5+ seeds for stability.** The 0.91 stability number from Table 9 is strong, but reporting mean ± std across 5-10 seeds would make it unassailable. If archetypal anchoring genuinely reduces seed variance, show it with error bars.

### P2 — Would improve the paper

8. **Qualify the ICC interpretation.** L0 ICC of 0.551 means the hierarchy is a gradient, not a binary separation. Reframe as "the matryoshka hierarchy captures a continuous spectrum from dataset-level to row-level concepts" rather than implying a clean split.

9. **Support the "20 concepts / 90% variance" claim.** Add a cumulative variance explained curve showing how many features are needed to reach 90% R² at each matryoshka level. Define precisely what "variance" means here.

10. **Sweep the correlation clustering threshold.** Show concept presence matrix stability across `corr_threshold` values [0.10, 0.15, 0.20, 0.25]. Appendix figure.

11. **Justify or drop RA-SAE.** The codebase has both strict and relaxed archetypal variants. Pick one and commit. If strict A-SAE (relaxation=0.0) is what you use, don't mention relaxation in the paper — it distracts.

---

## What Changed After Reading the Draft

| Original Concern | Status After Reading Draft |
|---|---|
| Matryoshka hierarchy is imposed | **Partially resolved** — ICC analysis provides real evidence of hierarchy, but TopK baseline still needed |
| Feature stability underexplored | **Mostly resolved** — Table 9 shows 0.91 stability for Matryoshka-Archetypal. More seeds would strengthen but the comparison is already compelling |
| Binary Jaccard discards signal | **Unchanged** — Paper defines activation correlation but doesn't use it in results |
| Missing performance correlation | **Unchanged** — Still the biggest gap |
| RA-SAE unmotivated | **Reframed** — The real issue is that Matryoshka-Archetypal is validated but not committed to in methodology |
| Correlation threshold ad-hoc | **Acknowledged in limitations** (Section 6, item 2) — mentions sweep in appendix |
| Layer selection inconsistent | **Resolved** — Section 8 shows principled layerwise CKA selection |
| *NEW:* Paper is TabPFN-heavy | Cross-model tables are placeholders — most critical gap |
| *NEW:* Missing key citations | DeepMind deprioritization and "Discover Not Act" paper should be cited |
| *NEW:* ICC values moderate at L0 | Paper overstates the dataset-vs-row separation |

---

## Key References from Recent Research

| Source | Relevance to This Paper |
|--------|-----------|
| [Use SAEs to Discover, Not to Act](https://arxiv.org/html/2506.23845v1) | Directly validates your discovery use case; should cite in Related Work |
| [DeepMind: Deprioritising SAE Research](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9) | Address head-on — your use case is different from their negative results |
| [SAE Survey (EMNLP 2025)](https://arxiv.org/abs/2503.05613) | Comprehensive landscape; good for positioning |
| [SAE Features for Classification](https://arxiv.org/abs/2502.11367) | Cross-model transfer of SAE features (Gemma 2B→9B); relevant precedent |
| [Scaling and Evaluating SAEs (ICLR 2025)](https://arxiv.org/abs/2406.04093) | k-sparse SAE baselines; scaling reference |
| [Goodfire: SAE Probes for PII Detection](https://www.goodfire.ai/research/rakuten-sae-probes-for-pii-detection) | First enterprise SAE deployment; shows SAEs can work in production for specific tasks |
| [Interpretability as Compression (MDL-SAEs)](https://arxiv.org/abs/2410.11179) | Alternative framing worth acknowledging |
| [Geometry of Concepts](https://www.mdpi.com/1099-4300/27/4/344) | SAE feature structure analysis; related to your geometric comparison |

---

## Bottom Line

The paper is stronger than the codebase alone suggested. The ICC analysis, the 2x2 CKA-vs-concept framework, the principled layer selection, and the stability comparison (Table 9) are all solid contributions. The core methodology is sound.

The three critical gaps, in order:

1. **Cross-model results are empty.** The paper is currently a TabPFN case study with a cross-model framework sketched but not executed. Fill Tables 2 and 5.
2. **No performance connection.** Without showing that concept alignment predicts something about downstream performance, the paper is descriptive geometry. Add the CKA/concepts vs. performance scatter.
3. **Commit to the method you validated.** Matryoshka-Archetypal won the HP sweep. Use it in the methodology section. Frame the stability improvement as a contribution.

Everything else is important but secondary to these three.
