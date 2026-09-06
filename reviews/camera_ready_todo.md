# Camera-ready / if-accepted TODO

Changes promised in the rebuttal or surfaced during rebuttal work. Tagged with the
reviewer that prompted each. Keep in sync with `rebuttal_draft_*.md`.

## A. Section 3 (Experimental Setup) additions
- [ ] **Random-SAE definition** — state it is NOT purely random: archetypal
      matryoshka top-k SAE; the baseline **retains the data-derived archetype
      dictionary** and randomizes only the concept assignments (logits +
      deviations). Move this *before* the results. (your #2; SD9t clarity, dVDs, nn7D)
      → draft paragraph in `rebuttal_draft_random_ofnl.md` (S3).
- [ ] **SAE selection rule** — lift from App. D.6 into main text: feasible set =
      {R² ≥ 0.80, alive ≥ 0.80, stability ≥ 0.75}, then pick the least-complex SAE
      (min √(hidden_dim · L0), the two capacity terms of the sweep objective
      test_recon·√(hidden_dim)·√(L0)/alive_frac). (SD9t Q2, nn7D)
- [ ] **TabDPT retrieval** — state explicitly that TabDPT runs in its native
      retrieval-augmented mode (per-query nearest-neighbour context), and that
      retrieval is preserved through the causal intervention. (SD9t Q5)
- [ ] **TabDPT transfers are distributed, so single-concept patching fails there**
      — candidate framing for App F.3: transfer into TabDPT succeeds (gc_deployed
      0.98) by spreading across ~60 co-active concepts per row, so no single
      concept is necessary. Evidence: median acceptance rank 34 (vs 3-8 for the
      other recipients), per-concept ablation ceiling 6-11x smaller (|interval|
      0.0015 vs 0.010-0.017), and measured movement 0.003 vs 0.28-0.37. Rank is
      the dominant driver — at matched rank <=3 TabDPT still lags (0.178 vs
      0.33-0.46), so lower receptivity is a real but secondary effect. Present as
      a positive result about distributed representation with concepts-per-row as
      an a-priori predictor of where single-concept intervention can reproduce a
      transfer, NOT as an exclusion. Coverage cost if TabDPT-recipient cells are
      dropped: 17 of 335 concepts (335 -> 318), all reachable only at median rank
      52 with movement -0.0001. Pairs with the TabDPT retrieval item above.
- [ ] **Transfer linear map** — lift a brief description from App. F.8 into the
      main text (what is fit for the concept map). (SD9t clarity)

## B. Metric / methodology clarifications
- [ ] **Strong/weak on low-event-rate datasets** — clarify that AUC is used for
      strong/weak, with a neg-logloss fallback when the 200-sample holdout is
      single-class (no class-1 present); affects ~4 datasets (seismic-bumps,
      hiva_agnostic, taiwanese_bankruptcy_prediction, Marketing_Campaign). Same
      fallback in both trained and random arms; datasets NOT dropped (matches paper).
      (your #1)
- [x] **Per-example strong/weak** — DONE 2026-09-05. Landed in §3.3 (Intervention),
      NOT §4.2 (that's where S/W is actually defined). Clarified S is the
      dataset-level stronger model and interventions are applied per row. (SD9t Q1)

## C. Tables / results
- [ ] **Add `acc` (acceptance) column to the ablation table (Table 1)** — parallel
      to the transfer table; its omission is an oversight. (your #3; SD9t Q3)
- [x] **Full-test-set / selection-bias answer** — DONE 2026-09-05. Decision: do NOT
      replace the published below-diagonal headline numbers (per-experiment mean:
      ablation 0.93, transfer 0.90). The symmetric pipeline pools by ROW, so its
      below-diag recompute is 0.815 (transfer) — a benign aggregation difference, not
      stochastic. Rather than reconcile two aggregations, we ADDED one paragraph to
      §4 (after the transfer discussion) reporting the previously-untreated
      above-diagonal rows in the paper's own row-pooled idiom: transfer/weak→strong
      0.98 over 46,446 rows, ablation 0.89 over 48,237 rows, two triangles cover
      99–100% of the test set. Frames per-row-winner knowledge as an experimental
      control, not a precondition (answers dVDs selection-bias without new tables).
      Numbers verified vs output/rebuttal/symmetric_summary.json.
- [x] **Add weak→strong transfer result** — gc≈0.98 now stated in the §4 paragraph
      above (surfaces that strong models do NOT already contain all of the weaker
      model's concepts). (SD9t Q1b)
- [x] **Add "no harm" result** — DONE 2026-09-05, §4 Transfer (paragraph after the
      full-test one). Verified fresh via scripts/rebuttal/count_transfer_harm.py:
      0 / 77,536 intervened transfer rows moved the recipient to a worse loss on
      y_true; 100% improved, 0 unchanged. Framed as the answer to dVDs' "does
      transfer preserve performance or harm the recipient?" (dVDs)
- [x] **Revise the "purely new capacity" sentence** — DONE 2026-09-06. Replaced the
      near-zero single-vector cosine (median 0.03, an artifact of high-dim concentration)
      with the PER-RECIPIENT subspace aligned/novel energy split from
      `subspace_by_recipient.py` (reads subspace_summary.json + forward_deltas).
      Off-manifold energy by recipient: tabpfn 0.49, tabicl 0.50, tabicl_v2 0.64,
      mitra 0.60, tabdpt 0.53, carte 0.45 (range 0.45–0.64); principal angle 27–66°;
      median K 3–6. Landed in: abstract (qualitative "roughly half … 0.45–0.64"),
      §4 (rewrite of the orthogonality paragraph + new Table tab:offmanifold, all 6
      recipients), conclusion (one quantitative sentence).
      DELIBERATELY NOT USED: the functional-decomposition pooled rel_off (~0.46) — it
      has no coherent reading (basis is each recipient's own eigenbasis; see the pooled
      caveat below). Per-recipient split is the coherent unit. Ridge-map alignment
      floor (~0.33) NOT cited this round (would need subspace on forward_deltas_random);
      revisit if a reviewer/AC presses on whether the alignment is above chance.
      NOTE: carte FLIPS between measures — most on-manifold here (0.45 off, 27°) vs
      highest rel_off (0.81), because carte is near-degenerate (active dim ≈3/300) so
      the delta concentrates in its few dominant directions.
- [ ] **IF the on/off split is used at all, use the 99% variance threshold** (settled
      2026-08-02 — a choice about how we would report it, NOT a commitment to report it;
      see the disaggregation result below, which weakens the whole thread).
      Rationale: 99% is the CONSERVATIVE choice for an off-manifold claim, because it
      hands the on-manifold subspace essentially all the recipient's embedding variance
      and therefore yields the SMALLEST rel_off. Pooled trained rel_off by threshold:
      80% 0.47, 90% 0.46, 95% 0.44, 99% 0.40 (random 0.46/0.43/0.42/0.37). Present
      80/90/95 as a **sensitivity check** — "the choice of threshold does not drive the
      conclusion, and we report the pessimistic end" — NOT as a robustness/replication
      claim; the four runs share rows, deltas and acceptance, and differ only in the
      dimension of E. Practical: the `_t99` dirs store `ke`/`var_threshold`; the
      unsuffixed 90% dir predates that field.
- [ ] **CAVEAT to state wherever the POOLED on/off number appears: it is a row-count
      weighted blend of six very different recipients, and carte dominates it.**
      Per-recipient rel_off @99% spans 0.11–0.89 (carte 0.73–0.89, mitra 0.27–0.45,
      tabicl 0.37–0.39, tabpfn ~0.31, tabdpt 0.22–0.25, tabicl_v2 0.11–0.14). carte is
      11,197 of 36,039 trained classification rows (~31%), and its on-manifold subspace
      is near-degenerate — ke = 1–3 of 300 dims at 80%, still only 1–11 at 99% — so
      almost the entire delta is off-manifold *by construction* for that recipient. The
      trained and random arms also carry different recipient mixes (carte 11,197 trained
      vs 6,304 random), so the pooled trained-vs-random contrast is confounded with
      composition. **The pooled number is not merely confounded — it has no coherent
      reading.** E is the recipient's own eigenbasis, so rel_off is defined against a
      different basis per recipient, with effective ranks from ke≈1–11 of 300 (carte) to
      43–128 of 768 (tabdpt). Averaging them averages quantities that do not share a
      definition, so "report pooled with a caveat" is NOT one of the options. What DOES
      survive: within a fixed recipient, trained-vs-random is a fair comparison because
      the basis is held constant. What does NOT: the pooled value, and any reading of
      rel_off's absolute LEVEL as evidence about concepts rather than about the
      recipient's spectrum (for carte, "off-manifold" means "nearly the whole space" and
      the measurement is close to vacuous). This weakens the on/off split as a vehicle
      for the ofnL Q2 "purely new capacity" answer — if that answer is still wanted, the
      fallback is the aligned/novel energy split + the ridge-map alignment floor.
      NOT established: whether ke/emb_dim
      explains the ordering generally — tabicl (48–122/512, 0.37) vs tabicl_v2
      (37–116/512, 0.11) breaks it. carte is a demonstrable outlier; a general
      "low-rank ⇒ high rel_off" law is an untested hypothesis.
- [ ] **Foreground the per-concept efficiency (parsimony)** as the random-baseline
      answer: trained closes 0.90 with K≈8.9 vs random 0.52 with K_R≈17.3 (~3.4×
      gap-closure per concept). (dVDs, nn7D)
- [ ] **(Appendix, camera-ready) Truly-random isotropic-SAE control for the
      on-manifold ENERGY question.** The archetype-random baseline keeps the
      data-derived archetype dictionary, so its deltas are on-manifold *by
      construction* — it cannot null whether the transfer delta's on-manifold energy
      concentration comes from the concepts vs. the acceptance/greedy search + the
      data-aligned dictionary. Isotropic k_e/d fails the other way (ignores
      acceptance entirely). The clean null is an SAE whose dictionary NEVER touches
      the archetypes (isotropic decoder directions), run through the SAME
      matching → caches → greedy transfer → decomposition pipeline; then measure the
      accepted deltas' on-manifold energy. Concentrates on-manifold ⇒ the acceptance
      search forces it; near chance ⇒ the data dictionary does. Full pipeline for a
      new SAE variant (random SAEs match differently → fresh matching). Doubles as a
      stronger R1 baseline (a dictionary that does NOT span the data subspace).
      NOT used for the dVDs discussion answer — that rests on the functional gc split
      (controlled by the archetype-random arm) and drops the energy claim.
      (ofnL Q2 follow-up; the "purely random control" idea)
- [ ] **(Appendix) "Disambiguation of apparently tied off-manifold gc."** On mean-of-gc
      the off-manifold split looks tied trained-vs-random (@99%: rel_off 0.40 vs 0.37) —
      but that's a gc-normalization artifact. gc = (loss_w−loss_int)/(loss_w−loss_s)
      divides out the stakes, so near-tie rows (tiny gap, gc≈1 on ~0 nats) get equal
      weight and drown out the high-stakes rows where the arms differ. Disambiguate via
      scripts/rebuttal/gap_stratified_decomposition.py (classification rows, logloss gap):
      1. **FIRST ITEM — quartile-stratified gc by absolute gap** (logloss_w − logloss_s):
         rel_off rises monotonically with the gap — TRAINED 0.33/0.38/0.39/0.48,
         RANDOM 0.33/0.37/0.37/0.42. Off-manifold matters MORE on high-stakes rows, and
         trained > random at the top. Robust presentation; lead with it.
      2. Loss-weighted rel = Σ(gc_c·gap)/Σ(gc_full·gap) (= nats-of-loss-removed ratio):
         TRAINED rel_off 0.78 vs RANDOM 0.50 (vs 0.40/0.37 mean-of-gc) — the learned-vs-
         random gap gc hid. CAVEAT: tail-sensitive (large-gap band runs to ~16 nats);
         report winsorized and note the tail. Functional comparison (dictionary+acceptance
         held fixed), so it's a valid learned-vs-random signal, not a geometry artifact.
      Run across the whole sweep (80/90/95/99), not just 99. Revise any main-text wording
      that calls off-manifold "redundant/tied"; this strengthens the latent-capacity
      reading rather than deflating it. (ofnL Q2)
      ROBUSTNESS — needed before this carries weight (newest, least-settled result):
      confirm the 0.33→0.48 quartile trend holds ACROSS PAIRS/recipients (trained rises,
      random stays flat), not driven by a subset of pairs. Only then does it earn a claim.
      **GATE RESULT (2026-08-02): the gate does not pass. This did not earn a claim.**
      Reproduce with `gap_stratified_decomposition.py --thr 99 --by recipient` and
      `--by pair` (both print the Q1..Q4 rel_off table and the Q4>Q1 tally per arm).
      - **Per recipient**: trained Q1→Q4 rel_off — carte 0.89→0.73, mitra 0.27→0.45,
        tabdpt 0.25→0.22, tabicl 0.37→0.39, tabicl_v2 0.13→0.11, tabpfn 0.32→0.31.
        Trained rises in 2/6; RANDOM rises in 3/6. The "trained rises, random flat"
        signature does not survive.
      - **Per pair** (the decisive run): trained Q4>Q1 in **13/27** pairs, random in
        **13/29** — the two arms are indistinguishable, so the pooled 0.33→0.48 rise is
        not a pair-level phenomenon.
      - Residual structure is set by the RECIPIENT and flips sign: mitra-recipient
        trained rises in 4/5 donors (+0.06..+0.35) while random is flat/negative;
        tabdpt-recipient is the MIRROR IMAGE (random rises 5/5 at +0.04..+0.18, trained
        0/5); carte-recipient trained falls in 5/5 (−0.08..−0.21). Random's own
        excursions (+0.18, +0.15, −0.21) are as large as trained's, so all of the
        mitra values except tabdpt→mitra (+0.35) sit INSIDE random's noise band.
      - That recipient-driven structure is **definitional, not empirical**: E is the
        recipient's own eigenbasis, so rel_off is parameterized by recipient geometry
        before any donor enters. Donor effects are second-order (within-recipient
        rel_off barely moves across 5 donors; across recipients it spans 0.11–0.94).
      - The threshold sweep could not have caught any of this: it varies dim(E) on a
        FIXED set of rows, orthogonal to the disaggregation problem.
      NOT tested, deliberately: the loss-weighted variant (0.78 vs 0.50). Deprioritized
      — the quartile disaggregation already answers the gate, and the loss-weighted
      number is tail-sensitive (large-gap band runs to ~16 nats). The script no longer
      computes it; re-add if ever revisited.

## D. Framing
- [x] **Reframe interventions as a causal DIAGNOSTIC, not a deployment/model-
      improvement method** — DONE 2026-09-06. Intro: "improve the weaker one" ->
      "close its per-row gap"; dropped "accelerate convergence" from the release
      contribution. Conclusion: "improves predictions" -> "closes their per-row gap",
      and replaced "principled basis for targeted model improvement" with an explicit
      "causal diagnostic … not a route to costless accuracy gains; per-row winner is
      the experimental control." (dVDs, nn7D)
- [ ] **State HyperFast / Tabula-8B exclusion scope** — causal conclusions are for
      the transformer-ICL family; those two lack the per-row embedding interface the
      intervention needs. (dVDs)
- [ ] **Note interpretability limitation** — systematic human/ground-truth concept-
      semantics validation is future work (beyond the patch examples). (dVDs, nn7D)
- [ ] **App F.5 — dense SAE latents & labeling difficulty.** Cite "Dense SAE latents
      are features, not bugs" (arXiv:2506.15679). Our archetypal-matryoshka SAEs
      contain dense latents; per that work these are *genuine features*, not artifacts
      — so their presence is not a defect in our dictionaries. But density = the latent
      fires on a large fraction of inputs, so there is *less discriminative firing
      evidence* to pin down what it encodes, which is a concrete reason concept
      labeling is harder for a subset of our concepts. Use it two ways in App F.5:
      (a) support that dense latents are legitimate (pre-empts "these look like broken
      features"); (b) explain/scope our labeling difficulty on dense concepts.
      NOTE: firing DENSITY (fraction of rows a latent activates on — an encoder
      property) is distinct from transfer ACCEPTANCE rate (the ≥1000/≈2% cutoff for
      the patch set — a transfer property). Do NOT assume the patch concepts are the
      dense ones; if we want to connect App F.5 to the patch-coverage story, measure
      firing density on the SAE activations directly and check the overlap.
- [ ] (optional) **Negative-R² reframe** — state that global map R² is the wrong
      yardstick; specificity is per-row directional edit + gap-closure-per-concept.
      (dVDs)
- [ ] **(CONDITIONAL — only if a reviewer/AC requests a retitle) Retitle to
      "INCEPT: Infusing Novel Concepts to Explain Pretrained Tabular Model Disparity."**
      Do NOT change unilaterally or propose it ourselves — apply only if reviewer/AC
      feedback explicitly asks for a title change.

## E. Internal / reproducibility (for open-source release; not necessarily paper text)
- [ ] **Baseline-prediction consistency between arms** — `perrow_importance` and
      `perrow_importance_random` were computed 2–3 days apart and their baseline
      preds drift (SAE-independent, so they should be identical). Decision-level
      impact: carte 7.8% class-swaps, mitra 1.4%, tabdpt 0.5%, tabpfn 0.06%,
      tabicl/tabicl_v2 0%. Fix: have the random arm REUSE the trained baselines
      (baseline is SAE-independent), or re-derive strong/weak from a single baseline.
      Root-cause the carte drift (preprocessing/RobustScaler timing vs. stochastic
      inference) — currently deferred.
- [ ] Confirm ensemble/retrieval inference is seeded so baselines are reproducible
      (tabpfn/mitra/tabdpt drift; tabicl/tabicl_v2 already deterministic).
- [ ] **CARTETail lacks `recapture`** — every other main-sweep tail (tabpfn, tabicl,
      tabicl_v2, mitra, tabdpt) implements `recapture(X_query_new)`; CARTETail does not,
      so `functional_decomposition.py` (and any re-injection path via
      `intervene_lib.batched_intervention`) fails carte-recipient datasets with
      `'CARTETail' object has no attribute 'recapture'`, silently dropping them. Adding it
      means re-running CARTE's star-graph transform on the modified central-node
      embeddings — non-trivial. Until then the functional decomposition excludes
      carte-recipient rows; state this scope where the on/off-manifold split is reported.
      (NOTE: superseded this session — carte-recipient now runs via
      `predict_row_batched`; the decomposition covers all 6 recipients. Update/close.)

## F. Main-text relocation — dVDs list committed in the reply (+ SD9t)
The specific "move into main text" commitments from the reply. Several overlap §A/§C
above; this is the consolidated dVDs checklist so nothing is dropped in the edit.
- [x] **Full-test-set numbers + weak→strong gc=0.981 into MAIN TEXT** — DONE
      2026-09-05, §4 paragraph (see §C item above); in the body, not appendix. (dVDs)
- [x] **Donor prediction as a REQUIRED INPUT** — DONE 2026-09-05. Landed in §3.3
      (Intervention), not §4.2. States S's per-row prediction is a required input,
      not inferred. (dVDs)
- [ ] **SAE stability metric promoted from App. D into main text** (ties to the §A
      "SAE selection rule" item — surface the stability≥0.75 criterion in the body). (dVDs)
- [ ] **Geometry-matched (random-SAE) control into Sec. 3, AHEAD of the numbers**
      (drafted in §A `rebuttal_draft_random_ofnl.md` S3 — confirm placement before
      results). (dVDs)
- [ ] **Per-concept gap-closure DIRECTLY in Tables 1 and 2** — as a table column, not
      only narrated (concretizes the §C parsimony item). (dVDs)
- [ ] **SD9t's four formatting fixes** — enumerate from §B and apply. (SD9t)
- [ ] **App. F.8 landmark / linear-map summary into main text** (the §A "Transfer
      linear map" item — include the landmark-count summary). (SD9t)

## G. Candidate: reproducibility appendix (2026-08-16)
- [ ] **Consider an appendix section on reproducibility**, drawing on the patching
      pipeline's measured behaviour. Candidate content if included: (i) bit-exact
      replication per host+commit through the full search (v20↔v21: 3,046/3,046
      identical patches, zero measurement drift); (ii) cross-GPU behaviour — TF32
      matmul defaults on Ampere/Ada make forwards architecture-dependent at ~1e-3
      relative, so near-tie argmax decisions cascade into different chosen patches
      across hosts (~12% patch identity 4090↔3090) while population metrics
      (gap_opened, blast, share-of-ceiling) remain equivalent — i.e., reproducibility
      claims should be stated as bit-exact per hardware, distribution-equivalent
      across hardware; (iii) determinism prerequisites (CUBLAS_WORKSPACE_CONFIG,
      deterministic algorithms, fixed query windows, extraction seeds) and the canary
      as the runtime integrity guard; (iv) the KNIFE-EDGE population: adjacent
      code-version comparisons on the same host re-measure ~0.1-0.7% of rows'
      recipient-LOO intervals differently (occasionally sign-flipping), concentrated
      in tabpfn/tabdpt-recipient rows -- model-internal tie amplification of ulp-level
      build-order differences, the same discontinuity class as tabpfn's cross-host
      behaviour; within-commit replication remains bit-exact, and population metrics
      are indifferent.
