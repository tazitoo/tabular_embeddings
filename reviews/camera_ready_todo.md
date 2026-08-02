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
- [ ] **Transfer linear map** — lift a brief description from App. F.8 into the
      main text (what is fit for the concept map). (SD9t clarity)

## B. Metric / methodology clarifications
- [ ] **Strong/weak on low-event-rate datasets** — clarify that AUC is used for
      strong/weak, with a neg-logloss fallback when the 200-sample holdout is
      single-class (no class-1 present); affects ~4 datasets (seismic-bumps,
      hiva_agnostic, taiwanese_bankruptcy_prediction, Marketing_Campaign). Same
      fallback in both trained and random arms; datasets NOT dropped (matches paper).
      (your #1)
- [ ] **Per-example strong/weak** — clarify wording in Sec 4.2 that strong/weak is
      the per-row winner under fixed shared context, not an overall ranking. (SD9t Q1)

## C. Tables / results
- [ ] **Add `acc` (acceptance) column to the ablation table (Table 1)** — parallel
      to the transfer table; its omission is an oversight. (your #3; SD9t Q3)
- [ ] **Replace oracle-subset numbers with FULL-TEST-SET numbers** — transfer
      gc≈0.883, ablation gc≈0.919 over ~100% of rows (below+above diagonal), vs the
      ~60% below-diagonal subset in the submission. **[substantive — new headline
      numbers/table]** (dVDs primary request)
- [ ] **Add weak→strong transfer result** — gc≈0.981: strong models do NOT already
      contain all of the weaker model's concepts. (SD9t Q1b)
- [ ] **Add "no harm" result** — 0 / 77,536 intervened rows moved the recipient to
      a worse loss (100% improve or unchanged); follows from the acceptance/overshoot
      rule. (dVDs)
- [ ] **Revise the "purely new capacity" sentence** — replace the near-zero cosine
      with the measured aligned/novel energy split + the functional decomposition
      (on-manifold vs off-manifold gap-closure), and note the ridge-map alignment
      floor (~0.33) so the geometry is not over-read. (ofnL Q2) — numbers pending compute.
      NOTE (updated 2026-08-02): the 5-recipient caveat is STALE — carte-recipient now
      runs via `predict_row_batched`, so the decomposition covers all 6 recipients.
- [ ] **DECIDED (2026-08-02): report the on/off split at the 99% variance threshold.**
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
      composition. DECIDE before writing: report per-recipient rather than pooled, or
      report pooled with this caveat stated inline. NOT established: whether ke/emb_dim
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
      **GATE RESULT (2026-08-02): FAILED. Do not claim this.** Run
      `gap_stratified_decomposition.py --thr 99 --by recipient`. The pooled quartile rise
      is NOT broad-based — per-recipient trained Q1→Q4 rel_off: carte 0.89→0.73,
      mitra 0.27→0.45, tabdpt 0.25→0.22, tabicl 0.37→0.39, tabicl_v2 0.13→0.11,
      tabpfn 0.32→0.31. Trained rises in only 2/6 recipients; RANDOM rises in 3/6 —
      i.e. the "trained rises, random stays flat" pattern does not survive. Only mitra
      shows the pooled pattern. The pooled trend is a composition (Simpson's) effect:
      rel_off differs ~8× across recipients and the gap quartiles do not draw evenly
      from them. The loss-weighted 0.78-vs-0.50 contrast is confounded the same way
      (different recipient mixes per arm — see the pooling caveat above). The threshold
      sweep could not have caught this: it varies the subspace dimension on a FIXED set
      of rows, so it is orthogonal to the composition problem.
      IF revisited: the honest version is per-recipient, and on current evidence only
      mitra supports it — one recipient is an anecdote, not a claim.

## D. Framing
- [ ] **Reframe interventions as a causal DIAGNOSTIC, not a deployment/model-
      improvement method** — remove wording implying free full-benchmark accuracy
      gains; per-row-winner knowledge is the experimental control. (dVDs, nn7D)
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
- [ ] **Full-test-set numbers + weak→strong gc=0.981 into MAIN TEXT** (were listed as
      §C additions; dVDs wants them in the body, not appendix). (dVDs)
- [ ] **Donor prediction as a REQUIRED INPUT in Sec. 4.2** — state that the
      intervention / strong–weak definition takes the donor's per-row prediction as a
      given input, not something inferred. (dVDs)
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
