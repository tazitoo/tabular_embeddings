# Paper Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh paper figures, tables, and quoted numbers against the
three refreshed experiment data dirs pulled from workers on 2026-04-19,
wire generators to emit PDFs into both `output/` and the paper repo,
then diff-report number movements for the author.

**Architecture:** Each generator reads from `output/<sweep>/`, writes
a PDF/TeX artefact into BOTH `output/paper_figures/<section>/` (working
copy) AND `/Users/brian/src/tabular_embedding_paper/figures/<section>/`
(paper repo). Generators are named `<artefact>.py` under
`scripts/paper/<section>/` or `scripts/tables/<name>/`. Rename
`table4/` → `ablation_summary/` because the paper now calls it Table 2
and the directory name was actively misleading.

**Tech Stack:** Python (matplotlib, numpy), LaTeX (matplotlib pgf or
plain \input{}), shell. No new dependencies. No GPU needed for any
refresh step.

**Refreshed data dirs (all verified in-sync with workers 2026-04-19):**
- `output/ablation_sweep_tols/` — ablation sweep trained SAEs
- `output/transfer_random/` — transfer with random baseline SAEs
- `output/transfer_global_mnnp90_trained_tols/` — transfer with trained SAEs

**Paper repo:** `/Users/brian/src/tabular_embedding_paper/`
- `sections/*.tex` — prose and `\includegraphics{figures/...}`
- `figures/{1_intro,4_results,A_appendix,…,G_observational}/` — PDFs
- `tables/*.tex` — 5 generated tables

**Labeling block:** Tasks touching `concept_labels_table.tex`,
per-feature validator accuracy, or pairing deltas (f_6, f_11, f_36,
f_86, f_92) are BLOCKED until task #28 (outside repo) lands all five
`f{6,11,36,86,92}_label_{A,Apairing}_v10.json` files into
`output/contrastive_examples/mitra/`. At plan time only f_92 has
landed. These tasks are gathered at the end under Phase 8.

---

## Open decisions to resolve before Phase 2

- **Ablation random baseline pairing.** `ablation_sweep_tols/`
  (Apr 16, trained) differs from `ablation_sweep_random/` (Apr 3,
  random) in one behaviour that affects aggregates: `min_gap=0.01`
  early-skip sets `gap_closed=1.0` on rows where the two models agree
  within 1% (`ablation_sweep.py:264-267`). Trained (tols) gets the
  free 1.0; old random didn't. Other tols changes
  (`gc_tolerance=0.99`, deterministic tails, per-feature `feature_preds`)
  do not meaningfully shift random's aggregate mean. **Plan proceeds
  with (c′): keep the existing random, but quantify the asymmetry in
  the diff report (Task 14 adds this measurement). If the magnitude
  exceeds ~1 pp on any headline cell, re-run random with the current
  `ablation_sweep.py` defaults (matching-tols protocol) — that is a
  contingency Task 14b, unblocked only if the audit exposes a real
  shift.**
- **Transfer "trained vs random" pairing.** Table and some figures
  compare trained-map transfer to random-map transfer.
  `transfer_global_mnnp90_trained_tols/` is the trained side;
  `transfer_random/` is the random side. They are from the same
  refreshed pull. Proceed with this pair.
- **Old figures with no refreshed input (geometric, per-model-layers):**
  Rerun as sanity regen only; expect no numeric change.

---

## File structure

**Scripts being modified or created:**

- Rename: `scripts/tables/table4/` → `scripts/tables/ablation_summary/`
  - Rename file: `table4.py` → `ablation_summary.py`
  - Keep file: `ablation_summary.tex` (regenerated output)
- Modify: `scripts/tables/ablation_summary/ablation_summary.py`
  - Swap `SWEEP_DIR` to `ablation_sweep_tols`
  - Add paper-repo dual-write path
- Modify: `scripts/paper/sec4/compute_row_intervention_data.py`
  - Add dual-write path (already emits JSON, figure builder consumes it)
- Modify: `scripts/paper/sec4/row_intervention_figure.py` — dual-write PDF
- Modify: `scripts/paper/sec4/fig_geometric_vs_concept.py` — dual-write
- Modify: `scripts/paper/appendix_a/fig_per_model_layers.py` — dual-write
- Modify: `scripts/figures/plot_intervention_distributions.py` — dual-write
- Modify: `scripts/figures/plot_intervention_example_3panel.py` — dual-write
- Modify: `scripts/figures/plot_importance_decay_grid.py` (or equivalent)
  - Drop PNG emission, keep PDF only
- Create: `scripts/paper/_paper_repo.py` — constant
  `PAPER_REPO = Path("/Users/brian/src/tabular_embedding_paper")`
  and helper `paper_figure_path(section, name)`. Single source of
  truth for dual-write.
- Create: `docs/reports/2026-04-19-paper-refresh-number-diff.md` —
  author-facing report of numbers that moved.

**Paper repo files being refreshed:**

- `figures/4_results/intervention_distributions.pdf`
- `figures/4_results/intervention_example_3panel.pdf`
- `figures/4_results/row_intervention_figure.pdf`
- `figures/4_results/importance_decay_grid.pdf` (PNG → PDF)
- `figures/4_results/geometric_vs_concept.pdf` (sanity regen)
- `figures/A_appendix/layerwise_cka_appendix_*.pdf` (sanity regen)
- `figures/A_appendix/layerwise_depth_all_models_combined.pdf` (sanity regen)
- `tables/section4_summary.tex` (from renamed generator)

**Paper repo files being removed (redundant):**

- `figures/6_appendix/layerwise_cka_appendix.png` (A_appendix has PDFs)
- `figures/6_appendix/layerwise_depth_all_models_combined.png` (same)
- `figures/4_results/domain_reconstruction.png` (D_appendix has PDF)
- `figures/4_results/feature_selectivity.png` (D_appendix has PDF)

---

## Phase 1: Baseline snapshot

Goal: Record what the paper currently contains before anything changes.
Enables the Phase 7 diff report.

### Task 1: Snapshot current paper artefacts

**Files:**
- Create: `output/paper_refresh_baseline_2026-04-19/` (scratch dir for comparison)

- [ ] **Step 1: Snapshot current generated tables and figures for later diff**

```bash
cd /Volumes/Samsung2TB/src/tabular_embeddings
mkdir -p output/paper_refresh_baseline_2026-04-19/tables
mkdir -p output/paper_refresh_baseline_2026-04-19/figures
cp /Users/brian/src/tabular_embedding_paper/tables/*.tex \
   output/paper_refresh_baseline_2026-04-19/tables/
cp -r /Users/brian/src/tabular_embedding_paper/figures \
   output/paper_refresh_baseline_2026-04-19/
cp scripts/tables/table4/ablation_summary.tex \
   output/paper_refresh_baseline_2026-04-19/tables/section4_summary_pre.tex
```

- [ ] **Step 2: Verify snapshot complete**

```bash
ls output/paper_refresh_baseline_2026-04-19/tables/
ls output/paper_refresh_baseline_2026-04-19/figures/
```

Expected: all 5 tables and all figure subdirs present.

- [ ] **Step 3: Commit**

```bash
git add docs/plans/2026-04-19-paper-refresh.md
git commit -m "docs: plan paper refresh against 2026-04-19 sweep pull"
```

Note: the snapshot dir is scratch; do not commit its contents.

---

## Phase 2: Dual-write helper + rename table4

Goal: Establish the single source of truth for paper-repo output paths,
then rename `table4/` → `ablation_summary/` and point its data source
at the refreshed `ablation_sweep_tols/`.

### Task 2: Add paper-repo path helper

**Files:**
- Create: `scripts/paper/_paper_repo.py`

- [ ] **Step 1: Write helper module**

```python
"""Canonical path resolver for the paper repo.

Generators in scripts/paper/ and scripts/figures/ dual-write their
final artefacts to both output/paper_figures/<section>/ (working copy)
and the paper repo at PAPER_REPO/figures/<section>/.
"""
from pathlib import Path

PAPER_REPO = Path("/Users/brian/src/tabular_embedding_paper")
PAPER_FIGURES = PAPER_REPO / "figures"
PAPER_TABLES = PAPER_REPO / "tables"


def paper_figure_path(section: str, name: str) -> Path:
    """Resolve absolute path for a figure in the paper repo.

    Args:
        section: One of "1_intro", "4_results", "A_appendix",
                 "B_appendix", "C_appendix", "D_appendix",
                 "E_appendix", "F_appendix", "G_observational".
        name: Filename with .pdf extension.

    Returns:
        Absolute Path; parent dir guaranteed to exist.
    """
    p = PAPER_FIGURES / section / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def paper_table_path(name: str) -> Path:
    """Resolve absolute path for a .tex table in the paper repo."""
    p = PAPER_TABLES / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
```

- [ ] **Step 2: Smoke-test the import**

```bash
cd /Volumes/Samsung2TB/src/tabular_embeddings
python -c "from scripts.paper._paper_repo import paper_figure_path, paper_table_path; \
           print(paper_figure_path('4_results', 'test.pdf')); \
           print(paper_table_path('test.tex'))"
```

Expected: Prints two absolute paths under
`/Users/brian/src/tabular_embedding_paper/…`. No exception.

- [ ] **Step 3: Commit**

```bash
git add scripts/paper/_paper_repo.py
git commit -m "feat: add paper-repo path helper for dual-write"
```

### Task 3: Rename table4 → ablation_summary

**Files:**
- Rename: `scripts/tables/table4/` → `scripts/tables/ablation_summary/`
- Rename inside: `table4.py` → `ablation_summary.py`

- [ ] **Step 1: Rename directory and file with git**

```bash
cd /Volumes/Samsung2TB/src/tabular_embeddings
git mv scripts/tables/table4 scripts/tables/ablation_summary
git mv scripts/tables/ablation_summary/table4.py \
       scripts/tables/ablation_summary/ablation_summary.py
```

- [ ] **Step 2: Search for references to the old path and update**

```bash
grep -rn "scripts.tables.table4\|scripts/tables/table4" \
    --include='*.py' --include='*.md' --include='*.tex' .
```

Expected output: none after the rename. If any hits appear, update the
reference using Edit — replace `scripts.tables.table4.table4` with
`scripts.tables.ablation_summary.ablation_summary`, and
`scripts/tables/table4/` with `scripts/tables/ablation_summary/`.

- [ ] **Step 3: Update the module docstring and usage line**

Edit `scripts/tables/ablation_summary/ablation_summary.py`:

Old docstring line 9:
```
    python -m scripts.tables.table4.table4
```

New:
```
    python -m scripts.tables.ablation_summary.ablation_summary
```

Old line 3:
```
Table 4: Mean gap closed when ablating unmatched concepts, by strong model.
```

New:
```
Section 4 ablation summary: mean gap closed when ablating unmatched
concepts, by strong model. Rendered as Table 2 in the paper draft.
```

- [ ] **Step 4: Verify module still runs with no behavioural change yet**

```bash
python -m scripts.tables.ablation_summary.ablation_summary
```

Expected: no exception. Output is still based on old `ablation_sweep/`
(the data swap happens in Task 4). Capture the baseline output:

```bash
cp scripts/tables/ablation_summary/ablation_summary.tex \
   output/paper_refresh_baseline_2026-04-19/tables/section4_summary_renamed_pre_swap.tex
```

- [ ] **Step 5: Commit**

```bash
git add -u  # picks up renames and docstring
git commit -m "refactor: rename table4 -> ablation_summary (paper Table 2)"
```

### Task 4: Swap ablation_summary data source to refreshed sweep

**Files:**
- Modify: `scripts/tables/ablation_summary/ablation_summary.py` lines 20-22

- [ ] **Step 1: Edit data source constants**

Old:
```python
SWEEP_DIR = PROJECT_ROOT / "output" / "ablation_sweep"
RANDOM_DIR = PROJECT_ROOT / "output" / "ablation_sweep_random"
OUTPUT_TEX = Path(__file__).parent / "ablation_summary.tex"
```

New:
```python
from scripts.paper._paper_repo import paper_table_path

SWEEP_DIR = PROJECT_ROOT / "output" / "ablation_sweep_tols"
RANDOM_DIR = PROJECT_ROOT / "output" / "ablation_sweep_random"
OUTPUT_TEX = Path(__file__).parent / "ablation_summary.tex"
PAPER_OUTPUT_TEX = paper_table_path("section4_summary.tex")
```

- [ ] **Step 2: Find the write site and add dual-write**

Read the end of `ablation_summary.py` to find where `OUTPUT_TEX` is
written (likely `OUTPUT_TEX.write_text(tex)` or `with open(OUTPUT_TEX,
"w")`). Immediately after the write, add:

```python
PAPER_OUTPUT_TEX.write_text(tex_str)
print(f"  → also wrote {PAPER_OUTPUT_TEX}")
```

(Use the same variable name that holds the rendered tex string. If the
script writes inside a with-block, dedent `PAPER_OUTPUT_TEX.write_text`
to match.)

- [ ] **Step 3: Run the generator**

```bash
cd /Volumes/Samsung2TB/src/tabular_embeddings
python -m scripts.tables.ablation_summary.ablation_summary
```

Expected:
- Terminal prints a trained-entry count and writes two paths.
- `scripts/tables/ablation_summary/ablation_summary.tex` exists and
  mtime is now.
- `/Users/brian/src/tabular_embedding_paper/tables/section4_summary.tex`
  exists and mtime is now.

- [ ] **Step 4: Diff the new table against the baseline**

```bash
diff output/paper_refresh_baseline_2026-04-19/tables/section4_summary_pre.tex \
     scripts/tables/ablation_summary/ablation_summary.tex | head -80
```

Record the diff output in the Phase 7 report. Numbers moving is
expected; the structural shape should match.

- [ ] **Step 5: Commit (code repo only)**

The paper-repo file `section4_summary.tex` belongs to a separate git
repo and is committed in Phase 7 Task 15.

```bash
cd /Volumes/Samsung2TB/src/tabular_embeddings
git add scripts/tables/ablation_summary/ablation_summary.py \
        scripts/tables/ablation_summary/ablation_summary.tex
git commit -m "feat(paper): swap ablation_summary to ablation_sweep_tols + dual-write"
```

---

## Phase 3: Rerun figures that directly consume refreshed dirs

Three generators point at the refreshed sweep data:
`plot_intervention_distributions.py`, `plot_intervention_example_3panel.py`,
and `scripts/paper/sec4/compute_row_intervention_data.py →
row_intervention_figure.py`.

### Task 5: Dual-write plot_intervention_distributions.py

**Files:**
- Modify: `scripts/figures/plot_intervention_distributions.py`

- [ ] **Step 1: Read current output paths in the script**

```bash
grep -n "savefig\|output_path\|output_dir\|paper_figures\|\.pdf\|\.png" \
    scripts/figures/plot_intervention_distributions.py
```

Note the existing savefig calls.

- [ ] **Step 2: Add dual-write**

At the top of the file, add:
```python
from scripts.paper._paper_repo import paper_figure_path
```

At each `plt.savefig(<existing_path>.pdf, ...)` site, add immediately after:
```python
plt.savefig(paper_figure_path("4_results", "intervention_distributions.pdf"),
            bbox_inches="tight")
```

Delete any `plt.savefig(..., ".png")` calls. PDFs only per user
directive.

- [ ] **Step 3: Run it**

```bash
python scripts/figures/plot_intervention_distributions.py
```

Expected: exits 0; prints two save paths; the paper-repo file
`/Users/brian/src/tabular_embedding_paper/figures/4_results/intervention_distributions.pdf`
has mtime = now.

- [ ] **Step 4: Eyeball the figure**

```bash
open /Users/brian/src/tabular_embedding_paper/figures/4_results/intervention_distributions.pdf
```

Expected: renders without error, axis ranges look sensible, model
labels present. No visual validation beyond smoke test.

- [ ] **Step 5: Commit**

```bash
git add scripts/figures/plot_intervention_distributions.py
git commit -m "feat(paper): dual-write intervention_distributions to paper repo"
```

### Task 6: Dual-write plot_intervention_example_3panel.py

**Files:**
- Modify: `scripts/figures/plot_intervention_example_3panel.py`

- [ ] **Step 1: Add import and dual-write at savefig site**

Top of file:
```python
from scripts.paper._paper_repo import paper_figure_path
```

After each existing savefig call for the main figure:
```python
plt.savefig(paper_figure_path("4_results", "intervention_example_3panel.pdf"),
            bbox_inches="tight")
```

Remove any `.png` savefig calls.

- [ ] **Step 2: Run**

```bash
python scripts/figures/plot_intervention_example_3panel.py
```

Expected: the 3-panel PDF is written both to the local output dir and
`/Users/brian/src/tabular_embedding_paper/figures/4_results/intervention_example_3panel.pdf`.

- [ ] **Step 3: Eyeball**

```bash
open /Users/brian/src/tabular_embedding_paper/figures/4_results/intervention_example_3panel.pdf
```

Expected: three panels render; the walking example (carte_vs_mitra,
credit-g, row 325) resolves consistently with `row_intervention_figure`
in Task 7.

- [ ] **Step 4: Commit**

```bash
git add scripts/figures/plot_intervention_example_3panel.py
git commit -m "feat(paper): dual-write intervention_example_3panel to paper repo"
```

### Task 7: Regenerate row_intervention_figure (sec4)

**Files:**
- Modify: `scripts/paper/sec4/compute_row_intervention_data.py`
- Modify: `scripts/paper/sec4/row_intervention_figure.py`

- [ ] **Step 1: Run compute step**

`compute_row_intervention_data.py` already reads refreshed
`ablation_figure_data/` and `transfer_figure_data/` and writes
`row_intervention_data.json`. No changes required there.

```bash
python scripts/paper/sec4/compute_row_intervention_data.py
```

Expected: prints per-feature ablation and transfer steps, saves to
`scripts/paper/sec4/row_intervention_data.json`.

- [ ] **Step 2: Add dual-write to row_intervention_figure.py**

Top of file, add:
```python
from scripts.paper._paper_repo import paper_figure_path
```

Find the `plt.savefig(...)` site and add, immediately after:
```python
plt.savefig(paper_figure_path("4_results", "row_intervention_figure.pdf"),
            bbox_inches="tight")
```

- [ ] **Step 3: Render the figure**

```bash
python scripts/paper/sec4/row_intervention_figure.py
```

Expected: PDF written to both
`scripts/paper/sec4/row_intervention_figure.pdf` and
`/Users/brian/src/tabular_embedding_paper/figures/4_results/row_intervention_figure.pdf`.

- [ ] **Step 4: Eyeball**

```bash
open /Users/brian/src/tabular_embedding_paper/figures/4_results/row_intervention_figure.pdf
```

- [ ] **Step 5: Commit**

```bash
git add scripts/paper/sec4/row_intervention_figure.py \
        scripts/paper/sec4/row_intervention_data.json \
        scripts/paper/sec4/row_intervention_figure.pdf
git commit -m "feat(paper): regen row_intervention_figure from refreshed sweep data"
```

---

## Phase 4: Sanity-regen unchanged-data figures

User asked to rerun these because they haven't been run recently.
Expect numbers to match the paper's existing values (within
floating-point). If they move, flag in the Phase 7 report.

### Task 8: Rerun fig_geometric_vs_concept

**Files:**
- Modify: `scripts/paper/sec4/fig_geometric_vs_concept.py` — add dual-write

- [ ] **Step 1: Add paper-repo dual-write**

Top of file:
```python
from scripts.paper._paper_repo import paper_figure_path
```

After existing `plt.savefig(...)`:
```python
plt.savefig(paper_figure_path("4_results", "geometric_vs_concept.pdf"),
            bbox_inches="tight")
```

- [ ] **Step 2: Run**

```bash
python scripts/paper/sec4/fig_geometric_vs_concept.py
```

- [ ] **Step 3: Diff against baseline**

```bash
cmp -s output/paper_refresh_baseline_2026-04-19/figures/4_results/geometric_vs_concept.pdf \
       /Users/brian/src/tabular_embedding_paper/figures/4_results/geometric_vs_concept.pdf \
    && echo "identical" || echo "differs — inspect"
```

Expected: either identical, or cosmetic differences only. Open both
PDFs side-by-side if `differs`.

- [ ] **Step 4: Commit**

```bash
git add scripts/paper/sec4/fig_geometric_vs_concept.py
git commit -m "feat(paper): dual-write geometric_vs_concept to paper repo"
```

### Task 9: Rerun fig_per_model_layers

**Files:**
- Modify: `scripts/paper/appendix_a/fig_per_model_layers.py`

- [ ] **Step 1: Add dual-write for each of the 7 per-model PDFs and the combined PDF**

Top of file:
```python
from scripts.paper._paper_repo import paper_figure_path
```

At each model's savefig site (there are 7 models + 1 combined):
```python
plt.savefig(
    paper_figure_path("A_appendix", f"layerwise_cka_appendix_{model}.pdf"),
    bbox_inches="tight",
)
```

And for the combined figure:
```python
plt.savefig(
    paper_figure_path("A_appendix", "layerwise_depth_all_models_combined.pdf"),
    bbox_inches="tight",
)
```

- [ ] **Step 2: Run**

```bash
python scripts/paper/appendix_a/fig_per_model_layers.py
```

Expected: 8 PDFs overwritten in
`/Users/brian/src/tabular_embedding_paper/figures/A_appendix/`. All 7
model names in the list `{carte, hyperfast, mitra, mitra_regressor,
tabdpt, tabicl, tabpfn, tabula8b}` should be covered. Missing files →
flag in Phase 7.

- [ ] **Step 3: Diff mtimes**

```bash
ls -la /Users/brian/src/tabular_embedding_paper/figures/A_appendix/ | grep -v ^d
```

Expected: 8 PDFs with mtime = now.

- [ ] **Step 4: Commit**

```bash
git add scripts/paper/appendix_a/fig_per_model_layers.py
git commit -m "feat(paper): dual-write per-model layerwise_cka figures"
```

---

## Phase 5: Purge PNGs and convert leftovers to PDF

User directive: PDFs only. Clean up the remaining PNGs in the paper
repo and make sure their generators produce PDF going forward.

### Task 10: Regenerate importance_decay_grid as PDF

**Files:**
- Modify: generator for `importance_decay_grid.pdf` (exact name TBD
  at run time — most likely
  `scripts/figures/plot_ablation_grid.py` or a sibling; grep to confirm)

- [ ] **Step 1: Locate the generator**

```bash
grep -rln "importance_decay_grid\|importance-decay-grid" scripts/
```

If multiple matches, pick the one whose output dir is under
`output/paper_figures/4_results/` or similar, and which saves with
`.png`. Record the path.

- [ ] **Step 2: Switch emission from PNG to PDF and add dual-write**

Edit the identified script:
- Replace `.png` extension in savefig with `.pdf`
- Add top-of-file import `from scripts.paper._paper_repo import paper_figure_path`
- After savefig, add:
```python
plt.savefig(paper_figure_path("4_results", "importance_decay_grid.pdf"),
            bbox_inches="tight")
```

- [ ] **Step 3: Run**

```bash
python <path-to-that-script>
```

- [ ] **Step 4: Delete the stale PNG from paper repo**

```bash
rm /Users/brian/src/tabular_embedding_paper/figures/4_results/importance_decay_grid.png
```

- [ ] **Step 5: Verify paper section still compiles**

```bash
cd /Users/brian/src/tabular_embedding_paper
grep -n importance_decay_grid sections/4_results.tex
```

If the .tex names the file with an explicit `.png` extension, change
it to `.pdf` (or remove the extension so LaTeX picks the PDF).

- [ ] **Step 6: Commit (code side)**

```bash
cd /Volumes/Samsung2TB/src/tabular_embeddings
git add <path-to-regenerator>
git commit -m "feat(paper): emit importance_decay_grid as PDF + dual-write"
```

### Task 11: Purge redundant 6_appendix PNGs

**Files:**
- Delete (paper repo):
  - `figures/6_appendix/layerwise_cka_appendix.png`
  - `figures/6_appendix/layerwise_depth_all_models_combined.png`

- [ ] **Step 1: Confirm A_appendix has canonical PDFs**

```bash
ls /Users/brian/src/tabular_embedding_paper/figures/A_appendix/*.pdf
```

Expected: 7 per-model PDFs + `layerwise_depth_all_models_combined.pdf`
all present (regenerated in Task 9).

- [ ] **Step 2: Check if any .tex references figures/6_appendix/**

```bash
cd /Users/brian/src/tabular_embedding_paper
grep -rn "6_appendix\|6 appendix" sections/
```

Expected: no references. If any exist, switch them to
`figures/A_appendix/…` before deleting the files.

- [ ] **Step 3: Delete (paper repo)**

```bash
cd /Users/brian/src/tabular_embedding_paper
rm figures/6_appendix/layerwise_cka_appendix.png
rm figures/6_appendix/layerwise_depth_all_models_combined.png
rmdir figures/6_appendix  # only if empty
```

- [ ] **Step 4: Commit (paper repo)**

(Deferred to Phase 7 aggregate paper-repo commit.)

### Task 12: Purge redundant 4_results PNGs

**Files:**
- Delete (paper repo):
  - `figures/4_results/domain_reconstruction.png`
  - `figures/4_results/feature_selectivity.png`

- [ ] **Step 1: Confirm D_appendix has PDFs for both**

```bash
ls /Users/brian/src/tabular_embedding_paper/figures/D_appendix/domain_reconstruction.pdf \
   /Users/brian/src/tabular_embedding_paper/figures/D_appendix/feature_selectivity.pdf
```

Expected: both present.

- [ ] **Step 2: Check .tex references**

```bash
cd /Users/brian/src/tabular_embedding_paper
grep -rn "4_results/domain_reconstruction\|4_results/feature_selectivity" sections/
```

If matched, switch references to the D_appendix PDFs before deleting.

- [ ] **Step 3: Delete (paper repo)**

```bash
cd /Users/brian/src/tabular_embedding_paper
rm figures/4_results/domain_reconstruction.png
rm figures/4_results/feature_selectivity.png
```

(Commit deferred to Phase 7.)

---

## Phase 6: Verify the paper still builds

### Task 13: Full LaTeX build of the paper

**Files:**
- (paper repo only)

- [ ] **Step 1: Build with latexmk**

```bash
cd /Users/brian/src/tabular_embedding_paper
latexmk -pdf -interaction=nonstopmode main.tex 2>&1 | tail -40
```

Expected: "Output written on main.pdf". If undefined references or
missing files surface, address them by either regenerating the missing
artefact (recheck Phase 3–5) or updating the .tex file path.

- [ ] **Step 2: Open the built PDF and skim**

```bash
open /Users/brian/src/tabular_embedding_paper/main.pdf
```

Focus areas:
- Section 4 Results: Table 2 (ablation summary) structure intact
- Section 4 Results: all figures render
- Appendix A: per-model layer plots all present
- Appendix E: per-dataset ablation grids still present (unchanged in
  this pass)

- [ ] **Step 3: Record the build outcome**

If the build succeeds with only the expected number of warnings, move
on. If it fails, fix the cause and retry before Phase 7.

---

## Phase 7: Narrative-number diff report

Produce an author-facing report of what moved so the human can update
quoted numbers in the prose.

### Task 14: Quantify trained-random min_gap asymmetry

**Files:**
- Create: `scripts/tables/ablation_summary/audit_min_gap_asymmetry.py`

The `min_gap=0.01` row-skip in `ablation_sweep_tols/` doesn't apply to
the older `ablation_sweep_random/`. Measure the bias this injects into
the trained-minus-random delta so the diff report can back the choice
to keep the old random (or flag a re-run).

- [ ] **Step 1: Write the audit script**

```python
"""Quantify the min_gap=0.01 row-skip asymmetry between
ablation_sweep_tols (new, with skip) and ablation_sweep_random (old,
without skip).

For each per-row tols npz:
  - count rows where optimal_k == 0 AND gap_closed == 1.0 (the auto-credited
    "agreed" rows)
  - report fraction and per-pair mean gc impact
"""
from pathlib import Path
import numpy as np
from scripts._project_root import PROJECT_ROOT

TOLS = PROJECT_ROOT / "output" / "ablation_sweep_tols"


def audit_pair(pair_dir: Path) -> dict:
    fractions, n_rows_total, n_agreed_total = [], 0, 0
    for npz in sorted(pair_dir.glob("*.npz")):
        try:
            d = np.load(npz, allow_pickle=True)
        except Exception:
            continue
        ok = d["optimal_k"]
        gc = d["gap_closed"]
        agreed = (ok == 0) & (np.isclose(gc, 1.0))
        fractions.append(float(agreed.mean()) if len(agreed) else 0.0)
        n_rows_total += len(agreed)
        n_agreed_total += int(agreed.sum())
    return {
        "pair": pair_dir.name,
        "n_datasets": len(fractions),
        "mean_agreed_fraction": float(np.mean(fractions)) if fractions else 0.0,
        "overall_agreed_fraction": n_agreed_total / n_rows_total if n_rows_total else 0.0,
    }


def main():
    rows = []
    for pair_dir in sorted(TOLS.iterdir()):
        if not pair_dir.is_dir() or "_vs_" not in pair_dir.name:
            continue
        rows.append(audit_pair(pair_dir))

    rows.sort(key=lambda r: -r["overall_agreed_fraction"])
    print(f"{'pair':<30} {'mean_frac':>10} {'overall':>10}")
    for r in rows:
        print(f"{r['pair']:<30} {r['mean_agreed_fraction']:>10.4f} "
              f"{r['overall_agreed_fraction']:>10.4f}")

    max_frac = max(r["overall_agreed_fraction"] for r in rows)
    print(f"\nMax pair-level agreed fraction: {max_frac:.4f}")
    # Worst-case asymmetry on headline: agreed_fraction * (1.0 - random_gc_estimate)
    # If we assume random_gc on agreed rows ≈ random_gc overall (~0.2-0.4),
    # worst-case bias ≈ agreed_fraction * 0.6.
    print(f"Worst-case delta inflation ≈ {max_frac * 0.6:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the audit**

```bash
cd /Volumes/Samsung2TB/src/tabular_embeddings
python scripts/tables/ablation_summary/audit_min_gap_asymmetry.py
```

Expected output: per-pair fractions + a worst-case bias estimate.

**Decision rule:**
- If `Worst-case delta inflation` < 0.01 (1 pp): keep old random; note
  the asymmetry in the diff report with the measured numbers.
- If `Worst-case delta inflation` ≥ 0.01: stop and escalate to Task
  14b (re-run random with current defaults) before finalising the diff
  report. Report this result to the user; don't decide unilaterally.

- [ ] **Step 3: Record the measurement**

Add the script's stdout to the top of
`docs/reports/2026-04-19-paper-refresh-number-diff.md` under a
"Methodology note" heading. Keep the script (not throwaway) — someone
will want to re-audit after any future sweep re-run.

- [ ] **Step 4: Commit**

```bash
git add scripts/tables/ablation_summary/audit_min_gap_asymmetry.py
git commit -m "feat(paper): audit min_gap=0.01 row-skip asymmetry for ablation summary"
```

### Task 14b: (CONTINGENT) Re-run random sweep with current defaults

Skip this task unless Task 14 Step 2 reported inflation ≥ 0.01 (1 pp).

- [ ] **Step 1: Stop and report to user**

Do not silently launch a large distributed job. Report the audit
result to the user with per-pair breakdown and wait for explicit
go-ahead before scheduling the re-run on workers. The launch
procedure follows `memory/feedback_launch_jobs_individually.md` and
the `ablation_sweep.py --output-dir output/ablation_sweep_random_tols`
pattern; once green-lit, the distributed launch details (workers,
ordering, lock files) become a separate mini-plan.

### Task 14c: Extract quoted numbers and candidate new values

**Files:**
- Create: `docs/reports/2026-04-19-paper-refresh-number-diff.md`

- [ ] **Step 1: Pull quoted numbers from the paper .tex**

Grep the paper sections for numeric claims tied to the refreshed data:

```bash
cd /Users/brian/src/tabular_embedding_paper
grep -nE "[0-9]+\.[0-9]+|gap closed|rejection rate|transfer > ablation" \
    sections/4_results.tex sections/C_appendix_concepts.tex
```

Copy each numeric claim and its context into the report.

- [ ] **Step 2: Extract corresponding fresh values**

For each claim, compute or read the corresponding value from the
refreshed generator output. Sources:
- Table values → `scripts/tables/ablation_summary/ablation_summary.tex`
- Per-model transfer numbers → aggregate from
  `output/transfer_global_mnnp90_trained_tols/<pair>/*.npz`
- Correlations (r=0.96, r=0.15, r=0.03) → recompute from refreshed
  sweeps if the source scripts exist; if not, note as "source script
  missing, recompute needed"

- [ ] **Step 3: Write the diff report**

Structure:
```markdown
# Paper refresh — number diff (2026-04-19)

For each row: old quoted number in paper → new value from refreshed
data → source. Integrate into prose at the author's discretion.

## Section 4 (4_results.tex)

| Claim (current prose) | Old | New | Delta | Source |
|-----------------------|-----|-----|-------|--------|
| 89% of strong model's per-row advantage | 0.89 | <X> | <D> | ablation_summary.tex |
| TabICL-v2 gap closed | 0.97 | <X> | <D> | ablation_summary.tex |
| …                                   | …   | …   | …     | …             |

## Section 4 transfer numbers

| Claim | Old | New | Delta | Source |
|-------|-----|-----|-------|--------|

## Open items

- <r-value> correlations: computed from <script> or pending
- Labeling-dependent numbers (f_6, f_11, f_36, f_86, f_92): BLOCKED on
  task #28 (only f_92_v10 has landed as of plan time).
```

- [ ] **Step 4: Commit the report**

```bash
cd /Volumes/Samsung2TB/src/tabular_embeddings
git add docs/reports/2026-04-19-paper-refresh-number-diff.md
git commit -m "docs: paper refresh number diff for author integration"
```

### Task 15: Commit paper-repo artefacts

- [ ] **Step 1: Stage all refreshed paper-repo files**

```bash
cd /Users/brian/src/tabular_embedding_paper
git status
```

Expected dirty files (from Phase 2–5):
- `tables/section4_summary.tex`
- `figures/4_results/intervention_distributions.pdf`
- `figures/4_results/intervention_example_3panel.pdf`
- `figures/4_results/row_intervention_figure.pdf`
- `figures/4_results/importance_decay_grid.pdf` (new)
- `figures/4_results/domain_reconstruction.png` (deleted)
- `figures/4_results/feature_selectivity.png` (deleted)
- `figures/4_results/geometric_vs_concept.pdf`
- `figures/A_appendix/*.pdf` (8 files)
- `figures/6_appendix/*.png` (deleted)

- [ ] **Step 2: Stage files individually (no `git add -A`)**

```bash
cd /Users/brian/src/tabular_embedding_paper
git add tables/section4_summary.tex \
        figures/4_results/intervention_distributions.pdf \
        figures/4_results/intervention_example_3panel.pdf \
        figures/4_results/row_intervention_figure.pdf \
        figures/4_results/importance_decay_grid.pdf \
        figures/4_results/geometric_vs_concept.pdf \
        figures/A_appendix/layerwise_cka_appendix_carte.pdf \
        figures/A_appendix/layerwise_cka_appendix_hyperfast.pdf \
        figures/A_appendix/layerwise_cka_appendix_mitra_regressor.pdf \
        figures/A_appendix/layerwise_cka_appendix_mitra.pdf \
        figures/A_appendix/layerwise_cka_appendix_tabdpt.pdf \
        figures/A_appendix/layerwise_cka_appendix_tabicl.pdf \
        figures/A_appendix/layerwise_cka_appendix_tabpfn.pdf \
        figures/A_appendix/layerwise_cka_appendix_tabula8b.pdf \
        figures/A_appendix/layerwise_depth_all_models_combined.pdf
git add -u figures/4_results/domain_reconstruction.png \
            figures/4_results/feature_selectivity.png \
            figures/6_appendix/layerwise_cka_appendix.png \
            figures/6_appendix/layerwise_depth_all_models_combined.png
```

- [ ] **Step 3: Commit**

```bash
git commit -m "Refresh paper figures and table2 against 2026-04-19 sweep data"
```

Do NOT push. Let the author review locally first.

---

## Phase 8: Labeling-dependent artefacts (BLOCKED)

These tasks cannot run until task #28 (label generation outside this
repo) completes and drops all five of
`f{6,11,36,86,92}_label_{A,Apairing}_v10.json` into
`output/contrastive_examples/mitra/`. As of plan time only f_92 has
landed.

### Task 16: Wait for label snapshots

- [ ] **Step 1: Poll for completeness**

```bash
cd /Volumes/Samsung2TB/src/tabular_embeddings
for f in 6 11 36 86 92; do
  for v in A Apairing; do
    p="output/contrastive_examples/mitra/f_${f}_label_${v}_v10.json"
    [ -f "$p" ] && echo "OK $p" || echo "MISSING $p"
  done
done
```

Expected (when unblocked): 10 "OK" lines. Until then, stop and hand
back to the user.

### Task 17: Regenerate concept_labels_table

**Files:**
- Modify: `scripts/tables/concept_labels_table.py` — add dual-write

- [ ] **Step 1: Add dual-write for `concept_labels_table.tex`**

Top of file:
```python
from scripts.paper._paper_repo import paper_table_path
```

After the existing `with open(OUTPUT_TEX, "w")`:
```python
paper_table_path("concept_labels_table.tex").write_text(
    Path(OUTPUT_TEX).read_text()
)
```

(Or inline the same tex string to both paths — pick the simpler variant
based on current write-site structure.)

- [ ] **Step 2: Run**

```bash
python scripts/tables/concept_labels_table.py
```

- [ ] **Step 3: Diff against baseline**

```bash
diff output/paper_refresh_baseline_2026-04-19/tables/concept_labels_table.tex \
     /Users/brian/src/tabular_embedding_paper/tables/concept_labels_table.tex \
     | head -80
```

- [ ] **Step 4: Commit (code + paper-repo)**

```bash
cd /Volumes/Samsung2TB/src/tabular_embeddings
git add scripts/tables/concept_labels_table.py \
        scripts/tables/concept_labels_table.tex
git commit -m "feat(paper): regen concept_labels_table from v10 labels"
cd /Users/brian/src/tabular_embedding_paper
git add tables/concept_labels_table.tex
git commit -m "Refresh concept_labels_table from v10 labels"
```

### Task 18: Update Phase 7 number-diff report

- [ ] **Step 1: Append per-feature validator accuracy + pairing delta section**

In `docs/reports/2026-04-19-paper-refresh-number-diff.md`, add:

```markdown
## Labeling validator accuracy (10+10 protocol, v10)

| Feature | A old | A new | Apairing old | Apairing new | Delta (pairing-A) |
|---------|-------|-------|--------------|--------------|-------------------|
| f_6  | … | … | … | … | … |
| f_11 | … | … | … | … | … |
| f_36 | … | … | … | … | … |
| f_86 | … | … | … | … | … |
| f_92 | … | … | … | … | … |

Mean validator acc across 5 features: old <X> → new <Y>.
Pairing delta range: old "+0.08 to +0.24" → new <range>.
```

Fill from the v10 JSON files.

- [ ] **Step 2: Commit**

```bash
cd /Volumes/Samsung2TB/src/tabular_embeddings
git add docs/reports/2026-04-19-paper-refresh-number-diff.md
git commit -m "docs: add labeling validator numbers to refresh diff report"
```

---

## Completion criteria

- `scripts/tables/ablation_summary/` exists (renamed from table4).
- `ablation_summary.py` reads `ablation_sweep_tols/` and writes to both
  local and paper-repo paths.
- Paper repo files under `figures/4_results/`, `figures/A_appendix/`,
  and `tables/section4_summary.tex` all have mtimes ≥ plan start.
- No `.png` files remain in the paper repo.
- `main.pdf` builds cleanly in the paper repo.
- `docs/reports/2026-04-19-paper-refresh-number-diff.md` enumerates
  every moved number with old → new values.
- Phase 8 tasks either completed (if task #28 has landed) or explicitly
  recorded as BLOCKED in a session-end note.
