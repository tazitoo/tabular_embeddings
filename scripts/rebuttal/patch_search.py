#!/usr/bin/env python3
"""App F.3: search input-level suppression patches, selectively, and read out the effect.

For a concept c = (donor, feat_id) accepted into a recipient, find a small edit to the
donor's input row that lowers c's activation while leaving the other k-1 concepts
accepted at that row alone, and report what that does to the recipient's prediction.

Design decisions this encodes (each settled by measurement earlier):

  patch site   Donor rows. The concept is a donor SAE feature; the recipient has no
               such feature.
  search space (column, value) pairs, values drawn ONLY from that column's observed
               support. No contrastive-example CSVs: they exist for 2 of 335 concepts,
               and their non-firing contrast requirement disqualifies the 76 always-on
               concepts outright (firing density >= 0.967) even though those have ample
               dynamic range to suppress (mean activation ~1/3 of max).
  columns      Ranked by rank-correlation with the concept's CACHED activation -- free,
               no forward passes, and better targeted than contrast separation.
  selectivity  delta_r = sum_j sign_j * a_j * v_j * std_w is LINEAR in a_j (verified to
               3.1e-05 by validate_delta_reconstruction.py). If only a_c moves, the
               delta change is exactly c's term and the recipient effect is attributable
               to c. If others move, it is a mixture. So disturbance to the other k-1 is
               recorded for every candidate, and optionally enforced as a CONSTRAINT
               (not a penalty with a lambda to tune).
  in-sample    SAE reconstruction error, against the real-row range from
               sae_insample_null.py. A patch that suppresses c by leaving the region the
               dictionary can represent is the search exploiting the objective, not
               evidence about c -- it counts as "no qualifying patch found".
  readout      Counterfactual delta rebuilt from the CORPUS activation scaled by the
               measured suppression ratio, so published deltas stay intact and only a
               dimensionless ratio comes from re-extraction.
  rows         Sampled across the accepted rows. sae_test rows are ordered/clustered
               (first-16 ||mean||=8.65 vs 1.17 over 200), so "first N" is a biased draw.

tabdpt is PARKED, not excluded. Its corpus draw is unreproducible, so its recipient
effects carry extra uncertainty (substituting a reproducible draw moves the injected
delta ~59%, vs 0.13% for a deterministic donor). Its donor-side patches are unaffected:
whether an input edit suppresses the concept does not depend on the corpus draw.

The canonical run takes NO arguments -- the defaults ARE the full process:

    python -m scripts.rebuttal.patch_search

Every flag below is a testing knob or a shard control for splitting work across hosts.
"""
import argparse
import glob
import json
import math
import os
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import (
    SPLITS_PATH, build_tail, get_extraction_layer_taskaware, load_dataset_context,
    load_norm_stats, load_sae, load_test_embeddings, batched_intervention,
)

FWD = PROJECT_ROOT / "output" / "rebuttal" / "forward_deltas"
IMPORTANCE = PROJECT_ROOT / "output" / "perrow_importance"
ATOMS = PROJECT_ROOT / "output" / "transfer_caches" / "global_trained"
EXTRACT_SEED = 13
# Close enough to the target to stop. Mirrors transfer_sweep_v2's gc_tolerance=0.99, which
# breaks its greedy once gap-closed reaches 0.99 rather than spending steps on the last
# sliver. Fixed, not a flag: it is a property of the convention, not something to tune per
# run. Here the target is the recipient's ORIGINAL pre-transfer prediction, so reversal
# >= this means the patch has undone the transfer at that row.
REVERSAL_TOLERANCE = 0.99
# The unmodified row rides in every batch and must reproduce to this. Same threshold the
# old runtime independence check used, kept so the number means the same thing.
CANARY_TOL = 1e-4
# A concept below this is not active, so a relative shift in it says nothing about whether
# we disturbed it -- and |da|/|a| on a near-zero baseline is what produced a reported
# 198,990,863%. Such concepts are dropped from the collateral, not floored.
ACTIVE_FLOOR = 1e-3

# Exponent on each term of the objective. Defaults reproduce
# drop x sqrt(reversal) / ((1+blast) x (1+recon_excess)).
#
# The reversal exponent is the live one, and its effect INVERTED when the crossing guard
# landed. Reversal is now bounded at 1, and on [0,1] a sqrt inflates small values and
# compresses the differences between them -- sqrt(0.19)=0.436 against sqrt(0.40)=0.632 is
# a ratio of 1.45 where linear would be 2.1. So the sqrt makes the search LESS willing to
# trade suppression for recipient movement, which is the opposite of what it was chosen
# for: dampening an unbounded term that was rewarding overshoot. Raising this exponent
# makes the search chase reversal harder.
EXPONENTS = {"drop": 1.0, "reversal": 0.5, "blast": 1.0, "recon": 1.0}
EPS = 1e-7   # the constant already used by _gc and the transfer sweep
# Nothing is excluded by default. tabdpt is PARKED, not dropped: its cached
# embeddings come from an unseeded retrieval draw, so re-extraction yields a
# different draw and the injected delta moves ~59% (vs 0.13% for a deterministic
# donor). That compromises only the tie between a patch's recipient effect and the
# PUBLISHED delta -- the donor-side claim ("editing column X suppresses concept c")
# does not depend on the corpus draw at all and is fully valid for tabdpt.
# Use --park-donors to set it aside explicitly for a given run.
EXCLUDED_DONORS: set[str] = set()

# CARTE is the ONLY recipient excluded, and only from the READOUT. Its tail is REFIT
# per dataset -- CARTETail.from_data "fit CARTE, capture hidden state" -- so a rebuild is
# a different model and the cached July delta lands in an August embedding space. That
# is a space mismatch, not numerical noise, so it cannot be differenced away. Measured
# over the patching universe (gc_drift_sweep.py, 3 arms): full-path prediction drift
# 3.22e-02, the largest of any recipient.
#
# Nothing else is filtered. Every other recipient's tail is frozen pretrained and
# reproduces: tabdpt 0.00e+00, mitra 1.19e-07, tabpfn 6.42e-03, tabicl 1.22e-02
# (tabicl_v2 untested -- its tails failed to build under tfm, and it needs tfm2).
# An earlier allowlist of {mitra, tabdpt} was built on gc medians, which order these
# models wrongly: tabicl's tail reproduces EXACTLY (0.00e+00) yet its gc median is 0.145,
# because gc clamps and divides by a near-zero gap. That excluded good rows for a bad
# reason and cut cells to 1-4 rows per concept.
#
# The DONOR-side patch is never affected: whether an input edit suppresses the concept
# does not involve the recipient tail at all, so carte cells stay patchable.
READOUT_EXCLUDED = {"carte"}

# stratified across donors and both firing-density regimes (--probe only)
PROBE_CONCEPTS = [
    ("tabicl", 96), ("tabicl_v2", 228), ("mitra", 107), ("tabpfn", 56), ("tabicl", 158),
]


# ── extraction ───────────────────────────────────────────────────────────────

def _reseed():
    torch.manual_seed(EXTRACT_SEED); np.random.seed(EXTRACT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(EXTRACT_SEED)


def extract_acts(donor, dataset, X_ctx, y_ctx, X_query, task, device):
    """SAE activations + relative reconstruction error for a batch of query rows.

    The fit must match the one that built the corpus, or the re-extracted activation is
    not comparable to the cached one it is measured against. 04_extract_all_layers passes
    cat_indices for tabpfn and hyperfast, which sets TabPFN's
    categorical_features_indices and changes the preprocessing at the head of the model
    (88 of MIC's 111 columns). Omitting it here fit TabPFN as if every column were
    numeric, so a_start was produced by a differently configured model than a_corpus.
    """
    from models.layer_extraction import extract_all_layers, load_and_fit, sort_layer_names
    _reseed()
    fit_kwargs = {}
    if donor in ("tabpfn", "hyperfast"):
        ci = sorted(column_types(donor, dataset, np.asarray(X_query)))
        if ci:
            fit_kwargs["cat_indices"] = ci
    clf = load_and_fit(donor, X_ctx, y_ctx, task=task, device=device, **fit_kwargs)
    embs = extract_all_layers(donor, clf, X_query, task=task, seed=EXTRACT_SEED)
    names = sort_layer_names(list(embs.keys()))
    idx = min(max(get_extraction_layer_taskaware(donor, dataset), 0), len(names) - 1)
    raw = np.asarray(embs[names[idx]], dtype=np.float32)
    mean, std = load_norm_stats(donor, dataset, device=device)
    sae, _ = load_sae(donor, device=device)
    with torch.no_grad():
        x = (torch.tensor(raw, device=device) - mean) / std
        h = sae.encode(x)
        xh = sae.decode(h)
        rel = (torch.linalg.norm(x - xh, dim=1) /
               torch.linalg.norm(x, dim=1).clamp_min(1e-8)).cpu().numpy()
        return h.cpu().numpy().astype(np.float64), rel


# ── search space ─────────────────────────────────────────────────────────────


def rank_columns(X: np.ndarray, a: np.ndarray, top_k: int) -> list[int]:
    """Columns ranked by |rank correlation| with the concept's activation. No forwards."""
    def rank(v):
        o = np.argsort(np.argsort(v))
        return (o - o.mean()) / (o.std() + 1e-12)
    ra = rank(a)
    scores = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.nanstd(col) < 1e-12:
            scores.append(0.0); continue
        scores.append(abs(float(np.mean(rank(np.nan_to_num(col)) * ra))))
    return [int(j) for j in np.argsort(-np.asarray(scores))[:top_k]]


def _same(a, b) -> bool:
    """Is a proposed value the one already there? Works for strings and for NaN.

    np.isclose(..., equal_nan=True) was used, which raises on raw string labels and
    treats two distinct category codes as equal when they are numerically close. A raw
    edit is an identity change, not a numeric one.
    """
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    fa, fb = float(a), float(b)
    if np.isnan(fa) or np.isnan(fb):
        return np.isnan(fa) and np.isnan(fb)
    return bool(np.isclose(fa, fb))


def _present(v: np.ndarray) -> np.ndarray:
    """Observed values of a column, missing dropped, for either space.

    Preprocessed columns are float and mark missing with NaN. RAW columns may hold
    strings, where np.isnan raises, and mark missing with None as well as NaN. This is
    the one place that difference is handled, so every helper below works unchanged on a
    raw column of category labels and on a float32 code column.
    """
    a = np.asarray(v)
    if a.dtype.kind in "fc":
        return a[~np.isnan(a)]
    return np.array([x for x in a.tolist() if x is not None and x == x], dtype=object)


def is_integral(v: np.ndarray) -> bool:
    """Does this NUMERIC column hold only whole numbers?

    Decides whether the line search may place a value BETWEEN two observed ones. An
    integral column cannot take 3.5, so its grid steps in whole numbers; a column with
    fractional values can, so its grid interpolates freely.

    Checked against the values rather than read off the dtype, because a column stored as
    float can hold only integers and a mis-cast one says the wrong thing either way. The
    test is exact -- no tolerance, no minimum sample, one value or a thousand gives the
    same answer.

    This does NO typing work and never sees a categorical column. Which columns are
    categorical is preprocessing's answer (cat_indices) and is settled before any grid is
    built; those go to candidate_levels and only ever get observed levels. Integrality
    cannot make that distinction anyway -- a category coded 0-4 and a count taking 0-4 are
    the same array -- which is exactly what the old "<=20 unique integers" heuristic got
    wrong in both directions.

    Replaces lattice_step, which tried to recover an ARBITRARY spacing so it could also
    catch half-steps and currency-in-cents. That needed an atol I chose, needed three
    distinct values to see two gaps, and degraded to a guess on sparse columns -- it
    returned "interpolate" for a column that was 99.65% missing. Those cases are also not
    worth catching: a currency column interpolated to a third decimal is an unusual value
    the model can still ingest, not an impossible one like a fabricated category code,
    which is the failure the rule was written for.
    """
    w = _present(v).astype(float)
    if w.size == 0:
        return False
    return bool(np.array_equal(w, np.rint(w)))


def line_search_values(v: np.ndarray, x0: float, direction: float,
                       n_points: int) -> np.ndarray:
    """Values from x0 out to the marginal's edge, for a dense scan of one column.

    The extent is the column's OBSERVED RANGE, not the point where pass-1 gradients stop
    being favourable. Those gradients are first-order main effects measured on the
    unpatched row; using them to bound the scan lets an estimate taken somewhere else veto
    a region the conditioned evaluation never gets to see. Direction is theirs to set,
    extent is not.

    Steps in whole numbers on an integral column and interpolates on a fractional one.
    Snapping to values observed IN X_query instead would cap the scan at 200 distinct
    targets regardless of how fine we want it, since that is all the query set holds.
    """
    w = _present(v).astype(float)
    if w.size == 0 or not np.isfinite(x0):
        return np.array([])
    edge = float(w.max()) if direction > 0 else float(w.min())
    if not np.isfinite(edge) or abs(edge - float(x0)) <= 0:
        return np.array([])
    grid = np.linspace(float(x0), edge, n_points + 1)[1:]     # exclude x0 itself
    if is_integral(w):
        grid = np.rint(grid)
    return np.unique(grid)


def column_histogram(v: np.ndarray, categorical: bool, nbins: int = 20):
    """(representative observed value, count) per occupied bin of the column.

    One object for both types: a categorical column's bins ARE its levels, a numeric
    column's are equal-width bins over the observed range. Only OCCUPIED bins are
    returned, so every representative is a value the column actually takes -- which is
    the in-support rule the search has always had, and it also means no count is ever
    zero, so log freq needs no flooring.

    The representative is the observed value nearest the bin's median, not the bin
    centre, which would be a value that may occur nowhere.
    """
    v = _present(v)
    if v.size == 0:
        return np.array([]), np.array([])
    if categorical:
        u, c = np.unique(v, return_counts=True)
        return u, c
    v = v.astype(float)
    lo, hi = float(v.min()), float(v.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        u, c = np.unique(v, return_counts=True)
        return u, c
    edges = np.linspace(lo, hi, nbins + 1)
    which = np.clip(np.digitize(v, edges) - 1, 0, nbins - 1)
    reps, cnts = [], []
    for b in range(nbins):
        m = which == b
        n = int(m.sum())
        if n == 0:
            continue
        vals = v[m]
        reps.append(float(vals[np.argmin(np.abs(vals - np.median(vals)))]))
        cnts.append(n)
    return np.asarray(reps), np.asarray(cnts)


def step_pool(v: np.ndarray, x0, max_vals: int, categorical: bool, nbins: int = 20):
    """Destinations available from where this row sits, with their step in log freq.

    There is no requested step size and no snapping. The column's histogram, read from
    x0's own bin, IS the menu: every other occupied bin is a destination carrying its own
    dL = |log count(dest) - log count(x0)|. So a step size parameter (step_frac) does not
    exist to be chosen -- it was only ever an artefact of pretending we pick a step when
    we are picking from what the column offers.

    Returns (value, dL) sorted by dL, thinned to max_vals spread across the RANGE of
    available dL rather than clustered: several destinations at similar rarity are
    several near-identical edits and give pass 2 no more choice than one.

    dL == 0 destinations are returned too, and are valid patch values -- they simply
    cannot carry a slope, since the denominator vanishes. Pass 1 drops them; pass 2 does
    not have to.
    """
    reps, cnts = column_histogram(v, categorical, nbins)
    if reps.size == 0:
        return []
    if categorical:
        hit = np.where(reps == x0)[0]
    else:
        try:
            hit = np.array([int(np.argmin(np.abs(reps.astype(float) - float(x0))))])
        except (TypeError, ValueError):
            hit = np.array([], dtype=int)
    if hit.size == 0:
        return []
    L = np.log(cnts.astype(float))
    L0 = L[hit[0]]
    cand = [(reps[b], float(abs(L[b] - L0)))
            for b in range(len(reps)) if b != hit[0]]
    if not cand:
        return []
    cand.sort(key=lambda t: t[1])
    if max_vals is not None and len(cand) > max_vals:
        cand = [cand[i] for i in np.linspace(0, len(cand) - 1, max_vals).astype(int)]
    return cand


def candidate_levels(v: np.ndarray, max_vals: int) -> np.ndarray:
    """Levels spread across a nominal column's MASS, the same way candidate_values is.

    Categorical values are .cat.codes, so quantiling them selects by code magnitude -- an
    arbitrary ordering with nothing to do with which levels are plausible. But the fix is
    not "take the commonest": that makes max_vals mean a different thing per column type,
    a SPREAD over the range for continuous columns and a HEAD of the distribution for
    categorical ones. Pairing an upper-decile value with a modal level is not a pair.

    The operation candidate_values performs is "pick max_vals values at even intervals of
    the column's mass". A nominal column has no value axis to walk, but it has a
    frequency-rank axis, which is the only ordering it does have -- and the same one the
    edit cost uses, -log p(level). So the levels are ordered by frequency, and the same
    interior quantiles are taken over cumulative mass. max_vals then means one thing in
    both paths: how many points are spread across what the column actually contains.

    Coverage rather than head-of-distribution matters because the search needs candidates
    that can MOVE the concept. Restricted to the 6 commonest of 1765 levels it can only
    reach 12% of the mass, and a concept only suppressible by a less common level is out
    of reach by construction.
    """
    v = _present(v)
    if v.size == 0:
        return np.array([])
    u, c = np.unique(v, return_counts=True)
    order = np.argsort(-c, kind="stable")
    u, c = u[order], c[order]
    if max_vals is None or len(u) <= max_vals:
        return u
    cum = np.cumsum(c) / float(c.sum())
    idx = np.clip(np.searchsorted(cum, np.linspace(0.05, 0.95, max_vals)), 0, len(u) - 1)
    picked = list(dict.fromkeys(idx.tolist()))
    for k in range(len(u)):        # collapse top-up, in frequency order
        if len(picked) >= max_vals:
            break
        if k not in picked:
            picked.append(k)
    return u[np.array(picked[:max_vals], dtype=int)]


def candidate_values(v: np.ndarray, max_vals: int,
                     interior: bool = True) -> np.ndarray:
    """Values that actually OCCUR in the column, weighted by how typical they are.

    Three things this has to get right, each of which the naive version got wrong:

    1. Never invent a value. np.quantile interpolates linearly by default, so quantiles
       of ordinal-coded categoricals produce codes like 2.571 that correspond to no
       category -- off-distribution by construction, and a violation of the only
       in-support guard we have now that contrastive rows are gone. method="nearest"
       snaps every proposal onto an observed value.
    2. Quantile the column WITH multiplicity, not its unique values. A column that is
       90% zeros has unique-quantiles spread over rare extremes (0, 98, 249, ... 1000)
       while data-quantiles correctly propose the typical value (0).
    3. Do not force the extremes. linspace(0,1) always includes min and max, biasing the
       search toward the most violent legal edit; interior quantiles avoid that.
    """
    v = _present(v)
    if v.size == 0:
        return np.array([])
    uniq = np.unique(v)
    if max_vals is None or len(uniq) <= max_vals:
        return uniq            # None = the column's entire observed support
    qs = np.linspace(0.05, 0.95, max_vals) if interior else np.linspace(0.0, 1.0, max_vals)
    out = np.unique(np.quantile(v, qs, method="nearest"))
    if len(out) < max_vals:
        # Snapping collapses quantiles onto the same value in skewed columns (90% zeros
        # -> two candidates). Top up with evenly spaced OBSERVED values so the search
        # keeps its coverage without proposing anything that does not occur.
        #
        # Topping up must not OVERSHOOT the cap: `extra` holds max_vals values and the
        # union of it with `out` was returned whole, so a column could come back with 10
        # or 11 candidates when 6 were asked for. That silently inflated the probe budget
        # per column, and unevenly -- skewed columns, the ones that trigger this branch,
        # got the most. Quantile picks are kept first, then extras fill to exactly the cap.
        extra = uniq[np.linspace(0, len(uniq) - 1, max_vals).astype(int)]
        picked = list(dict.fromkeys(out.tolist()))
        for x in extra.tolist():
            if len(picked) >= max_vals:
                break
            if x not in picked:
                picked.append(x)
        out = np.array(sorted(picked))
    return out


class Space:
    """The columns the search edits, and how an edited row becomes a model input.

    Two spaces, one interface:

      preprocessed  cols are the float32 columns the model ingests; materialize is the
                    identity. What the search has always done.
      raw           cols are the ORIGINAL table columns; materialize pushes an edited raw
                    row through the fitted AutoGluon generator, so the model input is
                    produced by the same code that built the corpus rather than assembled
                    by hand.

    Raw is the better space to define a patch in for three reasons the preprocessed space
    cannot give:

      1. A patch becomes model-independent. "Set ROLE_FAMILY to X" is one edit for every
         donor; only the transform differs. In preprocessed space the SAME column is
         categorical for tabpfn and numeric for tabicl, because imputation clears
         cat_indices, so the two donors search different spaces over identical data.
      2. Cross-column consistency is automatic. Editing one preprocessed column while a
         derived column still holds the old value makes a row the pipeline could never
         emit. Transforming from raw cannot produce one.
      3. Categorical typing comes from the raw dtype, which is what the generator itself
         keys on, instead of from cat_indices after the fact.
    """

    def __init__(self, cols, names, cat, materialize):
        self.cols = cols            # list of 1-D arrays, one per column
        self.names = names          # column labels: ints (preprocessed) or names (raw)
        self.cat = cat              # positional indices of categorical columns
        self.materialize = materialize   # list[row values] -> float32 model input
        self.n_cols = len(cols)
        self.n_rows = len(cols[0]) if cols else 0

    def row(self, i):
        return [c[i] for c in self.cols]


def preprocessed_space(model, dataset, X_query):
    """The existing space: edit the matrix the model ingests, no transform."""
    cat = column_types(model, dataset, X_query)
    return Space(cols=[X_query[:, j] for j in range(X_query.shape[1])],
                 names=list(range(X_query.shape[1])), cat=cat,
                 materialize=lambda rows: np.asarray(rows, dtype=np.float32))


def raw_space(model, dataset, row_indices, X_query_pp, splits=None):
    """Edit the ORIGINAL table; the fitted generator produces the model input.

    Refuses to return a space whose transform does not reproduce X_query exactly. If the
    generator rebuilt from raw disagrees with the cached matrix, then every activation,
    every SAE code and every cached delta was computed on different inputs, and a patch
    measured here would not be a patch on the published run. verify_preprocessor_refit
    established this holds for all 204 (model, dataset) pairs; this re-checks the actual
    query rows in use, since that is the slice the search perturbs.
    """
    from data.extended_loader import _load_tabarena_cached_v2
    from data.preprocessing import NAN_SAFE_MODELS, fit_preprocessor

    if splits is None:
        splits = json.loads(SPLITS_PATH.read_text())
    cached = _load_tabarena_cached_v2(dataset)
    if cached is None:
        raise FileNotFoundError(f"no raw TabArena cache for {dataset}")
    X_df, _ = cached
    train_idx = np.array(splits[dataset]["train_indices"])
    X_train_raw = X_df.iloc[train_idx].reset_index(drop=True)
    X_query_raw = X_df.iloc[np.asarray(row_indices)].reset_index(drop=True)

    pre = fit_preprocessor(X_train_raw, nan_safe=model in NAN_SAFE_MODELS)
    got = pre.transform(X_query_raw)
    ok_nan = np.array_equal(np.isnan(got), np.isnan(X_query_pp))
    ok_val = ok_nan and np.array_equal(got[~np.isnan(got)], X_query_pp[~np.isnan(X_query_pp)])
    if not ok_val:
        raise RuntimeError(
            f"{model}/{dataset}: refit generator does not reproduce X_query "
            f"(shape {got.shape} vs {X_query_pp.shape}, nan_pattern_ok={ok_nan}). "
            "Raw-space patching would edit a different input than the corpus used.")

    names = list(X_query_raw.columns)
    cat = {j for j, c in enumerate(names)
           if not pd.api.types.is_numeric_dtype(X_query_raw[c])}
    cols = [X_query_raw[c].to_numpy() for c in names]

    def materialize(rows):
        df = pd.DataFrame({c: [r[j] for r in rows] for j, c in enumerate(names)})
        return pre.transform(df.astype(X_query_raw.dtypes.to_dict()))

    return Space(cols=cols, names=names, cat=cat, materialize=materialize)


def column_types(model: str, dataset: str, X: np.ndarray) -> set[int]:
    """Categorical column indices, exactly as THIS model's preprocessing declared them.

    A "standard step" is meaningless for a categorical -- the values are pandas .cat.codes
    (preprocessing._df_to_float32), assigned by category order and carrying no magnitude,
    so moving code 2 to 2.5 names no category and 2->3 is not a step of any size. Those
    columns get alternative observed LEVELS instead of a gradient.

    Which columns those are is NOT ours to decide. The patch edits X_query, which IS the
    per-model preprocessed matrix the model ingests, so the only classification that
    produces matched predictions is the one that pipeline used. This previously ANNOTATED
    the cache with a "<=20 unique integer values" heuristic, which disagreed with
    preprocessing in both directions:

      MIC/tabpfn      cache declares 88 categorical (the layout is 23 numeric then 88
                      categorical); the heuristic called it 103, reclassifying 15 columns
                      that AutoGluon parsed as NUMERIC and that the model ingests as
                      numeric.
      Amazon/tabicl   cache declares 0, and the heuristic agreed -- but only because the
                      columns are ID codes with cardinality far above 20.

    Note that cat_indices is model-dependent BY CONSTRUCTION, and correctly so:
    _preprocess_autogluon median-imputes for TabICL and Mitra and then clears cat_indices,
    because after imputation the values are no longer category codes. TabPFN and TabDPT
    (nan_safe) keep them. So the same dataset is 88 categorical for tabpfn and 0 for
    tabicl. That is not an inconsistency to repair by borrowing the nan_safe model's
    indices -- it is what each model actually receives, and borrowing would impose a
    structure the model never sees.

    No fallback: if the cache cannot be read, the classification is unknown and the run
    must fail rather than silently searching the wrong value space.
    """
    from data.preprocessing import load_preprocessed, CACHE_DIR
    return set(load_preprocessed(model, dataset, CACHE_DIR).cat_indices or [])


def nearest_observed(v: np.ndarray, target: float) -> float:
    """Snap a proposed value onto the nearest value the column actually takes."""
    v = _present(v)
    if v.size == 0:
        return float(target)
    return float(v[np.argmin(np.abs(v - target))])


def edit_distance(v: np.ndarray, old, new, categorical: bool = False) -> float:
    """How costly is this edit, in the column's own terms.

    CATEGORICAL columns get the surprisal of the level moved TO, -log p(new), in nats.
    The values are pandas .cat.codes -- nominal labels whose order is assignment order --
    so |code_new - code_old| is arithmetic on labels and any scale built from it is
    meaningless. Measured on MIC, dividing by the code IQR made the identical operation
    (flip a binary column) cost between 1.0 and 9.1 across columns purely on class
    balance, and made a 15-level column's level change cost 0.118, i.e. cheaper than any
    binary flip. Surprisal uses only the frequency of the destination level, so it needs
    no ordering: moving to a rarer level is a bigger edit, moving to a common one is a
    small one.

    CONTINUOUS columns keep |new - old| / IQR, with std as the fallback when the IQR
    degenerates. There the order is real and the magnitude means something.

    The two are NOT commensurable, and summing them across a mixed row mixes nats with
    IQR-multiples. That is tolerated deliberately: edit distance never enters the
    objective, and it is the SECOND key of the tie-break behind the column count, which
    is well defined. It ranks candidates that are already tied on the objective and on
    suppression; it does not decide what a good patch is.

    Must always return a finite number. The tie-break picks with min(), and a nan
    compares False against everything, so a nan candidate is neither smaller nor larger
    than its rivals and min() keeps whichever it saw first -- the tie-break then silently
    does nothing. In the v3 sweep that hit 1.9% of rows via `old` being NaN: the original
    cell is MISSING, so the edit fills a hole rather than moving a value.
    """
    v = _present(v)
    if v.size == 0:
        return 0.0
    if categorical:
        # Frequency of the destination level. An unseen level floors at one occurrence,
        # so it is the most expensive edit available rather than an infinite one.
        p = max(int(np.count_nonzero(v == new)), 1) / float(v.size)
        d = -math.log(p)
        return float(d) if np.isfinite(d) else 0.0
    scale = float(np.subtract(*np.percentile(v, [75, 25])))
    if scale <= 1e-12:
        scale = float(np.std(v)) or 1.0
    ref = float(old)
    if not np.isfinite(ref):
        ref = float(np.median(v))
    d = abs(float(new) - ref) / max(scale, 1e-9)
    return float(d) if np.isfinite(d) else 0.0


# ── the search ───────────────────────────────────────────────────────────────

def make_evaluator(donor, dataset, X_ctx, y_ctx, X_query, space, task, device, row,
                   a_ref, tol=CANARY_TOL):
    """Evaluate candidate rows -> (activations, recon_rel), one entry per candidate.

    ONE path. There is no batched-vs-per-candidate branch and no fallback: a run that
    quietly switches evaluation method produces numbers whose meaning depends on a
    decision nobody saw. The previous branch sent mitra down a per-candidate path costing
    ~48 forwards per greedy step, on the strength of a check that measured the wrong
    property, and it took a whole sweep before anyone asked why.

    Every batch carries the UNMODIFIED row under test as its first entry, and its
    activation is compared against `a_ref`. That is a canary, not a correction -- it is
    not differenced out, because differencing would absorb exactly the effect we need to
    know about. If the unmodified row does not reproduce, the batch perturbed it, so every
    candidate measured alongside it is suspect and the run stops.

    This is what makes a shared window safe to use at all. The query set grows with the
    candidate count, and that count varies per row: pass 1 probes every column by default,
    which reaches 3,939 appended rows on wide datasets. tabpfn is stable to 48 appended
    rows and shifts 4.99e-02 at 128 -- above tolerance, and 45.9% of its rows exceed 128.
    A fixed "safe" window size cannot be trusted for that, because the threshold is
    model-specific and goes stale silently; checking the actual batch cannot.
    """
    base = X_query
    window = len(base)
    unmodified = space.row(row)

    def evaluate(variants):
        """Candidates are cycled through a window of exactly len(X_query) rows.

        The window NEVER grows. Candidates replace rows inside it and the dataset's own
        rows fill any remainder, so every forward -- the baseline pass, every greedy step,
        every concept -- runs on a query set of the size the corpus was built at. Growing
        it was the defect: pass 1 probed every column by default, reaching 4,139 rows on
        hiva_agnostic where the dataset has 200, and the model then saw a query set twenty
        times anything it was calibrated on.

        Padding SHORT datasets up to 200 was measured and rejected. It is safe for tabicl
        and tabdpt but not in general: tabpfn moves 1.23e-01 and mitra 2.59e+00 when anneal
        is padded 86 -> 200, while both are clean on MIC 162 -> 200. No rule separates
        those cases, so the window is simply the dataset's own size.

        Cost is ceil(n_candidates / (window - 1)) forwards instead of one, and that is the
        honest price of the single forward having produced unusable numbers.
        """
        acts, recons = [], []
        per = window - 1                       # one slot reserved for the canary
        for i in range(0, len(variants), per):
            chunk = list(variants[i:i + per])
            rows = [unmodified] + chunk
            j = 0
            while len(rows) < window:          # fill with the dataset's OWN rows
                rows.append(space.row(j)); j += 1
            Xq = np.asarray(space.materialize(rows), dtype=base.dtype)
            a, rel = extract_acts(donor, dataset, X_ctx, y_ctx, Xq, task, device)
            drift = float(np.abs(a[0] - a_ref).max())
            if drift > tol:
                raise RuntimeError(
                    f"{donor}/{dataset} row {row}: the unmodified row moved {drift:.3e} "
                    f"(tol {tol:.0e}) in a window of {window} holding {len(chunk)} "
                    "candidates. Every candidate measured beside it is unmeasurable.")
            acts.append(a[1:1 + len(chunk)]); recons.append(rel[1:1 + len(chunk)])
        return np.concatenate(acts), np.concatenate(recons)

    return evaluate


def weighted_blast(a_base, a_new, others, recip):
    """Collateral weighted by how much the recipient's prediction depends on each concept.

    Each of the k-1 co-accepted concepts moved by |da_j|/|a_j| -- per concept, relative to
    its own scale, since activations differ by an order of magnitude across features. That
    movement is then weighted by concept j's LOO effect: how far the prediction moves when
    j is removed from the completed delta.

    Unweighted, every concept counts the same. That cannot distinguish a 12% shift in a
    concept the prediction does not depend on -- which changes no outcome -- from a 3%
    shift in one carrying a third of it. Measured on v14's 481 early-stop rows, the
    90th-percentile other concept moved 0.306 against a target that moved 0.349, and
    nothing recorded says whether that collateral sat on concepts that mattered.

    Weighted, it is in the same currency as the rest of the objective: drop and the
    recipient term are both prediction effects, and this becomes "how much prediction
    effect did we disturb" against "how much did we intend to".

    Concepts whose activation is ~0 are excluded rather than floored. A concept that is not
    active carries no signal about whether we disturbed it, and |da|/|a| on a near-zero
    baseline is what produced a reported 198,990,863% -- the reason blast_radius was
    written as a norm in the first place.
    """
    if len(others) == 0 or recip is None or "loo_by_fid" not in recip:
        return None
    b, n = np.asarray(a_base)[others], np.asarray(a_new)[others]
    live = np.abs(b) > ACTIVE_FLOOR
    if not live.any():
        return None
    rel = np.abs(n[live] - b[live]) / np.abs(b[live])
    w = np.array([recip["loo_by_fid"].get(int(f), 0.0) for f in np.asarray(others)[live]])
    tot = float(w.sum())
    if tot <= EPS:                    # nothing here moves the prediction either way
        return 0.0
    return float((rel * w).sum() / tot)


def blast_radius(a_base, a_new, accepted_others):
    """Movement in the OTHER accepted concepts, as one scale-free number.

    Only concepts that fire AND were accepted have a term in delta_r, so they are the
    only ones whose movement can reach the recipient. Restricted to that set, with the
    patched concept excluded (we want it to move).

    A vector norm rather than a per-concept quantile: |da_j|/|a_j| explodes when a
    concept sits near zero -- that is what produced a reported 198,990,863% -- whereas
    a near-zero component contributes near-zero to both numerator and denominator here.
    No quantile means no arbitrary choice of 90 vs 95 vs max, and the whole distribution
    contributes rather than one sampled point.
    """
    if len(accepted_others) == 0:
        return 0.0
    d = a_new[accepted_others] - a_base[accepted_others]
    return float(np.linalg.norm(d) / (np.linalg.norm(a_base[accepted_others]) + EPS))


def reversal(L_transfer, L_target, L_mod, min_interval=1e-3):
    """Where the patch landed between the transfer and the ABLATION of this concept.

    0 = the patch changed nothing; 1 = it reached what ablating this concept alone
    achieves; >1 = it moved further than removing the concept, which is doing more than
    removing it and is the crossing the guard should reject.

    The target used to be the UNTRANSFERRED prediction, which asks one concept to undo a
    whole multi-concept transfer. That is unreachable wherever the concept carries a small
    share, so a perfect patch was filed as an undershoot -- one measured row suppressed the
    concept 90.7% with 3.6% collateral and scored reversal 0.190, which is what it should
    score when the concept carries a fifth of the delta.

    Returns nan when the interval is degenerate. Ablating the concept moves the prediction
    by less than min_interval on a quarter of rows (ceiling_effect p25 = 0.001), and
    normalising there divides by noise -- that is what drove capture_of_ceiling to p90 =
    11.17. Such a row cannot demonstrate anything about the recipient in either direction,
    so it is flagged and the patch is selected on the donor-side terms alone.
    """
    interval = L_target - L_transfer
    if not np.isfinite(interval) or abs(interval) < min_interval:
        return float("nan")
    return float((L_mod - L_transfer) / interval)


def objective(drop_frac, rev, blast, recon_excess=0.0):
    """drop x sqrt(reversal) / ((1 + blast) x (1 + recon_excess)).

    Products and ratios of dimensionless terms, so there are no weights to invent.
    sqrt compresses reversal's range: undoing the whole transfer scores 1.0 while
    removing one concept's share scores ~0.026, and the square root narrows that 38x
    spread to ~6x so the recipient term informs the choice without dictating it.
    `1 + blast` degrades smoothly to `drop x sqrt(rev)` for a clean patch rather than
    blowing up as blast -> 0.

    `recon_excess` is the IN-SAMPLE term: how far the patch inflates the SAE's
    reconstruction error above this row's own baseline, max(0, recon'/recon0 - 1).
    Relative rather than absolute because the baseline level is model-specific (0.12
    for tabpfn, 0.57 for tabicl_v2), and one-sided because reconstructing BETTER than
    the original row is not a problem. Without it the search can suppress the concept
    by pushing the row somewhere the dictionary cannot represent, which is the
    objective being exploited rather than evidence about the concept -- the same
    failure as an off-manifold patch.

    Reversal is not clipped and the sqrt is sign-preserving: a patch that moves the
    recipient opposite to the transfer scores negative, which is correct.
    """
    r = float(rev)
    p = EXPONENTS["reversal"]
    root = math.copysign(abs(r) ** p, r) if np.isfinite(r) else float("nan")
    return float(drop_frac ** EXPONENTS["drop"] * root
                 / ((1.0 + blast) ** EXPONENTS["blast"]
                    * (1.0 + max(0.0, recon_excess)) ** EXPONENTS["recon"]))


def build_recip_shared(donor, recipient, dataset, device):
    """Per-CELL recipient context: the tail, the atoms, and the donor's cached
    activations. Built once per (recipient, dataset) -- the tail does not depend on the
    row, and rebuilding it per row would dominate the cost.

    Returns None when the recipient's readout does not reproduce the transfer (carte),
    so the search runs donor-side only there rather than optimising against a number
    that cannot be trusted.
    """
    if recipient in READOUT_EXCLUDED:
        return None
    zc = np.load(ATOMS / f"{donor}_to_{recipient}.npz", allow_pickle=True)
    V = np.asarray(zc["virtual_atoms"], dtype=np.float64)
    fmap = {int(f): i for i, f in enumerate(np.asarray(zc["feature_ids"]))}
    _, std_w = load_norm_stats(recipient, dataset, device=device)
    std_w = np.asarray(std_w.cpu(), dtype=np.float64)

    sae, _ = load_sae(donor, device=device)
    with torch.no_grad():
        A = sae.encode(torch.tensor(np.asarray(load_test_embeddings(donor)[dataset],
                                               dtype=np.float32),
                                    device=device)).cpu().numpy().astype(np.float64)

    splits = json.loads(SPLITS_PATH.read_text())
    Xtr, ytr, Xq, _, _, task = load_dataset_context(recipient, dataset, splits)
    if ytr.dtype == np.int32:
        ytr = ytr.astype(np.int64)
    layer = get_extraction_layer_taskaware(recipient, dataset=dataset)
    cat_idx = None
    if recipient in ("hyperfast", "tabpfn"):
        from data.preprocessing import load_preprocessed, CACHE_DIR
        # No swallow. An unreadable cache means the recipient tail would be built without
        # cat_indices -- a differently configured model, silently. That is the same defect
        # as the donor-side fit bug, and it cannot be detected downstream: the run does not
        # crash, it just produces predictions from a model we did not intend.
        cat_idx = load_preprocessed(recipient, dataset, CACHE_DIR).cat_indices or None
    _reseed()
    tail = build_tail(recipient, Xtr, ytr, Xq, layer, task, device, cat_indices=cat_idx,
                      target_name=splits.get(dataset, {}).get("target", "target"))
    return {"V": V, "fmap": fmap, "std_w": std_w, "A": A, "tail": tail, "Xq": Xq}


def build_recip(shared, donor, recipient, dataset, npz_path, row, a_re, feat, device):
    """Per-ROW recipient context: this row's accepted atoms, signs, and the transfer's
    own endpoints (L_orig, L_transfer) that `reversal` is scaled by."""
    if shared is None:
        return None
    from scripts.intervention.intervene_lib import (
        SEQUENTIAL_MODELS, batched_ablation_sequential)

    V, fmap, std_w, A = shared["V"], shared["fmap"], shared["std_w"], shared["A"]
    tail, Xq = shared["tail"], shared["Xq"]
    z = np.load(npz_path, allow_pickle=True)
    dd = np.asarray(z["deployed_delta"], dtype=np.float64)[row]
    sel = np.asarray(z["selected_features"])[row]
    fids = [int(f) for f in np.unique(sel[sel >= 0]) if int(f) in fmap]
    if not fids:
        return None
    B = np.stack([V[fmap[f]] * std_w for f in fids])
    c, *_ = np.linalg.lstsq(B.T, dd, rcond=None)   # signs are not cached; recover them

    y = int(np.asarray(z["y_query"])[row])
    pw, pi = np.asarray(z["preds_weak"])[row], np.asarray(z["preds_intervened"])[row]

    def loss(p):
        p = np.asarray(p)
        if p.ndim >= 1 and p.size > 1:
            return float(-np.log(np.clip(p[y], EPS, 1 - EPS)))
        return float((float(p) - float(np.asarray(z["preds_strong"])[row])) ** 2)

    def predict(deltas):
        t = torch.tensor(np.asarray(deltas), dtype=torch.float32, device=device)
        if isinstance(tail, SEQUENTIAL_MODELS):
            return np.asarray(batched_ablation_sequential(tail, Xq[row:row+1], t, query_idx=row),
                              dtype=np.float64)
        return np.asarray(batched_intervention(tail, Xq[row:row+1], t, inject_context=False),
                          dtype=np.float64)

    # The ABLATION TARGET: the delta with THIS concept's term removed and every other
    # concept left at its corpus value. That is what a perfect patch achieves -- remove c,
    # disturb nothing else -- so it is the endpoint the recipient term should be scaled by.
    # Scaling by the untransferred prediction instead asks one concept to undo the whole
    # transfer, which is only reachable when c IS the whole transfer, and files a perfect
    # patch on a low-share concept as an undershoot.
    #
    # Defensible now because it was measured rather than assumed: the per-concept ablations
    # sum to the transfer's effect at 67% of rows (loo_additivity_sweep, median ratio
    # 0.846), so the ablation target is a real share of a decomposable whole. On the 26%
    # redundant rows it is not, which is a reason to flag them, not to rescale them.
    signs = np.sign(c)
    a_corpus = np.array([A[row, f] for f in fids])

    # LOO for EVERY accepted concept, not just the patched one, in the same batched call.
    # Variant i is the delta with concept i's term removed -- the transfer's own delta
    # minus that concept, which is what "ablate it" means. The reconstruction is faithful:
    # validate_delta_reconstruction agreed with the stored deployed_delta to 3.14e-05.
    #
    # The patched concept's entry gives L_ablated, the target the recipient term is scaled
    # by. The others give how much the recipient's prediction DEPENDS on each co-accepted
    # concept, which is what the collateral should be weighted by: a 12% shift in a concept
    # whose LOO is ~0 cannot move the prediction, while a 3% shift in one carrying a third
    # of it can. blast and selectivity_ratio both treat them as equally worth not
    # disturbing, and neither can tell those apart.
    #
    # Marginal contributions are already on disk (step_preds differenced across the
    # transfer's greedy) but they are the effect AT ACCEPTANCE given what was accepted so
    # far -- order-dependent, and it diverges from removal-from-the-completed-set exactly
    # when concepts are correlated. LOO is the one that answers "if this gets disturbed,
    # does the prediction care, given everything else present".
    variants = []
    for i in range(len(fids)):
        keep = np.ones(len(fids)); keep[i] = 0.0
        variants.append((signs * a_corpus * keep) @ B)
    L_loo = [loss(p) for p in predict(np.asarray(variants))]
    L_transfer = loss(pi)
    i_feat = fids.index(feat) if feat in fids else None
    L_ablated = L_loo[i_feat] if i_feat is not None else float("nan")
    # |prediction effect of removing concept j|, the weight for j's disturbance.
    loo_effect = np.array([abs(L - L_transfer) for L in L_loo])

    return {"fids": fids, "B": B, "signs": signs,
            "a_corpus": a_corpus,
            "a_re": {f: float(a_re[f]) for f in fids},   # our own baseline, for ratios
            "predict": predict, "loss": loss,
            "L_orig": loss(pw), "L_transfer": L_transfer, "L_ablated": L_ablated,
            "loo_effect": loo_effect, "loo_by_fid": dict(zip(fids, loo_effect.tolist())),
            "row": int(row)}


def recipient_reversal(recip, acts, feat):
    """Measured reversal for a batch of candidate activation vectors.

    For each candidate: rescale every accepted concept's term by its MEASURED ratio
    (not just the patched concept's -- assuming the others held still is the assumption
    under test), rebuild the delta, and run the recipient. One batched call for the
    whole batch, so a pass costs one recipient forward regardless of candidate count.
    """
    d = []
    for av in acts:
        r_vec = np.array([
            float(av[f] / recip["a_re"][f]) if abs(recip["a_re"][f]) > EPS else 1.0
            for f in recip["fids"]])
        d.append((recip["signs"] * recip["a_corpus"] * r_vec) @ recip["B"])
    preds = recip["predict"](np.asarray(d))
    return [reversal(recip["L_transfer"], recip["L_ablated"], recip["loss"](p))
            for p in preds]


def shift_metrics(a_base, a_new, others, feat):
    """How much did the OTHER accepted concepts move, per concept and relative to their own scale?

    An absolute max over k-1 concepts is not meaningful: activations differ by an order
    of magnitude across features, so the max is just whichever concept is largest. Every
    accepted concept has a_j > 0 by construction (candidacy required h_strong > 0), so a
    relative change is well defined.
    """
    tgt = abs(a_new[feat] - a_base[feat]) / max(abs(a_base[feat]), 1e-6)
    if len(others) == 0:
        return {"other_rel_median": 0.0, "other_rel_p90": 0.0, "other_rel_max": 0.0,
                "other_abs_max": 0.0, "target_rel": float(tgt),
                "selectivity_ratio": float("inf"), "n_others_moved_gt_10pct": 0}
    b, n = a_base[others], a_new[others]
    rel = np.abs(n - b) / np.maximum(np.abs(b), 1e-6)
    p90 = float(np.percentile(rel, 90))
    return {"other_rel_median": float(np.median(rel)), "other_rel_p90": p90,
            "other_rel_max": float(rel.max()),
            "other_abs_max": float(np.abs(n - b).max()),
            "target_rel": float(tgt),
            # >1 means the target moved relatively more than the 90th-pct other concept
            "selectivity_ratio": float(tgt / (p90 + 1e-9)),
            "n_others_moved_gt_10pct": int((rel > 0.10).sum())}


def column_sensitivity(ev, space, x0, a_base_row, feat, others, max_levels,
                       probe_cols=None, nbins=20):
    """Pass 1: how much can each column move THIS concept, per unit of log frequency.

    Both column types are probed the same way. The column's histogram, read from where
    this row sits, supplies the destinations and their steps dL = |log count(dest) -
    log count(x0)|; there is no chosen step size, so continuous and categorical columns
    are no longer probed at incomparable magnitudes. Previously continuous columns got
    +/- 0.5 IQR -- a cost pinned near 0.5 by construction -- while categorical columns
    got levels spread across the whole distribution, costing 1.5 to 8.5 nats. Categorical
    columns then produced larger responses and outranked continuous ones for reasons that
    had nothing to do with which column controls the concept.

    Slopes are ONE-SIDED and first order for both types:

        g = (a_c(x0) - a_c(x')) / dL          positive = suppresses

    Centred differencing was considered and rejected: it exists only for continuous
    columns, so the ranking would compare a second-order estimate against a first-order
    one. One-sided everywhere also means log freq never has to act as a signed coordinate,
    which it cannot -- it peaks at the mode and falls either side.

    Ranking is on the concept ALONE. This used to rank by selectivity, concept movement
    per unit of collateral, which applies the objective's blast penalty inside the
    generator: a column that moves c hard and others hard was discarded at pass 1, so
    pass 2 never discovered that a milder step down the same column is selective. The
    blast that disqualified it was a property of the probe's size, not of the column. The
    other concepts' response is still measured and returned, because pass 2 needs blast
    per candidate and selectivity_ratio remains a useful diagnostic -- it is simply not
    what orders the columns.

    dL == 0 destinations carry no slope (the denominator vanishes) and are dropped here.
    They remain legitimate patch values for pass 2, which does not divide by anything.
    """
    variants, meta = [], []
    for j in (range(space.n_cols) if probe_cols is None else probe_cols):
        col = space.cols[j]
        pool = step_pool(col, x0[j], max_levels, categorical=j in space.cat, nbins=nbins)
        for val, dL in pool:
            if _same(val, x0[j]) or not np.isfinite(dL):
                continue
            r = list(x0); r[j] = val
            variants.append(r); meta.append((int(j), val, float(dL)))
    if not variants:
        return []
    a, rel = ev(variants)
    out = []
    for i, (j, val, dL) in enumerate(meta):
        m = shift_metrics(a_base_row, a[i], others, feat)
        d = float(a_base_row[feat] - a[i][feat])          # positive = suppresses
        col = space.cols[j]
        rec = {"column": j, "column_name": str(space.names[j]), "value": val,
               "drop": d, "delta_log_freq": dL,
               # the pass-1 statistic: response per unit of log frequency, one-sided.
               # comparable across columns AND across column types, which is the whole
               # point of measuring the step in log freq.
               "slope": (d / dL) if dL > 0 else float("nan"),
               "activation_after": float(a[i][feat]), "recon_rel": float(rel[i]),
               # edit_distance is per-column: `v` used to leak from the probe loop, so
               # every record was scored against the LAST probed column's distribution.
               "edit_distance": edit_distance(col, x0[j], val, categorical=j in space.cat),
               "selectivity": d / (m["other_rel_p90"] + 1e-6)}
        rec.update(m)
        out.append(rec)
    return out


def search_row(donor, dataset, X_ctx, y_ctx, X_query, task, device, row, feat,
               others, space, sel_tol, recon_bar, value_search=True,
               max_levels=6, top_m=8, probe_cols=None, n_line=192,
               uninhibited=False,
               recip_shared=None, recipient=None, npz_path=None):
    """Greedy search over input (column, value) edits, scored on the joint objective.

    The search is over INPUT features and values. Concepts are measured, never searched:
    the transfer's concept selection is fixed history, the context we intervene in.

    Candidates are scored by  drop_frac * reversal / (1 + blast)  -- three MEASURED
    terms, none computed. The recipient term cannot be derived from the donor side: the
    delta is linear in the activations only for a fixed accepted set with fixed atoms,
    the recipient's prediction is nonlinear in the delta, the mapping carries its own
    normalisation, and the accepted set is itself a function of the activations. So the
    recipient forward sits inside the loop -- that is the coupling, and it is why
    tabicl <-> tabicl_v2 pairs cannot run in one process.

    Cost stays two batched calls per pass regardless of candidate count: one donor
    forward for all candidates, one recipient call for their deltas.

    `recip` carries the recipient context. Without it only the donor-side terms exist
    (carte cells, where the readout does not reproduce).
    """
    import itertools

    base = X_query.copy()
    a0, rel0 = extract_acts(donor, dataset, X_ctx, y_ctx, base, task, device)
    a_base_row = a0[row].copy()
    a_start = float(a_base_row[feat])
    x0 = space.row(row)   # the row in the SPACE's columns, not the model's
    ev = make_evaluator(donor, dataset, X_ctx, y_ctx, base, space, task, device, row,
                        a_base_row)
    # Built here, not by the caller: the ratios are taken against OUR re-extracted
    # baseline, which only exists once the donor forward above has run.
    recip = build_recip(recip_shared, donor, recipient, dataset, npz_path, row,
                        a_base_row, feat, device) if recip_shared is not None else None

    sens = column_sensitivity(ev, space, x0, a_base_row, feat, others,
                              max_levels, probe_cols=probe_cols)
    # Rank on the CONCEPT alone -- response per unit of log frequency. The objective
    # balances drop against blast, reconstruction and the recipient effect in pass 2,
    # where it has the whole menu to balance across; doing it here discards a column
    # because one probe of it happened to be large.
    #
    # Per column we keep the probe with the largest slope: pass 1 exists to answer "can
    # this column move the concept", and the steepest response answers it. The
    # smallest-step probe would be the more accurate derivative but answers a question
    # we are not asking.
    per_col = {}
    for s in sens:
        if s["drop"] <= 0 or not np.isfinite(s["slope"]):
            continue
        cur = per_col.get(s["column"])
        if cur is None or s["slope"] > cur["slope"]:
            per_col[s["column"]] = s
    ranked = sorted(per_col.values(), key=lambda s: -s["slope"])[:top_m]

    # The columns are ranked by their BEST probe, but every probed value on those columns
    # is kept. Pass 1 already measured them; discarding all but the maximum is what left
    # pass 2 with nothing but maximum-suppression candidates, so a row whose strongest
    # edit overshoots had no milder edit to fall back on and returned no patch at all.
    #
    # Costs nothing extra in forwards: top_cols(8) x max_vals(6) = 48 candidates against
    # ~199 usable window slots, so a greedy step is still one forward on any dataset with
    # a full window. `values` is one entry per column when value_search is off, which is
    # exactly the previous behaviour.
    by_col = defaultdict(list)
    for s in sens:
        if s["drop"] <= 0 or not np.isfinite(s["slope"]):
            continue
        by_col[s["column"]].append(s)
    if value_search:
        # Dense line search on the CONTINUOUS columns. Pass 1's probes are a coarse grid;
        # they set the direction -- the sign of the move that suppressed -- and the scan
        # runs from the row's current value out to the marginal's edge in that direction.
        # The extent is deliberately not bounded by where those gradients stop being
        # favourable: they are first-order main effects from the unpatched row, and using
        # them to veto a region means the conditioned evaluation never sees it.
        #
        # Categoricals have no line to walk, so their pool stays the admissible level set.
        for c in list(by_col):
            if not by_col[c]:
                continue
            lead = max(by_col[c], key=lambda s: s["slope"])
            col = space.cols[c]
            seen = {float(s["value"]) for s in by_col[c]
                    if not isinstance(s["value"], str)}
            def _add(val, is_cat):
                by_col[c].append({"column": c, "value": val, "slope": lead["slope"],
                                  "drop": lead["drop"], "delta_log_freq": np.nan,
                                  "edit_distance": edit_distance(col, x0[c], val,
                                                                 categorical=is_cat)})
            if c in space.cat:
                # UNINHIBITED: every observed level, including those whose pass-1 probe
                # moved the concept the wrong way. Those were excluded by a first-order
                # estimate from the UNPATCHED row -- the same reasoning that stopped
                # pass-1 gradients bounding the continuous scan. A level that looks
                # unhelpful before any edit is committed may not be after one.
                if not uninhibited:
                    continue
                for val in _present(col):
                    v = float(val) if not isinstance(val, str) else val
                    if _same(val, x0[c]) or (not isinstance(v, str) and v in seen):
                        continue
                    if not isinstance(v, str):
                        seen.add(v)
                    _add(v, True)
            else:
                # Uninhibited scans BOTH sides of the marginal; otherwise only the side
                # pass 1 found suppressing.
                dirs = (1.0, -1.0) if uninhibited else (
                    (np.sign(float(lead["value"]) - float(x0[c])),))
                for direction in dirs:
                    if direction == 0:
                        continue
                    for val in line_search_values(col, float(x0[c]), direction, n_line):
                        if float(val) in seen or _same(val, x0[c]):
                            continue
                        seen.add(float(val))
                        _add(float(val), False)
        for c in by_col:
            by_col[c].sort(key=lambda s: -s["slope"])
    else:
        by_col = {c: [per_col[c]] for c in per_col}

    best, stop = None, "no_sensitive_column"
    committed = []          # list of the sensitivity records already applied
    trajectory = []         # objective terms at each committed column
    if ranked:
        # GREEDY, commit-and-re-probe, structurally the same loop transfer_sweep_v2 runs
        # over concepts. Each step evaluates every not-yet-committed column ON TOP of what
        # is already applied, in one batched forward plus one batched recipient call, then
        # commits the best surviving candidate.
        #
        # Replaces enumerating every combination of columns. Enumeration
        # assumes the columns' effects ADD -- it scores each combination from the ORIGINAL
        # row, so it never sees that a column's effect changes once another has been
        # applied. Re-probing is what makes the coupling visible, and it is not more
        # expensive: 92 combinations at 8 columns and 3 steps against ~3 x 8 here.
        #
        # Values stay pinned to the pass-1 winner per column. Searching values in
        # combination is the next version; this one first establishes that greedy plus the
        # crossing guard behaves.
        # PER-COLUMN LINE SEARCH. Columns are taken in pass-1 slope order, and each gets
        # ONE forward spending the whole window on its own values, conditioned on the
        # columns already fixed.
        #
        # What this replaces: a pool of (column, value) pairs across all ranked columns,
        # evaluated together. That split the window eight ways, leaving 16 points to cover
        # an entire marginal -- a spacing of about a sixteenth of the column's range, which
        # steps over any narrow band where the concept actually responds. It was neither a
        # search over combinations nor a search over values, just a sparse sample of the
        # product space, and it could not resolve what it was looking for.
        #
        # Cost is one forward per column ADDED. A 3-column patch costs 3 forwards plus
        # pass 1, about what the pooled version spent, at an order of magnitude more
        # resolution; a row that stops at one column pays for one.
        col_order = [s["column"] for s in ranked]
        # No cap on how many columns a patch may edit. The greedy is already bounded by
        # the pool -- top_m columns, each leaving once committed -- and it stops on its own
        # when nothing improves, the target is reached, or the concept is fully suppressed.
        # A separate max_steps=3 overrode those rules on 27.3% of v10 patches: rows still
        # improving when the loop ran out of iterations, recorded as if they had stopped by
        # choice. It was also a leftover from enumeration, where it existed to keep
        # C(top_m, k) from exploding; greedy has no explosion to control, since a step
        # costs one forward whether it commits the first column or the sixth.
        #
        # Deliberately not replaced with a smaller cap "for interpretability". If
        # suppressing a concept takes six columns, that is what it costs, and deciding
        # otherwise in advance hides the cost rather than reducing it.
        for cand_col in col_order:
            rows_, meta_ = [], []
            for cand in by_col.get(cand_col, []):
                r = list(x0)
                for s in committed:
                    r[s["column"]] = s["value"]
                r[cand_col] = cand["value"]
                rows_.append(r); meta_.append(cand)
            if not rows_:
                continue
            a, rel = ev(rows_)
            revs = recipient_reversal(recip, a, feat) if recip else [float("nan")] * len(a)
            scored = []
            for i, cand in enumerate(meta_):
                m = shift_metrics(a_base_row, a[i], others, feat)
                drop = a_start - float(a[i][feat])
                df = drop / a_start if a_start > 0 else float("nan")
                bl_raw = blast_radius(a_base_row, a[i], others)
                bl_w = weighted_blast(a_base_row, a[i], others, recip)
                bl = bl_raw if bl_w is None else bl_w
                rev = float(revs[i])
                r0 = float(rel0[row])
                rex = max(0.0, float(rel[i]) / r0 - 1.0) if r0 > EPS else 0.0
                if drop <= 0 or (recon_bar is not None and float(rel[i]) > recon_bar):
                    continue
                # CROSSING GUARD, the convention transfer_sweep_v2 already uses: it
                # rejects any candidate whose intervened prediction goes PAST the strong
                # prediction, and _gc clamps gap_closed to [0,1]. The patch's target is the
                # recipient's ORIGINAL pre-transfer prediction, and reversal > 1 is exactly
                # "crossed past it", so the same guard applies here.
                #
                # It also repairs the objective without changing it: sqrt(reversal) is a
                # sound reward BELOW the target and only misbehaves above, so removing the
                # crossings removes the region where it selects the wrong candidate. That
                # region was 26.1% of chosen patches.
                if np.isfinite(rev) and rev > 1.0:
                    continue
                cols = [s["column"] for s in committed + [cand]]
                scored.append({"columns": cols,
                               "values": [s["value"] for s in committed + [cand]],
                               "activation_after": float(a[i][feat]), "drop": drop,
                               "drop_frac": df, "blast": bl, "blast_raw": bl_raw,
                               "blast_weighted": bl_w, "reversal": rev,
                               "recon_excess": rex,
                               "score": objective(df, rev if np.isfinite(rev) else 1.0,
                                                  bl, rex),
                               "recon_rel": float(rel[i]),
                               "edit_distance": float(sum(s["edit_distance"]
                                                          for s in committed + [cand])),
                               "_cand": cand, "_vec": a[i], **m})
            if not scored:
                # This column offered nothing admissible. Skip it and try the next: the
                # ranking is a first-order estimate from the unpatched row, so one column
                # failing says nothing about the rest.
                if not committed:
                    stop = "no_qualifying_combination"
                continue
            # No tie-break. It existed to counteract enumeration's bias toward larger
            # combinations -- there was always a 3-column combo scoring marginally higher,
            # so 74% of patches used all three. Greedy adds a column only when it improves
            # the objective, so size is governed by the stopping rule instead, and within a
            # step every candidate has the same column count so that half of the key was
            # inert anyway. --drop-tol went with it: it only existed to stop the tie-break
            # paying for a smaller edit with suppression.
            _s = lambda c: c["score"] if np.isfinite(c["score"]) else -np.inf
            step_best = max(scored, key=_s)
            # Commit only on improvement, as the transfer greedy does -- but SKIP rather
            # than stop, for the same reason as above. Columns are visited once each, in
            # rank order, and the search ends when the pool is exhausted or the concept is
            # suppressed.
            if best is not None and _s(step_best) <= _s(best):
                stop = "no_improvement"
                continue
            best = step_best
            committed.append(best.pop("_cand"))
            # One entry per column committed, carrying every term of the objective at that
            # step. The final `best` shows where the search ended; this shows how it got
            # there -- which column bought suppression, which bought recipient movement,
            # and where a term stopped improving.
            trajectory.append({"column": int(cand_col),
                               "column_name": str(space.names[cand_col]),
                               "value": best["values"][-1],
                               "n_cols": len(best["columns"]),
                               "score": best["score"], "drop_frac": best["drop_frac"],
                               "reversal": best["reversal"], "blast": best["blast"],
                               "recon_excess": best["recon_excess"],
                               "n_candidates_searched": len(rows_)})
            stop = ("fully_suppressed" if best["activation_after"] <= 0
                    else "best_combination")
            if best["activation_after"] <= 0:
                break
            # Target reached: stop rather than keep committing columns for a sliver of
            # reversal, which is what gc_tolerance does for the transfer greedy.
            if np.isfinite(best["reversal"]) and best["reversal"] >= REVERSAL_TOLERANCE:
                stop = "reversal_target_reached"
                break

    a_now_vec = best.pop("_vec") if best else a_base_row.copy()
    a_now = float(best["activation_after"]) if best else a_start
    cur = list(x0)
    if best:
        for c, v in zip(best["columns"], best["values"]):
            cur[c] = v

    acc = np.concatenate([[feat], others]).astype(int) if len(others) else np.array([feat])
    ratios = {int(j): float(a_now_vec[j] / a_base_row[j]) if abs(a_base_row[j]) > 1e-9 else 1.0
              for j in acc}
    return {"row": int(row), "a_start": a_start, "a_final": a_now,
            "ratio": (a_now / a_start) if a_start > 0 else float("nan"),
            "drop_frac": (1.0 - a_now / a_start) if a_start > 0 else float("nan"),
            "recon_rel_start": float(rel0[row]), "stop_reason": stop,
            # in the SPACE's columns: preprocessed values, or raw table values under
            # --space raw, where the row is reportable as-is with no inversion.
            "patched_row": [x.item() if hasattr(x, "item") else x for x in cur],
            "patched_columns": [str(space.names[c]) for c in best["columns"]] if best else [],
            "n_cols_changed": len(best["columns"]) if best else 0,
            "best": best, "trajectory": trajectory, "sensitivity_top": ranked[:5],
            "n_probes": len(sens), "n_sensitive_columns": len(per_col),
            "final_shift": shift_metrics(a_base_row, a_now_vec, others, feat),
            "accepted_ratios": ratios,
            "steps": [best] if best else []}


# ── cell selection and readout ───────────────────────────────────────────────

_IMP_CACHE: dict = {}


def row_importance(donor, dataset, feat):
    """This concept's importance at every row, from output/perrow_importance.

    row_feature_drops[r, i] is the loss change when feature i is ablated at row r --
    the same signal transfer_sweep_symmetric ranks its candidates by (:340, :577). It
    is computed per (donor, dataset) independently of the recipient and of the greedy's
    insertion order, and it is not censored by selected_features' 60-wide truncation,
    which makes it the right basis for choosing which rows to patch.

    Returns None when the dataset's tested feature set does not include this concept.
    """
    key = (donor, dataset)
    if key not in _IMP_CACHE:
        f = IMPORTANCE / donor / f"{dataset}.npz"
        if not f.exists():
            _IMP_CACHE[key] = None
        else:
            z = np.load(f, allow_pickle=True)
            _IMP_CACHE[key] = (np.asarray(z["row_feature_drops"], dtype=np.float64),
                               {int(v): i for i, v in enumerate(np.asarray(z["feature_indices"]))})
    got = _IMP_CACHE[key]
    if got is None:
        return None
    drops, fmap = got
    if feat not in fmap:
        return None
    return drops[:, fmap[feat]]


def _task_of(dataset):
    global _SPLITS_CACHE
    if _SPLITS_CACHE is None:
        _SPLITS_CACHE = json.loads(SPLITS_PATH.read_text())
    return _SPLITS_CACHE.get(dataset, {}).get("task_type", "?")


_SPLITS_CACHE = None


def cells_for_concept(donor, feat, min_rows, task_filter=None):
    """(recipient, dataset, accepted rows) where this concept was actually deployed."""
    out = []
    for f in sorted(glob.glob(str(FWD / "*" / "*.npz"))):
        try:
            z = np.load(f, allow_pickle=True)
        except Exception:
            continue
        if "selected_features" not in z.files or str(z["strong_model"]) != donor:
            continue
        sel = np.asarray(z["selected_features"])
        if sel.size == 0:
            continue
        # n_concepts = how many concepts were accepted at that row (NOT the SAE's
        # TopK -- k is overloaded, so it is not used for this anywhere). The delta is a
        # sum over n_concepts terms, so it governs how much collateral movement dilutes
        # attribution of any single one.
        dataset = os.path.basename(f)[:-4]
        # selected_features[r] is the greedy's accepted list in order, already
        # LOO-ranked (transfer_sweep_symmetric.py:727-732), so the concept's POSITION
        # is its importance rank at that row. Rank is ordinal, so it is unaffected by
        # the log-loss blowups that make raw row_feature_drops span 12 orders of
        # magnitude (p90 0.51, p99 5.5e4, max 5.8e6).
        rows = []
        for r in range(sel.shape[0]):
            acc = [int(x) for x in sel[r] if x >= 0]
            if feat in acc:
                rows.append((r, acc.index(feat)))   # (row, rank; 0 = accepted first)
        if len(rows) >= min_rows:
            out.append((str(z["weak_model"]), dataset, rows, f))
    # Rank by where this concept is most IMPORTANT, using the transfer's own per-row
    # importance. Earlier rules ranked on cell size or on how many concepts shared the
    # row; both were properties of the container rather than of the concept, and the
    # size rule steered us onto rows whose recorded concept list is truncated at 60.
    # The single filter: prefer any recipient other than carte, since a carte cell
    # yields a donor-side patch where another cell would yield a patch AND a readout.
    # It has to be a filter rather than a sort preference -- expressed as a preference
    # it was silently discarded by the dataset dedup downstream. Concepts with only
    # carte cells keep them and get donor-side patches with the readout withheld.
    usable = [c for c in out if c[0] not in READOUT_EXCLUDED]
    if usable:
        out = usable
    # Same shape of rule for the env. tabicl and tabicl_v2 cannot share an env, so a cell
    # whose RECIPIENT is tabicl_v2 cannot run in a tfm arm no matter what the donor is.
    # Filtering here rather than failing later keeps coverage: the concept picks a cell it
    # CAN run instead of losing that dataset entirely. Concepts whose every cell needs the
    # other env keep them, are reported as env_mismatch by run_one_dataset, and are picked
    # up by the arm running that interpreter -- a visible hole, not a silent one.
    runnable = [c for c in out if required_env(donor, c[0]) == current_env()]
    if runnable:
        out = runnable
    # Task-type split. Classification and regression are swept SEPARATELY so a concept's
    # evidence is not pooled across heads: whether a concept means the same thing when
    # the output head changes is the generalisation question, and pooling assumes the
    # answer. Filtering rather than sorting, for the same reason as the carte rule -- a
    # preference gets silently dropped by the dataset dedup downstream.
    #
    # STRICT, unlike the carte and env filters above. Those fall back to the full set
    # because a donor-side patch on an awkward cell beats no patch at all. A fallback
    # here would hand a regression sweep its classification cells and silently undo the
    # split: measured, tabicl f158 came back with 6 "regression" cells that were all
    # classification. A concept with no cells of this task type has none, and that is the
    # honest answer.
    if task_filter and task_filter != "all":
        out = [c for c in out if _task_of(c[1]) == task_filter]
    # then earliest acceptance of this concept, then size
    out.sort(key=lambda t: (min(v for _, v in t[2]), -len(t[2]), t[1]))
    return out


def readout(npz_path, feat, row, ratios, device):
    """Recipient prediction under the counterfactual delta.

    `ratios` maps accepted feature id -> measured a'/a, for EVERY accepted concept at
    this row, not just c. Published deltas are untouched: activations come from the
    corpus and only the dimensionless ratios come from the patch.
    """
    from scripts.rebuttal.functional_decomposition import _gc
    from scripts.intervention.intervene_lib import (
        SEQUENTIAL_MODELS, batched_ablation_sequential)

    z = np.load(npz_path, allow_pickle=True)
    donor, recipient = str(z["strong_model"]), str(z["weak_model"])
    dataset = os.path.basename(npz_path)[:-4]
    dd = np.asarray(z["deployed_delta"], dtype=np.float64)
    sel = np.asarray(z["selected_features"])
    zc = np.load(ATOMS / f"{donor}_to_{recipient}.npz", allow_pickle=True)
    V = np.asarray(zc["virtual_atoms"], dtype=np.float64)
    fmap = {int(f): i for i, f in enumerate(np.asarray(zc["feature_ids"]))}
    _, std_w = load_norm_stats(recipient, dataset, device=device)
    std_w = np.asarray(std_w.cpu(), dtype=np.float64)

    sae, _ = load_sae(donor, device=device)
    with torch.no_grad():
        A = sae.encode(torch.tensor(np.asarray(load_test_embeddings(donor)[dataset],
                                               dtype=np.float32),
                                    device=device)).cpu().numpy().astype(np.float64)

    fids = [int(f) for f in np.unique(sel[row][sel[row] >= 0]) if int(f) in fmap]
    if feat not in fids:
        return None
    B = np.stack([V[fmap[f]] * std_w for f in fids])
    c, *_ = np.linalg.lstsq(B.T, dd[row], rcond=None)
    signs = np.sign(c)
    sign_c = float(signs[fids.index(feat)])

    # Rebuild the counterfactual from the MEASURED ratio of EVERY accepted concept.
    # Ratios keep everything on the corpus scale (a_cf_j = a_j^corpus * ratio_j), so the
    # published deltas stay intact and only dimensionless ratios come from re-extraction.
    # Using c's ratio alone would ASSUME the selectivity we are trying to establish --
    # that assumption is what made the previous purity trivially 1.0.
    r_vec = np.array([float(ratios.get(int(f), 1.0)) for f in fids])
    a_corpus = np.array([A[row, f] for f in fids])
    d_cf = ((signs * a_corpus * r_vec) @ B)
    d_total = d_cf - dd[row]
    i_c = fids.index(feat)
    d_from_c = signs[i_c] * a_corpus[i_c] * (r_vec[i_c] - 1.0) * B[i_c]
    purity = float(np.linalg.norm(d_from_c) / (np.linalg.norm(d_total) + 1e-12))

    # CEILING CONTROL: ablate c outright -- remove its term and leave every other
    # concept untouched. That is the most any input patch could achieve, because a
    # perfect patch drives a_c to zero and disturbs nothing else. It also separates
    # "the patch failed" from "this concept barely matters at this row": if the ceiling
    # effect is ~0, no patch can show anything and that is a fact about the concept.
    r_abl = np.ones_like(r_vec)
    r_abl[i_c] = 0.0
    d_abl = ((signs * a_corpus * r_abl) @ B)

    splits = json.loads(SPLITS_PATH.read_text())
    Xtr, ytr, Xq, _, _, task = load_dataset_context(recipient, dataset, splits)
    if ytr.dtype == np.int32:
        ytr = ytr.astype(np.int64)
    layer = get_extraction_layer_taskaware(recipient, dataset=dataset)
    cat_idx = None
    if recipient in ("hyperfast", "tabpfn"):
        from data.preprocessing import load_preprocessed, CACHE_DIR
        # No swallow. An unreadable cache means the recipient tail would be built without
        # cat_indices -- a differently configured model, silently. That is the same defect
        # as the donor-side fit bug, and it cannot be detected downstream: the run does not
        # crash, it just produces predictions from a model we did not intend.
        cat_idx = load_preprocessed(recipient, dataset, CACHE_DIR).cat_indices or None
    _reseed()
    tail = build_tail(recipient, Xtr, ytr, Xq, layer, task, device, cat_indices=cat_idx,
                      target_name=splits.get(dataset, {}).get("target", "target"))
    deltas = torch.tensor(np.vstack([dd[row], d_cf, d_abl]), dtype=torch.float32, device=device)
    if isinstance(tail, SEQUENTIAL_MODELS):
        preds = np.asarray(batched_ablation_sequential(tail, Xq[row:row+1], deltas, query_idx=row),
                           dtype=np.float64)
    else:
        preds = np.asarray(batched_intervention(tail, Xq[row:row+1], deltas, inject_context=False),
                           dtype=np.float64)
    y = int(np.asarray(z["y_query"])[row])
    b, t = np.asarray(z["preds_weak"])[row], np.asarray(z["preds_strong"])[row]
    gc_dep = float(_gc(b, preds[0], t, y))
    gc_cf = float(_gc(b, preds[1], t, y))
    gc_ceil = float(_gc(b, preds[2], t, y))
    # everything in gap-closure units: how much of the achievable effect did the patch
    # deliver? capture near 1 = the patch does what ablating the concept does.
    ceiling_effect = gc_dep - gc_ceil
    patch_effect = gc_dep - gc_cf
    return {"recipient": recipient, "dataset": dataset, "row": int(row),
            "gc_deployed": gc_dep,
            "gc_counterfactual": gc_cf,
            "gc_ceiling_ablated": gc_ceil,
            "ceiling_effect": ceiling_effect,
            "patch_effect": patch_effect,
            "capture_of_ceiling": (float(patch_effect / ceiling_effect)
                                    if abs(ceiling_effect) > 1e-9 else float("nan")),
            "attribution_purity": purity, "sign_c": sign_c,
            "a_c_corpus": float(A[row, feat]), "ratio_c": float(r_vec[i_c]),
            "n_accepted": len(fids),
            "delta_rel_change": float(np.linalg.norm(d_total) /
                                      (np.linalg.norm(dd[row]) + 1e-12))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="run the 5 stratified concepts")
    ap.add_argument("--concepts", nargs="*", default=None, help="donor:feat pairs")
    ap.add_argument("--n-datasets", type=int, default=None,
                    help="datasets per concept; default ALL. Ranking is execution "
                         "ORDER, not eligibility, so this is purely a budget dial -- "
                         "lowering it stops earlier down an importance-ordered list, "
                         "it does not make rows ineligible.")
    ap.add_argument("--n-activation-bands", type=int, default=3,
                    help="activation strata to spread the sampled rows over; 1 would "
                         "characterise the concept from its top-activation rows alone")
    ap.add_argument("--n-rows", type=int, default=None,
                    help="rows patched per dataset; default ALL. Budget dial, as above.")
    ap.add_argument("--top-cols", type=int, default=None,
                    help="columns carried from the pass-1 sensitivity map into the "
                         "pass-2 combination search; default ALL that suppress. This "
                         "one is NOT a pure budget dial -- too small and a patch that "
                         "exists is never found, which would show up as a false 'no "
                         "qualifying patch' in the coverage figure.")
    ap.add_argument("--max-vals", type=int, default=None,
                    help="candidate values per column; default the column's ENTIRE "
                         "observed support. Also a search-space limit, not a budget.")
    ap.add_argument("--task", choices=("classification", "regression", "all"),
                    default="classification",
                    help="sweep one task type at a time. Classification and regression "
                         "are kept SEPARATE because whether a concept means the same "
                         "thing when the output head changes IS the generalisation "
                         "question -- pooling them assumes the answer. Restricting to "
                         "classification costs no concepts (0 of 335 have only regression "
                         "cells), 16.8%% of cells and 10.8%% of rows; it does leave 14 "
                         "concepts under 30 rows, up from 5.")
    ap.add_argument("--exponents", default=None,
                    help="override the objective's exponents as "
                         "drop,reversal,blast,recon (default 1,0.5,1,1). The reversal one "
                         "is the live knob: with the crossing guard bounding reversal at 1, "
                         "a sqrt COMPRESSES differences between low values and makes the "
                         "search less willing to trade suppression for recipient movement. "
                         "Raising it chases reversal harder.")
    ap.add_argument("--uninhibited", action="store_true",
                    help="scan BOTH sides of the marginal on continuous columns, and offer "
                         "every observed level on categoricals rather than only those whose "
                         "pass-1 probe suppressed. Costs 2-4 forwards per greedy step "
                         "instead of 1, measured over the datasets in play.")
    ap.add_argument("--n-line", type=int, default=192,
                    help="points in the line search along ONE continuous column, per "
                         "direction, from the row's current value toward the marginal's "
                         "edge. Sized to fill the query window, since each column now gets "
                         "its own forward -- it was 16 when eight columns shared one "
                         "window, a spacing of a sixteenth of the range that stepped over "
                         "any narrow band the concept responds in. Snapped to the column's "
                         "whole numbers on an integral column, interpolated where it is "
                         "continuous.")
    ap.add_argument("--no-value-search", action="store_true",
                    help="offer each ranked column only its single most-suppressive value, "
                         "as the search did before values were searched. Kept to reproduce "
                         "the v9 baseline; the default searches every probed value, which "
                         "costs no extra forwards (8 columns x 6 values = 48 candidates "
                         "against ~199 window slots).")
    ap.add_argument("--space", choices=("raw", "preprocessed"), default="raw",
                    help="which columns the search edits. RAW is the canonical path: it "
                         "edits the original table and transforms through the fitted "
                         "generator, so the model input is produced by the code that "
                         "built the corpus, the patch is model-independent, and the "
                         "reported columns and values are the table's own. Verified to "
                         "reproduce X_query exactly, and refuses to run if it does not. "
                         "'preprocessed' is kept only to reproduce pre-v7 sweeps.")
    ap.add_argument("--min-rows", type=int, default=1,
                    help="minimum accepted rows for a cell to be usable. Cells are "
                         "ranked largest-first, so this only excludes empties.")
    ap.add_argument("--selectivity-tol", type=float, default=None,
                    help="max allowed relative shift (p90) in the other accepted "
                         "concepts; omit to record without constraining")
    ap.add_argument("--recon-bar", type=float, default=None)
    ap.add_argument("--max-probe-cols", type=int, default=None,
                    help="prefilter pass-1 probes to the top-N columns by rank "
                         "correlation; essential for models that fail independence, "
                         "where each probe costs its own forward")
    ap.add_argument("--from-burndown", nargs="?", const=str(
        PROJECT_ROOT / "output" / "rebuttal" / "patching_burndown.csv"),
        default=str(PROJECT_ROOT / "output" / "rebuttal" / "patching_burndown.csv"),
                    help="concept list; defaults to the locked set, so a bare run does "
                         "the full sweep")
    ap.add_argument("--donors", nargs="*", default=None,
                    help="restrict to these donors (tabicl_v2 must run under tfm2)")
    ap.add_argument("--park-donors", nargs="*", default=None,
                    help="set these donors aside for this run; they are reported as "
                         "parked, never silently dropped")
    ap.add_argument("--shard", default=None, help="i/n -- take every n-th concept")
    ap.add_argument("--no-resume", action="store_true",
                    help="recompute concepts already present in --out (default resumes)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output" / "rebuttal" / "patch_search.json"))
    args = ap.parse_args()
    if args.exponents:
        vals = [float(x) for x in args.exponents.split(",")]
        if len(vals) != 4:
            raise SystemExit("--exponents needs four numbers: drop,reversal,blast,recon")
        EXPONENTS.update(zip(("drop", "reversal", "blast", "recon"), vals))
        print(f"objective exponents: {EXPONENTS}", flush=True)

    # explicit --concepts wins; --probe next; otherwise the locked set, so a bare
    # run does the full sweep
    if args.concepts:
        concepts = [(c.split(":")[0], int(c.split(":")[1])) for c in args.concepts]
    elif args.probe:
        concepts = PROBE_CONCEPTS
    else:
        import csv
        concepts = [(r["donor"], int(r["feat_id"]))
                    for r in csv.DictReader(open(args.from_burndown))]
    parked = set(args.park_donors or []) | EXCLUDED_DONORS
    n_parked = sum(1 for c in concepts if c[0] in parked)
    if n_parked:
        print(f"PARKED {n_parked} concepts from donors {sorted(parked)} "
              f"-- set aside for this run, not dropped from the population")
    concepts = [c for c in concepts if c[0] not in parked]
    if args.donors:
        concepts = [c for c in concepts if c[0] in set(args.donors)]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        concepts = concepts[i::n]

    results = []
    done = set()
    if not args.no_resume and os.path.exists(args.out):
        try:
            results = json.load(open(args.out))
            done = {(r["donor"], r["feat"]) for r in results}
            print(f"resuming: {len(done)} concepts already done")
        except Exception:
            results = []
    concepts = [c for c in concepts if c not in done]
    print(f"{len(concepts)} concepts to run -> {args.out}")
    torch.use_deterministic_algorithms(True)

    for donor, feat in concepts:
        try:
            results.append(run_concept(donor, feat, args))
        except Exception as exc:
            # one concept failing must not discard the whole run's results
            print(f"    FAILED {type(exc).__name__}: {exc}", flush=True)
            results.append({"donor": donor, "feat": feat, "status": "error",
                            "error": f"{type(exc).__name__}: {exc}"})
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2, default=float)
    print(f"\nwrote {args.out}")
    # Terminal marker, and the ONLY thing that means this run finished. The output file
    # is rewritten after every concept so a killed run leaves a complete-looking JSON,
    # and "no process on the host" is not completion either -- two v8 arms were SIGHUP'd
    # mid-sweep, left partial output, reported no error, and were read as finished. That
    # cost 34 concepts and 233 rows before the gap showed up in the coverage accounting.
    # Counts are on the line so a truncated run cannot be mistaken for a whole one.
    done = sum(1 for r in results if not r.get("status"))
    print(f"DONE {args.out} concepts={len(results)} ok={done} "
          f"errors={len(results) - done}", flush=True)


def run_concept(donor, feat, args):
    if True:
        if donor in EXCLUDED_DONORS:
            print(f"\n{donor} f{feat}: SKIPPED (donor excluded)")
            return {"donor": donor, "feat": feat, "status": "excluded_donor"}
        cells = cells_for_concept(donor, feat, args.min_rows, args.task)
        if not cells:
            print(f"\n{donor} f{feat}: no cell with >= {args.min_rows} accepted rows")
            return {"donor": donor, "feat": feat, "status": "no_cell"}
        # Deterministic variety: one cell per DATASET (the largest, since donor-side
        # activation does not depend on the recipient), datasets ranked largest-first,
        # top N taken. Anchoring on the single densest dataset would explain each
        # concept in one place; this spreads the evidence without letting the search
        # choose where it is easiest.
        # cells is already ranked; dedupe by dataset keeping the FIRST (best-ranked)
        # occurrence and preserve that order. Re-sorting here by row count discarded
        # both the recipient filter and the acceptance-rank ordering, so selection kept
        # landing on the largest carte cell regardless of what was chosen upstream.
        by_ds = {}
        for rec, ds, rows_k, path in cells:
            if ds not in by_ds:
                by_ds[ds] = (rec, ds, rows_k, path)
        picks = list(by_ds.values())[:args.n_datasets]
        print(f"\n{donor} f{feat}: {len(cells)} cells over {len(by_ds)} datasets -> "
              f"top {len(picks)}: " +
              ", ".join(f"{ds}({len(rk)}r,best_rank={min(v for _, v in rk)},{rec})"
                        for rec, ds, rk, _ in picks), flush=True)

        entry = {"donor": donor, "feat": feat, "n_cells": len(cells),
                 "n_datasets_available": len(by_ds), "datasets": []}
        for recipient, dataset, acc_rows_n, npz_path in picks:
            try:
                entry["datasets"].append(
                    run_one_dataset(donor, feat, recipient, dataset, acc_rows_n,
                                    npz_path, args))
            except Exception as exc:
                print(f"    {dataset}: FAILED {type(exc).__name__}: {exc}", flush=True)
                entry["datasets"].append({"dataset": dataset, "recipient": recipient,
                                          "error": f"{type(exc).__name__}: {exc}"})
        return entry


def ordinal(i):
    """1-based position in the greedy's acceptance order, for readable reports."""
    if i is None:
        return "?"
    n = int(i) + 1
    suf = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def required_env(*models) -> str:
    """The env a cell needs: the MOST RESTRICTIVE over every model it touches.

    tabicl and tabicl_v2 cannot coexist in one env, so tabicl_v2 anywhere in the cell --
    as donor OR as recipient -- forces tfm2. Selection previously keyed the env off the
    DONOR alone, so a tabicl_v2 RECIPIENT was dispatched to tfm and died building its
    tail with `TabICL.__init__() got an unexpected keyword argument 'num_quantiles'`,
    which is v1 being handed a v2 argument. The cell was then lost from the sweep.
    """
    ms = set(models)
    if "tabicl" in ms and "tabicl_v2" in ms:
        # No env hosts both, so there is no interpreter to route this cell to. Returning
        # tfm2 here would be a lie in the other direction: tabicl v1 is absent from tfm2,
        # so the cell would fail there exactly as it fails in tfm.
        return None
    return "tfm2" if "tabicl_v2" in ms else "tfm"


def current_env() -> str:
    """Which conda env is running, derived from the INTERPRETER, not the environment.

    CONDA_PREFIX is only set by `conda activate`. Every cluster launch invokes the
    interpreter by absolute path -- /home/brian/anaconda3/envs/tfm/bin/python -m ... --
    so CONDA_PREFIX is empty and this returned "unknown". Paired with the env guard that
    meant EVERY cell was skipped as env_mismatch: five arms finished in minutes, wrote
    their output files, and reported success having done no work at all.
    """
    import sys
    # sys.prefix is the env root whether or not conda activated it:
    # /home/brian/anaconda3/envs/tfm -> "tfm".
    return os.path.basename(sys.prefix) or "unknown"


def run_one_dataset(donor, feat, recipient, dataset, acc_rows_n, npz_path, args):
    """Patch one concept in one dataset -- the unit where columns are comparable."""
    need = required_env(donor, recipient)
    have = current_env()
    if need is None:
        return {"dataset": dataset, "recipient": recipient,
                "status": "env_impossible: tabicl and tabicl_v2 cannot share an env"}
    if have == "unknown":
        # Refuse rather than skip. An undetectable env cannot be compared to a
        # requirement, and treating that as a mismatch silently empties the whole sweep.
        raise RuntimeError(
            f"cannot determine the running conda env from {os.sys.executable!r}; "
            "the env guard would skip every cell and the run would look successful")
    if have != need:
        # Reported, never silent: an env-skipped cell is a hole in the sweep and has to
        # be visible as one so the arm can be re-run under the right interpreter.
        return {"dataset": dataset, "recipient": recipient,
                "status": f"env_mismatch: needs {need}, running {have}"}
    if True:
        ranks = [v for _, v in acc_rows_n]
        print(f"  {dataset} -> {recipient} ({len(acc_rows_n)} rows, "
              f"acceptance rank best={min(ranks)} median={np.median(ranks):.0f})", flush=True)

        X_ctx, y_ctx, X_query, _, row_indices, task = load_dataset_context(donor, dataset,
                                                                 query_source="holdout")
        if hasattr(X_query, "iloc"):
            print("    donor is a DataFrame model -- not supported here")
            return {"dataset": dataset, "status": "dataframe_donor"}
        sae, _ = load_sae(donor, device=args.device)
        with torch.no_grad():
            A = sae.encode(torch.tensor(np.asarray(load_test_embeddings(donor)[dataset],
                                                   dtype=np.float32),
                                        device=args.device)).cpu().numpy().astype(np.float64)
        # The space the search edits. --space raw defines the patch on the ORIGINAL
        # table and pushes each edited row back through the fitted AutoGluon generator,
        # so the model input is produced by the same code that built the corpus. It also
        # makes the patch model-independent: MIC is 94 categorical columns for every
        # donor in raw space, where preprocessed space calls it 88 for tabpfn/tabdpt and
        # 0 for tabicl/mitra, so two donors otherwise search different spaces over
        # identical data.
        if args.space == "raw":
            space = raw_space(donor, dataset, row_indices, X_query)
        else:
            space = preprocessed_space(donor, dataset, X_query)
        cat = space.cat
        # Prefilter which columns get a pass-1 probe. Free (rank correlation with the
        # cached activation, no forwards) and essential for models that fail the
        # independence check, where every probe costs its own forward.
        probe_cols = (rank_columns(X_query, A[:, feat], args.max_probe_cols)
                      if args.max_probe_cols else None)


        # Recipient tail built ONCE per cell -- it depends on (recipient, dataset), not
        # on the row. None for carte, where the readout does not reproduce.
        recip_shared = build_recip_shared(donor, recipient, dataset, args.device)
        if recip_shared is None:
            print(f"    readout unavailable for recipient={recipient}; "
                  f"scoring donor-side only", flush=True)

        z = np.load(npz_path, allow_pickle=True)
        sel = np.asarray(z["selected_features"])

        # Every row in the cell, up to the sample size -- no ordering by k. Selecting
        # rows on k picks the ones where attribution is easiest, which is the same
        # confound as selecting the dataset on k.
        # most important rows first -- the concept's own recorded importance, so the
        # patch is attempted where it actually does work rather than wherever it happens
        # to appear
        # Two axes. Acceptance rank says where the concept did the most work;
        # activation says where it is legible enough to label and has headroom to
        # suppress. Take rows stratified across activation bands rather than the top
        # band alone, so the patch is not characterised from high-activation rows only,
        # and within each band prefer the earliest-accepted row.
        act = {r: float(A[r, feat]) for r, _ in acc_rows_n}
        order = sorted(acc_rows_n, key=lambda t: -act[t[0]])
        n_bands = min(args.n_activation_bands, len(order))
        bands = np.array_split(np.array([r for r, _ in order], dtype=int), n_bands)
        rank_of = dict(acc_rows_n)
        per_band = max(1, args.n_rows // max(n_bands, 1))
        chosen = []
        for b in bands:
            chosen += sorted(b.tolist(), key=lambda r: rank_of[r])[:per_band]
        # top up from whatever is left, earliest-accepted first
        if len(chosen) < args.n_rows:
            rest = [r for r, _ in sorted(acc_rows_n, key=lambda t: t[1]) if r not in chosen]
            chosen += rest[:args.n_rows - len(chosen)]
        chosen = chosen[:args.n_rows]
        entry_act = {int(r): act[r] for r in chosen}
        entry = {"recipient": recipient, "dataset": dataset,
                 "n_accepted_rows": len(acc_rows_n),
                 "rank_best": int(min(ranks)), "rank_median": float(np.median(ranks)),
                 "row_activation": entry_act,
                 "n_categorical_cols": len(cat),
                 "readout_usable": recipient not in READOUT_EXCLUDED, "rows": []}
        for row in chosen:
            row = int(row)
            others = np.array([int(f) for f in np.unique(sel[row][sel[row] >= 0])
                               if int(f) != feat], dtype=int)
            res = search_row(donor, dataset, X_ctx, y_ctx, X_query, task, args.device,
                             row, feat, others, space, args.selectivity_tol, args.recon_bar,
                             max_levels=args.max_vals, top_m=args.top_cols,
                             probe_cols=probe_cols, n_line=args.n_line,
                             uninhibited=args.uninhibited,
                             recip_shared=recip_shared, recipient=recipient,
                             npz_path=npz_path,
                             value_search=not args.no_value_search)
            res.update({"donor": donor, "feat": feat, "recipient": recipient,
                        "dataset": dataset, "n_other_concepts": int(len(others)),
                        "n_concepts_at_row": int(len(others)) + 1,
                        "acceptance_rank": rank_of.get(row),
                        "activation": act.get(row)})
            # a large drop means nothing without the selectivity and in-sample numbers
            m = res["final_shift"]
            rec = max((s["recon_rel"] for s in res["steps"]), default=res["recon_rel_start"])
            # A row id alone is not a statement: the same row carries many concepts.
            # Identify (donor, recipient, dataset, concept, row) and say what else was
            # injected there.
            print(f"    {donor} f{feat} -> {recipient} / {dataset} row {row}: "
                  f"1 of {len(others)+1} concepts injected here, "
                  f"accepted {ordinal(rank_of.get(row))}, act={act.get(row, float('nan')):.2f}",
                  flush=True)
            print(f"      drop {res['drop_frac']:6.1%} ({res['n_cols_changed']} cols, "
                  f"{res['stop_reason']}) | target {m['target_rel']:.1%} vs others "
                  f"med {m['other_rel_median']:.1%} p90 {m['other_rel_p90']:.1%} "
                  f"(>10%: {m['n_others_moved_gt_10pct']}/{len(others)}) | "
                  f"sel-ratio {m['selectivity_ratio']:.2f} | "
                  f"recon {res['recon_rel_start']:.3f}->{rec:.3f}", flush=True)
            # Every term of the objective, and the objective itself, for the CHOSEN patch.
            # drop x sqrt(rev) / ((1+blast)(1+recon_excess)) -- printed factor by factor so
            # a score can be read rather than inferred. Without it a low score is
            # indistinguishable between weak suppression, a recipient that did not move,
            # collateral on the other concepts, and a row pushed off-manifold, which are
            # four different problems with four different fixes.
            b = res.get("best") or {}
            if b:
                rv = b.get("reversal", float("nan"))
                bl, rx, df = b.get("blast", 0.0), b.get("recon_excess", 0.0), b.get("drop_frac", 0.0)
                print(f"      objective {b.get('score', float('nan')):.4f}"
                      f" = drop {df:.3f}"
                      f" x (toward-ablation {rv:.3f})^{EXPONENTS['reversal']:g}"
                      f" / (1+blast {bl:.3f})"
                      f" / (1+recon_excess {rx:.3f})", flush=True)
            if recipient in READOUT_EXCLUDED:
                res["readout"] = {"status": "recipient_readout_excluded",
                                  "recipient": recipient,
                                  "reason": "rebuilt tail does not reproduce the cached "
                                            "transfer for this recipient; donor-side "
                                            "suppression result is unaffected"}
            elif np.isfinite(res["ratio"]):
                try:
                    res["readout"] = readout(npz_path, feat, row, res["accepted_ratios"],
                                             args.device)
                    if res["readout"]:
                        r = res["readout"]
                        cap = r.get('capture_of_ceiling')
                        cap_s = f"{cap:6.1%}" if cap is not None and np.isfinite(cap) else "   n/a"
                        print(f"      gc(all {r['n_accepted']} concepts) {r['gc_deployed']:.4f} -> {r['gc_counterfactual']:.4f} "
                              f"| CEILING (ablate c) {r['gc_ceiling_ablated']:.4f} "
                              f"= {r['ceiling_effect']:+.4f} | patch {r['patch_effect']:+.4f} "
                              f"| capture {cap_s} | purity {r['attribution_purity']:.3f}",
                              flush=True)
                except Exception as exc:
                    res["readout"] = {"error": f"{type(exc).__name__}: {exc}"}
                    print(f"       readout ERROR {type(exc).__name__}: {exc}", flush=True)
            entry["rows"].append(res)

        return entry


if __name__ == "__main__":
    main()
