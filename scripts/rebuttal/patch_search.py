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
from collections import defaultdict

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
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
    """SAE activations + relative reconstruction error for a batch of query rows."""
    from models.layer_extraction import extract_all_layers, load_and_fit, sort_layer_names
    _reseed()
    clf = load_and_fit(donor, X_ctx, y_ctx, task=task, device=device)
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

def check_independence(donor, dataset, X_ctx, y_ctx, X_query, task, device, n_dup=8):
    """Does appending candidate rows change the base rows' activations?

    The search evaluates a whole greedy step in ONE forward by appending every candidate
    as an extra query row. That is only valid if query rows do not influence each other.
    Verified rather than assumed: embed the base set, then the base set with `n_dup` of
    its own rows appended, and check (a) the base rows are unchanged and (b) each
    duplicate matches its original.
    """
    a_base, _ = extract_acts(donor, dataset, X_ctx, y_ctx, X_query, task, device)
    dup_idx = np.linspace(0, len(X_query) - 1, n_dup).astype(int)
    Xq = np.vstack([X_query, X_query[dup_idx]])
    a_aug, _ = extract_acts(donor, dataset, X_ctx, y_ctx, Xq, task, device)
    base_shift = float(np.abs(a_aug[:len(X_query)] - a_base).max())
    dup_shift = float(np.abs(a_aug[len(X_query):] - a_base[dup_idx]).max())
    return {"base_rows_max_shift": base_shift, "duplicate_vs_original_max_shift": dup_shift,
            "independent": bool(base_shift < 1e-4 and dup_shift < 1e-4)}


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


def candidate_values(X: np.ndarray, col: int, max_vals: int,
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
    v = X[~np.isnan(X[:, col]), col]
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
        extra = uniq[np.linspace(0, len(uniq) - 1, max_vals).astype(int)]
        out = np.unique(np.concatenate([out, extra]))
    return out


def column_types(model: str, dataset: str, X: np.ndarray) -> set[int]:
    """Categorical column indices, from the preprocessing cache where it declares them.

    A "standard step" is meaningless for an ordinal-coded categorical -- moving code 2
    to 2.5 names no category, and even 2->3 is not a step of any size. Those columns get
    alternative observed LEVELS instead of a gradient.
    """
    cat: set[int] = set()
    try:
        from data.preprocessing import load_preprocessed, CACHE_DIR
        cat = set(load_preprocessed(model, dataset, CACHE_DIR).cat_indices or [])
    except Exception:
        pass
    for j in range(X.shape[1]):
        v = X[~np.isnan(X[:, j]), j]
        if j not in cat and v.size and len(np.unique(v)) <= 20 and np.allclose(v, np.round(v)):
            cat.add(j)
    return cat


def nearest_observed(X: np.ndarray, col: int, target: float) -> float:
    """Snap a proposed value onto the nearest value the column actually takes."""
    v = X[~np.isnan(X[:, col]), col]
    if v.size == 0:
        return float(target)
    return float(v[np.argmin(np.abs(v - target))])


def edit_distance(X: np.ndarray, col: int, old: float, new: float) -> float:
    """Size of an edit in units of the column's own spread, so columns are comparable.

    Must always return a finite number. The minimal-edit tie-break picks with
    min(key=(edit_distance, n_cols)), and a nan compares False against everything, so a
    nan candidate is neither smaller nor larger than its rivals -- min() then keeps
    whichever it happened to see first and the tie-break silently does nothing. In the
    v3 sweep that hit 1.9% of rows, via `old` being NaN: the original cell is MISSING,
    so the edit fills a hole rather than moving a value. Measured from the column's
    median, which is what "no value" is worth in the column's own units.
    """
    v = X[~np.isnan(X[:, col]), col]
    scale = float(np.subtract(*np.percentile(v, [75, 25]))) if v.size else 0.0
    if scale <= 1e-12:
        scale = float(np.std(v)) if v.size else 1.0
    ref = float(old)
    if not np.isfinite(ref):
        ref = float(np.median(v)) if v.size else float(new)
    d = abs(float(new) - ref) / max(scale, 1e-9)
    return float(d) if np.isfinite(d) else 0.0


# ── the search ───────────────────────────────────────────────────────────────

def make_evaluator(donor, dataset, X_ctx, y_ctx, X_query, task, device, row, batched):
    """Evaluate candidate rows -> (activations, recon_rel), one entry per candidate.

    batched=True appends every candidate as an extra query row, so a whole greedy step
    costs one forward. That is only valid when query rows do not influence each other:
    tabpfn and tabicl pass check_independence exactly (shift 0.00e+00), but MITRA FAILS
    it (shift 2.44) because its 2D attention lets query rows attend to one another.
    For those models fall back to replacing the target row and paying one forward per
    candidate -- slow, but the only correct option.
    """
    base = X_query

    def batched_eval(variants):
        Xq = np.vstack([base, np.asarray(variants, dtype=base.dtype)])
        a, rel = extract_acts(donor, dataset, X_ctx, y_ctx, Xq, task, device)
        return a[len(base):], rel[len(base):]

    def replace_eval(variants):
        acts, recs = [], []
        for v in variants:
            Xq = base.copy()
            Xq[row] = v
            a, rel = extract_acts(donor, dataset, X_ctx, y_ctx, Xq, task, device)
            acts.append(a[row]); recs.append(rel[row])
        return np.asarray(acts), np.asarray(recs)

    return batched_eval if batched else replace_eval


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


def reversal(L_orig, L_transfer, L_mod):
    """Fraction of the transfer's gain that the patch undid.

    0 = the patch changed nothing; 1 = the recipient is back to its untransferred
    prediction. Uses the transfer's own endpoints, so it needs no separate ablation and
    carries no scale. The denominator is ~0 on rows where the transfer achieved nothing;
    those rows carry no signal in either direction and are flagged rather than scored.
    """
    return float((L_mod - L_transfer) / (L_orig - L_transfer + EPS))


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
    root = math.copysign(math.sqrt(abs(r)), r) if np.isfinite(r) else float("nan")
    return float(drop_frac * root / ((1.0 + blast) * (1.0 + max(0.0, recon_excess))))


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
        try:
            cat_idx = load_preprocessed(recipient, dataset, CACHE_DIR).cat_indices or None
        except Exception:
            pass
    _reseed()
    tail = build_tail(recipient, Xtr, ytr, Xq, layer, task, device, cat_indices=cat_idx,
                      target_name=splits.get(dataset, {}).get("target", "target"))
    return {"V": V, "fmap": fmap, "std_w": std_w, "A": A, "tail": tail, "Xq": Xq}


def build_recip(shared, donor, recipient, dataset, npz_path, row, a_re, device):
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

    return {"fids": fids, "B": B, "signs": np.sign(c),
            "a_corpus": np.array([A[row, f] for f in fids]),
            "a_re": {f: float(a_re[f]) for f in fids},   # our own baseline, for ratios
            "predict": predict, "loss": loss,
            "L_orig": loss(pw), "L_transfer": loss(pi), "row": int(row)}


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
    return [reversal(recip["L_orig"], recip["L_transfer"], recip["loss"](p)) for p in preds]


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


def column_sensitivity(ev, X_query, x0, a_base_row, feat, others, cat, step_frac, max_levels,
                       probe_cols=None):
    """Pass 1: one standard step per column -> local sensitivity of c, in one forward.

    Continuous columns get +/- step_frac of their IQR, snapped to an observed value, so
    the response is a finite-difference gradient in comparable units across columns.
    Categorical columns get alternative observed levels instead, since a "step" in
    ordinal-code space is not a step of any size.

    Returns one record per (column, value) probe, carrying both the concept's response
    and the other k-1 concepts' response -- so columns can be ranked by SELECTIVITY
    (how much c moves per unit of collateral) rather than by raw effect.
    """
    variants, meta = [], []
    for j in (range(X_query.shape[1]) if probe_cols is None else probe_cols):
        v = X_query[~np.isnan(X_query[:, j]), j]
        if v.size == 0:
            continue
        if j in cat:
            vals = [val for val in candidate_values(X_query, j, max_levels)
                    if not np.isclose(val, x0[j], equal_nan=True)]
        else:
            iqr = float(np.subtract(*np.percentile(v, [75, 25]))) or float(np.std(v)) or 1.0
            vals = []
            for sgn in (-1.0, 1.0):
                cand = nearest_observed(X_query, j, x0[j] + sgn * step_frac * iqr)
                if not np.isclose(cand, x0[j], equal_nan=True):
                    vals.append(cand)
        for val in dict.fromkeys(vals):
            r = x0.copy(); r[j] = val
            variants.append(r); meta.append((int(j), float(val)))
    if not variants:
        return []
    a, rel = ev(variants)
    out = []
    for i, (j, val) in enumerate(meta):
        m = shift_metrics(a_base_row, a[i], others, feat)
        d = float(a_base_row[feat] - a[i][feat])          # positive = suppresses
        out.append({"column": j, "value": val, "drop": d,
                    "activation_after": float(a[i][feat]), "recon_rel": float(rel[i]),
                    "edit_distance": edit_distance(X_query, j, x0[j], val),
                    # effect on c per unit of collateral: the quantity we actually want
                    "selectivity": d / (m["other_rel_p90"] + 1e-6), **m})
    return out


def search_row(donor, dataset, X_ctx, y_ctx, X_query, task, device, row, feat,
               others, cat, sel_tol, recon_bar, batched=True, step_frac=0.5,
               max_levels=6, top_m=8, max_cols=3, probe_cols=None,
               recip_shared=None, recipient=None, npz_path=None, tie_band=0.10, drop_tol=0.01):
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
    x0 = base[row].copy()
    ev = make_evaluator(donor, dataset, X_ctx, y_ctx, base, task, device, row, batched)
    # Built here, not by the caller: the ratios are taken against OUR re-extracted
    # baseline, which only exists once the donor forward above has run.
    recip = build_recip(recip_shared, donor, recipient, dataset, npz_path, row,
                        a_base_row, device) if recip_shared is not None else None

    sens = column_sensitivity(ev, X_query, x0, a_base_row, feat, others, cat,
                              step_frac, max_levels, probe_cols=probe_cols)
    # best suppressing probe per column, then rank columns by selectivity
    per_col = {}
    for s in sens:
        if s["drop"] <= 0:
            continue
        if s["column"] not in per_col or s["selectivity"] > per_col[s["column"]]["selectivity"]:
            per_col[s["column"]] = s
    ranked = sorted(per_col.values(), key=lambda s: -s["selectivity"])[:top_m]

    best, stop = None, "no_sensitive_column"
    if ranked:
        combos, cmeta = [], []
        for size in range(1, max_cols + 1):
            for sub in itertools.combinations(ranked, size):
                r = x0.copy()
                for s in sub:
                    r[s["column"]] = s["value"]
                combos.append(r); cmeta.append(sub)
        a, rel = ev(combos)
        # One batched recipient call for every candidate's delta. The recipient term is
        # measured, not derived: see the docstring on why it cannot be computed from the
        # donor side.
        revs = recipient_reversal(recip, a, feat) if recip else [float("nan")] * len(a)
        cands = []
        for i, sub in enumerate(cmeta):
            m = shift_metrics(a_base_row, a[i], others, feat)
            drop = a_start - float(a[i][feat])
            df = drop / a_start if a_start > 0 else float("nan")
            bl = blast_radius(a_base_row, a[i], others)
            rev = float(revs[i])
            # in-sample term: inflation of SAE reconstruction error over THIS row's
            # baseline, one-sided (reconstructing better than the original is fine)
            r0 = float(rel0[row])
            rex = max(0.0, float(rel[i]) / r0 - 1.0) if r0 > EPS else 0.0
            ok = recon_bar is None or float(rel[i]) <= recon_bar
            if drop > 0 and ok:
                cands.append({"columns": [s["column"] for s in sub],
                              "values": [s["value"] for s in sub],
                              "activation_after": float(a[i][feat]), "drop": drop,
                              "drop_frac": df, "blast": bl, "reversal": rev,
                              "recon_excess": rex,
                              # donor-only cells (carte) have no recipient term; scoring
                              # falls back to suppression against blast so the search
                              # still runs, and the record says the readout is absent.
                              "score": objective(df, rev if np.isfinite(rev) else 1.0,
                                                 bl, rex),
                              "recon_rel": float(rel[i]),
                              "edit_distance": float(sum(s["edit_distance"] for s in sub)),
                              "_vec": a[i], **m})
        if cands:
            # Minimal-edit tie-break: among candidates that achieve the SAME suppression
            # and score within `tie_band` of the best, take the smallest edit. Without it,
            # max(score) has no reason to prefer a 1-column patch over a 3-column one that
            # scores marginally higher, and 74% of chosen patches used all 3 columns while
            # the 1- and 2-column patches suppressed just as completely (drop 1.000).
            #
            # The suppression gate is not decoration. A band on the SCORE alone lets the
            # tie-break pay for a smaller edit with the target itself, because drop_frac is
            # a factor of the score: measured over 2,112 rows, a 10% score band regressed
            # suppression on 19.3% of them, and on 291 of those the patch was not even
            # smaller -- same size, less suppression, which is a pure loss. "Tied" has to
            # mean tied on what the patch is FOR.
            #
            # Order is (n_cols, edit_distance), and that order is the whole point.
            # edit_distance is a SUM over the edited columns, so keying on it first
            # minimises total edit MAGNITUDE, not patch size -- and `len(columns)` as a
            # second element is inert, because it only breaks EXACT float ties in a sum of
            # floats. Written the other way round, the "minimal-edit tie-break" never
            # selected on column count at all; the drop from 74% to 48% three-column
            # patches was a side effect of fewer columns usually summing to less. It also
            # let a same-size candidate with a smaller magnitude and worse suppression win,
            # which is what happened on 291 rows.
            #
            # Fewest columns first is also the criterion the appendix needs: a one-column
            # patch shown in full is legible, a three-column one is a list.
            #
            # nan scores sort to the bottom rather than winning by position: max() and
            # min() both compare with <, which is False against nan either way, so a nan
            # silently survives as "best" if it is seen first.
            _s = lambda c: c["score"] if np.isfinite(c["score"]) else -np.inf
            _d = lambda c: c["drop_frac"] if np.isfinite(c["drop_frac"]) else -np.inf
            top = max(_s(c) for c in cands)
            floor = top - tie_band * abs(top) if np.isfinite(top) else -np.inf
            near = [c for c in cands if _s(c) >= floor] or cands
            d_top = max(_d(c) for c in near)
            near = [c for c in near if _d(c) >= d_top - drop_tol] or near
            best = min(near, key=lambda c: (len(c["columns"]), c["edit_distance"]))
            stop = "fully_suppressed" if best["activation_after"] <= 0 else "best_combination"
        else:
            stop = "no_qualifying_combination"

    a_now_vec = best.pop("_vec") if best else a_base_row.copy()
    a_now = float(best["activation_after"]) if best else a_start
    cur = x0.copy()
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
            "patched_row": cur.tolist(),
            "n_cols_changed": len(best["columns"]) if best else 0,
            "best": best, "sensitivity_top": ranked[:5],
            "n_probes": len(sens), "n_sensitive_columns": len(per_col),
            "final_shift": shift_metrics(a_base_row, a_now_vec, others, feat),
            "accepted_ratios": ratios, "batched": bool(batched),
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


def cells_for_concept(donor, feat, min_rows):
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
        try:
            cat_idx = load_preprocessed(recipient, dataset, CACHE_DIR).cat_indices or None
        except Exception:
            pass
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
    ap.add_argument("--drop-tol", type=float, default=0.01,
                    help="a candidate counts as tied on SUPPRESSION only if its drop_frac "
                         "is within this of the best in the score band. Without the gate "
                         "the tie-break buys a smaller edit with the target: a 0.10 score "
                         "band alone cost suppression on 19.3%% of rows, 291 of which did "
                         "not even get a smaller patch.")
    ap.add_argument("--tie-band", type=float, default=0.10,
                    help="candidates within this fraction of the best score are treated "
                         "as tied, and the SMALLEST edit among them wins. Right-sizes "
                         "the patch: drop already saturates at 1.000 with one column, "
                         "so without this the search takes ~4x larger edits for nothing.")
    ap.add_argument("--max-steps", type=int, default=3,
                    help="largest column-combination size in pass 2. The one knob that "
                         "cannot simply be maximised: pass 2 evaluates C(top_cols, size) "
                         "combinations, so this is combinatorial in the column count. "
                         "Raise it only with the cost measured.")
    ap.add_argument("--min-rows", type=int, default=1,
                    help="minimum accepted rows for a cell to be usable. Cells are "
                         "ranked largest-first, so this only excludes empties.")
    ap.add_argument("--selectivity-tol", type=float, default=None,
                    help="max allowed relative shift (p90) in the other accepted "
                         "concepts; omit to record without constraining")
    ap.add_argument("--recon-bar", type=float, default=None)
    ap.add_argument("--step-frac", type=float, default=0.5,
                    help="standard step per continuous column, in IQR units (pass 1)")
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
    ap.add_argument("--check-independence", action="store_true",
                    help="verify query rows do not influence each other before trusting "
                         "the batched candidate evaluation")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output" / "rebuttal" / "patch_search.json"))
    args = ap.parse_args()

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


def run_concept(donor, feat, args):
    if True:
        if donor in EXCLUDED_DONORS:
            print(f"\n{donor} f{feat}: SKIPPED (donor excluded)")
            return {"donor": donor, "feat": feat, "status": "excluded_donor"}
        cells = cells_for_concept(donor, feat, args.min_rows)
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


def run_one_dataset(donor, feat, recipient, dataset, acc_rows_n, npz_path, args):
    """Patch one concept in one dataset -- the unit where columns are comparable."""
    if True:
        ranks = [v for _, v in acc_rows_n]
        print(f"  {dataset} -> {recipient} ({len(acc_rows_n)} rows, "
              f"acceptance rank best={min(ranks)} median={np.median(ranks):.0f})", flush=True)

        X_ctx, y_ctx, X_query, _, _, task = load_dataset_context(donor, dataset,
                                                                 query_source="holdout")
        if hasattr(X_query, "iloc"):
            print("    donor is a DataFrame model -- not supported here")
            return {"dataset": dataset, "status": "dataframe_donor"}
        sae, _ = load_sae(donor, device=args.device)
        with torch.no_grad():
            A = sae.encode(torch.tensor(np.asarray(load_test_embeddings(donor)[dataset],
                                                   dtype=np.float32),
                                        device=args.device)).cpu().numpy().astype(np.float64)
        cat = column_types(donor, dataset, X_query)
        # Prefilter which columns get a pass-1 probe. Free (rank correlation with the
        # cached activation, no forwards) and essential for models that fail the
        # independence check, where every probe costs its own forward.
        probe_cols = (rank_columns(X_query, A[:, feat], args.max_probe_cols)
                      if args.max_probe_cols else None)

        # Batching is only valid when query rows are independent. Decide by measurement,
        # not by model name: mitra's 2D attention makes its rows interact (shift 2.44)
        # while tabpfn/tabicl are exactly 0.00.
        ind = check_independence(donor, dataset, X_ctx, y_ctx, X_query, task, args.device)
        batched = ind["independent"]
        print(f"    independence: base shift={ind['base_rows_max_shift']:.2e} "
              f"dup shift={ind['duplicate_vs_original_max_shift']:.2e} -> "
              f"{'batched' if batched else 'ROWS INTERACT -> per-candidate forwards'}",
              flush=True)

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
                             row, feat, others, cat, args.selectivity_tol, args.recon_bar,
                             batched=batched, step_frac=args.step_frac,
                             max_levels=args.max_vals, top_m=args.top_cols,
                             max_cols=args.max_steps, probe_cols=probe_cols,
                             recip_shared=recip_shared, recipient=recipient,
                             npz_path=npz_path, tie_band=args.tie_band,
                             drop_tol=args.drop_tol)
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
