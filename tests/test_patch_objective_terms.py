"""Every term the patch search optimises, pinned by a test.

Each of these terms was found wrong by a sweep, days after it had produced numbers:
`reversal` scaled to an unreachable target, `blast` summed unbounded log-losses, the
objective's divisor read as a weight. All four are pure functions of a handful of floats
and cost milliseconds to check, which is the argument for this file existing -- the
property tests in test_query_window_invariance.py needed a GPU and a model, and these
need neither.

What is deliberately NOT asserted here: which term should be in the objective at all, and
what `min_interval` should be. Those are open (docs/plans/2026-08-12-patching-handoff.md).
These tests pin the arithmetic as it currently stands so that a change to the FORM is
visible as a failing test rather than a silent shift in what the sweep was optimising.
"""

import numpy as np
import pytest

from scripts.rebuttal.patch_search import (
    ACTIVE_FLOOR,
    EPS,
    EXPONENTS,
    blast_radius,
    collateral_detail,
    endpoint,
    objective,
    reversal,
    shift_metrics,
    weighted_blast,
)


# ── endpoint: the prediction, not a loss ─────────────────────────────────────

def test_endpoint_classification_is_the_true_class_probability():
    assert endpoint(np.array([0.2, 0.7, 0.1]), 1, None) == pytest.approx(0.7)
    assert endpoint(np.array([0.2, 0.7, 0.1]), 0, None) == pytest.approx(0.2)


def test_endpoint_classification_is_bounded_by_one():
    """The whole point of moving off -log(p[y]): a badly-wrong prediction contributes at
    most 1.0, so summing k of them cannot manufacture the k-tracking inflation that made
    v15's additivity read 85.67 at k=57."""
    for p in (np.array([1e-12, 1 - 1e-12]), np.array([0.5, 0.5]), np.array([1.0, 0.0])):
        for y in (0, 1):
            assert 0.0 <= endpoint(p, y, None) <= 1.0


def test_endpoint_regression_is_squared_distance_to_the_donor():
    assert endpoint(np.float64(3.0), 0, 5.0) == pytest.approx(4.0)
    assert endpoint(np.float64(5.0), 0, 5.0) == pytest.approx(0.0)


def test_endpoint_regression_is_not_bounded_by_one():
    """Which is why min_interval cannot be one constant across both heads."""
    assert endpoint(np.float64(100.0), 0, 0.0) == pytest.approx(1e4)


# ── reversal: where the patch landed between transfer and ablation ────────────

def test_reversal_endpoints():
    # L_transfer = 0.30, ablating the concept would reach 0.80.
    assert reversal(0.30, 0.80, 0.30) == pytest.approx(0.0)    # changed nothing
    assert reversal(0.30, 0.80, 0.80) == pytest.approx(1.0)    # reached the ablation
    assert reversal(0.30, 0.80, 0.55) == pytest.approx(0.5)    # halfway


def test_reversal_is_invariant_to_the_direction_of_the_ablation():
    """Ablating a concept can move p[y] either way. Landing on the target is 1.0 in both
    directions -- if it were not, the term would reward one sign of concept."""
    assert reversal(0.30, 0.80, 0.55) == pytest.approx(reversal(0.80, 0.30, 0.55))


def test_reversal_is_negative_when_the_patch_moves_the_wrong_way():
    """Not clipped: a patch that pushes the recipient FURTHER from the ablation is worse
    than doing nothing, and the objective's sign-preserving root keeps that visible."""
    assert reversal(0.30, 0.80, 0.20) < 0


def test_reversal_above_one_means_crossing():
    """The search rejects these (the crossing guard); the function still reports them, so
    the guard is one decision in one place rather than a clamp hidden in the arithmetic."""
    assert reversal(0.30, 0.80, 1.05) > 1.0


def test_reversal_small_interval_is_floored_not_gated():
    """A small interval is a MEASUREMENT -- the concept contributes little at this row --
    not a missing value. min_interval (0.01, borrowed from the sweeps' own min_gap)
    floors the DENOMINATOR: credit is capped at movement/min_interval, so full credit
    needs movement of at least the resolution floor, but the measured movement stays in
    the score with its sign. Gating returned nan here, and nan was a double defect at the
    call site: scored as 1.0 (FULL credit) and exempt from the crossing guard -- 59% of
    v17's chosen rows."""
    # interval 0.005, below the floor: denominator becomes 0.01
    assert reversal(0.30, 0.305, 0.302) == pytest.approx(0.2)
    assert reversal(0.30, 0.305, 0.30) == pytest.approx(0.0)
    # movement far past a near-zero target reads >1 and the crossing guard REJECTS it,
    # instead of the old gate waving it through as nan
    assert reversal(0.30, 0.305, 0.32) == pytest.approx(2.0)
    # negative interval: the sign of the floored denominator follows the interval
    assert reversal(0.80, 0.795, 0.79) == pytest.approx(1.0)
    # the floor is settable -- a regression sweep is in different units
    assert reversal(0.30, 0.305, 0.302, min_interval=1e-6) == pytest.approx(0.4)


def test_reversal_is_nan_only_for_the_genuinely_unmeasured():
    """Non-finite endpoints (or no recipient context at all, upstream). Everything
    measured stays a number."""
    assert np.isnan(reversal(0.30, np.nan, 0.30))
    assert np.isnan(reversal(0.30, np.inf, 0.30))
    assert np.isnan(reversal(0.30, 0.80, np.nan))
    assert np.isnan(reversal(np.nan, 0.80, 0.30))


# ── objective ────────────────────────────────────────────────────────────────

def _exp(**kw):
    """Temporarily set EXPONENTS, since the objective reads the module-level dict."""
    old = dict(EXPONENTS)
    EXPONENTS.update(kw)
    return old


def test_objective_is_monotone_in_every_term():
    base = objective(0.5, 0.5, 0.05, 0.1)
    assert objective(0.6, 0.5, 0.05, 0.1) > base       # more suppression is better
    assert objective(0.5, 0.6, 0.05, 0.1) > base       # more recipient movement is better
    assert objective(0.5, 0.5, 0.06, 0.1) < base       # more collateral is worse
    assert objective(0.5, 0.5, 0.05, 0.2) < base       # worse reconstruction is worse


def test_objective_equal_exponents_give_equal_sensitivity_to_relative_change():
    """The property the product-of-ratios form exists to have: at equal exponents a 10%
    change in any term is worth the same. This is what `1 + x` breaks, and it is the
    argument that retired `1 + blast`."""
    old = _exp(drop=1.0, reversal=1.0, blast=1.0, recon=1.0)
    try:
        base = objective(0.5, 0.5, 0.05, 0.0)
        assert objective(0.55, 0.5, 0.05, 0.0) / base == pytest.approx(1.1)
        assert objective(0.5, 0.55, 0.05, 0.0) / base == pytest.approx(1.1)
        assert objective(0.5, 0.5, 0.05 / 1.1, 0.0) / base == pytest.approx(1.1)
    finally:
        EXPONENTS.update(old)


def test_objective_recon_excess_does_not_have_that_property():
    """recon_excess enters as `1 + x`, so its influence depends on where x sits: the same
    10% relative change is worth 9.5% at x=0.01 and 4.8% at x=1.0. Documented as a known
    inconsistency in the form, not asserted as desirable -- open defect 2 in the handoff."""
    old = _exp(drop=1.0, reversal=1.0, blast=1.0, recon=1.0)
    try:
        near = objective(0.5, 0.5, 0.05, 0.01) / objective(0.5, 0.5, 0.05, 0.01 * 1.1)
        far = objective(0.5, 0.5, 0.05, 1.0) / objective(0.5, 0.5, 0.05, 1.0 * 1.1)
        assert near != pytest.approx(far)
    finally:
        EXPONENTS.update(old)


def test_objective_recon_term_is_blind_below_the_row_s_own_error():
    """rex = max(0, rel/rel_start - 1), so every candidate reconstructing at or better than
    the unpatched row scores identically on this term -- the trade stops existing there.
    Measured on v15: 38.4% of chosen patches sit in that flat region
    (patch_recon_position.py). Above it the term is monotone in rel, and rel_start is a row
    constant, so the reference divides out of the within-row ranking entirely and only the
    CLAMP makes the choice of reference bite."""
    old = _exp(drop=1.0, reversal=1.0, blast=1.0, recon=1.0)
    try:
        # two candidates, both reconstructing better than the row's own start: same score
        assert objective(0.5, 0.5, 0.05, 0.0) == objective(0.5, 0.5, 0.05, 0.0)
        flat = [objective(0.5, 0.5, 0.05, max(0.0, r / 0.40 - 1.0)) for r in (0.10, 0.39)]
        assert flat[0] == pytest.approx(flat[1])
        # above it, monotone -- and a common factor on the reference cancels in a ratio
        a = objective(0.5, 0.5, 0.05, 0.50 / 0.40 - 1.0)
        b = objective(0.5, 0.5, 0.05, 0.60 / 0.40 - 1.0)
        assert b < a
    finally:
        EXPONENTS.update(old)


def test_objective_preserves_the_sign_of_reversal():
    """A patch that moves the recipient opposite to the transfer must score below one that
    does nothing, not above it by way of a squared root."""
    assert objective(0.9, -0.5, 0.05, 0.0) < 0
    assert objective(0.9, -0.5, 0.05, 0.0) < objective(0.9, 0.0, 0.05, 0.0)


def test_objective_scales_as_a_power_when_all_exponents_scale():
    """Only exponent RATIOS matter: raising all of them by k raises the score to the k,
    which leaves the ranking untouched. This is why the exponent sweep fixes drop=1."""
    a = (0.5, 0.4, 0.05, 0.1)
    b = (0.7, 0.6, 0.02, 0.3)
    old = _exp(drop=1.0, reversal=1.0, blast=1.0, recon=1.0)
    try:
        one = objective(*a), objective(*b)
        _exp(drop=2.0, reversal=2.0, blast=2.0, recon=2.0)
        two = objective(*a), objective(*b)
        assert two[0] == pytest.approx(one[0] ** 2)
        assert (one[0] < one[1]) == (two[0] < two[1])
    finally:
        EXPONENTS.update(old)


def test_objective_zero_collateral_scores_enormously():
    """`blast + EPS` with EPS=1e-7, so a candidate that disturbs nothing scores ~1e7. That
    is the intended behaviour of a divisor with no reference point -- scores are compared
    only WITHIN a row, never across rows -- and it is the behaviour that looked like a bug
    when blast was being fed unscaled prediction effects at 1e-3."""
    assert objective(0.5, 0.25, 0.0, 0.0) == pytest.approx(0.5 * 0.5 / EPS)


def test_objective_is_nan_when_reversal_is_nan():
    """The caller substitutes 1.0 for a nan reversal. Since the denominator floor, nan
    only reaches the caller where NO readout exists at all (carte, READOUT_EXCLUDED), and
    there 1.0 is genuinely neutral -- it removes the factor from the product, scoring the
    row on the donor-side terms. It was NOT neutral when measured-but-small intervals also
    came back nan: 1.0 is the top of reversal's guarded range, so 59% of v17's rows got
    full recipient credit with the crossing guard bypassed."""
    assert np.isnan(objective(0.5, np.nan, 0.05, 0.0))
    assert objective(0.5, 1.0, 0.05, 0.0) == pytest.approx(objective(0.5, 1.0, 0.05, 0.0))


# ── collateral ───────────────────────────────────────────────────────────────

def _recip(loo_by_fid, L_transfer=0.30, L_orig=0.80):
    return {"loo_by_fid": loo_by_fid, "L_transfer": L_transfer, "L_orig": L_orig}


def test_weighted_blast_is_zero_when_nothing_moved():
    a = np.array([1.0, 2.0, 3.0])
    assert weighted_blast(a, a.copy(), np.array([1, 2]), _recip({1: 0.4, 2: 0.2})) == 0.0


def test_weighted_blast_is_a_sum_not_a_mean():
    """Two concepts each disturbed 10% cost twice one disturbed 10%. A mean would report
    the same for both, normalising away exactly the count that should accumulate."""
    a = np.array([1.0, 1.0, 1.0])
    one = np.array([1.0, 1.1, 1.0])
    two = np.array([1.0, 1.1, 1.1])
    r = _recip({1: 0.5, 2: 0.5})
    assert (weighted_blast(a, two, np.array([1, 2]), r)
            == pytest.approx(2 * weighted_blast(a, one, np.array([1, 2]), r)))


def test_weighted_blast_ignores_concepts_the_prediction_does_not_depend_on():
    """The reason for weighting at all: a 12% shift in a concept with LOO ~0 changes no
    outcome, and blast_radius cannot tell it from a 3% shift in one carrying a third."""
    a = np.array([1.0, 1.0])
    moved = np.array([1.0, 2.0])                      # concept 1 doubled
    assert weighted_blast(a, moved, np.array([1]), _recip({1: 0.0})) == pytest.approx(0.0)


def test_weighted_blast_excludes_inactive_concepts_rather_than_flooring_them():
    """|da|/|a| on a near-zero baseline is what produced a reported 198,990,863%."""
    a = np.array([1.0, ACTIVE_FLOOR / 10])
    moved = np.array([1.0, 1.0])                      # a huge RELATIVE move on a dead concept
    assert weighted_blast(a, moved, np.array([1]), _recip({1: 0.5})) is None


def test_weighted_blast_is_a_fraction_of_the_transfer_s_own_movement():
    """Scaled by |L_transfer - L_orig| so it reads as 'what fraction of the prediction
    effect in play did we spend on concepts we were not patching'."""
    a, moved = np.array([1.0, 1.0]), np.array([1.0, 1.5])
    r = _recip({1: 0.4}, L_transfer=0.30, L_orig=0.80)
    assert weighted_blast(a, moved, np.array([1]), r) == pytest.approx(0.5 * 0.4 / 0.5)


def test_weighted_blast_is_none_when_the_transfer_moved_nothing():
    a, moved = np.array([1.0, 1.0]), np.array([1.0, 1.5])
    r = _recip({1: 0.4}, L_transfer=0.30, L_orig=0.30)
    assert weighted_blast(a, moved, np.array([1]), r) is None


def test_weighted_blast_is_none_without_a_recipient():
    a, moved = np.array([1.0, 1.0]), np.array([1.0, 1.5])
    assert weighted_blast(a, moved, np.array([1]), None) is None
    assert weighted_blast(a, moved, np.array([]), _recip({1: 0.4})) is None


def test_collateral_detail_is_ordered_by_what_was_actually_spent():
    a = np.array([1.0, 1.0, 1.0])
    moved = np.array([1.0, 1.1, 1.5])                 # feat 1 moved 10%, feat 2 moved 50%
    d = collateral_detail(a, moved, np.array([1, 2]), _recip({1: 0.9, 2: 0.1}))
    assert [x["feat"] for x in d] == [1, 2]           # 0.10*0.9 = 0.09 beats 0.50*0.1 = 0.05
    assert d[0]["disturbed"] == pytest.approx(0.09)


def test_collateral_detail_marks_inactive_concepts_instead_of_scoring_them():
    a = np.array([1.0, ACTIVE_FLOOR / 10])
    d = collateral_detail(a, np.array([1.0, 1.0]), np.array([1]), _recip({1: 0.5}))
    assert d[0]["inactive"] is True and d[0]["disturbed"] is None


def test_blast_radius_is_scale_free_and_zero_when_unchanged():
    a = np.array([1.0, 2.0, 3.0])
    assert blast_radius(a, a.copy(), np.array([1, 2])) == pytest.approx(0.0)
    moved = a * 1.1
    assert (blast_radius(a, moved, np.array([1, 2]))
            == pytest.approx(blast_radius(10 * a, 10 * moved, np.array([1, 2]))))


def test_shift_metrics_with_no_other_concepts_is_perfectly_selective():
    a = np.array([1.0, 2.0])
    m = shift_metrics(a, np.array([0.5, 2.0]), np.array([], dtype=int), 0)
    assert m["selectivity_ratio"] == float("inf")
    assert m["target_rel"] == pytest.approx(0.5)
    assert m["n_others_moved_gt_10pct"] == 0
