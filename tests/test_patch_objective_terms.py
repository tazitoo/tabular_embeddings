"""Every term the patch search optimises, pinned by a test.

Each of these terms was found wrong by a sweep, days after it had produced numbers:
`toward_ablation` scaled to an unreachable target, `blast` summed unbounded log-losses, the
objective's divisor read as a weight. All four are pure functions of a handful of floats
and cost milliseconds to check, which is the argument for this file existing -- the
property tests in test_query_window_invariance.py needed a GPU and a model, and these
need neither.

What is deliberately NOT asserted here: the exponents, which are an experimental
question the exponent sweep answers against the baseline. These tests pin the arithmetic
as it stands so that a change to the FORM is visible as a failing test rather than a
silent shift in what the sweep was optimising.
"""

import numpy as np
import pytest

from scripts.rebuttal.patch_search import (
    ACTIVE_FLOOR,
    EPS,
    EXPONENTS,
    MIN_GAP,
    blast_radius,
    centrality,
    collateral_detail,
    donor_dist_sq,
    objective,
    gap_opened_metric,
    probe_effectiveness,
    recipient_toward_ablation,
    shift_metrics,
    toward_ablation,
    true_class_prob,
    weighted_blast,
)


# ── the recipient's scalar: a probability, or a squared distance ─────────────

def test_true_class_prob_reads_the_true_class():
    assert true_class_prob(np.array([0.2, 0.7, 0.1]), 1) == pytest.approx(0.7)
    assert true_class_prob(np.array([0.2, 0.7, 0.1]), 0) == pytest.approx(0.2)


def test_true_class_prob_is_bounded_by_one():
    """The whole point of moving off -log(p[y]): a badly-wrong prediction contributes at
    most 1.0, so summing k of them cannot manufacture the k-tracking inflation that made
    v15's additivity read 85.67 at k=57."""
    for p in (np.array([1e-12, 1 - 1e-12]), np.array([0.5, 0.5]), np.array([1.0, 0.0])):
        for y in (0, 1):
            assert 0.0 <= true_class_prob(p, y) <= 1.0


def test_donor_dist_sq_is_squared_distance_to_the_donor():
    assert donor_dist_sq(3.0, 5.0) == pytest.approx(4.0)
    assert donor_dist_sq(5.0, 5.0) == pytest.approx(0.0)


def test_donor_dist_sq_is_not_bounded_by_one():
    """Which is why toward_ablation's min_interval, calibrated in probability units,
    does not transfer to a regression sweep."""
    assert donor_dist_sq(100.0, 0.0) == pytest.approx(1e4)


# ── toward_ablation: where the patch landed between transfer and ablation ────────────

def test_toward_ablation_endpoints():
    # p_transfer = 0.30, ablating the concept would reach 0.80.
    assert toward_ablation(0.30, 0.80, 0.30) == pytest.approx(0.0)    # changed nothing
    assert toward_ablation(0.30, 0.80, 0.80) == pytest.approx(1.0)    # reached the ablation
    assert toward_ablation(0.30, 0.80, 0.55) == pytest.approx(0.5)    # halfway


def test_toward_ablation_is_invariant_to_the_direction_of_the_ablation():
    """Ablating a concept can move p[y] either way. Landing on the target is 1.0 in both
    directions -- if it were not, the term would reward one sign of concept."""
    assert toward_ablation(0.30, 0.80, 0.55) == pytest.approx(toward_ablation(0.80, 0.30, 0.55))


def test_toward_ablation_is_negative_when_the_patch_moves_the_wrong_way():
    """Not clipped: a patch that pushes the recipient FURTHER from the ablation is worse
    than doing nothing, and the objective's sign-preserving root keeps that visible."""
    assert toward_ablation(0.30, 0.80, 0.20) < 0


def test_toward_ablation_above_one_means_crossing():
    """The search rejects these (the crossing guard); the function still reports them, so
    the guard is one decision in one place rather than a clamp hidden in the arithmetic."""
    assert toward_ablation(0.30, 0.80, 1.05) > 1.0


def test_toward_ablation_small_interval_is_floored_not_gated():
    """A small interval is a MEASUREMENT -- the concept contributes little at this row --
    not a missing value. min_interval (0.01, borrowed from the sweeps' own min_gap)
    floors the DENOMINATOR: credit is capped at movement/min_interval, so full credit
    needs movement of at least the resolution floor, but the measured movement stays in
    the score with its sign. Gating returned nan here, and nan was a double defect at the
    call site: scored as 1.0 (FULL credit) and exempt from the crossing guard -- 59% of
    v17's chosen rows."""
    # interval 0.005, below the floor: denominator becomes 0.01
    assert toward_ablation(0.30, 0.305, 0.302) == pytest.approx(0.2)
    assert toward_ablation(0.30, 0.305, 0.30) == pytest.approx(0.0)
    # movement far past a near-zero target reads >1 and the crossing guard REJECTS it,
    # instead of the old gate waving it through as nan
    assert toward_ablation(0.30, 0.305, 0.32) == pytest.approx(2.0)
    # negative interval: the sign of the floored denominator follows the interval
    assert toward_ablation(0.80, 0.795, 0.79) == pytest.approx(1.0)
    # the floor is settable -- a regression sweep is in different units
    assert toward_ablation(0.30, 0.305, 0.302, min_interval=1e-6) == pytest.approx(0.4)


def test_toward_ablation_is_nan_only_for_the_genuinely_unmeasured():
    """Non-finite endpoints (or no recipient context at all, upstream). Everything
    measured stays a number."""
    assert np.isnan(toward_ablation(0.30, np.nan, 0.30))
    assert np.isnan(toward_ablation(0.30, np.inf, 0.30))
    assert np.isnan(toward_ablation(0.30, 0.80, np.nan))
    assert np.isnan(toward_ablation(np.nan, 0.80, 0.30))


# ── objective ────────────────────────────────────────────────────────────────

def _exp(**kw):
    """Temporarily set EXPONENTS, since the objective reads the module-level dict."""
    old = dict(EXPONENTS)
    EXPONENTS.update(kw)
    return old


def test_objective_is_monotone_in_every_term():
    base = objective(0.5, 0.5, 0.05, 0.9)
    assert objective(0.6, 0.5, 0.05, 0.9) > base       # more suppression is better
    assert objective(0.5, 0.6, 0.05, 0.9) > base       # more recipient movement is better
    assert objective(0.5, 0.5, 0.06, 0.9) < base       # more collateral is worse
    assert objective(0.5, 0.5, 0.05, 0.8) < base       # toward a tail is worse
    assert objective(0.5, 0.5, 0.05, 1.1) > base       # toward the density is better


def test_objective_equal_exponents_give_equal_sensitivity_to_relative_change():
    """The property the product-of-ratios form exists to have: at equal exponents a 10%
    change in any term is worth the same. This is what `1 + x` breaks, and it is the
    argument that retired `1 + blast` and, later, `1 + recon_excess`."""
    old = _exp(suppression=1.0, toward_ablation=1.0, blast=1.0, centrality=1.0)
    try:
        base = objective(0.5, 0.5, 0.05, 0.9)
        assert objective(0.55, 0.5, 0.05, 0.9) / base == pytest.approx(1.1)
        assert objective(0.5, 0.55, 0.05, 0.9) / base == pytest.approx(1.1)
        assert objective(0.5, 0.5, 0.05 / 1.1, 0.9) / base == pytest.approx(1.1)
        assert objective(0.5, 0.5, 0.05, 0.9 * 1.1) / base == pytest.approx(1.1)
    finally:
        EXPONENTS.update(old)


# ── centrality: position in the dataset's own reconstruction-loss distribution ─

def test_centrality_is_one_at_the_median_and_falls_in_both_tails():
    losses = np.sort(np.linspace(0.1, 0.5, 199))       # median 0.3
    mid = centrality(0.3, losses)
    assert mid == pytest.approx(1.0, abs=0.02)
    assert centrality(0.45, losses) < centrality(0.35, losses) < mid
    assert centrality(0.15, losses) < centrality(0.25, losses) < mid  # LOW tail too


def test_centrality_never_reaches_zero_beyond_the_observed_range():
    """Half-rank smoothing: a value beyond every real row keeps ~1/(n+1) of centrality,
    so the score degrades rather than zeroing -- and two candidates both beyond the range
    still compare by their other terms."""
    losses = np.sort(np.linspace(0.1, 0.5, 199))
    assert 0.0 < centrality(0.9, losses) < 0.02
    assert 0.0 < centrality(0.01, losses) < 0.02


def test_centrality_penalises_reconstructing_better_than_any_real_row():
    """The one-sided recon_excess charged nothing for a patch landing below every real
    row's loss (1.9% of v15's chosen patches). Both directions of atypical now cost."""
    losses = np.sort(np.linspace(0.1, 0.5, 199))
    assert centrality(0.01, losses) < centrality(0.3, losses)


def test_objective_rewards_moving_toward_the_density():
    """The term is the before/after centrality ratio: >1 the patch moved the row toward
    the crowd of real rows, <1 toward either tail. Within a row the start is a constant,
    so candidate selection is driven by where each candidate ENDS -- the ratio makes the
    recorded score readable, not the choice different."""
    losses = np.sort(np.linspace(0.1, 0.5, 199))
    start = centrality(0.45, losses)                    # row starts in the upper tail
    toward = centrality(0.32, losses) / start
    away = centrality(0.49, losses) / start
    assert toward > 1.0 > away
    assert objective(0.5, 0.5, 0.05, toward) > objective(0.5, 0.5, 0.05, away)


def test_objective_preserves_the_sign_of_toward_ablation():
    """A patch that moves the recipient opposite to the transfer must score below one that
    does nothing, not above it by way of a squared root."""
    assert objective(0.9, -0.5, 0.05, 1.0) < 0
    assert objective(0.9, -0.5, 0.05, 1.0) < objective(0.9, 0.0, 0.05, 1.0)


def test_objective_scales_as_a_power_when_all_exponents_scale():
    """Only exponent RATIOS matter: raising all of them by k raises the score to the k,
    which leaves the ranking untouched. This is why the exponent sweep fixes drop=1."""
    a = (0.5, 0.4, 0.05, 0.9)
    b = (0.7, 0.6, 0.02, 1.2)
    old = _exp(suppression=1.0, toward_ablation=1.0, blast=1.0, centrality=1.0)
    try:
        one = objective(*a), objective(*b)
        _exp(suppression=2.0, toward_ablation=2.0, blast=2.0, centrality=2.0)
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
    assert objective(0.5, 0.25, 0.0, 1.0) == pytest.approx(0.5 * 0.5 / EPS)


def test_objective_is_nan_when_toward_ablation_is_nan():
    """The caller substitutes 1.0 for a nan toward_ablation. Since the denominator floor, nan
    only reaches the caller where NO readout exists at all (carte, READOUT_EXCLUDED), and
    there 1.0 is genuinely neutral -- it removes the factor from the product, scoring the
    row on the donor-side terms. It was NOT neutral when measured-but-small intervals also
    came back nan: 1.0 is the top of toward_ablation's guarded range, so 59% of v17's rows got
    full recipient credit with the crossing guard bypassed."""
    assert np.isnan(objective(0.5, np.nan, 0.05, 1.0))
    assert objective(0.5, 1.0, 0.05, 1.0) == pytest.approx(objective(0.5, 1.0, 0.05, 1.0))


# ── attribution: the movement credited to c is observed minus bystanders ─────

def _fake_ctx(p_patched, p_transfer=0.50, p_ablated=0.60, loo_signed=(0.10, -0.03)):
    """Minimal recip context: two concepts, c = 7 and bystander 9, identity geometry so
    the delta arithmetic is inert and only the attribution arithmetic is exercised."""
    return {"fids": [7, 9], "B": np.eye(2), "signs": np.ones(2),
            "a_corpus": np.ones(2), "a_re": {7: 1.0, 9: 1.0},
            "predict": lambda d: np.array([[p_patched]] * len(d)),
            "loss": lambda pr: float(pr[0]),
            "p_transfer": p_transfer, "p_ablated": p_ablated,
            "loo_signed": list(loo_signed)}


def test_attribution_subtracts_the_bystanders_signed_share():
    """c suppressed fully, bystander 9 at ratio 0.8. Bystander's first-order share is
    (1 - 0.8) x (-0.03) = -0.006; observed movement 0.05 attributes to 0.056, and
    toward_ablation = 0.056 / 0.10."""
    out = recipient_toward_ablation(_fake_ctx(p_patched=0.55),
                                    [{7: 0.0, 9: 0.8}], feat=7)
    assert out[0]["movement_observed"] == pytest.approx(0.05)
    assert out[0]["est_bystander"] == pytest.approx(-0.006)
    assert out[0]["attribution_fallback"] is False
    assert out[0]["toward"] == pytest.approx(0.056 / 0.10)


def test_attribution_ignores_bystanders_that_did_not_move():
    """Ratio 1.0 contributes (1 - 1.0) x anything = 0: with no bystander movement the
    attributed and observed movements coincide."""
    out = recipient_toward_ablation(_fake_ctx(p_patched=0.55),
                                    [{7: 0.0, 9: 1.0}], feat=7)
    assert out[0]["est_bystander"] == pytest.approx(0.0)
    assert out[0]["toward"] == pytest.approx(0.05 / 0.10)


def test_attribution_falls_back_when_the_correction_leaves_the_probability_range():
    """A first-order estimate that implies movement outside [-1, 1] is out-of-model by
    construction (one v19-tested row overshot by 1.68). The UNCORRECTED movement is
    used and the fallback recorded -- no chosen constant involved."""
    out = recipient_toward_ablation(
        _fake_ctx(p_patched=0.55, loo_signed=(0.10, -20.0)),
        [{7: 0.0, 9: 0.8}], feat=7)
    assert out[0]["attribution_fallback"] is True
    assert out[0]["toward"] == pytest.approx(0.05 / 0.10)   # observed, uncorrected


# ── probe_effectiveness: the --rank-by effectiveness pass-1 ordering ─────────

def test_probe_effectiveness_is_gain_over_dL_when_bystanders_hold_still():
    a_base = np.array([2.0, 1.0])
    a_vec = np.array([1.0, 1.0])                       # c halved, bystander untouched
    # gain = (1.0/2.0) x 0.2 = 0.1; dL = 0.5 -> 0.2
    assert probe_effectiveness(a_vec, a_base, 0, np.array([1]), {1: 0.5},
                               interval=0.2, dL=0.5) == pytest.approx(0.2)


def test_probe_effectiveness_charges_loo_weighted_spend():
    a_base = np.array([2.0, 1.0])
    a_vec = np.array([1.0, 1.2])                       # bystander moved 20% of itself
    # spend = 0.2 x 0.5 = 0.1 -> net = (0.1 - 0.1)/0.5 = 0
    assert probe_effectiveness(a_vec, a_base, 0, np.array([1]), {1: 0.5},
                               interval=0.2, dL=0.5) == pytest.approx(0.0)
    # a bystander the prediction ignores costs nothing
    assert probe_effectiveness(a_vec, a_base, 0, np.array([1]), {1: 0.0},
                               interval=0.2, dL=0.5) == pytest.approx(0.2)


def test_probe_effectiveness_ignores_inactive_bystanders():
    a_base = np.array([2.0, ACTIVE_FLOOR / 10])
    a_vec = np.array([1.0, 1.0])                       # huge relative move on a dead concept
    assert probe_effectiveness(a_vec, a_base, 0, np.array([1]), {1: 0.5},
                               interval=0.2, dL=0.5) == pytest.approx(0.2)


def test_probe_effectiveness_degenerates_to_least_spend_on_tiny_intervals():
    """When c's interval is ~0, gain is ~0 for every column and net is minus spend/dL:
    the ordering becomes collateral-avoidance -- what LOO-weighting means when the
    prediction barely depends on c. Stated behaviour, not an accident."""
    a_base = np.array([2.0, 1.0])
    quiet = probe_effectiveness(np.array([1.0, 1.01]), a_base, 0, np.array([1]),
                                {1: 0.5}, interval=1e-6, dL=0.5)
    loud = probe_effectiveness(np.array([1.0, 1.5]), a_base, 0, np.array([1]),
                               {1: 0.5}, interval=1e-6, dL=0.5)
    assert loud < quiet < 0.001


def test_probe_effectiveness_rejects_degenerate_dL():
    a = np.array([2.0, 1.0])
    assert probe_effectiveness(a, a, 0, np.array([1]), {1: 0.5}, 0.2, 0.0) == float("-inf")


# ── gap_opened: a METRIC, never an objective term ────────────────────────────

def test_gap_opened_is_the_attributed_share_of_the_original_gap():
    """Weak 0.20, strong 0.80: original disagreement 0.60. Transfer took the recipient
    to 0.75; the patch moved it back by an attributed 0.06 toward weak -> re-opened 10%
    of the original gap."""
    g = gap_opened_metric(movement_observed=-0.05, est_bystander=0.01, fallback=False,
                          p_weak=0.20, p_transfer=0.75, p_strong=0.80)
    assert g == pytest.approx((-0.05 - 0.01) * -1.0 / 0.60)   # +0.10: re-opened


def test_gap_opened_is_signed_and_negative_when_the_patch_closes_further():
    g = gap_opened_metric(movement_observed=0.03, est_bystander=0.0, fallback=False,
                          p_weak=0.20, p_transfer=0.75, p_strong=0.80)
    assert g == pytest.approx(-0.05)                          # moved toward strong


def test_gap_opened_uses_the_uncorrected_movement_on_fallback():
    g = gap_opened_metric(movement_observed=-0.06, est_bystander=-5.0, fallback=True,
                          p_weak=0.20, p_transfer=0.75, p_strong=0.80)
    assert g == pytest.approx(0.10)


def test_gap_opened_is_none_when_unmeasured():
    """No readout, regression (p_strong None), or zero gap -- None, not zero."""
    assert gap_opened_metric(None, None, None, 0.2, 0.75, 0.8) is None
    assert gap_opened_metric(-0.05, 0.0, False, 0.2, 0.75, None) is None
    assert gap_opened_metric(-0.05, 0.0, False, 0.5, 0.5, 0.5) is None


# ── collateral ───────────────────────────────────────────────────────────────

def _recip(loo_by_fid, interval=0.50):
    return {"loo_by_fid": loo_by_fid, "interval": interval}


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


def test_weighted_blast_is_scaled_to_what_ablating_c_moves():
    """spend / max(|interval|, MIN_GAP): bystander spend in units of the concept under
    test's own effect -- the 2026-08-14 strictness redesign. The transfer's movement is
    no longer the scale; the same spend against a weak concept must read LARGER."""
    a, moved = np.array([1.0, 1.0]), np.array([1.0, 1.5])
    strong = weighted_blast(a, moved, np.array([1]), _recip({1: 0.4}, interval=0.50))
    weak = weighted_blast(a, moved, np.array([1]), _recip({1: 0.4}, interval=0.05))
    assert strong == pytest.approx(0.5 * 0.4 / 0.50)
    assert weak == pytest.approx(0.5 * 0.4 / 0.05)
    assert weak > strong


def test_weighted_blast_floors_the_interval_at_min_gap():
    """A sub-floor interval is capped at the resolution bound, not divided by noise --
    strict (spend / 0.01) but bounded."""
    a, moved = np.array([1.0, 1.0]), np.array([1.0, 1.5])
    r = weighted_blast(a, moved, np.array([1]), _recip({1: 0.4}, interval=0.002))
    assert r == pytest.approx(0.5 * 0.4 / MIN_GAP)


def test_weighted_blast_is_none_without_a_measured_interval():
    a, moved = np.array([1.0, 1.0]), np.array([1.0, 1.5])
    assert weighted_blast(a, moved, np.array([1]), _recip({1: 0.4}, interval=float("nan"))) is None


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
    assert m["target_moved_frac"] == pytest.approx(0.5)
    assert m["n_others_moved_gt_10pct"] == 0
