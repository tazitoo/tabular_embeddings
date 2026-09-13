"""The patching population is one rule applied to every accepted concept.

Rule (round 11, replaces the submission's acceptance window [200, 499]): off-manifold
fraction in [0.6, 0.8), acceptance >= 200 rows, firing density < 0.90. The floor keeps
the patch search from being data-limited; the density cap excludes always-on concepts,
which fire on nearly every row and offer no non-firing contrast for a suppression patch.
The old upper acceptance bound was a compute bound that, once transfers accepted more
atoms per row, cut off the concepts with the most rows.
"""
from scripts.rebuttal.off_manifold_concept_stratification import DEFAULT_CELL, in_patching_cell


def _row(off, acc, density):
    # (donor, feat, off_frac, acceptance, universality, density, n_datasets)
    return ("tabpfn", 1, off, acc, 3, density, 10)


def test_defaults_are_rule_b():
    assert DEFAULT_CELL == {"off_lo": 0.6, "off_hi": 0.8, "acc_lo": 200,
                            "acc_hi": float("inf"), "density_max": 0.90}


def test_cell_membership():
    assert in_patching_cell(_row(0.7, 250, 0.5))
    assert in_patching_cell(_row(0.7, 3000, 0.5))       # no upper acceptance bound
    assert not in_patching_cell(_row(0.7, 199, 0.5))    # below the acceptance floor
    assert not in_patching_cell(_row(0.7, 250, 0.90))   # always-on excluded (cap is exclusive)
    assert not in_patching_cell(_row(0.59, 250, 0.5))
    assert not in_patching_cell(_row(0.80, 250, 0.5))   # band is half-open


def test_submission_window_is_still_expressible():
    old = dict(DEFAULT_CELL, acc_hi=499, density_max=float("inf"))
    assert in_patching_cell(_row(0.7, 250, 0.99), **old)
    assert not in_patching_cell(_row(0.7, 500, 0.5), **old)
