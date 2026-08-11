"""The query-window property the patch search's batching depends on.

The search evaluates many candidate rows in one forward. That is only sound if a row's
activation is a function of the row and the fixed context -- not of where it sits in the
query set, and not of which other rows share the window. These tests assert exactly that,
per donor model.

Why this is a test and not a runtime check. It used to be measured inside the sweep, once
per (donor, dataset) cell, and branched on: a failure switched the search to a slow
per-candidate path. Re-deriving a fixed property every run makes it look like something to
handle rather than something to rely on, and that is how an unexplained measurement stood
as an architectural constraint for a whole sweep. It also mismeasured: it grew the query
set by appending duplicates, so it detected sensitivity to query-set SIZE and reported it
as rows interacting. Mitra is exactly invariant to order and to companions at fixed count;
only changing the count moves it.

So the evaluator's contract is a FIXED-SIZE window filled by REPLACING rows, and these
tests are what license it. A failure here is a bug to fix, not a case to handle.

Requires the model weights and the preprocessing cache, so these skip on a machine that
has neither -- they are meant to run on a worker.
"""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from data.preprocessing import CACHE_DIR  # noqa: E402

DATASET = "Marketing_Campaign"
WINDOW = 32
TOL = 1e-4          # the same threshold the old runtime check used


def _ctx(model):
    from scripts.intervention.intervene_lib import SPLITS_PATH, load_dataset_context
    splits = json.loads(SPLITS_PATH.read_text())
    Xtr, ytr, Xq, _, _, task = load_dataset_context(model, DATASET, splits)
    if ytr.dtype == np.int32:
        ytr = ytr.astype(np.int64)
    return Xtr, ytr, Xq, task


def _acts(model, Xtr, ytr, rows, task):
    from scripts.rebuttal.patch_search import extract_acts
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a, _ = extract_acts(model, DATASET, Xtr, ytr, rows, task, device)
    return a


pytestmark = pytest.mark.skipif(
    not (CACHE_DIR / "tabpfn" / f"{DATASET}.npz").exists(),
    reason="preprocessing cache not available (run on a worker)",
)


@pytest.mark.parametrize("model", ["tabpfn", "tabicl", "mitra", "tabdpt"])
def test_activation_is_invariant_to_query_order(model):
    """Permuting the query set must not change any row's activation."""
    Xtr, ytr, Xq, task = _ctx(model)
    Xq = Xq[:WINDOW]
    a1 = _acts(model, Xtr, ytr, Xq, task)

    perm = np.random.RandomState(0).permutation(len(Xq))
    a2 = _acts(model, Xtr, ytr, Xq[perm], task)

    shift = float(np.abs(a2[np.argsort(perm)] - a1).max())
    assert shift < TOL, f"{model}: query order changed activations by {shift:.3e}"


@pytest.mark.parametrize("model", ["tabpfn", "tabicl", "mitra", "tabdpt"])
def test_activation_is_invariant_to_companions_at_fixed_count(model):
    """A row's activation must not depend on which other rows share the window.

    This is the property the evaluator relies on: candidates are swapped into a
    constant-size window, so each candidate sits alongside different neighbours.
    """
    Xtr, ytr, Xq, task = _ctx(model)
    test = Xq[0:1]
    companions_a = Xq[1:WINDOW]
    companions_b = Xq[WINDOW:2 * WINDOW - 1]
    assert len(companions_a) == len(companions_b)

    a_a = _acts(model, Xtr, ytr, np.vstack([test, companions_a]), task)[0]
    a_b = _acts(model, Xtr, ytr, np.vstack([test, companions_b]), task)[0]

    shift = float(np.abs(a_b - a_a).max())
    assert shift < TOL, f"{model}: companions changed the test row by {shift:.3e}"


@pytest.mark.parametrize("model", ["tabpfn", "tabicl", "mitra", "tabdpt"])
def test_activation_is_invariant_to_position_in_window(model):
    """Moving the row under test from the front to the back must change nothing."""
    Xtr, ytr, Xq, task = _ctx(model)
    test, block = Xq[0:1], Xq[1:WINDOW]

    front = _acts(model, Xtr, ytr, np.vstack([test, block]), task)[0]
    back = _acts(model, Xtr, ytr, np.vstack([block, test]), task)[-1]

    shift = float(np.abs(back - front).max())
    assert shift < TOL, f"{model}: position in the window changed the row by {shift:.3e}"


def test_mitra_is_sensitive_to_window_size():
    """Documents WHY the window is fixed, and fails if that reason ever stops holding.

    Growing the query set moves mitra's existing rows (measured 2.42), which is what the
    old runtime check detected and misread as rows interacting. If this ever passes,
    appending has become safe and the fixed-window constraint can be revisited -- so it is
    asserted rather than left as a comment.
    """
    Xtr, ytr, Xq, task = _ctx("mitra")
    base = Xq[:64]
    a1 = _acts("mitra", Xtr, ytr, base, task)
    dup = base[np.linspace(0, len(base) - 1, 8).astype(int)]
    a2 = _acts("mitra", Xtr, ytr, np.vstack([base, dup]), task)

    shift = float(np.abs(a2[:len(base)] - a1).max())
    assert shift > TOL, (
        "mitra no longer moves when the query set grows; the fixed-window constraint "
        f"may be unnecessary (shift {shift:.3e})")
