"""retrain_selected can take its hyperparameters from a sweep study and override the
structural ones (expansion, top-k), writing to a caller-chosen directory.

This is how one-off candidate SAEs are trained for comparison without touching the
sweep's own validated checkpoint.
"""
import pytest

from scripts._project_root import PROJECT_ROOT

ROUND10_TABDPT_DB = (PROJECT_ROOT / "output" / "sae_tabarena_sweep_round10" / "tabdpt"
                     / "tabdpt_matryoshka_archetypal.db")

PARAM_KEYS = {"expansion", "topk", "sparsity_penalty", "learning_rate",
              "archetypal_temp", "archetypal_n", "archetypal_relaxation"}


@pytest.mark.skipif(not ROUND10_TABDPT_DB.exists(), reason="round-10 study not on this host")
def test_params_from_study_returns_the_trials_hyperparameters():
    from scripts.sae.retrain_selected import params_from_study

    p = params_from_study(ROUND10_TABDPT_DB, trial=17)
    assert set(p) == PARAM_KEYS | {"trial"}
    assert p["trial"] == 17 and p["expansion"] == 4 and p["topk"] == 16


def test_overrides_replace_only_the_structural_params():
    from scripts.sae.retrain_selected import with_overrides

    base = dict(trial=13, expansion=4, topk=128, sparsity_penalty=1e-3, learning_rate=1e-4,
                archetypal_temp=0.2, archetypal_n=1000, archetypal_relaxation=0.5)
    out = with_overrides(base, expansion=2, topk=64)
    assert out["expansion"] == 2 and out["topk"] == 64
    assert {k: out[k] for k in PARAM_KEYS - {"expansion", "topk"}} == \
        {k: base[k] for k in PARAM_KEYS - {"expansion", "topk"}}
    assert with_overrides(base) == base


def test_candidate_dir_is_tagged_and_outside_the_sweep_checkpoint_dir():
    from scripts.round_paths import sae_sweep_dir
    from scripts.sae.retrain_selected import candidate_dir

    d = candidate_dir("tabdpt", "t13_2x_k64")
    assert d == sae_sweep_dir() / "tabdpt_candidates" / "t13_2x_k64"
    assert d != sae_sweep_dir() / "tabdpt"
