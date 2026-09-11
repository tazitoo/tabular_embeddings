"""The paper's SAE selection rule, applied to a sweep's Optuna study.

Among trials with R² >= 0.80, alive >= 0.80 and cross-seed stability >= 0.75, pick the
least complex: smallest sqrt(d_hidden * L0). When no trial qualifies the sweep's own
validated choice (Optuna best by the efficiency objective) stands, and the report must
say so rather than silently promoting a non-qualifying trial.
"""
import numpy as np
import pytest

from scripts._project_root import PROJECT_ROOT

ROUND10_TABDPT_DB = (PROJECT_ROOT / "output" / "sae_tabarena_sweep_round10" / "tabdpt"
                     / "tabdpt_matryoshka_archetypal.db")


def _trials():
    """Three synthetic trials: two qualify, one fails the alive floor."""
    return [
        dict(trial=1, expansion=4, topk=64, l0=64.0, alive=0.90, stability=0.80,
             test_recon=0.15, input_dim=768, value=10.0),
        dict(trial=2, expansion=4, topk=32, l0=32.0, alive=0.85, stability=0.78,
             test_recon=0.18, input_dim=768, value=9.0),
        dict(trial=3, expansion=4, topk=16, l0=16.0, alive=0.60, stability=0.90,
             test_recon=0.19, input_dim=768, value=8.0),
    ]


def test_select_picks_least_complex_qualifier():
    from scripts.sae.select_sae_config import select

    chosen, qualifiers = select(_trials(), var_per_dim=1.0)
    assert [q["trial"] for q in qualifiers] == [1, 2]
    assert chosen["trial"] == 2  # sqrt(3072*32) < sqrt(3072*64)


def test_select_reports_no_qualifier_instead_of_promoting_one():
    from scripts.sae.select_sae_config import select

    trials = [dict(t, alive=0.5) for t in _trials()]
    chosen, qualifiers = select(trials, var_per_dim=1.0)
    assert chosen is None and qualifiers == []


def test_r2_uses_the_corpus_variance():
    from scripts.sae.select_sae_config import r2

    assert r2(test_recon=0.25, var_per_dim=1.0) == pytest.approx(0.75)
    assert r2(test_recon=0.25, var_per_dim=0.5) == pytest.approx(0.5)


@pytest.mark.skipif(not ROUND10_TABDPT_DB.exists(), reason="round-10 study not on this host")
def test_round10_tabdpt_study_loads_and_had_no_qualifier():
    """Round 10 recorded that TabDPT fell back to the Optuna best (no trial cleared the
    alive floor); the loader must reproduce that reading from the study itself."""
    from scripts.sae.select_sae_config import load_trials, select

    trials = load_trials(ROUND10_TABDPT_DB)
    assert len(trials) == 30
    chosen, qualifiers = select(trials, var_per_dim=1.0)
    assert chosen is None
    assert min(trials, key=lambda t: t["value"])["trial"] == 17  # Optuna number (0-based)
