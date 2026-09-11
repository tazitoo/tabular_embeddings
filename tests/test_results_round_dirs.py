"""Every rerun stage writes under output/round{N}/ and reads its inputs from there.

The NeurIPS-era results (output/perrow_importance, output/rebuttal/symmetric_*,
forward_deltas, functional_decomposition, patch_search.json, the matching JSONs) are the
published record and are never rewritten. One constant, RESULTS_DIR, derived from
DEFAULT_SAE_ROUND, names the round's results root; each stage's defaults hang off it so a
bare launch cannot land in a legacy directory.
"""
import importlib.util

from scripts._project_root import PROJECT_ROOT


def _by_path(rel):
    path = PROJECT_ROOT / rel
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _under_results(p):
    from scripts.intervention.intervene_lib import RESULTS_DIR
    return RESULTS_DIR in p.parents or p == RESULTS_DIR


def test_results_dir_is_the_default_round():
    from scripts.intervention.intervene_lib import RESULTS_DIR
    from scripts.sae.compare_sae_cross_model import DEFAULT_SAE_ROUND

    assert RESULTS_DIR == PROJECT_ROOT / "output" / f"round{DEFAULT_SAE_ROUND}"


def test_perrow_importance_and_baseline_cache_write_under_results_dir():
    from scripts.intervention import cache_baseline_predictions, perrow_importance

    assert _under_results(perrow_importance.OUTPUT_DIR)
    assert _under_results(cache_baseline_predictions.OUTPUT_DIR)


def test_symmetric_sweeps_read_and_write_under_results_dir():
    from scripts.intervention.intervene_lib import DEFAULT_MATCHING_FILE, IMPORTANCE_DIR
    from scripts.rebuttal import ablation_sweep_symmetric as abl
    from scripts.rebuttal import transfer_sweep_symmetric as tr

    for mod in (abl, tr):
        assert _under_results(mod.OUTPUT_DIR)
        assert mod.IMPORTANCE_DIR == IMPORTANCE_DIR
        assert mod.DEFAULT_MATCHING_FILE == DEFAULT_MATCHING_FILE
    assert _under_results(IMPORTANCE_DIR)
    assert _under_results(DEFAULT_MATCHING_FILE)


def test_decomposition_and_patching_follow_forward_deltas_under_results_dir():
    from scripts.intervention.intervene_lib import FORWARD_DELTAS_DIR, IMPORTANCE_DIR
    from scripts.rebuttal import functional_decomposition as fd
    from scripts.rebuttal import patch_search as ps

    assert _under_results(FORWARD_DELTAS_DIR)
    assert fd.FWD_DIR == FORWARD_DELTAS_DIR
    assert _under_results(fd.OUT_DIR)
    assert ps.FWD == FORWARD_DELTAS_DIR
    assert ps.IMPORTANCE == IMPORTANCE_DIR
    assert _under_results(ps.ATOMS)
    assert _under_results(ps.DEFAULT_OUT)
    assert _under_results(ps.DEFAULT_BURNDOWN)


def test_transfer_cache_builder_uses_default_round_sae_and_results_dir():
    from scripts.analysis import build_transfer_caches as btc
    from scripts.sae.compare_sae_cross_model import sae_sweep_dir

    from scripts.round_paths import random_sae_dir

    assert btc.SAE_DIRS["trained"] == sae_sweep_dir()
    assert btc.SAE_DIRS["random"] == random_sae_dir()
    assert _under_results(btc.OUT_ROOT)


def test_matching_pipeline_defaults_live_under_results_dir():
    from scripts.intervention.intervene_lib import (
        CROSS_CORR_DIR, CROSS_MODEL_BASELINE_FILE, DEFAULT_MATCHING_FILE,
    )

    m01 = _by_path("scripts/matching/01_match_sae_concepts_mnn.py")
    m02 = _by_path("scripts/matching/02_build_concept_graph.py")
    assert m01.DEFAULT_MATCHING_FILE == DEFAULT_MATCHING_FILE
    assert m01.CROSS_CORR_DIR == CROSS_CORR_DIR
    assert m01.CROSS_MODEL_BASELINE_FILE == CROSS_MODEL_BASELINE_FILE
    assert m02.DEFAULT_MATCHING_FILE == DEFAULT_MATCHING_FILE
    assert m02.CROSS_CORR_DIR == CROSS_CORR_DIR
    assert _under_results(CROSS_CORR_DIR)
    assert _under_results(CROSS_MODEL_BASELINE_FILE)
