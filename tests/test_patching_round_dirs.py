"""The patching prerequisites follow the round like every other stage.

patch_search.py reads its concept list from the burndown, which is derived from the
off-manifold stratification dump of the round's forward deltas, whose firing densities
come from the round's concept-activation cache, whose dataset ranking comes from the
round's dataset-quality cache. All four were hard-wired to the round-10 locations, so a
round-11 burndown would have ranked TabDPT concepts by an SAE that no longer exists.
Every one of them now resolves through scripts/round_paths.py, and promote_round links
the unchanged models' caches so only the regenerated model is rebuilt.
"""
import json
from pathlib import Path

from scripts import round_paths
from scripts._project_root import PROJECT_ROOT


def _under_results(p: Path) -> bool:
    return round_paths.RESULTS_DIR in Path(p).parents


def test_round_paths_name_the_patching_prerequisites():
    assert _under_results(round_paths.CONCEPT_ACTIVATIONS_DIR)
    assert _under_results(round_paths.QUALITY_CACHE_FILE)
    assert _under_results(round_paths.CONTRASTIVE_EXAMPLES_DIR)
    assert _under_results(round_paths.off_manifold_dump_file("trained"))
    assert round_paths.off_manifold_dump_file("random") != round_paths.off_manifold_dump_file("trained")


def test_concept_activation_cache_reads_and_writes_the_round():
    from scripts.sae import build_concept_activation_cache as b

    assert b.OUTPUT_DIR == round_paths.CONCEPT_ACTIVATIONS_DIR
    assert Path(b.sae_checkpoint("tabdpt")).parent.parent == round_paths.sae_sweep_dir()
    assert Path(b.test_corpus("tabdpt")).parent == round_paths.sae_training_dir()


def test_quality_cache_default_path_and_corpus_follow_the_round():
    from scripts.concepts import build_dataset_quality_cache as q
    from scripts.concepts.dataset_quality_cache import DEFAULT_CACHE_PATH

    assert DEFAULT_CACHE_PATH == round_paths.QUALITY_CACHE_FILE
    assert q.SAE_DATA_DIR == round_paths.sae_training_dir()


def test_quality_cache_carry_over_copies_only_models_not_rebuilt(tmp_path):
    from scripts.concepts.build_dataset_quality_cache import carry_over_models

    old = tmp_path / "old.json"
    old.write_text(json.dumps({
        "metadata": {"cache_version": 2},
        "models": {"tabpfn": {"features": {"1": {}}}, "tabdpt": {"features": {"9": {}}}},
    }))
    cache = {"metadata": {"cache_version": 2}, "models": {"tabdpt": {"features": {"2": {}}}}}
    carry_over_models(cache, old, rebuilt=["tabdpt"])
    assert set(cache["models"]) == {"tabpfn", "tabdpt"}
    assert cache["models"]["tabdpt"] == {"features": {"2": {}}}  # rebuilt entry wins
    assert cache["metadata"]["carried_over"] == {"tabpfn": str(old)}


def test_contrastive_examples_read_and_write_the_round():
    from scripts.concepts import build_contrastive_examples as c

    assert c.SAE_DATA_DIR == round_paths.sae_training_dir()
    assert c.OUTPUT_DIR == round_paths.CONTRASTIVE_EXAMPLES_DIR


def test_stratification_reads_the_round_for_both_arms():
    from scripts.rebuttal import off_manifold_concept_stratification as s

    fwd, cache = s.arm_inputs("trained")
    assert fwd == round_paths.FORWARD_DELTAS_DIR
    assert cache == round_paths.TRANSFER_CACHES_DIR / "global_trained"
    fwd, cache = s.arm_inputs("random")
    assert fwd == round_paths.FORWARD_DELTAS_RANDOM_DIR
    assert cache == round_paths.TRANSFER_CACHES_DIR / "global_random"
    assert s.ACTIVATIONS_DIR == round_paths.CONCEPT_ACTIVATIONS_DIR
    assert s.default_dump_out("trained") == round_paths.off_manifold_dump_file("trained")


def test_burndown_defaults_follow_the_round():
    from scripts.rebuttal import build_patching_burndown as b

    assert b.DEFAULT_DUMP == round_paths.off_manifold_dump_file("trained")
    assert b.DEFAULT_OUT == round_paths.PATCHING_BURNDOWN_FILE
    assert b.CONTRASTIVE_DIR == round_paths.CONTRASTIVE_EXAMPLES_DIR


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def test_promote_round_links_labeling_caches_except_the_regenerated_model(tmp_path):
    from scripts.sae_corpus.promote_round import promote_round

    for m in ("tabpfn", "tabdpt"):
        _touch(tmp_path / "sae_training_round10" / f"{m}_taskaware_sae_test.npz")
        _touch(tmp_path / "sae_tabarena_sweep_round10" / m / "sae_matryoshka_archetypal_validated.pt")
        _touch(tmp_path / "concept_activations_cache" / m / "adult.npz")   # round 10: legacy path
        _touch(tmp_path / "contrastive_examples" / m / "f1_adult.csv")
    promote_round(tmp_path, src_round=10, dst_round=11, regenerate=["tabdpt"])

    acts = tmp_path / "round11" / "concept_activations_cache"
    assert (acts / "tabpfn").is_symlink() and (acts / "tabpfn" / "adult.npz").read_bytes() == b"x"
    assert not (acts / "tabdpt").exists()
    ex = tmp_path / "round11" / "contrastive_examples"
    assert (ex / "tabpfn").is_symlink()
    assert not (ex / "tabdpt").exists()


def test_promote_round_reads_labeling_caches_from_the_round_tree_after_round10(tmp_path):
    from scripts.sae_corpus.promote_round import promote_round

    for m in ("tabpfn", "tabdpt"):
        _touch(tmp_path / "sae_training_round11" / f"{m}_taskaware_sae_test.npz")
        _touch(tmp_path / "sae_tabarena_sweep_round11" / m / "sae_matryoshka_archetypal_validated.pt")
        _touch(tmp_path / "round11" / "concept_activations_cache" / m / "adult.npz")
    promote_round(tmp_path, src_round=11, dst_round=12, regenerate=["tabpfn"])
    assert (tmp_path / "round12" / "concept_activations_cache" / "tabdpt").is_symlink()
    assert not (tmp_path / "round12" / "concept_activations_cache" / "tabpfn").exists()
