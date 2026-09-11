"""Round 11 = round 10 with one model's corpus and SAE regenerated.

Existing results are never overwritten: a new round directory is seeded with relative
symlinks to the previous round's files for every model that is NOT being regenerated,
and the regenerated model's files are written fresh into it. Every script that resolves
its SAE inputs from DEFAULT_SAE_ROUND then follows the new round without edits.
"""
import os
from pathlib import Path

import numpy as np


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def _seed_round10(root: Path):
    for m in ("tabpfn", "tabdpt"):
        _touch(root / "sae_training_round10" / f"{m}_taskaware_sae_test.npz")
        _touch(root / "sae_training_round10" / f"{m}_taskaware_norm_stats.npz")
        _touch(root / "sae_tabarena_sweep_round10" / m / "sae_matryoshka_archetypal_validated.pt")
    _touch(root / "sae_training_round10" / "layer_comparison_plot.png")  # not a model file
    for m in ("tabpfn", "tabdpt"):
        _touch(root / "sae_random_baseline_round10" / m / "sae_matryoshka_archetypal_validated.pt")


def test_promote_round_links_every_model_except_the_regenerated_one(tmp_path):
    from scripts.sae_corpus.promote_round import promote_round

    _seed_round10(tmp_path)
    linked = promote_round(tmp_path, src_round=10, dst_round=11, regenerate=["tabdpt"])

    dst = tmp_path / "sae_training_round11"
    assert (dst / "tabpfn_taskaware_sae_test.npz").is_symlink()
    assert (dst / "tabpfn_taskaware_norm_stats.npz").is_symlink()
    assert not (dst / "tabdpt_taskaware_sae_test.npz").exists()
    assert not (dst / "layer_comparison_plot.png").exists()
    sweep = tmp_path / "sae_tabarena_sweep_round11"
    assert (sweep / "tabpfn").is_symlink()
    assert not (sweep / "tabdpt").exists()
    rnd = tmp_path / "sae_random_baseline_round11"
    assert (rnd / "tabpfn").is_symlink()
    assert not (rnd / "tabdpt").exists()
    assert sorted(linked) == ["tabpfn"]


def test_promote_round_tolerates_a_missing_random_baseline_dir(tmp_path):
    from scripts.sae_corpus.promote_round import promote_round

    _seed_round10(tmp_path)
    (tmp_path / "sae_random_baseline_round10" / "tabpfn" / "sae_matryoshka_archetypal_validated.pt").unlink()
    (tmp_path / "sae_random_baseline_round10" / "tabpfn").rmdir()
    (tmp_path / "sae_random_baseline_round10" / "tabdpt" / "sae_matryoshka_archetypal_validated.pt").unlink()
    (tmp_path / "sae_random_baseline_round10" / "tabdpt").rmdir()
    (tmp_path / "sae_random_baseline_round10").rmdir()
    promote_round(tmp_path, src_round=10, dst_round=11, regenerate=["tabdpt"])
    assert (tmp_path / "sae_training_round11" / "tabpfn_taskaware_sae_test.npz").is_symlink()


def test_random_sae_dir_is_round_tagged():
    from scripts._project_root import PROJECT_ROOT
    from scripts.round_paths import DEFAULT_SAE_ROUND, random_sae_dir

    assert random_sae_dir() == PROJECT_ROOT / "output" / f"sae_random_baseline_round{DEFAULT_SAE_ROUND}"


def test_promote_round_links_are_relative_and_resolve(tmp_path):
    from scripts.sae_corpus.promote_round import promote_round

    _seed_round10(tmp_path)
    promote_round(tmp_path, src_round=10, dst_round=11, regenerate=["tabdpt"])
    link = tmp_path / "sae_training_round11" / "tabpfn_taskaware_sae_test.npz"
    assert not os.path.isabs(os.readlink(link))
    assert link.resolve() == (tmp_path / "sae_training_round10" / "tabpfn_taskaware_sae_test.npz").resolve()
    assert (tmp_path / "sae_tabarena_sweep_round11" / "tabpfn" /
            "sae_matryoshka_archetypal_validated.pt").read_bytes() == b"x"


def test_promote_round_is_idempotent(tmp_path):
    from scripts.sae_corpus.promote_round import promote_round

    _seed_round10(tmp_path)
    promote_round(tmp_path, src_round=10, dst_round=11, regenerate=["tabdpt"])
    promote_round(tmp_path, src_round=10, dst_round=11, regenerate=["tabdpt"])
    assert (tmp_path / "sae_training_round11" / "tabpfn_taskaware_sae_test.npz").is_symlink()


def test_sweep_reads_prebuilt_corpus_from_the_default_round():
    from scripts.sae.compare_sae_cross_model import DEFAULT_SAE_ROUND
    from scripts.sae.sae_tabarena_sweep import prebuilt_corpus_dir
    from scripts._project_root import PROJECT_ROOT

    assert prebuilt_corpus_dir() == PROJECT_ROOT / "output" / f"sae_training_round{DEFAULT_SAE_ROUND}"


def test_corpus_builder_writes_to_the_default_round():
    import importlib.util

    from scripts.sae.compare_sae_cross_model import DEFAULT_SAE_ROUND
    from scripts._project_root import PROJECT_ROOT

    path = PROJECT_ROOT / "scripts" / "sae_corpus" / "07_build_sae_training_data.py"
    spec = importlib.util.spec_from_file_location("build_sae_training_data_07", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert builder.default_output_dir() == PROJECT_ROOT / "output" / f"sae_training_round{DEFAULT_SAE_ROUND}"


def test_default_round_is_11():
    from scripts.sae.compare_sae_cross_model import DEFAULT_SAE_ROUND

    assert DEFAULT_SAE_ROUND == 11
