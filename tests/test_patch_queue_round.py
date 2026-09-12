"""patch_queue dispatches a round's patching sweep from that round's burndown.

The dispatcher used to pool concepts from a previous run's per-concept files and write
under output/rebuttal/<run>, the NeurIPS record. A round re-derives its concept list
(the burndown), writes every per-concept file under output/round{N}/patch_runs/<run>
on every host, and runs the model with the default CUDA allocator like every other
round-11 launcher (expandable_segments changed Mitra's numerics on wide datasets).
"""
from pathlib import Path

from scripts import round_paths


def test_run_dir_lives_under_the_round_on_mac_and_remote():
    from scripts.rebuttal.patch_queue import REPO, remote_run_dir, run_dir

    assert run_dir("r11v31") == round_paths.PATCH_RUNS_DIR / "r11v31"
    assert round_paths.RESULTS_DIR in run_dir("r11v31").parents
    rel = run_dir("r11v31").relative_to(round_paths.PROJECT_ROOT)
    assert remote_run_dir("r11v31") == f"{REPO}/{rel}"


def test_pool_from_burndown_keeps_impact_order_and_dedupes(tmp_path):
    from scripts.rebuttal.patch_queue import pool_from_burndown

    bd = tmp_path / "patching_burndown.csv"
    bd.write_text("donor,feat_id,off_frac,off_mass_share\n"
                  "tabdpt,38,0.7,0.02\ntabicl,5,0.65,0.01\ntabdpt,38,0.7,0.02\n")
    assert pool_from_burndown(bd) == [("tabdpt", 38), ("tabicl", 5)]


def test_launch_env_uses_the_default_cuda_allocator():
    from scripts.rebuttal.patch_queue import ENV

    assert "expandable_segments" not in ENV
    assert "OMP_NUM_THREADS=8" in ENV


def test_slot_spec_parses_host_gpu_pairs():
    from scripts.rebuttal.patch_queue import parse_slots

    assert parse_slots(["morg.local:1", "terrax.local:0"]) == [("morg.local", 1), ("terrax.local", 0)]
