"""The forward (paper) ablation direction writes to its own round directory.

The paper's ablation table is the FORWARD sweep (ablate the overall-stronger model's
unique concepts on the rows it wins); the rebuttal's REVERSE sweep is the
above-diagonal check. Both come from ablation_sweep_symmetric.py, and until now the
forward run (opt-in --forward) defaulted into the reverse directory, so a round could
silently mix the two directions in one tree.
"""
from scripts import round_paths


def test_round_paths_name_the_forward_ablation_dirs():
    assert round_paths.FORWARD_ABLATION_DIR == round_paths.RESULTS_DIR / "forward_ablation"
    assert round_paths.FORWARD_ABLATION_RANDOM_DIR == round_paths.RESULTS_DIR / "forward_ablation_random"


def test_default_output_dir_follows_the_direction():
    from scripts.rebuttal.ablation_sweep_symmetric import default_output_dir

    assert default_output_dir(reverse=True) == round_paths.SYMMETRIC_ABLATION_DIR
    assert default_output_dir(reverse=False) == round_paths.FORWARD_ABLATION_DIR
