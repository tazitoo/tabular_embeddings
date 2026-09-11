"""Every stage of the intervention chain enables deterministic kernels and records
provenance in what it writes.

The NeurIPS-era sweeps seeded once per process and set no determinism flags, which is
half of why their outputs do not reproduce (docs/reproducibility.md). This scans each
chain script's source: its main() must call configure_determinism(), and every
np.savez_compressed must spread provenance() into the saved dict.
"""
import ast

from scripts._project_root import PROJECT_ROOT

CHAIN_SCRIPTS = [
    "scripts/intervention/perrow_importance.py",
    "scripts/rebuttal/ablation_sweep_symmetric.py",
    "scripts/rebuttal/transfer_sweep_symmetric.py",
]


def _calls(tree, name):
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == name]


def _main_calls(tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return bool(_calls(node, name))
    return False


def _every_savez_spreads_provenance(src):
    """Each savez_compressed(...) must be preceded (in the same block) by a
    provenance() spread into the dict it saves."""
    tree = ast.parse(src)
    saves = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "savez_compressed"]
    assert saves, "no savez_compressed found"
    for s in saves:
        window = src.splitlines()[max(0, s.lineno - 6):s.lineno]
        assert any("provenance()" in line for line in window), \
            f"savez_compressed at line {s.lineno} without provenance()"


def test_every_chain_script_enables_determinism_in_main():
    for rel in CHAIN_SCRIPTS:
        tree = ast.parse((PROJECT_ROOT / rel).read_text())
        assert _main_calls(tree, "configure_determinism"), rel


def test_every_chain_script_records_provenance_in_its_npz():
    for rel in CHAIN_SCRIPTS:
        _every_savez_spreads_provenance((PROJECT_ROOT / rel).read_text())
