"""Every extracted embedding file records where and how it was produced.

Bit-exactness holds per host and commit (docs/reproducibility.md), so an npz whose
producing host is unknown cannot be checked against a re-extraction. Round 11 records
host, commit, fit seed and torch version in each file.
"""
import importlib.util

from scripts._project_root import PROJECT_ROOT


def _load_04():
    path = PROJECT_ROOT / "scripts" / "sae_corpus" / "04_extract_all_layers.py"
    spec = importlib.util.spec_from_file_location("extract_all_layers_04", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_provenance_records_host_commit_seed_and_torch():
    from models.layer_extraction import FIT_SEED

    prov = _load_04().provenance()
    assert set(prov) >= {"host", "commit", "fit_seed", "torch"}
    assert prov["fit_seed"] == FIT_SEED
    assert len(prov["commit"]) >= 7
    assert prov["host"]


def test_provenance_values_are_npz_safe_scalars():
    import numpy as np

    prov = _load_04().provenance()
    for k, v in prov.items():
        np.array(v)  # must be storable as a 0-d array in savez
        assert isinstance(v, (str, int)), k
