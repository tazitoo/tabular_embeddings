"""TabDPT must take the same deterministic path at corpus extraction and in the tail.

TabDPT has two sources of randomness that seeding its predict() does not fix:
  * over its 100-feature cap it fits a randomized PCA (torch.pca_lowrank) at fit time;
  * regression predict() is an 8-member ensemble whose member seeds come from OS
    entropy and whose members each permute the features.
Passing `seed` to predict() turns ON a feature permutation the SAE corpus never had,
so it changes the path instead of pinning it. The reproducible path is: pin the torch
RNG before fit (PCA), and predict with a single unpermuted member.
"""
import sys
import types

import numpy as np
import pytest
import torch


class _Recorder:
    """Stand-in for a fitted TabDPT estimator that records predict kwargs."""

    def __init__(self):
        self.calls = []

    def predict(self, X, **kw):
        self.calls.append(("predict", kw))
        return np.zeros(len(X), dtype=np.float32)

    def predict_proba(self, X, **kw):
        self.calls.append(("predict_proba", kw))
        return np.full((len(X), 2), 0.5, dtype=np.float32)


class TabDPTRegressor(_Recorder):
    pass


class TabDPTClassifier(_Recorder):
    pass


X = np.zeros((4, 3), dtype=np.float32)


def test_extraction_predict_uses_single_unpermuted_member_for_regression():
    from models.layer_extraction import predict

    clf = TabDPTRegressor()
    predict(clf, X, task="regression")
    name, kw = clf.calls[0]
    assert name == "predict"
    assert kw.get("n_ensembles") == 1
    assert "seed" not in kw


def test_extraction_predict_passes_no_seed_for_classification():
    from models.layer_extraction import predict

    clf = TabDPTClassifier()
    predict(clf, X, task="classification")
    name, kw = clf.calls[0]
    assert name == "predict_proba"
    assert "seed" not in kw


def test_extract_all_layers_has_no_seed_parameter():
    from models.layer_extraction import extract_all_layers

    with pytest.raises(TypeError):
        extract_all_layers("tabdpt", TabDPTClassifier(), X, task="classification", seed=13)


def test_load_and_fit_pins_torch_rng_before_fit(monkeypatch):
    """Two fits under the same seed must see the same RNG stream (this is what pins
    TabDPT's randomized PCA), regardless of what the process drew in between."""
    draws = []

    class StubClassifier:
        def __init__(self, device="cpu", compile=False):
            pass

        def fit(self, X, y):
            draws.append(torch.rand(3).clone())

    stub = types.ModuleType("tabdpt")
    stub.TabDPTClassifier = StubClassifier
    stub.TabDPTRegressor = StubClassifier
    monkeypatch.setitem(sys.modules, "tabdpt", stub)

    from models.layer_extraction import load_and_fit

    y = np.zeros(4, dtype=np.int32)
    load_and_fit("tabdpt", X, y, task="classification", device="cpu", seed=13)
    torch.manual_seed(999)
    torch.rand(7)
    load_and_fit("tabdpt", X, y, task="classification", device="cpu", seed=13)
    assert torch.equal(draws[0], draws[1])


def _tail(clf, task):
    from scripts.intervention.intervene_sae import TabDPTTail

    layer = torch.nn.Identity()
    return TabDPTTail(clf=clf, encoder_layers=[layer], hidden_state=torch.zeros(4, 3),
                      extraction_layer=0, n_ctx=2, n_query=2, X_query=X[:2],
                      task=task, device="cpu")


def test_tail_regression_predict_uses_single_member():
    clf = TabDPTRegressor()
    tail = _tail(clf, "regression")
    tail.predict(torch.zeros(4, 3))  # delta spans the full context+query state
    name, kw = clf.calls[-1]
    assert name == "predict"
    assert kw.get("n_ensembles") == 1


def test_tail_regression_recapture_uses_single_member():
    class ForwardingRegressor(TabDPTRegressor):
        """predict() runs the hooked layer so recapture has a hidden state to keep."""

        def predict(self, X, **kw):
            self.layer(torch.zeros(4, 3))
            return super().predict(X, **kw)

    clf = ForwardingRegressor()
    tail = _tail(clf, "regression")
    clf.layer = tail.encoder_layers[0]
    tail.recapture(X[:2])
    name, kw = clf.calls[-1]
    assert name == "predict"
    assert kw.get("n_ensembles") == 1


def test_build_tail_pins_torch_rng_before_every_build(monkeypatch):
    """A resumed sweep must build the same tail as a fresh one: the RNG is pinned
    per build, not once per process (this is what pins TabDPT's fit-time PCA)."""
    from scripts.intervention import intervene_sae

    draws = []

    def fake_from_data(*args, **kwargs):
        draws.append(torch.rand(3).clone())
        return "tail"

    monkeypatch.setattr(intervene_sae.TabDPTTail, "from_data", staticmethod(fake_from_data))
    y = np.zeros(4, dtype=np.int32)
    intervene_sae.build_tail("tabdpt", X, y, X[:2], extraction_layer=1, device="cpu")
    torch.manual_seed(999)
    torch.rand(7)
    intervene_sae.build_tail("tabdpt", X, y, X[:2], extraction_layer=1, device="cpu")
    assert torch.equal(draws[0], draws[1])
