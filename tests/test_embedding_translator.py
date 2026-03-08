"""Tests for the universal embedding translator."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.embedding_translator import (
    EmbeddingTranslator,
    evaluate_per_dataset,
    load_aligned_embeddings,
    load_translator,
    save_translator,
    split_train_val,
    train_translator,
    translate,
    translate_delta,
    translation_loss,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_pairs():
    """Synthetic aligned pairs: Y = X @ W_true.T + noise."""
    rng = np.random.RandomState(42)
    d_src, d_tgt = 16, 32
    W_true = rng.randn(d_tgt, d_src) * 0.5
    datasets = {}
    for i in range(6):
        n = 100
        X = rng.randn(n, d_src).astype(np.float32)
        Y = (X @ W_true.T + rng.randn(n, d_tgt) * 0.1).astype(np.float32)
        datasets[f"dataset_{i}"] = (X, Y)
    return datasets, W_true


# ── Model Tests ──────────────────────────────────────────────────────────────


class TestEmbeddingTranslator:
    def test_linear_shape(self):
        model = EmbeddingTranslator(16, 32, hidden=0)
        x = torch.randn(10, 16)
        assert model(x).shape == (10, 32)

    def test_mlp_shape(self):
        model = EmbeddingTranslator(16, 32, hidden=64)
        x = torch.randn(10, 16)
        assert model(x).shape == (10, 32)

    def test_linear_is_linear(self):
        model = EmbeddingTranslator(8, 4, hidden=0)
        x = torch.randn(5, 8)
        a, b = 2.0, 3.0
        # Linear: f(a*x1 + b*x2) = a*f(x1) + b*f(x2) ... approximately
        # Actually just check it's a single nn.Linear
        assert isinstance(model.net, torch.nn.Linear)

    def test_mlp_is_sequential(self):
        model = EmbeddingTranslator(8, 4, hidden=16)
        assert isinstance(model.net, torch.nn.Sequential)


# ── Loss Tests ───────────────────────────────────────────────────────────────


class TestTranslationLoss:
    def test_perfect_match_zero_loss(self):
        x = torch.randn(10, 32)
        loss = translation_loss(x, x, cosine_weight=0.5)
        assert loss.item() < 1e-6

    def test_mse_only(self):
        pred = torch.randn(10, 32)
        target = torch.randn(10, 32)
        loss = translation_loss(pred, target, cosine_weight=0.0)
        expected = torch.nn.functional.mse_loss(pred, target)
        assert abs(loss.item() - expected.item()) < 1e-6

    def test_cosine_only(self):
        pred = torch.randn(10, 32)
        target = torch.randn(10, 32)
        loss = translation_loss(pred, target, cosine_weight=1.0)
        cos = torch.nn.functional.cosine_similarity(pred, target, dim=1).mean()
        expected = 1.0 - cos
        assert abs(loss.item() - expected.item()) < 1e-5


# ── Split Tests ──────────────────────────────────────────────────────────────


class TestSplitTrainVal:
    def test_explicit_holdout(self, synthetic_pairs):
        dataset_pairs, _ = synthetic_pairs
        X_tr, Y_tr, X_v, Y_v, tr_ds, v_ds = split_train_val(
            dataset_pairs, holdout_datasets=["dataset_0", "dataset_1"]
        )
        assert "dataset_0" in v_ds
        assert "dataset_1" in v_ds
        assert len(tr_ds) == 4
        assert X_tr.shape[0] == 400
        assert X_v.shape[0] == 200

    def test_random_holdout(self, synthetic_pairs):
        dataset_pairs, _ = synthetic_pairs
        X_tr, Y_tr, X_v, Y_v, tr_ds, v_ds = split_train_val(
            dataset_pairs, holdout_frac=0.33
        )
        assert len(v_ds) >= 1
        assert len(tr_ds) + len(v_ds) == 6
        assert X_tr.shape[0] + X_v.shape[0] == 600

    def test_shapes_consistent(self, synthetic_pairs):
        dataset_pairs, _ = synthetic_pairs
        X_tr, Y_tr, X_v, Y_v, _, _ = split_train_val(dataset_pairs)
        assert X_tr.shape[1] == 16
        assert Y_tr.shape[1] == 32
        assert X_v.shape[1] == 16
        assert Y_v.shape[1] == 32


# ── Training Tests ───────────────────────────────────────────────────────────


class TestTrainTranslator:
    def test_linear_recovers_map(self, synthetic_pairs):
        """Linear translator should recover the true linear map."""
        dataset_pairs, W_true = synthetic_pairs
        X_tr, Y_tr, X_v, Y_v, _, _ = split_train_val(
            dataset_pairs, holdout_datasets=["dataset_5"]
        )
        model, history = train_translator(
            X_tr, Y_tr, X_v, Y_v,
            arch="linear", lr=1e-2, n_epochs=200, patience=50,
            cosine_weight=0.0,
        )
        assert history["best_val_r2"] > 0.9, f"R²={history['best_val_r2']}"

    def test_mlp_trains(self, synthetic_pairs):
        """MLP should train without errors and achieve decent R²."""
        dataset_pairs, _ = synthetic_pairs
        X_tr, Y_tr, X_v, Y_v, _, _ = split_train_val(
            dataset_pairs, holdout_datasets=["dataset_5"]
        )
        model, history = train_translator(
            X_tr, Y_tr, X_v, Y_v,
            arch="mlp", hidden=64, lr=3e-3, n_epochs=300, patience=100,
            cosine_weight=0.0, batch_size=128,
        )
        assert history["best_val_r2"] > 0.5, f"R²={history['best_val_r2']}"

    def test_early_stopping(self, synthetic_pairs):
        """With patience=10 and many epochs, should stop early."""
        dataset_pairs, _ = synthetic_pairs
        X_tr, Y_tr, X_v, Y_v, _, _ = split_train_val(
            dataset_pairs, holdout_datasets=["dataset_5"]
        )
        model, history = train_translator(
            X_tr, Y_tr, X_v, Y_v,
            arch="linear", lr=1e-2, n_epochs=1000, patience=10,
            cosine_weight=0.0,
        )
        assert len(history["train_loss"]) < 1000

    def test_history_keys(self, synthetic_pairs):
        dataset_pairs, _ = synthetic_pairs
        X_tr, Y_tr, X_v, Y_v, _, _ = split_train_val(
            dataset_pairs, holdout_datasets=["dataset_5"]
        )
        _, history = train_translator(
            X_tr, Y_tr, X_v, Y_v,
            arch="linear", n_epochs=10, patience=50,
        )
        for key in ["train_loss", "val_loss", "val_r2", "val_cosine",
                     "best_epoch", "best_val_loss", "best_val_r2", "best_val_cosine"]:
            assert key in history, f"Missing key: {key}"


# ── Translate Tests ──────────────────────────────────────────────────────────


class TestTranslate:
    def test_translate_shape(self):
        model = EmbeddingTranslator(16, 32, hidden=0)
        X = np.random.randn(20, 16).astype(np.float32)
        result = translate(model, X)
        assert result.shape == (20, 32)

    def test_translate_delta_linear(self):
        """For a linear model, translate_delta should equal delta @ W.T."""
        model = EmbeddingTranslator(8, 4, hidden=0)
        X = np.random.randn(10, 8).astype(np.float32)
        delta = np.random.randn(10, 8).astype(np.float32) * 0.1

        result = translate_delta(model, X, delta)
        # For linear: f(x+d) - f(x) = d @ W.T (bias cancels)
        W = model.net.weight.detach().numpy()  # (4, 8)
        expected = delta @ W.T
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_translate_delta_mlp_nonzero(self):
        """MLP translate_delta should produce non-zero results."""
        model = EmbeddingTranslator(8, 4, hidden=16)
        X = np.random.randn(10, 8).astype(np.float32)
        delta = np.random.randn(10, 8).astype(np.float32)
        result = translate_delta(model, X, delta)
        assert result.shape == (10, 4)
        assert np.abs(result).sum() > 0

    def test_translate_delta_zero_delta(self):
        """Zero delta should produce zero output."""
        model = EmbeddingTranslator(8, 4, hidden=16)
        X = np.random.randn(10, 8).astype(np.float32)
        delta = np.zeros((10, 8), dtype=np.float32)
        result = translate_delta(model, X, delta)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)


# ── Save/Load Tests ──────────────────────────────────────────────────────────


class TestSaveLoad:
    def test_roundtrip_linear(self):
        model = EmbeddingTranslator(16, 32, hidden=0)
        X = np.random.randn(5, 16).astype(np.float32)
        original_out = translate(model, X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_translator(model, path, "src", "tgt", {}, ["d1"], ["d2"])
            loaded, meta = load_translator(path)
            assert meta["source_model"] == "src"
            assert meta["target_model"] == "tgt"
            assert meta["arch"] == "linear"
            loaded_out = translate(loaded, X)
            np.testing.assert_allclose(original_out, loaded_out, atol=1e-6)

    def test_roundtrip_mlp(self):
        model = EmbeddingTranslator(16, 32, hidden=64)
        X = np.random.randn(5, 16).astype(np.float32)
        original_out = translate(model, X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_translator(model, path, "src", "tgt", {}, ["d1"], ["d2"])
            loaded, meta = load_translator(path)
            assert meta["arch"] == "mlp"
            assert meta["hidden"] == 64
            loaded_out = translate(loaded, X)
            np.testing.assert_allclose(original_out, loaded_out, atol=1e-6)


# ── Evaluate Tests ───────────────────────────────────────────────────────────


class TestEvaluate:
    def test_per_dataset_keys(self, synthetic_pairs):
        dataset_pairs, _ = synthetic_pairs
        model = EmbeddingTranslator(16, 32, hidden=0)
        results = evaluate_per_dataset(model, dataset_pairs, ["dataset_0"])
        assert "dataset_0" in results
        for key in ["r2", "cosine", "mse"]:
            assert key in results["dataset_0"]

    def test_missing_dataset_skipped(self, synthetic_pairs):
        dataset_pairs, _ = synthetic_pairs
        model = EmbeddingTranslator(16, 32, hidden=0)
        results = evaluate_per_dataset(model, dataset_pairs, ["nonexistent"])
        assert "nonexistent" not in results
