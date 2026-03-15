#!/usr/bin/env python3
"""Universal embedding translator between tabular foundation models.

Trains a map (linear or MLP) on ALL aligned embedding pairs across TabArena
datasets, rather than fitting per-dataset. This is the universal geometry
thesis: if models learn similar representations, one map should translate
between embedding spaces regardless of the dataset.

Usage:
    # Train universal TabPFN→TabICL map
    python scripts/embedding_translator.py \
        --source tabpfn --target tabicl --arch mlp --device cuda

    # Evaluate on held-out datasets
    python scripts/embedding_translator.py \
        --source tabpfn --target tabicl --eval-only \
        --checkpoint output/embedding_translator/tabpfn_to_tabicl.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scripts._project_root import PROJECT_ROOT

logger = logging.getLogger(__name__)

EMB_DIR = PROJECT_ROOT / "output" / "embeddings" / "tabarena"
OUT_DIR = PROJECT_ROOT / "output" / "embedding_translator"


# ── Model ────────────────────────────────────────────────────────────────────


class EmbeddingTranslator(nn.Module):
    """MLP map from source embedding space to target embedding space."""

    def __init__(self, d_source: int, d_target: int, hidden: int = 0, dropout: float = 0.1):
        super().__init__()
        if hidden <= 0:
            # Linear map (equivalent to Ridge but trained with SGD)
            self.net = nn.Linear(d_source, d_target)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_source, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, d_target),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_aligned_embeddings(
    source_model: str,
    target_model: str,
    holdout_datasets: Optional[List[str]] = None,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], List[str]]:
    """Load aligned embedding pairs from all overlapping TabArena datasets.

    Returns:
        dataset_pairs: {dataset_name: (emb_source, emb_target)} for all overlapping datasets
        overlap: sorted list of overlapping dataset names
    """
    src_dir = EMB_DIR / source_model
    tgt_dir = EMB_DIR / target_model
    if not src_dir.exists():
        raise FileNotFoundError(f"Source embedding dir not found: {src_dir}")
    if not tgt_dir.exists():
        raise FileNotFoundError(f"Target embedding dir not found: {tgt_dir}")

    src_datasets = {
        f.stem.replace("tabarena_", ""): f
        for f in src_dir.glob("tabarena_*.npz")
    }
    tgt_datasets = {
        f.stem.replace("tabarena_", ""): f
        for f in tgt_dir.glob("tabarena_*.npz")
    }
    overlap = sorted(set(src_datasets) & set(tgt_datasets))
    if not overlap:
        raise ValueError(f"No overlapping datasets between {source_model} and {target_model}")

    dataset_pairs = {}
    for ds in overlap:
        src = np.load(src_datasets[ds], allow_pickle=True)["embeddings"]
        tgt = np.load(tgt_datasets[ds], allow_pickle=True)["embeddings"]
        if src.shape[0] != tgt.shape[0]:
            logger.warning(f"{ds}: row mismatch ({src.shape[0]} vs {tgt.shape[0]}), skipping")
            continue
        dataset_pairs[ds] = (src, tgt)

    return dataset_pairs, overlap


def split_train_val(
    dataset_pairs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    holdout_datasets: Optional[List[str]] = None,
    holdout_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """Split aligned pairs into train/val by dataset (not by row).

    Holding out entire datasets tests generalization to unseen data distributions.
    If holdout_datasets is None, randomly selects ~holdout_frac of datasets.

    Returns:
        X_train, Y_train, X_val, Y_val, train_datasets, val_datasets
    """
    all_datasets = sorted(dataset_pairs.keys())
    rng = np.random.RandomState(seed)

    if holdout_datasets is None:
        n_val = max(1, int(len(all_datasets) * holdout_frac))
        val_datasets = sorted(rng.choice(all_datasets, n_val, replace=False).tolist())
    else:
        val_datasets = sorted(d for d in holdout_datasets if d in dataset_pairs)

    train_datasets = sorted(set(all_datasets) - set(val_datasets))

    X_train = np.concatenate([dataset_pairs[d][0] for d in train_datasets])
    Y_train = np.concatenate([dataset_pairs[d][1] for d in train_datasets])
    X_val = np.concatenate([dataset_pairs[d][0] for d in val_datasets])
    Y_val = np.concatenate([dataset_pairs[d][1] for d in val_datasets])

    return X_train, Y_train, X_val, Y_val, train_datasets, val_datasets


# ── Loss ─────────────────────────────────────────────────────────────────────


def translation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cosine_weight: float = 0.5,
) -> torch.Tensor:
    """MSE + cosine similarity loss for embedding translation.

    MSE ensures magnitude fidelity; cosine ensures directional alignment.
    """
    mse = nn.functional.mse_loss(pred, target)
    # Cosine similarity: 1 = perfect alignment, loss = 1 - cos
    cos = nn.functional.cosine_similarity(pred, target, dim=1).mean()
    return (1.0 - cosine_weight) * mse + cosine_weight * (1.0 - cos)


# ── Training ─────────────────────────────────────────────────────────────────


def train_translator(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    arch: str = "mlp",
    hidden: int = 0,
    dropout: float = 0.1,
    cosine_weight: float = 0.5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    n_epochs: int = 200,
    batch_size: int = 512,
    patience: int = 20,
    device: str = "cpu",
) -> Tuple[EmbeddingTranslator, Dict]:
    """Train the universal embedding translator.

    Args:
        X_train, Y_train: Training aligned pairs
        X_val, Y_val: Validation aligned pairs (held-out datasets)
        arch: "linear" or "mlp"
        hidden: Hidden dim for MLP (0 = auto from max(d_source, d_target))
        cosine_weight: Weight of cosine loss in [0, 1]
        n_epochs: Maximum training epochs
        patience: Early stopping patience
        device: torch device

    Returns:
        model: Trained EmbeddingTranslator
        history: Training metrics dict
    """
    d_source = X_train.shape[1]
    d_target = Y_train.shape[1]

    if arch == "linear":
        hidden_dim = 0
    else:
        hidden_dim = hidden if hidden > 0 else max(d_source, d_target)

    model = EmbeddingTranslator(d_source, d_target, hidden=hidden_dim, dropout=dropout)
    model = model.to(device)

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_r2": [], "val_cosine": []}

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = translation_loss(pred, yb, cosine_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # Validate
        model.eval()
        val_losses, val_preds, val_targets = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(translation_loss(pred, yb, cosine_weight).item())
                val_preds.append(pred.cpu())
                val_targets.append(yb.cpu())

        val_loss = np.mean(val_losses)
        val_pred_all = torch.cat(val_preds)
        val_tgt_all = torch.cat(val_targets)

        # R² on validation
        ss_res = ((val_pred_all - val_tgt_all) ** 2).sum().item()
        ss_tot = ((val_tgt_all - val_tgt_all.mean(dim=0)) ** 2).sum().item()
        val_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Mean cosine similarity on validation
        val_cos = nn.functional.cosine_similarity(val_pred_all, val_tgt_all, dim=1).mean().item()

        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(val_loss)
        history["val_r2"].append(val_r2)
        history["val_cosine"].append(val_cos)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            logger.info(
                f"Epoch {epoch:3d}: train_loss={np.mean(train_losses):.4f}  "
                f"val_loss={val_loss:.4f}  val_R²={val_r2:.4f}  val_cos={val_cos:.4f}"
            )

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    # Restore best
    model.load_state_dict(best_state)
    model.eval()

    history["best_epoch"] = int(np.argmin(history["val_loss"]))
    history["best_val_loss"] = float(best_val_loss)
    history["best_val_r2"] = float(history["val_r2"][history["best_epoch"]])
    history["best_val_cosine"] = float(history["val_cosine"][history["best_epoch"]])

    return model, history


# ── Per-Dataset Evaluation ───────────────────────────────────────────────────


def evaluate_per_dataset(
    model: EmbeddingTranslator,
    dataset_pairs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    datasets: List[str],
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """Evaluate translation quality per dataset.

    Returns:
        {dataset: {"r2": float, "cosine": float, "mse": float}}
    """
    model.eval()
    results = {}
    for ds in datasets:
        if ds not in dataset_pairs:
            continue
        X, Y = dataset_pairs[ds]
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32, device=device)
            yt = torch.tensor(Y, dtype=torch.float32, device=device)
            pred = model(xt)

            mse = nn.functional.mse_loss(pred, yt).item()
            cos = nn.functional.cosine_similarity(pred, yt, dim=1).mean().item()
            ss_res = ((pred - yt) ** 2).sum().item()
            ss_tot = ((yt - yt.mean(dim=0)) ** 2).sum().item()
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        results[ds] = {"r2": r2, "cosine": cos, "mse": mse}
    return results


# ── Convenience: translate embeddings ────────────────────────────────────────


def translate(
    model: EmbeddingTranslator,
    embeddings: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Translate source embeddings to target space.

    Args:
        model: Trained EmbeddingTranslator
        embeddings: (n_samples, d_source) source embeddings

    Returns:
        (n_samples, d_target) predicted target embeddings
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(embeddings, dtype=torch.float32, device=device)
        return model(x).cpu().numpy()


def translate_delta(
    model: EmbeddingTranslator,
    embeddings: np.ndarray,
    delta: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Translate a source-space delta to target space via finite difference.

    For an MLP, delta mapping is nonlinear:
        delta_target = translate(emb + delta) - translate(emb)

    This preserves nonlinear structure that a simple delta @ W.T misses.

    Args:
        model: Trained EmbeddingTranslator
        embeddings: (n_samples, d_source) base source embeddings
        delta: (n_samples, d_source) delta in source space

    Returns:
        (n_samples, d_target) delta in target space
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(embeddings, dtype=torch.float32, device=device)
        d = torch.tensor(delta, dtype=torch.float32, device=device)
        target_base = model(x)
        target_perturbed = model(x + d)
        return (target_perturbed - target_base).cpu().numpy()


# ── Save / Load ──────────────────────────────────────────────────────────────


def save_translator(
    model: EmbeddingTranslator,
    path: Path,
    source_model: str,
    target_model: str,
    history: Dict,
    train_datasets: List[str],
    val_datasets: List[str],
):
    """Save trained translator with metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "source_model": source_model,
            "target_model": target_model,
            "d_source": model.net[0].in_features if isinstance(model.net, nn.Sequential) else model.net.in_features,
            "d_target": model.net[-1].out_features if isinstance(model.net, nn.Sequential) else model.net.out_features,
            "hidden": model.net[0].out_features if isinstance(model.net, nn.Sequential) else 0,
            "arch": "mlp" if isinstance(model.net, nn.Sequential) else "linear",
            "history": history,
            "train_datasets": train_datasets,
            "val_datasets": val_datasets,
        },
        path,
    )
    logger.info(f"Saved translator to {path}")


def load_translator(path: Path, device: str = "cpu") -> Tuple[EmbeddingTranslator, Dict]:
    """Load a trained translator from checkpoint.

    Returns:
        model: EmbeddingTranslator in eval mode
        metadata: Dict with training info
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = EmbeddingTranslator(
        d_source=ckpt["d_source"],
        d_target=ckpt["d_target"],
        hidden=ckpt["hidden"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model = model.to(device)
    return model, ckpt


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train universal embedding translator")
    parser.add_argument("--source", required=True, help="Source model (e.g. tabpfn)")
    parser.add_argument("--target", required=True, help="Target model (e.g. tabicl)")
    parser.add_argument("--arch", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--hidden", type=int, default=0, help="MLP hidden dim (0=auto)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cosine-weight", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--holdout", nargs="*", help="Datasets to hold out for validation")
    parser.add_argument("--holdout-frac", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load all aligned embeddings
    dataset_pairs, overlap = load_aligned_embeddings(args.source, args.target)
    logger.info(f"Loaded {len(dataset_pairs)} overlapping datasets, "
                f"{sum(v[0].shape[0] for v in dataset_pairs.values())} aligned pairs")
    logger.info(f"Source dim: {next(iter(dataset_pairs.values()))[0].shape[1]}, "
                f"Target dim: {next(iter(dataset_pairs.values()))[1].shape[1]}")

    if args.eval_only:
        if not args.checkpoint:
            parser.error("--checkpoint required with --eval-only")
        model, meta = load_translator(args.checkpoint, args.device)
        logger.info(f"Loaded {meta['arch']} translator from {args.checkpoint}")
        logger.info(f"Train datasets: {meta['train_datasets']}")
        logger.info(f"Val datasets: {meta['val_datasets']}")

        results = evaluate_per_dataset(model, dataset_pairs, overlap, args.device)
        logger.info(f"\n{'Dataset':<45} {'R²':>8} {'Cosine':>8} {'MSE':>8}  Split")
        logger.info("-" * 80)
        for ds in sorted(results):
            r = results[ds]
            split = "VAL" if ds in meta.get("val_datasets", []) else "train"
            logger.info(f"{ds:<45} {r['r2']:8.4f} {r['cosine']:8.4f} {r['mse']:8.4f}  {split}")

        # Summary
        val_ds = [ds for ds in results if ds in meta.get("val_datasets", [])]
        train_ds = [ds for ds in results if ds not in meta.get("val_datasets", [])]
        if val_ds:
            val_r2 = np.mean([results[d]["r2"] for d in val_ds])
            val_cos = np.mean([results[d]["cosine"] for d in val_ds])
            logger.info(f"\nVal mean:   R²={val_r2:.4f}  cos={val_cos:.4f} ({len(val_ds)} datasets)")
        if train_ds:
            train_r2 = np.mean([results[d]["r2"] for d in train_ds])
            train_cos = np.mean([results[d]["cosine"] for d in train_ds])
            logger.info(f"Train mean: R²={train_r2:.4f}  cos={train_cos:.4f} ({len(train_ds)} datasets)")
        return

    # Train/val split
    X_train, Y_train, X_val, Y_val, train_datasets, val_datasets = split_train_val(
        dataset_pairs, holdout_datasets=args.holdout, holdout_frac=args.holdout_frac, seed=args.seed,
    )
    logger.info(f"Train: {X_train.shape[0]} pairs from {len(train_datasets)} datasets")
    logger.info(f"Val:   {X_val.shape[0]} pairs from {len(val_datasets)} datasets: {val_datasets}")

    # Train
    model, history = train_translator(
        X_train, Y_train, X_val, Y_val,
        arch=args.arch,
        hidden=args.hidden,
        dropout=args.dropout,
        cosine_weight=args.cosine_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        device=args.device,
    )

    logger.info(f"\nBest epoch {history['best_epoch']}: "
                f"val_loss={history['best_val_loss']:.4f}  "
                f"val_R²={history['best_val_r2']:.4f}  "
                f"val_cos={history['best_val_cosine']:.4f}")

    # Per-dataset breakdown
    results = evaluate_per_dataset(model, dataset_pairs, overlap, args.device)
    logger.info(f"\n{'Dataset':<45} {'R²':>8} {'Cosine':>8} {'MSE':>8}  Split")
    logger.info("-" * 80)
    for ds in sorted(results):
        r = results[ds]
        split = "VAL" if ds in val_datasets else "train"
        logger.info(f"{ds:<45} {r['r2']:8.4f} {r['cosine']:8.4f} {r['mse']:8.4f}  {split}")

    # Save
    ckpt_name = f"{args.source}_to_{args.target}.pt"
    ckpt_path = OUT_DIR / ckpt_name
    save_translator(model, ckpt_path, args.source, args.target, history, train_datasets, val_datasets)

    # Save per-dataset results
    results_path = OUT_DIR / f"{args.source}_to_{args.target}_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "arch": args.arch,
                "hidden": args.hidden or max(X_train.shape[1], Y_train.shape[1]),
                "best_val_r2": history["best_val_r2"],
                "best_val_cosine": history["best_val_cosine"],
                "n_train_pairs": X_train.shape[0],
                "n_val_pairs": X_val.shape[0],
                "train_datasets": train_datasets,
                "val_datasets": val_datasets,
                "per_dataset": results,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
