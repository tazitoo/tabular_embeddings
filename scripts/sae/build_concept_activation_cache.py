#!/usr/bin/env python3
"""
Build a per-model, per-dataset cache of SAE concept activations.

Uses the round 10 SAE test set (already per-dataset normalized) as input
and encodes it with each model's validated SAE checkpoint. Caches the
resulting activations per dataset for interactive concept analysis.

Cache structure:
    output/concept_activations_cache/
        {model}/
            {dataset}.npz
                activations: (n_rows, n_hidden) float32
                row_indices: (n_rows,) int32  -- original dataset row indices
                alive_mask: (n_hidden,) bool  -- features that fire on any row
                n_rows: int
                n_hidden: int
                model: str
                dataset: str
        _global_alive.npz
            alive_mask: (n_hidden,) bool  -- union across all datasets

Usage:
    python -m scripts.sae.build_concept_activation_cache --model tabpfn
    python -m scripts.sae.build_concept_activation_cache --all
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.sae.analyze_sae_concepts_deep import load_sae_checkpoint

MODELS = ["tabpfn", "tabicl", "tabicl_v2", "tabdpt", "mitra", "carte"]

# SAE checkpoint path per model (round 10 validated)
SAE_CKPT = {
    m: f"sae_tabarena_sweep_round10/{m}/sae_matryoshka_archetypal_validated.pt"
    for m in MODELS
}

# Pre-processed test embeddings (already per-dataset normalized for SAE)
TEST_DATA = {
    m: f"sae_training_round10/{m}_taskaware_sae_test.npz"
    for m in MODELS
}

OUTPUT_DIR = PROJECT_ROOT / "output" / "concept_activations_cache"


def build_model_cache(model_name: str) -> None:
    """Build concept activation cache for one model across all datasets."""
    ckpt_path = PROJECT_ROOT / "output" / SAE_CKPT[model_name]
    test_path = PROJECT_ROOT / "output" / TEST_DATA[model_name]

    if not ckpt_path.exists():
        print(f"  SAE checkpoint not found: {ckpt_path}")
        return
    if not test_path.exists():
        print(f"  Test data not found: {test_path}")
        return

    print(f"\n=== {model_name} ===")
    print(f"Loading SAE: {ckpt_path.name}")
    sae, config, _ = load_sae_checkpoint(ckpt_path)
    sae.eval()

    print(f"Loading test data: {test_path.name}")
    d = np.load(test_path, allow_pickle=True)
    embeddings = d["embeddings"].astype(np.float32)
    row_indices_all = d["row_indices"]
    samples_per_dataset = d["samples_per_dataset"]

    print(f"  {len(embeddings)} rows, {embeddings.shape[1]} dim, {len(samples_per_dataset)} datasets")

    # Encode everything at once
    print("Encoding...")
    with torch.no_grad():
        x = torch.tensor(embeddings, dtype=torch.float32)
        codes = sae.encode(x).numpy().astype(np.float32)
    print(f"  Codes: {codes.shape}")

    n_hidden = codes.shape[1]
    out_dir = OUTPUT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    global_alive = np.zeros(n_hidden, dtype=bool)
    row_offset = 0

    for item in samples_per_dataset:
        ds_name = str(item["dataset"])
        count = int(item["count"])
        ds_codes = codes[row_offset : row_offset + count]
        ds_row_idx = row_indices_all[row_offset : row_offset + count]
        row_offset += count

        alive_mask = ds_codes.max(axis=0) > 0.001
        global_alive |= alive_mask

        np.savez_compressed(
            out_dir / f"{ds_name}.npz",
            activations=ds_codes,
            row_indices=ds_row_idx.astype(np.int32),
            alive_mask=alive_mask,
            n_rows=count,
            n_hidden=n_hidden,
            model=model_name,
            dataset=ds_name,
        )
        print(
            f"  {ds_name}: {count} rows, {alive_mask.sum()} alive features"
        )

    np.savez_compressed(
        out_dir / "_global_alive.npz",
        alive_mask=global_alive,
        n_datasets=len(samples_per_dataset),
        total_alive=int(global_alive.sum()),
        n_hidden=n_hidden,
    )
    print(f"  Global alive: {global_alive.sum()}/{n_hidden}")


def main():
    parser = argparse.ArgumentParser(
        description="Build SAE concept activation cache"
    )
    parser.add_argument("--model", choices=MODELS, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = MODELS if args.all else ([args.model] if args.model else None)
    if not models:
        parser.error("Specify --model or --all")

    for m in models:
        t0 = time.time()
        build_model_cache(m)
        print(f"{m} done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
