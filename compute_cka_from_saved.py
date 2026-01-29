#!/usr/bin/env python3
"""
Compute pairwise CKA, cosine similarity, and Procrustes distance from saved
.npz embedding files.

Reads embeddings saved by extract_embeddings.py and produces a CSV with
per-dataset pairwise metrics, plus an aggregated summary.

Usage:
    python compute_cka_from_saved.py \
        --embedding-dir 3_output/embeddings/tabarena \
        --output 3_output/geometric_sweep_full.csv

    # With LaTeX table output
    python compute_cka_from_saved.py \
        --embedding-dir 3_output/embeddings/tabarena \
        --output 3_output/geometric_sweep_full.csv \
        --latex 3_output/geometric_sweep_full.tex
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from analysis.similarity import (
    centered_kernel_alignment,
    cosine_similarity_paired,
    procrustes_align,
)


def discover_models(embedding_dir: Path) -> Dict[str, List[Path]]:
    """Scan embedding_dir for model subdirectories and their .npz files."""
    models = {}
    for subdir in sorted(embedding_dir.iterdir()):
        if not subdir.is_dir():
            continue
        npz_files = sorted(subdir.glob("*.npz"))
        if npz_files:
            models[subdir.name] = npz_files
    return models


def load_model_embeddings(npz_files: List[Path]) -> Dict[str, np.ndarray]:
    """Load primary embeddings from all .npz files for one model."""
    embeddings = {}
    for f in npz_files:
        dataset_name = f.stem
        data = np.load(str(f), allow_pickle=True)
        embeddings[dataset_name] = data["embeddings"]
    return embeddings


def compute_pairwise_metrics(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
) -> Dict[str, float]:
    """Compute CKA, mean cosine similarity, and Procrustes distance."""
    cka = centered_kernel_alignment(emb_a, emb_b)
    cos_sims = cosine_similarity_paired(emb_a, emb_b)
    proc_dist, _, _ = procrustes_align(emb_a, emb_b)
    return {
        "cka_score": float(cka),
        "mean_cosine_sim": float(cos_sims.mean()),
        "procrustes_distance": float(proc_dist),
    }


def generate_latex_table(summary_df: pd.DataFrame, output_path: Path):
    """Generate a LaTeX table from the aggregated summary."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Pairwise embedding similarity across TabArena datasets}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Model A & Model B & CKA & Cosine Sim & Procrustes \\",
        r"\midrule",
    ]

    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['model_a']} & {row['model_b']} & "
            f"{row['cka_mean']:.3f} $\\pm$ {row['cka_std']:.3f} & "
            f"{row['cosine_mean']:.3f} $\\pm$ {row['cosine_std']:.3f} & "
            f"{row['procrustes_mean']:.3f} $\\pm$ {row['procrustes_std']:.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_path.write_text("\n".join(lines))
    print(f"LaTeX table saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute pairwise CKA from saved embedding .npz files"
    )
    parser.add_argument("--embedding-dir", type=str, required=True,
                        help="Directory containing model subdirectories with .npz files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path")
    parser.add_argument("--latex", type=str, default=None,
                        help="Output LaTeX table path")
    args = parser.parse_args()

    embedding_dir = Path(args.embedding_dir)
    if not embedding_dir.exists():
        print(f"Error: {embedding_dir} does not exist")
        sys.exit(1)

    # Discover models
    model_files = discover_models(embedding_dir)
    model_names = list(model_files.keys())
    print(f"Found {len(model_names)} models: {', '.join(model_names)}")

    if len(model_names) < 2:
        print("Need at least 2 models for pairwise comparison")
        sys.exit(1)

    # Load all embeddings
    all_embeddings = {}
    for name, files in model_files.items():
        all_embeddings[name] = load_model_embeddings(files)
        print(f"  {name}: {len(all_embeddings[name])} datasets")

    # Compute pairwise metrics
    rows = []
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i + 1:]:
            datasets_a = set(all_embeddings[model_a].keys())
            datasets_b = set(all_embeddings[model_b].keys())
            common = sorted(datasets_a & datasets_b)

            if not common:
                print(f"  {model_a} vs {model_b}: no common datasets")
                continue

            print(f"\n{model_a} vs {model_b}: {len(common)} common datasets")

            for dataset_name in common:
                emb_a = all_embeddings[model_a][dataset_name]
                emb_b = all_embeddings[model_b][dataset_name]

                # Must have same number of query samples
                if emb_a.shape[0] != emb_b.shape[0]:
                    print(f"  {dataset_name}: sample count mismatch "
                          f"({emb_a.shape[0]} vs {emb_b.shape[0]}), skipping")
                    continue

                try:
                    metrics = compute_pairwise_metrics(emb_a, emb_b)
                    rows.append({
                        "dataset": dataset_name,
                        "model_a": model_a,
                        "model_b": model_b,
                        "dim_a": emb_a.shape[1] if emb_a.ndim > 1 else 1,
                        "dim_b": emb_b.shape[1] if emb_b.ndim > 1 else 1,
                        **metrics,
                    })
                except Exception as e:
                    print(f"  {dataset_name}: FAILED ({e})")

    if not rows:
        print("No pairwise results computed")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Print aggregated summary
    print(f"\n{'=' * 70}")
    print("AGGREGATED RESULTS")
    print(f"{'=' * 70}")

    summary_rows = []
    for (ma, mb), group in df.groupby(["model_a", "model_b"]):
        summary = {
            "model_a": ma,
            "model_b": mb,
            "n_datasets": len(group),
            "cka_mean": group["cka_score"].mean(),
            "cka_std": group["cka_score"].std(),
            "cosine_mean": group["mean_cosine_sim"].mean(),
            "cosine_std": group["mean_cosine_sim"].std(),
            "procrustes_mean": group["procrustes_distance"].mean(),
            "procrustes_std": group["procrustes_distance"].std(),
        }
        summary_rows.append(summary)
        print(f"\n{ma} vs {mb} (n={summary['n_datasets']}):")
        print(f"  CKA:        {summary['cka_mean']:.4f} ± {summary['cka_std']:.4f}")
        print(f"  Cosine:     {summary['cosine_mean']:.4f} ± {summary['cosine_std']:.4f}")
        print(f"  Procrustes: {summary['procrustes_mean']:.4f} ± {summary['procrustes_std']:.4f}")

    summary_df = pd.DataFrame(summary_rows)

    # Save outputs
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nPer-dataset results saved to {output_path}")

        # Also save summary
        summary_path = output_path.with_name(output_path.stem + "_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")

    if args.latex:
        generate_latex_table(summary_df, Path(args.latex))


if __name__ == "__main__":
    main()
