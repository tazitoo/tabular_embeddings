#!/usr/bin/env python3
"""
Verify embedding sample counts across all models and datasets.

Compares actual embedding shapes against expected query sizes from
load_context_query() to detect any extraction bugs (e.g. hook overwrite).

Usage:
    PYTHONPATH=. python scripts/verify_embedding_counts.py
    PYTHONPATH=. python scripts/verify_embedding_counts.py --model mitra
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.extended_loader import TABARENA_DATASETS
from scripts.extract_layer_embeddings import load_context_query


def get_expected_query_size(dataset_name: str) -> int:
    """Compute expected query size for a dataset."""
    _, _, X_query = load_context_query(dataset_name)
    return len(X_query)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None,
                        help='Check specific model (default: all)')
    args = parser.parse_args()

    emb_dir = PROJECT_ROOT / 'output' / 'embeddings' / 'tabarena'
    if not emb_dir.exists():
        print(f"Error: {emb_dir} not found")
        return

    models = [args.model] if args.model else sorted(
        d.name for d in emb_dir.iterdir() if d.is_dir()
    )

    # Cache expected sizes
    print("Computing expected query sizes...")
    expected = {}
    for ds_name in sorted(TABARENA_DATASETS.keys()):
        try:
            expected[ds_name] = get_expected_query_size(ds_name)
        except Exception as e:
            print(f"  Warning: failed to load {ds_name}: {e}")

    print(f"Expected sizes computed for {len(expected)} datasets\n")

    all_ok = True
    for model_name in models:
        model_dir = emb_dir / model_name
        if not model_dir.exists():
            print(f"{model_name}: directory not found, skipping")
            continue

        files = sorted(model_dir.glob('tabarena_*.npz'))
        n_ok = 0
        n_bad = 0
        bad_details = []

        for f in files:
            ds_name = f.stem.replace('tabarena_', '')
            if ds_name not in expected:
                continue

            data = np.load(f, allow_pickle=True)
            actual = data['embeddings'].shape[0]
            exp = expected[ds_name]

            if actual == exp:
                n_ok += 1
            else:
                n_bad += 1
                bad_details.append((ds_name, actual, exp))
                all_ok = False

        status = "OK" if n_bad == 0 else "MISMATCH"
        print(f"{model_name}: {n_ok} OK, {n_bad} mismatched out of {len(files)} files [{status}]")
        for ds, actual, exp in bad_details:
            print(f"  {ds}: got {actual}, expected {exp} (delta {actual - exp})")

    if all_ok:
        print("\nAll models match expected counts.")
    else:
        print("\nSome models have mismatched counts — re-extraction needed.")


if __name__ == '__main__':
    main()
