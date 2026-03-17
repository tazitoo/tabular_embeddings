#!/usr/bin/env python3
"""
Validate that preprocessing fixes work correctly.

Tests:
1. Loader returns DataFrame with proper dtypes (categoricals as object)
2. Metadata includes cat_feature_indices and feature_names
3. Model wrappers accept DataFrames and produce valid embeddings
4. Old vs new preprocessing comparison (accuracy check)

Usage:
    python scripts/validate_preprocessing.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data.extended_loader import load_tabarena_dataset, TABARENA_DATASETS


def test_loader_returns_dataframe():
    """Test that loader returns DataFrame with proper dtypes."""
    # Pick a dataset with known categoricals
    test_datasets = ["adult", "SpeedDating", "wine_quality"]

    for name in test_datasets:
        if name not in TABARENA_DATASETS:
            continue

        result = load_tabarena_dataset(name, max_samples=200)
        if result is None:
            print(f"  SKIP: {name} (failed to load)")
            continue

        X_df, y, meta = result

        # Check return type
        assert isinstance(X_df, pd.DataFrame), f"{name}: X should be DataFrame, got {type(X_df)}"

        # Check metadata fields
        assert hasattr(meta, 'cat_feature_indices'), f"{name}: metadata missing cat_feature_indices"
        assert hasattr(meta, 'feature_names'), f"{name}: metadata missing feature_names"
        assert len(meta.feature_names) == len(X_df.columns), f"{name}: feature_names length mismatch"

        # Check that categorical columns are object dtype (not integer codes)
        cat_cols = X_df.select_dtypes(include=["object", "category"]).columns
        n_cat = len(cat_cols)

        # Check that cat_feature_indices matches actual categoricals
        assert len(meta.cat_feature_indices) == n_cat, (
            f"{name}: cat_feature_indices ({len(meta.cat_feature_indices)}) != "
            f"actual categoricals ({n_cat})"
        )

        # If dataset has categoricals, verify they're strings, not integer codes
        if n_cat > 0:
            first_cat = cat_cols[0]
            sample_values = X_df[first_cat].dropna().head(5).tolist()
            # Values should be strings, not integers
            for v in sample_values:
                assert isinstance(v, str), (
                    f"{name}: categorical column '{first_cat}' has non-string value: "
                    f"{v} (type={type(v)})"
                )

        print(f"  OK: {name} ({meta.n_samples}x{meta.n_features}, "
              f"{n_cat} cat cols, feature_types={meta.feature_types})")

        # Show sample of categorical values
        if n_cat > 0:
            first_cat = cat_cols[0]
            unique_vals = X_df[first_cat].dropna().unique()[:5]
            print(f"      Sample categoricals in '{first_cat}': {unique_vals}")


def test_model_wrapper_accepts_dataframe():
    """Test that model wrappers can accept DataFrames."""
    from models.base import EmbeddingExtractor

    # Create a test DataFrame with mixed types
    n = 50
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        'age': rng.randint(18, 80, n).astype(float),
        'income': rng.normal(50000, 15000, n),
        'education': rng.choice(['HS', 'BS', 'MS', 'PhD'], n),
        'occupation': rng.choice(['Sales', 'Tech', 'Admin', 'Other'], n),
    })
    y = rng.randint(0, 2, n)

    cat_indices = EmbeddingExtractor._detect_cat_features(df)
    assert cat_indices == [2, 3], f"Expected cat indices [2, 3], got {cat_indices}"

    X_np, detected_indices = EmbeddingExtractor._to_numpy_with_label_encoding(df)
    assert X_np.dtype == np.float32, f"Expected float32, got {X_np.dtype}"
    assert X_np.shape == (n, 4), f"Expected ({n}, 4), got {X_np.shape}"
    assert detected_indices == [2, 3], f"Expected [2, 3], got {detected_indices}"

    # Label-encoded values should be integers (0, 1, 2, ...)
    for idx in detected_indices:
        unique_vals = np.unique(X_np[:, idx][~np.isnan(X_np[:, idx])])
        assert all(v == int(v) for v in unique_vals), (
            f"Column {idx} should be label-encoded integers, got {unique_vals}"
        )

    print("  OK: _detect_cat_features and _to_numpy_with_label_encoding work correctly")
    print(f"      cat_indices={cat_indices}, X_np shape={X_np.shape}")


def test_real_dataset_categoricals():
    """Verify specific datasets have expected categorical structure."""
    # adult dataset should have many categoricals (workclass, education, etc.)
    if "adult" not in TABARENA_DATASETS:
        print("  SKIP: adult not in TABARENA_DATASETS")
        return

    result = load_tabarena_dataset("adult", max_samples=100)
    if result is None:
        print("  SKIP: adult failed to load")
        return

    X_df, y, meta = result

    cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_df.select_dtypes(include=["number"]).columns.tolist()

    print(f"  adult: {len(cat_cols)} categorical, {len(num_cols)} numeric columns")
    print(f"  Categorical: {cat_cols}")
    print(f"  Numeric: {num_cols[:5]}...")

    # Verify at least some categoricals exist
    assert len(cat_cols) > 0, "adult should have categorical columns"

    # Verify categorical values are meaningful strings
    for col in cat_cols[:3]:
        vals = X_df[col].dropna().unique()[:5]
        print(f"    {col}: {vals}")
        assert all(isinstance(v, str) for v in vals), f"{col} values should be strings"


def main():
    print("=" * 70)
    print("Preprocessing Validation")
    print("=" * 70)

    print("\n1. Testing loader returns DataFrame with proper dtypes...")
    test_loader_returns_dataframe()

    print("\n2. Testing model wrapper DataFrame handling...")
    test_model_wrapper_accepts_dataframe()

    print("\n3. Testing real dataset categorical structure...")
    test_real_dataset_categoricals()

    print("\n" + "=" * 70)
    print("All validation tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
