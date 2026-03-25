#!/usr/bin/env python3
"""One-off diagnostic: run HyperFast on credit-g to check prediction quality.

Tests whether passing cat_indices to HyperFast fixes discrete predictions.
Compares: (1) current pipeline (ordinal, no cat_indices)
          (2) ordinal + cat_indices passed to HyperFast
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import numpy as np
import pandas as pd

from scripts._project_root import PROJECT_ROOT
from data.preprocessing import load_preprocessed, CACHE_DIR
from data.extended_loader import load_tabarena_dataset
from scripts.intervention.intervene_lib import SPLITS_PATH

splits = json.loads(SPLITS_PATH.read_text())
info = splits["credit-g"]
train_idx = np.array(info["train_indices"])
test_idx = np.array(info["test_indices"])

# Load raw data to identify categorical columns
result = load_tabarena_dataset("credit-g", max_samples=10000)
X_raw = result.X
cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
print(f"Raw data: {X_raw.shape}, {len(cat_cols)} categorical columns")
print(f"Cat columns: {cat_cols}")
print()

# --- Method 1: Current pipeline (ordinal encoded, no cat_indices) ---
print("=" * 60)
print("Method 1: Current pipeline (ordinal, no cat_indices)")
print("=" * 60)
data = load_preprocessed("hyperfast", "credit-g", CACHE_DIR)
print(f"X_train: {data.X_train.shape}  range: [{data.X_train.min():.1f}, {data.X_train.max():.1f}]")

from hyperfast import HyperFastClassifier

clf1 = HyperFastClassifier(device="cuda")
clf1.fit(data.X_train, data.y_train)
preds1 = clf1.predict_proba(data.X_test)
p1 = preds1[:, 1]
print(f"Unique P(1): {len(np.unique(np.round(p1, 3)))}")
print(f"P(1) stats: min={p1.min():.4f} max={p1.max():.4f} std={p1.std():.4f}")
print()

# --- Method 2: Ordinal encoded + cat_indices passed to HyperFast ---
print("=" * 60)
print("Method 2: Ordinal + cat_indices (HyperFast does one-hot internally)")
print("=" * 60)

# Identify which columns in the ordinal-encoded array are categorical
# The preprocessing preserves column order, so cat column positions match
all_cols = list(X_raw.columns)
cat_indices = [all_cols.index(c) for c in cat_cols]
print(f"cat_indices: {cat_indices}")

clf2 = HyperFastClassifier(device="cuda", cat_features=cat_indices)
clf2.fit(data.X_train, data.y_train)
preds2 = clf2.predict_proba(data.X_test)
p2 = preds2[:, 1]
print(f"Unique P(1): {len(np.unique(np.round(p2, 3)))}")
print(f"P(1) stats: min={p2.min():.4f} max={p2.max():.4f} std={p2.std():.4f}")
print()

# --- Compare ---
print("=" * 60)
print("Comparison")
print("=" * 60)
print(f"Method 1 unique values: {len(np.unique(np.round(p1, 3)))}")
print(f"Method 2 unique values: {len(np.unique(np.round(p2, 3)))}")
print()
print("Method 1 sample:", preds1[:5, 1].round(4))
print("Method 2 sample:", preds2[:5, 1].round(4))
