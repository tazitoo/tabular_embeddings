#!/usr/bin/env python3
"""One-off diagnostic: run HyperFast on credit-g to check prediction quality.

Tests whether HyperFast produces continuous probabilities or discrete outputs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import numpy as np

from scripts._project_root import PROJECT_ROOT
from data.preprocessing import load_preprocessed, CACHE_DIR
from scripts.intervention.intervene_lib import SPLITS_PATH

splits = json.loads(SPLITS_PATH.read_text())
info = splits["credit-g"]

data = load_preprocessed("hyperfast", "credit-g", CACHE_DIR)
print(f"X_train: {data.X_train.shape}  dtype: {data.X_train.dtype}")
print(f"X_test:  {data.X_test.shape}")
print(f"y_train: unique={np.unique(data.y_train)}, counts={np.bincount(data.y_train)}")
print(f"NaN: {np.isnan(data.X_train).any()}")
print(f"X range: [{data.X_train.min():.3f}, {data.X_train.max():.3f}]")
print()
print("X_train[0, :10]:", data.X_train[0, :10])
print()

from hyperfast import HyperFastClassifier

clf = HyperFastClassifier(device="cuda")
clf.fit(data.X_train, data.y_train)
preds = clf.predict_proba(data.X_test)

print(f"Predictions: {preds.shape}")
print(f"Unique P(1) values (rounded .001): {len(np.unique(np.round(preds[:, 1], 3)))}")
print()

p1 = preds[:, 1]
print(f"P(1) stats: min={p1.min():.4f}  max={p1.max():.4f}  "
      f"mean={p1.mean():.4f}  std={p1.std():.4f}")
print()

print("P(1) distribution:")
for lo, hi in [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
    n = ((p1 >= lo) & (p1 < hi)).sum()
    print(f"  [{lo:.1f}, {hi:.1f}): {n}")

print()
print("Sample predictions (first 15):")
for i in range(min(15, len(preds))):
    print(f"  row {i}: P(0)={preds[i, 0]:.4f}  P(1)={preds[i, 1]:.4f}  "
          f"y={data.y_test[i]}")
