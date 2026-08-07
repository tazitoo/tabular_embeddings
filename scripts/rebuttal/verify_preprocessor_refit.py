#!/usr/bin/env python3
"""Does refitting the outer preprocessing reproduce the cached matrix EXACTLY?

We want to hold the fitted feature generator so an edited raw row can be put through the
same preprocessing the model was fit on, instead of editing the preprocessed matrix and
inverting back to raw values for reporting. That is only sound if a refit reproduces the
cache bit for bit. If it does not, a persisted generator silently produces model inputs
that differ from the ones the corpus, the SAE and every cached activation were built on.

Two specific ways this could fail, both checked here rather than assumed:

  fit_transform vs transform   The cache was built with fg.fit_transform(X_train) for the
                               train split. FittedPreprocessor.transform necessarily calls
                               fg.transform(X_train). AutoGluon generators are not
                               required to agree on those two.
  determinism                  Category ORDER decides .cat.codes. If a refit orders
                               categories differently, every code shifts and nothing about
                               it looks wrong -- the matrix is still valid, just not the
                               one the model was fit on.

Equality is exact, including the NaN pattern: NaN marks a category unseen in training
(code -1), which is information, and a tolerance would hide a shifted code as a small
difference.

Usage:
    python -m scripts.rebuttal.verify_preprocessor_refit
    python -m scripts.rebuttal.verify_preprocessor_refit --models tabpfn tabicl --datasets MIC
"""
import argparse
import json

import numpy as np
import pandas as pd

from scripts._project_root import PROJECT_ROOT
from data.preprocessing import (
    CACHE_DIR, NAN_SAFE_MODELS, fit_preprocessor, load_preprocessed,
)
from scripts.intervention.intervene_lib import SPLITS_PATH

RAW_CACHE = PROJECT_ROOT / "data" / "cache" / "tabarena"


def raw_frame(dataset):
    p = RAW_CACHE / f"{dataset}_v2.parquet"
    y = RAW_CACHE / f"{dataset}_v2_y.npy"
    if not p.exists() or not y.exists():
        return None, None
    return pd.read_parquet(p), np.load(y, allow_pickle=True)


def compare(a, b):
    """Exact match including NaN placement. Returns (ok, detail)."""
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    na, nb = np.isnan(a), np.isnan(b)
    if not np.array_equal(na, nb):
        return False, f"NaN pattern differs in {int((na != nb).sum())} cells"
    same = np.array_equal(a[~na], b[~nb])
    if same:
        return True, "exact"
    d = np.abs(a[~na] - b[~nb])
    return False, (f"{int((d > 0).sum())}/{d.size} cells differ, "
                   f"max |diff| {float(d.max()):.6g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["tabpfn", "tabdpt", "tabicl", "mitra"])
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="default: every dataset with a raw v2 cache AND a preprocessed cache")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output" / "rebuttal" /
                                         "preprocessor_refit_check.json"))
    args = ap.parse_args()

    splits = json.loads(SPLITS_PATH.read_text())
    datasets = args.datasets or sorted(
        p.name[:-len("_v2.parquet")] for p in RAW_CACHE.glob("*_v2.parquet"))

    rows, n_ok, n_bad, n_skip = [], 0, 0, 0
    for model in args.models:
        for ds in datasets:
            if ds not in splits:
                n_skip += 1; continue
            try:
                cached = load_preprocessed(model, ds, CACHE_DIR)
            except Exception:
                n_skip += 1; continue
            X_df, _ = raw_frame(ds)
            if X_df is None:
                n_skip += 1; continue

            si = splits[ds]
            X_train = X_df.iloc[np.array(si["train_indices"])].reset_index(drop=True)
            X_test = X_df.iloc[np.array(si["test_indices"])].reset_index(drop=True)
            try:
                pre = fit_preprocessor(X_train, nan_safe=model in NAN_SAFE_MODELS)
                ok_tr, d_tr = compare(pre.transform(X_train), cached.X_train)
                ok_te, d_te = compare(pre.transform(X_test), cached.X_test)
                ok_ci = sorted(pre.cat_indices) == sorted(cached.cat_indices or [])
            except Exception as exc:
                rows.append({"model": model, "dataset": ds,
                             "error": f"{type(exc).__name__}: {exc}"})
                n_bad += 1
                print(f"  {model:9s} {ds:32s} ERROR {type(exc).__name__}: {exc}", flush=True)
                continue

            ok = ok_tr and ok_te and ok_ci
            n_ok += ok; n_bad += not ok
            rows.append({"model": model, "dataset": ds, "ok": bool(ok),
                         "train": d_tr, "test": d_te, "cat_indices_match": bool(ok_ci),
                         "n_cat_refit": len(pre.cat_indices),
                         "n_cat_cached": len(cached.cat_indices or [])})
            if not ok:
                print(f"  {model:9s} {ds:32s} MISMATCH  train[{d_tr}]  test[{d_te}]  "
                      f"cat_indices {'ok' if ok_ci else 'DIFFER'}", flush=True)

    print(f"\n{n_ok} exact, {n_bad} mismatched, {n_skip} skipped (no cache)")
    json.dump({"n_ok": n_ok, "n_bad": n_bad, "n_skipped": n_skip, "rows": rows},
              open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")
    if n_bad:
        print("\nA mismatch means a persisted generator would NOT reproduce the inputs the\n"
              "corpus was built on. Do not build raw-space patching on it until this is 0.")


if __name__ == "__main__":
    main()
