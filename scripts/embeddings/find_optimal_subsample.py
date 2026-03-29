#!/usr/bin/env python3
"""
Find the minimum subsample size that represents each dataset's distribution.

For each TabArena dataset, draws stratified subsamples at increasing sizes
and measures distribution divergence (energy distance) against the full
dataset. Reports the smallest size where the distribution stabilizes.

No GPU required — operates on raw features.

Output:
  output/sae_subsample_analysis/subsample_recommendations.json

Usage:
  python scripts/embeddings/find_optimal_subsample.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from scripts._project_root import PROJECT_ROOT
from data.extended_loader import load_tabarena_dataset, TABARENA_DATASETS

OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_subsample_analysis"
SPLIT_SEED = 42
CANDIDATE_SIZES = [100, 200, 350, 500, 1000, 2000, 3500, 5000]
N_TRIALS = 5  # Average over multiple draws to reduce noise
CONVERGENCE_THRESHOLD = 0.05  # Relative improvement < 5% = converged


def energy_distance(X: np.ndarray, Y: np.ndarray, max_samples: int = 2000) -> float:
    """Compute energy distance between two multivariate samples.

    Energy distance is 0 iff the distributions are identical.
    More robust than KL for multivariate continuous distributions —
    no binning, no density estimation, works in any dimension.

    Subsamples both sets to max_samples for computational tractability.
    """
    if len(X) > max_samples:
        rng = np.random.RandomState(SPLIT_SEED + 99)
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        rng = np.random.RandomState(SPLIT_SEED + 100)
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]

    # E(d) = 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]
    dXY = cdist(X, Y, metric="euclidean").mean()
    dXX = cdist(X, X, metric="euclidean").mean()
    dYY = cdist(Y, Y, metric="euclidean").mean()

    return float(2 * dXY - dXX - dYY)


def stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    n: int,
    task: str,
    seed: int,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a stratified subsample of size n.

    Classification: stratify by class label.
    Regression: stratify by quantile-binned target.
    """
    if n >= len(X):
        return X, y

    if task == "classification":
        strat = y
    else:
        # Bin continuous target into quantiles for stratification
        bins = min(n_bins, len(np.unique(y)))
        try:
            strat = np.digitize(y, np.percentile(y, np.linspace(0, 100, bins + 1)[1:-1]))
        except Exception:
            strat = None

    frac = n / len(X)
    try:
        X_sub, _, y_sub, _ = train_test_split(
            X, y, train_size=frac, random_state=seed, stratify=strat,
        )
    except ValueError:
        # Stratification failed (too few samples per stratum)
        X_sub, _, y_sub, _ = train_test_split(
            X, y, train_size=frac, random_state=seed,
        )

    return X_sub, y_sub


def find_convergence(sizes: list[int], distances: list[float]) -> int:
    """Find the smallest size where energy distance has converged.

    Convergence = relative improvement from size[i] to size[i+1] < threshold.
    """
    for i in range(len(distances) - 1):
        if distances[i] == 0:
            return sizes[i]
        rel_improvement = (distances[i] - distances[i + 1]) / distances[i]
        if rel_improvement < CONVERGENCE_THRESHOLD:
            return sizes[i]

    return sizes[-1]


def analyze_dataset(name: str, task: str) -> dict | None:
    """Find optimal subsample size for one dataset."""
    result = load_tabarena_dataset(name, max_samples=999999)
    if result is None:
        return None

    X, y, _ = result
    n_full = len(X)

    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Standardize features for distance computation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Test each candidate size
    valid_sizes = [s for s in CANDIDATE_SIZES if s < n_full * 0.7]  # Leave room for holdout
    if not valid_sizes:
        return {
            "n_full": n_full,
            "recommended_size": n_full,
            "note": "dataset too small for subsampling",
            "distances": {},
        }

    size_distances = {}
    for size in valid_sizes:
        trial_dists = []
        for trial in range(N_TRIALS):
            X_sub, _ = stratified_subsample(
                X_scaled, y, size, task, seed=SPLIT_SEED + trial,
            )
            d = energy_distance(X_sub, X_scaled)
            trial_dists.append(d)
        size_distances[size] = {
            "mean": round(float(np.mean(trial_dists)), 6),
            "std": round(float(np.std(trial_dists)), 6),
        }

    # Find convergence point
    dist_means = [size_distances[s]["mean"] for s in valid_sizes]
    recommended = find_convergence(valid_sizes, dist_means)

    return {
        "n_full": n_full,
        "task": task,
        "recommended_size": recommended,
        "distances": size_distances,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    print(f"{'Dataset':<40} {'N':>7} {'Task':<6} {'Recommended':>12}")
    print("-" * 70)

    for name in sorted(TABARENA_DATASETS.keys()):
        task = TABARENA_DATASETS[name].get("task", "classification")
        print(f"  {name}...", end=" ", flush=True)

        r = analyze_dataset(name, task)
        if r is None:
            print("SKIP")
            continue

        results[name] = r
        print(f"\r{name:<40} {r['n_full']:>7} {task:<6} {r['recommended_size']:>12}")

    # Summary
    sizes = [r["recommended_size"] for r in results.values()]
    total_recommended = sum(min(r["recommended_size"], r["n_full"]) for r in results.values())
    total_train = int(total_recommended * 0.7)
    total_test = total_recommended - total_train

    summary = {
        "n_datasets": len(results),
        "total_recommended_rows": total_recommended,
        "total_train_70pct": total_train,
        "total_test_30pct": total_test,
        "median_recommended_size": int(np.median(sizes)),
        "mean_recommended_size": int(np.mean(sizes)),
        "max_recommended_size": int(np.max(sizes)),
        "size_distribution": {
            str(s): sum(1 for sz in sizes if sz == s)
            for s in sorted(set(sizes))
        },
    }

    output = {"summary": summary, "datasets": results}
    out_path = OUTPUT_DIR / "subsample_recommendations.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Total recommended: {total_recommended:,} rows ({total_train:,} train / {total_test:,} test)")
    print(f"Median per dataset: {summary['median_recommended_size']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
