"""
Data loading utilities for tabular embedding experiments.

Supports:
1. TabZilla benchmark suite (36 "hard" datasets from OpenML)
2. OpenML-CC18 benchmark suite
3. Standard tabular datasets (classification and regression)
4. Synthetic data for controlled experiments

References:
- TabZilla: https://github.com/naszilla/tabzilla
- OpenML-CC18: https://www.openml.org/search?type=study&study_type=task&id=99
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# TabZilla "Hard" Benchmark Suite (36 datasets)
# From: "TabZilla: When Do Neural Nets Outperform Boosted Trees on Tabular Data?"
# These are datasets where simple baselines cannot reach top performance.
# =============================================================================
TABZILLA_HARD = {
    # Classification - Binary
    "electricity": 44120,
    "eye_movements": 1044,
    "MagicTelescope": 1120,
    "bank-marketing": 44234,
    "california": 44090,
    "credit": 44089,
    "default-of-credit-card-clients": 44233,
    "Diabetes130US": 44225,
    "heloc": 44230,
    "jannis": 44226,
    "MiniBooNE": 44128,
    "numerai28.6": 44229,
    "pol": 44231,
    "road-safety": 44161,

    # Classification - Multiclass
    "covertype": 44121,
    "albert": 44223,
    "dionis": 44232,
    "helena": 44227,
    "jungle_chess_2pcs_raw_endgame_complete": 44227,

    # Regression
    "Allstate_Claims_Severity": 44227,
    "Bike_Sharing_Demand": 44063,
    "Brazilian_houses": 44227,
    "house_sales": 44093,
    "particulate-matter-ukair-2017": 44163,
    "sulfur": 44160,
    "superconduct": 44126,
    "wine_quality": 44091,
    "yprop_4_1": 44162,
}

# =============================================================================
# OpenML-CC18 Benchmark Suite (72 classification datasets)
# Curated for ML benchmarking with controlled difficulty
# =============================================================================
OPENML_CC18 = {
    # Small datasets (good for few-shot evaluation)
    "iris": 61,
    "wine": 187,
    "breast-cancer": 13,
    "diabetes": 37,
    "vehicle": 54,
    "segment": 36,
    "vowel": 307,
    "balance-scale": 11,

    # Medium datasets
    "credit-g": 31,
    "kr-vs-kp": 3,
    "mushroom": 24,
    "tic-tac-toe": 50,
    "cmc": 23,
    "car": 21,
    "nursery": 26,
    "splice": 46,
    "waveform-5000": 60,
    "optdigits": 28,
    "pendigits": 32,
    "letter": 6,
    "satimage": 182,
    "texture": 40499,
    "mfeat-factors": 12,
    "mfeat-fourier": 14,
    "mfeat-karhunen": 16,
    "mfeat-morphological": 18,
    "mfeat-pixel": 20,
    "mfeat-zernike": 22,

    # Larger datasets
    "adult": 1590,
    "bank-marketing": 1461,
    "electricity": 151,
    "covertype": 1596,
    "shuttle": 40685,
    "phoneme": 1489,
    "wall-robot-navigation": 1497,
    "spambase": 44,
}

# =============================================================================
# Regression Benchmarks
# =============================================================================
REGRESSION_DATASETS = {
    # UCI / OpenML regression
    "boston": 531,           # Boston housing (classic)
    "diamonds": 44229,       # Diamond prices
    "california_housing": 44090,
    "bike_sharing": 44063,
    "wine_quality": 44091,
    "superconduct": 44126,
    "cpu_act": 44132,
    "pol": 44133,
    "elevators": 44134,
    "house_16H": 44136,
    "houses": 44137,
    "yprop_4_1": 44162,
}

# =============================================================================
# Quick Benchmark (subset for fast iteration)
# =============================================================================
QUICK_BENCHMARK = {
    # Small classification (few-shot friendly)
    "iris": 61,
    "wine": 187,
    "breast-cancer": 13,
    "vehicle": 54,

    # Medium classification
    "credit-g": 31,
    "adult": 1590,
    "electricity": 151,

    # Regression
    "california_housing": 44090,
    "bike_sharing": 44063,
}


def load_openml_dataset(
    dataset_id: int,
    max_samples: int = 10000,
    task: str = "auto",
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
    """
    Load a dataset from OpenML.

    Args:
        dataset_id: OpenML dataset ID
        max_samples: Maximum samples to return
        task: "classification", "regression", or "auto" (infer from target)

    Returns:
        (X, y, metadata) tuple or None
    """
    try:
        from sklearn.datasets import fetch_openml

        data = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
        X = data.data
        y = data.target

        # Determine task type
        if task == "auto":
            if y.dtype in ['float64', 'float32'] and len(y.unique()) > 20:
                task = "regression"
            else:
                task = "classification"

        # Handle categorical features
        cat_cols = X.select_dtypes(include=['category', 'object']).columns
        for col in cat_cols:
            X[col] = X[col].astype('category').cat.codes

        X = X.values.astype(np.float32)

        # Handle target
        if task == "classification":
            if y.dtype == 'object' or y.dtype.name == 'category':
                y = pd.Categorical(y).codes
            y = np.asarray(y).astype(int)
        else:
            y = np.asarray(y).astype(np.float32)

        # Remove samples with NaN in y
        valid_idx = ~np.isnan(y) if task == "regression" else np.ones(len(y), dtype=bool)
        X = X[valid_idx]
        y = y[valid_idx]

        # Subsample if needed
        if len(X) > max_samples:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(X), max_samples, replace=False)
            X = X[indices]
            y = y[indices]

        # Handle NaN in features
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        metadata = {
            "name": data.details.get("name", f"openml_{dataset_id}"),
            "openml_id": dataset_id,
            "task": task,
            "n_features": X.shape[1],
            "n_samples": len(X),
        }

        if task == "classification":
            metadata["n_classes"] = len(np.unique(y))
            metadata["class_balance"] = float(y.mean()) if len(np.unique(y)) == 2 else None
        else:
            metadata["y_mean"] = float(y.mean())
            metadata["y_std"] = float(y.std())

        return X, y, metadata

    except Exception as e:
        print(f"Error loading OpenML dataset {dataset_id}: {e}")
        return None


def load_dataset(
    name: str,
    max_samples: int = 10000,
    source: str = "auto",
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
    """
    Load a dataset by name from various benchmark suites.

    Args:
        name: Dataset name
        max_samples: Maximum samples to return
        source: "tabzilla", "cc18", "regression", "quick", "tabarena", or "auto"

    Returns:
        (X, y, metadata) tuple or None
    """
    # Try to find dataset in various suites
    dataset_id = None
    task = "auto"

    if source == "auto":
        if name in TABZILLA_HARD:
            dataset_id = TABZILLA_HARD[name]
        elif name in OPENML_CC18:
            dataset_id = OPENML_CC18[name]
        elif name in REGRESSION_DATASETS:
            dataset_id = REGRESSION_DATASETS[name]
            task = "regression"
        elif name in QUICK_BENCHMARK:
            dataset_id = QUICK_BENCHMARK[name]
        else:
            # Try TabArena via extended_loader
            try:
                from data.extended_loader import TABARENA_DATASETS, load_tabarena_dataset
                if name in TABARENA_DATASETS:
                    result = load_tabarena_dataset(name, max_samples=max_samples)
                    if result:
                        X, y, meta = result
                        return X, y, {
                            "name": meta.name,
                            "task": meta.task,
                            "n_features": meta.n_features,
                            "n_samples": meta.n_samples,
                            "n_classes": meta.n_classes,
                            "source": "tabarena",
                        }
            except ImportError:
                pass
            # Not found anywhere
            if dataset_id is None:
                print(f"Dataset '{name}' not found in source '{source}'")
                print(f"Available in tabzilla: {list(TABZILLA_HARD.keys())[:5]}...")
                print(f"Available in cc18: {list(OPENML_CC18.keys())[:5]}...")
                return None
    elif source == "tabzilla":
        dataset_id = TABZILLA_HARD.get(name)
    elif source == "cc18":
        dataset_id = OPENML_CC18.get(name)
    elif source == "regression":
        dataset_id = REGRESSION_DATASETS.get(name)
        task = "regression"
    elif source == "quick":
        dataset_id = QUICK_BENCHMARK.get(name)
    elif source == "tabarena":
        try:
            from data.extended_loader import load_tabarena_dataset
            result = load_tabarena_dataset(name, max_samples=max_samples)
            if result:
                X, y, meta = result
                return X, y, {
                    "name": meta.name,
                    "task": meta.task,
                    "n_features": meta.n_features,
                    "n_samples": meta.n_samples,
                    "n_classes": meta.n_classes,
                    "source": "tabarena",
                }
        except ImportError:
            print("Extended loader not available for TabArena datasets")
        return None

    if dataset_id is None:
        print(f"Dataset '{name}' not found in source '{source}'")
        return None

    return load_openml_dataset(dataset_id, max_samples=max_samples, task=task)


def generate_synthetic_classification(
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 10,
    n_classes: int = 2,
    class_sep: float = 1.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic classification data.

    Args:
        n_samples: Number of samples
        n_features: Total features
        n_informative: Number of informative features
        n_classes: Number of classes
        class_sep: Class separation factor
        random_state: Random seed

    Returns:
        (X, y, metadata) tuple
    """
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=max(0, n_features - n_informative - 2),
        n_clusters_per_class=2,
        n_classes=n_classes,
        class_sep=class_sep,
        random_state=random_state,
    )

    X = X.astype(np.float32)
    y = y.astype(int)

    metadata = {
        "name": f"synthetic_clf_{n_samples}x{n_features}",
        "task": "classification",
        "n_features": n_features,
        "n_samples": n_samples,
        "n_classes": n_classes,
        "n_informative": n_informative,
        "class_sep": class_sep,
    }

    return X, y, metadata


def generate_synthetic_regression(
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 10,
    noise: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic regression data.

    Args:
        n_samples: Number of samples
        n_features: Total features
        n_informative: Number of informative features
        noise: Noise level
        random_state: Random seed

    Returns:
        (X, y, metadata) tuple
    """
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state,
    )

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    metadata = {
        "name": f"synthetic_reg_{n_samples}x{n_features}",
        "task": "regression",
        "n_features": n_features,
        "n_samples": n_samples,
        "n_informative": n_informative,
        "noise": noise,
    }

    return X, y, metadata


def list_datasets(source: str = "all") -> Dict[str, List[str]]:
    """
    List available datasets by source.

    Args:
        source: "tabzilla", "cc18", "regression", "quick", "tabarena",
                "relbench", or "all"

    Returns:
        Dict mapping source name to list of dataset names
    """
    sources = {}

    if source in ["tabzilla", "all"]:
        sources["tabzilla"] = list(TABZILLA_HARD.keys())
    if source in ["cc18", "all"]:
        sources["cc18"] = list(OPENML_CC18.keys())
    if source in ["regression", "all"]:
        sources["regression"] = list(REGRESSION_DATASETS.keys())
    if source in ["quick", "all"]:
        sources["quick"] = list(QUICK_BENCHMARK.keys())
    if source in ["tabarena", "all"]:
        try:
            from data.extended_loader import TABARENA_DATASETS
            sources["tabarena"] = list(TABARENA_DATASETS.keys())
        except ImportError:
            pass
    if source in ["relbench", "all"]:
        try:
            from data.extended_loader import RELBENCH_TASKS
            sources["relbench"] = [
                f"{ds}/{task}" for ds, tasks in RELBENCH_TASKS.items()
                for task, _ in tasks
            ]
        except ImportError:
            pass

    return sources


def load_benchmark_suite(
    suite: str = "quick",
    max_samples: int = 5000,
    max_datasets: int = None,
) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
    """
    Load all datasets from a benchmark suite.

    Args:
        suite: "tabzilla", "cc18", "regression", or "quick"
        max_samples: Max samples per dataset
        max_datasets: Max number of datasets to load

    Returns:
        List of (X, y, metadata) tuples
    """
    if suite == "tabzilla":
        datasets = TABZILLA_HARD
    elif suite == "cc18":
        datasets = OPENML_CC18
    elif suite == "regression":
        datasets = REGRESSION_DATASETS
    elif suite == "quick":
        datasets = QUICK_BENCHMARK
    elif suite == "tabarena":
        try:
            from data.extended_loader import load_tabarena_suite
            return load_tabarena_suite(
                max_samples=max_samples, max_datasets=max_datasets
            )
        except ImportError:
            raise ValueError("Extended loader not available for TabArena suite")
    elif suite == "relbench":
        try:
            from data.extended_loader import load_relbench_suite
            return load_relbench_suite(
                max_samples=max_samples, max_datasets=max_datasets
            )
        except ImportError:
            raise ValueError("Extended loader not available for RelBench suite")
    else:
        raise ValueError(f"Unknown suite: {suite}")

    results = []
    for i, name in enumerate(datasets.keys()):
        if max_datasets and i >= max_datasets:
            break

        print(f"Loading {name}...", end=" ", flush=True)
        result = load_dataset(name, max_samples=max_samples, source=suite)
        if result:
            print(f"OK ({result[2]['n_samples']} samples, {result[2]['n_features']} features)")
            results.append(result)
        else:
            print("FAILED")

    return results


if __name__ == "__main__":
    # Test data loading
    print("=" * 60)
    print("Testing Data Loaders")
    print("=" * 60)

    # List available datasets
    print("\nAvailable datasets:")
    for source, names in list_datasets("all").items():
        print(f"  {source}: {len(names)} datasets")
        print(f"    Examples: {names[:3]}...")

    # Test synthetic data
    print("\n" + "-" * 60)
    print("Synthetic Classification:")
    X, y, meta = generate_synthetic_classification(n_samples=500, n_features=15)
    print(f"  {meta}")
    print(f"  X: {X.shape}, y: {y.shape}, balance: {y.mean():.2%}")

    print("\nSynthetic Regression:")
    X, y, meta = generate_synthetic_regression(n_samples=500, n_features=15)
    print(f"  {meta}")
    print(f"  X: {X.shape}, y: {y.shape}, y_mean: {y.mean():.2f}")

    # Test OpenML loading
    print("\n" + "-" * 60)
    print("OpenML Datasets:")

    for name in ["iris", "adult", "california_housing"]:
        result = load_dataset(name, max_samples=1000)
        if result:
            X, y, meta = result
            print(f"  {name}: {meta['n_samples']}x{meta['n_features']}, task={meta['task']}")
