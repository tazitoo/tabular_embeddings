"""
Extended data loading for probing embedding space corners.

This module provides diverse data sources to systematically probe
different regions of the embedding space:

1. Standard Benchmarks: PMLB, TabZilla, OpenML-CC18
2. Scientific Domains: Materials science, genomics-like, physics
3. Controlled Synthetic: Systematic variation of data properties
4. Edge Cases: High-dim, sparse, highly imbalanced, etc.

The goal is to find datasets that activate different "neurons" or
features in the foundation models' embedding spaces.
"""

from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class DatasetMetadata:
    """Rich metadata for dataset characterization."""
    name: str
    source: str  # openml, pmlb, synthetic, kaggle, etc.
    task: str  # classification, regression
    n_samples: int
    n_features: int
    n_classes: Optional[int] = None

    # Data characteristics (for probing analysis)
    feature_types: str = "numeric"  # numeric, categorical, mixed
    sparsity: float = 0.0  # Fraction of zeros
    noise_level: str = "unknown"  # clean, moderate, noisy
    difficulty: str = "unknown"  # easy, medium, hard
    domain: str = "unknown"  # medical, financial, scientific, etc.

    # Derived properties
    dim_ratio: float = 0.0  # n_features / n_samples
    class_balance: Optional[float] = None


# =============================================================================
# PMLB (Penn Machine Learning Benchmarks)
# ~150 curated datasets with good metadata
# =============================================================================
PMLB_DATASETS = {
    # Small datasets (good for few-shot, TabPFN sweet spot)
    "analcatdata_authorship": {"task": "classification", "domain": "text"},
    "analcatdata_dmft": {"task": "classification", "domain": "medical"},
    "appendicitis": {"task": "classification", "domain": "medical"},
    "australian": {"task": "classification", "domain": "financial"},
    "backache": {"task": "classification", "domain": "medical"},
    "breast_cancer": {"task": "classification", "domain": "medical"},
    "breast_cancer_wisconsin": {"task": "classification", "domain": "medical"},
    "chess": {"task": "classification", "domain": "game"},
    "churn": {"task": "classification", "domain": "business"},
    "clean1": {"task": "classification", "domain": "synthetic"},
    "cleve": {"task": "classification", "domain": "medical"},
    "coil2000": {"task": "classification", "domain": "insurance"},
    "corral": {"task": "classification", "domain": "synthetic"},
    "credit_a": {"task": "classification", "domain": "financial"},
    "credit_g": {"task": "classification", "domain": "financial"},
    "diabetes": {"task": "classification", "domain": "medical"},
    "dis": {"task": "classification", "domain": "medical"},
    "glass": {"task": "classification", "domain": "materials"},
    "heart_c": {"task": "classification", "domain": "medical"},
    "heart_h": {"task": "classification", "domain": "medical"},
    "hepatitis": {"task": "classification", "domain": "medical"},
    "horse_colic": {"task": "classification", "domain": "veterinary"},
    "house_votes_84": {"task": "classification", "domain": "politics"},
    "hungarian": {"task": "classification", "domain": "medical"},
    "hypothyroid": {"task": "classification", "domain": "medical"},
    "ionosphere": {"task": "classification", "domain": "physics"},
    "iris": {"task": "classification", "domain": "biology"},
    "kr_vs_kp": {"task": "classification", "domain": "game"},
    "labor": {"task": "classification", "domain": "economics"},
    "led24": {"task": "classification", "domain": "synthetic"},
    "letter": {"task": "classification", "domain": "image"},
    "liver_disorder": {"task": "classification", "domain": "medical"},
    "magic": {"task": "classification", "domain": "physics"},
    "mofn_3_7_10": {"task": "classification", "domain": "synthetic"},
    "molecular_biology_promoters": {"task": "classification", "domain": "genomics"},
    "monk1": {"task": "classification", "domain": "synthetic"},
    "monk2": {"task": "classification", "domain": "synthetic"},
    "monk3": {"task": "classification", "domain": "synthetic"},
    "mushroom": {"task": "classification", "domain": "biology"},
    "page_blocks": {"task": "classification", "domain": "document"},
    "parity5": {"task": "classification", "domain": "synthetic"},
    "pendigits": {"task": "classification", "domain": "image"},
    "pima": {"task": "classification", "domain": "medical"},
    "postoperative_patient_data": {"task": "classification", "domain": "medical"},
    "prnn_crabs": {"task": "classification", "domain": "biology"},
    "prnn_synth": {"task": "classification", "domain": "synthetic"},
    "profb": {"task": "classification", "domain": "sports"},
    "promoters": {"task": "classification", "domain": "genomics"},
    "ring": {"task": "classification", "domain": "synthetic"},
    "saheart": {"task": "classification", "domain": "medical"},
    "satimage": {"task": "classification", "domain": "satellite"},
    "segment": {"task": "classification", "domain": "image"},
    "shuttle": {"task": "classification", "domain": "aerospace"},
    "sleep": {"task": "classification", "domain": "medical"},
    "sonar": {"task": "classification", "domain": "physics"},
    "soybean": {"task": "classification", "domain": "agriculture"},
    "spambase": {"task": "classification", "domain": "text"},
    "spect": {"task": "classification", "domain": "medical"},
    "spectf": {"task": "classification", "domain": "medical"},
    "splice": {"task": "classification", "domain": "genomics"},
    "threeOf9": {"task": "classification", "domain": "synthetic"},
    "tic_tac_toe": {"task": "classification", "domain": "game"},
    "tokyo1": {"task": "classification", "domain": "unknown"},
    "twonorm": {"task": "classification", "domain": "synthetic"},
    "vehicle": {"task": "classification", "domain": "image"},
    "vote": {"task": "classification", "domain": "politics"},
    "vowel": {"task": "classification", "domain": "audio"},
    "waveform_21": {"task": "classification", "domain": "synthetic"},
    "waveform_40": {"task": "classification", "domain": "synthetic"},
    "wine_quality_red": {"task": "classification", "domain": "chemistry"},
    "wine_quality_white": {"task": "classification", "domain": "chemistry"},
    "wine_recognition": {"task": "classification", "domain": "chemistry"},
    "xd6": {"task": "classification", "domain": "synthetic"},

    # Regression
    "1027_ESL": {"task": "regression", "domain": "unknown"},
    "1028_SWD": {"task": "regression", "domain": "unknown"},
    "1029_LEV": {"task": "regression", "domain": "unknown"},
    "1030_ERA": {"task": "regression", "domain": "unknown"},
    "1089_USCrime": {"task": "regression", "domain": "social"},
    "1096_FacultySalaries": {"task": "regression", "domain": "economics"},
    "192_vineyard": {"task": "regression", "domain": "agriculture"},
    "195_auto_price": {"task": "regression", "domain": "automotive"},
    "197_cpu_act": {"task": "regression", "domain": "computing"},
    "201_pol": {"task": "regression", "domain": "unknown"},
    "207_autoPrice": {"task": "regression", "domain": "automotive"},
    "210_cloud": {"task": "regression", "domain": "meteorology"},
    "215_2dplanes": {"task": "regression", "domain": "synthetic"},
    "218_house_8L": {"task": "regression", "domain": "real_estate"},
    "225_puma8NH": {"task": "regression", "domain": "robotics"},
    "228_elusage": {"task": "regression", "domain": "energy"},
    "229_pwLinear": {"task": "regression", "domain": "synthetic"},
    "230_machine_cpu": {"task": "regression", "domain": "computing"},
    "294_satellite_image": {"task": "regression", "domain": "satellite"},
    "344_mv": {"task": "regression", "domain": "unknown"},
    "4544_GeographicalOriginalofMusic": {"task": "regression", "domain": "music"},
    "503_wind": {"task": "regression", "domain": "energy"},
    "505_tecator": {"task": "regression", "domain": "food"},
    "519_vinnie": {"task": "regression", "domain": "unknown"},
    "522_pm10": {"task": "regression", "domain": "environment"},
    "523_analcatdata_neavote": {"task": "regression", "domain": "politics"},
    "527_analcatdata_election2000": {"task": "regression", "domain": "politics"},
    "529_pollen": {"task": "regression", "domain": "biology"},
    "537_houses": {"task": "regression", "domain": "real_estate"},
    "542_pollution": {"task": "regression", "domain": "environment"},
    "547_no2": {"task": "regression", "domain": "environment"},
    "556_analcatdata_apnea2": {"task": "regression", "domain": "medical"},
    "557_analcatdata_apnea1": {"task": "regression", "domain": "medical"},
    "561_cpu": {"task": "regression", "domain": "computing"},
    "564_fried": {"task": "regression", "domain": "synthetic"},
    "573_cpu_small": {"task": "regression", "domain": "computing"},
    "574_house_16H": {"task": "regression", "domain": "real_estate"},
}


def load_pmlb_dataset(
    name: str,
    max_samples: int = 10000,
) -> Optional[Tuple[np.ndarray, np.ndarray, DatasetMetadata]]:
    """
    Load a dataset from PMLB.

    Args:
        name: Dataset name from PMLB_DATASETS
        max_samples: Maximum samples to return

    Returns:
        (X, y, metadata) tuple or None
    """
    try:
        import pmlb

        if name not in PMLB_DATASETS:
            print(f"Dataset '{name}' not in PMLB_DATASETS catalog")
            return None

        info = PMLB_DATASETS[name]

        # Load dataset
        X, y = pmlb.fetch_data(name, return_X_y=True, local_cache_dir=".pmlb_cache")

        # Convert to numpy
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        # Handle task type
        if info["task"] == "classification":
            if y.dtype == 'object' or y.dtype.name == 'category':
                y = pd.Categorical(y).codes
            y = y.astype(int)
        else:
            y = y.astype(np.float32)

        # Subsample if needed
        if len(X) > max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), max_samples, replace=False)
            X, y = X[idx], y[idx]

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Create metadata
        meta = DatasetMetadata(
            name=name,
            source="pmlb",
            task=info["task"],
            n_samples=len(X),
            n_features=X.shape[1],
            n_classes=len(np.unique(y)) if info["task"] == "classification" else None,
            domain=info.get("domain", "unknown"),
            dim_ratio=X.shape[1] / len(X),
            sparsity=(X == 0).mean(),
        )

        if info["task"] == "classification" and len(np.unique(y)) == 2:
            meta.class_balance = float(y.mean())

        return X, y, meta

    except ImportError:
        print("PMLB not installed. Run: pip install pmlb")
        return None
    except Exception as e:
        print(f"Error loading PMLB dataset {name}: {e}")
        return None


# =============================================================================
# Controlled Synthetic Data Generators
# For systematic probing of embedding space
# =============================================================================

def generate_linear_separable(
    n_samples: int = 1000,
    n_features: int = 20,
    margin: float = 1.0,
    noise: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """
    Generate linearly separable data with controlled margin.

    Tests: Does the model encode linear decision boundaries?
    """
    rng = np.random.RandomState(random_state)

    # Generate separating hyperplane
    w = rng.randn(n_features)
    w = w / np.linalg.norm(w)

    # Generate points
    X = rng.randn(n_samples, n_features).astype(np.float32)

    # Labels based on which side of hyperplane
    scores = X @ w
    y = (scores > 0).astype(int)

    # Add margin
    X[y == 0] -= margin * w / 2
    X[y == 1] += margin * w / 2

    # Add noise
    X += noise * rng.randn(n_samples, n_features)

    meta = DatasetMetadata(
        name=f"linear_sep_m{margin}_n{noise}",
        source="synthetic",
        task="classification",
        n_samples=n_samples,
        n_features=n_features,
        n_classes=2,
        difficulty="easy" if margin > 0.5 else "medium",
        noise_level="clean" if noise < 0.1 else "moderate",
        class_balance=float(y.mean()),
    )

    return X, y, meta


def generate_xor_pattern(
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 4,
    noise: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """
    Generate XOR-like pattern (nonlinear decision boundary).

    Tests: Does the model encode nonlinear feature interactions?
    """
    rng = np.random.RandomState(random_state)

    X = rng.randn(n_samples, n_features).astype(np.float32)

    # XOR on first n_informative features (pairwise)
    y = np.zeros(n_samples, dtype=int)
    for i in range(0, n_informative, 2):
        if i + 1 < n_informative:
            y ^= ((X[:, i] > 0) ^ (X[:, i + 1] > 0)).astype(int)

    # Add noise
    X += noise * rng.randn(n_samples, n_features)

    meta = DatasetMetadata(
        name=f"xor_pattern_{n_informative}feat",
        source="synthetic",
        task="classification",
        n_samples=n_samples,
        n_features=n_features,
        n_classes=2,
        difficulty="hard",
        noise_level="clean" if noise < 0.1 else "moderate",
        class_balance=float(y.mean()),
    )

    return X, y, meta


def generate_hierarchical_features(
    n_samples: int = 1000,
    n_groups: int = 4,
    features_per_group: int = 5,
    within_group_corr: float = 0.8,
    between_group_corr: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """
    Generate features with hierarchical correlation structure.

    Tests: Does the model encode feature group structure?
    """
    rng = np.random.RandomState(random_state)
    n_features = n_groups * features_per_group

    # Generate group factors
    group_factors = rng.randn(n_samples, n_groups)

    # Generate correlated features within groups
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    for g in range(n_groups):
        start = g * features_per_group
        end = start + features_per_group
        for f in range(features_per_group):
            X[:, start + f] = (
                within_group_corr * group_factors[:, g] +
                np.sqrt(1 - within_group_corr**2) * rng.randn(n_samples)
            )

    # Add between-group correlation via global factor
    global_factor = rng.randn(n_samples)
    X += between_group_corr * global_factor[:, np.newaxis]

    # Target based on group means
    group_means = group_factors.mean(axis=1)
    y = (group_means > 0).astype(int)

    meta = DatasetMetadata(
        name=f"hierarchical_{n_groups}g_{features_per_group}f",
        source="synthetic",
        task="classification",
        n_samples=n_samples,
        n_features=n_features,
        n_classes=2,
        difficulty="medium",
        feature_types="numeric",
        class_balance=float(y.mean()),
    )

    return X, y, meta


def generate_high_dimensional_sparse(
    n_samples: int = 500,
    n_features: int = 1000,
    sparsity: float = 0.95,
    n_informative: int = 20,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """
    Generate high-dimensional sparse data.

    Tests: How does the model handle high-dim sparse inputs?
    """
    rng = np.random.RandomState(random_state)

    # Sparse features
    X = rng.randn(n_samples, n_features).astype(np.float32)
    mask = rng.rand(n_samples, n_features) < sparsity
    X[mask] = 0

    # Informative features determine label
    informative_idx = rng.choice(n_features, n_informative, replace=False)
    scores = X[:, informative_idx].sum(axis=1)
    y = (scores > np.median(scores)).astype(int)

    meta = DatasetMetadata(
        name=f"high_dim_sparse_{n_features}d_{sparsity:.0%}sp",
        source="synthetic",
        task="classification",
        n_samples=n_samples,
        n_features=n_features,
        n_classes=2,
        sparsity=sparsity,
        difficulty="hard",
        dim_ratio=n_features / n_samples,
        class_balance=float(y.mean()),
    )

    return X, y, meta


def generate_categorical_heavy(
    n_samples: int = 1000,
    n_numeric: int = 5,
    n_categorical: int = 10,
    cardinality: int = 10,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """
    Generate data with many categorical features (one-hot encoded).

    Tests: How does the model handle categorical-heavy data?
    """
    rng = np.random.RandomState(random_state)

    # Numeric features
    X_numeric = rng.randn(n_samples, n_numeric).astype(np.float32)

    # Categorical features (one-hot)
    X_cat_list = []
    for _ in range(n_categorical):
        cat_vals = rng.randint(0, cardinality, n_samples)
        one_hot = np.eye(cardinality)[cat_vals]
        X_cat_list.append(one_hot)

    X_cat = np.hstack(X_cat_list).astype(np.float32)
    X = np.hstack([X_numeric, X_cat])

    # Target based on both numeric and categorical
    scores = X_numeric.sum(axis=1) + X_cat[:, :cardinality].sum(axis=1)
    y = (scores > np.median(scores)).astype(int)

    n_features = n_numeric + n_categorical * cardinality

    meta = DatasetMetadata(
        name=f"categorical_heavy_{n_categorical}cat_{cardinality}card",
        source="synthetic",
        task="classification",
        n_samples=n_samples,
        n_features=n_features,
        n_classes=2,
        feature_types="mixed",
        difficulty="medium",
        class_balance=float(y.mean()),
    )

    return X, y, meta


def generate_time_series_tabular(
    n_samples: int = 1000,
    n_timesteps: int = 10,
    n_features_per_step: int = 5,
    include_lags: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """
    Generate time series data flattened to tabular format.

    Tests: Does the model capture temporal patterns in tabular form?
    """
    rng = np.random.RandomState(random_state)

    # Generate underlying time series
    # AR(1) process with some structure
    base_series = np.zeros((n_samples, n_timesteps))
    base_series[:, 0] = rng.randn(n_samples)
    for t in range(1, n_timesteps):
        base_series[:, t] = 0.7 * base_series[:, t-1] + 0.3 * rng.randn(n_samples)

    # Generate features at each timestep
    feature_list = []
    for t in range(n_timesteps):
        step_features = rng.randn(n_samples, n_features_per_step) + base_series[:, t:t+1]
        feature_list.append(step_features)

    X = np.hstack(feature_list).astype(np.float32)

    # Add lag features if requested
    if include_lags:
        lags = []
        for lag in [1, 2, 5]:
            if lag < n_timesteps:
                lag_feat = np.roll(base_series, lag, axis=1)
                lag_feat[:, :lag] = 0
                lags.append(lag_feat)
        if lags:
            X = np.hstack([X] + lags).astype(np.float32)

    # Target: trend direction (up or down)
    trend = base_series[:, -1] - base_series[:, 0]
    y = (trend > 0).astype(int)

    meta = DatasetMetadata(
        name=f"ts_tabular_{n_timesteps}steps",
        source="synthetic",
        task="classification",
        n_samples=n_samples,
        n_features=X.shape[1],
        n_classes=2,
        domain="time_series",
        difficulty="medium",
        class_balance=float(y.mean()),
    )

    return X, y, meta


def generate_imbalanced_data(
    n_samples: int = 1000,
    n_features: int = 20,
    imbalance_ratio: float = 0.05,  # Minority class fraction
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """
    Generate highly imbalanced classification data.

    Tests: How does embedding quality change with class imbalance?
    """
    from sklearn.datasets import make_classification

    n_minority = int(n_samples * imbalance_ratio)
    n_majority = n_samples - n_minority

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_classes=2,
        weights=[1 - imbalance_ratio, imbalance_ratio],
        random_state=random_state,
    )

    X = X.astype(np.float32)
    y = y.astype(int)

    meta = DatasetMetadata(
        name=f"imbalanced_{imbalance_ratio:.0%}minority",
        source="synthetic",
        task="classification",
        n_samples=n_samples,
        n_features=n_features,
        n_classes=2,
        difficulty="hard",
        class_balance=float(y.mean()),
    )

    return X, y, meta


def generate_noisy_labels(
    n_samples: int = 1000,
    n_features: int = 20,
    label_noise: float = 0.2,  # Fraction of labels to flip
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """
    Generate data with noisy (flipped) labels.

    Tests: How robust are embeddings to label noise?
    """
    from sklearn.datasets import make_classification

    rng = np.random.RandomState(random_state)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_classes=2,
        random_state=random_state,
    )

    X = X.astype(np.float32)
    y = y.astype(int)

    # Flip labels
    n_flip = int(n_samples * label_noise)
    flip_idx = rng.choice(n_samples, n_flip, replace=False)
    y[flip_idx] = 1 - y[flip_idx]

    meta = DatasetMetadata(
        name=f"noisy_labels_{label_noise:.0%}noise",
        source="synthetic",
        task="classification",
        n_samples=n_samples,
        n_features=n_features,
        n_classes=2,
        noise_level=f"{label_noise:.0%}_label_noise",
        difficulty="hard",
        class_balance=float(y.mean()),
    )

    return X, y, meta


# =============================================================================
# Probing Suite: Systematic dataset collection
# =============================================================================

PROBING_GENERATORS = {
    # Linear vs nonlinear
    "linear_easy": lambda: generate_linear_separable(margin=2.0, noise=0.05),
    "linear_hard": lambda: generate_linear_separable(margin=0.2, noise=0.3),
    "xor_4feat": lambda: generate_xor_pattern(n_informative=4),
    "xor_8feat": lambda: generate_xor_pattern(n_informative=8),

    # Feature structure
    "hierarchical_4g": lambda: generate_hierarchical_features(n_groups=4),
    "hierarchical_8g": lambda: generate_hierarchical_features(n_groups=8),

    # Dimensionality
    "high_dim_500d": lambda: generate_high_dimensional_sparse(n_features=500, sparsity=0.9),
    "high_dim_1000d": lambda: generate_high_dimensional_sparse(n_features=1000, sparsity=0.95),

    # Feature types
    "categorical_5cat": lambda: generate_categorical_heavy(n_categorical=5),
    "categorical_20cat": lambda: generate_categorical_heavy(n_categorical=20),

    # Temporal
    "ts_5steps": lambda: generate_time_series_tabular(n_timesteps=5),
    "ts_20steps": lambda: generate_time_series_tabular(n_timesteps=20),

    # Edge cases
    "imbalanced_5pct": lambda: generate_imbalanced_data(imbalance_ratio=0.05),
    "imbalanced_1pct": lambda: generate_imbalanced_data(imbalance_ratio=0.01),
    "noisy_10pct": lambda: generate_noisy_labels(label_noise=0.1),
    "noisy_30pct": lambda: generate_noisy_labels(label_noise=0.3),
}


def load_probing_suite(
    subset: Optional[List[str]] = None,
) -> List[Tuple[np.ndarray, np.ndarray, DatasetMetadata]]:
    """
    Load the full probing suite of synthetic datasets.

    Args:
        subset: Optional list of specific generators to use

    Returns:
        List of (X, y, metadata) tuples
    """
    generators = subset or list(PROBING_GENERATORS.keys())
    results = []

    for name in generators:
        if name not in PROBING_GENERATORS:
            print(f"Unknown generator: {name}")
            continue

        print(f"Generating {name}...", end=" ", flush=True)
        try:
            X, y, meta = PROBING_GENERATORS[name]()
            print(f"OK ({meta.n_samples}x{meta.n_features})")
            results.append((X, y, meta))
        except Exception as e:
            print(f"FAILED: {e}")

    return results


def get_domain_datasets(domain: str) -> List[str]:
    """Get PMLB datasets for a specific domain."""
    return [
        name for name, info in PMLB_DATASETS.items()
        if info.get("domain") == domain
    ]


# =============================================================================
# TabArena Benchmark (51 curated datasets from NeurIPS 2025)
# OpenML Suite ID 457 ("tabarena-v0.1")
# From: "TabArena: A Comprehensive Benchmark for Tabular Learning"
# =============================================================================
TABARENA_DATASETS = {
    "airfoil_self_noise": {"openml_id": 46904, "task": "regression", "domain": "engineering"},
    "Amazon_employee_access": {"openml_id": 46905, "task": "classification", "domain": "business"},
    "anneal": {"openml_id": 46906, "task": "classification", "domain": "materials"},
    "Another-Dataset-on-used-Fiat-500": {"openml_id": 46907, "task": "regression", "domain": "automotive"},
    "APSFailure": {"openml_id": 46908, "task": "classification", "domain": "engineering"},
    "bank-marketing": {"openml_id": 46910, "task": "classification", "domain": "financial"},
    "Bank_Customer_Churn": {"openml_id": 46911, "task": "classification", "domain": "financial"},
    "Bioresponse": {"openml_id": 46912, "task": "classification", "domain": "biology"},
    "blood-transfusion-service-center": {"openml_id": 46913, "task": "classification", "domain": "medical"},
    "churn": {"openml_id": 46915, "task": "classification", "domain": "business"},
    "coil2000_insurance_policies": {"openml_id": 46916, "task": "classification", "domain": "insurance"},
    "concrete_compressive_strength": {"openml_id": 46917, "task": "regression", "domain": "engineering"},
    "credit-g": {"openml_id": 46918, "task": "classification", "domain": "financial"},
    "credit_card_clients_default": {"openml_id": 46919, "task": "classification", "domain": "financial"},
    "customer_satisfaction_in_airline": {"openml_id": 46920, "task": "classification", "domain": "business"},
    "diabetes": {"openml_id": 46921, "task": "classification", "domain": "medical"},
    "Diabetes130US": {"openml_id": 46922, "task": "classification", "domain": "medical"},
    "diamonds": {"openml_id": 46923, "task": "regression", "domain": "retail"},
    "E-CommereShippingData": {"openml_id": 46924, "task": "classification", "domain": "logistics"},
    "Fitness_Club": {"openml_id": 46927, "task": "classification", "domain": "business"},
    "Food_Delivery_Time": {"openml_id": 46928, "task": "regression", "domain": "logistics"},
    "GiveMeSomeCredit": {"openml_id": 46929, "task": "classification", "domain": "financial"},
    "hazelnut-spread-contaminant-detection": {"openml_id": 46930, "task": "classification", "domain": "food"},
    "healthcare_insurance_expenses": {"openml_id": 46931, "task": "regression", "domain": "medical"},
    "heloc": {"openml_id": 46932, "task": "classification", "domain": "financial"},
    "hiva_agnostic": {"openml_id": 46933, "task": "classification", "domain": "biology"},
    "houses": {"openml_id": 46934, "task": "regression", "domain": "real_estate"},
    "HR_Analytics_Job_Change_of_Data_Scientists": {"openml_id": 46935, "task": "classification", "domain": "hr"},
    "in_vehicle_coupon_recommendation": {"openml_id": 46937, "task": "classification", "domain": "marketing"},
    "Is-this-a-good-customer": {"openml_id": 46938, "task": "classification", "domain": "business"},
    "kddcup09_appetency": {"openml_id": 46939, "task": "classification", "domain": "marketing"},
    "Marketing_Campaign": {"openml_id": 46940, "task": "classification", "domain": "marketing"},
    "maternal_health_risk": {"openml_id": 46941, "task": "classification", "domain": "medical"},
    "miami_housing": {"openml_id": 46942, "task": "regression", "domain": "real_estate"},
    "online_shoppers_intention": {"openml_id": 46947, "task": "classification", "domain": "business"},
    "physiochemical_protein": {"openml_id": 46949, "task": "regression", "domain": "biology"},
    "polish_companies_bankruptcy": {"openml_id": 46950, "task": "classification", "domain": "financial"},
    "qsar-biodeg": {"openml_id": 46952, "task": "classification", "domain": "chemistry"},
    "QSAR-TID-11": {"openml_id": 46953, "task": "regression", "domain": "chemistry"},
    "QSAR_fish_toxicity": {"openml_id": 46954, "task": "regression", "domain": "chemistry"},
    "SDSS17": {"openml_id": 46955, "task": "classification", "domain": "astronomy"},
    "seismic-bumps": {"openml_id": 46956, "task": "classification", "domain": "geology"},
    "splice": {"openml_id": 46958, "task": "classification", "domain": "genomics"},
    "students_dropout_and_academic_success": {"openml_id": 46960, "task": "classification", "domain": "education"},
    "superconductivity": {"openml_id": 46961, "task": "regression", "domain": "physics"},
    "taiwanese_bankruptcy_prediction": {"openml_id": 46962, "task": "classification", "domain": "financial"},
    "website_phishing": {"openml_id": 46963, "task": "classification", "domain": "security"},
    "wine_quality": {"openml_id": 46964, "task": "regression", "domain": "chemistry"},
    "NATICUSdroid": {"openml_id": 46969, "task": "classification", "domain": "security"},
    "jm1": {"openml_id": 46979, "task": "classification", "domain": "software"},
    "MIC": {"openml_id": 46980, "task": "classification", "domain": "unknown"},
}


from pathlib import Path as _Path

_TABARENA_CACHE_DIR = _Path(__file__).parent / "cache" / "tabarena"


def _load_tabarena_cached(name: str, info: dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load preprocessed (X, y) from cache, or return None."""
    cache_path = _TABARENA_CACHE_DIR / f"{name}.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"]
    return None


def _save_tabarena_cache(name: str, X: np.ndarray, y: np.ndarray, task: str) -> None:
    """Save preprocessed (X, y) to cache."""
    _TABARENA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(_TABARENA_CACHE_DIR / f"{name}.npz"),
        X=X, y=y, task=np.array(task),
    )


def load_tabarena_dataset(
    name: str,
    max_samples: int = 10000,
) -> Optional[Tuple[np.ndarray, np.ndarray, DatasetMetadata]]:
    """
    Load a dataset from the TabArena benchmark (OpenML suite 457).

    Uses a persistent cache at data/cache/tabarena/ to avoid repeated
    OpenML downloads. The cache stores the full preprocessed dataset;
    subsampling is applied after loading.

    Args:
        name: Dataset name from TABARENA_DATASETS
        max_samples: Maximum samples to return

    Returns:
        (X, y, metadata) tuple or None
    """
    try:
        if name not in TABARENA_DATASETS:
            print(f"Dataset '{name}' not in TABARENA_DATASETS catalog")
            return None

        info = TABARENA_DATASETS[name]
        task = info["task"]

        # Try cache first
        cached = _load_tabarena_cached(name, info)
        if cached is not None:
            X, y = cached
        else:
            from sklearn.datasets import fetch_openml

            dataset_id = info["openml_id"]
            data = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
            X = data.data.copy()
            y = data.target

            # Handle categorical features
            cat_cols = X.select_dtypes(include=["category", "object"]).columns
            for col in cat_cols:
                X[col] = X[col].astype("category").cat.codes

            X = X.values.astype(np.float32)

            # Handle target
            if task == "classification":
                if y.dtype == "object" or y.dtype.name == "category":
                    y = pd.Categorical(y).codes
                y = np.asarray(y).astype(int)
            else:
                y = np.asarray(y).astype(np.float32)

            # Remove NaN targets
            if task == "regression":
                valid = ~np.isnan(y)
                X, y = X[valid], y[valid]

            # Handle NaN in features
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Cache the full preprocessed dataset
            _save_tabarena_cache(name, X, y, task)

        # Subsample
        if len(X) > max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), max_samples, replace=False)
            X, y = X[idx], y[idx]

        n_classes = len(np.unique(y)) if task == "classification" else None

        meta = DatasetMetadata(
            name=f"tabarena_{name}",
            source="tabarena",
            task=task,
            n_samples=len(X),
            n_features=X.shape[1],
            n_classes=n_classes,
            domain=info.get("domain", "unknown"),
            dim_ratio=X.shape[1] / len(X),
            sparsity=float((X == 0).mean()),
        )

        if task == "classification" and n_classes == 2:
            meta.class_balance = float(y.mean())

        return X, y, meta

    except ImportError:
        print("scikit-learn not installed. Run: pip install scikit-learn")
        return None
    except Exception as e:
        print(f"Error loading TabArena dataset {name}: {e}")
        return None


def load_tabarena_suite(
    max_samples: int = 5000,
    max_datasets: Optional[int] = None,
    task_filter: Optional[str] = None,
    domain_filter: Optional[str] = None,
) -> List[Tuple[np.ndarray, np.ndarray, DatasetMetadata]]:
    """
    Load datasets from the TabArena benchmark suite.

    Args:
        max_samples: Maximum samples per dataset
        max_datasets: Maximum number of datasets to load
        task_filter: Only load "classification" or "regression" tasks
        domain_filter: Only load datasets from a specific domain

    Returns:
        List of (X, y, metadata) tuples
    """
    results = []
    loaded = 0

    for name, info in TABARENA_DATASETS.items():
        if max_datasets and loaded >= max_datasets:
            break
        if task_filter and info["task"] != task_filter:
            continue
        if domain_filter and info.get("domain") != domain_filter:
            continue

        print(f"Loading TabArena/{name}...", end=" ", flush=True)
        result = load_tabarena_dataset(name, max_samples=max_samples)
        if result:
            X, y, meta = result
            print(f"OK ({meta.n_samples}x{meta.n_features})")
            results.append(result)
            loaded += 1
        else:
            print("FAILED")

    return results


# =============================================================================
# RelBench (Stanford Relational Learning Benchmark)
# Multi-table relational data flattened to tabular format.
# From: "RelBench: A Benchmark for Deep Learning on Relational Databases"
# =============================================================================

# Datasets with tasks that have reasonable entity-level features when flattened.
# We focus on entity-level classification/regression tasks (not link prediction).
RELBENCH_TASKS = {
    # Dataset -> list of (task_name, task_type) that work well flattened
    "rel-amazon": [
        ("user-churn", "classification"),
        ("item-churn", "classification"),
    ],
    "rel-avito": [
        ("ad-ctr", "regression"),
    ],
    "rel-event": [
        ("user-attendance", "classification"),
        ("user-repeat", "classification"),
    ],
    "rel-f1": [
        ("driver-position", "regression"),
        ("driver-dnf", "classification"),
    ],
    "rel-hm": [
        ("user-churn", "classification"),
        ("item-sales", "regression"),
    ],
    "rel-stack": [
        ("user-engagement", "classification"),
        ("post-votes", "regression"),
    ],
    "rel-trial": [
        ("study-outcome", "classification"),
        ("study-adverse", "classification"),
    ],
}


def _flatten_relbench_tables(
    db,
    entity_table_name: str,
    entity_col: str,
    time_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Flatten a RelBench database into a single feature table by joining
    and aggregating related tables.

    Strategy:
    1. Start from the entity table.
    2. For each related table (via FK), compute aggregate features
       (count, mean, std, min, max of numeric columns).
    3. Join aggregates back to entity table.
    """
    entity_table = db.table_dict[entity_table_name]
    result_df = entity_table.df.copy()

    # For each other table, check if it has a FK pointing to our entity
    for table_name, table in db.table_dict.items():
        if table_name == entity_table_name:
            continue

        # Check for FK relationship to entity table
        fk_col = None
        for fk, pk_table in table.fkey_col_to_pkey_table.items():
            if pk_table == entity_table_name:
                fk_col = fk
                break

        if fk_col is None:
            continue

        # Aggregate numeric columns from related table
        related_df = table.df
        numeric_cols = related_df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude the FK column itself from aggregation
        numeric_cols = [c for c in numeric_cols if c != fk_col]

        if not numeric_cols:
            # At minimum, count related rows
            agg_df = (
                related_df.groupby(fk_col)
                .size()
                .reset_index(name=f"{table_name}_count")
            )
        else:
            agg_funcs = {}
            for col in numeric_cols[:10]:  # Limit to first 10 numeric cols
                agg_funcs[f"{table_name}_{col}_mean"] = (col, "mean")
                agg_funcs[f"{table_name}_{col}_std"] = (col, "std")

            agg_df = (
                related_df.groupby(fk_col)
                .agg(**agg_funcs)
                .reset_index()
            )
            # Also add count
            count_df = (
                related_df.groupby(fk_col)
                .size()
                .reset_index(name=f"{table_name}_count")
            )
            agg_df = agg_df.merge(count_df, on=fk_col, how="left")

        # Join to entity table
        join_col = entity_table.pkey_col if entity_table.pkey_col else entity_col
        if fk_col in result_df.columns or join_col in agg_df.columns:
            merge_on = fk_col if fk_col in result_df.columns else join_col
            if merge_on not in agg_df.columns:
                agg_df = agg_df.rename(columns={fk_col: merge_on})
            result_df = result_df.merge(agg_df, on=merge_on, how="left")

    return result_df


def load_relbench_dataset(
    dataset_name: str,
    task_name: str,
    max_samples: int = 10000,
    split: str = "train",
) -> Optional[Tuple[np.ndarray, np.ndarray, DatasetMetadata]]:
    """
    Load a RelBench dataset, flatten to tabular format, and return (X, y).

    The flattening strategy joins entity tables with aggregated features
    from related tables (count, mean, std of numeric columns per entity).

    Args:
        dataset_name: RelBench dataset (e.g., "rel-amazon")
        task_name: Task within the dataset (e.g., "user-churn")
        max_samples: Maximum samples
        split: "train", "val", or "test"

    Returns:
        (X, y, metadata) tuple or None
    """
    try:
        from relbench.datasets import get_dataset
        from relbench.tasks import get_task

        # Load dataset and task
        dataset = get_dataset(dataset_name, download=True)
        task = get_task(dataset_name, task_name, download=True)
        db = dataset.get_db()

        # Get task split
        task_table = task.get_table(split)
        task_df = task_table.df.copy()

        # Identify target column (last column that isn't an ID or timestamp)
        id_cols = set()
        if task_table.pkey_col:
            id_cols.add(task_table.pkey_col)
        if task_table.time_col:
            id_cols.add(task_table.time_col)
        for fk in task_table.fkey_col_to_pkey_table:
            id_cols.add(fk)

        potential_targets = [c for c in task_df.columns if c not in id_cols]
        if not potential_targets:
            print(f"No target column found in task {task_name}")
            return None

        target_col = potential_targets[-1]  # Target is typically the last column
        y = task_df[target_col].values

        # Determine task type from the catalog or from target values
        task_type = "classification"
        if dataset_name in RELBENCH_TASKS:
            for tn, tt in RELBENCH_TASKS[dataset_name]:
                if tn == task_name:
                    task_type = tt
                    break

        # Build features: start with task table columns, then add flattened DB features
        # Get entity table name from FK relationships
        entity_table_name = None
        entity_col = None
        for fk, pk_table in task_table.fkey_col_to_pkey_table.items():
            entity_table_name = pk_table
            entity_col = fk
            break

        if entity_table_name and entity_table_name in db.table_dict:
            flat_df = _flatten_relbench_tables(db, entity_table_name, entity_col)
            # Merge flattened features into task table
            merge_col = entity_col if entity_col in task_df.columns else None
            if merge_col and merge_col in flat_df.columns:
                task_df = task_df.merge(flat_df, on=merge_col, how="left", suffixes=("", "_flat"))

        # Select numeric columns as features
        feature_cols = []
        for col in task_df.columns:
            if col == target_col or col in id_cols:
                continue
            if task_df[col].dtype in [np.float64, np.float32, np.int64, np.int32, int, float]:
                feature_cols.append(col)
            elif task_df[col].dtype.name == "category" or task_df[col].dtype == object:
                # Encode categorical
                task_df[col] = task_df[col].astype("category").cat.codes.astype(np.float32)
                feature_cols.append(col)

        if not feature_cols:
            print(f"No usable features after flattening {dataset_name}/{task_name}")
            return None

        X = task_df[feature_cols].values.astype(np.float32)

        # Handle target
        if task_type == "classification":
            if y.dtype == object or (hasattr(y, "dtype") and y.dtype.name == "category"):
                y = pd.Categorical(y).codes
            y = np.asarray(y).astype(int)
        else:
            y = np.asarray(y).astype(np.float32)

        # Remove NaN targets
        valid = ~pd.isna(y)
        X, y = X[valid], y[valid]

        # Handle NaN in features
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Subsample
        if len(X) > max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), max_samples, replace=False)
            X, y = X[idx], y[idx]

        n_classes = len(np.unique(y)) if task_type == "classification" else None

        meta = DatasetMetadata(
            name=f"relbench_{dataset_name}_{task_name}",
            source="relbench",
            task=task_type,
            n_samples=len(X),
            n_features=X.shape[1],
            n_classes=n_classes,
            domain=dataset_name.replace("rel-", ""),
            dim_ratio=X.shape[1] / max(len(X), 1),
            sparsity=float((X == 0).mean()),
        )

        if task_type == "classification" and n_classes == 2:
            meta.class_balance = float(y.mean())

        return X, y, meta

    except ImportError:
        print("RelBench not installed. Run: pip install relbench")
        return None
    except Exception as e:
        print(f"Error loading RelBench {dataset_name}/{task_name}: {e}")
        return None


def load_relbench_suite(
    max_samples: int = 5000,
    max_datasets: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray, DatasetMetadata]]:
    """
    Load all curated RelBench tasks from RELBENCH_TASKS.

    Args:
        max_samples: Maximum samples per dataset
        max_datasets: Maximum number of dataset-task pairs to load

    Returns:
        List of (X, y, metadata) tuples
    """
    results = []
    loaded = 0

    for dataset_name, tasks in RELBENCH_TASKS.items():
        for task_name, task_type in tasks:
            if max_datasets and loaded >= max_datasets:
                return results

            print(f"Loading RelBench/{dataset_name}/{task_name}...", end=" ", flush=True)
            result = load_relbench_dataset(
                dataset_name, task_name, max_samples=max_samples
            )
            if result:
                X, y, meta = result
                print(f"OK ({meta.n_samples}x{meta.n_features})")
                results.append(result)
                loaded += 1
            else:
                print("FAILED")

    return results


def list_all_sources() -> Dict[str, int]:
    """List all available data sources and counts."""
    relbench_task_count = sum(len(tasks) for tasks in RELBENCH_TASKS.values())
    return {
        "pmlb": len(PMLB_DATASETS),
        "tabarena": len(TABARENA_DATASETS),
        "relbench": relbench_task_count,
        "probing_synthetic": len(PROBING_GENERATORS),
        "domains": len(set(info.get("domain", "unknown") for info in PMLB_DATASETS.values())),
    }


def get_tabarena_domains() -> Dict[str, List[str]]:
    """Get TabArena datasets grouped by domain."""
    domains: Dict[str, List[str]] = {}
    for name, info in TABARENA_DATASETS.items():
        domain = info.get("domain", "unknown")
        domains.setdefault(domain, []).append(name)
    return domains


if __name__ == "__main__":
    print("=" * 60)
    print("Extended Data Loader - Probing Suite")
    print("=" * 60)

    # Show available sources
    print("\nAvailable sources:")
    for source, count in list_all_sources().items():
        print(f"  {source}: {count}")

    # Show PMLB domains
    print("\nPMLB domains:")
    domains = set(info.get("domain", "unknown") for info in PMLB_DATASETS.values())
    for domain in sorted(domains):
        count = len(get_domain_datasets(domain))
        print(f"  {domain}: {count} datasets")

    # Show TabArena domains
    print("\nTabArena domains:")
    for domain, datasets in sorted(get_tabarena_domains().items()):
        print(f"  {domain}: {len(datasets)} datasets")

    # Show RelBench tasks
    print("\nRelBench tasks:")
    for dataset, tasks in RELBENCH_TASKS.items():
        task_names = [f"{t[0]} ({t[1]})" for t in tasks]
        print(f"  {dataset}: {', '.join(task_names)}")

    # Test probing generators
    print("\n" + "-" * 60)
    print("Testing probing generators:")
    for name in list(PROBING_GENERATORS.keys())[:5]:
        X, y, meta = PROBING_GENERATORS[name]()
        print(f"  {meta.name}: {meta.n_samples}x{meta.n_features}, "
              f"difficulty={meta.difficulty}, balance={meta.class_balance:.2%}")
