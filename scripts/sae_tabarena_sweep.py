#!/usr/bin/env python3
"""
SAE Architecture Comparison on TabArena.

1. Split TabArena datasets into train/test (70/30)
2. Pool embeddings from train datasets
3. Run Optuna HP sweep for each SAE type
4. Evaluate best configs on test datasets

Usage:
    # Define splits and check data
    python scripts/sae_tabarena_sweep.py --setup

    # Run HP sweep for one model
    python scripts/sae_tabarena_sweep.py --model tabpfn --n-trials 30

    # Evaluate on test set
    python scripts/sae_tabarena_sweep.py --model tabpfn --evaluate
"""

import argparse
import json
import hashlib
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.sparse_autoencoder import (
    SAEConfig,
    train_sae,
    measure_dictionary_richness,
    compare_dictionaries,
)

# Fixed random seed for reproducible splits
SPLIT_SEED = 42

# TabArena train/test split (70/30, stratified by size)
# This is deterministic based on dataset names
def get_tabarena_splits() -> Tuple[List[str], List[str]]:
    """
    Get train/test split of TabArena datasets.

    Split is deterministic (based on hash of dataset name).
    Roughly 70% train, 30% test.
    """
    all_datasets = [
        "airfoil_self_noise", "Amazon_employee_access", "anneal",
        "Another-Dataset-on-used-Fiat-500", "APSFailure", "Bank_Customer_Churn",
        "bank-marketing", "Bioresponse", "blood-transfusion-service-center",
        "churn", "coil2000_insurance_policies", "concrete_compressive_strength",
        "credit_card_clients_default", "credit-g", "customer_satisfaction_in_airline",
        "diabetes", "Diabetes130US", "diamonds", "E-CommereShippingData",
        "Fitness_Club", "GiveMeSomeCredit", "healthcare_insurance_expenses",
        "in_vehicle_coupon_recommendation", "Is-this-a-good-customer",
        "jasmine", "KDDCup09_appetency", "kdd_ipums_la_97-small",
        "maternal_health_risk", "MIC", "MiniBooNE", "NATICUSdroid",
        "national_longitudinal_survey_binary", "ozone-level-8hr",
        "page_blocks_binary", "particulate_matter_ukair_2017",
        "phoneme", "PortoSeguro", "profb", "road-safety-drivers-sex",
        "Satellite", "sick", "taiwanese_bankruptcy_prediction",
        "tamilnadu_electricity", "telco-customer-churn", "us_crime",
        "vehicle", "wilt", "wine_quality",
    ]

    train_datasets = []
    test_datasets = []

    for ds in all_datasets:
        # Deterministic split based on hash
        h = int(hashlib.md5(ds.encode()).hexdigest(), 16)
        if h % 10 < 7:  # 70% train
            train_datasets.append(ds)
        else:
            test_datasets.append(ds)

    return train_datasets, test_datasets


def load_embeddings(model_name: str, dataset_name: str) -> Optional[np.ndarray]:
    """Load cached embeddings for a model/dataset."""
    path = PROJECT_ROOT / f"output/embeddings/tabarena/{model_name}/tabarena_{dataset_name}.npz"
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return data['embeddings'].astype(np.float32)
    return None


def pool_embeddings(
    model_name: str,
    datasets: List[str],
    max_per_dataset: int = 500,
    normalize: bool = True,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Pool embeddings from multiple datasets.

    Args:
        model_name: Model to load embeddings for
        datasets: List of dataset names
        max_per_dataset: Max samples per dataset (for balance)
        normalize: Whether to normalize embeddings

    Returns:
        (pooled_embeddings, dataset_counts)
    """
    all_embeddings = []
    dataset_counts = {}

    for ds in datasets:
        emb = load_embeddings(model_name, ds)
        if emb is None:
            print(f"  Warning: No embeddings for {ds}")
            continue

        # Subsample if too large
        if len(emb) > max_per_dataset:
            np.random.seed(SPLIT_SEED)
            idx = np.random.choice(len(emb), max_per_dataset, replace=False)
            emb = emb[idx]

        all_embeddings.append(emb)
        dataset_counts[ds] = len(emb)

    if not all_embeddings:
        raise ValueError(f"No embeddings found for {model_name}")

    pooled = np.concatenate(all_embeddings, axis=0)

    if normalize:
        # Per-dimension normalization
        std = pooled.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        pooled = pooled / std

    return pooled, dataset_counts


def compute_stability(
    embeddings: np.ndarray,
    config: SAEConfig,
    n_runs: int = 2,
    return_per_scale: bool = False,
) -> float:
    """
    Train SAE twice and measure dictionary stability.

    For Matryoshka SAEs, also measures stability at each nested scale.

    Args:
        embeddings: Training data
        config: SAE configuration
        n_runs: Number of training runs
        return_per_scale: If True, return dict with per-scale stability (Matryoshka only)

    Returns:
        Overall stability score (or dict with per-scale if return_per_scale=True)
    """
    dicts = []
    for seed in [123, 456][:n_runs]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model, result = train_sae(embeddings, config, verbose=False)
        dicts.append(result.dictionary)

    comp = compare_dictionaries(dicts[0], dicts[1])
    overall_stability = comp['mean_best_match_a']

    # For Matryoshka, compute per-scale stability
    if config.sparsity_type == "matryoshka" and return_per_scale:
        mat_dims = config.matryoshka_dims or [32, 64, 128, 256]
        mat_dims = [d for d in mat_dims if d <= config.hidden_dim]

        per_scale = {}
        prev_dim = 0
        for dim in mat_dims:
            # Compare features in this scale only
            scale_dict1 = dicts[0][prev_dim:dim]
            scale_dict2 = dicts[1][prev_dim:dim]
            if len(scale_dict1) > 0:
                scale_comp = compare_dictionaries(scale_dict1, scale_dict2)
                per_scale[f"scale_{prev_dim}_{dim}"] = scale_comp['mean_best_match_a']
            prev_dim = dim

        # Remaining features
        if prev_dim < config.hidden_dim:
            scale_dict1 = dicts[0][prev_dim:]
            scale_dict2 = dicts[1][prev_dim:]
            scale_comp = compare_dictionaries(scale_dict1, scale_dict2)
            per_scale[f"scale_{prev_dim}_{config.hidden_dim}"] = scale_comp['mean_best_match_a']

        return {
            "overall": overall_stability,
            "per_scale": per_scale,
        }

    return overall_stability


def run_sae_trial(
    embeddings: np.ndarray,
    sae_type: str,
    expansion: int = 4,
    sparsity_penalty: float = 1e-3,
    learning_rate: float = 1e-3,
    topk: int = 32,
    archetypal_n_archetypes: int = 500,
    archetypal_temp: float = 0.1,
    archetypal_relaxation: float = 0.0,
    n_epochs: int = 100,
    measure_stability: bool = True,
) -> Dict:
    """Run a single SAE training trial and return metrics."""
    input_dim = embeddings.shape[1]
    hidden_dim = input_dim * expansion

    config = SAEConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        sparsity_penalty=sparsity_penalty,
        sparsity_type=sae_type,
        topk=topk,
        archetypal_n_archetypes=archetypal_n_archetypes,
        archetypal_simplex_temp=archetypal_temp,
        archetypal_relaxation=archetypal_relaxation,
        archetypal_use_centroids=True,
        use_aux_loss=(sae_type not in ["archetypal", "matryoshka_archetypal"]),
        n_epochs=n_epochs,
        batch_size=128,
        learning_rate=learning_rate,
    )

    # Train and evaluate
    torch.manual_seed(42)
    model, result = train_sae(embeddings, config, verbose=False)
    richness = measure_dictionary_richness(result, input_features=embeddings, sae_model=model)

    metrics = {
        "sae_type": sae_type,
        "expansion": expansion,
        "sparsity_penalty": sparsity_penalty,
        "learning_rate": learning_rate,
        "r2": richness.get("explained_variance", 0.0),
        "l0_sparsity": richness["l0_sparsity"],
        "richness_score": richness["richness_score"],
        "reconstruction_loss": result.reconstruction_loss,
        "alive_features": result.alive_features,
    }

    if measure_stability:
        stability = compute_stability(embeddings, config, n_runs=2)
        metrics["stability"] = stability

    return metrics


def create_optuna_objective(embeddings: np.ndarray, sae_type: str):
    """Create Optuna objective for a specific SAE type."""
    import optuna

    def objective(trial: optuna.Trial) -> float:
        # Common hyperparameters
        expansion = trial.suggest_categorical("expansion", [4, 8])
        sparsity_penalty = trial.suggest_float("sparsity_penalty", 1e-4, 1e-2, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        # Type-specific parameters
        if sae_type in ("topk", "matryoshka_archetypal"):
            topk = trial.suggest_categorical("topk", [16, 32, 64, 128])
        else:
            topk = 32

        if sae_type in ("archetypal", "matryoshka_archetypal"):
            archetypal_temp = trial.suggest_float("archetypal_temp", 0.05, 0.5, log=True)
            archetypal_n = trial.suggest_categorical("archetypal_n", [256, 512, 1000])
            archetypal_relax = trial.suggest_float("archetypal_relaxation", 0.0, 2.0)
        else:
            archetypal_temp = 0.1
            archetypal_n = 500
            archetypal_relax = 0.0

        try:
            metrics = run_sae_trial(
                embeddings,
                sae_type=sae_type,
                expansion=expansion,
                sparsity_penalty=sparsity_penalty,
                learning_rate=learning_rate,
                topk=topk,
                archetypal_n_archetypes=archetypal_n,
                archetypal_temp=archetypal_temp,
                archetypal_relaxation=archetypal_relax,
                n_epochs=100,
                measure_stability=True,
            )

            # Store metrics
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    trial.set_user_attr(key, val)

            # Objective: balance R² and stability
            # Weight stability higher since that's the key differentiator
            r2 = metrics["r2"]
            stability = metrics["stability"]

            # Composite score: 0.4 * R² + 0.6 * stability
            # (stability is more important for interpretability)
            score = 0.4 * max(0, r2) + 0.6 * stability

            return score

        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0

    return objective


def run_sweep(
    model_name: str,
    n_trials: int = 30,
    output_dir: Optional[Path] = None,
):
    """Run HP sweep for all SAE types on train datasets."""
    import optuna

    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / "sae_tabarena_sweep" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get train datasets
    train_datasets, test_datasets = get_tabarena_splits()
    print(f"Train datasets: {len(train_datasets)}")
    print(f"Test datasets: {len(test_datasets)}")

    # Pool train embeddings
    print(f"\nPooling {model_name} embeddings from train datasets...")
    embeddings, counts = pool_embeddings(model_name, train_datasets, max_per_dataset=200)
    print(f"  Total samples: {len(embeddings)}")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print(f"  Datasets loaded: {len(counts)}")

    # Save split info
    split_info = {
        "train_datasets": train_datasets,
        "test_datasets": test_datasets,
        "train_counts": counts,
        "total_train_samples": len(embeddings),
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Run sweep for each SAE type
    sae_types = ["l1", "topk", "matryoshka", "archetypal", "matryoshka_archetypal"]
    best_configs = {}

    for sae_type in sae_types:
        print(f"\n{'='*60}")
        print(f"HP Sweep: {sae_type.upper()}")
        print('='*60)

        study_name = f"{model_name}_{sae_type}"
        storage = f"sqlite:///{output_dir}/{study_name}.db"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        objective = create_optuna_objective(embeddings, sae_type)

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
        )

        print(f"\nBest {sae_type}:")
        print(f"  Score: {study.best_value:.4f}")
        print(f"  Params: {study.best_params}")
        if study.best_trial.user_attrs:
            print(f"  R²: {study.best_trial.user_attrs.get('r2', 'N/A'):.4f}")
            print(f"  Stability: {study.best_trial.user_attrs.get('stability', 'N/A'):.4f}")

        best_configs[sae_type] = {
            "params": study.best_params,
            "score": study.best_value,
            "metrics": dict(study.best_trial.user_attrs),
        }

    # Save best configs
    with open(output_dir / "best_configs.json", "w") as f:
        json.dump(best_configs, f, indent=2)

    # Summary comparison
    print(f"\n{'='*60}")
    print("ARCHITECTURE COMPARISON")
    print('='*60)
    print(f"{'Type':<12} {'Score':>8} {'R²':>8} {'Stability':>10} {'L0':>8}")
    print('-'*60)

    for sae_type, config in best_configs.items():
        m = config["metrics"]
        print(f"{sae_type:<12} {config['score']:>8.4f} {m.get('r2', 0):>8.4f} "
              f"{m.get('stability', 0):>10.4f} {m.get('l0_sparsity', 0):>8.1f}")

    return best_configs


def evaluate_on_test(
    model_name: str,
    output_dir: Optional[Path] = None,
):
    """Evaluate best configs on test datasets."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / "sae_tabarena_sweep" / model_name

    # Load best configs
    config_path = output_dir / "best_configs.json"
    if not config_path.exists():
        raise ValueError(f"No best configs found. Run --sweep first.")

    with open(config_path) as f:
        best_configs = json.load(f)

    # Get test datasets
    _, test_datasets = get_tabarena_splits()

    print(f"Evaluating on {len(test_datasets)} test datasets...")

    # Pool test embeddings
    embeddings, counts = pool_embeddings(model_name, test_datasets, max_per_dataset=200)
    print(f"  Total test samples: {len(embeddings)}")

    # Evaluate each SAE type
    results = {}

    for sae_type, config in best_configs.items():
        print(f"\nEvaluating {sae_type}...")

        params = config["params"]

        metrics = run_sae_trial(
            embeddings,
            sae_type=sae_type,
            expansion=params.get("expansion", 4),
            sparsity_penalty=params.get("sparsity_penalty", 1e-3),
            learning_rate=params.get("learning_rate", 1e-3),
            topk=params.get("topk", 32),
            archetypal_n_archetypes=params.get("archetypal_n", 500),
            archetypal_temp=params.get("archetypal_temp", 0.1),
            archetypal_relaxation=params.get("archetypal_relaxation", 0.0),
            n_epochs=100,
            measure_stability=True,
        )

        results[sae_type] = metrics
        print(f"  R²: {metrics['r2']:.4f}, Stability: {metrics['stability']:.4f}")

    # Save test results
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("TEST SET RESULTS")
    print('='*60)
    print(f"{'Type':<12} {'R²':>8} {'Stability':>10} {'L0':>8}")
    print('-'*60)

    for sae_type, m in results.items():
        print(f"{sae_type:<12} {m['r2']:>8.4f} {m['stability']:>10.4f} {m['l0_sparsity']:>8.1f}")

    return results


def setup_check(model_name: str = "tabpfn"):
    """Check data availability and show split info."""
    train_datasets, test_datasets = get_tabarena_splits()

    print("TabArena Train/Test Split")
    print("="*60)
    print(f"Train datasets ({len(train_datasets)}):")
    for ds in train_datasets:
        emb = load_embeddings(model_name, ds)
        status = f"{len(emb)} samples" if emb is not None else "MISSING"
        print(f"  {ds}: {status}")

    print(f"\nTest datasets ({len(test_datasets)}):")
    for ds in test_datasets:
        emb = load_embeddings(model_name, ds)
        status = f"{len(emb)} samples" if emb is not None else "MISSING"
        print(f"  {ds}: {status}")


def main():
    parser = argparse.ArgumentParser(description="SAE TabArena Sweep")
    parser.add_argument("--setup", action="store_true", help="Check data and show splits")
    parser.add_argument("--model", type=str, default="tabpfn", help="Model name")
    parser.add_argument("--n-trials", type=int, default=30, help="Trials per SAE type")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on test set")
    args = parser.parse_args()

    if args.setup:
        setup_check(args.model)
    elif args.evaluate:
        evaluate_on_test(args.model)
    else:
        run_sweep(args.model, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
