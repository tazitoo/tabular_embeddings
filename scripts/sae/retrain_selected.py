#!/usr/bin/env python3
"""Retrain an SAE with chosen HPs and save the validated-artifact set.

Trains at the validation seed, runs the stability seeds, and saves the validated
model, seed models and the geometry-matched random baseline -- the same artifact set
the sweep's own validate_and_save writes.

HPs come from one of:
  * SELECTED_PARAMS below (the round-10 floor picks), the default per model;
  * a sweep study trial (--trial N, --round R), optionally with the structural
    params overridden (--expansion, --topk) to train one-off candidates.

Candidates go to a tagged directory beside the sweep's checkpoint dir
(--tag NAME -> sae_tabarena_sweep_round{R}/{model}_candidates/NAME) so the sweep's
validated checkpoint is never overwritten; a summary.json with R2/alive/L0/stability
is written next to them for comparison.

Usage:
    python -m scripts.sae.retrain_selected --model tabpfn --device cuda
    python -m scripts.sae.retrain_selected --model tabdpt --trial 13 --tag t13_4x_k128
    python -m scripts.sae.retrain_selected --model tabdpt --trial 13 --expansion 2 --topk 64 --tag t13_2x_k64
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._project_root import PROJECT_ROOT
from scripts.sae.sae_tabarena_sweep import (
    _load_prebuilt_embeddings, run_sae_trial, save_sae_model,
    compute_stability,
)
from scripts.round_paths import sae_sweep_dir
from analysis.sparse_autoencoder import create_random_baseline

STRUCTURAL = ("expansion", "topk")
HP_KEYS = ("sparsity_penalty", "learning_rate", "archetypal_temp", "archetypal_n",
           "archetypal_relaxation")


def params_from_study(db_path: Path, trial: int) -> dict:
    """The HPs of one completed trial of an Optuna study, keyed like SELECTED_PARAMS."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=db_path.stem, storage=f"sqlite:///{db_path}")
    t = next(t for t in study.trials if t.number == trial)
    p = t.params
    return {"trial": trial, "expansion": int(p["expansion"]), "topk": int(p["topk"]),
            **{k: p[k] for k in HP_KEYS}}


def with_overrides(params: dict, expansion: int | None = None, topk: int | None = None) -> dict:
    """Copy of `params` with the structural params replaced where given."""
    out = dict(params)
    if expansion is not None:
        out["expansion"] = int(expansion)
    if topk is not None:
        out["topk"] = int(topk)
    return out


def candidate_dir(model: str, tag: str) -> Path:
    """Where a one-off candidate lands: beside, never inside, the sweep's model dir."""
    return sae_sweep_dir() / f"{model}_candidates" / tag

# Floor-selected trial HPs (from sweep analysis)
SELECTED_PARAMS = {
    "tabpfn": {
        "trial": 14,
        "expansion": 4,
        "sparsity_penalty": 0.00010360852294206857,
        "learning_rate": 0.0002516986097067781,
        "topk": 256,
        "archetypal_temp": 0.20129888873505528,
        "archetypal_n": 1000,
        "archetypal_relaxation": 1.4114268837203008,
    },
    "tabula8b": {
        "trial": 15,
        "expansion": 1,
        "sparsity_penalty": 0.0035127674530828464,
        "learning_rate": 0.00010995890162062483,
        "topk": 64,
        "archetypal_temp": 0.11473925496243119,
        "archetypal_n": 512,
        "archetypal_relaxation": 0.6950379195220894,
    },
    "tabicl": {
        "trial": 29,
        "expansion": 4,
        "sparsity_penalty": 0.0007423816641646234,
        "learning_rate": 8.423668005698826e-05,
        "topk": 256,
        "archetypal_temp": 0.2580488578046695,
        "archetypal_n": 512,
        "archetypal_relaxation": 0.7220216414656098,
    },
    "tabicl_v2": {
        "trial": 29,
        "expansion": 4,
        "sparsity_penalty": 0.00032521618213540205,
        "learning_rate": 8.423668005698826e-05,
        "topk": 256,
        "archetypal_temp": 0.06643408066084037,
        "archetypal_n": 512,
        "archetypal_relaxation": 0.34535969625859775,
    },
    "tabdpt": {
        "trial": 13,
        "expansion": 4,
        "sparsity_penalty": 0.009759858149775247,
        "learning_rate": 0.0003404220148189038,
        "topk": 128,
        "archetypal_temp": 0.22039732738806497,
        "archetypal_n": 1000,
        "archetypal_relaxation": 0.026728089405926442,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trial", type=int, default=None,
                        help="take HPs from this trial of the model's sweep study")
    parser.add_argument("--round", type=int, default=None, help="study round (default: current)")
    parser.add_argument("--expansion", type=int, default=None, help="override expansion")
    parser.add_argument("--topk", type=int, default=None, help="override top-k")
    parser.add_argument("--tag", default=None,
                        help="write to {model}_candidates/TAG instead of the model's sweep dir")
    args = parser.parse_args()

    model_name = args.model
    if args.trial is not None:
        db = sae_sweep_dir(args.round) / model_name / f"{model_name}_matryoshka_archetypal.db"
        params = with_overrides(params_from_study(db, args.trial), args.expansion, args.topk)
    else:
        if model_name not in SELECTED_PARAMS:
            parser.error(f"no SELECTED_PARAMS for {model_name}; pass --trial")
        params = with_overrides(SELECTED_PARAMS[model_name], args.expansion, args.topk)
    output_dir = candidate_dir(model_name, args.tag) if args.tag else sae_sweep_dir() / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    prebuilt = _load_prebuilt_embeddings(model_name)
    train_emb, test_emb, datasets, _ = prebuilt
    print(f"Train: {train_emb.shape}, Test: {test_emb.shape}")

    sae_type = "matryoshka_archetypal"

    # Train with validation seed (different from sweep seed 42)
    print(f"\nTraining {model_name} with floor-selected HPs (trial {params['trial']})...")
    print(f"  Params: {json.dumps({k: v for k, v in params.items() if k != 'trial'}, indent=4)}")

    validation_seed = 12345
    metrics, model, config, seed_models, seed_ids = run_sae_trial(
        train_emb,
        sae_type=sae_type,
        expansion=params["expansion"],
        sparsity_penalty=params["sparsity_penalty"],
        learning_rate=params["learning_rate"],
        topk=params["topk"],
        archetypal_n_archetypes=params["archetypal_n"],
        archetypal_temp=params["archetypal_temp"],
        archetypal_relaxation=params["archetypal_relaxation"],
        n_epochs=100,
        measure_stability=True,
        return_model=True,
        seed=validation_seed,
        device=args.device,
    )

    # Test set evaluation
    test_tensor = torch.tensor(test_emb, dtype=torch.float32, device=args.device)
    model.eval()
    with torch.no_grad():
        recon, _ = model(test_tensor)
        test_recon_loss = torch.nn.functional.mse_loss(recon, test_tensor).item()

    hidden_dim = params["expansion"] * train_emb.shape[1]
    alive_frac = metrics["alive_features"] / hidden_dim
    l0 = metrics["l0_sparsity"]
    obj = test_recon_loss * np.sqrt(hidden_dim) * np.sqrt(l0) / alive_frac

    var_per_dim = float(test_emb.var(axis=0).mean())
    r2 = 1.0 - test_recon_loss / var_per_dim
    print(f"\n  Results:")
    print(f"    train_recon:  {metrics['reconstruction_loss']:.6f}")
    print(f"    test_recon:   {test_recon_loss:.6f}  (R2 {r2:.4f})")
    print(f"    alive:        {metrics['alive_features']}/{hidden_dim} ({alive_frac*100:.1f}%)")
    print(f"    L0:           {l0:.1f}")
    print(f"    stability:    {metrics['stability']:.4f}")
    print(f"    objective:    {obj:.4f}")
    with open(output_dir / "summary.json", "w") as fh:
        json.dump({"model": model_name, "params": params, "hidden_dim": int(hidden_dim),
                   "test_recon": float(test_recon_loss), "r2": float(r2),
                   "alive": float(alive_frac), "l0": float(l0),
                   "stability": float(metrics["stability"]), "objective": float(obj)},
                  fh, indent=2)

    # Save validated model
    best_params = {k: v for k, v in params.items() if k != "trial"}
    model_path = output_dir / f"sae_{sae_type}_validated.pt"
    save_sae_model(model, config, metrics, best_params, model_path)

    # Save random baseline
    baseline = create_random_baseline(config)
    baseline_path = output_dir / f"sae_{sae_type}_random_baseline.pt"
    save_sae_model(baseline, baseline.config, {"random_baseline": True}, best_params, baseline_path)

    # Save stability seed models
    for seed_model, seed_id in zip(seed_models, seed_ids):
        seed_path = output_dir / f"sae_{sae_type}_seed{seed_id}.pt"
        save_sae_model(seed_model, config, metrics, best_params, seed_path)

    print(f"\nSaved to {output_dir}/")
    print(f"  sae_{sae_type}_validated.pt")
    print(f"  sae_{sae_type}_random_baseline.pt")
    for seed_id in seed_ids:
        print(f"  sae_{sae_type}_seed{seed_id}.pt")


if __name__ == "__main__":
    main()
