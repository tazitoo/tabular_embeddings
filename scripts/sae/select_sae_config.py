#!/usr/bin/env python3
"""Apply the paper's SAE selection rule to a sweep's Optuna study and report it.

Rule (paper Sec. 3.2 / App. B): among trials with R² >= 0.80, alive fraction >= 0.80
and cross-seed stability >= 0.75, select the least complex, i.e. the smallest
sqrt(d_hidden * L0). If no trial qualifies, the sweep's own validated choice (Optuna
best under the efficiency objective recon * sqrt(d_hidden) * sqrt(L0) / alive) stands,
and this report says so explicitly -- it never promotes a non-qualifying trial.

R² is 1 - test_recon / var, with var the mean per-dimension variance of the model's
test corpus (per-dataset standardised, so close to 1).

Usage:
    python -m scripts.sae.select_sae_config --model tabdpt
    python -m scripts.sae.select_sae_config --model tabdpt --round 10
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.round_paths import DEFAULT_SAE_ROUND, sae_sweep_dir, sae_training_dir

R2_FLOOR, ALIVE_FLOOR, STABILITY_FLOOR = 0.80, 0.80, 0.75


def r2(test_recon: float, var_per_dim: float) -> float:
    return 1.0 - test_recon / var_per_dim


def capacity(t: dict) -> float:
    return math.sqrt(t["expansion"] * t["input_dim"] * t["l0"])


def load_trials(db_path: Path, input_dim: int = 768) -> list[dict]:
    """Completed trials of the study in `db_path` as flat dicts."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=db_path.stem, storage=f"sqlite:///{db_path}")
    out = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE or t.value is None:
            continue
        a, p = t.user_attrs, t.params
        expansion = int(p["expansion"])
        hidden = expansion * input_dim
        out.append(dict(
            trial=t.number, expansion=expansion, topk=int(p["topk"]),
            l0=float(a["l0_sparsity"]), alive=float(a["alive_features"]) / hidden,
            stability=float(a["stability"]), test_recon=float(a["test_reconstruction_loss"]),
            input_dim=input_dim, value=float(t.value), params=dict(p),
        ))
    return out


def select(trials: list[dict], var_per_dim: float):
    """Return (chosen_or_None, qualifiers) under the paper's rule."""
    qualifiers = [t for t in trials
                  if r2(t["test_recon"], var_per_dim) >= R2_FLOOR
                  and t["alive"] >= ALIVE_FLOOR and t["stability"] >= STABILITY_FLOOR]
    if not qualifiers:
        return None, []
    return min(qualifiers, key=capacity), qualifiers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--round", type=int, default=DEFAULT_SAE_ROUND)
    ap.add_argument("--sae-type", default="matryoshka_archetypal")
    args = ap.parse_args()

    sweep = sae_sweep_dir(args.round) / args.model
    db = sweep / f"{args.model}_{args.sae_type}.db"
    test = np.load(sae_training_dir(args.round) / f"{args.model}_taskaware_sae_test.npz")
    emb = test["embeddings"].astype(np.float32)
    var = float(emb.var(axis=0).mean())
    trials = load_trials(db, input_dim=emb.shape[1])
    chosen, qualifiers = select(trials, var)
    best = min(trials, key=lambda t: t["value"])

    print(f"{args.model} round{args.round}: {len(trials)} trials, test var/dim={var:.4f}\n")
    print(f"{'trial':>5} {'exp':>3} {'topk':>4} {'L0':>6} {'alive':>6} {'stab':>6} "
          f"{'recon':>7} {'R2':>6} {'objective':>9} {'capacity':>8}  qualifies")
    for t in sorted(trials, key=lambda t: t["value"]):
        q = t in qualifiers
        print(f"{t['trial']:5d} {t['expansion']:3d} {t['topk']:4d} {t['l0']:6.1f} {t['alive']:6.2f} "
              f"{t['stability']:6.3f} {t['test_recon']:7.4f} {r2(t['test_recon'], var):6.3f} "
              f"{t['value']:9.2f} {capacity(t):8.0f}  {'yes' if q else ''}")

    print(f"\nOptuna best (efficiency objective): trial {best['trial']}")
    if chosen is None:
        print(f"No trial clears R2>={R2_FLOOR}, alive>={ALIVE_FLOOR}, stability>={STABILITY_FLOOR}: "
              f"the sweep's validated choice (trial {best['trial']}) stands.")
    else:
        print(f"Paper rule picks trial {chosen['trial']} "
              f"(least complex of {len(qualifiers)} qualifiers)"
              + ("" if chosen["trial"] == best["trial"] else
                 f" -- DIFFERS from the validated checkpoint; retrain with its params:\n"
                 f"{json.dumps(chosen['params'], indent=2)}"))


if __name__ == "__main__":
    main()
