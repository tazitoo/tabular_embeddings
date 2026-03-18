#!/usr/bin/env python3
"""
Run TabArena's TabPFNv2 pipeline on 5 validation datasets and save per-row OOF predictions.

Uses TabArena's exact preprocessing and inference pipeline (AutoGluon wrapper around
TabPFNv2), fold=0/repeat=0 (TabArena-Lite), producing the ground-truth per-row OOF
predictions we compare against in 01_validate_inference.py.

Output:
    output/sae_training_round9/tabarena_oof_predictions/{dataset_name}.json
    {
        "task_id": int,
        "fold": 0,
        "repeat": 0,
        "train_indices": [int, ...],
        "y_pred_proba_val": [[float, ...], ...],   # per-row OOF probabilities
        "classes": [int, ...]
    }

Usage (run on GPU worker):
    python scripts/sae_corpus/run_tabarena_tabpfnv2.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._project_root import PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_training_round9" / "tabarena_oof_predictions"
EXPERIMENT_DIR = "/tmp/tabarena_tabpfnv2_runs"

# 5 small classification datasets — task IDs from OpenML suite "tabarena-v0.1"
VALIDATION_TASKS = {
    "blood-transfusion-service-center": 363621,
    "diabetes": 363629,
    "website_phishing": 363707,
    "anneal": 363614,
    "credit-g": 363626,
}


def main():
    from tabarena.benchmark.experiment import run_experiments_new
    from tabarena.models.utils import get_configs_generator_from_name

    config_gen = get_configs_generator_from_name("TabPFNv2")
    # generate_all_bag_experiments returns the same refit=True config TabArena uses
    model_experiments = config_gen.generate_all_bag_experiments(
        num_random_configs=0,
        fold_fitting_strategy="sequential_local",
    )

    task_ids = list(VALIDATION_TASKS.values())
    print(f"Running TabArena TabPFNv2 on {len(task_ids)} tasks: {list(VALIDATION_TASKS.keys())}")
    print(f"  repetitions_mode=TabArena-Lite (fold=0, repeat=0 only)")

    results_lst = run_experiments_new(
        output_dir=EXPERIMENT_DIR,
        model_experiments=model_experiments,
        tasks=task_ids,
        repetitions_mode="TabArena-Lite",
    )

    print(f"\nGot {len(results_lst)} results, saving per-row OOF predictions...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # task_id -> dataset name reverse map
    tid_to_name = {v: k for k, v in VALIDATION_TASKS.items()}

    saved = []
    for result in results_lst:
        task_id = result.task_id
        ds_name = tid_to_name.get(task_id, str(task_id))

        try:
            # y_pred_proba_val_as_pd: DataFrame with row indices as index, class cols
            val_preds = result.y_pred_proba_val_as_pd
            if val_preds is None:
                print(f"  WARNING: {ds_name} — no val predictions")
                continue

            out = {
                "task_id": task_id,
                "fold": result.fold,
                "repeat": result.repeat,
                "train_indices": val_preds.index.tolist(),
                "classes": [str(c) for c in val_preds.columns.tolist()],
                "y_pred_proba_val": val_preds.values.tolist(),
            }

            out_path = OUTPUT_DIR / f"{ds_name}.json"
            out_path.write_text(json.dumps(out, indent=2))
            print(f"  ✓ {ds_name}: {len(val_preds)} rows → {out_path.name}")
            saved.append(ds_name)

        except Exception as e:
            print(f"  ✗ {ds_name}: ERROR — {e}")
            import traceback; traceback.print_exc()

    print(f"\nSaved {len(saved)}/{len(task_ids)} datasets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
