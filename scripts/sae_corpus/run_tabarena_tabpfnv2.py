#!/usr/bin/env python3
"""
Run TabArena's RealTabPFN-v2.5 pipeline on 5 validation datasets and save per-row OOF predictions.

Uses TabArena's exact preprocessing and inference pipeline (AutoGluon wrapper around
TabPFN v2.5), fold=0/repeat=0 (TabArena-Lite), producing the ground-truth per-row OOF
predictions we compare directly against our own TabPFN 2.5 inference pipeline.

Note: TabArena's NeurIPS 2025 paper used TabPFNv2 (older). RealTabPFN-v2.5 was added
in November 2025. Since our corpus pipeline uses v2.5, we validate against v2.5.

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

    # RealTabPFN-v2.5: TabArena's wrapper around TabPFN 2.5 — same version as our pipeline.
    # custom_model_dir is a class attribute (not a hyperparameter), so patch it directly
    # to use local checkpoints instead of attempting a HuggingFace download.
    LOCAL_CHECKPOINT_DIR = "/data/models/tabular_fm/tabpfn"
    from tabarena.benchmark.models.ag.tabpfnv2_5.tabpfnv2_5_model import RealTabPFNv25Model
    RealTabPFNv25Model.custom_model_dir = LOCAL_CHECKPOINT_DIR

    config_gen = get_configs_generator_from_name("RealTabPFN-v2.5")
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

    # run_experiments_new returns dicts; per-row predictions are in the cached pkl files
    import pickle
    from pathlib import Path as _Path

    method_key = "TA-RealTabPFN-v2.5_c1_BAG_L1"
    saved = []
    for task_id, ds_name in tid_to_name.items():
        pkl_path = _Path(EXPERIMENT_DIR) / "data" / method_key / str(task_id) / "0_0" / "results.pkl"
        if not pkl_path.exists():
            print(f"  ✗ {ds_name}: pkl not found at {pkl_path}")
            continue
        try:
            result = pickle.load(open(pkl_path, "rb"))
            sa = result["simulation_artifacts"]

            y_val_idx = sa["y_val_idx"].tolist()          # row indices in original dataset
            pred_dict = sa["pred_proba_dict_val"]          # {model_key: (n_val, n_classes) array}
            classes = sa["ordered_class_labels_transformed"]

            # Average across ensemble members (same as AutoGluon's OOF scoring)
            import numpy as np
            preds_arr = np.mean(list(pred_dict.values()), axis=0)

            out = {
                "task_id": task_id,
                "fold": 0,
                "repeat": 0,
                "train_indices": y_val_idx,
                "classes": [str(c) for c in classes],
                "y_pred_proba_val": preds_arr.tolist(),
            }

            out_path = OUTPUT_DIR / f"{ds_name}.json"
            out_path.write_text(json.dumps(out, indent=2))
            print(f"  ✓ {ds_name}: {len(y_val_idx)} rows → {out_path.name}")
            saved.append(ds_name)

        except Exception as e:
            print(f"  ✗ {ds_name}: ERROR — {e}")
            import traceback; traceback.print_exc()

    print(f"\nSaved {len(saved)}/{len(task_ids)} datasets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
