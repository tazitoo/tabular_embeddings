#!/usr/bin/env python3
"""
Run TabArena's RealTabPFN-v2.5 pipeline on 5 validation datasets and save test predictions.

Uses TabArena's exact preprocessing (AutoMLPipelineFeatureGenerator + LabelCleaner)
with a straightforward train/predict split: train on the outer train fold, predict on
the outer test fold. Saves predicted probabilities and ground-truth labels so
01_validate_inference.py can compare log-loss against our pipeline.

Output:
    output/sae_training_round9/tabarena_oof_predictions/{dataset_name}.json
    {
        "task_id": int,
        "fold": 0,
        "repeat": 0,
        "test_indices": [int, ...],
        "classes": [str, ...],
        "y_pred_proba_test": [[float, ...], ...],   # (n_test, n_classes)
        "y_test": [int, ...]
    }

Usage (run on GPU worker):
    python scripts/sae_corpus/run_tabarena_tabpfnv2.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._project_root import PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_training_round9" / "tabarena_oof_predictions"
LOCAL_CHECKPOINT_DIR = "/data/models/tabular_fm/tabpfn"

# 5 small classification datasets — task IDs from OpenML suite "tabarena-v0.1"
VALIDATION_TASKS = {
    "blood-transfusion-service-center": 363621,
    "diabetes": 363629,
    "website_phishing": 363707,
    "anneal": 363614,
    "credit-g": 363626,
}


def main():
    import json as _json

    from autogluon.core.data import LabelCleaner
    from autogluon.features.generators import AutoMLPipelineFeatureGenerator
    from tabarena.benchmark.models.ag.tabpfnv2_5.tabpfnv2_5_model import RealTabPFNv25Model
    from tabarena.benchmark.task.openml import OpenMLTaskWrapper

    # Patch local checkpoint dir to avoid HuggingFace download
    RealTabPFNv25Model.custom_model_dir = LOCAL_CHECKPOINT_DIR

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits_path = PROJECT_ROOT / "output" / "sae_training_round9" / "tabarena_splits.json"
    splits = _json.loads(splits_path.read_text())

    saved = []
    for ds_name, task_id in VALIDATION_TASKS.items():
        print(f"\n[{ds_name}] task_id={task_id}")
        split_info = splits[ds_name]

        try:
            # Load full dataset via OpenML task (TabArena's data path)
            task = OpenMLTaskWrapper.from_task_id(task_id)
            train_idx = np.array(split_info["train_indices"])
            test_idx = np.array(split_info["test_indices"])

            # get_train_test_split accepts explicit indices, returns (X_train, y_train, X_test, y_test)
            X_train, y_train, X_test, y_test = task.get_train_test_split(
                train_indices=train_idx,
                test_indices=test_idx,
            )

            problem_type = "binary" if y_train.nunique() == 2 else "multiclass"
            print(f"  problem_type={problem_type}, n_train={len(X_train)}, n_test={len(X_test)}")

            # TabArena's preprocessing pipeline (exact same as benchmark runs)
            feature_generator = AutoMLPipelineFeatureGenerator()
            label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y_train)

            X_train_proc = feature_generator.fit_transform(X_train)
            y_train_proc = label_cleaner.transform(y_train)
            X_test_proc = feature_generator.transform(X_test)
            y_test_proc = label_cleaner.transform(y_test)

            # Fit TabPFN v2.5 — no CV, no bagging, straight train → predict
            model = RealTabPFNv25Model(problem_type=problem_type)
            model.fit(X=X_train_proc, y=y_train_proc)
            pred_proba = model.predict_proba(X_test_proc)  # (n_test, n_classes)

            # Inverse-transform labels back to original class names
            classes = label_cleaner.ordered_class_labels

            out = {
                "task_id": task_id,
                "fold": 0,
                "repeat": 0,
                "test_indices": test_idx.tolist(),
                "classes": [str(c) for c in classes],
                "y_pred_proba_test": pred_proba.tolist(),
                "y_test": y_test_proc.tolist(),
            }

            out_path = OUTPUT_DIR / f"{ds_name}.json"
            out_path.write_text(json.dumps(out, indent=2))
            print(f"  ✓ saved {len(test_idx)} test predictions → {out_path.name}")
            saved.append(ds_name)

        except Exception as e:
            import traceback
            print(f"  ✗ ERROR — {e}")
            traceback.print_exc()

    print(f"\nSaved {len(saved)}/{len(VALIDATION_TASKS)} datasets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
