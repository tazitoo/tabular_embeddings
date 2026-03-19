#!/usr/bin/env python3
"""Generate LaTeX table of small datasets (< 700 holdout rows) for appendix.

These datasets contribute all holdout rows to the SAE corpus rather than the
standard 700-row cap (500 train + 200 test).  The table documents which
datasets are affected and their actual sample counts.

Output:
    scripts/tables/small_datasets_table.tex
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._project_root import PROJECT_ROOT

SPLITS_PATH = PROJECT_ROOT / "output" / "sae_training_round9" / "tabarena_splits.json"
SAMPLE_CAP = 700
TRAIN_FRAC = 500 / 700  # ≈ 71.4%

splits = json.loads(SPLITS_PATH.read_text())

# Collect small datasets
rows = []
for name, info in sorted(splits.items()):
    n_test = len(info["test_indices"])
    if n_test < SAMPLE_CAP:
        n_train_split = round(n_test * TRAIN_FRAC)
        n_test_split = n_test - n_train_split
        rows.append((name, info["task_type"], n_test, n_train_split, n_test_split))

rows.sort(key=lambda r: r[2])  # ascending by holdout size

# Build LaTeX
lines = []
lines.append(r"\begin{table}[h]")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(r"\caption{Datasets below the 700-row holdout threshold. These datasets")
lines.append(r"contribute all available holdout rows to the SAE corpus rather than the")
lines.append(r"standard 500/200 train/test sample. The train/test split maintains the")
lines.append(r"same 71/29 ratio used for larger datasets.}")
lines.append(r"\label{tab:small_datasets}")
lines.append(r"\begin{tabular}{lcrrrr}")
lines.append(r"\toprule")
lines.append(r"Dataset & Task & Holdout & Train & Test & \% of Cap \\")
lines.append(r"\midrule")

for name, task, n_holdout, n_train, n_test_split in rows:
    pct = 100 * n_holdout / SAMPLE_CAP
    display_name = name.replace("_", r"\_")
    task_short = "C" if task == "classification" else "R"
    lines.append(
        f"{display_name} & {task_short} & {n_holdout} & {n_train} & {n_test_split} & {pct:.0f}\\% \\\\"
    )

lines.append(r"\midrule")
total_holdout = sum(r[2] for r in rows)
total_train = sum(r[3] for r in rows)
total_test = sum(r[4] for r in rows)
lines.append(
    rf"\textbf{{Total ({len(rows)} datasets)}} & & \textbf{{{total_holdout}}} & "
    rf"\textbf{{{total_train}}} & \textbf{{{total_test}}} & \\"
)
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

tex = "\n".join(lines) + "\n"

out_path = Path(__file__).parent / "small_datasets_table.tex"
out_path.write_text(tex)
print(tex)
print(f"\n→ {out_path}")
