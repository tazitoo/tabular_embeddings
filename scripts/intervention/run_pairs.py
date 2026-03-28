#!/usr/bin/env python3
"""Run ablation_sweep for a list of model pairs sequentially.

Reads pairs from a CSV file (one pair per line: model_a,model_b) and
runs ablation_sweep for each pair via subprocess.  Sequential execution
avoids GPU contention when multiple pairs share the same worker.

CSV format (no header):
    tabpfn,mitra
    tabpfn,tabicl
    mitra,tabicl

Usage:
    # Default: reads pairs.csv in the project root
    python -m scripts.intervention.run_pairs

    # Explicit path
    python -m scripts.intervention.run_pairs --pairs /tmp/my_pairs.csv

    # Override device / datasets
    python -m scripts.intervention.run_pairs --device cuda:1
    python -m scripts.intervention.run_pairs --datasets credit-g adult

    # Resume: skip pairs whose output directory already has all datasets
    python -m scripts.intervention.run_pairs --resume
"""
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

from scripts._project_root import PROJECT_ROOT

DEFAULT_PAIRS_CSV = PROJECT_ROOT / "pairs.csv"


def load_pairs(csv_path: Path) -> list[tuple[str, str]]:
    pairs = []
    with open(csv_path) as f:
        for row in csv.reader(f):
            row = [c.strip() for c in row if c.strip()]
            if len(row) == 2:
                pairs.append((row[0], row[1]))
            elif row:
                print(f"[run_pairs] WARNING: skipping malformed row: {row}", flush=True)
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Run ablation_sweep for a list of model pairs sequentially")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS_CSV,
                        help="CSV file with model_a,model_b pairs (default: pairs.csv)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Pass --resume to each ablation_sweep invocation")
    args = parser.parse_args()

    if not args.pairs.exists():
        print(f"[run_pairs] ERROR: pairs file not found: {args.pairs}", flush=True)
        sys.exit(1)

    pairs = load_pairs(args.pairs)
    if not pairs:
        print("[run_pairs] No pairs found — exiting.", flush=True)
        sys.exit(0)

    print(f"[run_pairs] {len(pairs)} pair(s) to run", flush=True)
    for i, (a, b) in enumerate(pairs):
        cmd = [
            sys.executable, "-m", "scripts.intervention.ablation_sweep",
            "--models", a, b,
            "--device", args.device,
        ]
        if args.datasets:
            cmd += ["--datasets"] + args.datasets
        if args.resume:
            cmd.append("--resume")

        print(f"\n[run_pairs] [{i+1}/{len(pairs)}] {a} vs {b}", flush=True)
        print(f"[run_pairs]   {' '.join(cmd)}", flush=True)
        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0
        status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
        print(f"[run_pairs]   {status} in {elapsed:.0f}s", flush=True)

    print("\n[run_pairs] All pairs complete.", flush=True)


if __name__ == "__main__":
    main()
