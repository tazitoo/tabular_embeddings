#!/usr/bin/env python3
"""Seed a new SAE round from the previous one without touching any existing file.

A round is three directories under output/:

    sae_training_round{N}/         {model}_taskaware_{sae_training,sae_test,norm_stats}.npz
    sae_tabarena_sweep_round{N}/   {model}/  (Optuna study, validated / seed / random checkpoints)
    sae_random_baseline_round{N}/  {model}/  (geometry-matched random-SAE control; round 10's is
                                   the untagged output/sae_random_baseline, linked as _round10)

Regenerating one model's corpus and SAE must not rewrite the previous round, and the
other models' artifacts must stay bit-identical, so the new round holds RELATIVE
symlinks to the previous round for every model that is not being regenerated. The
regenerated models are left absent: 04_extract_all_layers -> 07_build_sae_training_data
-> sae_tabarena_sweep write them fresh. Every consumer resolves its inputs through
DEFAULT_SAE_ROUND (scripts/sae/compare_sae_cross_model.py), so bumping that constant
moves the whole pipeline onto the new round.

Run on every host that holds the outputs (links are relative, so the tree is portable):

    python -m scripts.sae_corpus.promote_round --from 10 --to 11 --regenerate tabdpt
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._project_root import PROJECT_ROOT  # noqa: E402

MODEL_FILE_SUFFIXES = ("_sae_training.npz", "_sae_test.npz", "_norm_stats.npz")


def _model_of(filename: str) -> str | None:
    for suffix in MODEL_FILE_SUFFIXES:
        if filename.endswith(suffix):
            stem = filename[: -len(suffix)]
            return stem.split("_taskaware")[0].split("_layer")[0]
    return None


def _link(dst: Path, src: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        raise FileExistsError(f"{dst} exists and is not a link to {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.relpath(src, dst.parent), dst)


def promote_round(output_root: Path, src_round: int, dst_round: int,
                  regenerate: list[str]) -> list[str]:
    """Create round `dst_round` from `src_round`; return the models that were linked."""
    src_train = output_root / f"sae_training_round{src_round}"
    dst_train = output_root / f"sae_training_round{dst_round}"
    src_sweep = output_root / f"sae_tabarena_sweep_round{src_round}"
    dst_sweep = output_root / f"sae_tabarena_sweep_round{dst_round}"
    src_random = output_root / f"sae_random_baseline_round{src_round}"
    dst_random = output_root / f"sae_random_baseline_round{dst_round}"
    skip = set(regenerate)
    linked: set[str] = set()

    for f in sorted(src_train.iterdir()):
        model = _model_of(f.name)
        if model is None or model in skip:
            continue
        _link(dst_train / f.name, f)
        linked.add(model)

    for d in sorted(p for p in src_sweep.iterdir() if p.is_dir()):
        if d.name in skip:
            continue
        _link(dst_sweep / d.name, d)
        linked.add(d.name)

    if src_random.is_dir():
        for d in sorted(p for p in src_random.iterdir() if p.is_dir()):
            if d.name in skip:
                continue
            _link(dst_random / d.name, d)
        dst_random.mkdir(parents=True, exist_ok=True)

    dst_train.mkdir(parents=True, exist_ok=True)
    dst_sweep.mkdir(parents=True, exist_ok=True)
    return sorted(linked)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--from", dest="src_round", type=int, required=True)
    ap.add_argument("--to", dest="dst_round", type=int, required=True)
    ap.add_argument("--regenerate", nargs="+", required=True,
                    help="models whose corpus and SAE will be rebuilt in the new round")
    ap.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output")
    args = ap.parse_args()
    linked = promote_round(args.output_root, args.src_round, args.dst_round, args.regenerate)
    print(f"round{args.dst_round} <- round{args.src_round}: linked {linked}; "
          f"to regenerate: {sorted(args.regenerate)}")


if __name__ == "__main__":
    main()
