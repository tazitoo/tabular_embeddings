#!/usr/bin/env python3
"""App F.3: build the prioritized patching burndown list for the 335 off-manifold concepts.

Takes the locked concept cell dumped by off_manifold_concept_stratification.py
(--dump) and, for each concept:
  - its share of the total off-manifold "contribution" mass (already in the dump,
    off_mass_share) -- this is what makes the 335 cell 20.5% of the total; used to
    order patching effort by impact.
  - the recipients it was accepted into (already in the dump).
  - candidate patch datasets, ranked by the existing dataset-quality cache
    (scripts/concepts/dataset_quality_cache.py, built from round-10 sae_test
    activations) via the same select_top_datasets() build_contrastive_examples.py
    uses -- NOT an arbitrary re-derivation of dataset choice.
  - whether contrastive evidence (output/contrastive_examples/{model}/f{feat}_*)
    already exists for the concept, since optimize_activation_suppression.py needs
    it and only a small fraction of the 335 have it yet.

Usage:
    python -m scripts.rebuttal.build_patching_burndown
    python -m scripts.rebuttal.build_patching_burndown --dump output/rebuttal/off_manifold_concept_dump_trained.csv
"""
import argparse
import csv
import re
from pathlib import Path

from scripts._project_root import PROJECT_ROOT
from scripts.concepts.dataset_quality_cache import (
    DEFAULT_CACHE_PATH,
    load_quality_cache,
    select_top_datasets,
)

CONTRASTIVE_DIR = PROJECT_ROOT / "output" / "contrastive_examples"


def _evidence_datasets(model: str, feat: int) -> set[str]:
    """Datasets for which output/contrastive_examples/{model}/f{feat}_{dataset}.csv exists."""
    d = CONTRASTIVE_DIR / model
    if not d.is_dir():
        return set()
    pat = re.compile(rf"^f{feat}_(.+)\.csv$")
    out = set()
    for f in d.iterdir():
        m = pat.match(f.name)
        if m:
            out.add(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(
        PROJECT_ROOT / "output" / "rebuttal" / "off_manifold_concept_dump_trained.csv"))
    ap.add_argument("--max-datasets-per-concept", type=int, default=3)
    ap.add_argument("--quality-cache", default=str(DEFAULT_CACHE_PATH))
    ap.add_argument("--out", default=str(
        PROJECT_ROOT / "output" / "rebuttal" / "patching_burndown.csv"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.dump)))
    cache = load_quality_cache(args.quality_cache)
    if cache is None:
        raise FileNotFoundError(f"Quality cache not found: {args.quality_cache}")

    out_rows = []
    n_no_quality_entry = 0
    n_no_workable_dataset = 0
    for r in rows:
        model, feat = r["donor"], int(r["feat_id"])
        entry = cache.get("models", {}).get(model, {}).get("features", {}).get(str(feat))
        if not entry:
            n_no_quality_entry += 1
            out_rows.append({**r, "chosen_datasets": "", "chosen_fire_rates": "",
                              "chosen_n_active": "", "chosen_n_inactive": "",
                              "n_with_evidence": 0, "n_without_evidence": 0,
                              "note": "NO_QUALITY_CACHE_ENTRY"})
            continue
        feature_entries = entry.get("datasets", {})
        chosen = select_top_datasets(feature_entries, args.max_datasets_per_concept)
        if not chosen:
            n_no_workable_dataset += 1
        have_evidence = _evidence_datasets(model, feat)
        n_with = sum(1 for ds in chosen if ds in have_evidence)
        n_without = len(chosen) - n_with
        out_rows.append({
            **r,
            "chosen_datasets": "|".join(chosen),
            "chosen_fire_rates": "|".join(
                f"{feature_entries[ds].get('fire_rate', float('nan')):.3f}" for ds in chosen),
            "chosen_n_active": "|".join(str(feature_entries[ds].get("n_active", "")) for ds in chosen),
            "chosen_n_inactive": "|".join(str(feature_entries[ds].get("n_inactive", "")) for ds in chosen),
            "n_with_evidence": n_with,
            "n_without_evidence": n_without,
            "note": "" if chosen else "NO_WORKABLE_DATASET",
        })

    # already sorted by off_mass descending from the dump; keep that order (impact-first)
    fieldnames = list(out_rows[0].keys())
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    n = len(out_rows)
    n_zero_evidence = sum(1 for r in out_rows if r.get("n_with_evidence", 0) == 0 and r["note"] != "NO_WORKABLE_DATASET" and r["note"] != "NO_QUALITY_CACHE_ENTRY")
    cum_share = 0.0
    print(f"{n} concepts written to {args.out}")
    print(f"  no quality-cache entry: {n_no_quality_entry}")
    print(f"  no workable dataset (all filtered_out): {n_no_workable_dataset}")
    print(f"  workable but zero contrastive evidence built yet: {n_zero_evidence}")
    print(f"\n  top 10 by off-manifold contribution share:")
    print(f"  {'donor':10s} {'feat':>5s} {'share':>7s} {'cum':>7s} {'univ':>4s} {'datasets':>3s} "
          f"{'w/evid':>6s} {'chosen datasets'}")
    for r in out_rows[:10]:
        cum_share += float(r["off_mass_share"])
        print(f"  {r['donor']:10s} {r['feat_id']:>5s} {float(r['off_mass_share']):7.2%} "
              f"{cum_share:7.2%} {r['universality']:>4s} {r['n_datasets']:>3s} "
              f"{r['n_with_evidence']:>6} {r['chosen_datasets']}")


if __name__ == "__main__":
    main()
