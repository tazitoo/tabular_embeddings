#!/usr/bin/env python3
"""
Batch helper for labeling individual (model, feat_idx) concepts.

Targets the `unmatched_features.explained` bucket in
`output/cross_model_concept_labels_round10.json` — these are per-model
features with their own prompts but no cross-model group label.

The actual labeling is expected to happen inside a Claude Code session:
a small team of Haiku agents each read the prompt and propose a 1-2
sentence label; a single Sonnet agent then adjudicates and writes the
final label. This script only handles (a) selecting which concepts
need labels, (b) serialising them to a batch file the dispatcher can
consume, and (c) merging the final labels back with provenance.

Subcommands:
    prepare   Select concepts + write a batch JSON of prompts
    merge     Write labels from a labels JSON back into the groups file
    status    Print labeled/unlabeled counts per model

Selectors:
    --from-gap-index        Top-5 unmatched features per (weak, task) from
                            output/concept_labeling/weak_gap_index.json
    --top-unmatched N       Top-N unmatched features per model by
                            ablation-selection frequency (from ablation_sweep)
    --ids STR               Explicit comma-separated IDs, e.g.
                            "TabPFN:12,TabICL-v2:2"
    --only-unlabeled        Drop items that already have a real label

Usage:
    python -m scripts.concepts.label_individual_concepts prepare \\
        --from-gap-index --only-unlabeled --output batch.json

    python -m scripts.concepts.label_individual_concepts merge \\
        --labels labels.json

    python -m scripts.concepts.label_individual_concepts status
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scripts._project_root import PROJECT_ROOT

GROUPS_PATH = (
    PROJECT_ROOT / "output" / "cross_model_concept_labels_round10.json"
)
GAP_INDEX_PATH = (
    PROJECT_ROOT / "output" / "concept_labeling" / "weak_gap_index.json"
)
ABLATION_SWEEP_DIR = PROJECT_ROOT / "output" / "ablation_sweep"

WEAK_ORDER = ["CARTE", "Mitra", "TabDPT", "TabICL", "TabICL-v2", "TabPFN"]
TASKS = ["classification", "regression"]


# ── IO ───────────────────────────────────────────────────────────────────


def load_groups_file() -> dict:
    with open(GROUPS_PATH) as f:
        return json.load(f)


def save_groups_file(data: dict) -> None:
    with open(GROUPS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _iter_explained(data: dict):
    """Yield (model, feat_idx:str, entry_dict) for all explained unmatched
    features in the groups JSON."""
    exp = data.get("unmatched_features", {}).get("explained", {})
    for model, feats in exp.items():
        for fi_str, entry in feats.items():
            yield model, fi_str, entry


def _is_unlabeled(entry: dict) -> bool:
    label = entry.get("label")
    return (not label) or label == "unlabeled"


# ── Selectors ────────────────────────────────────────────────────────────


def select_from_gap_index(top_k: int = 5) -> list[tuple[str, int]]:
    """Return (model, feat_idx) pairs from top-K of each (weak, task)."""
    gap = json.loads(GAP_INDEX_PATH.read_text())
    out: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for weak in WEAK_ORDER:
        for task in TASKS:
            entries = gap.get(weak, {}).get(task, {}).get("top_gaps", [])
            for e in entries[:top_k]:
                key = (e["strong_model"], int(e["feat_idx"]))
                if key not in seen:
                    seen.add(key)
                    out.append(key)
    return out


def select_top_unmatched_per_model(n_per_model: int) -> list[tuple[str, int]]:
    """Rank each model's unmatched features by how often ablation_sweep
    accepted them (summed across all pairs and datasets), return top N per
    model. Uses `selected_features` so only features actually chosen by the
    greedy search are counted."""
    code_to_display = {
        "carte": "CARTE", "mitra": "Mitra", "tabdpt": "TabDPT",
        "tabicl": "TabICL", "tabicl_v2": "TabICL-v2", "tabpfn": "TabPFN",
    }
    counts: dict[str, Counter] = defaultdict(Counter)
    import numpy as np
    for pair_dir in sorted(ABLATION_SWEEP_DIR.iterdir()):
        if not pair_dir.is_dir():
            continue
        for npz_path in pair_dir.glob("*.npz"):
            d = np.load(npz_path, allow_pickle=True)
            if "selected_features" not in d or "strong_model" not in d:
                continue
            strong = code_to_display.get(str(d["strong_model"]))
            if strong is None:
                continue
            sel = d["selected_features"]  # (n_rows, max_k)
            for row in sel:
                for fi in row:
                    if fi < 0:
                        break
                    counts[strong][int(fi)] += 1
    out: list[tuple[str, int]] = []
    for model in WEAK_ORDER:
        for fi, _cnt in counts[model].most_common(n_per_model):
            out.append((model, fi))
    return out


def parse_ids(arg: str) -> list[tuple[str, int]]:
    out = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        model, fi = token.split(":")
        out.append((model.strip(), int(fi)))
    return out


# ── Commands ─────────────────────────────────────────────────────────────


def cmd_prepare(args: argparse.Namespace) -> None:
    data = load_groups_file()

    selections: list[tuple[str, int]] = []
    if args.from_gap_index:
        selections.extend(select_from_gap_index(args.gap_top_k))
    if args.top_unmatched:
        selections.extend(select_top_unmatched_per_model(args.top_unmatched))
    if args.ids:
        selections.extend(parse_ids(args.ids))

    if not selections:
        print("No selector given (use --from-gap-index / --top-unmatched N / --ids)")
        sys.exit(2)

    # Deduplicate while preserving order
    seen: set[tuple[str, int]] = set()
    unique: list[tuple[str, int]] = []
    for key in selections:
        if key not in seen:
            seen.add(key)
            unique.append(key)

    # Resolve each selection in the groups file
    exp = data.get("unmatched_features", {}).get("explained", {})
    items = []
    missing = []
    skipped_labeled = 0
    for model, fi in unique:
        entry = exp.get(model, {}).get(str(fi))
        if entry is None:
            missing.append((model, fi))
            continue
        if args.only_unlabeled and not _is_unlabeled(entry):
            skipped_labeled += 1
            continue
        items.append({
            "id": f"{model}:{fi}",
            "model": model,
            "feat_idx": fi,
            "prompt": entry.get("prompt", ""),
            "meta": {
                "r2": entry.get("r2"),
                "signature": entry.get("signature"),
                "top_probes": entry.get("top_probes"),
                "current_label": entry.get("label"),
            },
        })

    batch = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": {
            "from_gap_index": bool(args.from_gap_index),
            "gap_top_k": args.gap_top_k if args.from_gap_index else None,
            "top_unmatched": args.top_unmatched,
            "ids": args.ids,
            "only_unlabeled": bool(args.only_unlabeled),
        },
        "n_items": len(items),
        "items": items,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(batch, f, indent=2)

    print(f"Wrote {len(items)} items → {out_path}")
    if missing:
        print(f"  Not in explained bucket: {len(missing)}")
        for model, fi in missing:
            print(f"    {model}:{fi}")
    if skipped_labeled:
        print(f"  Skipped already-labeled: {skipped_labeled}")


def cmd_merge(args: argparse.Namespace) -> None:
    with open(args.labels) as f:
        labels: dict[str, str] = json.load(f)

    data = load_groups_file()
    exp = data.get("unmatched_features", {}).get("explained", {})

    written = 0
    missing = []
    skipped_no_change = 0
    for item_id, label in labels.items():
        if not isinstance(label, str) or not label.strip():
            continue
        try:
            model, fi = item_id.split(":")
        except ValueError:
            print(f"  bad id: {item_id}")
            continue
        entry = exp.get(model, {}).get(fi)
        if entry is None:
            missing.append(item_id)
            continue
        if entry.get("label") == label.strip():
            skipped_no_change += 1
            continue
        entry["label"] = label.strip()
        entry.setdefault("label_provenance", {})
        entry["label_provenance"] = {
            "written_at": datetime.now(timezone.utc).isoformat(),
            "source": args.source or "agent-consensus",
        }
        written += 1

    if written and not args.dry_run:
        save_groups_file(data)

    print(
        f"{'(dry-run) would write' if args.dry_run else 'Wrote'} "
        f"{written} labels; unchanged={skipped_no_change}; "
        f"missing={len(missing)}"
    )
    if missing:
        for m in missing:
            print(f"  missing: {m}")


def cmd_status(args: argparse.Namespace) -> None:
    data = load_groups_file()
    totals: Counter = Counter()
    labeled: Counter = Counter()
    for model, _fi, entry in _iter_explained(data):
        totals[model] += 1
        if not _is_unlabeled(entry):
            labeled[model] += 1
    print(f"{'model':12s} {'labeled':>10s} {'total':>10s} {'pct':>8s}")
    for model in sorted(totals):
        n = totals[model]
        k = labeled[model]
        pct = 100 * k / n if n else 0
        print(f"{model:12s} {k:10d} {n:10d} {pct:7.1f}%")


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="Write batch JSON of prompts")
    p_prep.add_argument("--from-gap-index", action="store_true")
    p_prep.add_argument("--gap-top-k", type=int, default=5)
    p_prep.add_argument("--top-unmatched", type=int, default=None,
                        help="Top N unmatched features per model by ablation usage")
    p_prep.add_argument("--ids", type=str, default=None,
                        help='Comma-separated "Model:feat" ids')
    p_prep.add_argument("--only-unlabeled", action="store_true")
    p_prep.add_argument("--output", type=str, required=True)
    p_prep.set_defaults(func=cmd_prepare)

    p_merge = sub.add_parser("merge", help="Write labels JSON back")
    p_merge.add_argument("--labels", type=str, required=True,
                         help='JSON file mapping "Model:feat" → label string')
    p_merge.add_argument("--source", type=str, default=None)
    p_merge.add_argument("--dry-run", action="store_true")
    p_merge.set_defaults(func=cmd_merge)

    p_status = sub.add_parser("status", help="Print labeled / unlabeled counts")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
