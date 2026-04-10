#!/usr/bin/env python3
"""
Annotate a concept group with investigation findings.

Adds two fields to a concept group in cross_model_concept_labels_round10.json:
  - notes: curated investigation details (firing rate, characterization,
           open questions, linked investigation markdown files)
  - final_answer: the canonical one-sentence summary used in the paper

The `label` field (auto-generated) is left untouched. `final_answer` supersedes
`label` for downstream analyses once a concept has been investigated.

Usage:
    # From a JSON payload file
    python -m scripts.concepts.annotate_concept_group --gid 231 --from-json notes.json

    # Inline (set final_answer only)
    python -m scripts.concepts.annotate_concept_group --gid 231 \\
        --final-answer "Rows where X happens"

    # Show current annotation
    python -m scripts.concepts.annotate_concept_group --gid 231 --show
"""

import argparse
import json
import time
from pathlib import Path

from scripts._project_root import PROJECT_ROOT

GROUPS_PATH = PROJECT_ROOT / "output" / "cross_model_concept_labels_round10.json"


def load_groups() -> dict:
    with open(GROUPS_PATH) as f:
        return json.load(f)


def save_groups(data: dict) -> None:
    with open(GROUPS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def show(gid: str) -> None:
    data = load_groups()
    g = data["concept_groups"].get(gid)
    if g is None:
        print(f"Group {gid} not found")
        return

    print(f"=== Concept Group {gid} ===")
    print(f"Models: {g.get('n_models')}/8")
    print(f"Members: {len(g.get('members', []))}")
    print()
    print(f"LABEL (auto-generated):")
    print(f"  {g.get('label', '(none)')}")
    print()

    final = g.get("final_answer")
    if final:
        print(f"FINAL ANSWER (curated):")
        print(f"  {final}")
    else:
        print(f"FINAL ANSWER: (not set)")
    print()

    notes = g.get("notes")
    if notes:
        print("NOTES:")
        for k, v in notes.items():
            if isinstance(v, list):
                print(f"  {k}:")
                for item in v:
                    print(f"    - {item}")
            else:
                print(f"  {k}: {v}")
    else:
        print("NOTES: (none)")


def annotate(
    gid: str,
    final_answer: str | None = None,
    notes_payload: dict | None = None,
    merge_notes: bool = True,
) -> None:
    data = load_groups()
    groups = data["concept_groups"]
    if gid not in groups:
        raise KeyError(f"Group {gid} not found")

    g = groups[gid]

    if final_answer is not None:
        g["final_answer"] = final_answer

    if notes_payload is not None:
        existing = g.get("notes", {}) if merge_notes else {}
        existing.update(notes_payload)
        existing.setdefault("annotated", time.strftime("%Y-%m-%dT%H:%M:%S"))
        g["notes"] = existing

    save_groups(data)
    print(f"Updated group {gid}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gid", required=True, help="Concept group ID")
    parser.add_argument("--show", action="store_true", help="Show current annotation")
    parser.add_argument(
        "--final-answer", type=str, default=None,
        help="Set the canonical final_answer string",
    )
    parser.add_argument(
        "--from-json", type=Path, default=None,
        help="Load notes payload from a JSON file",
    )
    parser.add_argument(
        "--replace-notes", action="store_true",
        help="Replace existing notes instead of merging",
    )
    args = parser.parse_args()

    if args.show:
        show(args.gid)
        return

    notes_payload = None
    if args.from_json:
        with open(args.from_json) as f:
            payload = json.load(f)
        # If the payload has a top-level final_answer, pull it out
        if "final_answer" in payload and args.final_answer is None:
            args.final_answer = payload.pop("final_answer")
        notes_payload = payload

    if notes_payload is None and args.final_answer is None:
        parser.error("Specify --show, --final-answer, or --from-json")

    annotate(
        args.gid,
        final_answer=args.final_answer,
        notes_payload=notes_payload,
        merge_notes=not args.replace_notes,
    )
    show(args.gid)


if __name__ == "__main__":
    main()
