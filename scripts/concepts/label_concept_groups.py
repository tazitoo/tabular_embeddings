#!/usr/bin/env python3
"""
Label cross-model concept groups with an LLM.

Five-step workflow:
  1. `prepare`  — extract prompts into numbered batch files, ordered by confidence
  2. `dispatch` — split each batch into agent-sized chunks (~50 groups each)
  3. `label`    — label batch 0 first as seed (API or Claude Code agents),
                  then remaining batches use few-shot examples from batch 0
  4. `combine`  — merge chunk label files back into per-batch label files
  5. `merge`    — write all labels back into the groups JSON with provenance

Usage:
    # Step 1: prepare batches (100 groups each, sorted by confidence)
    python -m scripts.concepts.label_concept_groups prepare

    # Step 2: dispatch a batch into agent-sized chunks
    python -m scripts.concepts.label_concept_groups dispatch --batch 5

    # Step 3a: label via API (requires ANTHROPIC_API_KEY)
    # python -m scripts.concepts.label_concept_groups label --batch 0

    # Step 3b: label via Claude Code Sonnet agents (actual method used)
    # Each agent reads a chunk file and writes labels_batch_NN_chunk_M.json
    # See dispatch output for agent prompt templates.

    # Step 4: combine chunk labels into batch label file
    python -m scripts.concepts.label_concept_groups combine --batch 5

    # Step 5: merge all batch labels back into groups JSON
    python -m scripts.concepts.label_concept_groups merge
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.sae.compare_sae_cross_model import DEFAULT_SAE_ROUND

GROUPS_PATH = PROJECT_ROOT / "output" / f"cross_model_concept_labels_round{DEFAULT_SAE_ROUND}.json"
CONCEPTS_PATH = PROJECT_ROOT / "output" / f"sae_concept_analysis_round{DEFAULT_SAE_ROUND}.json"
LABELING_DIR = PROJECT_ROOT / "output" / "concept_labeling"

BATCH_SIZE = 50   # groups per batch (smaller for reliable agent processing)
CHUNK_SIZE = 25   # groups per agent chunk


# ── Confidence scoring ───────────────────────────────────────────────────


def score_group(gid: str, group: dict, concepts: dict) -> dict:
    """Score a concept group by labeling confidence.

    Confidence = (n_models, frac_members_with_good_examples, mean_r2).
    "Good examples" means ≥5 non-zero activating rows in the top examples.
    """
    members = group["members"]
    n_good = 0
    for model, fidx in members:
        pf = concepts["models"].get(model, {}).get("per_feature", {})
        feat = pf.get(str(fidx), {})
        top = feat.get("examples", {}).get("top", [])
        n_nonzero = sum(1 for r in top if r.get("activation", 0) > 0)
        if n_nonzero >= 5:
            n_good += 1
    frac_good = n_good / max(len(members), 1)
    return {
        "gid": gid,
        "n_models": group["n_models"],
        "frac_good": frac_good,
        "mean_r2": group["mean_r2"],
        "n_members": len(members),
    }


# ── Prepare ──────────────────────────────────────────────────────────────


def cmd_prepare(args):
    """Extract prompts into numbered batch files, ordered by confidence."""
    with open(args.groups) as f:
        data = json.load(f)
    with open(args.concepts) as f:
        concepts = json.load(f)

    groups = data["concept_groups"]

    # Score and sort all groups except group 0 (mega-group, handled separately)
    scored = []
    for gid, g in groups.items():
        if gid == "0":
            continue
        if not g.get("prompt"):
            continue
        scored.append(score_group(gid, g, concepts))

    scored.sort(
        key=lambda x: (x["n_models"], x["frac_good"], x["mean_r2"]),
        reverse=True,
    )

    # Write batches
    LABELING_DIR.mkdir(parents=True, exist_ok=True)

    n_batches = 0
    for i in range(0, len(scored), BATCH_SIZE):
        batch = scored[i : i + BATCH_SIZE]
        batch_data = []
        for s in batch:
            g = groups[s["gid"]]
            batch_data.append({
                "gid": s["gid"],
                "prompt": g["prompt"],
                "n_members": s["n_members"],
                "n_models": s["n_models"],
                "frac_good": round(s["frac_good"], 3),
                "mean_r2": round(s["mean_r2"], 4),
            })

        batch_path = LABELING_DIR / f"batch_{i // BATCH_SIZE:02d}.json"
        with open(batch_path, "w") as f:
            json.dump(batch_data, f, indent=2)
        n_batches += 1

    # Write manifest
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "groups_file": str(args.groups),
        "concepts_file": str(args.concepts),
        "sae_round": DEFAULT_SAE_ROUND,
        "n_groups": len(scored),
        "n_batches": n_batches,
        "batch_size": BATCH_SIZE,
        "ordering": "confidence_desc (n_models, frac_good_examples, mean_r2)",
        "skipped": ["group_0 (mega-group, requires splitting)"],
        "system_prompt": data.get("metadata", {}).get("system_prompt", ""),
    }
    with open(LABELING_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Prepared {n_batches} batches ({len(scored)} groups) in {LABELING_DIR}")
    print(f"  Batch 0: groups with highest confidence (top {BATCH_SIZE})")
    print(f"  Batch {n_batches - 1}: groups with lowest confidence")
    print(f"  Skipped: group 0 ({len(groups['0']['members'])} members)")


# ── Label (API path — commented out, documented) ─────────────────────────


def cmd_label(args):
    """Label a single batch via LLM API.

    ACTUAL METHOD USED: Claude Code with Sonnet agents.
    The Sonnet agent reads batch_NN.json, labels each group, and writes
    labels_batch_NN.json as {gid: label_string}.

    The API path below is provided for reproducibility if an API key is
    available, but was not used for the paper results.
    """
    batch_path = LABELING_DIR / f"batch_{args.batch:02d}.json"
    if not batch_path.exists():
        print(f"Batch file not found: {batch_path}")
        return

    with open(batch_path) as f:
        batch = json.load(f)

    manifest_path = LABELING_DIR / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    system_prompt = manifest.get("system_prompt", "")

    # Load seed labels as few-shot examples (from batch 0 labeling)
    seed_path = LABELING_DIR / "labels_batch_00.json"
    few_shot = ""
    if seed_path.exists() and args.batch > 0:
        with open(seed_path) as f:
            seed_labels = json.load(f)
        examples = list(seed_labels.items())[:5]
        few_shot = "\n\nHere are examples of previously labeled concepts for style reference:\n"
        for gid, label in examples:
            few_shot += f"  Group {gid}: {label}\n"

    labels = {}

    # ── API labeling (not used for paper results) ──
    # Uncomment and set ANTHROPIC_API_KEY to use:
    #
    # import anthropic
    # client = anthropic.Anthropic()
    # for item in batch:
    #     prompt = item["prompt"] + few_shot
    #     message = client.messages.create(
    #         model="claude-sonnet-4-20250514",
    #         max_tokens=256,
    #         system=system_prompt,
    #         messages=[{"role": "user", "content": prompt}],
    #     )
    #     labels[item["gid"]] = message.content[0].text.strip()
    #     print(f"  Group {item['gid']}: {labels[item['gid']]}")

    # ── Claude Code labeling (actual method used) ──
    # Labels produced by Claude Code Sonnet agents reading batch files.
    # The agent receives the system prompt and few-shot examples, processes
    # all prompts in the batch, and writes labels_batch_NN.json.
    print(f"Batch {args.batch}: {len(batch)} groups")
    print(f"To label this batch in Claude Code, the Sonnet agent reads:")
    print(f"  {batch_path}")
    print(f"And writes labels to:")
    print(f"  {LABELING_DIR / f'labels_batch_{args.batch:02d}.json'}")

    if labels:
        out_path = LABELING_DIR / f"labels_batch_{args.batch:02d}.json"
        with open(out_path, "w") as f:
            json.dump(labels, f, indent=2)
        print(f"Wrote {len(labels)} labels to {out_path}")


# ── Dispatch (split batches into agent-sized chunks) ─────────────────────


AGENT_PROMPT_TEMPLATE = """\
You are labeling concept groups from a Sparse Autoencoder analysis of tabular
foundation models. Each group contains contrastive examples: raw data rows where
the feature fires strongly vs nearby rows where it stays silent. Each row also
shows PMI (target predictiveness), surprise (value rarity), and compression
contribution (inter-row novelty) with dataset ranges.

For EACH group, write a concise label describing the single coherent data pattern
visible in the RAW DATA VALUES. Do not cite specific column names, specific
numeric values, or dataset names. Describe the abstract structural pattern
(magnitude, sparsity, scale heterogeneity, categorical uniformity, etc.).

{few_shot}
Read the file: {chunk_path}

For each entry in the JSON array, read the "prompt" field and produce a label.
Write a JSON file mapping group IDs to labels:

    {{"<gid>": "<label>", ...}}

Write your output to: {output_path}
"""


def _load_few_shot(n: int = 5) -> str:
    """Load seed labels from batch 0 as few-shot examples.

    Bootstrapping: batch 0 is labeled first WITHOUT few-shot examples
    (labels_batch_00.json doesn't exist yet, so this returns "").
    Once batch 0 is labeled and combined, its labels become the few-shot
    seed for all subsequent batches (1–N), ensuring style consistency
    without leaking labels from prior SAE rounds.
    """
    seed_path = LABELING_DIR / "labels_batch_00.json"
    if not seed_path.exists():
        return ""
    with open(seed_path) as f:
        seed_labels = json.load(f)
    examples = list(seed_labels.items())[:n]
    lines = ["Here are examples of previously labeled concepts for style reference:"]
    for gid, label in examples:
        lines.append(f"  Group {gid}: {label}")
    return "\n".join(lines)


def cmd_dispatch(args):
    """Split a batch into agent-sized chunks with full prompt context.

    Each chunk is a self-contained JSON file small enough for a Sonnet agent
    to read (~50 groups, ~250KB). The dispatch also writes an agent_prompt
    text file documenting exactly what the agent should do.

    Usage:
        python -m scripts.concepts.label_concept_groups dispatch --batch 5
        # Creates: output/concept_labeling/chunks/batch_05_chunk_0.json
        #          output/concept_labeling/chunks/batch_05_chunk_1.json
        #          output/concept_labeling/chunks/batch_05_agent_prompt.txt
    """
    batch_path = LABELING_DIR / f"batch_{args.batch:02d}.json"
    if not batch_path.exists():
        print(f"Batch file not found: {batch_path}")
        return

    with open(batch_path) as f:
        batch = json.load(f)

    chunk_dir = LABELING_DIR / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    few_shot = _load_few_shot()

    n_chunks = 0
    for i in range(0, len(batch), CHUNK_SIZE):
        chunk = batch[i : i + CHUNK_SIZE]
        chunk_idx = i // CHUNK_SIZE
        chunk_path = chunk_dir / f"batch_{args.batch:02d}_chunk_{chunk_idx}.json"
        with open(chunk_path, "w") as f:
            json.dump(chunk, f, indent=2)

        output_path = chunk_dir / f"labels_batch_{args.batch:02d}_chunk_{chunk_idx}.json"
        prompt = AGENT_PROMPT_TEMPLATE.format(
            few_shot=few_shot,
            chunk_path=chunk_path,
            output_path=output_path,
        )
        prompt_path = chunk_dir / f"batch_{args.batch:02d}_chunk_{chunk_idx}_prompt.txt"
        with open(prompt_path, "w") as f:
            f.write(prompt)

        n_chunks += 1
        print(f"  Chunk {chunk_idx}: {len(chunk)} groups -> {chunk_path.name}")

    print(f"\nDispatched batch {args.batch} into {n_chunks} chunks in {chunk_dir}")
    print(f"Agent prompts written alongside chunk files.")
    print(f"\nTo label in Claude Code, launch Sonnet agents with:")
    for chunk_idx in range(n_chunks):
        prompt_path = chunk_dir / f"batch_{args.batch:02d}_chunk_{chunk_idx}_prompt.txt"
        print(f"  Agent(prompt=read('{prompt_path}'), model='sonnet')")


def cmd_combine(args):
    """Combine chunk label files back into a single batch label file.

    Reads all labels_batch_NN_chunk_*.json from chunks/ and writes the
    combined labels_batch_NN.json to the labeling directory.
    """
    chunk_dir = LABELING_DIR / "chunks"
    pattern = f"labels_batch_{args.batch:02d}_chunk_*.json"
    chunk_files = sorted(chunk_dir.glob(pattern))

    if not chunk_files:
        print(f"No chunk label files found matching {pattern} in {chunk_dir}")
        return

    combined = {}
    for cf in chunk_files:
        with open(cf) as f:
            labels = json.load(f)
        combined.update(labels)
        print(f"  {cf.name}: {len(labels)} labels")

    out_path = LABELING_DIR / f"labels_batch_{args.batch:02d}.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nCombined {len(combined)} labels -> {out_path}")


# ── Merge ────────────────────────────────────────────────────────────────


def cmd_merge(args):
    """Merge all label files back into the groups JSON."""
    with open(args.groups) as f:
        data = json.load(f)

    groups = data["concept_groups"]

    # Collect all label files
    label_files = sorted(LABELING_DIR.glob("labels_batch_*.json"))
    if not label_files:
        print(f"No label files found in {LABELING_DIR}")
        return

    total = 0
    sources = []
    for lf in label_files:
        with open(lf) as f:
            labels = json.load(f)
        for gid, label in labels.items():
            if gid in groups:
                groups[gid]["label"] = label
                total += 1
        sources.append(lf.name)

    # Update metadata
    data.setdefault("metadata", {})
    data["metadata"]["labeling"] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": "Claude Code with Sonnet agents (batch labeling)",
        "n_labeled": total,
        "label_files": sources,
        "sae_round": DEFAULT_SAE_ROUND,
    }

    # Update summary
    n_labeled = sum(
        1 for g in groups.values()
        if g.get("label") and g["label"] != "unlabeled"
    )
    data["summary"]["n_labeled"] = n_labeled

    with open(args.groups, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Merged {total} labels from {len(label_files)} files into {args.groups}")
    print(f"  Total labeled: {n_labeled}/{len(groups)}")


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Label cross-model concept groups (prepare/label/merge)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Extract prompts into batch files")
    p_prepare.add_argument(
        "--groups", type=Path, default=GROUPS_PATH,
        help="Groups JSON (default: cross_model_concept_labels_round{N}.json)",
    )
    p_prepare.add_argument(
        "--concepts", type=Path, default=CONCEPTS_PATH,
        help="Concept analysis JSON (default: sae_concept_analysis_round{N}.json)",
    )

    p_label = sub.add_parser("label", help="Label a batch (API or instructions)")
    p_label.add_argument("--batch", type=int, required=True, help="Batch number")

    p_dispatch = sub.add_parser("dispatch", help="Split batch into agent-sized chunks")
    p_dispatch.add_argument("--batch", type=int, required=True, help="Batch number")

    p_combine = sub.add_parser("combine", help="Combine chunk labels into batch label file")
    p_combine.add_argument("--batch", type=int, required=True, help="Batch number")

    p_merge = sub.add_parser("merge", help="Merge label files back into groups JSON")
    p_merge.add_argument(
        "--groups", type=Path, default=GROUPS_PATH,
        help="Groups JSON to update",
    )

    args = parser.parse_args()

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "label":
        cmd_label(args)
    elif args.command == "dispatch":
        cmd_dispatch(args)
    elif args.command == "combine":
        cmd_combine(args)
    elif args.command == "merge":
        cmd_merge(args)


if __name__ == "__main__":
    main()
