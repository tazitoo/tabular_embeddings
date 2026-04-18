#!/usr/bin/env python3
"""Generate a LaTeX table of SAE concept labels from the labeling pipeline.

Pulls `label` (and optional validator headline) from each
`output/contrastive_examples/{model}/f{feat}_label_A.json` snapshot
and emits a three-column table:

    TFM name | concept | label

Defaults to the mitra features we've run in the PROMPT_ORDER=A sweep
(f_6, f_11, f_36). Pass --features / --model to override.

Usage:
    python -m scripts.tables.concept_labels_table
    python -m scripts.tables.concept_labels_table --model mitra --features 6 11 36 86 92
    python -m scripts.tables.concept_labels_table --include-validator
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._project_root import PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "output" / "contrastive_examples"
TEX_OUT = Path(__file__).with_suffix(".tex")

TFM_DISPLAY = {
    "mitra": "Mitra",
    "tabpfn": "TabPFN",
    "tabicl": "TabICL",
    "tabicl_v2": "TabICL-v2",
    "tabdpt": "TabDPT",
    "carte": "CARTE",
}


def _escape_tex(s: str) -> str:
    """Minimal LaTeX escaping for free-text label content."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    out = s
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def _load_label_snapshot(model: str, feat: int) -> dict | None:
    """Return the parsed _A snapshot if it exists, else the latest label, else None."""
    model_dir = OUTPUT_DIR / model
    for suffix in ("_label_A.json", "_label.json"):
        path = model_dir / f"f{feat}{suffix}"
        if path.exists():
            data = json.loads(path.read_text())
            data["_source_path"] = str(path.relative_to(PROJECT_ROOT))
            return data
    return None


def _headline(snapshot: dict) -> str | None:
    """Formatted validator headline 'acc 0.68 (f1 0.68)' if present, else None."""
    vr = snapshot.get("validator_results") or {}
    overall = vr.get("overall") or {}
    acc = overall.get("accuracy")
    f1 = overall.get("f1")
    if acc is None:
        return None
    pieces = [f"acc {acc:.2f}"]
    if f1 is not None:
        pieces.append(f"f1 {f1:.2f}")
    return ", ".join(pieces)


def build_table(model: str, feats: list[int], include_validator: bool) -> str:
    tfm_display = TFM_DISPLAY.get(model, model)
    rows = []
    for feat in feats:
        snap = _load_label_snapshot(model, feat)
        if snap is None:
            print(f"[warn] no label for {model}/f{feat}; skipping", file=sys.stderr)
            continue
        label = snap.get("label", "")
        if not label:
            print(f"[warn] empty label for {model}/f{feat}", file=sys.stderr)
            continue
        rows.append((tfm_display, feat, label, _headline(snap) if include_validator else None))

    if not rows:
        raise RuntimeError("No labels found for the given (model, feature) list")

    # Column spec
    if include_validator:
        colspec = "l c p{9.5cm} l"
        header = (
            r"TFM & Concept & Label & Validator \\"
        )
    else:
        colspec = "l c p{11cm}"
        header = r"TFM & Concept & Label \\"

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        (
            r"\caption{Final SAE concept labels produced by the judge-gated "
            r"labeling pipeline (\texttt{PROMPT\_ORDER=A}, all-opus agents). "
            r"Each label is the judge's synthesis after round-by-round "
            r"refinement against held-out contrastive rows. "
            r"Validator column (when shown) is held-out accuracy / f1 on 50 "
            r"held-out rows per feature.}"
            if include_validator
            else r"\caption{Final SAE concept labels produced by the "
                 r"judge-gated labeling pipeline (\texttt{PROMPT\_ORDER=A}, "
                 r"all-opus agents). Each label is the judge's synthesis "
                 r"after round-by-round refinement against held-out "
                 r"contrastive rows.}"
        ),
        r"\label{tab:concept_labels}",
        r"\begin{tabular}{" + colspec + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for tfm, feat, label, val in rows:
        label_tex = _escape_tex(label)
        concept_tex = f"$f_{{{feat}}}$"
        if include_validator:
            val_tex = _escape_tex(val) if val else "--"
            lines.append(f"{tfm} & {concept_tex} & {label_tex} & {val_tex} \\\\")
        else:
            lines.append(f"{tfm} & {concept_tex} & {label_tex} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mitra")
    parser.add_argument(
        "--features", type=int, nargs="+", default=[6, 11, 36],
        help="Feature indices (default: 6 11 36 — the PROMPT_ORDER=A sweep so far)",
    )
    parser.add_argument(
        "--include-validator", action="store_true",
        help="Add a Validator column with held-out accuracy / f1",
    )
    parser.add_argument(
        "--output", type=Path, default=TEX_OUT,
        help=f"Output .tex path (default: {TEX_OUT.name} next to this script)",
    )
    args = parser.parse_args()

    tex = build_table(args.model, args.features, args.include_validator)
    args.output.write_text(tex)
    print(f"Wrote {args.output} ({len(tex)} bytes, {len(args.features)} rows)")


if __name__ == "__main__":
    main()
