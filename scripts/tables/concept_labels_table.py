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


_METRIC_KEYS = {
    "macro_acc": ("accuracy_macro", "macro acc"),
    "micro_acc": ("accuracy", "micro acc"),
    "f1": ("f1", "f1"),
}


def _metric_value(snapshot: dict, metric: str) -> tuple[str | None, float | None]:
    """Return (formatted value, numeric value) for the chosen metric.

    Falls back to macro (if present) → micro accuracy if the requested key
    is missing (older label files pre-date the macro column).
    """
    vr = snapshot.get("validator_results") or {}
    overall = vr.get("overall") or {}
    key, _ = _METRIC_KEYS[metric]
    val = overall.get(key)
    if val is None and metric == "macro_acc":
        # Older snapshots only had `accuracy` (micro).
        val = overall.get("accuracy")
    if val is None:
        return None, None
    return f"{val:.2f}", float(val)


def build_table(
    model: str,
    feats: list[int],
    metric: str | None,
    mark_drafts: bool,
) -> str:
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
        if metric is not None:
            val_str, _ = _metric_value(snap, metric)
        else:
            val_str = None
        rows.append((tfm_display, feat, label, val_str))

    if not rows:
        raise RuntimeError("No labels found for the given (model, feature) list")

    metric_header = _METRIC_KEYS[metric][1] if metric else None

    if metric:
        colspec = "l c p{10cm} c"
        header = fr"TFM & Concept & Label & {metric_header} \\"
    else:
        colspec = "l c p{11cm}"
        header = r"TFM & Concept & Label \\"

    draft_note = (
        " Labels are flagged as drafts pending polarity-inversion fixes on "
        r"$f_{86}$ and $f_{92}$."
        if mark_drafts
        else ""
    )
    metric_note = (
        fr" The \emph{{{metric_header}}} column is validator accuracy on 50 held-out "
        "rows per feature (5 activating + 5 non-activating per dataset)."
        if metric
        else ""
    )
    caption = (
        r"\caption{Draft SAE concept labels produced by the judge-gated "
        r"labeling pipeline (\texttt{PROMPT\_ORDER=A}, all-opus agents). "
        r"Each label is the judge's synthesis after round-by-round "
        r"refinement against held-out contrastive rows." + metric_note + draft_note + "}"
    )

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        caption,
        r"\label{tab:concept_labels}",
        r"\begin{tabular}{" + colspec + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for tfm, feat, label, val_str in rows:
        label_tex = _escape_tex(label)
        concept_tex = f"$f_{{{feat}}}$"
        if metric:
            val_tex = val_str if val_str is not None else "--"
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
        "--features", type=int, nargs="+", default=[6, 11, 36, 86, 92],
        help="Feature indices (default: all 5 mitra features swept so far)",
    )
    parser.add_argument(
        "--metric", choices=list(_METRIC_KEYS.keys()) + ["none"],
        default="macro_acc",
        help="Validator metric column; 'none' to omit (default: macro_acc)",
    )
    parser.add_argument(
        "--no-draft-note", action="store_true",
        help="Suppress the 'labels are drafts' caption note",
    )
    parser.add_argument(
        "--output", type=Path, default=TEX_OUT,
        help=f"Output .tex path (default: {TEX_OUT.name} next to this script)",
    )
    args = parser.parse_args()

    metric = None if args.metric == "none" else args.metric
    tex = build_table(args.model, args.features, metric, mark_drafts=not args.no_draft_note)
    args.output.write_text(tex)
    print(f"Wrote {args.output} ({len(tex)} bytes, {len(args.features)} rows)")


if __name__ == "__main__":
    main()
