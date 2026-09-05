#!/usr/bin/env python3
"""App F.3 patch explanations: the diagnosis, the worked examples, and the full data.

Three artefacts, because the same 877 (concept, dataset, recipient) cells answer three
different questions and no single table answers all of them.

  Diagnostic matrix   For a model builder: which recipient lacks how much, from which
                      donor, and what a concept is worth there. Distinct concepts, so
                      it does not double count a concept deployed into several datasets.
  Quartile panels     For a reader who wants to see the pipeline work: the input edit,
                      the concept it silences, and the recipient's prediction change
                      against what perfect ablation of that concept achieves.
  CSV                 Everything, machine readable, so nothing is lost to the page
                      budget and the panels do not have to be comprehensive.

The panels are stratified by the ABLATION effect, not by the patch's own effect.
Ordering by our result would float successes to the top and bury failures; ordering by
the ceiling orders by how much each concept matters to the transfer, which is a
property of the concept, and lets the patch column report honestly against it.

Prediction change is in true-class probability rather than gap-closure units. gc
divides by the strong-weak gap, which is right for aggregation and misleading for a
single row: a small denominator turns a 0.008 probability move into a gc of 0.48.

Run with no arguments; the defaults are the v31 canonical paths.
"""
import argparse
import collections
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.paper._paper_repo import paper_table_path

REBUTTAL = PROJECT_ROOT / "output" / "rebuttal"
SWEEP_GLOBS = [str(REBUTTAL / "v31q" / "*.json"), str(REBUTTAL / "v31q_f78.json")]
FWD = REBUTTAL / "forward_deltas"
OUT_DIR = Path(__file__).parent
# The four datasets carrying the most concepts; enough to show the pipeline without
# being a catalogue. Coverage is the CSV's job.
PANEL_DATASETS = ["splice", "Amazon_employee_access", "E-CommereShippingData", "anneal"]
PER_DATASET = 3
N_QUARTILES = 4
QUARTILE_LABEL = ["largest", "upper-middle", "lower-middle", "smallest"]
MILLI = 1e3
SEED = 0


def _tex(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def _val(v):
    """Table values are read by people: 4 significant figures, not float repr."""
    return f"{v:.4g}" if isinstance(v, float) else str(v)[:15]


def load_cells():
    """One record per (concept, dataset, recipient) that produced a measurable patch.

    The row kept for each cell is the one the search's OBJECTIVE selected, not the one
    with the largest suppression: reporting the best suppression would be reporting the
    acceptance criterion rather than what the pipeline does.
    """
    out = []
    for pattern in SWEEP_GLOBS:
        for path in sorted(glob.glob(pattern)):
            for concept in json.load(open(path)):
                for cell in concept.get("datasets", []):
                    if not (cell.get("rows") and cell.get("readout_usable")):
                        continue
                    won = max((r for r in cell["rows"] if r.get("best")),
                              key=lambda r: r["best"].get("score", -1), default=None)
                    if won is None:
                        continue
                    readout = won.get("readout") or {}
                    ablate = readout.get("interval_readout")
                    patch = readout.get("movement_total_measured")
                    out.append({
                        "key": (concept["donor"], int(concept["feat"])),
                        "donor": concept["donor"], "feat": int(concept["feat"]),
                        "dataset": cell["dataset"], "recipient": cell["recipient"],
                        "n_rows": cell.get("n_accepted_rows"),
                        "row": won.get("row"), "won": won,
                        "a_start": won.get("a_start"), "a_final": won.get("a_final"),
                        "ablate": ablate * MILLI if isinstance(ablate, (int, float)) else None,
                        "patch": patch * MILLI if isinstance(patch, (int, float)) else None,
                    })
    return out


def diagnostic_matrix(cells):
    """(donor, recipient) -> distinct concepts and the median |ablation effect|.

    Counted over DISTINCT concepts: a concept deployed into three datasets is one thing
    the recipient lacks, not three.
    """
    grouped = collections.defaultdict(lambda: {"keys": set(), "ablate": []})
    for c in cells:
        g = grouped[(c["donor"], c["recipient"])]
        g["keys"].add(c["key"])
        if c["ablate"] is not None:
            g["ablate"].append(abs(c["ablate"]))
    return grouped


def _totals(cells, index, value):
    keys, ablate = set(), []
    for c in cells:
        if c[index] == value:
            keys.add(c["key"])
            if c["ablate"] is not None:
                ablate.append(abs(c["ablate"]))
    return keys, ablate


def render_matrix(cells):
    grouped = diagnostic_matrix(cells)
    donors = sorted({d for d, _ in grouped})
    recips = sorted({r for _, r in grouped}, key=lambda r: -len(_totals(cells, "recipient", r)[0]))
    out = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{What each recipient is missing. Each cell counts the distinct "
        r"off-manifold concepts a donor carries that the recipient lacks, and the "
        r"median effect of ablating one, in true-class probability ($\times 10^{-3}$). "
        r"Row and column totals count distinct concepts, so they are smaller than the "
        r"sums: a concept missing from two recipients is counted once in each column "
        r"but once overall. TabICL and TabICL-v2 cannot share an environment, so that "
        r"pair is unmeasured rather than empty.}",
        r"\label{tab:patch_diagnosis}", r"\small",
        r"\begin{tabular}{l" + "r" * (len(recips) + 1) + "}", r"\toprule",
        "Donor & " + " & ".join(_tex(r) for r in recips) + r" & Distinct \\",
        r"\midrule",
    ]
    for donor in donors:
        row = [_tex(donor)]
        for recip in recips:
            g = grouped.get((donor, recip))
            row.append("--" if not g else
                       rf"{len(g['keys'])} / {np.median(g['ablate']):.1f}"
                       if g["ablate"] else f"{len(g['keys'])} / --")
        row.append(str(len(_totals(cells, "donor", donor)[0])))
        out.append(" & ".join(row) + r" \\")
    out.append(r"\midrule")
    tot = ["Distinct"]
    for recip in recips:
        keys, ablate = _totals(cells, "recipient", recip)
        tot.append(rf"{len(keys)} / {np.median(ablate):.1f}" if ablate else f"{len(keys)} / --")
    tot.append(str(len({c['key'] for c in cells})))
    out += [" & ".join(tot) + r" \\", r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def _row_indices():
    """(donor, recipient, dataset) -> original table row for each searched row."""
    out = {}
    for f in sorted(glob.glob(str(FWD / "*" / "*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "row_indices" in z.files:
            out[(str(z["strong_model"]), str(z["weak_model"]),
                 os.path.basename(f)[:-4])] = z["row_indices"]
    return out


def _raw_tables():
    from data.extended_loader import _load_tabarena_cached_v2
    tables = {}
    for ds in PANEL_DATASETS:
        cached = _load_tabarena_cached_v2(ds)
        if cached is not None:
            tables[ds] = cached[0]
    return tables


def patch_text(cell, table, row_idx, max_lines=3):
    """The edit as the original table's own columns: `name: old -> new`, one per line.

    Repair steps, which walk back collateral rather than suppress, are greyed so a
    reader is not told that cleanup is part of the explanation.
    """
    original = None
    if table is not None and row_idx is not None and cell["row"] < len(row_idx):
        original = table.iloc[int(row_idx[cell["row"]])]
    steps = cell["won"].get("trajectory") or []
    lines = []
    for step in steps[:max_lines]:
        name, new = step["column_name"], _val(step["value"])
        old = _val(original[name]) if original is not None and name in original else "?"
        text = rf"{_tex(name)}: {_tex(old)} $\to$ {_tex(new)}"
        lines.append(rf"\rep{{{text}}}" if step.get("repair") else text)
    if len(steps) > max_lines:
        lines.append(rf"\textit{{+{len(steps) - max_lines} more}}")
    return r" \\ ".join(lines)


def quartile_panels(cells):
    """Up to PER_DATASET concepts per dataset per quartile, no concept used twice.

    Sampling is random within the quartile so the panel is not a gallery of the cases
    that worked. A dataset with fewer concepts in a quartile contributes fewer rows;
    that imbalance is real -- effect size is not spread evenly across datasets.
    """
    pool = [c for c in cells
            if c["dataset"] in PANEL_DATASETS and c["ablate"] is not None
            and c["patch"] is not None]
    pool.sort(key=lambda c: -abs(c["ablate"]))
    rng = np.random.default_rng(SEED)
    used, panels = set(), []
    for chunk in np.array_split(np.arange(len(pool)), N_QUARTILES):
        chosen = []
        for ds in PANEL_DATASETS:
            cand = [i for i in chunk if pool[i]["dataset"] == ds and pool[i]["key"] not in used]
            for i in rng.permutation(cand)[:PER_DATASET]:
                used.add(pool[int(i)]["key"])
                chosen.append(pool[int(i)])
        chosen.sort(key=lambda c: (PANEL_DATASETS.index(c["dataset"]), -abs(c["ablate"])))
        span = (abs(pool[chunk[-1]]["ablate"]), abs(pool[chunk[0]]["ablate"]))
        panels.append((chosen, span))
    return panels


def render_panels(cells):
    row_idx, tables = _row_indices(), _raw_tables()
    out = [r"\providecommand{\rep}[1]{\textcolor{black!45}{\textit{#1}}}"]
    for q, (chosen, (lo, hi)) in enumerate(quartile_panels(cells), start=1):
        if not chosen:
            continue
        has_repair = any(s.get("repair") for c in chosen
                         for s in (c["won"].get("trajectory") or []))
        repair_note = (r"Repair edits, which walk back collateral rather than suppress, "
                       r"are \rep{greyed}. " if has_repair else "")
        out += [
            r"\begin{table}[t]", r"\centering",
            rf"\caption{{Patch explanations, quartile Q{q} of the ablation effect "
            rf"({QUARTILE_LABEL[q - 1]}; $|\Delta_{{\mathrm{{ablate}}}}|$ from {lo:.1f} "
            rf"to {hi:.1f} $\times 10^{{-3}}$). Up to {PER_DATASET} concepts per "
            rf"dataset, drawn at random within the quartile and not selected for "
            rf"quality; no concept repeats across quartiles. A dataset contributes "
            rf"fewer rows, or none, where the quartile holds fewer of its concepts. "
            + repair_note +
            rf"Prediction change is in true-class probability ($\times 10^{{-3}}$): "
            rf"what ablating the concept achieves, and what the input patch achieved.}}",
            rf"\label{{tab:patch_q{q}}}", r"\small", r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabularx}{\textwidth}{rllXrrr}", r"\toprule",
            r"Rows & Donor & Recip. & Patch & Act. & Ablate & Patch \\",
        ]
        current = None
        for c in chosen:
            if c["dataset"] != current:
                current = c["dataset"]
                out += [r"\midrule",
                        rf"\multicolumn{{7}}{{l}}{{\textit{{{_tex(current)}}}}} \\"]
            text = patch_text(c, tables.get(c["dataset"]),
                              row_idx.get((c["donor"], c["recipient"], c["dataset"])))
            out.append(
                rf"{c['n_rows']} & {_tex(c['donor'])} $f_{{{c['feat']}}}$ & "
                rf"{_tex(c['recipient'])} & \makecell[tl]{{{text}}} & "
                rf"{c['a_start']:.2f} $\to$ {c['a_final']:.2f} & "
                rf"{c['ablate']:+.1f} & {c['patch']:+.1f} \\")
        out += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(out)


def write_csv(cells, path):
    """Every cell, so the panels never have to stand in for the population."""
    cols = ["donor", "feat", "dataset", "recipient", "n_rows", "row",
            "a_start", "a_final", "suppression_frac", "n_cols_changed",
            "n_suppressors", "n_repairs", "ablate_x1e3", "patch_x1e3", "patch_columns"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for c in cells:
            steps = c["won"].get("trajectory") or []
            w.writerow({
                "donor": c["donor"], "feat": c["feat"], "dataset": c["dataset"],
                "recipient": c["recipient"], "n_rows": c["n_rows"], "row": c["row"],
                "a_start": c["a_start"], "a_final": c["a_final"],
                "suppression_frac": (c["won"].get("best") or {}).get("suppression_frac"),
                "n_cols_changed": c["won"].get("n_cols_changed"),
                "n_suppressors": sum(1 for s in steps if not s.get("repair")),
                "n_repairs": sum(1 for s in steps if s.get("repair")),
                "ablate_x1e3": c["ablate"], "patch_x1e3": c["patch"],
                "patch_columns": "|".join(str(s["column_name"]) for s in steps),
            })


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-paper-copy", action="store_true",
                    help="skip the dual-write to the paper repo")
    args = ap.parse_args()

    cells = load_cells()
    print(f"{len(cells)} cells, {len({c['key'] for c in cells})} distinct concepts, "
          f"{len({c['dataset'] for c in cells})} datasets")

    tex = render_matrix(cells) + "\n\n" + render_panels(cells) + "\n"
    for path in [OUT_DIR / "patch_explanations.tex"] + (
            [] if args.no_paper_copy else [paper_table_path("patch_explanations.tex")]):
        path.write_text(tex)
        print(f"wrote {path}")

    csv_path = OUT_DIR / "patch_explanations.csv"
    write_csv(cells, csv_path)
    print(f"wrote {csv_path} ({len(cells)} rows)")


if __name__ == "__main__":
    main()
