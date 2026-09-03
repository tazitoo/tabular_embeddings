#!/usr/bin/env python3
"""App F.3 tables: how the input-patch pipeline performs over the locked concept set.

Two tables, deliberately separate because they have different denominators and
conflating them is the error the coverage report warns about:

  Table 1  attrition -- every concept in the locked set, and where each one left
           the pipeline. Denominator: the burndown population.
  Table 2  patch character -- what the patches look like, stratified by how much
           of the transfer the concept actually carries. Denominator: only the
           concepts that yielded a qualifying patch.

Table 2 is stratified by the concept's acceptance rank (its position in the
greedy's LOO-ordered accept list at the rows where it was deployed), not by donor
or recipient. Rank is a property of the concept, so the strata partition the
population without overlap -- a concept deployed into three recipients appears
once. It also separates the distributed-transfer cases on its own: concepts that
only reach TabDPT land in the high-rank stratum because that is what being
distributed means, so nothing has to be excluded to keep the table honest.

Run with no arguments; the defaults are the v31 canonical paths.
"""
import csv
import glob
import json
import statistics as st
from pathlib import Path

from scripts._project_root import PROJECT_ROOT
from scripts.paper._paper_repo import paper_table_path

REBUTTAL = PROJECT_ROOT / "output" / "rebuttal"
SWEEP_GLOBS = [str(REBUTTAL / "v31q" / "*.json"), str(REBUTTAL / "v31q_f78.json")]
BURNDOWN = REBUTTAL / "patching_burndown.csv"
TEX_OUT = Path(__file__).parent / "patch_pipeline.tex"
PAPER_TEX_OUT = paper_table_path("patch_pipeline.tex")

OFF_DUMP = REBUTTAL / "off_manifold_concept_dump_all.csv"
FWD = REBUTTAL / "forward_deltas"
# A concept counts as off-manifold at the same threshold the locked set was cut on.
OFF_THRESHOLD = 0.6
N_STRATA = 4


def _median(values):
    """Median over the values that are real numbers; None if none are."""
    vals = [v for v in values if isinstance(v, (int, float)) and v == v]
    return st.median(vals) if vals else None


def off_manifold_fracs():
    """off_frac for every accepted concept, not only the locked set."""
    with open(OFF_DUMP) as fh:
        return {(r["donor"], int(r["feat_id"])): float(r["off_frac"])
                for r in csv.DictReader(fh)}


def coactive_counts(concepts, population):
    """Per concept: median number of co-active concepts drawn from this study's set.

    `n_concepts_at_row` counts every concept the greedy accepted at that row, nearly
    all of which fall outside the locked set and were never patched. Reporting it
    alone invites reading the rank denominator as the study population. This counts
    only the co-active concepts that are themselves in the locked set, so a reader
    can see how many patched concepts share a row -- rows in the high-rank quartiles
    carry many, which is why their patches are not independent observations.
    """
    import numpy as np
    npz_by, out = {}, {}
    for f in sorted(glob.glob(str(FWD / "*" / "*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "selected_features" in z.files:
            npz_by[(str(z["strong_model"]), str(z["weak_model"]),
                    Path(f).stem)] = f
    cache = {}
    for c in concepts:
        donor, _ = c["key"]
        vals = []
        for r in c["patched"]:
            key = (donor, r["recipient"], r["dataset"])
            if key not in npz_by:
                continue
            if key not in cache:
                cache[key] = np.load(npz_by[key], allow_pickle=True)["selected_features"]
            acc = [int(x) for x in cache[key][r["row"]] if x >= 0]
            vals.append(sum(1 for a in acc if (donor, a) in population))
        out[c["key"]] = _median(vals)
    return out


def load_population():
    """The locked concept set: (donor, feat) pairs the sweep was meant to cover."""
    with open(BURNDOWN) as fh:
        return {(r["donor"], int(r["feat_id"])) for r in csv.DictReader(fh)}


def load_concepts():
    """One record per attempted concept, with the fields both tables need.

    `cells` counts only cells that produced candidate rows: a cell that failed
    before searching (env collision, load error) carries no evidence either way,
    and folding it in with the searched-but-empty cells would report an
    infrastructure limit as a scientific negative.
    """
    out = []
    for pattern in SWEEP_GLOBS:
        for path in sorted(glob.glob(pattern)):
            for concept in json.load(open(path)):
                cells = [d for d in concept.get("datasets", []) if d.get("rows")]
                patched = [r for d in cells for r in d["rows"] if r.get("best")]
                out.append({
                    "key": (concept["donor"], int(concept["feat"])),
                    "n_cells": len(cells),
                    "patched": patched,
                    "readout": any(d.get("readout_usable") for d in cells),
                    "rank": _median(r.get("acceptance_rank") for r in patched),
                    "coactive": _median(r.get("n_concepts_at_row") for r in patched),
                    "cols": _median(r.get("n_cols_changed") for r in patched),
                    "suppression": _median(r["best"].get("suppression_frac") for r in patched),
                    "selectivity": _median(r["best"].get("selectivity_ratio") for r in patched),
                    "centrality": _median(r["best"].get("centrality_ratio") for r in patched),
                })
    return out


def attrition(population, concepts):
    """Table 1 rows: (label, count, is_loss). Counts close by construction."""
    attempted = {c["key"] for c in concepts}
    searched = [c for c in concepts if c["n_cells"]]
    patched = [c for c in searched if c["patched"]]
    with_readout = [c for c in patched if c["readout"]]
    return [
        ("Starting concept universe", len(population), False),
        ("not dispatched", len(population - attempted), True),
        ("Attempted", len(attempted), False),
        ("never searchable: conflicting TabICL versions required",
         len(attempted) - len(searched), True),
        ("Searched", len(searched), False),
        ("no qualifying patch found", len(searched) - len(patched), True),
        ("Qualifying patch", len(patched), False),
        ("CARTE as recipient, refit induces noise larger than effect",
         len(patched) - len(with_readout), True),
        ("Patch with measured recipient prediction", len(with_readout), False),
    ]


def visible(rows):
    """Drop losses that did not happen, and any stage they leave as a repeat.

    The stage stays in attrition() so the accounting is always complete; hiding a
    zero here is cosmetic. If a concept ever does go undispatched the row returns,
    rather than that loss being silently absorbed into the next stage's drop.
    """
    out = []
    for label, n, is_loss in rows:
        if is_loss and n == 0:
            continue
        if not is_loss and out and not out[-1][2] and out[-1][1] == n:
            continue          # same count, no loss between: keep the earlier name
        out.append((label, n, is_loss))
    # Intermediate running totals are implied by the losses either side of them, so
    # only the universe and the final surviving count are shown. The losses still sum
    # to the difference between the two, which is what makes the table checkable.
    return [r for i, r in enumerate(out) if r[2] or i == 0 or i == len(out) - 1]


def strata_rows(concepts):
    """Table 2 rows, over the concepts Table 1 ends on.

    Restricted to concepts with a measured recipient prediction, so this table
    describes exactly the set Table 1 terminates at and the two reconcile without
    explanation. The 11 CARTE-only concepts dropped here do have valid donor-side
    patches -- every column below is donor-side and never touches the recipient --
    but including them shifts no median by more than 0.05, so nothing is bought by
    carrying a second denominator.
    """
    import numpy as np
    patched = [c for c in concepts
               if c["patched"] and c["rank"] is not None and c["readout"]]
    ranks = [c["rank"] for c in patched]
    # Quartiles of the observed rank, not hand-set thresholds: no cut point in the
    # pipeline corresponds to a label like "dominant", so any fixed edge would be
    # invented. Ties make the groups only approximately equal.
    edges = [float(np.percentile(ranks, 100 * i / N_STRATA))
             for i in range(1, N_STRATA)]
    rows, lo = [], float("-inf")
    for i, hi in enumerate(edges + [float("inf")]):
        group = [c for c in patched if lo < c["rank"] <= hi]
        if lo == float("-inf"):
            band = f"rank <= {hi:.0f}"
        elif hi == float("inf"):
            band = f"rank > {lo:.0f}"
        else:
            band = f"{lo:.0f} < rank <= {hi:.0f}"
        if group:
            rows.append((f"Q{i + 1}", band, group))
        lo = hi
    rows.append(("All", "", patched))
    return rows


BLOCKS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
N_BINS = 10


def suppression_hist(group, scale):
    """Counts per suppression bin over [0,1], as a fraction of the group."""
    import numpy as np
    v = [c["suppression"] for c in group if c["suppression"] is not None]
    if not v:
        return []
    return list(np.histogram(v, bins=N_BINS, range=(0, 1))[0] / len(v) / scale)


def hist_scale(rows):
    """Tallest bin anywhere, so every row shares one vertical scale."""
    import numpy as np
    tallest = 0.0
    for *_, group in rows:
        v = [c["suppression"] for c in group if c["suppression"] is not None]
        if v:
            tallest = max(tallest, (np.histogram(v, bins=N_BINS, range=(0, 1))[0]
                                    / len(v)).max())
    return tallest or 1.0


def extinguished(group):
    """Fraction whose activation the patch drove to exactly zero.

    Reported instead of a median because the distribution is bimodal with nothing
    between 0.9 and 1.0: the median is then a step function of this fraction, reading
    1.00 whenever it clears 50% and dropping into the lower cluster when it does not.
    """
    v = [c["suppression"] for c in group if c["suppression"] is not None]
    return sum(1 for x in v if x == 1.0) / len(v) if v else None


def _tex_band(band):
    """Inequalities need math mode in LaTeX."""
    return band.replace("<=", r"$\leq$").replace("<", "$<$").replace(">", "$>$")


def _fmt(value, spec):
    return "--" if value is None else format(value, spec)


def render_text(population, concepts, offc):
    lines = []
    total = len(population)
    lines.append(f"TABLE 1  Causal patching of off-manifold transferred concepts "
                 f"(off-manifold fraction 0.6--0.8): attrition over {total} concepts\n")
    lines.append(f"  {"Stage":<60}{"N":>6}{"% of set":>10}")
    lines.append("  " + "-" * 76)
    for label, n, is_loss in visible(attrition(population, concepts)):
        name = label
        lines.append(f"  {name:<60}{n:>6}{100 * n / total:>9.1f}%")

    rows = strata_rows(concepts)
    n_patched = len(rows[-1][2])
    lines.append(f"\n\nTABLE 2  Patch character over those {n_patched} concepts\n")
    scale = hist_scale(rows)
    head = (f"  {'Quartile':<10}{'transfer rank':<20}{'N':>5}  "
            f"{'suppression':<13}{'ext.':>6}{'cols':>7}{'selectivity':>13}"
            f"{'centrality':>12}")
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))
    for label, band, group in rows:
        if label == "All":
            lines.append("  " + "-" * (len(head) - 2))
        lines.append(
            f"  {label:<10}{band:<20}{len(group):>5}  "
            f"{''.join(BLOCKS[min(7, int(round(h * 7)))] for h in suppression_hist(group, scale)):<13}"
            f"{_fmt(extinguished(group) and 100 * extinguished(group), '>5.0f')}%"
            f"{_fmt(_median(c['cols'] for c in group), '>7.1f')}"
            f"{_fmt(_median(c['selectivity'] for c in group), '>13.1f')}"
            f"{_fmt(_median(c['centrality'] for c in group), '>12.2f')}")
    return "\n".join(lines)


def render_tex(population, concepts, offc):
    """Two independent tables, each with a caption short enough to read.

    The mechanism prose that a reader needs -- why the two losses are not negative
    results, what the ratios mean, why high-rank quartiles are not independent --
    is body text, not caption text, so it is emitted as comments at the top of the
    file for pasting into the section rather than crammed above the rules.
    """
    rows = strata_rows(concepts)
    scale = hist_scale(rows)
    out = [
        r"% ---------------------------------------------------------------------",
        r"% BODY TEXT (not caption).",
        r"%",
        r"% Attrition table: the starting universe is the 335 concepts drawn from the",
        r"% 6088 accepted across the transfer sweep by two criteria -- off-manifold",
        r"% fraction in [0.6, 0.8], the share of the concept's mapped direction lying",
        r"% outside the recipient's principal subspace (taken at 90 percent of",
        r"% variance), and acceptance count in [200, 499]. Every concept is accounted",
        r"% for, and neither loss is a negative result. One concept's donor and",
        r"% recipient require mutually incompatible TabICL versions, so it could never",
        r"% be searched. Eleven concepts transfer only into CARTE, the one recipient",
        r"% refit per dataset rather than frozen; rebuilding its tail to read a patch",
        r"% moves predictions by 0.032 against a median transfer effect of 0.012 on",
        r"% those rows, so the measurement noise exceeds the quantity measured and the",
        r"% readout is not attempted. That is missing measurement, not a measured",
        r"% null. The donor-side patch is unaffected in both cases: whether an input",
        r"% edit suppresses the concept never involves the recipient.",
        r"%",
        r"% Character table: quartiles are of the observed rank distribution, not",
        r"% fixed thresholds. Suppression is bimodal -- a patch either drives the",
        r"% concept's activation to exactly zero or stalls partway, with almost",
        r"% nothing between -- so the extinguished rate is reported rather than a",
        r"% median, which would be a step function of that rate: it reads 1.00",
        r"% whenever the rate clears 50 percent and drops into the lower cluster",
        r"% when it does not. Transfer rank is taken over every concept the greedy",
        r"% accepted at a row, nearly all of which fall outside the locked set. The",
        r"% concepts in a row's accept list that are themselves in the locked set",
        r"% number 1 at the Q1 median and 16 at the Q4 median, so the high-rank",
        r"% quartiles concentrate on shared rows and their patches are not independent",
        r"% observations.",
        r"% ---------------------------------------------------------------------",
        r"",
        r"\begin{table}[t]", r"\centering",
        r"\caption{Attrition for causal patching of off-manifold transferred "
        r"concepts.}",
        r"\label{tab:patch_attrition}", r"\small",
        r"\begin{tabular}{lr}", r"\toprule",
        r"Stage & N \\", r"\midrule",
    ]
    for label, n, is_loss in visible(attrition(population, concepts)):
        out.append(rf"{label} & {n} \\")
    out += [
        r"\bottomrule", r"\end{tabular}", r"\end{table}", r"",
        r"\ifdefined\suppbin\else\newlength{\suppbh}\setlength{\suppbh}{4.5mm}\fi",
        r"\providecommand{\suppbin}[1]{\rule{1.5mm}{#1\suppbh}\hspace{0.25mm}}",
        r"\providecommand{\supphist}[1]{%",
        r"  \raisebox{0pt}[4.5mm][0pt]{\textcolor{black!55}{#1}}}",
        r"",
        r"\begin{table}[t]", r"\centering",
        r"\caption{Patch character over the "
        rf"{len(rows[-1][2])} concepts with a measured recipient prediction, by "
        r"quartile of the concept's transfer rank. `ext.' is the fraction whose "
        r"activation the patch drove to exactly zero; histograms are the per-concept "
        r"suppression over $[0,1]$ in ten bins, common vertical scale. `select.' is "
        r"how many times more the target concept moved than any other concept at that "
        r"row; `centr.' is the patched row's centrality over its original. Medians "
        r"throughout.}",
        r"\label{tab:patch_character}", r"\small",
        r"\begin{tabular}{llrlrrrr}", r"\toprule",
        r"Quartile & transfer rank & N & suppression & ext. "
        r"& cols & select. & centr. \\",
        r"\midrule",
    ]
    for label, band, group in rows:
        if label == "All":
            out.append(r"\midrule")
        bars = "".join(rf"\suppbin{{{h:.3f}}}" for h in suppression_hist(group, scale))
        out.append(
            rf"{label} & {_tex_band(band)} & {len(group)} "
            rf"& \supphist{{{bars}}} "
            rf"& {_fmt(extinguished(group) and 100 * extinguished(group), '.0f')}\% "
            rf"& {_fmt(_median(c['cols'] for c in group), '.1f')} "
            rf"& {_fmt(_median(c['selectivity'] for c in group), '.1f')} "
            rf"& {_fmt(_median(c['centrality'] for c in group), '.2f')} \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def main():
    population = load_population()
    concepts = load_concepts()
    unexpected = {c["key"] for c in concepts} - population
    if unexpected:
        raise SystemExit(f"results outside the locked set: {sorted(unexpected)}")
    offc = coactive_counts(concepts, population)
    print(render_text(population, concepts, offc))
    tex = render_tex(population, concepts, offc)
    TEX_OUT.write_text(tex)
    print(f"\nwrote {TEX_OUT.relative_to(PROJECT_ROOT)} ({len(tex)} bytes)")
    PAPER_TEX_OUT.write_text(tex)
    print(f"  -> also wrote {PAPER_TEX_OUT}")


if __name__ == "__main__":
    main()
