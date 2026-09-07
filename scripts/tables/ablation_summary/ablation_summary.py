#!/usr/bin/env python3
"""
Section 4 ablation summary: mean gap closed when ablating unmatched
concepts, by strong model. Rendered as Table 1 in the paper draft.

Reads all ablation sweep NPZ files, groups by which model is "strong"
(the one being ablated), and reports mean/median gap_closed, mean concepts
ablated (K), and the per-concept acceptance rate (acc), parallel to the
transfer table.

acc = (concepts ablated) / (concepts tried), pooled over strong-win rows.
"Tried" is the candidate set the greedy ranks per row: unmatched, firing,
positive-importance concepts (row_feature_drops > 0 implies the concept
fires), read from output/perrow_importance/<strong>/<dataset>.npz. This is the
ablation analogue of the transfer table's firing-unmatched acceptance pool.

Usage:
    python -m scripts.tables.ablation_summary.ablation_summary
"""

from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.paper._paper_repo import paper_table_path
from scripts.intervention.ablation_sweep import get_unmatched_features, IMPORTANCE_DIR

SWEEP_DIR = PROJECT_ROOT / "output" / "ablation_sweep_tols"
RANDOM_DIR = PROJECT_ROOT / "output" / "ablation_sweep_random_tols"
OUTPUT_TEX = Path(__file__).parent / "ablation_summary.tex"
PAPER_OUTPUT_TEX = paper_table_path("section4_summary.tex")

# Display name mapping
DISPLAY = {
    "tabpfn": "TabPFN", "mitra": "Mitra", "tabicl": "TabICL",
    "tabicl_v2": "TabICL-v2", "tabdpt": "TabDPT", "carte": "CARTE",
}

# Exclude from main table
EXCLUDE = {"hyperfast", "tabula8b"}


def _acc_counts(data, strong, weak, dataset):
    """(n_used, n_tried) over strong-win rows, or None if importance missing.

    used  = concepts ablated (optimal_k).
    tried = candidate concepts the greedy ranked per row (unmatched, firing,
            positive importance). Mirrors `ranked` in ablation_sweep.py.
    """
    imp_path = IMPORTANCE_DIR / strong / f"{dataset}.npz"
    if not imp_path.exists() or "strong_wins" not in data or "optimal_k" not in data:
        return None
    imp = np.load(imp_path, allow_pickle=True)
    drops = np.asarray(imp["row_feature_drops"])           # (n_query, n_feat)
    feat_idx = [int(x) for x in imp["feature_indices"]]
    unmatched = {int(x) for x in get_unmatched_features(strong, weak)}
    cols = [i for i, fi in enumerate(feat_idx) if fi in unmatched]
    if not cols:
        return None
    wins = np.asarray(data["strong_wins"], dtype=bool)
    k = np.asarray(data["optimal_k"])
    if len(wins) != drops.shape[0]:
        return None
    n_tried = int((drops[np.ix_(wins, cols)] > 0).sum())
    n_used = int(k[wins].sum())
    return n_used, n_tried


def load_ablation_results(sweep_dir, compute_acc=False):
    """Load ablation NPZ files.

    Returns list of (strong_model, pair_dir_name, dataset, gc, mean_k, acc)
    where acc is (n_used, n_tried) or None (only computed when compute_acc).
    """
    results = []
    for pair_dir in sorted(sweep_dir.iterdir()):
        if not pair_dir.is_dir():
            continue
        parts = pair_dir.name.split("_vs_")
        if len(parts) != 2:
            continue
        model_a, model_b = parts
        if model_a in EXCLUDE or model_b in EXCLUDE:
            continue

        for npz_path in sorted(pair_dir.glob("*.npz")):
            try:
                data = np.load(npz_path, allow_pickle=True)
            except Exception:
                continue

            strong = str(data["strong_model"]) if "strong_model" in data else None
            if strong is None:
                continue

            gc = float(data["mean_gap_closed"]) if "mean_gap_closed" in data else None
            if gc is None:
                continue

            n_strong = int(data["n_strong_wins"]) if "n_strong_wins" in data else 0
            if n_strong == 0:
                continue

            mean_k = float(data["mean_optimal_k"]) if "mean_optimal_k" in data else None
            dataset = npz_path.stem

            acc = None
            if compute_acc:
                weak = model_b if strong == model_a else model_a
                acc = _acc_counts(data, strong, weak, dataset)

            results.append((strong, pair_dir.name, dataset, gc, mean_k, acc))

    return results


def main():
    trained = load_ablation_results(SWEEP_DIR, compute_acc=True)
    print(f"Trained: {len(trained)} entries")

    # Load random baseline and index by (pair, dataset)
    random_gc = {}
    random_k = {}
    if RANDOM_DIR.exists():
        random_results = load_ablation_results(RANDOM_DIR)
        print(f"Random:  {len(random_results)} entries")
        for strong, pair, dataset, gc, mean_k, _ in random_results:
            random_gc[(pair, dataset)] = gc
            if mean_k is not None:
                random_k[(pair, dataset)] = mean_k
    else:
        print("WARNING: no random baseline directory, gc_R/K_R will be missing")

    # Group by strong model
    by_model_gc = defaultdict(list)
    by_model_k = defaultdict(list)
    by_model_random = defaultdict(list)
    by_model_random_k = defaultdict(list)
    by_model_used = defaultdict(int)
    by_model_tried = defaultdict(int)
    acc_missing = 0
    for strong, pair, dataset, gc, mean_k, acc in trained:
        by_model_gc[strong].append(gc)
        if mean_k is not None:
            by_model_k[strong].append(mean_k)
        rgc = random_gc.get((pair, dataset))
        if rgc is not None:
            by_model_random[strong].append(rgc)
        rk = random_k.get((pair, dataset))
        if rk is not None:
            by_model_random_k[strong].append(rk)
        if acc is not None:
            by_model_used[strong] += acc[0]
            by_model_tried[strong] += acc[1]
        else:
            acc_missing += 1
    if acc_missing:
        print(f"WARNING: acc unavailable for {acc_missing}/{len(trained)} (pair, dataset) entries")

    def acc_of(used, tried):
        return (used / tried) if tried else None

    # Sort by N descending
    model_stats = []
    for model, gcs in by_model_gc.items():
        display = DISPLAY.get(model, model)
        ks = by_model_k.get(model, [])
        randoms = by_model_random.get(model, [])
        random_ks = by_model_random_k.get(model, [])
        model_stats.append({
            "key": model,
            "display": display,
            "n": len(gcs),
            "mean_gc": np.mean(gcs),
            "std_gc": np.std(gcs),
            "mean_k": np.mean(ks) if ks else 0,
            "std_k": np.std(ks) if ks else 0,
            "mean_gc_r": np.mean(randoms) if randoms else None,
            "std_gc_r": np.std(randoms) if randoms else None,
            "n_random": len(randoms),
            "mean_k_r": np.mean(random_ks) if random_ks else None,
            "std_k_r": np.std(random_ks) if random_ks else None,
            "acc": acc_of(by_model_used[model], by_model_tried[model]),
        })
    model_stats.sort(key=lambda x: -x["n"])

    overall_acc = acc_of(sum(by_model_used.values()), sum(by_model_tried.values()))

    # Print summary
    print(f"\n{'Model':<15s} {'N':>4s} {'gc':>12s} {'gc_R':>12s} {'K':>12s} {'K_R':>12s} {'acc':>7s}")
    print("-" * 82)
    for s in model_stats:
        gc_r_str = (f"{s['mean_gc_r']:.3f}±{s['std_gc_r']:.3f}"
                    if s["mean_gc_r"] is not None else "---")
        k_r_str = (f"{s['mean_k_r']:.1f}±{s['std_k_r']:.1f}"
                   if s["mean_k_r"] is not None else "---")
        acc_str = f"{s['acc']:.3f}" if s["acc"] is not None else "---"
        print(f"{s['display']:<15s} {s['n']:>4d} "
              f"{s['mean_gc']:.3f}±{s['std_gc']:.3f} "
              f"{gc_r_str:>12s} "
              f"{s['mean_k']:.1f}±{s['std_k']:.1f}".ljust(56) +
              f"{k_r_str:>12s} {acc_str:>7s}")
    all_gcs = [gc for _, _, _, gc, _, _ in trained]
    all_ks = [k for _, _, _, _, k, _ in trained if k is not None]
    all_randoms = [random_gc[(pair, ds)] for _, pair, ds, _, _, _ in trained
                   if (pair, ds) in random_gc]
    all_random_ks = [random_k[(pair, ds)] for _, pair, ds, _, _, _ in trained
                     if (pair, ds) in random_k]
    print("-" * 82)
    acc_o_str = f"{overall_acc:.3f}" if overall_acc is not None else "---"
    print(f"{'Overall':<15s} {len(all_gcs):>4d} "
          f"{np.mean(all_gcs):.3f}±{np.std(all_gcs):.3f} "
          f"{np.mean(all_randoms):.3f}±{np.std(all_randoms):.3f} "
          f"{np.mean(all_ks):.1f}±{np.std(all_ks):.1f}".ljust(56) +
          f"{np.mean(all_random_ks):.1f}±{np.std(all_random_ks):.1f} {acc_o_str:>7s}")

    # Generate LaTeX
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Ablation results by strong model, sorted by $N$. "
        r"$gc/gc_R$ = mean gap closed using trained/random SAEs. "
        r"$K/K_R$ = mean concepts ablated using trained/random SAEs. "
        r"\emph{acc} = fraction of candidate concepts accepted during ablation "
        r"(concepts ablated / firing unmatched concepts considered).}"
    )
    lines.append(r"\label{tab:ablation_summary}")
    lines.append(r"\begin{tabular}{lrlllll}")
    lines.append(r"\toprule")
    lines.append(r"Model (when strong) & $N$ & gc & $gc_R$ & $K$ & $K_R$ & acc \\")
    lines.append(r"\midrule")

    for s in model_stats:
        gc_r_str = (f"{s['mean_gc_r']:.2f} $\\pm$ {s['std_gc_r']:.2f}"
                    if s["mean_gc_r"] is not None else "---")
        k_r_str = (f"{s['mean_k_r']:.1f} $\\pm$ {s['std_k_r']:.1f}"
                   if s["mean_k_r"] is not None else "---")
        acc_str = f"{s['acc']:.3f}" if s["acc"] is not None else "---"
        lines.append(
            f"{s['display']} & {s['n']} & "
            f"{s['mean_gc']:.2f} $\\pm$ {s['std_gc']:.2f} & "
            f"{gc_r_str} & "
            f"{s['mean_k']:.1f} $\\pm$ {s['std_k']:.1f} & "
            f"{k_r_str} & {acc_str} \\\\"
        )

    lines.append(r"\midrule")
    gc_r_overall = (f"{np.mean(all_randoms):.2f} $\\pm$ {np.std(all_randoms):.2f}"
                    if all_randoms else "---")
    k_r_overall = (f"{np.mean(all_random_ks):.1f} $\\pm$ {np.std(all_random_ks):.1f}"
                   if all_random_ks else "---")
    acc_o_tex = f"{overall_acc:.3f}" if overall_acc is not None else "---"
    lines.append(
        f"Overall & {len(all_gcs)} & "
        f"{np.mean(all_gcs):.2f} $\\pm$ {np.std(all_gcs):.2f} & "
        f"{gc_r_overall} & "
        f"{np.mean(all_ks):.1f} $\\pm$ {np.std(all_ks):.1f} & "
        f"{k_r_overall} & {acc_o_tex} \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    OUTPUT_TEX.write_text(tex + "\n")
    print(f"\nSaved to {OUTPUT_TEX}")
    PAPER_OUTPUT_TEX.write_text(tex + "\n")
    print(f"  → also wrote {PAPER_OUTPUT_TEX}")
    print(tex)


if __name__ == "__main__":
    main()
