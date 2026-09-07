#!/usr/bin/env python3
"""
Section 4 transfer summary: mean gap closed when injecting unmatched
concepts from strong into weak, by *weak* (recipient) model.

Companion to ``ablation_summary.py``. Ablation groups by the model
being intervened on; for transfer, that is the *weak* model receiving
injected concepts. gc, K, and acceptance all describe the recipient's
response to the injection.

Δ is computed only on (pair, dataset) entries present in both trained
and random directories.

Usage:
    python -m scripts.tables.transfer_summary.transfer_summary
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.paper._paper_repo import paper_table_path
from scripts.tables._rank_depth_acc import rank_depth_counts

SWEEP_DIR = PROJECT_ROOT / "output" / "transfer_global_mnnp90_trained_tols"
RANDOM_DIR = PROJECT_ROOT / "output" / "transfer_global_mnnp90_random"
OUTPUT_TEX = Path(__file__).parent / "transfer_summary.tex"
PAPER_OUTPUT_TEX = paper_table_path("transfer_summary.tex")

DISPLAY = {
    "tabpfn": "TabPFN", "mitra": "Mitra", "tabicl": "TabICL",
    "tabicl_v2": "TabICL-v2", "tabdpt": "TabDPT", "carte": "CARTE",
}

EXCLUDE = {"hyperfast", "tabula8b"}


def load_transfer_results(sweep_dir, compute_acc=False):
    """Load all transfer NPZ files from a sweep directory.

    Returns list of (weak_model, pair_dir_name, dataset, gc, mean_k, n_tried, n_accepted)
    tuples, keeping only entries with at least one strong-win row. The weak model is
    the recipient of the injection — gc/K all describe its response. acc = n_accepted /
    n_tried is the rank-depth acceptance (see scripts.tables._rank_depth_acc): tried is
    how far down the donor's importance ranking the greedy reached before stopping.
    """
    results = []
    if not sweep_dir.exists():
        return results
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

            weak = str(data["weak_model"]) if "weak_model" in data else None
            if weak is None:
                continue

            gc = float(data["mean_gap_closed"]) if "mean_gap_closed" in data else None
            if gc is None:
                continue

            n_strong = int(data["n_strong_wins"]) if "n_strong_wins" in data else 0
            if n_strong == 0:
                continue

            mean_k = float(data["mean_optimal_k"]) if "mean_optimal_k" in data else None
            dataset = npz_path.stem

            n_tried, n_accepted = 0, 0
            if compute_acc:
                strong = str(data["strong_model"]) if "strong_model" in data else None
                rd = (rank_depth_counts(data, strong, weak, dataset, use_abs=True)
                      if strong is not None else None)
                if rd is not None:
                    n_accepted, n_tried = rd

            results.append((weak, pair_dir.name, dataset, gc, mean_k,
                            n_tried, n_accepted))

    return results


def main():
    trained = load_transfer_results(SWEEP_DIR, compute_acc=True)
    print(f"Trained: {len(trained)} entries")

    random_gc = {}
    random_k = {}
    if RANDOM_DIR.exists():
        random_results = load_transfer_results(RANDOM_DIR)
        print(f"Random:  {len(random_results)} entries")
        for weak, pair, dataset, gc, mean_k, _, _ in random_results:
            random_gc[(pair, dataset)] = gc
            if mean_k is not None:
                random_k[(pair, dataset)] = mean_k
    else:
        print("WARNING: no random baseline directory, gc_R/K_R will be missing")

    by_model_gc = defaultdict(list)
    by_model_k = defaultdict(list)
    by_model_random = defaultdict(list)
    by_model_random_k = defaultdict(list)
    by_model_tried = defaultdict(int)
    by_model_accepted = defaultdict(int)
    for weak, pair, dataset, gc, mean_k, n_tried, n_accepted in trained:
        by_model_gc[weak].append(gc)
        if mean_k is not None:
            by_model_k[weak].append(mean_k)
        by_model_tried[weak] += n_tried
        by_model_accepted[weak] += n_accepted
        rgc = random_gc.get((pair, dataset))
        if rgc is not None:
            by_model_random[weak].append(rgc)
        rk = random_k.get((pair, dataset))
        if rk is not None:
            by_model_random_k[weak].append(rk)

    model_stats = []
    for model, gcs in by_model_gc.items():
        display = DISPLAY.get(model, model)
        ks = by_model_k.get(model, [])
        randoms = by_model_random.get(model, [])
        random_ks = by_model_random_k.get(model, [])
        tried = by_model_tried[model]
        accepted = by_model_accepted[model]
        acc_rate = accepted / tried if tried > 0 else None
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
            "n_tried": tried,
            "n_accepted": accepted,
            "acc_rate": acc_rate,
        })
    model_stats.sort(key=lambda x: -x["n"])

    print(f"\n{'Model':<15s} {'N':>4s} {'gc':>12s} {'gc_R':>12s} "
          f"{'K':>10s} {'K_R':>10s} {'acc':>8s}")
    print("-" * 85)
    for s in model_stats:
        gc_r_str = (f"{s['mean_gc_r']:.3f}±{s['std_gc_r']:.3f}"
                    if s["mean_gc_r"] is not None else "---")
        k_r_str = (f"{s['mean_k_r']:.1f}±{s['std_k_r']:.1f}"
                   if s["mean_k_r"] is not None else "---")
        acc_str = f"{s['acc_rate']:.3f}" if s["acc_rate"] is not None else "---"
        print(f"{s['display']:<15s} {s['n']:>4d} "
              f"{s['mean_gc']:.3f}±{s['std_gc']:.3f} "
              f"{gc_r_str:>12s} "
              f"{s['mean_k']:>4.1f}±{s['std_k']:<4.1f} "
              f"{k_r_str:>10s} "
              f"{acc_str:>8s}")
    all_gcs = [gc for _, _, _, gc, _, _, _ in trained]
    all_ks = [k for _, _, _, _, k, _, _ in trained if k is not None]
    all_tried = sum(t for _, _, _, _, _, t, _ in trained)
    all_accepted = sum(a for _, _, _, _, _, _, a in trained)
    all_randoms = [random_gc[(pair, ds)] for _, pair, ds, _, _, _, _ in trained
                   if (pair, ds) in random_gc]
    all_random_ks = [random_k[(pair, ds)] for _, pair, ds, _, _, _, _ in trained
                     if (pair, ds) in random_k]
    overall_acc = all_accepted / all_tried if all_tried > 0 else None
    print("-" * 85)
    print(f"{'Overall':<15s} {len(all_gcs):>4d} "
          f"{np.mean(all_gcs):.3f}±{np.std(all_gcs):.3f} "
          f"{np.mean(all_randoms):.3f}±{np.std(all_randoms):.3f} "
          f"{np.mean(all_ks):>4.1f}±{np.std(all_ks):<4.1f} "
          f"{np.mean(all_random_ks):>4.1f}±{np.std(all_random_ks):<4.1f} "
          f"{overall_acc:>8.3f}")

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Transfer results by weak model, sorted by $N$. "
        r"$gc/gc_R$ = mean gap closed using trained/random SAEs. "
        r"$K/K_R$ = mean concepts injected using trained/random SAEs. "
        r"\emph{acc} = concepts accepted / concepts tried, where \emph{tried} is the "
        r"rank depth reached in the donor's importance-ordered candidate list (the "
        r"position of the deepest accepted concept).}"
    )
    lines.append(r"\label{tab:transfer_summary}")
    lines.append(r"\begin{tabular}{lrlllll}")
    lines.append(r"\toprule")
    lines.append(r"Model (when weak) & $N$ & gc & $gc_R$ & $K$ & $K_R$ & acc \\")
    lines.append(r"\midrule")

    for s in model_stats:
        gc_r_str = (f"{s['mean_gc_r']:.2f} $\\pm$ {s['std_gc_r']:.2f}"
                    if s["mean_gc_r"] is not None else "---")
        k_r_str = (f"{s['mean_k_r']:.1f} $\\pm$ {s['std_k_r']:.1f}"
                   if s["mean_k_r"] is not None else "---")
        acc_str = f"{s['acc_rate']:.3f}" if s["acc_rate"] is not None else "---"
        lines.append(
            f"{s['display']} & {s['n']} & "
            f"{s['mean_gc']:.2f} $\\pm$ {s['std_gc']:.2f} & "
            f"{gc_r_str} & "
            f"{s['mean_k']:.1f} $\\pm$ {s['std_k']:.1f} & "
            f"{k_r_str} & "
            f"{acc_str} \\\\"
        )

    lines.append(r"\midrule")
    gc_r_overall = (f"{np.mean(all_randoms):.2f} $\\pm$ {np.std(all_randoms):.2f}"
                    if all_randoms else "---")
    k_r_overall = (f"{np.mean(all_random_ks):.1f} $\\pm$ {np.std(all_random_ks):.1f}"
                   if all_random_ks else "---")
    acc_overall = f"{overall_acc:.3f}" if overall_acc is not None else "---"
    lines.append(
        f"Overall & {len(all_gcs)} & "
        f"{np.mean(all_gcs):.2f} $\\pm$ {np.std(all_gcs):.2f} & "
        f"{gc_r_overall} & "
        f"{np.mean(all_ks):.1f} $\\pm$ {np.std(all_ks):.1f} & "
        f"{k_r_overall} & "
        f"{acc_overall} \\\\"
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
