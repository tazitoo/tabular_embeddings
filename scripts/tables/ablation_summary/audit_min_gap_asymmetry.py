"""Quantify the min_gap=0.01 row-skip asymmetry between
ablation_sweep_tols (new, with skip) and ablation_sweep_random (old,
without skip).

For each per-row tols npz, count the fraction of rows where the trained
sweep auto-credited gap_closed=1.0 via the min_gap=0.01 shortcut
(optimal_k == 0 AND gap_closed == 1.0). Report per-pair and overall
fractions; estimate worst-case inflation of the trained-minus-random
delta under the approximation that random would have scored
(1.0 - 0.6) = 0.4 on those rows (so the trained excess ≈ 0.6 per row).
"""
from pathlib import Path
import numpy as np
from scripts._project_root import PROJECT_ROOT

TOLS = PROJECT_ROOT / "output" / "ablation_sweep_tols"


def audit_pair(pair_dir: Path) -> dict:
    fractions = []
    n_rows_total = 0
    n_agreed_total = 0
    for npz in sorted(pair_dir.glob("*.npz")):
        try:
            d = np.load(npz, allow_pickle=True)
        except Exception:
            continue
        if "optimal_k" not in d or "gap_closed" not in d:
            continue
        ok = d["optimal_k"]
        gc = d["gap_closed"]
        # Restrict to strong-wins rows if available — that is the
        # population the paper's headline number is taken over.
        if "strong_wins" in d:
            sw = d["strong_wins"]
            if sw.dtype == bool:
                mask = sw
            else:
                mask = sw.astype(bool)
            ok = ok[mask]
            gc = gc[mask]
        agreed = (ok == 0) & (np.isclose(gc, 1.0))
        if len(agreed):
            fractions.append(float(agreed.mean()))
            n_rows_total += len(agreed)
            n_agreed_total += int(agreed.sum())
    return {
        "pair": pair_dir.name,
        "n_datasets": len(fractions),
        "mean_agreed_fraction": float(np.mean(fractions)) if fractions else 0.0,
        "overall_agreed_fraction": (n_agreed_total / n_rows_total) if n_rows_total else 0.0,
        "n_rows": n_rows_total,
        "n_agreed": n_agreed_total,
    }


def main():
    rows = []
    for pair_dir in sorted(TOLS.iterdir()):
        if not pair_dir.is_dir() or "_vs_" not in pair_dir.name:
            continue
        rows.append(audit_pair(pair_dir))

    rows.sort(key=lambda r: -r["overall_agreed_fraction"])
    print(f"{'pair':<30} {'mean_frac':>10} {'overall':>10} "
          f"{'n_agreed':>10} {'n_rows':>10}")
    for r in rows:
        print(f"{r['pair']:<30} {r['mean_agreed_fraction']:>10.4f} "
              f"{r['overall_agreed_fraction']:>10.4f} "
              f"{r['n_agreed']:>10d} {r['n_rows']:>10d}")

    max_frac = max((r["overall_agreed_fraction"] for r in rows), default=0.0)
    overall_n_agreed = sum(r["n_agreed"] for r in rows)
    overall_n = sum(r["n_rows"] for r in rows)
    overall_frac = overall_n_agreed / overall_n if overall_n else 0.0

    print()
    print(f"Overall strong-wins rows: {overall_n}")
    print(f"Overall agreed rows: {overall_n_agreed}")
    print(f"Overall agreed fraction: {overall_frac:.4f}")
    print(f"Max pair-level agreed fraction: {max_frac:.4f}")
    # Worst-case asymmetry on headline ≈ agreed_fraction * 0.6
    print(f"Worst-case headline inflation (0.6 per-row): "
          f"max pair {max_frac * 0.6:.4f}, overall {overall_frac * 0.6:.4f}")


if __name__ == "__main__":
    main()
