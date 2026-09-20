"""Why is ablation's acceptance rate lower than transfer's?

Hypothesis: ablation ranks candidates by LOO drop (largest first) but must hit a
bounded target (the weak model's per-row loss). With squared distance, removing a
concept whose drop exceeds 2x the remaining gap lands farther from the target
than the start, so it is rejected and the greedy walks deeper.

For every strong-win classification row in the forward ablation sweep:
  gap      = weak_loss - strong_loss (loss units; ablation raises strong_loss)
  ranked   = firing unmatched concepts with drop > 0, descending (the sweep's order)
  skipped  = ranked above the deepest accepted concept, not accepted
Reports: fraction of skipped concepts with drop > 2*gap (overshoot), the same for
accepted concepts, and acceptance rate by initial-gap tercile.

Second report (both sweeps, all strong-win rows with at least one step): batches
consumed per row and concepts accepted per batch. Both sweeps accept the single
best subset of at most 3 from each batch of 12 candidate entries and then move
on; transfer's entries come in +/- pairs, so one transfer batch spans 6 concepts
where an ablation batch spans 12. Rank-depth acc counts concepts, so its ceiling
is 3/6 for transfer and 3/12 for ablation.
"""
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._project_root import PROJECT_ROOT
from scripts.intervention.ablation_sweep import get_unmatched_features

SWEEP = PROJECT_ROOT / "output" / "ablation_sweep_tols"
IMP = PROJECT_ROOT / "output" / "perrow_importance"


def main():
    tot = defaultdict(int)
    gap_rows = []  # (gap, n_used, n_tried)
    for pair_dir in sorted(SWEEP.iterdir()):
        if not pair_dir.is_dir() or "_vs_" not in pair_dir.name:
            continue
        for npz in sorted(pair_dir.glob("*.npz")):
            d = np.load(npz, allow_pickle=True)
            if "strong_model" not in d.files or d["preds_strong"].ndim != 2:
                continue  # regression: loss units differ, skip
            strong, weak = str(d["strong_model"]), str(d["weak_model"])
            imp_path = IMP / strong / f"{npz.stem}.npz"
            if not imp_path.exists():
                continue
            imp = np.load(imp_path, allow_pickle=True)
            drops = np.asarray(imp["row_feature_drops"], dtype=np.float64)
            feat_idx = [int(x) for x in imp["feature_indices"]]
            unmatched = {int(x) for x in get_unmatched_features(strong, weak)}
            upos = [i for i, fi in enumerate(feat_idx) if fi in unmatched]
            wins = np.asarray(d["strong_wins"], dtype=bool)
            sel = np.asarray(d["selected_features"])
            if len(wins) != drops.shape[0] or sel.ndim != 2 or sel.shape[0] != len(wins):
                continue  # e.g. carte_vs_mitra/hiva_agnostic stores an empty selection array
            gap_all = np.asarray(d["baseline_loss_weak"]) - np.asarray(d["baseline_loss_strong"])
            for r in np.where(wins)[0]:
                row_sel = [int(f) for f in np.atleast_1d(sel[r]) if int(f) >= 0]
                if not row_sel:
                    continue
                gap = float(gap_all[r])
                if gap <= 0:
                    continue
                cand = sorted((c for c in upos if drops[r, c] > 0), key=lambda c: -drops[r, c])
                rank_of = {feat_idx[c]: i for i, c in enumerate(cand)}
                depths = [rank_of[f] for f in row_sel if f in rank_of]
                if not depths:
                    continue
                deepest = max(depths)
                acc_set = set(row_sel)
                for i, c in enumerate(cand[: deepest + 1]):
                    key = "acc" if feat_idx[c] in acc_set else "skip"
                    tot[key] += 1
                    if drops[r, c] > 2 * gap:
                        tot[key + "_over"] += 1
                    if drops[r, c] > gap:
                        tot[key + "_over1"] += 1
                gap_rows.append((gap, len(row_sel), deepest + 1))
    print(f"rows: {len(gap_rows)}")
    for k in ("skip", "acc"):
        n = tot[k]
        print(f"{k:5s} n={n:7d}  drop>2*gap: {tot[k+'_over']/n:.3f}   drop>gap: {tot[k+'_over1']/n:.3f}")
    g = np.array(gap_rows)
    q = np.quantile(g[:, 0], [1 / 3, 2 / 3])
    for lo, hi, name in ((-np.inf, q[0], "small"), (q[0], q[1], "mid"), (q[1], np.inf, "large")):
        m = (g[:, 0] > lo) & (g[:, 0] <= hi)
        print(f"gap {name:5s} (n={m.sum():6d}, median gap {np.median(g[m,0]):.3f}): "
              f"acc = {g[m,1].sum()/g[m,2].sum():.3f}, mean K = {g[m,1].mean():.2f}, mean depth = {g[m,2].mean():.1f}")


def batch_stats():
    for name, sweep in (("ablation", SWEEP), ("transfer", PROJECT_ROOT / "output" / "transfer_global_mnnp90_trained_tols")):
        c = defaultdict(int)
        for f in sorted(glob.glob(f"{sweep}/*/*.npz")):
            d = np.load(f, allow_pickle=True)
            if "step_sizes" not in d.files or "strong_wins" not in d.files:
                continue
            st, w = d["step_sizes"], d["strong_wins"]
            for r in np.where(w)[0]:
                s = st[r]
                s = s[s > 0]
                if len(s) == 0:
                    continue
                c["rows"] += 1
                c["steps"] += len(s)
                c["accepted"] += int(s.sum())
                for k in (1, 2, 3):
                    c[f"size{k}"] += int((s == k).sum())
        n, st_ = c["rows"], c["steps"]
        print(f"{name}: rows {n}, batches/row {st_/n:.2f}, accepted/row {c['accepted']/n:.2f}, "
              f"accepted/batch {c['accepted']/st_:.2f}, subset sizes 1/2/3: "
              f"{c['size1']/st_:.2f}/{c['size2']/st_:.2f}/{c['size3']/st_:.2f}")


if __name__ == "__main__":
    main()
    print()
    batch_stats()
