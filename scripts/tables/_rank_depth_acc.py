"""Shared rank-depth acceptance rate for the ablation / transfer summary tables.

acc = concepts accepted / concepts tried, where "tried" is how deep into the
strong (donor) model's importance-ranked candidate list the greedy reached
before stopping: the number of unmatched concepts ranked at or above the
*deepest accepted* concept. Concepts ranked below the last acceptance are never
tried (the greedy stops at its gap-closure tolerance).

Ranking matches each sweep's candidate ordering:
  - ablation ranks by the ablation drop (positive only; it removes helpful
    concepts) -> use_abs=False
  - transfer ranks by |drop| magnitude -> use_abs=True
A non-firing concept has ~zero drop, so it falls below the accepted threshold
and firing is implicit; the transfer virtual-atom filter is a no-op (every
unmatched concept has a virtual atom), so it needs no reconstruction.

Counts are pooled over the strong-win rows of one (pair, dataset).
"""
import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.ablation_sweep import get_unmatched_features

IMPORTANCE_DIR = PROJECT_ROOT / "output" / "perrow_importance"


def rank_depth_counts(data, strong, weak, dataset, use_abs):
    """(n_used, n_tried) pooled over strong-win rows, or None if unavailable.

    `data` is a loaded ablation/transfer sweep npz for (strong vs weak, dataset).
    """
    imp_path = IMPORTANCE_DIR / strong / f"{dataset}.npz"
    if (not imp_path.exists() or "strong_wins" not in data
            or "selected_features" not in data):
        return None
    imp = np.load(imp_path, allow_pickle=True)
    drops = np.asarray(imp["row_feature_drops"], dtype=np.float64)   # (n_query, n_feat)
    feat_idx = [int(x) for x in imp["feature_indices"]]
    pos_of = {fi: i for i, fi in enumerate(feat_idx)}
    unmatched = {int(x) for x in get_unmatched_features(strong, weak)}
    umask = np.array([fi in unmatched for fi in feat_idx])
    if not umask.any():
        return None
    wins = np.asarray(data["strong_wins"], dtype=bool)
    sel = np.asarray(data["selected_features"])
    if len(wins) != drops.shape[0] or len(sel) != drops.shape[0]:
        return None

    umatched_positions = [i for i in range(len(feat_idx)) if umask[i]]
    n_used = n_tried = 0
    for r in np.where(wins)[0]:
        row_sel = [int(f) for f in np.atleast_1d(sel[r]) if int(f) >= 0]
        if not row_sel:
            continue
        metric = np.abs(drops[r]) if use_abs else drops[r]
        # Rebuild the row's candidate list in the greedy's order: unmatched
        # concepts with metric > 0 (a non-firing concept has ~zero drop), sorted
        # by descending metric. Python's sort is stable, preserving
        # feature_indices order on ties, matching the sweeps.
        cand = sorted((c for c in umatched_positions if metric[c] > 0),
                      key=lambda c: -metric[c])
        rank_of = {feat_idx[c]: i for i, c in enumerate(cand)}
        depths = [rank_of[f] for f in row_sel if f in rank_of]
        if not depths:
            continue
        n_tried += max(depths) + 1     # rank depth reached = deepest accepted + 1
        n_used += len(row_sel)
    return (n_used, n_tried) if n_tried else None
