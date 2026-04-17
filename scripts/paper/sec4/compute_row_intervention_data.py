#!/usr/bin/env python3
"""Extract per-feature replay data from sweep npz files into JSON for plotting.

Reads ablation + transfer npz files that contain feature_preds (per-feature
cumulative predictions computed inside the sweep with the same tail).
No GPU needed.

Usage:
    python scripts/paper/sec4/compute_row_intervention_data.py
"""
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PAIR = "carte_vs_mitra"
DATASET = "credit-g"
TARGET_ROW = 325
ABLATION_FILE = PROJECT_ROOT / "output" / "ablation_figure_data" / PAIR / f"{DATASET}.npz"
TRANSFER_FILE = PROJECT_ROOT / "output" / "transfer_figure_data" / PAIR / f"{DATASET}.npz"
IMPORTANCE_DIR = PROJECT_ROOT / "output" / "perrow_importance"
MNN_FILE = PROJECT_ROOT / "output" / "sae_feature_matching_mnn_floor_p90.json"
OUTPUT_FILE = Path(__file__).parent / "row_intervention_data.json"


def main():
    # ── Ablation ──
    ab = np.load(ABLATION_FILE, allow_pickle=True)
    ri = int(np.where(ab["row_indices"] == TARGET_ROW)[0][0])
    strong = str(ab["strong_model"])
    weak = str(ab["weak_model"])
    p_strong = float(ab["preds_strong"][ri][1])
    p_weak = float(ab["preds_weak"][ri][1])
    k_ab = int(ab["optimal_k"][ri])
    ab_feats = ab["selected_features"][ri][:k_ab].tolist()
    ab_fp = ab["feature_preds"][ri]

    print(f"Strong={strong} P={p_strong:.4f}, Weak={weak} P={p_weak:.4f}")
    print(f"Ablation: k={k_ab}, gc={ab['gap_closed'][ri]:.4f}")
    ab_preds = []
    prev = p_strong
    for i, feat in enumerate(ab_feats):
        p = float(ab_fp[i, 1])
        print(f"  remove f_{feat}: P={p:.4f} (delta={p - prev:+.04f})")
        ab_preds.append(p)
        prev = p

    # ── Transfer ──
    tr_feats = []
    tr_preds = []
    if TRANSFER_FILE.exists():
        tr = np.load(TRANSFER_FILE, allow_pickle=True)
        tri = int(np.where(tr["row_indices"] == TARGET_ROW)[0][0])
        k_tr = int(tr["optimal_k"][tri])
        sel_raw = tr["selected_features"][tri][:k_tr].tolist()
        seen = set()
        tr_feats = [f for f in sel_raw if not (f in seen or seen.add(f))]
        tr_fp = tr["feature_preds"][tri]

        print(f"\nTransfer: k={k_tr}, gc={tr['gap_closed'][tri]:.4f}")
        prev = p_weak
        for i, feat in enumerate(tr_feats):
            if np.isnan(tr_fp[i]).any():
                break
            p = float(tr_fp[i, 1])
            print(f"  inject f_{feat}: P={p:.4f} (delta={p - prev:+.04f})")
            tr_preds.append(p)
            prev = p
    else:
        print(f"\nNo transfer file at {TRANSFER_FILE}")

    # ── Pool ──
    imp = np.load(IMPORTANCE_DIR / strong / f"{DATASET}.npz", allow_pickle=True)
    imp_features = imp["feature_indices"]
    row_drops = imp["row_feature_drops"][ri]

    with open(MNN_FILE) as f:
        mnn = json.load(f)
    p_mnn = mnn["pairs"].get("CARTE__Mitra", mnn["pairs"].get("Mitra__CARTE", {}))
    unmatched = set(p_mnn.get("unmatched_b", []))

    pool = []
    for i, fi in enumerate(imp_features):
        if row_drops[i] != 0 and int(fi) in unmatched:
            pool.append({"feature": int(fi), "importance": float(row_drops[i])})
    pool.sort(key=lambda x: -x["importance"])
    print(f"\nPool: {len(pool)} firing unmatched features")

    # ── Save ──
    result = {
        "dataset": DATASET,
        "pair": PAIR,
        "row": TARGET_ROW,
        "strong_model": strong,
        "weak_model": weak,
        "p_strong": p_strong,
        "p_weak": p_weak,
        "ablation": {
            "features": ab_feats,
            "step_preds": ab_preds,
        },
        "transfer": {
            "features": tr_feats,
            "step_preds": tr_preds,
        },
        "pool": pool,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
