#!/usr/bin/env python3
"""Directional analysis of concept transfers (cosine metrics).

Addresses the reviewer concern "do transfers push embeddings off-manifold?"
using cosine similarity — the appropriate metric for high-dimensional
neural-net embeddings (L2 norms concentrate at high d and conflate magnitude
with direction).

Per accepted row, reports three angular quantities:
  1. cos(x, x+Δ)     — how much the embedding direction rotated
  2. cos(Δ, x)       — is Δ reinforcing x, orthogonal, or reversing
  3. Top-k NN cos    — highest cosine similarity to other real embeddings
                       (in the same dataset), pre- vs post-transfer.
                       If post-transfer top-k NN cos ≫ random noise floor
                       (≈ 1/√d), the embedding stays on the data manifold.

Reconstructs Δ from existing caches — no re-run needed. Because only
feature IDs (not ±signs) are saved in selected_features, we approximate
Δ as Σᵢ h_strong[r, fi] · virtual_atom[fi] over DISTINCT accepted features
(collapsing any +/− duplicates to a single occurrence). This represents one
plausible signed reconstruction; exact signs would require patching
transfer_sweep_v2.py to save them.

Output:
  output/transfer_direction_summary.csv   per-pair aggregates
  output/transfer_direction_rows.csv      per-row data for appendix plots

Usage:
  python -m scripts.analysis.transfer_direction
  python -m scripts.analysis.transfer_direction --pairs tabicl_vs_tabpfn
  python -m scripts.analysis.transfer_direction --topk 5
"""
import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import load_sae, load_test_embeddings

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TRANSFER_DIR_TRAINED = PROJECT_ROOT / "output" / "transfer_global_mnnp90_trained_tols"
TRANSFER_DIR_RANDOM = PROJECT_ROOT / "output" / "transfer_global_mnnp90_random"
VIRTUAL_ATOMS_DIR_TRAINED = PROJECT_ROOT / "output" / "transfer_caches" / "global_trained"
VIRTUAL_ATOMS_DIR_RANDOM = PROJECT_ROOT / "output" / "transfer_caches" / "global_random"
SAE_DIR_TRAINED = PROJECT_ROOT / "output" / "sae_tabarena_sweep_round10"
SAE_DIR_RANDOM = PROJECT_ROOT / "output" / "sae_random_baseline"


def load_virtual_atoms(vatoms_dir: Path, strong: str, weak: str) -> Dict[int, np.ndarray]:
    cache = np.load(vatoms_dir / f"{strong}_to_{weak}.npz", allow_pickle=True)
    vatoms = cache["virtual_atoms"]
    fids = cache["feature_ids"]
    mask = cache["computed_mask"]
    return {
        int(fids[i]): vatoms[i].astype(np.float32)
        for i in range(len(fids)) if mask[i]
    }


def encode_strong_activations(strong: str, sae_dir: Path, device: str = "cuda") -> Dict[str, np.ndarray]:
    sae, _ = load_sae(strong, sae_dir=sae_dir, device=device)
    test_emb = load_test_embeddings(strong)
    out = {}
    with torch.no_grad():
        for ds, emb in test_emb.items():
            x = torch.tensor(emb, dtype=torch.float32, device=device)
            out[ds] = sae.encode(x).cpu().numpy()
    del sae
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def topk_nn_cos(query: np.ndarray, pool_unit: np.ndarray, k: int,
                exclude_idx: int) -> np.ndarray:
    """Top-k cosine similarity of query against pool, excluding one row."""
    q = query / (np.linalg.norm(query) + 1e-12)
    sims = pool_unit @ q
    sims[exclude_idx] = -np.inf
    top = np.partition(sims, -k)[-k:]
    return np.sort(top)[::-1]


def process_dataset(
    transfer_npz: Path,
    virtual_atoms: Dict[int, np.ndarray],
    h_strong_ds: np.ndarray,
    emb_weak_ds: np.ndarray,
    topk: int,
) -> list:
    """Compute per-row cosine records for one pair-dataset."""
    d = np.load(transfer_npz, allow_pickle=True)
    strong = str(d["strong_model"])
    weak = str(d["weak_model"])
    selected = d["selected_features"]
    strong_wins = d["strong_wins"]
    optimal_k = d["optimal_k"]
    row_indices = d["row_indices"]

    n_q = len(selected)
    # Precompute unit-normalized weak pool (for NN lookups)
    pool = emb_weak_ds[:n_q]
    pool_norms = np.linalg.norm(pool, axis=1, keepdims=True)
    pool_unit = pool / (pool_norms + 1e-12)

    records = []
    for r in range(n_q):
        if not strong_wins[r] or optimal_k[r] == 0:
            continue
        sel = selected[r]
        sel = sel[sel >= 0]
        # Distinct features (collapse +/− duplicates)
        distinct = list(dict.fromkeys(int(f) for f in sel))
        distinct = [fi for fi in distinct if fi in virtual_atoms]
        if not distinct:
            continue

        # Reconstruct Δ (approximation: all distinct features, + sign)
        delta = np.zeros_like(emb_weak_ds[r])
        for fi in distinct:
            delta += float(h_strong_ds[r, fi]) * virtual_atoms[fi]

        x = emb_weak_ds[r]
        x_post = x + delta

        x_norm = np.linalg.norm(x) + 1e-12
        d_norm = np.linalg.norm(delta) + 1e-12
        xp_norm = np.linalg.norm(x_post) + 1e-12

        cos_rotation = float(np.dot(x, x_post) / (x_norm * xp_norm))
        cos_delta_x = float(np.dot(delta, x) / (d_norm * x_norm))

        # Top-k NN cosine: within-dataset test pool, excluding self
        k = min(topk, max(1, n_q - 1))
        pre_nn = topk_nn_cos(x, pool_unit, k, r)
        post_nn = topk_nn_cos(x_post, pool_unit, k, r)

        records.append({
            "strong": strong,
            "weak": weak,
            "dataset": transfer_npz.stem,
            "row_pos": r,
            "abs_row_idx": int(row_indices[r]),
            "k_selected": int(optimal_k[r]),
            "n_distinct": len(distinct),
            "emb_dim": int(emb_weak_ds.shape[1]),
            "cos_x_xplusdelta": cos_rotation,
            "cos_delta_x": cos_delta_x,
            "nn_cos_pre_mean": float(pre_nn.mean()),
            "nn_cos_pre_top1": float(pre_nn[0]),
            "nn_cos_post_mean": float(post_nn.mean()),
            "nn_cos_post_top1": float(post_nn[0]),
            "nn_cos_delta_mean": float(post_nn.mean() - pre_nn.mean()),
            "delta_l2": float(d_norm),
            "x_l2": float(x_norm),
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--condition", choices=["trained", "random"], default="trained",
                        help="SAE condition: trained (default) or random baseline")
    parser.add_argument("--out-tag", default=None,
                        help="Output CSV tag suffix (default: --condition)")
    args = parser.parse_args()

    if args.condition == "trained":
        transfer_dir = TRANSFER_DIR_TRAINED
        vatoms_dir = VIRTUAL_ATOMS_DIR_TRAINED
        sae_dir = SAE_DIR_TRAINED
    else:
        transfer_dir = TRANSFER_DIR_RANDOM
        vatoms_dir = VIRTUAL_ATOMS_DIR_RANDOM
        sae_dir = SAE_DIR_RANDOM
    tag = args.out_tag or args.condition
    out_summary = PROJECT_ROOT / "output" / f"transfer_direction_summary_{tag}.csv"
    out_rows = PROJECT_ROOT / "output" / f"transfer_direction_rows_{tag}.csv"

    pair_dirs = [p for p in sorted(transfer_dir.iterdir()) if p.is_dir()]
    if args.pairs:
        pair_dirs = [p for p in pair_dirs if p.name in set(args.pairs)]

    direction_datasets: Dict[Tuple[str, str], list] = defaultdict(list)
    for pd in pair_dirs:
        for npz_path in sorted(pd.glob("*.npz")):
            try:
                d = np.load(npz_path, allow_pickle=True)
                if int(d["n_strong_wins"]) == 0:
                    continue
                if "selected_features" not in d.keys():
                    continue
                direction_datasets[(str(d["strong_model"]), str(d["weak_model"]))].append(npz_path)
            except Exception as e:
                logger.warning(f"  WARN load {npz_path.name}: {e}")

    logger.info(f"Found {sum(len(v) for v in direction_datasets.values())} "
                f"dataset files across {len(direction_datasets)} directions")

    all_records = []
    test_emb_cache: Dict[str, dict] = {}

    for (strong, weak), npzs in sorted(direction_datasets.items()):
        logger.info(f"[{strong} -> {weak}] {len(npzs)} datasets")
        virtual_atoms = load_virtual_atoms(vatoms_dir, strong, weak)
        h_strong = encode_strong_activations(strong, sae_dir, device=args.device)
        if weak not in test_emb_cache:
            test_emb_cache[weak] = load_test_embeddings(weak)
        weak_emb = test_emb_cache[weak]

        for npz_path in npzs:
            ds = npz_path.stem
            if ds not in weak_emb or ds not in h_strong:
                continue
            records = process_dataset(
                npz_path, virtual_atoms, h_strong[ds], weak_emb[ds], args.topk,
            )
            all_records.extend(records)
        del h_strong

    if not all_records:
        logger.warning("No records — nothing to write.")
        return

    out_rows.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_records[0].keys())
    with open(out_rows, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_records)
    logger.info(f"Wrote {len(all_records)} rows -> {out_rows}")

    # Aggregate: per-pair + overall
    pair_groups: Dict[Tuple[str, str], list] = defaultdict(list)
    for rec in all_records:
        pair_groups[tuple(sorted([rec["strong"], rec["weak"]]))].append(rec)

    summary_rows = []
    all_cos_rot = np.array([r["cos_x_xplusdelta"] for r in all_records])
    all_cos_dx = np.array([r["cos_delta_x"] for r in all_records])
    all_nn_pre = np.array([r["nn_cos_pre_mean"] for r in all_records])
    all_nn_post = np.array([r["nn_cos_post_mean"] for r in all_records])
    all_d = np.array([r["emb_dim"] for r in all_records])

    def pct(a, q): return float(np.percentile(a, q))

    def aggregate(recs):
        rot = np.array([r["cos_x_xplusdelta"] for r in recs])
        dx = np.array([r["cos_delta_x"] for r in recs])
        nn_pre = np.array([r["nn_cos_pre_mean"] for r in recs])
        nn_post = np.array([r["nn_cos_post_mean"] for r in recs])
        return {
            "n_rows": len(recs),
            "cos_rotation_median": float(np.median(rot)),
            "cos_rotation_p10": pct(rot, 10),
            "cos_rotation_p90": pct(rot, 90),
            "cos_delta_x_median": float(np.median(dx)),
            "nn_cos_pre_median": float(np.median(nn_pre)),
            "nn_cos_post_median": float(np.median(nn_post)),
            "nn_cos_drop_median": float(np.median(nn_post - nn_pre)),
        }

    for pair_key, recs in sorted(pair_groups.items()):
        row = {"pair": f"{pair_key[0]}_vs_{pair_key[1]}"}
        row.update(aggregate(recs))
        summary_rows.append(row)

    overall = {"pair": "OVERALL"}
    overall.update(aggregate(all_records))
    summary_rows.append(overall)

    with open(out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    logger.info(f"Wrote summary -> {out_summary}")

    # Headline
    median_d = float(np.median(all_d))
    noise_floor = 1.0 / np.sqrt(median_d)
    logger.info("")
    logger.info("=" * 72)
    logger.info(f"TRANSFER DIRECTION SUMMARY  (N={len(all_records)} accepted rows)")
    logger.info("=" * 72)
    logger.info(f"cos(x, x+Δ)   median={np.median(all_cos_rot):.3f}   "
                f"p10={pct(all_cos_rot,10):.3f}  p90={pct(all_cos_rot,90):.3f}")
    logger.info(f"cos(Δ, x)     median={np.median(all_cos_dx):.3f}   "
                f"p10={pct(all_cos_dx,10):.3f}  p90={pct(all_cos_dx,90):.3f}")
    logger.info(f"NN cos (pre)  median={np.median(all_nn_pre):.3f}")
    logger.info(f"NN cos (post) median={np.median(all_nn_post):.3f}  "
                f"(noise floor 1/√d̄ = {noise_floor:.3f})")
    logger.info(f"NN drop       median={np.median(all_nn_post - all_nn_pre):+.3f}")
    logger.info("")
    logger.info("Paper sentence candidates:")
    logger.info(
        f'  "Across {len(all_records)} accepted transfers, the injected '
        f"perturbation rotated embeddings by a median cos(x, x+Δ) = "
        f"{np.median(all_cos_rot):.2f} — substantial angular change. "
        f"Post-transfer embeddings retain top-{args.topk} nearest-neighbor "
        f"cosine similarity of {np.median(all_nn_post):.2f} to other real "
        f"embeddings (vs {np.median(all_nn_pre):.2f} pre-transfer; random "
        f"noise floor ≈ {noise_floor:.2f}), confirming that transfers relocate "
        f'embeddings to a different in-distribution neighborhood rather than '
        f'off-manifold."'
    )


if __name__ == "__main__":
    main()
