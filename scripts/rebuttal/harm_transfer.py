#!/usr/bin/env python3
"""REBUTTAL: can transfer from a WEAKER model HARM a STRONGER model?

Answers SD9t Q1 ("does strong get worse?") and dVDs Q3 ("can transfer harm?").

Mechanism (the only coherent harm channel, per the design discussion): additive
transfer of a weak model's *unique* concepts cannot cleanly harm — you are adding
information the strong model lacked, and the paper's greedy acceptance filters
anything that moves the prediction the wrong way. The clean harm channel is
*subtractive, through a shared concept*: ablate a concept the weak model shares
with the strong one, transfer that ablation delta into the strong model, and it
loses a concept it was using -> its prediction degrades.

Concretely, for a MATCHED (shared) concept `c` in the weak model and a query row r:

    delta_src[r] = decode(h_weak[r] with c zeroed) - decode(h_weak[r])   # weak (normalized) emb space
    delta_tgt[r] = delta_src[r] @ M.T                                    # map weak->strong (same ridge M as transfer)
    delta_raw[r] = delta_tgt[r] * std_strong                            # denormalize to strong raw emb space
    strong_pred'[r] = strong_tail(query=r, +delta_raw[r])              # inject into query position (inject_context=False)

The claim rests ENTIRELY on the strong model's measured output dropping. We never
assert the delta "landed on" the strong model's matched concept, so the result
does NOT depend on the matching being correct. The matched-concept ordering is
only a *search heuristic* for where harm is likely (shared concepts are what the
strong model also uses). No direct-ablation ceiling; no reliance on the match.

Existence proof: find at least one concept whose transferred ablation delta harms
the strong model on the lower-triangle rows (rows where strong already wins).
Single concept at a time; record the per-concept reduction.

Deterministic: no args needed. Defaults reproduce the canonical case:
    strong = tabpfn, weak = tabdpt, dataset = credit-g, lower-triangle rows.
The SAE / importance / matching defaults are the SAME as the transfer sweep
(mnn_floor_p90), so the transfer channel is identical to E1 — just carrying a
subtractive matched-concept delta instead of an additive virtual atom.

Usage:
    python -m scripts.rebuttal.harm_transfer
    python -m scripts.rebuttal.harm_transfer --models tabpfn tabdpt --dataset credit-g
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import (
    SPLITS_PATH,
    load_sae, get_extraction_layer_taskaware, build_tail,
    load_dataset_context, load_test_embeddings,
    compute_per_row_loss, compute_importance_metric,
    batched_intervention,
)
from scripts.matching.utils import load_norm_stats as load_norm_stats_matching
from scripts.intervention.transfer_virtual_nodes import (
    extract_decoder_atoms, fit_concept_map, filter_landmarks,
)
from scripts.rebuttal.transfer_sweep_symmetric import (
    get_matched_pairs, DEFAULT_MATCHING_FILE,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "output" / "rebuttal" / "harm_transfer"
IMPORTANCE_DIR = PROJECT_ROOT / "output" / "perrow_importance"


def _weak_importance_rank(imp_weak) -> dict:
    """Mean |per-row drop| per weak SAE feature index -> importance score."""
    drops = np.abs(imp_weak["row_feature_drops"]).mean(axis=0)  # (n_feat_cols,)
    fidx = imp_weak["feature_indices"]
    return {int(fidx[j]): float(drops[j]) for j in range(len(fidx))}


def run(strong: str, weak: str, dataset: str, device: str,
        matching_file: Path, sae_dir: Path, imp_dir: Path,
        mode: str = "matched") -> dict:
    """Transfer single matched-concept ablation deltas weak->strong on the
    lower-triangle rows; record per-concept harm to the strong model.

    mode:
      "raw"     — map the true ablation delta (-h[r,c]*atom_c) straight through M.
                  Faithful, but the near-null map shrinks it to near-noise.
      "matched" — magnitude-match the E1 transfer channel: unit-map the concept
                  atom through M, renormalize, rescale by ||atom||*median_norm_ratio,
                  then inject -a_s*virtual_atom*std_strong. Same perturbation size
                  E1 uses as *help*, signed for removal (apples-to-apples).
    """
    splits = json.loads(SPLITS_PATH.read_text())

    # ── Load SAEs, norm stats, test embeddings ───────────────────────────────
    saes, norm_stats, test_emb = {}, {}, {}
    for m in (strong, weak):
        sae, _ = load_sae(m, device=device,
                          **({"sae_dir": sae_dir} if sae_dir else {}))
        sae.eval()
        saes[m] = sae
        norm_stats[m] = load_norm_stats_matching(m)
        test_emb[m] = load_test_embeddings(m)

    # ── Baseline predictions + row alignment (from cached importance) ─────────
    imp_strong = np.load(imp_dir / strong / f"{dataset}.npz", allow_pickle=True)
    imp_weak = np.load(imp_dir / weak / f"{dataset}.npz", allow_pickle=True)
    assert np.array_equal(imp_strong["row_indices"], imp_weak["row_indices"]), \
        "row index mismatch between models"
    y_query = imp_strong["y_query"].astype(np.int64)
    n_query = len(y_query)
    preds_strong = imp_strong["baseline_preds"].astype(np.float32)
    preds_weak = imp_weak["baseline_preds"].astype(np.float32)
    task = "classification"

    # Confirm the performance orientation matches the requested strong/weak.
    m_strong, metric_name = compute_importance_metric(y_query, preds_strong, task)
    m_weak, _ = compute_importance_metric(y_query, preds_weak, task)
    logger.info(f"  {metric_name}: {strong}(strong)={m_strong:.4f}  "
                f"{weak}(weak)={m_weak:.4f}")
    if m_strong < m_weak:
        logger.warning(f"  NOTE: on {dataset}, requested strong={strong} is NOT "
                       f"the better model ({metric_name} {m_strong:.4f} < "
                       f"{m_weak:.4f}). Harm direction is defined by this flag.")

    # Lower-triangle rows: strong wins per-row (strong_loss < weak_loss).
    loss_strong = compute_per_row_loss(y_query, preds_strong, task)
    loss_weak = compute_per_row_loss(y_query, preds_weak, task)
    lower = np.where(loss_strong < loss_weak)[0]
    logger.info(f"  Lower-triangle rows (strong {strong} wins per-row): "
                f"{len(lower)}/{n_query}")
    if len(lower) == 0:
        return {"strong": strong, "weak": weak, "dataset": dataset,
                "n_lower": 0, "concepts": []}

    # ── Cross-model map M: weak -> strong (same ridge as transfer sweep) ──────
    atoms_weak = extract_decoder_atoms(saes[weak]).numpy()    # (n_feat, d_weak)
    atoms_strong = extract_decoder_atoms(saes[strong]).numpy()  # (n_feat, d_strong)
    m_pairs = get_matched_pairs(weak, strong, matching_file=matching_file)  # (weak_idx, strong_idx)
    if len(m_pairs) < 5:
        raise RuntimeError(f"too few matched pairs {weak}->{strong}: {len(m_pairs)}")
    src_idx = [si for si, _ in m_pairs]
    tgt_idx = [ti for _, ti in m_pairs]
    filt_src, filt_tgt, filt_pairs, quality = filter_landmarks(
        atoms_weak[src_idx], atoms_strong[tgt_idx], m_pairs, min_cosine=0.0, alpha=1.0)
    M, r2 = fit_concept_map(filt_src, filt_tgt, alpha=1.0)  # d_strong = d_weak @ M.T
    logger.info(f"  Landmark map {weak}->{strong}: {len(filt_pairs)} pairs kept, "
                f"R²={r2:.3f}, M shape={M.shape}")

    # Sign-flip resolution (oracle-free): filter_landmarks negates the strong-
    # side atom of any pair whose LOO cosine is negative (MNN uses |r|). For a
    # FLIPPED concept the map is anti-aligned, so the true "remove from strong"
    # direction is +delta; for an aligned concept it is -delta.
    flipped_src = {int(pair[0]) for pair, _cos in quality["flipped"]}
    logger.info(f"  Sign-flipped concepts (anti-aligned map): {len(flipped_src)}/{len(m_pairs)} pairs")

    # Candidate concepts = matched WEAK indices, deduped, ranked by weak importance.
    imp_rank = _weak_importance_rank(imp_weak)
    cand = sorted(set(src_idx), key=lambda c: -imp_rank.get(c, 0.0))
    cand_flipped = np.array([c in flipped_src for c in cand], dtype=bool)  # (K,)
    # tgt matches per weak concept (metadata only; not used in the delta)
    strong_matches = {}
    for si, ti in m_pairs:
        strong_matches.setdefault(si, []).append(int(ti))
    logger.info(f"  Candidate matched weak concepts: {len(cand)} "
                f"({int(cand_flipped.sum())} sign-flipped)")

    # ── Weak SAE activations on the lower-triangle rows ───────────────────────
    with torch.no_grad():
        emb_w = torch.tensor(test_emb[weak][dataset], dtype=torch.float32, device=device)
        h_weak = saes[weak].encode(emb_w).cpu().numpy()  # (n_query, n_feat) normalized-space acts
    std_strong = np.asarray(norm_stats[strong][dataset][1], dtype=np.float32)  # (d_strong,)
    M_t = M.T.astype(np.float32)  # (d_weak, d_strong)
    atoms_weak_cand = atoms_weak[cand].astype(np.float32)  # (K, d_weak) decoder columns

    # Self-check: analytic ablation delta == decode difference (SAE decode is linear).
    with torch.no_grad():
        h0 = torch.tensor(h_weak[lower[0]], dtype=torch.float32, device=device)
        recon_full = saes[weak].decode(h0.unsqueeze(0))[0].cpu().numpy()
        h_ab = h0.clone(); h_ab[cand[0]] = 0.0
        recon_ab = saes[weak].decode(h_ab.unsqueeze(0))[0].cpu().numpy()
    analytic = -h_weak[lower[0], cand[0]] * atoms_weak_cand[0]
    assert np.allclose(recon_ab - recon_full, analytic, atol=1e-4), \
        "SAE decode is not linear as assumed; use compute_feature_deltas instead"

    # Magnitude-matched channel: precompute per-concept virtual atoms exactly as
    # the transfer sweep does (unit-map through M, renorm, rescale by
    # ||atom||*median_norm_ratio). Injected delta = -a_s*virtual_atom*std_strong.
    virtual_atoms = None
    if mode == "matched":
        src_n = np.linalg.norm(filt_src, axis=1)
        tgt_n = np.linalg.norm(filt_tgt, axis=1)
        vmask = (src_n > 1e-8) & (tgt_n > 1e-8)
        median_norm_ratio = float(np.median(tgt_n[vmask] / src_n[vmask])) if vmask.any() else 1.0
        atom_norms = np.linalg.norm(atoms_weak_cand, axis=1, keepdims=True)  # (K,1)
        unit = atoms_weak_cand / np.clip(atom_norms, 1e-8, None)
        vdir = unit @ M_t                                                   # (K, d_strong)
        vdir_n = np.linalg.norm(vdir, axis=1, keepdims=True)
        virtual_atoms = ((vdir / np.clip(vdir_n, 1e-8, None))
                         * atom_norms * median_norm_ratio).astype(np.float32)
        logger.info(f"  Magnitude-matched channel: median_norm_ratio={median_norm_ratio:.3f}")

    # ── Build strong recipient tail ──────────────────────────────────────────
    t0 = time.time()
    X_train_s, y_train_s, X_query_s, _, _, task_s = load_dataset_context(
        strong, dataset, splits)
    if y_train_s.dtype == np.int32:
        y_train_s = y_train_s.astype(np.int64)
    layer_s = get_extraction_layer_taskaware(strong, dataset=dataset)
    cat_indices = None
    if strong in ("hyperfast", "tabpfn"):
        from data.preprocessing import load_preprocessed, CACHE_DIR
        try:
            pre = load_preprocessed(strong, dataset, CACHE_DIR)
            cat_indices = pre.cat_indices if pre.cat_indices else None
        except Exception:
            pass
    target_name = splits.get(dataset, {}).get("target", "target")
    torch.manual_seed(13); np.random.seed(13)
    tail_s = build_tail(strong, X_train_s, y_train_s, X_query_s, layer_s, task_s,
                        device, cat_indices=cat_indices, target_name=target_name)
    logger.info(f"  Strong tail ({strong}) built in {time.time()-t0:.1f}s")

    # ── Per row: tail pass over [baseline, +delta (K), -delta (K)] ───────────
    # BOTH signs, exactly as the transfer sweep does ("cross-correlation uses
    # |r|, sign may flip"): delta_raw = a_s * virtual_atom * std_strong, and we
    # evaluate +delta (inject/reinforce the concept) AND -delta (ablate/remove
    # it). The removal (-delta) direction is the harm-relevant one; we record
    # both and select nothing.
    K = len(cand)
    eps = 1e-7
    base_p_true = np.zeros(len(lower), dtype=np.float64)
    base_p1 = np.zeros(len(lower), dtype=np.float64)
    p_true_plus = np.zeros((len(lower), K), dtype=np.float64)   # +delta (inject)
    p_true_minus = np.zeros((len(lower), K), dtype=np.float64)  # -delta (ablate)
    p1_plus = np.zeros((len(lower), K), dtype=np.float64)
    p1_minus = np.zeros((len(lower), K), dtype=np.float64)

    t1 = time.time()
    for i, r in enumerate(lower):
        a_s = h_weak[r, cand][:, None]  # (K,1) donor (weak) activation, == transfer's a_s
        if mode == "raw":              # map the true decoder atom straight through M
            pos = ((a_s * atoms_weak_cand) @ M_t * std_strong[None, :]).astype(np.float32)
        else:                          # matched: identical to the transfer sweep
            pos = (a_s * virtual_atoms * std_strong[None, :]).astype(np.float32)
        deltas = np.vstack([np.zeros((1, pos.shape[1]), np.float32), pos, -pos])  # (2K+1, d)
        deltas_t = torch.tensor(deltas, dtype=torch.float32, device=device)
        preds = batched_intervention(tail_s, X_query_s[r:r + 1], deltas_t,
                                     inject_context=False)  # (2K+1, 2)
        preds = np.asarray(preds, dtype=np.float64)
        y_r = int(y_query[r])
        base_p_true[i] = np.clip(preds[0, y_r], eps, 1 - eps)
        base_p1[i] = preds[0, 1]
        p_true_plus[i] = np.clip(preds[1:1 + K, y_r], eps, 1 - eps)
        p1_plus[i] = preds[1:1 + K, 1]
        p_true_minus[i] = np.clip(preds[1 + K:1 + 2 * K, y_r], eps, 1 - eps)
        p1_minus[i] = preds[1 + K:1 + 2 * K, 1]
        if (i + 1) % 10 == 0 or i + 1 == len(lower):
            logger.info(f"    row {i+1}/{len(lower)} "
                        f"({(i+1)/(time.time()-t1):.1f} rows/s)")

    # ── Raw per-(row, concept) prediction deltas — NO aggregation ────────────
    # One record per FIRING (row, concept): the change in the strong model's
    # P(true class) under +delta (inject) and -delta (ablate/remove). For the
    # harm question, delta_p_true_minus < 0 == removal harmed the strong model.
    y_lower = y_query[lower]
    dpt_plus = p_true_plus - base_p_true[:, None]
    dpt_minus = p_true_minus - base_p_true[:, None]
    dp1_plus = p1_plus - base_p1[:, None]
    dp1_minus = p1_minus - base_p1[:, None]
    # Sign-corrected REMOVAL: for a flipped concept the true removal is +delta;
    # for an aligned concept it is -delta. delta_*_removal < 0 == ablation harmed.
    dpt_removal = np.where(cand_flipped[None, :], dpt_plus, dpt_minus)   # (n_lower, K)
    dp1_removal = np.where(cand_flipped[None, :], dp1_plus, dp1_minus)
    fires = h_weak[np.ix_(lower, cand)] > 0              # (n_lower, K) bool
    fi, fk = np.where(fires)
    rows = []
    for i, k in zip(fi.tolist(), fk.tolist()):
        rows.append({
            "row": int(lower[i]),
            "weak_concept": int(cand[k]),
            "concept_flipped": bool(cand_flipped[k]),
            "strong_matches": strong_matches.get(cand[k], []),
            "h_activation": float(h_weak[lower[i], cand[k]]),
            "y_true": int(y_lower[i]),
            "base_p_true": float(base_p_true[i]),
            "delta_p_true_removal": float(dpt_removal[i, k]),  # sign-corrected; <0 == harm
            "delta_p_true_ablate": float(dpt_minus[i, k]),     # raw -delta
            "delta_p_true_inject": float(dpt_plus[i, k]),      # raw +delta
            "base_p1": float(base_p1[i]),
            "delta_p1_removal": float(dp1_removal[i, k]),
        })

    arrays = {
        "lower_rows": lower.astype(np.int64),
        "cand": np.asarray(cand, dtype=np.int64),
        "cand_flipped": cand_flipped,
        "y_lower": y_lower.astype(np.int64),
        "base_p1": base_p1.astype(np.float32),
        "base_p_true": base_p_true.astype(np.float32),
        "delta_p_true_removal": dpt_removal.astype(np.float32),
        "delta_p_true_ablate": dpt_minus.astype(np.float32),
        "delta_p_true_inject": dpt_plus.astype(np.float32),
        "delta_p1_removal": dp1_removal.astype(np.float32),
        "fires": fires,
        "h_weak_lower": h_weak[np.ix_(lower, cand)].astype(np.float32),
    }
    return {
        "strong": strong, "weak": weak, "dataset": dataset,
        "mode": mode,
        "metric_name": metric_name,
        "metric_strong": float(m_strong), "metric_weak": float(m_weak),
        "n_query": int(n_query), "n_lower": int(len(lower)),
        "map_r2": float(r2), "n_landmarks": int(len(filt_pairs)),
        "n_candidates": int(K),
        "n_firing_events": int(fires.sum()),
        "rows": rows,
        "_arrays": arrays,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs=2, default=["tabpfn", "tabdpt"],
                    metavar=("STRONG", "WEAK"),
                    help="strong (recipient) then weak (ablation donor); "
                         "default: tabpfn tabdpt")
    ap.add_argument("--dataset", default="credit-g")
    ap.add_argument("--mode", choices=["matched", "raw"], default="matched",
                    help="matched (default): magnitude-match E1's injection scale; "
                         "raw: map the true ablation delta straight through M")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--matching-file", type=Path, default=DEFAULT_MATCHING_FILE)
    ap.add_argument("--sae-dir", type=Path, default=None)
    ap.add_argument("--importance-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    strong, weak = args.models
    imp_dir = args.importance_dir if args.importance_dir else IMPORTANCE_DIR
    out_dir = (args.output_dir if args.output_dir else OUTPUT_DIR) / f"{strong}_from_{weak}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"HARM-VIA-TRANSFER [{args.mode}]  weak={weak} -> strong={strong}  [{args.dataset}]")
    result = run(strong, weak, args.dataset, args.device,
                 args.matching_file, args.sae_dir, imp_dir, mode=args.mode)

    arrays = result.pop("_arrays")

    stem = f"{args.dataset}_{args.mode}"
    (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(out_dir / f"{stem}.npz", **arrays)

    # Long-form CSV: one row per firing (row, concept). No aggregation.
    # delta_p_true_removal = sign-corrected ablation (flip-aware); _ablate/_inject = raw +/-.
    csv_path = out_dir / f"{stem}.csv"
    cols = ["row", "weak_concept", "concept_flipped", "h_activation", "y_true",
            "base_p_true", "delta_p_true_removal",
            "delta_p_true_ablate", "delta_p_true_inject"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for rec in result["rows"]:
            f.write(",".join(str(rec[c]) for c in cols) + "\n")

    print(f"\n{'='*70}\nPER-ROW ABLATION->TRANSFER DELTAS [{args.mode}]: "
          f"{weak} -> {strong} [{args.dataset}]\n{'='*70}")
    if result["n_lower"] == 0:
        print("  No lower-triangle rows; nothing to test.")
    else:
        R = result["rows"]
        rem = np.array([r["delta_p_true_removal"] for r in R])
        print(f"  lower-triangle rows (strong {strong} wins): {result['n_lower']}/{result['n_query']}  "
              f"| candidate concepts: {result['n_candidates']}  | map R²={result['map_r2']:.3f}")
        print(f"  firing (row, concept) events: {result['n_firing_events']}")
        print(f"\n  Sign-corrected REMOVAL (flip-aware ablation; Δp(y) < 0 == removal harmed strong):")
        print(f"    events harmed (Δ<0): {(rem<0).sum()}/{len(rem)}   "
              f"helped (Δ>0): {(rem>0).sum()}   |  worst harm Δp(y)={rem.min():+.4f}  "
              f"events harm>0.02: {(rem< -0.02).sum()}")
        print(f"\n  Top-15 removal-harm events:")
        print(f"  {'row':>4} {'weak_c':>7} {'flip':>4} {'h_act':>7} {'y':>2} "
              f"{'base_p(y)':>10} {'Δp(y)|removal':>14}")
        for rec in sorted(R, key=lambda r: r["delta_p_true_removal"])[:15]:
            print(f"  {rec['row']:>4d} {rec['weak_concept']:>7d} "
                  f"{'Y' if rec['concept_flipped'] else 'n':>4} "
                  f"{rec['h_activation']:>7.3f} {rec['y_true']:>2d} "
                  f"{rec['base_p_true']:>10.4f} {rec['delta_p_true_removal']:>+14.4f}")
    print(f"\nWrote {out_dir}/{stem}.{{json,npz,csv}}")


if __name__ == "__main__":
    main()
