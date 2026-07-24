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
from sklearn.metrics import roc_auc_score

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
        matching_file: Path, sae_dir: Path, imp_dir: Path) -> dict:
    """Transfer single matched-concept ablation deltas weak->strong on the
    lower-triangle rows; record per-concept harm to the strong model."""
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

    # Candidate concepts = matched WEAK indices, deduped, ranked by weak importance.
    imp_rank = _weak_importance_rank(imp_weak)
    cand = sorted(set(src_idx), key=lambda c: -imp_rank.get(c, 0.0))
    # tgt matches per weak concept (metadata only; not used in the delta)
    strong_matches = {}
    for si, ti in m_pairs:
        strong_matches.setdefault(si, []).append(int(ti))
    logger.info(f"  Candidate matched weak concepts: {len(cand)}")

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

    # ── Per row: one tail pass over [baseline, all candidate deltas] ──────────
    K = len(cand)
    eps = 1e-7
    # harmed_p_true[r_i, k] = strong P(true class) after transferring concept k's
    # ablation delta on lower row r_i; col -1 slot is the zero-delta baseline.
    base_p_true = np.zeros(len(lower), dtype=np.float64)
    harmed_p_true = np.zeros((len(lower), K), dtype=np.float64)
    harmed_p1 = np.zeros((len(lower), K), dtype=np.float64)  # P(class 1) for AUC
    base_p1 = np.zeros(len(lower), dtype=np.float64)

    t1 = time.time()
    for i, r in enumerate(lower):
        # analytic per-concept normalized ablation delta in weak space
        d_norm_src = -(h_weak[r, cand][:, None]) * atoms_weak_cand   # (K, d_weak)
        d_norm_tgt = d_norm_src @ M_t                               # (K, d_strong)
        d_raw = (d_norm_tgt * std_strong[None, :]).astype(np.float32)
        deltas = np.vstack([np.zeros((1, d_raw.shape[1]), np.float32), d_raw])  # (K+1, d_strong)
        deltas_t = torch.tensor(deltas, dtype=torch.float32, device=device)
        preds = batched_intervention(tail_s, X_query_s[r:r + 1], deltas_t,
                                     inject_context=False)  # (K+1, 2)
        preds = np.asarray(preds, dtype=np.float64)
        y_r = int(y_query[r])
        base_p_true[i] = np.clip(preds[0, y_r], eps, 1 - eps)
        base_p1[i] = preds[0, 1]
        harmed_p_true[i] = np.clip(preds[1:, y_r], eps, 1 - eps)
        harmed_p1[i] = preds[1:, 1]
        if (i + 1) % 10 == 0 or i + 1 == len(lower):
            logger.info(f"    row {i+1}/{len(lower)} "
                        f"({(i+1)/(time.time()-t1):.1f} rows/s)")

    # ── Per-concept harm metrics ─────────────────────────────────────────────
    base_loss = -np.log(base_p_true)                       # (n_lower,)
    harmed_loss = -np.log(harmed_p_true)                   # (n_lower, K)
    loss_increase = harmed_loss - base_loss[:, None]       # >0 == harm
    prob_drop = base_p_true[:, None] - harmed_p_true       # >0 == harm

    y_lower = y_query[lower]
    both_classes = len(np.unique(y_lower)) == 2
    base_auc = roc_auc_score(y_lower, base_p1) if both_classes else float("nan")

    concepts = []
    for k in range(K):
        auc_k = roc_auc_score(y_lower, harmed_p1[:, k]) if both_classes else float("nan")
        concepts.append({
            "weak_concept": int(cand[k]),
            "strong_matches": strong_matches.get(cand[k], []),
            "weak_importance": float(imp_rank.get(cand[k], 0.0)),
            "mean_loss_increase": float(loss_increase[:, k].mean()),
            "max_loss_increase": float(loss_increase[:, k].max()),
            "mean_prob_drop": float(prob_drop[:, k].mean()),
            "n_rows_harmed": int((prob_drop[:, k] > 0).sum()),
            "auc_after": float(auc_k),
            "auc_drop": float(base_auc - auc_k) if both_classes else float("nan"),
        })
    concepts.sort(key=lambda c: -c["mean_loss_increase"])

    return {
        "strong": strong, "weak": weak, "dataset": dataset,
        "metric_name": metric_name,
        "metric_strong": float(m_strong), "metric_weak": float(m_weak),
        "n_query": int(n_query), "n_lower": int(len(lower)),
        "map_r2": float(r2), "n_landmarks": int(len(filt_pairs)),
        "base_auc_lower": float(base_auc),
        "n_candidates": int(K),
        "n_concepts_harmful_loss": int(sum(c["mean_loss_increase"] > 0 for c in concepts)),
        "n_concepts_harmful_auc": int(sum((c["auc_drop"] > 0) for c in concepts if not np.isnan(c["auc_drop"]))),
        "concepts": concepts,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs=2, default=["tabpfn", "tabdpt"],
                    metavar=("STRONG", "WEAK"),
                    help="strong (recipient) then weak (ablation donor); "
                         "default: tabpfn tabdpt")
    ap.add_argument("--dataset", default="credit-g")
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

    logger.info(f"HARM-VIA-TRANSFER  weak={weak} -> strong={strong}  [{args.dataset}]")
    result = run(strong, weak, args.dataset, args.device,
                 args.matching_file, args.sae_dir, imp_dir)

    out_path = out_dir / f"{args.dataset}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\n{'='*70}\nHARM VIA WEAK->STRONG TRANSFER: {weak} -> {strong} [{args.dataset}]\n{'='*70}")
    if result["n_lower"] == 0:
        print("  No lower-triangle rows; nothing to test.")
    else:
        print(f"  rows tested (strong wins): {result['n_lower']}/{result['n_query']}  "
              f"| baseline AUC over these rows: {result['base_auc_lower']:.4f}")
        print(f"  matched candidate concepts: {result['n_candidates']}  "
              f"| map R²={result['map_r2']:.3f} ({result['n_landmarks']} landmarks)")
        print(f"  concepts that HARM (mean loss ↑): "
              f"{result['n_concepts_harmful_loss']}/{result['n_candidates']}  "
              f"| that drop AUC: {result['n_concepts_harmful_auc']}/{result['n_candidates']}")
        print(f"\n  Top-10 most harmful concepts (by mean per-row loss increase):")
        print(f"  {'weak_c':>7} {'importⁿ':>8} {'loss↑':>8} {'maxloss↑':>9} "
              f"{'p(y)drop':>9} {'rows✗':>6} {'AUC↓':>7}")
        for c in result["concepts"][:10]:
            print(f"  {c['weak_concept']:>7d} {c['weak_importance']:>8.4f} "
                  f"{c['mean_loss_increase']:>8.4f} {c['max_loss_increase']:>9.4f} "
                  f"{c['mean_prob_drop']:>9.4f} {c['n_rows_harmed']:>6d} "
                  f"{c['auc_drop']:>7.4f}")
        verdict = "YES" if result["n_concepts_harmful_loss"] > 0 else "NO"
        print(f"\n  >> Can weak->strong transfer harm strong? {verdict} "
              f"({result['n_concepts_harmful_loss']} concept(s) impose harm)")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
