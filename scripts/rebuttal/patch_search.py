#!/usr/bin/env python3
"""App F.3: search input-level suppression patches, selectively, and read out the effect.

For a concept c = (donor, feat_id) accepted into a recipient, find a small edit to the
donor's input row that lowers c's activation while leaving the other k-1 concepts
accepted at that row alone, and report what that does to the recipient's prediction.

Design decisions this encodes (each settled by measurement earlier):

  patch site   Donor rows. The concept is a donor SAE feature; the recipient has no
               such feature.
  search space (column, value) pairs, values drawn ONLY from that column's observed
               support. No contrastive-example CSVs: they exist for 2 of 335 concepts,
               and their non-firing contrast requirement disqualifies the 76 always-on
               concepts outright (firing density >= 0.967) even though those have ample
               dynamic range to suppress (mean activation ~1/3 of max).
  columns      Ranked by rank-correlation with the concept's CACHED activation -- free,
               no forward passes, and better targeted than contrast separation.
  selectivity  delta_r = sum_j sign_j * a_j * v_j * std_w is LINEAR in a_j (verified to
               3.1e-05 by validate_delta_reconstruction.py). If only a_c moves, the
               delta change is exactly c's term and the recipient effect is attributable
               to c. If others move, it is a mixture. So disturbance to the other k-1 is
               recorded for every candidate, and optionally enforced as a CONSTRAINT
               (not a penalty with a lambda to tune).
  in-sample    SAE reconstruction error, against the real-row range from
               sae_insample_null.py. A patch that suppresses c by leaving the region the
               dictionary can represent is the search exploiting the objective, not
               evidence about c -- it counts as "no qualifying patch found".
  readout      Counterfactual delta rebuilt from the CORPUS activation scaled by the
               measured suppression ratio, so published deltas stay intact and only a
               dimensionless ratio comes from re-extraction.
  rows         Sampled across the accepted rows. sae_test rows are ordered/clustered
               (first-16 ||mean||=8.65 vs 1.17 over 200), so "first N" is a biased draw.

tabdpt is excluded: its corpus draw is unreproducible and substituting any reproducible
draw moves the injected delta by ~59% (vs 0.13% for a deterministic donor).

Usage:
    python -m scripts.rebuttal.patch_search --probe --device cuda
"""
import argparse
import glob
import json
import os
from collections import defaultdict

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import (
    SPLITS_PATH, build_tail, get_extraction_layer_taskaware, load_dataset_context,
    load_norm_stats, load_sae, load_test_embeddings, batched_intervention,
)

FWD = PROJECT_ROOT / "output" / "rebuttal" / "forward_deltas"
ATOMS = PROJECT_ROOT / "output" / "transfer_caches" / "global_trained"
EXTRACT_SEED = 13
EXCLUDED_DONORS = {"tabdpt"}

# stratified across donors and both firing-density regimes; tabdpt excluded
PROBE_CONCEPTS = [
    ("tabicl", 96), ("tabicl_v2", 228), ("mitra", 107), ("tabpfn", 56), ("tabicl", 158),
]


# ── extraction ───────────────────────────────────────────────────────────────

def _reseed():
    torch.manual_seed(EXTRACT_SEED); np.random.seed(EXTRACT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(EXTRACT_SEED)


def extract_acts(donor, dataset, X_ctx, y_ctx, X_query, task, device):
    """SAE activations + relative reconstruction error for a batch of query rows."""
    from models.layer_extraction import extract_all_layers, load_and_fit, sort_layer_names
    _reseed()
    clf = load_and_fit(donor, X_ctx, y_ctx, task=task, device=device)
    embs = extract_all_layers(donor, clf, X_query, task=task, seed=EXTRACT_SEED)
    names = sort_layer_names(list(embs.keys()))
    idx = min(max(get_extraction_layer_taskaware(donor, dataset), 0), len(names) - 1)
    raw = np.asarray(embs[names[idx]], dtype=np.float32)
    mean, std = load_norm_stats(donor, dataset, device=device)
    sae, _ = load_sae(donor, device=device)
    with torch.no_grad():
        x = (torch.tensor(raw, device=device) - mean) / std
        h = sae.encode(x)
        xh = sae.decode(h)
        rel = (torch.linalg.norm(x - xh, dim=1) /
               torch.linalg.norm(x, dim=1).clamp_min(1e-8)).cpu().numpy()
        return h.cpu().numpy().astype(np.float64), rel


# ── search space ─────────────────────────────────────────────────────────────

def check_independence(donor, dataset, X_ctx, y_ctx, X_query, task, device, n_dup=8):
    """Does appending candidate rows change the base rows' activations?

    The search evaluates a whole greedy step in ONE forward by appending every candidate
    as an extra query row. That is only valid if query rows do not influence each other.
    Verified rather than assumed: embed the base set, then the base set with `n_dup` of
    its own rows appended, and check (a) the base rows are unchanged and (b) each
    duplicate matches its original.
    """
    a_base, _ = extract_acts(donor, dataset, X_ctx, y_ctx, X_query, task, device)
    dup_idx = np.linspace(0, len(X_query) - 1, n_dup).astype(int)
    Xq = np.vstack([X_query, X_query[dup_idx]])
    a_aug, _ = extract_acts(donor, dataset, X_ctx, y_ctx, Xq, task, device)
    base_shift = float(np.abs(a_aug[:len(X_query)] - a_base).max())
    dup_shift = float(np.abs(a_aug[len(X_query):] - a_base[dup_idx]).max())
    return {"base_rows_max_shift": base_shift, "duplicate_vs_original_max_shift": dup_shift,
            "independent": bool(base_shift < 1e-4 and dup_shift < 1e-4)}


def rank_columns(X: np.ndarray, a: np.ndarray, top_k: int) -> list[int]:
    """Columns ranked by |rank correlation| with the concept's activation. No forwards."""
    def rank(v):
        o = np.argsort(np.argsort(v))
        return (o - o.mean()) / (o.std() + 1e-12)
    ra = rank(a)
    scores = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.nanstd(col) < 1e-12:
            scores.append(0.0); continue
        scores.append(abs(float(np.mean(rank(np.nan_to_num(col)) * ra))))
    return [int(j) for j in np.argsort(-np.asarray(scores))[:top_k]]


def candidate_values(X: np.ndarray, col: int, max_vals: int) -> np.ndarray:
    """Values from the column's OBSERVED support -- never invents a value."""
    v = np.unique(X[~np.isnan(X[:, col]), col])
    if len(v) <= max_vals:
        return v
    return np.unique(np.quantile(v, np.linspace(0.0, 1.0, max_vals)))


# ── the search ───────────────────────────────────────────────────────────────

def search_row(donor, dataset, X_ctx, y_ctx, X_query, task, device, row, feat,
               others, cols, max_vals, max_steps, sel_tol, recon_bar):
    """Greedy (column, value) suppression for one row, batching all candidates."""
    base = X_query.copy()
    cur = base[row].copy()
    a0, rel0 = extract_acts(donor, dataset, X_ctx, y_ctx, base, task, device)
    a_start = float(a0[row, feat])
    hist, used = [], set()
    a_now = a_start
    stop = "max_steps"

    for _ in range(max_steps):
        variants, meta = [], []
        for c in cols:
            if c in used:
                continue
            for val in candidate_values(X_query, c, max_vals):
                if np.isclose(val, cur[c], equal_nan=True):
                    continue
                r = cur.copy(); r[c] = val
                variants.append(r); meta.append((c, float(val)))
        if not variants:
            stop = "no_candidates"; break

        # append candidates as extra query rows: one forward for the whole step
        Xq = np.vstack([base, np.asarray(variants, dtype=base.dtype)])
        a, rel = extract_acts(donor, dataset, X_ctx, y_ctx, Xq, task, device)
        off = len(base)
        best = None
        for i, (c, val) in enumerate(meta):
            av = a[off + i]
            drop = a_now - float(av[feat])
            dist = (float(np.max(np.abs(av[others] - a0[row][others]))) if len(others) else 0.0)
            ok_sel = sel_tol is None or dist <= sel_tol
            ok_rec = recon_bar is None or float(rel[off + i]) <= recon_bar
            cand = {"column": int(c), "value": val, "activation_after": float(av[feat]),
                    "drop": drop, "max_other_shift": dist,
                    "recon_rel": float(rel[off + i]),
                    "qualifies": bool(ok_sel and ok_rec)}
            if drop > 0 and ok_sel and ok_rec and (best is None or drop > best["drop"]):
                best = cand
        if best is None:
            stop = "no_qualifying_column"; break
        cur[best["column"]] = best["value"]
        used.add(best["column"])
        a_now = best["activation_after"]
        hist.append(best)
        if a_now <= 0:
            stop = "fully_suppressed"; break

    return {"row": int(row), "a_start": a_start, "a_final": a_now,
            "ratio": (a_now / a_start) if a_start > 0 else float("nan"),
            "drop_frac": (1.0 - a_now / a_start) if a_start > 0 else float("nan"),
            "recon_rel_start": float(rel0[row]), "steps": hist, "stop_reason": stop,
            "patched_row": cur.tolist(), "n_cols_changed": len(used)}


def placebo_row(donor, dataset, X_ctx, y_ctx, X_query, task, device, row, feat,
                others, cols_low, max_vals):
    """Edit a LOW-association column: what does touching the row at all cost?

    This is the calibrator for the selectivity tolerance -- the repeat null is exactly
    zero now that extraction is deterministic, so it cannot set a meaningful bar.
    """
    base = X_query.copy()
    a0, _ = extract_acts(donor, dataset, X_ctx, y_ctx, base, task, device)
    variants, meta = [], []
    for c in cols_low:
        for val in candidate_values(X_query, c, max_vals):
            if np.isclose(val, base[row, c], equal_nan=True):
                continue
            r = base[row].copy(); r[c] = val
            variants.append(r); meta.append((int(c), float(val)))
    if not variants:
        return None
    Xq = np.vstack([base, np.asarray(variants, dtype=base.dtype)])
    a, rel = extract_acts(donor, dataset, X_ctx, y_ctx, Xq, task, device)
    off = len(base)
    drops = [float(a0[row, feat] - a[off + i, feat]) for i in range(len(meta))]
    shifts = [float(np.max(np.abs(a[off + i][others] - a0[row][others]))) if len(others) else 0.0
              for i in range(len(meta))]
    return {"n": len(meta),
            "drop_median": float(np.median(drops)), "drop_p95": float(np.percentile(drops, 95)),
            "drop_max": float(np.max(drops)),
            "other_shift_median": float(np.median(shifts)),
            "other_shift_p95": float(np.percentile(shifts, 95)),
            "recon_rel_median": float(np.median(rel[off:]))}


# ── cell selection and readout ───────────────────────────────────────────────

def cells_for_concept(donor, feat, min_rows):
    """(recipient, dataset, accepted rows) where this concept was actually deployed."""
    out = []
    for f in sorted(glob.glob(str(FWD / "*" / "*.npz"))):
        try:
            z = np.load(f, allow_pickle=True)
        except Exception:
            continue
        if "selected_features" not in z.files or str(z["strong_model"]) != donor:
            continue
        sel = np.asarray(z["selected_features"])
        if sel.size == 0:
            continue
        rows = [r for r in range(sel.shape[0]) if feat in set(sel[r][sel[r] >= 0].tolist())]
        if len(rows) >= min_rows:
            out.append((str(z["weak_model"]), os.path.basename(f)[:-4], rows, f))
    out.sort(key=lambda t: -len(t[2]))
    return out


def readout(npz_path, feat, row, ratio, device):
    """Recipient prediction under the counterfactual delta (c's term scaled by `ratio`).

    Published deltas are untouched: a_c comes from the CORPUS activation and only the
    dimensionless ratio comes from the patch.
    """
    from scripts.rebuttal.functional_decomposition import _gc
    from scripts.intervention.intervene_sae import SEQUENTIAL_MODELS
    from scripts.intervention.ablation_sweep import batched_ablation_sequential

    z = np.load(npz_path, allow_pickle=True)
    donor, recipient = str(z["strong_model"]), str(z["weak_model"])
    dataset = os.path.basename(npz_path)[:-4]
    dd = np.asarray(z["deployed_delta"], dtype=np.float64)
    sel = np.asarray(z["selected_features"])
    zc = np.load(ATOMS / f"{donor}_to_{recipient}.npz", allow_pickle=True)
    V = np.asarray(zc["virtual_atoms"], dtype=np.float64)
    fmap = {int(f): i for i, f in enumerate(np.asarray(zc["feature_ids"]))}
    _, std_w = load_norm_stats(recipient, dataset, device=device)
    std_w = np.asarray(std_w.cpu(), dtype=np.float64)

    sae, _ = load_sae(donor, device=device)
    with torch.no_grad():
        A = sae.encode(torch.tensor(np.asarray(load_test_embeddings(donor)[dataset],
                                               dtype=np.float32),
                                    device=device)).cpu().numpy().astype(np.float64)

    fids = [int(f) for f in np.unique(sel[row][sel[row] >= 0]) if int(f) in fmap]
    if feat not in fids:
        return None
    B = np.stack([V[fmap[f]] * std_w for f in fids])
    c, *_ = np.linalg.lstsq(B.T, dd[row], rcond=None)
    sign_c = float(np.sign(c[fids.index(feat)]))
    term = sign_c * A[row, feat] * V[fmap[feat]] * std_w
    d_cf = dd[row] - (1.0 - ratio) * term
    purity = float(np.linalg.norm((1.0 - ratio) * term) /
                   (np.linalg.norm(dd[row] - d_cf) + 1e-12))

    splits = json.loads(SPLITS_PATH.read_text())
    Xtr, ytr, Xq, _, _, task = load_dataset_context(recipient, dataset, splits)
    if ytr.dtype == np.int32:
        ytr = ytr.astype(np.int64)
    layer = get_extraction_layer_taskaware(recipient, dataset=dataset)
    cat_idx = None
    if recipient in ("hyperfast", "tabpfn"):
        from data.preprocessing import load_preprocessed, CACHE_DIR
        try:
            cat_idx = load_preprocessed(recipient, dataset, CACHE_DIR).cat_indices or None
        except Exception:
            pass
    _reseed()
    tail = build_tail(recipient, Xtr, ytr, Xq, layer, task, device, cat_indices=cat_idx,
                      target_name=splits.get(dataset, {}).get("target", "target"))
    deltas = torch.tensor(np.vstack([dd[row], d_cf]), dtype=torch.float32, device=device)
    if isinstance(tail, SEQUENTIAL_MODELS):
        preds = np.asarray(batched_ablation_sequential(tail, Xq[row:row+1], deltas, query_idx=row),
                           dtype=np.float64)
    else:
        preds = np.asarray(batched_intervention(tail, Xq[row:row+1], deltas, inject_context=False),
                           dtype=np.float64)
    y = int(np.asarray(z["y_query"])[row])
    b, t = np.asarray(z["preds_weak"])[row], np.asarray(z["preds_strong"])[row]
    return {"recipient": recipient, "dataset": dataset, "row": int(row),
            "gc_deployed": float(_gc(b, preds[0], t, y)),
            "gc_counterfactual": float(_gc(b, preds[1], t, y)),
            "attribution_purity": purity, "sign_c": sign_c,
            "a_c_corpus": float(A[row, feat]), "ratio_applied": float(ratio)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="run the 5 stratified concepts")
    ap.add_argument("--concepts", nargs="*", default=None, help="donor:feat pairs")
    ap.add_argument("--rows-per-concept", type=int, default=2)
    ap.add_argument("--top-cols", type=int, default=6)
    ap.add_argument("--max-vals", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--min-rows", type=int, default=8)
    ap.add_argument("--selectivity-tol", type=float, default=None,
                    help="max allowed shift in the other k-1; omit to record only "
                         "(the probe measures the placebo null that sets this)")
    ap.add_argument("--recon-bar", type=float, default=None)
    ap.add_argument("--check-independence", action="store_true",
                    help="verify query rows do not influence each other before trusting "
                         "the batched candidate evaluation")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output" / "rebuttal" / "patch_search.json"))
    args = ap.parse_args()

    concepts = (PROBE_CONCEPTS if args.probe else
                [(c.split(":")[0], int(c.split(":")[1])) for c in (args.concepts or [])])
    torch.use_deterministic_algorithms(True)

    results = []
    for donor, feat in concepts:
        if donor in EXCLUDED_DONORS:
            print(f"\n{donor} f{feat}: SKIPPED (donor excluded)"); continue
        cells = cells_for_concept(donor, feat, args.min_rows)
        if not cells:
            print(f"\n{donor} f{feat}: no cell with >= {args.min_rows} accepted rows")
            results.append({"donor": donor, "feat": feat, "status": "no_cell"}); continue
        recipient, dataset, acc_rows, npz_path = cells[0]
        print(f"\n{donor} f{feat} -> {recipient} / {dataset}  "
              f"({len(acc_rows)} accepted rows, {len(cells)} cells)", flush=True)

        X_ctx, y_ctx, X_query, _, _, task = load_dataset_context(donor, dataset,
                                                                 query_source="holdout")
        if hasattr(X_query, "iloc"):
            print("    donor is a DataFrame model -- not supported here"); continue
        sae, _ = load_sae(donor, device=args.device)
        with torch.no_grad():
            A = sae.encode(torch.tensor(np.asarray(load_test_embeddings(donor)[dataset],
                                                   dtype=np.float32),
                                        device=args.device)).cpu().numpy().astype(np.float64)
        cols = rank_columns(X_query, A[:, feat], args.top_cols)
        # placebo columns: the LOWEST-association ones, i.e. the tail of the same ranking
        all_ranked = rank_columns(X_query, A[:, feat], X_query.shape[1])
        low = [j for j in reversed(all_ranked) if j not in cols][:2]

        if args.check_independence:
            ind = check_independence(donor, dataset, X_ctx, y_ctx, X_query, task, args.device)
            print(f"    independence: base shift={ind['base_rows_max_shift']:.2e} "
                  f"dup shift={ind['duplicate_vs_original_max_shift']:.2e} -> "
                  f"{'OK' if ind['independent'] else 'ROWS INTERACT (batching invalid)'}",
                  flush=True)

        z = np.load(npz_path, allow_pickle=True)
        sel = np.asarray(z["selected_features"])

        # sample rows ACROSS the accepted set -- sae_test rows are ordered
        pick = (np.linspace(0, len(acc_rows) - 1, args.rows_per_concept).astype(int)
                if len(acc_rows) > args.rows_per_concept else np.arange(len(acc_rows)))
        entry = {"donor": donor, "feat": feat, "recipient": recipient, "dataset": dataset,
                 "n_accepted_rows": len(acc_rows), "ranked_columns": cols,
                 "rows": [], "placebo": None}
        for pi in pick:
            row = int(acc_rows[int(pi)])
            others = np.array([int(f) for f in np.unique(sel[row][sel[row] >= 0])
                               if int(f) != feat], dtype=int)
            res = search_row(donor, dataset, X_ctx, y_ctx, X_query, task, args.device,
                             row, feat, others, cols, args.max_vals, args.max_steps,
                             args.selectivity_tol, args.recon_bar)
            res["n_other_concepts"] = int(len(others))
            print(f"    row {row}: a {res['a_start']:.3f} -> {res['a_final']:.3f} "
                  f"(drop {res['drop_frac']:.1%}, {res['n_cols_changed']} cols, "
                  f"{res['stop_reason']})", flush=True)
            if np.isfinite(res["ratio"]):
                try:
                    res["readout"] = readout(npz_path, feat, row, res["ratio"], args.device)
                    if res["readout"]:
                        r = res["readout"]
                        print(f"       recipient gc {r['gc_deployed']:.4f} -> "
                              f"{r['gc_counterfactual']:.4f}  purity={r['attribution_purity']:.3f}",
                              flush=True)
                except Exception as exc:
                    res["readout"] = {"error": f"{type(exc).__name__}: {exc}"}
                    print(f"       readout ERROR {type(exc).__name__}: {exc}", flush=True)
            entry["rows"].append(res)

        try:
            row0 = int(acc_rows[int(pick[0])])
            others0 = np.array([int(f) for f in np.unique(sel[row0][sel[row0] >= 0])
                                if int(f) != feat], dtype=int)
            entry["placebo"] = placebo_row(donor, dataset, X_ctx, y_ctx, X_query, task,
                                           args.device, row0, feat, others0, low, args.max_vals)
            if entry["placebo"]:
                p = entry["placebo"]
                print(f"    placebo ({p['n']} edits to low-association cols): "
                      f"drop median={p['drop_median']:.4f} p95={p['drop_p95']:.4f}  "
                      f"other-shift p95={p['other_shift_p95']:.4f}", flush=True)
        except Exception as exc:
            entry["placebo"] = {"error": f"{type(exc).__name__}: {exc}"}
        results.append(entry)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
