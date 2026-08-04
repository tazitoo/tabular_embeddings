#!/usr/bin/env python3
"""Can we rebuild the cached deployed_delta from its ingredients?

The patch readout is a counterfactual on the deployed delta: suppress concept c in the
donor, and the delta loses c's term. That rests on the delta being

    delta_r = sum_j  sign_j * a_j * v_j * std_w        (transfer_sweep_symmetric.py:598, :749)

with a_j the donor SAE activation, v_j the virtual atom, std_w the recipient's
per-dataset norm-stat std. Every ingredient is on disk, so before building anything on
this formula we should reproduce the cached deployed_delta from it. If that fails, the
counterfactual is wrong and every downstream number is void.

The greedy tried both signs and only feature ids were saved, so signs are recovered
rather than assumed: solve delta_r = sum_j c_j * (v_j * std_w) by least squares over the
k accepted atoms, then check |c_j| == a_j. With k <= 20 coefficients in d = 300-768
dimensions the system is heavily overdetermined, so agreement is not luck -- it is
simultaneously a check on the formula, the atom indexing, and the sign recovery.

Second use: substitute a different draw's activations for a_j (tabdpt's corpus draw is
unreproducible, see tabdpt_seed_search.py) and measure how far the delta moves. The
delta, not the raw activation, is what reaches the recipient, and it may be far more
draw-stable since only accepted concepts contribute.

Usage:
    python -m scripts.rebuttal.validate_delta_reconstruction
    python -m scripts.rebuttal.validate_delta_reconstruction --donor tabdpt
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import load_norm_stats, load_sae, load_test_embeddings

FWD = PROJECT_ROOT / "output" / "rebuttal" / "forward_deltas"
ATOMS = PROJECT_ROOT / "output" / "transfer_caches" / "global_trained"


def donor_acts(donor, dataset, device="cpu"):
    """a_j exactly as the transfer computed it: sae.encode(cached embeddings)."""
    sae, _ = load_sae(donor, device=device)
    Xn = np.asarray(load_test_embeddings(donor)[dataset], dtype=np.float32)
    with torch.no_grad():
        return sae.encode(torch.tensor(Xn, device=device)).cpu().numpy().astype(np.float64)


def load_atoms(donor, recipient):
    f = ATOMS / f"{donor}_to_{recipient}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    V = np.asarray(z["virtual_atoms"], dtype=np.float64)
    fids = np.asarray(z["feature_ids"])
    return V, {int(fid): i for i, fid in enumerate(fids)}


def reconstruct(npz_path, acts=None, device="cpu"):
    z = np.load(npz_path, allow_pickle=True)
    if "deployed_delta" not in z.files or "selected_features" not in z.files:
        return None
    donor, recipient = str(z["strong_model"]), str(z["weak_model"])
    dataset = os.path.basename(npz_path)[:-4]
    dd = np.asarray(z["deployed_delta"], dtype=np.float64)
    sel = np.asarray(z["selected_features"])
    at = load_atoms(donor, recipient)
    if at is None:
        return None
    V, fmap = at
    _, std_w = load_norm_stats(recipient, dataset, device=device)
    std_w = np.asarray(std_w.cpu(), dtype=np.float64)
    A = donor_acts(donor, dataset, device) if acts is None else acts

    rows, rel_errs, coef_match = [], [], []
    for r in range(len(dd)):
        fids = [int(f) for f in np.unique(sel[r][sel[r] >= 0]) if int(f) in fmap]
        if not fids or not np.abs(dd[r]).any():
            continue
        B = np.stack([V[fmap[f]] * std_w for f in fids])          # (k, d)
        c, *_ = np.linalg.lstsq(B.T, dd[r], rcond=None)            # solve for signed coeffs
        resid = float(np.linalg.norm(dd[r] - B.T @ c) / (np.linalg.norm(dd[r]) + 1e-12))
        a = np.array([A[r, f] for f in fids])
        # |c_j| should equal a_j; compare on the larger coefficients where it is meaningful
        big = np.abs(a) > 1e-6
        m = (float(np.median(np.abs(np.abs(c[big]) - a[big]) / np.maximum(a[big], 1e-9)))
             if big.any() else np.nan)
        rows.append(r); rel_errs.append(resid); coef_match.append(m)

    if not rows:
        return None
    return {
        "file": os.path.basename(npz_path), "donor": donor, "recipient": recipient,
        "dataset": dataset, "n_rows_with_delta": len(rows),
        "recon_rel_err_median": float(np.median(rel_errs)),
        "recon_rel_err_p95": float(np.percentile(rel_errs, 95)),
        "recon_rel_err_max": float(np.max(rel_errs)),
        "coef_vs_activation_median_rel": float(np.nanmedian(coef_match)),
    }


def reextracted_acts(donor, dataset, seed, device):
    """a_j from a re-extraction at a pinned seed, instead of the cached corpus draw."""
    from scripts.intervention.intervene_lib import load_dataset_context
    from scripts.rebuttal.patch_baseline_gate import layer_embeddings

    X_train, y_train, X_query, _, _, task = load_dataset_context(
        donor, dataset, query_source="holdout")
    emb, _, _ = layer_embeddings(donor, dataset, X_train, y_train, X_query, task)
    mean, std = load_norm_stats(donor, dataset, device=device)
    Xn = (emb - mean.cpu().numpy()) / std.cpu().numpy()
    sae, _ = load_sae(donor, device=device)
    with torch.no_grad():
        return sae.encode(torch.tensor(np.asarray(Xn, dtype=np.float32),
                                       device=device)).cpu().numpy().astype(np.float64)


def substitute_test(npz_path, seed, device):
    """How far does the injected delta move if a_j comes from a different draw?

    Signs and the accepted set are held fixed at their cached values; only the donor
    activations change. The delta -- not the raw activation -- is what reaches the
    recipient, so this is the quantity that decides whether a donor whose corpus draw is
    unreproducible can still be patched against the published deltas.
    """
    z = np.load(npz_path, allow_pickle=True)
    donor, recipient = str(z["strong_model"]), str(z["weak_model"])
    dataset = os.path.basename(npz_path)[:-4]
    dd = np.asarray(z["deployed_delta"], dtype=np.float64)
    sel = np.asarray(z["selected_features"])
    at = load_atoms(donor, recipient)
    if at is None:
        return None
    V, fmap = at
    _, std_w = load_norm_stats(recipient, dataset, device=device)
    std_w = np.asarray(std_w.cpu(), dtype=np.float64)

    A0 = donor_acts(donor, dataset, device)
    A1 = reextracted_acts(donor, dataset, seed, device)
    n = min(len(A0), len(A1), len(dd))

    rel, cos = [], []
    for r in range(n):
        fids = [int(f) for f in np.unique(sel[r][sel[r] >= 0]) if int(f) in fmap]
        if not fids or not np.abs(dd[r]).any():
            continue
        B = np.stack([V[fmap[f]] * std_w for f in fids])
        c, *_ = np.linalg.lstsq(B.T, dd[r], rcond=None)
        signs = np.sign(c)
        a1 = np.array([A1[r, f] for f in fids])
        d1 = (signs * a1) @ B
        rel.append(float(np.linalg.norm(d1 - dd[r]) / (np.linalg.norm(dd[r]) + 1e-12)))
        cos.append(float(d1 @ dd[r] / (np.linalg.norm(d1) * np.linalg.norm(dd[r]) + 1e-12)))
    if not rel:
        return None
    return {"donor": donor, "recipient": recipient, "dataset": dataset, "seed": seed,
            "n_rows": len(rel), "delta_rel_change_median": float(np.median(rel)),
            "delta_rel_change_p95": float(np.percentile(rel, 95)),
            "delta_cos_median": float(np.median(cos)), "delta_cos_min": float(np.min(cos))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", default=None, help="restrict to this donor model")
    ap.add_argument("--substitute-seed", type=int, default=None,
                    help="re-extract donor activations at this seed and report how far "
                         "the rebuilt delta moves from the cached one")
    ap.add_argument("--max-files", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(
        PROJECT_ROOT / "output" / "rebuttal" / "delta_reconstruction.json"))
    args = ap.parse_args()

    files = sorted(glob.glob(str(FWD / "*" / "*.npz")))
    if args.donor:
        files = [f for f in files if os.path.basename(os.path.dirname(f)).startswith(args.donor + "_vs_")]
    files = files[:args.max_files]

    if args.substitute_seed is not None:
        print(f"substituting seed-{args.substitute_seed} activations into {len(files)} "
              f"cached deltas (accepted set and signs held fixed)\n")
        print(f"  {'pair/dataset':46s} {'rows':>5s} {'delta rel chg':>14s} {'p95':>9s} {'cos':>8s}")
        subs = []
        for f in files:
            try:
                r = substitute_test(f, args.substitute_seed, args.device)
            except Exception as exc:
                print(f"  {os.path.basename(f):46s} ERROR {type(exc).__name__}: {exc}")
                continue
            if r is None:
                continue
            subs.append(r)
            tag = f"{r['donor']}->{r['recipient']}/{r['dataset']}"
            print(f"  {tag:46s} {r['n_rows']:>5d} {r['delta_rel_change_median']:14.4f} "
                  f"{r['delta_rel_change_p95']:9.4f} {r['delta_cos_median']:8.4f}")
        if subs:
            print(f"\n  median delta relative change: "
                  f"{np.median([s['delta_rel_change_median'] for s in subs]):.4f}")
            print(f"  median delta cosine to cached: "
                  f"{np.median([s['delta_cos_median'] for s in subs]):.4f}")
            print("\n  Small change => the injected delta is draw-stable even though the raw")
            print("  activations are not, and this donor can be patched against published deltas.")
        with open(args.out.replace(".json", f"_subseed{args.substitute_seed}.json"), "w") as fh:
            json.dump(subs, fh, indent=2)
        return

    print(f"reconstructing deployed_delta for {len(files)} files\n")
    print(f"  {'pair/dataset':46s} {'rows':>5s} {'rel_err med':>12s} {'p95':>9s} {'|c| vs a':>10s}")
    out = []
    for f in files:
        try:
            r = reconstruct(f, device=args.device)
        except Exception as exc:
            print(f"  {os.path.basename(f):46s} ERROR {type(exc).__name__}: {exc}")
            continue
        if r is None:
            continue
        out.append(r)
        tag = f"{r['donor']}->{r['recipient']}/{r['dataset']}"
        print(f"  {tag:46s} {r['n_rows_with_delta']:>5d} {r['recon_rel_err_median']:12.2e} "
              f"{r['recon_rel_err_p95']:9.2e} {r['coef_vs_activation_median_rel']:10.2e}")

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    if out:
        med = float(np.median([r["recon_rel_err_median"] for r in out]))
        cm = float(np.nanmedian([r["coef_vs_activation_median_rel"] for r in out]))
        print(f"\n  median reconstruction rel err across files: {med:.2e}")
        print(f"  median |coefficient| vs activation rel diff:  {cm:.2e}")
        print("\n  Near-zero on both => the delta formula, atom indexing and sign recovery")
        print("  are all confirmed, and the patch counterfactual can be built on them.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
