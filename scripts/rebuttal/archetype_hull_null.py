#!/usr/bin/env python3
"""App F.3: what hull residual do REAL rows have? (the in-distribution null)

Input-level patches are found by search. A search is free to suppress a concept by
pushing the row off the data manifold -- that is the optimizer exploiting the loss,
not evidence about the concept, so such a patch must be counted as "no qualifying
patch found", not as coverage. To enforce that we need a principled notion of "off
the manifold", and a null to compare against.

The SAEs here are Matryoshka-Archetypal: the dictionary is built as D = WC + L, where
C = `sae.reference_data` (K-means centroids of the training embeddings, persisted in
the checkpoint). conv(C) is therefore the SAE's own operational definition of where
the data lives. For an embedding e:

    r(e) = min ||e - C^T w||_2   s.t.  w >= 0, sum(w) = 1

This script computes r(e) for REAL, unpatched rows, so a later patch search can be
judged against the range real data actually occupies rather than an invented epsilon.

Two reference sets are reported, because they bound different things:
  global    : sae.reference_data -- pooled over all datasets the SAE trained on. An
              embedding can sit in this hull while being atypical for its own dataset.
  perdataset: that dataset's own cached test embeddings -- the tighter, more
              meaningful bound.

All CPU, no model forward passes.

Usage:
    python -m scripts.rebuttal.archetype_hull_null
"""
import argparse
import json

import numpy as np
from scipy.optimize import nnls

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import load_sae, load_test_embeddings

DEFAULT_PROBES = [
    ("tabpfn", "heloc"), ("tabicl", "Bioresponse"), ("tabicl_v2", "miami_housing"),
    ("mitra", "APSFailure"), ("tabdpt", "miami_housing"),
]


def hull_residual(E: np.ndarray, C: np.ndarray, penalty: float = 1e3) -> np.ndarray:
    """Distance from each row of E to conv(C).

    Simplex-constrained least squares via NNLS on an augmented system: appending a
    row of `penalty` to C^T and `penalty` to the target drives sum(w) -> 1 while NNLS
    supplies w >= 0.
    """
    A = np.vstack([C.T, np.full((1, C.shape[0]), penalty)])
    out = np.empty(len(E))
    for i, e in enumerate(E):
        b = np.concatenate([e, [penalty]])
        w, _ = nnls(A, b)
        out[i] = float(np.linalg.norm(e - C.T @ w))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=200,
                    help="rows per dataset to profile (NNLS is the cost)")
    ap.add_argument("--probes", nargs="*", default=None,
                    help="model:dataset pairs; default = one per donor")
    ap.add_argument("--out", default=str(
        PROJECT_ROOT / "output" / "rebuttal" / "archetype_hull_null.json"))
    args = ap.parse_args()

    probes = ([tuple(p.split(":", 1)) for p in args.probes] if args.probes
              else DEFAULT_PROBES)

    results = []
    for model, dataset in probes:
        sae, _ = load_sae(model, device="cpu")
        C = getattr(sae, "reference_data", None)
        if C is None:
            print(f"{model}: no reference_data -- not an archetypal SAE, skipping")
            continue
        C = np.asarray(C.detach().cpu(), dtype=np.float64)
        E = np.asarray(load_test_embeddings(model)[dataset], dtype=np.float64)
        E = E[:args.max_rows]
        scale = float(np.linalg.norm(E, axis=1).mean())

        r_glob = hull_residual(E, C)
        r_self = hull_residual(E, E)   # per-dataset hull: rows vs their own dataset

        entry = {
            "model": model, "dataset": dataset, "n_rows": len(E),
            "emb_dim": E.shape[1], "n_ref": len(C), "mean_emb_norm": scale,
            "global": {"median": float(np.median(r_glob)), "p95": float(np.percentile(r_glob, 95)),
                        "max": float(r_glob.max()),
                        "median_rel": float(np.median(r_glob) / scale)},
            "perdataset": {"median": float(np.median(r_self)), "p95": float(np.percentile(r_self, 95)),
                            "max": float(r_self.max()),
                            "median_rel": float(np.median(r_self) / scale)},
        }
        results.append(entry)
        print(f"{model:10s} {dataset:22s} dim={E.shape[1]:4d} n_ref={len(C):5d} "
              f"|e|~{scale:7.2f}")
        for k in ("global", "perdataset"):
            b = entry[k]
            print(f"     {k:11s} r: median={b['median']:8.3f}  p95={b['p95']:8.3f}  "
                  f"max={b['max']:8.3f}   (median/|e| = {b['median_rel']:.3f})")

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")
    print("A patched embedding is IN-DISTRIBUTION only if its residual sits inside the")
    print("range real rows occupy above. Suppression achieved outside it is the search")
    print("exploiting the objective -- count it as 'no qualifying patch found'.")


if __name__ == "__main__":
    main()
