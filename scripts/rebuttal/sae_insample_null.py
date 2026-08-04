#!/usr/bin/env python3
"""Is an embedding in-sample for the SAE? (and is a re-extracted draw as good?)

Replaces the convex-hull idea in archetype_hull_null.py, which was category-confused:
get_cached_kmeans is SPHERICAL k-means, so sae.reference_data is a unit-norm codebook
of DIRECTIONS (||C_i||=1), not a region of data space. Real embeddings (norm 14-23)
sit ~0.9*||e|| outside conv(C), so hull distance rejects reality.

The correct in-sample measure follows from the architecture. Reconstruction is
x_hat = h @ D + b_dec with h >= 0 and atoms D = C^T softmax(logits) + L, so the
representable set is the CONIC hull of the dictionary, and the residual against it is
just the SAE's reconstruction error. If the SAE cannot reconstruct an embedding, any
concept activation read off it is not a trustworthy reading of anything -- which is
exactly the guard a patch search needs.

Two questions here:
  1. What reconstruction error do REAL rows have? (the null a patched row is judged by)
  2. Is a re-extracted draw as in-sample as the stored corpus draw? This decides
     tabdpt: its corpus embeddings came from an unseeded retrieval draw that cannot be
     reproduced (see tabdpt_seed_search.py), but if a seeded re-extraction reconstructs
     just as well, the two draws are equally valid samples and neither is privileged.

Also reports max cosine to any archetype direction -- scale-free, so unlike hull
distance it is meaningful: "is this a direction the dictionary knows about".

Usage:
    python -m scripts.rebuttal.sae_insample_null --device cuda
"""
import argparse
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import (
    load_dataset_context, load_norm_stats, load_sae, load_test_embeddings,
)

DEFAULT_PROBES = [
    ("tabpfn", "heloc"), ("tabicl", "Bioresponse"), ("tabicl_v2", "miami_housing"),
    ("mitra", "APSFailure"), ("tabdpt", "miami_housing"),
]
EXTRACT_SEED = 13


def recon_stats(sae, X: np.ndarray, C: np.ndarray | None, device: str) -> dict:
    """Relative reconstruction error and archetype-direction alignment per row."""
    with torch.no_grad():
        x = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
        xh = sae.decode(sae.encode(x))
        num = torch.linalg.norm(x - xh, dim=1)
        den = torch.linalg.norm(x, dim=1).clamp_min(1e-8)
        rel = (num / den).cpu().numpy()
        fvu = float(((x - xh) ** 2).sum() / ((x - x.mean(0, keepdim=True)) ** 2).sum())
        out = {"recon_rel_median": float(np.median(rel)),
               "recon_rel_p95": float(np.percentile(rel, 95)),
               "fvu": fvu, "n_rows": int(len(X))}
        if C is not None:
            xn = torch.nn.functional.normalize(x, dim=1)
            cn = torch.nn.functional.normalize(
                torch.tensor(np.asarray(C, dtype=np.float32), device=device), dim=1)
            best = (xn @ cn.T).max(dim=1).values.cpu().numpy()
            out["archetype_cos_median"] = float(np.median(best))
            out["archetype_cos_min"] = float(best.min())
        return out


def reextract(model, dataset, device, n_rows):
    from scripts.rebuttal.patch_baseline_gate import layer_embeddings
    X_train, y_train, X_query, _, _, task = load_dataset_context(
        model, dataset, query_source="holdout")
    q = X_query[:n_rows] if not hasattr(X_query, "iloc") else X_query.iloc[:n_rows]
    emb, _, _ = layer_embeddings(model, dataset, X_train, y_train, q, task)
    mean, std = load_norm_stats(model, dataset, device="cpu")
    return (emb - mean.cpu().numpy()) / std.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rows", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict to these donors; tabicl_v2 must run under tfm2")
    ap.add_argument("--skip-reextract", action="store_true")
    ap.add_argument("--out", default=str(
        PROJECT_ROOT / "output" / "rebuttal" / "sae_insample_null.json"))
    args = ap.parse_args()

    torch.use_deterministic_algorithms(True)
    probes = ([p for p in DEFAULT_PROBES if p[0] in set(args.models)]
              if args.models else DEFAULT_PROBES)
    results = []
    for model, dataset in probes:
        sae, _ = load_sae(model, device=args.device)
        C = getattr(sae, "reference_data", None)
        C = C.detach().cpu().numpy() if C is not None else None
        E = np.asarray(load_test_embeddings(model)[dataset])[:args.n_rows]

        entry = {"model": model, "dataset": dataset,
                 "corpus": recon_stats(sae, E, C, args.device)}
        print(f"\n{model} / {dataset}")
        c = entry["corpus"]
        print(f"    corpus       recon_rel median={c['recon_rel_median']:.4f} "
              f"p95={c['recon_rel_p95']:.4f}  fvu={c['fvu']:.4f}  "
              f"archetype_cos median={c.get('archetype_cos_median', float('nan')):.4f}")

        if not args.skip_reextract:
            try:
                R = reextract(model, dataset, args.device, args.n_rows)
                entry["reextracted"] = recon_stats(sae, R, C, args.device)
                r = entry["reextracted"]
                print(f"    re-extracted recon_rel median={r['recon_rel_median']:.4f} "
                      f"p95={r['recon_rel_p95']:.4f}  fvu={r['fvu']:.4f}  "
                      f"archetype_cos median={r.get('archetype_cos_median', float('nan')):.4f}")
                verdict = ("as in-sample as the corpus draw"
                           if r["recon_rel_median"] <= c["recon_rel_median"] * 1.25
                           else "WORSE than the corpus draw")
                print(f"    -> re-extraction is {verdict}")
            except Exception as exc:
                print(f"    re-extract ERROR {type(exc).__name__}: {exc}")
                entry["reextracted"] = {"error": f"{type(exc).__name__}: {exc}"}
        results.append(entry)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")
    print("recon_rel on real rows is the null: a patched embedding whose reconstruction")
    print("error leaves this range is outside the instrument's valid range, and any")
    print("suppression measured there is the search exploiting the objective.")


if __name__ == "__main__":
    main()
