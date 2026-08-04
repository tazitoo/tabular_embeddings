#!/usr/bin/env python3
"""App F.3 GATE: does the patch pipeline reproduce the SAE corpus's activations?

Input-level suppression patching recomputes donor SAE activations by re-running the
embedding extractor. Every downstream number (activation drop, and the delta rebuilt
from a patched activation) is measured against the transfer's cached activations, which
came from `sae.encode(load_test_embeddings(donor)[dataset])` -- i.e. the SAE corpus
built by scripts/sae_corpus/04_extract_all_layers.py.

So the recomputed activation for an UNPATCHED row must equal the cached one. If it does
not, the patch pipeline is measuring against a different regime and every result is void.

There is concrete reason to expect a mismatch. 04_extract_all_layers.py routes every
model except {tabula8b, carte} through the preprocessing cache with stratified context
(dataframe_style=False, :326). scripts/concepts/patch_activation_probe.py instead loads
RAW frames via load_tabarena_dataset and hardcodes dataframe_style=True (:153), so for
mitra/tabdpt/tabicl/tabicl_v2/tabpfn it differs on BOTH preprocessing and context rows.
For in-context learners the context set is a first-order determinant of the embedding.

This script compares three routes on unpatched rows:
  cached      : sae.encode(load_test_embeddings(model)[dataset])          <- ground truth
  corpus_route: extractor via intervene_lib.load_dataset_context(...)     <- proposed fix
  probe_route : extractor via patch_activation_probe._load_raw_context_query(...)

Usage:
    python -m scripts.rebuttal.patch_baseline_gate
    python -m scripts.rebuttal.patch_baseline_gate --model tabpfn --datasets heloc
"""
import argparse
import json

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import (
    load_dataset_context,
    load_norm_stats,
    load_sae,
    load_test_embeddings,
)

# one classification + one regression dataset per donor: the stratified-vs-simple
# context divergence should bite on classification and may vanish on regression.
DEFAULT_PROBES = [
    ("tabpfn", "heloc"),
    ("tabdpt", "miami_housing"),
    ("tabicl", "Bioresponse"),
]


def _extractor(model: str, device: str):
    """Extractor dispatch.

    The COMMITTED patch_activation_probe is mitra-only (it raises NotImplementedError
    for every other model); the multi-model dispatch exists only in uncommitted WIP.
    Prefer that dispatch when present, but fall back to a local one so this gate --
    the check that decides whether any patch result is trustworthy -- does not depend
    on whether that WIP has been committed.
    """
    try:
        from scripts.concepts.patch_activation_probe import _embedding_extractor_for_model
        return _embedding_extractor_for_model(model, device)
    except ImportError:
        pass
    if model == "mitra":
        from models.mitra_embeddings import MitraEmbeddingExtractor
        return MitraEmbeddingExtractor(device=device, n_estimators=1, fine_tune=False, seed=13)
    if model == "tabpfn":
        from models.tabpfn_embeddings import TabPFNEmbeddingExtractor
        return TabPFNEmbeddingExtractor(device=device)
    if model == "tabicl":
        from models.tabicl_embeddings import TabICLEmbeddingExtractor
        return TabICLEmbeddingExtractor(device=device)
    if model == "tabicl_v2":
        from models.tabicl_v2_embeddings import TabICLV2EmbeddingExtractor
        return TabICLV2EmbeddingExtractor(device=device)
    if model == "tabdpt":
        from models.tabdpt_embeddings import TabDPTEmbeddingExtractor
        return TabDPTEmbeddingExtractor(device=device)
    if model == "carte":
        from models.carte_embeddings import CARTEEmbeddingExtractor
        return CARTEEmbeddingExtractor(device=device)
    raise NotImplementedError(f"Unsupported model: {model}")


def _sae_acts(model: str, emb: np.ndarray, dataset: str, device: str) -> np.ndarray:
    mean, std = load_norm_stats(model, dataset, device=device)
    sae, _ = load_sae(model, device=device)
    with torch.no_grad():
        x = torch.tensor(np.asarray(emb, dtype=np.float32), device=device)
        return sae.encode((x - mean) / std).cpu().numpy()


def cached_acts(model: str, dataset: str, device: str) -> np.ndarray:
    """Ground truth: exactly what transfer_sweep_symmetric.py:312 computed."""
    sae, _ = load_sae(model, device=device)
    Xn = np.asarray(load_test_embeddings(model)[dataset], dtype=np.float32)
    with torch.no_grad():
        return sae.encode(torch.tensor(Xn, device=device)).cpu().numpy()


def corpus_route_acts(model: str, dataset: str, device: str, n_rows: int) -> np.ndarray:
    X_train, y_train, X_query, _, _, task = load_dataset_context(
        model, dataset, query_source="holdout")
    ex = _extractor(model, device)
    q = X_query[:n_rows] if not hasattr(X_query, "iloc") else X_query.iloc[:n_rows]
    res = ex.extract_embeddings(X_train, y_train, q, task=task)
    return _sae_acts(model, res.embeddings, dataset, device)


def probe_route_acts(model: str, dataset: str, device: str, n_rows: int) -> np.ndarray:
    """The existing patch pipeline's route. Only exists in the multi-model WIP:
    the committed loader is _load_raw_mitra_context_query and the committed
    _extract_feature_activations refuses any model except mitra."""
    try:
        from scripts.concepts.patch_activation_probe import _load_raw_context_query
    except ImportError:
        raise RuntimeError(
            "multi-model patch_activation_probe not committed -- probe_route unavailable "
            "(committed version is mitra-only)")
    X_context, y_context, X_query_raw, _, _, task = _load_raw_context_query(model, dataset)
    ex = _extractor(model, device)
    res = ex.extract_embeddings(
        X_context, y_context, X_query_raw.iloc[:n_rows].reset_index(drop=True), task=task)
    return _sae_acts(model, res.embeddings, dataset, device)


def _report(name: str, a: np.ndarray, ref: np.ndarray) -> dict:
    n = min(len(a), len(ref))
    a, ref = a[:n], ref[:n]
    d = np.abs(a - ref)
    denom = np.maximum(np.abs(ref), 1e-8)
    fired_ref, fired_a = ref > 0, a > 0
    agree = float((fired_ref == fired_a).mean())
    cos = float((a.ravel() @ ref.ravel()) /
                (np.linalg.norm(a.ravel()) * np.linalg.norm(ref.ravel()) + 1e-12))
    out = {"route": name, "n_rows": int(n), "max_abs_diff": float(d.max()),
           "mean_abs_diff": float(d.mean()),
           "median_rel_diff_on_fired": float(np.median((d / denom)[fired_ref]))
           if fired_ref.any() else float("nan"),
           "firing_agreement": agree, "cosine": cos,
           "reproduces": bool(d.max() < 1e-3)}
    print(f"    {name:14s} max|d|={out['max_abs_diff']:9.4f}  mean|d|={out['mean_abs_diff']:8.4f}  "
          f"fire-agree={agree:6.1%}  cos={cos:7.4f}  -> "
          f"{'REPRODUCES' if out['reproduces'] else 'MISMATCH'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--n-rows", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(
        PROJECT_ROOT / "output" / "rebuttal" / "patch_baseline_gate.json"))
    args = ap.parse_args()

    probes = ([(args.model, d) for d in args.datasets] if args.model and args.datasets
              else [(args.model, d) for m, d in DEFAULT_PROBES if m == args.model]
              if args.model else DEFAULT_PROBES)

    results = []
    for model, dataset in probes:
        print(f"\n{model} / {dataset}  (first {args.n_rows} sae_test rows)")
        ref = cached_acts(model, dataset, args.device)[:args.n_rows]
        entry = {"model": model, "dataset": dataset, "routes": []}
        for name, fn in (("corpus_route", corpus_route_acts), ("probe_route", probe_route_acts)):
            try:
                entry["routes"].append(_report(name, fn(model, dataset, args.device, args.n_rows), ref))
            except Exception as exc:
                print(f"    {name:14s} ERROR {type(exc).__name__}: {exc}")
                entry["routes"].append({"route": name, "error": f"{type(exc).__name__}: {exc}"})
        results.append(entry)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")
    ok = [r for e in results for r in e["routes"]
          if r.get("route") == "corpus_route" and r.get("reproduces")]
    print(f"corpus_route reproduces cached activations on {len(ok)}/{len(results)} probes")
    print("A patch pipeline is only trustworthy on probes where corpus_route REPRODUCES.")


if __name__ == "__main__":
    main()
