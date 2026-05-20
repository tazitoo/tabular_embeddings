"""Span-restricted concept transfer — virtual-atom cache builder.

Premise (see plan / memory `transfer_concept_map_near_null`): the deployed
global atom→atom ridge map has near-null CV R² because it fits an
unidentifiable per-atom geometric relation on behaviorally-matched pairs and
silently injects the model-private (out-of-shared-span) part of each unmatched
direction. CKA guarantees the *shared subspace* (decoder column span) aligns,
not individual atoms.

This builder produces an alternative virtual-atom set:

  1. Per model, build the top-k principal **span** of its round-10 decoder
     atoms (k = effective rank at 95% squared-singular energy).
  2. Fit a low-dim k_t×k_s alignment M_span on the *matched* landmark atoms
     projected into those spans (matched pairs only pin a rotation).
  3. For each unmatched strong concept, transfer only its **in-span
     component** through M_span, lift into the weak span, and apply the
     *identical* magnitude recipe the deployed builder uses — so the ONLY
     thing that differs from production is the injected direction.

Orthogonality contract (hard): this file edits no existing pipeline code. It
reuses `build_global_ridge_virtual_atoms` (the deployed parity anchor) and the
landmark/atom helpers by import only, and writes the canonical npz cache schema
(mirrored from `build_transfer_caches.save_cache`) under a dedicated output
root, refusing to overwrite. The cache is consumed by the unmodified
`transfer_sweep_v2 --virtual-atoms-cache-dir` path, so downstream injection /
greedy search / gap-closed are byte-identical to the deployed pipeline.

Run (builder only; eval is a separate step):
    python -m scripts.intervention.transfer_span_restricted
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import permutations
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import load_sae
from scripts.intervention.transfer_sweep_v2 import (
    DEFAULT_MATCHING_FILE,
    get_matched_pairs,
    get_unmatched_features,
)
from scripts.intervention.transfer_virtual_nodes import (
    extract_decoder_atoms,
    filter_landmarks,
    fit_concept_map,
)
from scripts.analysis.build_transfer_caches import build_global_ridge_virtual_atoms
from scripts.analysis.compare_transfer_maps import sha256_file


# Inlined from the local-only diagnostic scripts.analysis.transfer_subspace_r2
# to keep this file self-contained on the orthogonal branch (the diagnostic is
# not committed; this file is the only one on the span-restricted branch).
def _ortho_basis(mat, k):
    """Top-k right singular (orthonormal, target-space) basis of ``mat``,
    columns ordered by singular value. ``mat`` is (n_dirs, d_target)."""
    _, s, vt = np.linalg.svd(mat, full_matrices=False)
    k = min(k, vt.shape[0])
    return vt[:k].T, s


def _effective_rank(s, frac):
    """Smallest #components capturing ``frac`` of squared-singular energy."""
    energy = np.cumsum(s ** 2)
    if energy[-1] <= 0:
        return 1
    return int(np.searchsorted(energy / energy[-1], frac) + 1)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Dedicated, orthogonal output root — never an existing results dir.
OUT_ROOT = PROJECT_ROOT / "output" / "span_restricted_transfer"
EXISTING_RESULTS_DIRS = [
    PROJECT_ROOT / "output" / "transfer_sweep_v2",
    PROJECT_ROOT / "output" / "transfer_caches",
    PROJECT_ROOT / "output" / "transfer_subspace_r2",
]

# `random` is the project's canonical untrained-SAE control (sae_random_baseline);
# it is the established baseline used by build_transfer_caches — not a new arm.
SAE_DIRS = {
    "trained": PROJECT_ROOT / "output" / "sae_tabarena_sweep_round10",
    "random": PROJECT_ROOT / "output" / "sae_random_baseline",
}

ENERGY_FRAC = 0.95   # principal-span effective-rank threshold (no k sweep)
ALPHA = 1.0          # same ridge alpha as the deployed concept map
MIN_COSINE = 0.0     # same landmark filter as the deployed pipeline


def _assert_orthogonal(path: Path) -> None:
    """Refuse to write outside OUT_ROOT, into an existing results dir, or over
    an existing file (orthogonality contract)."""
    rp = path.resolve()
    if OUT_ROOT.resolve() not in rp.parents:
        raise RuntimeError(f"Refusing to write outside {OUT_ROOT}: {rp}")
    for d in EXISTING_RESULTS_DIRS:
        if d.resolve() == rp or d.resolve() in rp.parents:
            raise RuntimeError(f"Refusing to write into existing results dir: {rp}")
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing cache: {rp}")


def _save_cache(variant, condition, source, target, vatoms, computed,
                unmatched_ids, n_landmarks, d_target, matching_file,
                map_params, concept_map_r2):
    """Write the canonical virtual-atoms npz. Schema mirrors
    `build_transfer_caches.save_cache` exactly (kept in sync by the parity
    assertion in the runner); duplicated here only so the output root can be
    the orthogonal dir without editing the pipeline writer."""
    out_dir = OUT_ROOT / "virtual_atoms" / f"{variant}_{condition}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{source}_to_{target}.npz"
    _assert_orthogonal(path)
    np.savez_compressed(
        path,
        virtual_atoms=vatoms,
        feature_ids=np.array(unmatched_ids, dtype=np.int64),
        computed_mask=computed,
        variant=np.array(variant),
        sae_condition=np.array(condition),
        source_model=np.array(source),
        target_model=np.array(target),
        n_landmarks=np.int64(n_landmarks),
        n_unmatched=np.int64(len(unmatched_ids)),
        n_computed=np.int64(int(computed.sum())),
        d_target=np.int64(d_target),
        ridge_alpha=np.float32(ALPHA),
        min_cosine=np.float32(MIN_COSINE),
        matching_file=np.array(str(matching_file)),
        matching_file_sha256=np.array(sha256_file(matching_file)),
        map_params_json=np.array(json.dumps(map_params)),
        concept_map_r2=np.float64(concept_map_r2),
    )
    return path


def build_span_restricted_virtual_atoms(
    atoms_src_full: np.ndarray,
    atoms_tgt_full: np.ndarray,
    filt_src: np.ndarray,
    filt_tgt: np.ndarray,
    unmatched_src_ids: list,
):
    """In-span-only transfer. Span = top-k principal subspace (95% energy) of
    each model's decoder atoms. M_span fit on matched atoms projected into the
    spans. Magnitude recipe is identical to the deployed builder
    (`build_global_ridge_virtual_atoms`) — only the direction differs.

    Returns (vatoms, computed, r2_span, k_src, k_tgt).
    """
    # Principal spans (right singular vectors of the atom clouds).
    U_src, s_src = _ortho_basis(atoms_src_full, atoms_src_full.shape[1])
    U_tgt, s_tgt = _ortho_basis(atoms_tgt_full, atoms_tgt_full.shape[1])
    k_src = _effective_rank(s_src, ENERGY_FRAC)
    k_tgt = _effective_rank(s_tgt, ENERGY_FRAC)
    U_src, U_tgt = U_src[:, :k_src], U_tgt[:, :k_tgt]  # (d, k)

    # Fit the low-dim alignment on matched landmarks projected into the spans.
    M_span, r2_span = fit_concept_map(filt_src @ U_src, filt_tgt @ U_tgt,
                                      alpha=ALPHA)  # (k_tgt, k_src)

    # Magnitude recipe: identical to build_global_ridge_virtual_atoms
    # (transfer_sweep_v2 production semantics) — median target/source landmark
    # norm ratio, applied to a unit direction scaled by the full atom norm.
    sn = np.linalg.norm(filt_src, axis=1)
    tn = np.linalg.norm(filt_tgt, axis=1)
    valid = (sn > 1e-8) & (tn > 1e-8)
    median_norm_ratio = float(np.median(tn[valid] / sn[valid])) if valid.sum() else 1.0

    n_unm = len(unmatched_src_ids)
    d_tgt = atoms_tgt_full.shape[1]
    vatoms = np.zeros((n_unm, d_tgt), dtype=np.float32)
    computed = np.zeros(n_unm, dtype=bool)
    for out_idx, fi in enumerate(unmatched_src_ids):
        atom_s = atoms_src_full[fi]
        atom_norm = float(np.linalg.norm(atom_s))
        if atom_norm < 1e-8:
            continue
        in_span = atom_s @ U_src              # (k_src,) drop out-of-span part
        mapped = in_span @ M_span.T           # (k_tgt,)
        virtual_dir = mapped @ U_tgt.T        # (d_tgt,) lift into weak space
        vdir_norm = float(np.linalg.norm(virtual_dir))
        if vdir_norm < 1e-8:
            continue
        vatoms[out_idx] = ((virtual_dir / vdir_norm)
                           * atom_norm * median_norm_ratio).astype(np.float32)
        computed[out_idx] = True
    return vatoms, computed, float(r2_span), k_src, k_tgt


def build_pair(source: str, target: str, condition: str) -> dict:
    """Build span + global(parity) caches for one directed pair under one SAE
    condition. Both written so the runner can compare like-for-like."""
    sae_dir = SAE_DIRS[condition]
    sae_s, _ = load_sae(source, sae_dir=sae_dir, device="cpu")  # weights only
    sae_t, _ = load_sae(target, sae_dir=sae_dir, device="cpu")
    atoms_s = extract_decoder_atoms(sae_s).numpy()
    atoms_t = extract_decoder_atoms(sae_t).numpy()

    m_pairs = get_matched_pairs(source, target)  # default = round-10 p90 file
    if len(m_pairs) < 10:
        logger.info(f"  SKIP {source}->{target}/{condition}: "
                    f"{len(m_pairs)} matched pairs")
        return {"pair": f"{source}->{target}", "condition": condition,
                "status": f"too few matched ({len(m_pairs)})"}
    src_idx = [p[0] for p in m_pairs]
    tgt_idx = [p[1] for p in m_pairs]
    filt_src, filt_tgt, _, _ = filter_landmarks(
        atoms_s[src_idx], atoms_t[tgt_idx], m_pairs,
        min_cosine=MIN_COSINE, alpha=ALPHA)
    n = len(filt_src)
    if n < 12:
        logger.info(f"  SKIP {source}->{target}/{condition}: "
                    f"{n} landmarks post-filter")
        return {"pair": f"{source}->{target}", "condition": condition,
                "status": f"too few landmarks ({n})"}

    unmatched = get_unmatched_features(source, target)
    matching_file = Path(DEFAULT_MATCHING_FILE)
    d_tgt = atoms_t.shape[1]

    # Parity anchor: the *deployed* global-ridge builder, reused unchanged.
    # Tuple-slice so this works whether the deployed builder returns 2 values
    # (worker HEAD) or 3 with r2_global appended (local-only change). r2 for
    # cache metadata is computed independently via fit_concept_map below so
    # the script never depends on the optional third return.
    ret_g = build_global_ridge_virtual_atoms(
        atoms_s, filt_src, filt_tgt, unmatched)
    vg, cg = ret_g[0], ret_g[1]
    _, r2g = fit_concept_map(filt_src, filt_tgt, alpha=ALPHA)
    _save_cache("global", condition, source, target, vg, cg, unmatched, n,
                d_tgt, matching_file,
                {"method": "global_ridge_parity", "alpha": ALPHA}, r2g)

    # Span-restricted variant.
    vs, cs, r2s, k_s, k_t = build_span_restricted_virtual_atoms(
        atoms_s, atoms_t, filt_src, filt_tgt, unmatched)
    _save_cache("span", condition, source, target, vs, cs, unmatched, n,
                d_tgt, matching_file,
                {"method": "span_restricted", "alpha": ALPHA,
                 "energy_frac": ENERGY_FRAC, "k_src": k_s, "k_tgt": k_t}, r2s)

    # In-span delta norm must not exceed the full (global) delta norm.
    both = cg & cs
    norm_ok = bool(np.all(
        np.linalg.norm(vs[both], axis=1)
        <= np.linalg.norm(vg[both], axis=1) + 1e-4)) if both.any() else True

    logger.info(
        f"  {source}->{target}/{condition}: n_land={n} "
        f"k_src={k_s}/{atoms_s.shape[1]} k_tgt={k_t}/{d_tgt} "
        f"r2_span={r2s:+.3f} r2_global={r2g:+.3f} "
        f"computed span/global={int(cs.sum())}/{int(cg.sum())} "
        f"norm_ok={norm_ok}")
    return {"pair": f"{source}->{target}", "condition": condition,
            "status": "ok", "n_landmarks": n, "k_src": k_s, "k_tgt": k_t,
            "r2_span": r2s, "r2_global": r2g,
            "n_computed_span": int(cs.sum()),
            "n_computed_global": int(cg.sum()),
            "inspan_norm_le_global": norm_ok}


RUNS_ROOT = OUT_ROOT / "runs"
PUBLISHED_ROOT = PROJECT_ROOT / "output" / "transfer_sweep_v2"
PARITY_TOL = 1e-3


def _pair_dir_name(a: str, b: str) -> str:
    """transfer_sweep_v2 writes to {sorted(a,b) joined by _vs_}."""
    x, y = sorted([a, b])
    return f"{x}_vs_{y}"


def _collect_gc(run_dir: Path) -> dict:
    """{dataset: (mean_gap_closed, n_strong_wins)} for one run dir."""
    out = {}
    if not run_dir.is_dir():
        return out
    for npz in sorted(run_dir.glob("*.npz")):
        c = np.load(npz, allow_pickle=True)
        if "mean_gap_closed" not in c.files:
            continue
        out[npz.stem] = (float(c["mean_gap_closed"]),
                         int(c["n_strong_wins"]) if "n_strong_wins" in c.files else 0)
    return out


def _wmean(items):
    """n_strong_wins-weighted and simple mean of per-dataset gap closed."""
    if not items:
        return float("nan"), float("nan"), 0
    gcs = np.array([g for g, _ in items], float)
    w = np.array([max(n, 0) for _, n in items], float)
    simple = float(gcs.mean())
    weighted = float((gcs * w).sum() / w.sum()) if w.sum() > 0 else simple
    return weighted, simple, len(items)


def aggregate(models: list, conditions: list) -> dict:
    """Decision table + parity check from existing transfer_sweep_v2 runs.
    Reads only our runs/ namespace and the published results (read-only)."""
    pairs = sorted({_pair_dir_name(a, b)
                    for a, b in permutations(models, 2)})
    report = {"pairs": {}, "parity": {}, "decision": {}}
    for pair in pairs:
        row = {}
        for cond in conditions:
            for variant in ("global", "span"):
                rd = RUNS_ROOT / f"{variant}_{cond}" / pair
                w, s, n = _wmean(list(_collect_gc(rd).values()))
                row[f"{variant}_{cond}"] = {"gc_weighted": w,
                                            "gc_simple": s, "n_datasets": n}
        # Parity: our global_trained vs published deployed result.
        ours = _collect_gc(RUNS_ROOT / "global_trained" / pair)
        pub = _collect_gc(PUBLISHED_ROOT / pair)
        common = sorted(set(ours) & set(pub))
        diffs = [abs(ours[d][0] - pub[d][0]) for d in common]
        max_diff = max(diffs) if diffs else float("nan")
        report["parity"][pair] = {
            "n_common_datasets": len(common),
            "max_abs_gc_diff": max_diff,
            "pass": bool(diffs) and max_diff < PARITY_TOL,
        }
        # Verdict (explicit thresholds; evidence, not a definitive label).
        gt = row.get("global_trained", {}).get("gc_weighted", float("nan"))
        st = row.get("span_trained", {}).get("gc_weighted", float("nan"))
        sr = row.get("span_random", {}).get("gc_weighted", float("nan"))
        verdict = "inconclusive"
        if not (np.isnan(gt) or np.isnan(st)):
            if st >= 0.8 * gt and (np.isnan(sr) or st > sr + 0.10):
                verdict = "rescue (in-span suffices, beats random)"
            elif st < 0.5 * gt:
                verdict = "reveal (deployed gc leaned on out-of-span)"
            elif not np.isnan(sr) and abs(st - sr) <= 0.10:
                verdict = "artifact (span ~ random)"
        report["pairs"][pair] = row
        report["decision"][pair] = verdict
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aggregate", action="store_true",
                    help="Report decision table + parity from existing "
                         "transfer_sweep_v2 runs (no GPU; read-only).")
    # Default = primary pilot (TabPFN<->Mitra, both directions): highest CKA,
    # atom-map R²≈0, healthy landmarks — the unconfounded best-case.
    ap.add_argument("--models", nargs="+", default=["tabpfn", "mitra"])
    ap.add_argument("--conditions", nargs="+", default=["trained", "random"],
                    choices=["trained", "random"])
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.aggregate:
        rep = aggregate(args.models, args.conditions)
        for pair, row in rep["pairs"].items():
            par = rep["parity"][pair]
            logger.info(f"\n[{pair}]  parity: "
                        f"{'PASS' if par['pass'] else 'FAIL/NA'} "
                        f"(max Δgc={par['max_abs_gc_diff']:.4g}, "
                        f"n={par['n_common_datasets']})")
            for k, v in row.items():
                logger.info(f"  {k:16s} gc_w={v['gc_weighted']:+.3f} "
                            f"gc_s={v['gc_simple']:+.3f} n={v['n_datasets']}")
            logger.info(f"  -> verdict: {rep['decision'][pair]}")
        dp = OUT_ROOT / "decision.json"
        dp.write_text(json.dumps(rep, indent=2))
        logger.info(f"\nDecision report: {dp}")
        return

    summary = []
    for cond in args.conditions:
        for s, t in permutations(args.models, 2):
            summary.append(build_pair(s, t, cond))

    # build_summary.json is idempotent run metadata (not an experiment
    # artifact), so it may be refreshed across runs; the no-overwrite guard
    # protects only the virtual-atom caches.
    sp = OUT_ROOT / "build_summary.json"
    sp.write_text(json.dumps(summary, indent=2))
    logger.info(f"\nWrote caches under {OUT_ROOT/'virtual_atoms'}\nSummary: {sp}")


if __name__ == "__main__":
    main()
