#!/usr/bin/env python3
"""Does re-estimating the recipient drop from the bystanders' shifts recover the ceiling?

The claim under test (2026-08-14 review): the deviation between the OBSERVED recipient
movement and the PLANNED one (suppression_frac x c's LOO interval) is the bystanders'
doing, to first order. If so, adding the bystanders' signed first-order contributions

    est_bystander = sum over j != c of (1 - r_j) x (p_loo_j - p_transfer)

(with r_j the sweep's measured final activation ratio for j) should bring the estimate
CLOSER to the observed movement than the planned term alone. If it does not, the
first-order model is missing the story (interactions, nonlinearity) and the
re-estimation design needs a rethink before it goes into the objective.

Two layers, one script:

  OFFLINE (no forwards): a necessary condition. |observed - planned| should be within
  the bystanders' unsigned spend (their magnitude bound, already recorded per row). A
  deviation exceeding the bound cannot be explained by bystanders at first order no
  matter the signs.

  SIGNED (recipient forwards): the decisive test. The sweep stores LOO effects
  UNSIGNED, so p_loo_j - p_transfer is recomputed here via one batched recipient
  forward per row, using build_recip's own machinery -- ratios come from the sweep
  output, so no donor forwards at all.

Cells needing the tfm2 env are skipped and counted when running under tfm (and vice
versa) -- same env rule as the sweep itself.

Usage (offline layer only, local):    python -m scripts.rebuttal.test_reestimated_drop --offline
Full run (one GPU worker):            python -m scripts.rebuttal.test_reestimated_drop
"""
import argparse
import glob
import json
import os
from collections import defaultdict

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

from scripts._project_root import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "output" / "rebuttal"
FLOOR = 0.01


def load_rows(pattern):
    """Chosen v19 patches with everything the test needs from the sweep itself."""
    out = []
    for p in sorted(glob.glob(str(OUT_DIR / pattern))):
        for c in json.load(open(p)):
            for ds in c.get("datasets") or []:
                for r in ds.get("rows") or []:
                    b = r.get("best")
                    iv = r.get("ablation_interval")
                    tw = (b or {}).get("toward_ablation")
                    if not b or iv is None or tw is None:
                        continue
                    if not (np.isfinite(iv) and np.isfinite(tw)):
                        continue
                    denom = float(np.copysign(max(abs(iv), FLOOR), iv if iv != 0 else 1.0))
                    out.append({
                        "donor": c["donor"], "feat": int(c["feat"]),
                        "recipient": ds["recipient"], "dataset": ds["dataset"],
                        "row": int(r["row"]),
                        "interval": float(iv),
                        "suppression_frac": float(b["suppression_frac"]),
                        "observed": float(tw) * denom,        # p_patched - p_transfer
                        "spend": sum(x.get("disturbed") or 0.0
                                     for x in (r.get("collateral") or [])
                                     if not x.get("inactive")),
                        "ratios": {int(k): float(v)
                                   for k, v in (r.get("accepted_ratios") or {}).items()},
                    })
    return out


def offline_report(rows):
    """Necessary condition: the deviation must be within the bystanders' unsigned bound."""
    dev = np.array([abs(r["observed"] - r["suppression_frac"] * r["interval"])
                    for r in rows])
    spend = np.array([r["spend"] for r in rows])
    within = dev <= spend + 1e-3          # 1e-3 headroom for path noise
    perfect = np.array([r["suppression_frac"] >= 0.99 for r in rows])
    print(f"OFFLINE necessary condition, {len(rows)} rows "
          f"(perfect suppression: {perfect.sum()})")
    print(f"  |observed - planned|: med {np.median(dev):.4f}  p90 {np.percentile(dev, 90):.4f}")
    print(f"  bystander spend:      med {np.median(spend):.4f}  p90 {np.percentile(spend, 90):.4f}")
    for name, m in (("all rows", np.ones(len(rows), bool)), ("perfect suppression", perfect)):
        print(f"  deviation within spend [{name}]: {within[m].mean():.1%} of {m.sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="skip the forwards")
    ap.add_argument("--n-cells", type=int, default=12)
    ap.add_argument("--n-rows", type=int, default=8, help="rows sampled per cell")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(OUT_DIR / "reestimated_drop_test.json"))
    args = ap.parse_args()

    rows = load_rows("patchv19clf_*.json")
    offline_report(rows)
    if args.offline:
        return

    import torch
    torch.use_deterministic_algorithms(True)
    from scripts.rebuttal.patch_search import (
        FWD, build_recip, build_recip_shared, current_env, required_env)

    # Sample cells largest-first for coverage, spread over recipients; env-mismatched
    # cells are counted, not silently dropped.
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["donor"], r["recipient"], r["dataset"])].append(r)
    ranked = sorted(by_cell.items(), key=lambda kv: -len(kv[1]))
    skipped_env = sum(1 for (d, rec, _), _ in ranked
                      if required_env(d, rec) != current_env())
    runnable = [(k, v) for k, v in ranked
                if required_env(k[0], k[1]) == current_env()]
    per_rec = defaultdict(int)
    cells = []
    for k, v in runnable:                       # round-robin-ish spread over recipients
        if per_rec[k[1]] >= max(1, args.n_cells // 4):
            continue
        cells.append((k, v)); per_rec[k[1]] += 1
        if len(cells) >= args.n_cells:
            break
    print(f"\nSIGNED test: {len(cells)} cells of {len(runnable)} runnable "
          f"({skipped_env} skipped for env), {args.n_rows} rows each")

    results = []
    for (donor, recipient, dataset), cell_rows in cells:
        npzs = [f for f in glob.glob(str(FWD / "*" / f"{dataset}.npz"))
                if str(np.load(f, allow_pickle=True)["strong_model"]) == donor
                and str(np.load(f, allow_pickle=True)["weak_model"]) == recipient]
        if not npzs:
            print(f"  {donor}->{recipient}/{dataset}: no npz, skipped"); continue
        try:
            shared = build_recip_shared(donor, recipient, dataset, args.device)
        except Exception as exc:
            print(f"  {donor}->{recipient}/{dataset}: shared FAILED {exc}"); continue
        if shared is None:
            continue
        for r in sorted(cell_rows, key=lambda x: -abs(x["interval"]))[:args.n_rows]:
            try:
                recip = build_recip(shared, donor, recipient, dataset, npzs[0],
                                    r["row"], defaultdict(float), r["feat"], args.device)
                if recip is None:
                    continue
                fids, B = recip["fids"], recip["B"]
                signs, a_corpus = recip["signs"], recip["a_corpus"]
                predict, loss = recip["predict"], recip["loss"]
                variants = []
                for i in range(len(fids)):
                    keep = np.ones(len(fids)); keep[i] = 0.0
                    variants.append((signs * a_corpus * keep) @ B)
                p_loo = [loss(p) for p in predict(np.asarray(variants))]
                est_by = sum(
                    (1.0 - r["ratios"].get(f, 1.0)) * (p_loo[i] - recip["p_transfer"])
                    for i, f in enumerate(fids) if f != r["feat"])
                planned = r["suppression_frac"] * r["interval"]
                results.append({**{k: r[k] for k in
                                   ("donor", "feat", "recipient", "dataset", "row",
                                    "interval", "suppression_frac", "observed", "spend")},
                                "planned": planned, "est_bystander": float(est_by),
                                "err_naive": abs(r["observed"] - planned),
                                "err_reest": abs(r["observed"] - planned - est_by)})
            except Exception as exc:
                print(f"  {donor}->{recipient}/{dataset} row {r['row']}: {type(exc).__name__}: {exc}")
        done = [x for x in results if x["dataset"] == dataset and x["recipient"] == recipient]
        print(f"  {donor}->{recipient}/{dataset}: {len(done)} rows", flush=True)
        json.dump(results, open(args.out, "w"), indent=2)

    if results:
        en = np.array([x["err_naive"] for x in results])
        er = np.array([x["err_reest"] for x in results])
        closer = er < en - 1e-9
        print(f"\n{len(results)} rows tested")
        print(f"  |observed - planned|            med {np.median(en):.4f}  p90 {np.percentile(en, 90):.4f}")
        print(f"  |observed - re-estimated|       med {np.median(er):.4f}  p90 {np.percentile(er, 90):.4f}")
        print(f"  re-estimate is CLOSER on {closer.mean():.1%} of rows "
              f"(ties/worse: {(~closer).sum()})")
        perf = np.array([x["suppression_frac"] >= 0.99 for x in results])
        if perf.any():
            print(f"  perfect-suppression rows ({perf.sum()}): closer on "
                  f"{closer[perf].mean():.1%}, med err {np.median(en[perf]):.4f} -> "
                  f"{np.median(er[perf]):.4f}")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
