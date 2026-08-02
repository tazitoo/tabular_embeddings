#!/usr/bin/env python3
"""Which installed packages are actually LOADED during tail-model compute?

Worker envs drift in ~65 transitive dependencies. Most are obviously irrelevant
(matplotlib, boto3), but "obviously" is not evidence. This builds the tail models
used by the intervention/decomposition path and records sys.modules before and
after, so a package's presence in the diff can be checked against whether it is
imported at all.

A package that is never imported cannot change numerics; one that IS imported
still might not (it may only be touched at config/registry time), so treat a hit
as "worth checking", not "guilty".

Usage:
    python -m scripts.rebuttal.compute_path_modules --dataset anneal \
        --models carte mitra tabdpt tabicl tabpfn --freeze /tmp/freeze.txt
"""
import argparse
import json
import re
import sys

import numpy as np
import torch

from scripts.intervention.intervene_lib import (
    SPLITS_PATH, get_extraction_layer_taskaware, build_tail, load_dataset_context,
)

# import name -> distribution name, where they differ
ALIAS = {
    "sklearn": "scikit-learn", "PIL": "pillow", "cv2": "opencv-python",
    "yaml": "pyyaml", "dateutil": "python-dateutil", "pkg_resources": "setuptools",
    "torch_geometric": "torch-geometric", "google": "protobuf",
    "attr": "attrs", "OpenSSL": "pyopenssl", "jwt": "pyjwt",
}


def top_level(mods):
    """Distribution-ish names for the top-level packages in a module set."""
    out = set()
    for m in mods:
        t = m.split(".")[0]
        if t.startswith("_") or t in sys.builtin_module_names:
            continue
        out.add(ALIAS.get(t, t).lower().replace("_", "-"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", nargs="+",
                    default=["carte", "mitra", "tabdpt", "tabicl", "tabpfn"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--freeze", help="pip freeze file; report only these packages")
    ap.add_argument("--out", help="write the loaded-package set as JSON")
    args = ap.parse_args()

    baseline = set(sys.modules)
    splits = json.loads(SPLITS_PATH.read_text())
    per_model = {}
    for model in args.models:
        before = set(sys.modules)
        Xtr, ytr, Xq, _, _, task = load_dataset_context(model, args.dataset, splits)
        if ytr.dtype == np.int32:
            ytr = ytr.astype(np.int64)
        layer = get_extraction_layer_taskaware(model, dataset=args.dataset)
        cat_indices = None
        if model in ("hyperfast", "tabpfn"):
            from data.preprocessing import load_preprocessed, CACHE_DIR
            cat_indices = load_preprocessed(model, args.dataset, CACHE_DIR).cat_indices or None
        torch.manual_seed(13); np.random.seed(13)
        build_tail(model, Xtr, ytr, Xq, layer, task, args.device,
                   cat_indices=cat_indices,
                   target_name=splits.get(args.dataset, {}).get("target", "target"))
        per_model[model] = top_level(set(sys.modules) - before)
        print(f"{model:<12} newly imported top-level packages: {len(per_model[model])}")

    loaded = top_level(set(sys.modules) - baseline) | top_level(baseline)
    print(f"\nTOTAL distinct top-level packages loaded: {len(loaded)}")

    if args.freeze:
        pkgs = set()
        for ln in open(args.freeze):
            m = re.match(r"^([A-Za-z0-9._-]+)\s*(?:==|@)", ln.strip())
            if m:
                pkgs.add(m.group(1).lower().replace("_", "-"))
        hit = sorted(pkgs & loaded)
        miss = sorted(pkgs - loaded)
        print(f"\n=== IMPORTED during compute ({len(hit)}/{len(pkgs)}) ===")
        print("   ", ", ".join(hit))
        print(f"\n=== NEVER IMPORTED ({len(miss)}) -- cannot affect numerics ===")
        print("   ", ", ".join(miss))

    if args.out:
        json.dump(sorted(loaded), open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
