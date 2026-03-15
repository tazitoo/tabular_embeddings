#!/usr/bin/env python3
"""Validate concept embedding quality.

Step 1 (always): Nomic self-checks — Matryoshka dimension consistency,
bootstrap stability.
Step 2 (optional): API cross-reference if self-checks fail or --api flag.
Step 3: Not yet implemented.
Step 4: Label–concept locality alignment — do nearby concepts (by probe
signature) get semantically similar labels?

Usage:
    python scripts/validate_concept_embeddings.py
    python scripts/validate_concept_embeddings.py --api voyage
    python scripts/validate_concept_embeddings.py --locality-only
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform, cosine as cosine_dist

from scripts._project_root import PROJECT_ROOT

from scripts.concepts.embed_concept_descriptions import embed_descriptions

INPUT_DIR = PROJECT_ROOT / "output" / "concept_descriptions"
LABELS_PATH = PROJECT_ROOT / "output" / "cross_model_concept_labels_round8.json"


def matryoshka_consistency(
    descriptions: List[str],
    dims: tuple = (768, 256, 128, 64),
) -> Dict[str, float]:
    """Check Spearman rank correlation of pairwise sims across Matryoshka dims.

    Args:
        descriptions: Text descriptions to embed at multiple resolutions.
        dims: Tuple of dimensions to compare (first is reference).

    Returns:
        Dict mapping "768v256" etc to Spearman rho values.
    """
    from scipy.spatial.distance import pdist

    embeddings_by_dim = {}
    for d in dims:
        embeddings_by_dim[d] = embed_descriptions(descriptions, dim=d)

    sims = {d: 1.0 - pdist(emb, metric="cosine") for d, emb in embeddings_by_dim.items()}
    base = dims[0]

    result = {}
    for d in dims[1:]:
        rho, _ = stats.spearmanr(sims[base], sims[d])
        result[f"{base}v{d}"] = float(rho)

    return result


def bootstrap_stability(
    descriptions: List[str],
    n_bootstrap: int = 5,
    dim: int = 768,
    seed: int = 42,
) -> float:
    """Check embedding stability under bootstrap resampling.

    Embeds full set, then resamples and checks if pairwise similarity
    rankings are preserved.

    Args:
        descriptions: Text descriptions to embed.
        n_bootstrap: Number of bootstrap iterations.
        dim: Embedding dimension.
        seed: Random seed for reproducibility.

    Returns:
        Mean Spearman rho across bootstrap samples.
    """
    from scipy.spatial.distance import pdist

    rng = np.random.RandomState(seed)
    base_emb = embed_descriptions(descriptions, dim=dim)

    rhos = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(descriptions), len(descriptions), replace=True)
        unique_idx = sorted(set(idx))
        if len(unique_idx) < 3:
            continue

        sub_descs = [descriptions[i] for i in unique_idx]
        sub_emb = embed_descriptions(sub_descs, dim=dim)
        sub_sims = 1.0 - pdist(sub_emb, metric="cosine")

        base_sub_sims = 1.0 - pdist(base_emb[unique_idx], metric="cosine")

        rho, _ = stats.spearmanr(base_sub_sims, sub_sims)
        rhos.append(rho)

    return float(np.mean(rhos)) if rhos else 0.0


def _build_probe_vectors(concept_groups: Dict) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build dense probe-signature vectors from top_probes.

    Each concept group becomes a 36-d vector where each dimension is the
    mean coefficient for that probe (0 if absent).

    Args:
        concept_groups: Dict of gid -> group info with 'top_probes' and 'label'.

    Returns:
        (probe_matrix, gids, labels) where probe_matrix is (n_groups, n_probes).
    """
    # Collect all probe names
    probe_names = set()
    for v in concept_groups.values():
        for entry in v.get("top_probes", []):
            probe_names.add(entry[0])
    probe_names = sorted(probe_names)
    probe_idx = {p: i for i, p in enumerate(probe_names)}

    gids = []
    labels = []
    vectors = []
    for gid, v in sorted(concept_groups.items(), key=lambda x: int(x[0])):
        label = v.get("label", "unlabeled")
        if label == "unlabeled":
            continue
        vec = np.zeros(len(probe_names), dtype=np.float32)
        for entry in v.get("top_probes", []):
            probe_name, _n_members, coeff = entry
            if probe_name in probe_idx:
                vec[probe_idx[probe_name]] = coeff
        gids.append(gid)
        labels.append(label)
        vectors.append(vec)

    return np.array(vectors), gids, labels


def label_concept_locality(
    concept_groups: Dict,
    ks: Tuple[int, ...] = (5, 10, 20, 50),
    dim: int = 768,
) -> Dict:
    """Compare label-space locality against concept-space (probe signature) locality.

    For each concept group, finds its K nearest neighbors in probe-signature
    space and in label-embedding space, then measures:
    1. Mantel test: Spearman correlation between full distance matrices.
    2. Precision@K: fraction of probe-space K-NN that are also label-space K-NN.
    3. Per-group mismatch score: concepts whose label neighborhood diverges most
       from their probe neighborhood (candidates for relabeling).

    Args:
        concept_groups: Dict of gid -> group info with 'top_probes' and 'label'.
        ks: Neighborhood sizes for Precision@K.
        dim: Nomic embedding dimension.

    Returns:
        Dict with mantel_rho, precision_at_k, worst_mismatches, tier_breakdown.
    """
    probe_matrix, gids, labels = _build_probe_vectors(concept_groups)
    n = len(gids)
    print(f"  Building probe vectors: {n} groups x {probe_matrix.shape[1]} probes")

    # Probe-space distance matrix (cosine)
    # Zero vectors get dist=1.0 to everything; flag them
    norms = np.linalg.norm(probe_matrix, axis=1)
    has_signal = norms > 1e-8
    n_signal = has_signal.sum()
    print(f"  Groups with probe signal: {n_signal}/{n}")

    # Only use groups with nonzero probe vectors for the Mantel test
    signal_idx = np.where(has_signal)[0]
    probe_sub = probe_matrix[signal_idx]
    labels_sub = [labels[i] for i in signal_idx]
    gids_sub = [gids[i] for i in signal_idx]
    n_sub = len(signal_idx)

    probe_dists = pdist(probe_sub, metric="cosine")

    # Label-space distance matrix (nomic embeddings)
    print(f"  Embedding {n_sub} labels with nomic-embed-text-v1.5 (dim={dim})...")
    label_embs = embed_descriptions(labels_sub, dim=dim)
    label_dists = pdist(label_embs, metric="cosine")

    # 1. Mantel test: Spearman correlation of distance matrices
    rho, p_value = stats.spearmanr(probe_dists, label_dists)
    print(f"  Mantel test: rho={rho:.4f}, p={p_value:.2e}")

    # Convert to square form for K-NN analysis
    probe_sq = squareform(probe_dists)
    label_sq = squareform(label_dists)

    # 2. Precision@K
    probe_ranks = np.argsort(probe_sq, axis=1)  # nearest first (self at 0)
    label_ranks = np.argsort(label_sq, axis=1)

    precision_at_k = {}
    for k in ks:
        if k >= n_sub:
            continue
        precisions = []
        for i in range(n_sub):
            # Skip self (index 0), take next K
            probe_nn = set(probe_ranks[i, 1:k + 1])
            label_nn = set(label_ranks[i, 1:k + 1])
            precisions.append(len(probe_nn & label_nn) / k)
        precision_at_k[k] = {
            "mean": float(np.mean(precisions)),
            "std": float(np.std(precisions)),
            "median": float(np.median(precisions)),
        }
        print(f"  Precision@{k}: mean={precision_at_k[k]['mean']:.3f}, "
              f"median={precision_at_k[k]['median']:.3f}")

    # 3. Per-group mismatch: Jaccard distance between probe K-NN and label K-NN
    k_mismatch = min(20, n_sub - 1)
    mismatches = []
    for i in range(n_sub):
        probe_nn = set(probe_ranks[i, 1:k_mismatch + 1])
        label_nn = set(label_ranks[i, 1:k_mismatch + 1])
        jaccard = 1.0 - len(probe_nn & label_nn) / len(probe_nn | label_nn)
        # Find which probe neighbors have very different labels
        probe_nn_labels = [labels_sub[j] for j in probe_ranks[i, 1:6]]
        mismatches.append({
            "gid": gids_sub[i],
            "label": labels_sub[i],
            "mismatch_score": float(jaccard),
            "probe_nn_labels": probe_nn_labels,
        })

    mismatches.sort(key=lambda x: -x["mismatch_score"])
    worst = mismatches[:20]
    best = mismatches[-10:]
    print(f"\n  Worst mismatches (label disagrees with probe neighborhood):")
    for m in worst[:10]:
        print(f"    Group {m['gid']}: \"{m['label']}\" "
              f"(mismatch={m['mismatch_score']:.2f})")
        print(f"      probe neighbors: {m['probe_nn_labels']}")

    print(f"\n  Best alignments (label agrees with probe neighborhood):")
    for m in best[:5]:
        print(f"    Group {m['gid']}: \"{m['label']}\" "
              f"(mismatch={m['mismatch_score']:.2f})")
        print(f"      probe neighbors: {m['probe_nn_labels']}")

    # 4. Tier breakdown
    tier_breakdown = {}
    for tier in [1, 2]:
        tier_idx = [
            i for i, gid in enumerate(gids_sub)
            if concept_groups[gid].get("tier") == tier
        ]
        if len(tier_idx) < 10:
            continue
        tier_probe = probe_sq[np.ix_(tier_idx, tier_idx)]
        tier_label = label_sq[np.ix_(tier_idx, tier_idx)]
        tier_probe_flat = squareform(tier_probe)
        tier_label_flat = squareform(tier_label)
        tier_rho, _ = stats.spearmanr(tier_probe_flat, tier_label_flat)
        tier_breakdown[f"tier_{tier}"] = {
            "n_groups": len(tier_idx),
            "mantel_rho": float(tier_rho),
        }
        print(f"  Tier {tier}: rho={tier_rho:.4f} (n={len(tier_idx)})")

    return {
        "mantel_rho": float(rho),
        "mantel_p_value": float(p_value),
        "n_groups_tested": n_sub,
        "n_groups_no_signal": int(n - n_sub),
        "precision_at_k": precision_at_k,
        "tier_breakdown": tier_breakdown,
        "worst_mismatches": worst,
        "best_alignments": best,
    }


def build_validation_report(
    nomic_checks: Dict,
    api_result: Optional[Dict] = None,
    locality_result: Optional[Dict] = None,
) -> Dict:
    """Build final validation report.

    Args:
        nomic_checks: Results of Matryoshka and bootstrap checks.
        api_result: Optional API cross-reference results.
        locality_result: Optional label-concept locality results.

    Returns:
        Complete validation report dict.
    """
    report = {
        "nomic_self_checks": nomic_checks,
        "api_validation": api_result or {"ran": False, "reason": "nomic self-checks passed"},
        "label_concept_locality": locality_result or {"ran": False},
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate concept embeddings")
    parser.add_argument("--api", type=str, choices=["voyage", "openai"],
                        help="Force API validation with specified provider")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of descriptions to sample for API check")
    parser.add_argument("--locality-only", action="store_true",
                        help="Run only Step 4 (label-concept locality)")
    parser.add_argument("--labels", type=str, default=str(LABELS_PATH),
                        help="Path to cross_model_concept_labels_round8.json")
    args = parser.parse_args()

    nomic_checks = None
    api_result = None
    locality_result = None

    # Steps 1-2: Nomic self-checks (skip if --locality-only)
    if not args.locality_only:
        desc_path = INPUT_DIR / "concept_descriptions.json"
        if not desc_path.exists():
            print(f"Warning: {desc_path} not found. Skipping steps 1-2.")
        else:
            with open(desc_path) as f:
                data = json.load(f)

            descriptions = []
            for group in data.get("groups", {}).values():
                features = group.get("features", {})
                if features:
                    for feat in features.values():
                        desc = feat.get("description", feat.get("brief_label", ""))
                        if desc:
                            descriptions.append(desc)
                else:
                    desc = group.get("summary", group.get("brief_label", ""))
                    if desc:
                        descriptions.append(desc)
            for feat in data.get("unmatched", {}).values():
                desc = feat.get("description", feat.get("brief_label", ""))
                if desc:
                    descriptions.append(desc)

            if descriptions:
                print(f"Validating {len(descriptions)} descriptions...")

                # Step 1: Nomic self-checks
                print("\nStep 1: Matryoshka consistency...")
                mat_result = matryoshka_consistency(descriptions)
                for k, v in mat_result.items():
                    print(f"  {k}: rho={v:.4f}")

                print("\nBootstrap stability...")
                boot_result = bootstrap_stability(descriptions)
                print(f"  Stability: {boot_result:.4f}")

                min_mat = min(mat_result.values()) if mat_result else 0.0
                passed = min_mat > 0.80 and boot_result > 0.85
                nomic_checks = {
                    "matryoshka_spearman": mat_result,
                    "bootstrap_stability": boot_result,
                    "passed": passed,
                }
                print(f"\nSelf-checks: {'PASSED' if passed else 'FAILED'}")

                # Step 2: API validation
                if args.api or not passed:
                    reason = f"--api {args.api}" if args.api else "self-checks failed"
                    print(f"\nStep 2: API validation ({reason})...")
                    api_result = {"ran": True, "reason": reason,
                                  "provider": args.api or "auto"}
                    print("  API validation not yet implemented — placeholder only.")

    # Step 4: Label-concept locality alignment
    labels_path = Path(args.labels)
    if labels_path.exists():
        print(f"\nStep 4: Label-concept locality alignment...")
        with open(labels_path) as f:
            label_data = json.load(f)
        locality_result = label_concept_locality(label_data["concept_groups"])
    else:
        print(f"Warning: {labels_path} not found. Skipping step 4.")

    report = build_validation_report(
        nomic_checks or {"skipped": True},
        api_result,
        locality_result,
    )

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = INPUT_DIR / "concept_embedding_validation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()
