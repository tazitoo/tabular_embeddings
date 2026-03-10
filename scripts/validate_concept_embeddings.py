#!/usr/bin/env python3
"""Validate concept embedding quality.

Step 1 (always): Nomic self-checks — Matryoshka dimension consistency,
bootstrap stability.
Step 2 (optional): API cross-reference if self-checks fail or --api flag.

Usage:
    python scripts/validate_concept_embeddings.py
    python scripts/validate_concept_embeddings.py --api voyage
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.embed_concept_descriptions import embed_descriptions

INPUT_DIR = PROJECT_ROOT / "output" / "concept_descriptions"


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


def build_validation_report(
    nomic_checks: Dict,
    api_result: Optional[Dict] = None,
) -> Dict:
    """Build final validation report.

    Args:
        nomic_checks: Results of Matryoshka and bootstrap checks.
        api_result: Optional API cross-reference results.

    Returns:
        Complete validation report dict.
    """
    report = {
        "nomic_self_checks": nomic_checks,
        "api_validation": api_result or {"ran": False, "reason": "nomic self-checks passed"},
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate concept embeddings")
    parser.add_argument("--api", type=str, choices=["voyage", "openai"],
                        help="Force API validation with specified provider")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of descriptions to sample for API check")
    args = parser.parse_args()

    desc_path = INPUT_DIR / "concept_descriptions.json"
    if not desc_path.exists():
        print(f"Error: {desc_path} not found. Run generate_concept_descriptions.py first.")
        sys.exit(1)

    with open(desc_path) as f:
        data = json.load(f)

    # Collect all descriptions
    descriptions = []
    for group in data.get("groups", {}).values():
        features = group.get("features", {})
        if features:
            for feat in features.values():
                desc = feat.get("description", feat.get("brief_label", ""))
                if desc:
                    descriptions.append(desc)
        else:
            # Fallback: use group-level summary/brief_label
            desc = group.get("summary", group.get("brief_label", ""))
            if desc:
                descriptions.append(desc)
    for feat in data.get("unmatched", {}).values():
        desc = feat.get("description", feat.get("brief_label", ""))
        if desc:
            descriptions.append(desc)

    if not descriptions:
        print("No descriptions found in input file.")
        sys.exit(1)

    print(f"Validating {len(descriptions)} descriptions...")

    # Step 1: Nomic self-checks
    print("\nMatryoshka consistency...")
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

    # Step 2: API validation (if requested or self-checks failed)
    api_result = None
    if args.api or not passed:
        reason = f"--api {args.api}" if args.api else "self-checks failed"
        print(f"\nRunning API validation ({reason})...")
        # TODO: Implement API embedding call for voyage/openai
        api_result = {"ran": True, "reason": reason, "provider": args.api or "auto"}
        print("  API validation not yet implemented — placeholder only.")

    report = build_validation_report(nomic_checks, api_result)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = INPUT_DIR / "concept_embedding_validation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()
