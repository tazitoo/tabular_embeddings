#!/usr/bin/env python3
"""Generate a complete concept dictionary from SAE analysis."""

import json
import sys
from pathlib import Path

def main():
    analysis_path = Path("output/sae_full_analysis.json")

    with open(analysis_path) as f:
        data = json.load(f)

    features = data["features"]
    clusters = data.get("clustering", {}).get("clusters", {})
    feature_to_cluster = data.get("clustering", {}).get("feature_to_cluster", {})

    # Build concept dictionary
    concept_dict = []

    for fid, feat in features.items():
        effects = feat["effect_sizes"]
        sorted_effects = sorted(effects.items(), key=lambda x: -abs(x[1]))[:3]

        # Handle both string and int keys
        cluster_id = feature_to_cluster.get(str(fid), feature_to_cluster.get(fid, -1))

        # Generate compact pattern description
        interp_parts = []
        for name, d in sorted_effects:
            if abs(d) > 0.3:
                sign = "+" if d > 0 else "-"
                # Shorten common prefixes
                short_name = name.replace("numeric_", "n_").replace("categorical_", "cat_")
                short_name = short_name.replace("_zscore", "").replace("_rate", "")
                interp_parts.append(f"{sign}{short_name[:10]}")

        concept_dict.append({
            "id": int(fid),
            "cluster": cluster_id,
            "top_effect": sorted_effects[0][0] if sorted_effects else None,
            "top_d": round(sorted_effects[0][1], 2) if sorted_effects else 0,
            "pattern": " ".join(interp_parts) if interp_parts else "weak",
            "interpretation": feat.get("interpretation", ""),
        })

    # Sort by cluster, then by top effect magnitude
    concept_dict.sort(key=lambda x: (x["cluster"], -abs(x["top_d"])))

    print("=" * 80)
    print("COMPLETE TABPFN SAE CONCEPT DICTIONARY")
    print(f"497 alive features → 30 clusters")
    print("=" * 80)

    current_cluster = None
    for c in concept_dict:
        if c["cluster"] != current_cluster:
            current_cluster = c["cluster"]
            cluster_info = clusters.get(str(current_cluster), clusters.get(current_cluster, {}))
            n_feat = cluster_info.get("n_features", "?")
            interp = cluster_info.get("interpretation", "No interpretation")
            print(f"\n{'='*80}")
            print(f"CLUSTER {current_cluster}: {n_feat} features")
            print(f"  Theme: {interp[:75]}")
            print("-" * 80)

        # Print feature with its pattern
        print(f"  F{c['id']:4d} | {c['top_effect']:22s} | d={c['top_d']:+5.2f} | {c['pattern']}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("CLUSTER SIZE DISTRIBUTION")
    print("=" * 80)

    cluster_sizes = [(cid, info.get("n_features", 0))
                     for cid, info in clusters.items()]
    cluster_sizes.sort(key=lambda x: -x[1])

    for cid, size in cluster_sizes:
        bar = "█" * (size // 3)
        print(f"  Cluster {cid:2s}: {size:3d} {bar}")

    # Save as structured JSON
    output = {
        "n_features": len(concept_dict),
        "n_clusters": len(clusters),
        "features": {c["id"]: {
            "cluster": c["cluster"],
            "top_effect": c["top_effect"],
            "effect_size": c["top_d"],
            "pattern": c["pattern"],
        } for c in concept_dict},
        "cluster_themes": {cid: info.get("interpretation", "")
                          for cid, info in clusters.items()},
    }

    with open("output/concept_dictionary.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to output/concept_dictionary.json")


if __name__ == "__main__":
    main()
