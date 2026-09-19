#!/usr/bin/env bash
# Build the anonymized code release as a single-commit repository under $OUT.
#
# Exports a whitelist of tracked paths from the current HEAD (code, tests, env
# freezes, the reproducibility doc), scrubs identifying strings (user names,
# host names, absolute home paths, the source repository URL), replaces the
# project README with the release README, and commits everything once as
# "Anonymous Authors". Result outputs (output/) are NOT included; the release
# README says how they are provided. Run from the repository root:
#
#   bash scripts/release/build_anon_release.sh /path/to/export
#
# Then `git remote add origin <private repo>` in $OUT, push, and point
# Anonymous GitHub (anonymous.4open.science) at that repository.
set -euo pipefail

OUT="${1:?export directory}"
REPO=$(git rev-parse --show-toplevel)
cd "$REPO"

WHITELIST=(models data scripts tests envs config docs/reproducibility.md
           LICENSE pyproject.toml uv.lock .gitignore
           compare_embeddings.py cluster.py extract_embeddings.py cache_tabarena.py
           compute_cka_from_saved.py compare_sae_concepts.py)

rm -rf "$OUT"; mkdir -p "$OUT"
git archive HEAD "${WHITELIST[@]}" | tar -x -C "$OUT"
cp docs/RELEASE_README.md "$OUT/README.md"
rm -rf "$OUT/scripts/release"

# ---- scrub -------------------------------------------------------------------
# Order matters: longer host names before their prefixes. Host names become
# generic worker names; home paths become generic; the source repo URL goes.
scrub() {
  local pat="$1" rep="$2"
  # grep exits 1 when a pattern has no matches; that is not an error here
  (grep -rIl -- "$pat" "$OUT" 2>/dev/null || true) | while read -r f; do
    sed -i '' -e "s|$pat|$rep|g" "$f"
  done
}
scrub "/Users/brian" "/Users/user"
scrub "/home/brian" "/home/user"
scrub "github.com/tazitoo/tabular_embeddings" "<this-repository>"
scrub "tazitoo" "anonymous"
scrub "brian" "user"
scrub "morg\.local" "gpuhost"
scrub "galactus" "head"
scrub "nova4" "worker5"
scrub "firelord4" "worker4"
scrub "firelord" "worker4"
scrub "surfer4" "worker1"
scrub "surfer" "worker1"
scrub "terrax4" "worker2"
scrub "terrax" "worker2"
scrub "octo4" "worker3"
scrub "[[:<:]]octo[[:>:]]" "worker3"
scrub "[[:<:]]morg[[:>:]]" "gpuhost"
scrub "/private/tmp/claude-501/[^\"' ]*" "/tmp/labels"
scrub "192\.168\.[0-9]*\.[0-9]*" "10.0.0.1"

# ---- verify ------------------------------------------------------------------
leftover=$(grep -rIn -i -E "brian|tazitoo|surfer|terrax|octo4|firelord|morg\.local|galactus|nova4|/Users/brian|/home/brian" "$OUT" || true)
if [[ -n "$leftover" ]]; then
  echo "IDENTIFYING STRINGS REMAIN:"; echo "$leftover" | head -20; exit 1
fi

# ---- single anonymous commit ------------------------------------------------
cd "$OUT"
git init -q -b main
git add -A
GIT_AUTHOR_NAME="Anonymous Authors" GIT_AUTHOR_EMAIL="anonymous@example.com" \
GIT_COMMITTER_NAME="Anonymous Authors" GIT_COMMITTER_EMAIL="anonymous@example.com" \
git commit -q -m "INCEPT: code release (anonymized)"
echo "export at $OUT: $(git ls-files | wc -l) files, $(du -sh --exclude=.git . 2>/dev/null | cut -f1 || du -sh . | cut -f1)"
