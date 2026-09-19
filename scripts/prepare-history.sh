#!/usr/bin/env bash
# Prepends the release-drafter draft notes for TAG to the matching HISTORY.md,
# mirroring pydantic's release process: curate locally, land via a normal PR,
# only then publish the GitHub release (avoids pushing to a protected main directly).
set -euo pipefail

usage() {
  echo "Usage: $0 [--push] <tag>   e.g. $0 moors-0.2.11" >&2
  exit 1
}

PUSH=false
if [[ "${1:-}" == "--push" ]]; then
  PUSH=true
  shift
fi

[[ $# -eq 1 ]] || usage
TAG="$1"

if [[ "$TAG" == moors-* ]]; then
  HISTORY_PATH="moors/HISTORY.md"
elif [[ "$TAG" == pymoors-* ]]; then
  HISTORY_PATH="pymoors/HISTORY.md"
else
  echo "Tag must start with 'moors-' or 'pymoors-'" >&2
  exit 1
fi

command -v gh >/dev/null || { echo "GitHub CLI ('gh') is required: https://cli.github.com/" >&2; exit 1; }

echo "Fetching draft release body for $TAG..."
RELEASE_BODY="$(gh release view "$TAG" --json body -q .body)"

DATE="$(date -u +%F)"
HEADER="## [$TAG] - $DATE"
TMP="$(mktemp)"
{
  echo "$HEADER"
  echo
  echo "$RELEASE_BODY"
  echo
  echo
  if [[ -f "$HISTORY_PATH" ]]; then cat "$HISTORY_PATH"; fi
} > "$TMP"
mv "$TMP" "$HISTORY_PATH"

echo "Updated $HISTORY_PATH — review and curate it now (fix markdown, mark breaking changes, etc.)."

if [[ "$PUSH" == false ]]; then
  echo
  echo "When ready, run:"
  echo "  git checkout -b history/$TAG"
  echo "  git add $HISTORY_PATH"
  echo "  git commit -m \"docs(history): update from release $TAG\""
  echo "  git push -u origin history/$TAG"
  echo "  gh pr create --fill"
  exit 0
fi

BRANCH="history/$TAG"
git checkout -b "$BRANCH"
git add "$HISTORY_PATH"
git commit -m "docs(history): update from release $TAG"
git push -u origin "$BRANCH"
gh pr create --fill
