#!/usr/bin/env bash

set -euo pipefail

ROOT="${1:-$(pwd)}"
cd "$(cd "$ROOT" && pwd)"

if [[ ! -d .git ]]; then
  echo "codex-sync: no git repository at $(pwd)" >&2
  exit 1
fi

STATUS_BRIEF="$(git status --short --branch)"
DIRTY="$(git status --porcelain)"

if [[ -z "${DIRTY}" ]]; then
  echo "No repository changes to process."
  echo "${STATUS_BRIEF}"
  echo "Syncing branch with origin..."
  git pull --ff-only
  if echo "${STATUS_BRIEF}" | grep -qE '\\[ahead [0-9]+\\]'; then
    git push
  fi
  exit 0
fi

TMP_ALL="$(mktemp)"
TMP_SUMMARY="$(mktemp)"
trap 'rm -f "$TMP_ALL" "$TMP_SUMMARY"' EXIT

RUST_LOG=error codex --ask-for-approval never exec --cd "$(pwd)" --sandbox danger-full-access --ephemeral --output-last-message "$TMP_SUMMARY" <<'EOF' >"$TMP_ALL" 2>&1
You are the repo-sync assistant for this workspace.

Do exactly this workflow:

1) Inspect working state with:
   - `git status --short --branch`
   - `git diff --name-status`
   - `git diff --cached --name-status`
   - `git diff`
   - `git diff --cached`

2) Produce a concise summary (4-10 bullet points) of what changed.

3) Update CHANGELOG.md under `## Unreleased`:
   - If this section contains only the existing placeholder line, replace it with your new summary.
   - If `## Unreleased` already has content, add a new heading `### Auto-sync (YYYY-MM-DD)` using today’s date and append your summary bullets.
   - Do not remove prior release entries.

4) Stage and commit all edits into minimal semantic commit groups with imperative messages you generate.
   (Include changelog edits in the same commit group as the code changes they describe.)

5) Run `git pull --ff-only`.
   If pull fails, resolve merge conflicts conservatively (preserve local intended changes + upstream changes), then continue.

6) Run `git push`.

7) Final response: short final summary including
   - what was summarized,
   - changelog update made,
   - commits pushed,
   - pull/push result.

Rules:
- No CI/build/test commands.
- No extra files.
- Keep commands strictly within this workflow.
EOF

if [[ -s "${TMP_SUMMARY}" ]]; then
  cat "${TMP_SUMMARY}"
else
  cat "${TMP_ALL}"
fi
