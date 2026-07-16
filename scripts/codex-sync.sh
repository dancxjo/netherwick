#!/usr/bin/env bash

set -euo pipefail

ROOT="${1:-$(pwd)}"
cd "$(cd "$ROOT" && pwd)"

if [[ ! -d .git ]]; then
  echo "codex-sync: no git repository at $(pwd)" >&2
  exit 1
fi

if [[ -z "$(git status --porcelain)" ]]; then
  echo "No repository changes to process."
  git status -sb
  exit 0
fi

RUST_LOG=error codex --ask-for-approval never exec --cd "$(pwd)" --sandbox danger-full-access --ephemeral <<'EOF'
You are continuing the workflow in this repository. Do exactly this:

1) Inspect repository state and current changes with `git diff` and `git status`.
2) Split unstaged/staged edits into a small number of semantically coherent commit groups.
3) Stage and commit each group with a distinct, imperative commit message that you generate.
4) Run `git pull --ff-only`.
5) If `git pull --ff-only` fails, resolve any merge conflicts in a conservative way (preserve local intended changes while honoring upstream changes) and continue.
6) Push the result with `git push`.

Rules:
- Keep commits minimal and focused.
- Do not include unrelated files in a group.
- If no changes are present, report that clearly and stop.
- Keep running commands limited to the workflow above.
- Show a brief final summary of commits and any conflict resolutions.
EOF
