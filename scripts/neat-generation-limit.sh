#!/usr/bin/env bash
set -euo pipefail

state_path="${1:?state checkpoint path required}"
explicit_limit="${2:-}"
default_limit="${3:-120}"
increment="${4:-120}"

if [[ -n "$explicit_limit" ]]; then
    printf '%s\n' "$explicit_limit"
    exit 0
fi

limit="$default_limit"
if [[ -f "$state_path" ]] && command -v jq >/dev/null 2>&1; then
    completed="$(jq -er '.generation_in_stage' "$state_path" 2>/dev/null || true)"
    if [[ "$completed" =~ ^[0-9]+$ ]]; then
        limit="$((completed + increment))"
    fi
fi
printf '%s\n' "$limit"
