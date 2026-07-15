#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state="$(mktemp)"
trap 'rm -f "$state"' EXIT

printf '{"generation_in_stage":243}\n' > "$state"
[[ "$("$root/scripts/neat-generation-limit.sh" "$state" "" 120 120)" == 363 ]]
printf '{"generation_in_stage":363}\n' > "$state"
[[ "$("$root/scripts/neat-generation-limit.sh" "$state" "" 120 120)" == 483 ]]
[[ "$("$root/scripts/neat-generation-limit.sh" "$state" 77 120 120)" == 77 ]]

echo "NEAT continuation generation limit tests passed"
