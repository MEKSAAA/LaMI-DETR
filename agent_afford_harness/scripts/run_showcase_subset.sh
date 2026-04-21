#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

SUBSET_JSON="${1:-}"
OUT_JSON="${2:-}"

ARGS=()
if [[ -n "${SUBSET_JSON}" ]]; then
  ARGS+=(--subset-json "${SUBSET_JSON}")
fi
if [[ -n "${OUT_JSON}" ]]; then
  ARGS+=(--output "${OUT_JSON}")
fi

run_py agent_afford_harness.harness.run_subset "${ARGS[@]}"

