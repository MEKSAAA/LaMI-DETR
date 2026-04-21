#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

TRACE_PATH="${1:-${HARNESS_DIR}/outputs/traces/demo_richhf.json}"
OUT_PATH="${2:-${HARNESS_DIR}/outputs/debug_images/demo.png}"

run_py agent_afford_harness.harness.visualize_trace \
  --trace "${TRACE_PATH}" \
  --out "${OUT_PATH}"

