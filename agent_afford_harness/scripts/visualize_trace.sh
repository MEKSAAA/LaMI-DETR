#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

TRACE_PATH="/data9/data/miaojw/projects26/LaMI-DETR/agent_afford_harness/outputs/traces/demo_api_doubao_grounding.json"
OUT_PATH="${2:-${HARNESS_DIR}/outputs/debug_images/demo.png}"

run_py agent_afford_harness.harness.visualize_trace \
  --trace "${TRACE_PATH}" \
  --out "${OUT_PATH}"

