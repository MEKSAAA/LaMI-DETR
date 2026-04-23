#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

TRACES_DIR="${1:-/data9/data/miaojw/projects26/LaMI-DETR/agent_afford_harness/outputs/traces}"
GT_FILE="${2:-/data9/data/miaojw/projects26/RoboAfford/annotations_normxy.json}"
DATA_ROOT="${3:-/data9/data/miaojw/projects26/RoboAfford}"
OUTPUT_FILE="${4:-/data9/data/miaojw/projects26/LaMI-DETR/agent_afford_harness/outputs/predictions/trace_eval_summary.json}"

if ! python -c "import numpy, PIL, tqdm" >/dev/null 2>&1; then
  echo "ERROR: missing Python deps. Please install: numpy pillow tqdm" >&2
  exit 1
fi

run_py agent_afford_harness.harness.eval_traces \
  --traces-dir "${TRACES_DIR}" \
  --gt-file "${GT_FILE}" \
  --data-root "${DATA_ROOT}" \
  --output-file "${OUTPUT_FILE}"
