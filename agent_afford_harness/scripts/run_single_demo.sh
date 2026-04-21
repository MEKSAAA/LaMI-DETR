#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

IMAGE_PATH="${1:-${REPO_DIR}/examples/richhf/images/1.jpg}"
QUESTION="${2:-Highlight points on the flower.}"
SAMPLE_ID="${3:-demo_richhf}"

run_py agent_afford_harness.harness.run_single_case \
  --image "${IMAGE_PATH}" \
  --question "${QUESTION}" \
  --sample-id "${SAMPLE_ID}"

