#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

IMAGE_PATH="${1:-${REPO_DIR}/examples/richhf/images/1.jpg}"
QUESTION="${2:-What part of a mug should be gripped to lift it?}"
SAMPLE_ID="${3:-demo_local_qwen}"
LOCAL_MODEL_PATH="${AGENT_HARNESS_LOCAL_VLM_PATH:-/NEW_EDS/miaojw/models/Qwen2.5-VL-3B-Instruct}"

export AGENT_HARNESS_LLM_MODE="local_qwen_vl"
export AGENT_HARNESS_LOCAL_VLM_PATH="${LOCAL_MODEL_PATH}"

run_py agent_afford_harness.harness.run_single_case \
  --image "${IMAGE_PATH}" \
  --question "${QUESTION}" \
  --sample-id "${SAMPLE_ID}"

