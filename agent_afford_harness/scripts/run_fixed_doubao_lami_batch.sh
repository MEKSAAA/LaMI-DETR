#!/usr/bin/env bash
set -eo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

ANNOTATIONS_FILE="/data9/data/miaojw/projects26/RoboAfford/annotations_normxy.json"
START="0"
LIMIT="0"

export CUDA_VISIBLE_DEVICES="4"

export AGENT_HARNESS_API_MODEL="doubao-seed-2-0-lite-260215"
export AGENT_HARNESS_API_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"

if [[ -z "$ARK_API_KEY" && -z "$AGENT_HARNESS_API_KEY" ]]; then
  echo "ERROR: missing ARK_API_KEY (or AGENT_HARNESS_API_KEY)." >&2
  exit 1
fi

run_py agent_afford_harness.harness.run_fixed_doubao_lami_batch \
  --annotations-file "$ANNOTATIONS_FILE" \
  --start "$START" \
  --limit "$LIMIT"

