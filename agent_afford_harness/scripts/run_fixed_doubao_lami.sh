#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

IMAGE_PATH="/data9/data/miaojw/projects26/RoboAfford/images/00.jpg"
QUESTION="What part of a mug should be gripped to lift it? Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be rounded to two decimal places, indicating the absolute pixel locations of the points in the image."
SAMPLE_ID="fixed_doubao_lami"

export AGENT_HARNESS_API_MODEL="${AGENT_HARNESS_API_MODEL:-doubao-seed-2-0-lite-260215}"
export AGENT_HARNESS_API_BASE_URL="${AGENT_HARNESS_API_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}"
export CUDA_VISIBLE_DEVICES="4"

if [[ -z "${ARK_API_KEY:-}" && -z "${AGENT_HARNESS_API_KEY:-}" ]]; then
  echo "ERROR: missing ARK_API_KEY (or AGENT_HARNESS_API_KEY)." >&2
  exit 1
fi

run_py agent_afford_harness.harness.run_fixed_doubao_lami \
  --image "${IMAGE_PATH}" \
  --question "${QUESTION}" \
  --sample-id "${SAMPLE_ID}"

