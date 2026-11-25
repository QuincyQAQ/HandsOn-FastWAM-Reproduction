#!/usr/bin/env bash
set -euo pipefail
HANDSON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HANDSON_ROOT/outputs/fastwam-cup"
exec bash "$HANDSON_ROOT/run_handson_npu.sh" scripts/serve_paper_policy.py \
  --model fastwam \
  --checkpoint /data1/qli736/robot/data/weights/handson/fastwam/step-100000/mp_rank_00_model_states.pt \
  --checkpoint-sha256 9d8532d0ad8b62596ba070f7f663c9e58d67a059523b7fe4c21ea7028bedf46f \
  --text-cache /data1/qli736/robot/weights/handson/fastwam/care-tasks.safetensors \
  --task t25_place_cup_on_plate --bind 127.0.0.1 --port 8630 \
  --metadata-output outputs/fastwam-cup/policy-metadata.json
