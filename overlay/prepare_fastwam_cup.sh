#!/usr/bin/env bash
set -euo pipefail
HANDSON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEXT_ROOT=/data1/qli736/robot/weights/handson/wan-text
CACHE=/data1/qli736/robot/weights/handson/fastwam/care-tasks.safetensors
if [[ ! -f "$TEXT_ROOT/models_t5_umt5-xxl-enc-bf16.pth" ]]; then
  echo 'Missing frozen UMT5 encoder. Run scripts/download_fastwam_text_assets.py in tmux first.' >&2
  exit 1
fi
if [[ -e "$CACHE" ]]; then
  echo "Cache already exists; inspect its receipt instead of replacing it: $CACHE" >&2
  exit 1
fi
bash "$HANDSON_ROOT/run_handson_npu.sh" -m handson.policies.encode_text \
  --model fastwam --device "${HANDSON_FASTWAM_DEVICE:-npu:8}" \
  --encoder "$TEXT_ROOT/models_t5_umt5-xxl-enc-bf16.pth" \
  --tokenizer "$TEXT_ROOT/google/umt5-xxl" \
  --task t25_place_cup_on_plate --task t26_place_fruit_in_bucket --task t28_place_tableware \
  --output "$CACHE"
