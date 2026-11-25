#!/usr/bin/env bash
set -euo pipefail
HANDSON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
cd "$HANDSON_ROOT"
export PYTHONPATH="$HANDSON_ROOT/handson"
export PYTHONNOUSERSITE=1
export HANDSON_FASTWAM_DEVICE="${HANDSON_FASTWAM_DEVICE:-npu:8}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TORCHDYNAMO_DISABLE=1
exec "$HANDSON_ROOT/.venvs/fastwam-npu/bin/python" "$@"
