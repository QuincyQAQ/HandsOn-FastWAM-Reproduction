#!/usr/bin/env bash
# Requires the existing remote FastWAM server and local port-8630 SSH tunnel.
set -euo pipefail
HANDSON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK="t25_place_cup_on_plate"
OUTPUT="${1:-outputs/fastwam-${TASK}-$(date +%Y%m%d_%H%M%S_%N)}"
SEED="${2:-0}"
exec bash "$HANDSON_ROOT/run_handson_sim.sh" -m handson.evaluation \
  --model fastwam --task "$TASK" \
  --policy-url http://127.0.0.1:8630 \
  --policy-metadata "$HANDSON_ROOT/local_runtime/fastwam-cup/policy-metadata.json" \
  --seed "$SEED" --output "$OUTPUT"
