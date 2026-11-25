#!/usr/bin/env bash
set -euo pipefail
HANDSON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$HANDSON_ROOT")"
cd "$HANDSON_ROOT"
exec env -u LD_LIBRARY_PATH -u PYTHONHOME \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="$HANDSON_ROOT/handson:$HANDSON_ROOT/third_party/IsaacLab-HandsOn/source/isaaclab" \
  OMNI_KIT_ACCEPT_EULA=YES \
  "$WORKSPACE_ROOT/.venvs/handson-sim/bin/python" "$@"
