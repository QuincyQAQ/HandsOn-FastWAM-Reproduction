#!/usr/bin/env bash
# Live Isaac Sim window, fresh model requests and native action execution.
set -euo pipefail
HANDSON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo 'Run this command in a terminal on the local PC desktop, not in the server SSH terminal.' >&2
  exit 1
fi
export HANDSON_LIVE_VIEW=1 HEADLESS=0 LIVESTREAM=0
export HANDSON_LIVE_RENDER_SPP="${HANDSON_LIVE_RENDER_SPP:-32}"
export HANDSON_LIVE_FRESH_CAPTURE="${HANDSON_LIVE_FRESH_CAPTURE:-0}"
export PYTHONUNBUFFERED=1
"$HANDSON_ROOT/../.venvs/handson-sim/bin/python" - <<'PY'
import json
import urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with opener.open('http://127.0.0.1:8630', timeout=10) as response:
        metadata = json.load(response)
    assert metadata['model'] == 'fastwam'
    assert metadata['checkpoint_sha256'] == '9d8532d0ad8b62596ba070f7f663c9e58d67a059523b7fe4c21ea7028bedf46f'
except Exception as exc:
    raise SystemExit(f'FastWAM server/tunnel unavailable on 127.0.0.1:8630: {exc}')
print('Live FastWAM checkpoint verified. Starting Isaac Sim GUI; each episode uses new camera observations.', flush=True)
PY
exec bash "$HANDSON_ROOT/run_fastwam_cup_eval.sh" "$@"
