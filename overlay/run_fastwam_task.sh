#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HANDSON_LOCAL_PYTHON:-/home/quincylee/miniconda3/bin/python}" "$ROOT/scripts/local/run_fastwam_task.py" "$@"
