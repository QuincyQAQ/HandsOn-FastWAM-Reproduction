#!/usr/bin/env bash
#SBATCH --job-name=handson-fastwam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=handson-%j.log
set -euo pipefail
umask 077

root=$(realpath "${HANDSON_ROOT:-${SLURM_SUBMIT_DIR:?submit from the restored HandsOn directory}}")
cd "$root"
: "${SLURM_JOB_ID:?run under Slurm}" "${CUDA_VISIBLE_DEVICES:?missing Slurm GPU mask}"
: "${FASTWAM_PYTHON:?provide the absolute CUDA policy Python path}"
: "${FASTWAM_CHECKPOINT:?provide the step-100000 checkpoint}" "${FASTWAM_TEXT_CACHE:?provide the task text cache}"
mode=${1:-serve}
if (( $# )); then shift; fi
case "$mode" in
  serve) if (( $# )); then echo "serve takes no episode arguments" >&2; exit 2; fi ;;
  episode) : "${HANDSON_SIM_PYTHON:?provide the simulation Python or container wrapper}" ;;
  *) echo "Usage: sbatch hpc/fastwam.sbatch {serve|episode --task TASK [--seed N | --paper-trial N]}" >&2; exit 2 ;;
esac
[[ -x "$FASTWAM_PYTHON" && -r "$FASTWAM_CHECKPOINT" && -r "$FASTWAM_TEXT_CACHE" && -f hpc/run_episode.py ]] || {
  echo "Check restored source, model/cache paths and CUDA environment" >&2; exit 2;
}

run_parent=${HANDSON_RUNS:-$root/outputs/hpc}
mkdir -p "$run_parent"
run=$(mktemp -d "$(realpath "$run_parent")/job-${SLURM_JOB_ID}.XXXXXX")
job_tmp=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/handson-${SLURM_JOB_ID}.XXXXXX")
policy_pid=
cleanup() {
  if [[ -n "$policy_pid" ]]; then
    kill "$policy_pid" 2>/dev/null || true
    wait "$policy_pid" 2>/dev/null || true
  fi
  rm -rf -- "$job_tmp"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export TMPDIR="$job_tmp" XDG_CACHE_HOME="$run/cache" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" HANDSON_FASTWAM_DEVICE=cuda:0
export PYTHONPATH="$root/handson${PYTHONPATH:+:$PYTHONPATH}"
port=${HANDSON_PORT:-$((20000 + SLURM_JOB_ID % 30000))}
url="http://127.0.0.1:$port"
receipt="$run/policy-metadata.json"
echo "Job=$SLURM_JOB_ID node=$(hostname) run=$run policy=$url"
"$FASTWAM_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Expected one Slurm-visible CUDA device")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("FastWAM needs BF16 on the allocated GPU")
print("CUDA", torch.version.cuda, torch.cuda.get_device_name(0), flush=True)
PY

"$FASTWAM_PYTHON" scripts/serve_paper_policy.py --model fastwam \
  --checkpoint "$FASTWAM_CHECKPOINT" --checkpoint-sha256 9d8532d0ad8b62596ba070f7f663c9e58d67a059523b7fe4c21ea7028bedf46f \
  --text-cache "$FASTWAM_TEXT_CACHE" --bind 127.0.0.1 --port "$port" \
  --metadata-output "$receipt" >"$run/policy.log" 2>&1 &
policy_pid=$!
# The server writes this fresh receipt after loading weights and binding the port.
until [[ -s "$receipt" ]] && "$FASTWAM_PYTHON" -m json.tool "$receipt" >/dev/null 2>&1; do
  if ! kill -0 "$policy_pid" 2>/dev/null; then
    echo "Policy exited during startup. Inspect $run/policy.log" >&2; exit 1
  fi
  sleep 2
done
"$FASTWAM_PYTHON" "$root/hpc/run_episode.py" --check-endpoint --url "$url" --metadata "$receipt"
echo "READY node=$(hostname) port=$port metadata=$receipt"
if [[ "$mode" == serve ]]; then
  wait "$policy_pid"
else
  export HANDSON_JOB_RUN="$run"
  "$FASTWAM_PYTHON" "$root/hpc/run_episode.py" --url "$url" --metadata "$receipt" --output "$run/episode" "$@"
fi
