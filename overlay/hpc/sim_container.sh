#!/usr/bin/env bash
# Simulation interpreter wrapper used by the "episode" job.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
: "${HANDSON_SIM_IMAGE:?set the prepared Isaac Lab 2.3.2 SIF or unpacked container}"
: "${HANDSON_JOB_RUN:?launch through the HPC episode job}"
: "${CUDA_VISIBLE_DEVICES:?missing Slurm GPU mask}"
runtime=${HANDSON_CONTAINER_RUNTIME:-/hpc2ssd/softwares/singularity-ce/4.1.3/bin/singularity}
run=$HANDSON_JOB_RUN
mkdir -p "$run"/{home,kit-cache,kit-data,kit-logs}
# Keep the host absolute episode and receipt paths valid inside the evaluator worker.
exec "$runtime" exec --nv --cleanenv --home "$run/home:/home/handson" \
  --bind "$root:$root" --bind "$run:$run" --bind "${TMPDIR:?}:/tmp" \
  --bind "$run/kit-cache:/isaac-sim/kit/cache" --bind "$run/kit-data:/isaac-sim/kit/data" \
  --bind "$run/kit-logs:/isaac-sim/kit/logs" --pwd "$root" \
  --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" --env PYTHONNOUSERSITE=1 \
  --env "PYTHONPATH=$root/handson:$root/third_party/IsaacLab-HandsOn/source/isaaclab:$root/.sim-packages" \
  --env ACCEPT_EULA=Y --env OMNI_KIT_ACCEPT_EULA=YES --env HANDSON_LIVE_VIEW=0 \
  "$HANDSON_SIM_IMAGE" /isaac-sim/python.sh "$@"
