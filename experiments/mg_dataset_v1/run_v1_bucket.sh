#!/bin/bash
# Generate one (task, bucket) of the v1 dataset.
#
# Usage:
#   sbatch scripts/compute_rtx6000_mimicgen.sh \
#       bash experiments/mg_dataset_v1/run_v1_bucket.sh <task> <bucket>
# where:
#   <task>   ∈ {square, threading, coffee}
#   <bucket> ∈ {q5, q3_termjitter}
set -euxo pipefail

TASK="${1:?usage: $0 <task> <bucket>}"
BUCKET="${2:?usage: $0 <task> <bucket>}"

DATASET_ROOT="/storage/home/hcoda1/6/vgiridhar6/shared/mimicgen/lerobot_datasets/v1_q5_q3jitter_play"
CONFIG="${DATASET_ROOT}/configs/${TASK}_${BUCKET}.json"

if [[ ! -f "${CONFIG}" ]]; then
    echo "config not found: ${CONFIG}"
    exit 1
fi

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

PYBIN="/storage/home/hcoda1/6/vgiridhar6/.conda/envs/lerobot-mg-datagen/bin/python"

cd /storage/home/hcoda1/6/vgiridhar6/forks/lerobot
"${PYBIN}" -u -m mimicgen.scripts.generate_dataset \
    --config "${CONFIG}" \
    --auto-remove-exp
