#!/bin/bash
# Train BC (diffusion) on the merged q5+q3_termjitter successes for one task.
# Saves ckpts often (every 500 steps) so we get an undertrained early ckpt
# (step 1000) for play rollouts.
#
# Usage:
#   sbatch scripts/compute_rtx6000_mimicgen.sh \
#       bash experiments/mg_dataset_v1/run_bc_train.sh <task_slug>
# where <task_slug> ∈ {square, threading, coffee}.
set -euxo pipefail

TASK_SLUG="${1:?usage: $0 <square|threading|coffee>}"
case "$TASK_SLUG" in
    square)    ENV_TASK=Square_D0    ;;
    threading) ENV_TASK=Threading_D0 ;;
    coffee)    ENV_TASK=Coffee_D0    ;;
    *) echo "unknown task: $TASK_SLUG"; exit 1 ;;
esac

DATASET_ROOT="/storage/home/hcoda1/6/vgiridhar6/shared/mimicgen/lerobot_datasets/v1_q5_q3jitter_play"
TASK_DIR="${DATASET_ROOT}/${TASK_SLUG}"
BC_INPUT_HDF5="${TASK_DIR}/bc_input/demo.hdf5"
LEROBOT_DS_ROOT="${TASK_DIR}/bc_input/lerobot_ds"
BC_OUTPUT_DIR="${DATASET_ROOT}/bc_ckpts/${TASK_SLUG}"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::UserWarning:torchvision.io._video_deprecation_warning"
PYBIN="/storage/home/hcoda1/6/vgiridhar6/.conda/envs/lerobot-mimicgen/bin/python"

# torchcodec needs FFmpeg7 libs from PyAV; mirror run_mimicgen_train.sh's tweak.
CONDA_LIB="/storage/project/r-agarg35-0/vgiridhar6/.conda/envs/lerobot-mimicgen/lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${CONDA_LIB}/python3.10/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

cd /storage/home/hcoda1/6/vgiridhar6/forks/lerobot

# Step 1: merge q5 + q3_termjitter successes.
"${PYBIN}" -u experiments/mg_dataset_v1/merge_for_bc.py --task "${TASK_SLUG}"

# Step 2: convert merged HDF5 → LeRobot v3.0 dataset format.
rm -rf "${LEROBOT_DS_ROOT}"
"${PYBIN}" -u scripts/convert_mimicgen_to_lerobot.py \
    --hdf5_path "${BC_INPUT_HDF5}" \
    --repo_id "local/v1_${TASK_SLUG}_bc_input" \
    --root "${LEROBOT_DS_ROOT}" \
    --save_init_states \
    --force

# Step 3: train. save_freq=500 gives us step 1000/2000/etc ckpts for play rollouts.
"${PYBIN}" -u -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="local/v1_${TASK_SLUG}_bc_input" \
    --dataset.root="${LEROBOT_DS_ROOT}" \
    --policy.type=diffusion \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --output_dir="${BC_OUTPUT_DIR}" \
    --job_name="v1_bc_${TASK_SLUG}" \
    --steps=10000 \
    --batch_size=32 \
    --num_workers=8 \
    --log_freq=200 \
    --save_freq=500 \
    --eval_freq=10000 \
    --env.type=mimicgen \
    --env.task="${ENV_TASK}" \
    --env.init_states_path="${LEROBOT_DS_ROOT}/meta/init_states.pt" \
    --eval.n_episodes=10 \
    --eval.batch_size=1 \
    --wandb.enable=false
