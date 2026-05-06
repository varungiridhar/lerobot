#!/bin/bash
# Train + in-loop eval a diffusion policy on a MimicGen task. Pass the task
# slug as $1: coffee_d0 | threading_d0 | square_d0.
# Submit via compute_rtx6000_mimicgen.sh.
set -euxo pipefail

TASK_SLUG="${1:?usage: $0 <coffee_d0|threading_d0|square_d0>}"
case "$TASK_SLUG" in
    coffee_d0)    ENV_TASK=Coffee_D0    ;;
    threading_d0) ENV_TASK=Threading_D0 ;;
    square_d0)    ENV_TASK=Square_D0    ;;
    *) echo "unknown task slug: $TASK_SLUG"; exit 1 ;;
esac

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::UserWarning:torchvision.io._video_deprecation_warning"
export PYTHONPATH="/storage/home/hcoda1/6/vgiridhar6/forks/lerobot-mimicgen-env/src:${PYTHONPATH:-}"

# torchcodec needs FFmpeg7 libs that PyAV ships with hashed names. Symlinks
# already exist in $CONDA_LIB; av.libs goes on the path so transitive deps
# resolve by literal hashed filenames.
CONDA_LIB="/storage/project/r-agarg35-0/vgiridhar6/.conda/envs/lerobot-mimicgen/lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${CONDA_LIB}/python3.10/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

JOB_NAME="mimicgen_${TASK_SLUG}_diffusion"
RUN_ID="$(date +%Y%m%d_%H%M%S)-${JOB_NAME}"
OUTPUT_DIR="/storage/home/hcoda1/6/vgiridhar6/forks/lerobot-mimicgen-env/outputs/train/${JOB_NAME}"
DATASET_ROOT="/storage/home/hcoda1/6/vgiridhar6/shared/mimicgen/lerobot_datasets/mimicgen_${TASK_SLUG}"

if [ "${CLEAN:-0}" = "1" ]; then
    rm -rf "${OUTPUT_DIR}"
fi

echo "=== Train + in-loop eval (task=${ENV_TASK}) ==="
python -u -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="local/mimicgen_${TASK_SLUG}" \
    --dataset.root="${DATASET_ROOT}" \
    --policy.type=diffusion \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    --steps=100000 \
    --batch_size=32 \
    --num_workers=8 \
    --log_freq=200 \
    --save_freq=10000 \
    --eval_freq=25000 \
    --env.type=mimicgen \
    --env.task="${ENV_TASK}" \
    --env.init_states_path="${DATASET_ROOT}/meta/init_states.pt" \
    --eval.n_episodes=10 \
    --eval.batch_size=1 \
    --wandb.enable=true \
    --wandb.project=awm \
    --wandb.entity=pair-diffusion \
    --wandb.disable_artifact=true \
    --wandb.run_id="${RUN_ID}"

echo "=== Done ==="
