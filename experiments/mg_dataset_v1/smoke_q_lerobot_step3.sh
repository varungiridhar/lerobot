#!/bin/bash
# Re-run just step 3 of smoke_q_lerobot.sh, reusing the converted+precached
# artifacts from a previous successful steps 1+2.
set -euxo pipefail

TASK_SLUG="coffee"
SHARED_ROOT="/storage/home/hcoda1/6/vgiridhar6/shared/mimicgen/lerobot_datasets"
SMOKE_ROOT="${SHARED_ROOT}/v1_lerobot_smoke"
Q_OUTPUT_DIR="${SHARED_ROOT}/q_lerobot_ckpts_smoke/${TASK_SLUG}"
PRECACHE_ROOT="${SMOKE_ROOT}/precache"

REPO_Q5="mg_${TASK_SLUG}_q5"
REPO_Q3="mg_${TASK_SLUG}_q3_termjitter"
REPO_PLAY="mg_${TASK_SLUG}_play"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::UserWarning:torchvision.io._video_deprecation_warning"
PYBIN="/storage/home/hcoda1/6/vgiridhar6/.conda/envs/lerobot-mimicgen/bin/python"
CONDA_LIB="/storage/project/r-agarg35-0/vgiridhar6/.conda/envs/lerobot-mimicgen/lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${CONDA_LIB}/python3.10/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

cd /storage/home/hcoda1/6/vgiridhar6/forks/lerobot
rm -rf "${Q_OUTPUT_DIR}"

"${PYBIN}" -u -m lerobot.scripts.lerobot_train \
    --dataset.repo_ids="[${REPO_Q5},${REPO_Q3},${REPO_PLAY}]" \
    --dataset.root="${SMOKE_ROOT}" \
    --dataset.use_imagenet_stats=false \
    --policy.type=q_function \
    --policy.vision_backbone=resnet18_cached \
    --policy.precache_root="${PRECACHE_ROOT}" \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --output_dir="${Q_OUTPUT_DIR}" \
    --job_name="smoke_q_${TASK_SLUG}" \
    --steps=10 \
    --batch_size=4 \
    --num_workers=0 \
    --log_freq=2 \
    --save_freq=10 \
    --eval_freq=0 \
    --wandb.enable=false
