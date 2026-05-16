#!/bin/bash
# Smoke-test the full Q pipeline end-to-end on coffee with 4 episodes per bucket.
# Validates: convert_v1_to_lerobot.py, precache_features.py, factory.py multi-dataset
# patch, QValueLabelDataset, and lerobot-train Q-function dispatch.
#
# Submit:
#   sbatch compute_rtx6000_mimicgen.sh \
#       bash experiments/mg_dataset_v1/smoke_q_lerobot.sh
set -euxo pipefail

TASK_SLUG="coffee"
SHARED_ROOT="/storage/home/hcoda1/6/vgiridhar6/shared/mimicgen/lerobot_datasets"
SRC_HDF5_ROOT="${SHARED_ROOT}/v1_q5_q3jitter_play"
SMOKE_ROOT="${SHARED_ROOT}/v1_lerobot_smoke"

POLICY_CKPT="/storage/home/hcoda1/6/vgiridhar6/forks/lerobot/outputs/train/mimicgen_${TASK_SLUG}_d0_act_simple/checkpoints/100000/pretrained_model"
Q_OUTPUT_DIR="${SHARED_ROOT}/q_lerobot_ckpts_smoke/${TASK_SLUG}"
# Isolated cache dir (smoke datasets have different N_frames than real ones; can't
# share the policy-checkpoint default cache without collision).
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

# Clean smoke output dirs (lerobot-train will recreate Q_OUTPUT_DIR itself).
rm -rf "${SMOKE_ROOT}" "${Q_OUTPUT_DIR}"
mkdir -p "${SMOKE_ROOT}"

echo "===== STEP 1: convert (4 eps per bucket) ====="
"${PYBIN}" -u experiments/mg_dataset_v1/convert_v1_to_lerobot.py \
    --src_root "${SRC_HDF5_ROOT}" \
    --dst_root "${SMOKE_ROOT}" \
    --task "${TASK_SLUG}" \
    --max_episodes 4

echo "===== STEP 2: precache ====="
for REPO_ID in "${REPO_Q5}" "${REPO_Q3}" "${REPO_PLAY}"; do
    "${PYBIN}" -u src/lerobot/policies/act_simple/precache_features.py \
        --policy_checkpoint "${POLICY_CKPT}" \
        --dataset_root "${SMOKE_ROOT}/${REPO_ID}" \
        --repo_id "${REPO_ID}" \
        --cache_root "${PRECACHE_ROOT}" \
        --chunk 32
done

echo "===== STEP 3: lerobot-train (10 steps) ====="
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

echo "===== SMOKE OK ====="
