#!/bin/bash
# End-to-end Q-function training for one task using the standard `lerobot-train` CLI.
#
# Steps performed:
#   1. Convert this task's three HDF5 buckets to LeRobotDatasets (if not already done).
#   2. Pre-cache vision features through the trained act_simple backbone for each.
#   3. Launch `lerobot.scripts.lerobot_train` with three repo_ids → MultiLeRobotDataset,
#      policy.type=q_function, vision_backbone=resnet18_cached.
#
# Usage:
#   sbatch compute_rtx6000_mimicgen.sh \
#       bash experiments/mg_dataset_v1/run_q_train_lerobot.sh <task_slug>
# where <task_slug> ∈ {square, threading, coffee}.
set -euxo pipefail

TASK_SLUG="${1:?usage: $0 <square|threading|coffee>}"
case "$TASK_SLUG" in
    square|threading|coffee) ;;
    *) echo "unknown task: $TASK_SLUG"; exit 1 ;;
esac

# ── Paths ─────────────────────────────────────────────────────────────────────
SHARED_ROOT="/storage/home/hcoda1/6/vgiridhar6/shared/mimicgen/lerobot_datasets"
SRC_HDF5_ROOT="${SHARED_ROOT}/v1_q5_q3jitter_play"
LEROBOT_ROOT="${SHARED_ROOT}/v1_lerobot"

POLICY_CKPT="/storage/home/hcoda1/6/vgiridhar6/forks/lerobot/outputs/train/mimicgen_${TASK_SLUG}_d0_act_simple/checkpoints/100000/pretrained_model"

# Mirror the BC naming convention: outputs/train/mimicgen_<task>_d0_<policy>.
# Suffix _h10 (matched to BC.chunk_size) + optional VARIANT_TAG env var so
# distributional-critic ablations don't clobber each other.
VARIANT_TAG="${VARIANT_TAG:-}"
JOB_NAME="mimicgen_${TASK_SLUG}_d0_q_function_h10${VARIANT_TAG:+_${VARIANT_TAG}}"
Q_OUTPUT_DIR="/storage/home/hcoda1/6/vgiridhar6/forks/lerobot/outputs/train/${JOB_NAME}"

REPO_Q5="mg_${TASK_SLUG}_q5"
REPO_Q3="mg_${TASK_SLUG}_q3_termjitter"
REPO_PLAY="mg_${TASK_SLUG}_play"

# BUCKETS env var override: comma-separated subset of {q5,q3_termjitter,play}.
# When set, only the named buckets are converted/precached/trained on. Default
# is the full 3-bucket mix.
BUCKETS="${BUCKETS:-q5,q3_termjitter,play}"
IFS=',' read -ra USE_BUCKETS <<< "$BUCKETS"

# ── Env ───────────────────────────────────────────────────────────────────────
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::UserWarning:torchvision.io._video_deprecation_warning"
PYBIN="/storage/home/hcoda1/6/vgiridhar6/.conda/envs/lerobot-mimicgen/bin/python"
CONDA_LIB="/storage/project/r-agarg35-0/vgiridhar6/.conda/envs/lerobot-mimicgen/lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${CONDA_LIB}/python3.10/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

cd /storage/home/hcoda1/6/vgiridhar6/forks/lerobot

# ── Step 1: convert (idempotent — skips if dst already a LeRobotDataset) ──────
for BUCKET in "${USE_BUCKETS[@]}"; do
    DST="${LEROBOT_ROOT}/mg_${TASK_SLUG}_${BUCKET}"
    if [[ -f "${DST}/meta/info.json" ]]; then
        echo "[convert] skipping ${DST} (already exists)"
        continue
    fi
    "${PYBIN}" -u experiments/mg_dataset_v1/convert_v1_to_lerobot.py \
        --src_root "${SRC_HDF5_ROOT}" \
        --dst_root "${LEROBOT_ROOT}" \
        --task "${TASK_SLUG}" \
        --bucket "${BUCKET}"
done

# ── Step 2: pre-cache features (idempotent — skips if cache exists) ──
# Cache lives under the policy checkpoint so it's bound to the BC backbone version:
#   ${POLICY_CKPT}/encoded_backbone/<repo_id>/{meta.json, <cam>.mmap}
PRECACHE_ROOT="${POLICY_CKPT}/encoded_backbone"
REPO_IDS_TO_PROCESS=()
for BUCKET in "${USE_BUCKETS[@]}"; do REPO_IDS_TO_PROCESS+=( "mg_${TASK_SLUG}_${BUCKET}" ); done
for REPO_ID in "${REPO_IDS_TO_PROCESS[@]}"; do
    DS_ROOT="${LEROBOT_ROOT}/${REPO_ID}"
    if [[ -f "${PRECACHE_ROOT}/${REPO_ID}/meta.json" ]]; then
        echo "[precache] skipping ${REPO_ID} (cache exists at ${PRECACHE_ROOT}/${REPO_ID}/)"
        continue
    fi
    "${PYBIN}" -u src/lerobot/policies/act_simple/precache_features.py \
        --policy_checkpoint "${POLICY_CKPT}" \
        --dataset_root "${DS_ROOT}" \
        --repo_id "${REPO_ID}"
done

# ── Step 3: launch lerobot-train with the three buckets joined ────────────────
# Optional distributional-critic overrides via env vars; left unset → defaults.
TRAIN_EXTRA=()
[[ -n "${NUM_BINS:-}" ]]       && TRAIN_EXTRA+=( --policy.num_bins="${NUM_BINS}" )
[[ -n "${HL_GAUSS_SIGMA:-}" ]] && TRAIN_EXTRA+=( --policy.hl_gauss_sigma="${HL_GAUSS_SIGMA}" )
[[ -n "${V_MIN:-}" ]]          && TRAIN_EXTRA+=( --policy.v_min="${V_MIN}" )
[[ -n "${V_MAX:-}" ]]          && TRAIN_EXTRA+=( --policy.v_max="${V_MAX}" )
[[ -n "${REWARD_MODE:-}" ]]    && TRAIN_EXTRA+=( --policy.reward_mode="${REWARD_MODE}" )
[[ -n "${TARGET_TAU:-}" ]]     && TRAIN_EXTRA+=( --policy.target_tau="${TARGET_TAU}" )
# Optional: override Q's action-normalization stats with an external safetensors
# (typically BC's unnormalizer) so Q's and BC's action frames line up at eval.
[[ -n "${ACTION_STATS_PATH:-}" ]] && TRAIN_EXTRA+=( --policy.action_stats_path="${ACTION_STATS_PATH}" )
# STEPS handled inline below — not duplicated here.

# Comma-join repo_ids in draccus-list syntax.
REPO_LIST="$(IFS=,; echo "${REPO_IDS_TO_PROCESS[*]}")"
"${PYBIN}" -u -m lerobot.scripts.lerobot_train \
    --dataset.repo_ids="[${REPO_LIST}]" \
    --dataset.root="${LEROBOT_ROOT}" \
    --dataset.use_imagenet_stats=false \
    --policy.type=q_function \
    --policy.h=10 \
    --policy.vision_backbone=resnet18_cached \
    --policy.precache_root="${PRECACHE_ROOT}" \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --output_dir="${Q_OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    --steps="${STEPS:-20000}" \
    --batch_size=32 \
    --num_workers=4 \
    --log_freq=200 \
    --save_freq=2000 \
    --eval_freq=0 \
    --wandb.enable=true \
    --wandb.project=awm \
    --wandb.entity=pair-diffusion \
    "${TRAIN_EXTRA[@]}"
