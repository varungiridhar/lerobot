#!/bin/bash
# Train the Q-function policy on RoboTwin data (multicam dataset).
#
# The Q-function is a categorical HL-Gauss critic (DINOv2-large per camera view +
# frozen T5 + 18-layer transformer decoder). lerobot-train auto-wraps the dataset
# with QValueLabelDataset; RoboTwin demos are all expert successes so
# reward_mode=all_success (every episode terminal gets +1, no buckets).
#
# MODE=smoke  -> short pipeline-proving run (rtx6000)
# MODE=probe  -> probe-scale training run (l40s)
# Submit via scripts/train_q_robotwin_{smoke,probe}_sbatch.sh
set -uxo pipefail

MODE="${MODE:-smoke}"
WORKTREE=/storage/home/hcoda1/6/vgiridhar6/forks/lerobot-qfunction-robotwin
DATASET_ROOT="${DATASET_ROOT:-/storage/home/hcoda1/6/vgiridhar6/r-agarg35-0/robotwin/dataset/robotwin2.0_multicam}"
OUT_BASE=/storage/home/hcoda1/6/vgiridhar6/r-agarg35-0/robotwin/outputs/train

export HF_HOME=/storage/home/hcoda1/6/vgiridhar6/r-agarg35-0/hf_cache
export TMPDIR=/storage/project/r-agarg35-0/vgiridhar6/robotwin/tmp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Use this worktree's source without repointing the shared env's editable install.
export PYTHONPATH="$WORKTREE/src:${PYTHONPATH:-}"

module load anaconda3/2022.05.0.1
module load cuda/12.6.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /storage/project/r-agarg35-0/vgiridhar6/.conda/envs/lerobot-robotwin

if [ "$MODE" = "smoke" ]; then
    STEPS="${STEPS:-300}";   BATCH="${BATCH:-4}";  EVAL_FREQ=150;  SAVE_FREQ=300
else
    STEPS="${STEPS:-20000}"; BATCH="${BATCH:-12}"; EVAL_FREQ=2000; SAVE_FREQ=5000
fi

cd "$WORKTREE"
accelerate launch --num_processes=1 --mixed_precision=bf16 "$(which lerobot-train)" \
    --job_name="qf_robotwin_multicam_${MODE}" \
    --policy.type=q_function \
    --policy.push_to_hub=false \
    --policy.dino_model_name=facebook/dinov2-large \
    --policy.dim_model=1024 \
    --policy.n_heads=16 \
    --policy.dim_feedforward=4096 \
    --policy.n_decoder_layers=18 \
    --policy.image_resize_h=224 \
    --policy.image_resize_w=224 \
    --policy.camera_keys='[observation.images.cam_high, observation.images.cam_left_wrist, observation.images.cam_right_wrist]' \
    --policy.reward_mode=sparse \
    --policy.bucket_overrides='{local/robotwin2.0_multicam: q5}' \
    --policy.terminal_bonuses='{q5: 1.0}' \
    --policy.step_reward=0.0 \
    --policy.v_min=-0.01 \
    --policy.v_max=1.01 \
    --policy.hl_gauss_sigma=0.0075 \
    --policy.use_text_conditioning=true \
    --policy.text_encoder_model=google/t5-v1_1-base \
    --policy.language_key=task \
    --policy.h=32 \
    --policy.gamma=0.99 \
    --policy.target_tau=0.005 \
    --policy.optimizer_lr=1e-4 \
    --policy.optimizer_lr_backbone=3e-5 \
    --policy.optimizer_weight_decay=1e-4 \
    --dataset.repo_id=local/robotwin2.0_multicam \
    --dataset.root="$DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --batch_size="$BATCH" \
    --steps="$STEPS" \
    --log_freq=10 \
    --save_freq="$SAVE_FREQ" \
    --eval_freq="$EVAL_FREQ" \
    --test_split_ratio=0.05 \
    --num_workers=4 \
    --cudnn_deterministic=false \
    --output_dir="${OUTPUT_DIR:-$OUT_BASE/qf_robotwin_multicam_${MODE}}" \
    --wandb.enable=false
echo "TRAIN DONE ($MODE)"
