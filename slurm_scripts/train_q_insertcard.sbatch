#!/bin/bash
# Q-function on the BIMANUAL insert-card-into-wallet merged dataset (task 2).
# 14-dim action chunks, h=32 == BC chunk, ALL THREE cameras, text conditioning
# ("Insert card into wallet" from the task key), per-episode rewards via the
# episode_labels sentinel (43 success -> terminal 1.0, 8 failure -> 0.0).
# Base L2 negatives recipe (swap + tube + trev, margin 0.25) — the recipe that
# produced the shipped task-1 value function. H100 inferno (L40S inferno queue
# has been stuck for days).
#
# Submit from repo root:
#   sbatch slurm_scripts/train_q_stackcups.sbatch
#   PROBE=1 sbatch slurm_scripts/train_q_stackcups.sbatch
#
#SBATCH -J q_insertcard
#SBATCH --account=gts-agarg35-ideasci23_dgx
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -p gpu-h100
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=64G
#SBATCH -t12:00:00
#SBATCH -q inferno
#SBATCH --requeue
#SBATCH --output=slurm_out/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vgiridhar6@gatech.edu

set -euo pipefail

REPO=/storage/project/r-agarg35-0/vgiridhar6/lerobot
cd "$REPO"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot-q
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME=/storage/scratch1/6/vgiridhar6/hf
export WANDB_MODE=offline
export WANDB_DIR="$HOME/scratch/wandb"
export WANDB_CACHE_DIR="$HOME/scratch/wandb/cache"
export WANDB_ARTIFACT_DIR="$HOME/scratch/wandb/artifacts"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# The train/test episode holdout is seeded via hash((seed, repo_id, bucket)) and
# Python string hashing is salted per process — pin it so the split is identical
# across the training run, requeue resumes, and the G2 gate reconstruction.
export PYTHONHASHSEED=0

DATASET=VarunGiridhar3/KT_insert_card_into_wallet_20260808_merged
DATA_ROOT=/storage/scratch1/6/vgiridhar6/lerobot_cache/VarunGiridhar3/KT_insert_card_into_wallet_20260808_merged

PROBE="${PROBE:-0}"
STEPS="${STEPS:-30000}"
BATCH="${BATCH:-32}"
JOB="${JOB:-qf_insertcard_h32}"
SAVE_FREQ=2500
LOG_FREQ=50
TEST_FREQ=200
TEST_N_BATCHES=4
WARMUP=2000

if [ "$PROBE" = "1" ]; then
  STEPS=30; LOG_FREQ=1; SAVE_FREQ=30; TEST_FREQ=30; TEST_N_BATCHES=2; WARMUP=10
  JOB="${JOB}_probe"
fi

OUT_DIR="$REPO/outputs/train/$JOB"

# A dir left by a run that died before its first checkpoint has nothing to
# resume — remove it so the fresh-start path doesn't hit FileExistsError.
if [ -d "$OUT_DIR" ] && [ ! -d "$OUT_DIR/checkpoints" ]; then
  echo "=== removing stale checkpoint-less $OUT_DIR ==="
  rm -rf "$OUT_DIR"
fi

if [ -f "$OUT_DIR/checkpoints/last/pretrained_model/train_config.json" ]; then
  echo "=== resuming from $OUT_DIR/checkpoints/last ==="
  exec accelerate launch --num_processes=1 --mixed_precision=bf16 \
    $(which lerobot-train) \
    --config_path="$OUT_DIR/checkpoints/last/pretrained_model/train_config.json" \
    --resume=true
fi

exec accelerate launch --num_processes=1 --mixed_precision=bf16 \
  $(which lerobot-train) \
  --job_name="$JOB" \
  --policy.type=q_function \
  --policy.push_to_hub=false \
  --policy.dino_model_name=facebook/dinov2-large \
  --policy.dim_model=1024 \
  --policy.n_heads=16 \
  --policy.dim_feedforward=4096 \
  --policy.n_decoder_layers=18 \
  --policy.image_resize_h=224 \
  --policy.image_resize_w=224 \
  --policy.camera_keys='[observation.images.left_wrist,observation.images.right_wrist,observation.images.top]' \
  --policy.reward_mode=sparse \
  --policy.step_reward=0.0 \
  --policy.terminal_bonuses='{q5: 1.0, play: 0.0}' \
  --policy.bucket_overrides="{$DATASET: episode_labels}" \
  --policy.v_min=-0.01 \
  --policy.v_max=1.01 \
  --policy.hl_gauss_sigma=0.0075 \
  --policy.use_text_conditioning=true \
  --policy.text_encoder_model=google/t5-v1_1-base \
  --policy.language_key=task \
  --policy.h=32 \
  --policy.gamma=0.99 \
  --policy.target_tau=0.005 \
  --policy.neg_margin_weight=0.25 \
  --policy.optimizer_lr=2e-4 \
  --policy.optimizer_lr_backbone=6e-5 \
  --policy.optimizer_weight_decay=1e-4 \
  --policy.lr_scheduler=cosine_decay_with_warmup \
  --policy.lr_warmup_steps="$WARMUP" \
  --policy.lr_decay_steps="$STEPS" \
  --policy.lr_decay_min=1e-6 \
  --dataset.repo_id="$DATASET" \
  --dataset.root="$DATA_ROOT" \
  --dataset.video_backend=pyav \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --log_freq="$LOG_FREQ" \
  --save_freq="$SAVE_FREQ" \
  --eval_freq=0 \
  --test_split_ratio=0.1 \
  --test_freq="$TEST_FREQ" \
  --test_n_batches="$TEST_N_BATCHES" \
  --num_workers=6 \
  --cudnn_deterministic=false \
  --seed=42 \
  --output_dir="$OUT_DIR" \
  --wandb.enable=true \
  --wandb.entity=vgiridhar6 \
  --wandb.project=awm \
  --wandb.disable_artifact=true
