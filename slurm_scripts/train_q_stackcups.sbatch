#!/bin/bash
# Q-function on the right-arm stack-cups dataset (real robot, Q-planning rebuttal).
#
# Adapted from scripts/train_q_libero_ddp.sh to:
#   * single L40S (inferno, gts-agarg35-ideas_l40s), batch 32, lr sqrt-scaled
#   * h=32 == BC chunk_size (ABC-DiT and diffusion fallback both trained at 32) (planner loader enforces equality)
#   * cameras right_wrist + top; task string from meta/tasks.parquet ("Stack cups")
#   * per-episode rewards from meta/episode_labels.json via the new
#     bucket_overrides sentinel "episode_labels": success ep -> q5 (terminal 1.0),
#     failure ep -> play (terminal 0.0). Reward def: sparse, r=0 in-episode,
#     terminal frame += bonus; gamma=0.99; HL-Gauss over [-0.01, 1.01].
#   * synthetic negatives ON (margin 0.25, swap + tube(1,2)sigma + temporal) —
#     with ~100 mostly-success demos they carry the contrastive signal.
#
# Submit from repo root:
#   sbatch slurm_scripts/train_q_stackcups.sbatch
#   PROBE=1 sbatch slurm_scripts/train_q_stackcups.sbatch
#
#SBATCH -J q_stackcups
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH -N1 --gres=gpu:L40S:1
#SBATCH -p gpu-l40s
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

DATASET=VarunGiridhar3/KT_stack_cups_20260807_211827_right
DATA_ROOT=/storage/scratch1/6/vgiridhar6/lerobot_cache

PROBE="${PROBE:-0}"
STEPS="${STEPS:-30000}"
BATCH="${BATCH:-32}"
JOB="${JOB:-qf_stackcups_h32}"
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
  --policy.camera_keys='[observation.images.right_wrist,observation.images.top]' \
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
