#!/bin/bash
# Phase-3 round 1: Q-only finetune on autonomous rollouts (BC stays frozen).
#
# Data mix (multi-repo, per-repo episode_labels buckets):
#   * demo:    VarunGiridhar3/KT_stack_cups_20260807_211827_right (114 eps, 101 succ) — 68.9% of frames
#   * bc_r0:   VarunGiridhar3/rollout_stackcups_bc_r0_right       (30 eps, 12 succ, BC-only 40%)
#   * bcq_r0:  VarunGiridhar3/rollout_stackcups_bcq_r0_right      (20 eps, 13 succ, BC+Q 65%)
# Rollout successes are positives (terminal 1.0), rollout failures are REAL
# zero-return negatives (terminal 0.0). ACTION normalization stats pinned to the
# demo repo so Q1's action space == Q0's (warm-start + G2 comparability).
#
# Warm start from Q0 = qf_stackcups_h32_negft/010000 (weights only, fresh optim);
# negatives recipe inherited unchanged from the checkpoint config so the ONLY
# change vs Q0 is the rollout data. Stratified per-(repo,bucket) holdout keeps
# held-out demo AND rollout episodes for the round-1 gate.
#
#   PROBE=1 sbatch slurm_scripts/train_q_stackcups_q1.sbatch   # 30-step wiring probe
#   sbatch slurm_scripts/train_q_stackcups_q1.sbatch           # full 10k finetune
#
#SBATCH -J q_q1_stackcups
#SBATCH --account=gts-agarg35-ideasci23_dgx
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -p gpu-h100
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=64G
#SBATCH -t8:00:00
#SBATCH -q inferno
#SBATCH --requeue
#SBATCH --output=slurm_out/Report-%j.out
#SBATCH --mail-type=END,FAIL
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
mkdir -p "$WANDB_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

DEMO=VarunGiridhar3/KT_stack_cups_20260807_211827_right
BC_R0=VarunGiridhar3/rollout_stackcups_bc_r0_right
BCQ_R0=VarunGiridhar3/rollout_stackcups_bcq_r0_right
DATA_ROOT=/storage/scratch1/6/vgiridhar6/lerobot_cache
# Staged copy of Q0 (negft/010000) whose config.json carries the 3-repo
# bucket_overrides + action_stats_repo_id BAKED IN: dict-typed --policy.X CLI
# overrides are broken under --policy.path in this draccus version (they reach
# the choice-class decoder as a raw string), so they must come from the file.
BASE_CKPT="${BASE_CKPT:-/storage/scratch1/6/vgiridhar6/q1_warmstart/pretrained_model}"

PROBE="${PROBE:-0}"
STEPS="${STEPS:-10000}"
JOB="${JOB:-qf_stackcups_h32_q1}"
SAVE_FREQ=2500
LOG_FREQ=50
TEST_FREQ=200
TEST_N_BATCHES=4
WARMUP=500

if [ "$PROBE" = "1" ]; then
  STEPS=30; LOG_FREQ=1; SAVE_FREQ=30; TEST_FREQ=30; TEST_N_BATCHES=2; WARMUP=10
  JOB="${JOB}_probe"
fi

OUT_DIR="$REPO/outputs/train/$JOB"

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
  --policy.path="$BASE_CKPT" \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=3e-5 \
  --policy.optimizer_lr_backbone=1e-5 \
  --policy.lr_scheduler=cosine_decay_with_warmup \
  --policy.lr_warmup_steps="$WARMUP" \
  --policy.lr_decay_steps="$STEPS" \
  --policy.lr_decay_min=1e-6 \
  --dataset.repo_ids="[$DEMO, $BC_R0, $BCQ_R0]" \
  --dataset.root="$DATA_ROOT" \
  --dataset.video_backend=pyav \
  --batch_size=32 \
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
