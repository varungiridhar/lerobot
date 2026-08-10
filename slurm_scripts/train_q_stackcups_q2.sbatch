#!/bin/bash
# Phase-3 round 2 (FINAL): Q-only finetune Q_1 -> Q_2 on ALL loop rollout data
# (demo 60.4% + bc_r0 + bcq_r0 + bcq_q1_r1; filtered-BC eval data EXCLUDED per
# user decision — pure loop). Recipe identical to round 1; only the data grows.
# 4-repo bucket_overrides + action stats pin BAKED into the q2_warmstart config
# (draccus dict CLI override broken under --policy.path).
#
#   PROBE=1 sbatch slurm_scripts/train_q_stackcups_q2.sbatch
#   sbatch slurm_scripts/train_q_stackcups_q2.sbatch
#SBATCH -J q_q2_stackcups
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
R1=VarunGiridhar3/rollout_stackcups_bcq_q1_r1_right
DATA_ROOT=/storage/scratch1/6/vgiridhar6/lerobot_cache
# Staged copy of Q0 (negft/010000) whose config.json carries the 3-repo
# bucket_overrides + action_stats_repo_id BAKED IN: dict-typed --policy.X CLI
# overrides are broken under --policy.path in this draccus version (they reach
# the choice-class decoder as a raw string), so they must come from the file.
BASE_CKPT="${BASE_CKPT:-/storage/scratch1/6/vgiridhar6/q2_warmstart/pretrained_model}"

PROBE="${PROBE:-0}"
STEPS="${STEPS:-10000}"
JOB="${JOB:-qf_stackcups_h32_q2}"
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
  --dataset.repo_ids="[$DEMO, $BC_R0, $BCQ_R0, $R1]" \
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
