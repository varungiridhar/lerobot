#!/bin/bash
# Q negatives-finetune: warm-start from the 30k base Q and strengthen action
# discrimination. Response to the G2 FAIL on criterion (a): the base Q ranks
# tube/hold negatives well (0.86-0.97) but is near chance on frame-matched
# swaps (0.48), reversed (0.64), shuffled (0.55) — it learned V(s), not Q(s,a).
#
# Changes vs the base run:
#   * warm start --policy.path=<30k pretrained_model> (weights only, fresh optim)
#   * NEW negatives: neg_use_matched_swap (nearest-fraction cross-episode chunk)
#     + neg_use_shift [8,16] ("right actions, wrong time" from the 2h window)
#   * margin weight 0.25 -> 0.5, delta 0.1 -> 0.15
#   * finetune LR 5e-5 (backbone 1.5e-5), 10k steps
# Same dataset/split/seed (PYTHONHASHSEED=0) so G2 stays comparable.
#
#   sbatch slurm_scripts/train_q_stackcups_negft.sbatch            # H100 default
#
#SBATCH -J q_negft
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

DATASET=VarunGiridhar3/KT_stack_cups_20260807_211827_right
DATA_ROOT=/storage/scratch1/6/vgiridhar6/lerobot_cache
BASE_CKPT="${BASE_CKPT:-$REPO/outputs/train/qf_stackcups_h32/checkpoints/030000/pretrained_model}"
STEPS="${STEPS:-10000}"
JOB="${JOB:-qf_stackcups_h32_negft}"
OUT_DIR="$REPO/outputs/train/$JOB"

if [ -d "$OUT_DIR" ] && [ ! -d "$OUT_DIR/checkpoints" ]; then
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
  --policy.neg_margin_weight=0.5 \
  --policy.neg_margin_delta=0.15 \
  --policy.neg_use_matched_swap=true \
  --policy.neg_use_shift=true \
  --policy.neg_shift_frames='[8,16]' \
  --policy.optimizer_lr=5e-5 \
  --policy.optimizer_lr_backbone=1.5e-5 \
  --policy.lr_scheduler=cosine_decay_with_warmup \
  --policy.lr_warmup_steps=500 \
  --policy.lr_decay_steps="$STEPS" \
  --policy.lr_decay_min=1e-6 \
  --dataset.repo_id="$DATASET" \
  --dataset.root="$DATA_ROOT" \
  --dataset.video_backend=pyav \
  --batch_size=32 \
  --steps="$STEPS" \
  --log_freq=50 \
  --save_freq=2500 \
  --eval_freq=0 \
  --test_split_ratio=0.1 \
  --test_freq=200 \
  --test_n_batches=4 \
  --num_workers=6 \
  --cudnn_deterministic=false \
  --seed=42 \
  --output_dir="$OUT_DIR" \
  --wandb.enable=true \
  --wandb.entity=vgiridhar6 \
  --wandb.project=awm \
  --wandb.disable_artifact=true
