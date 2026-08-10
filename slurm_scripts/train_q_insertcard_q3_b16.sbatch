#!/bin/bash
# Task-2 round 3 (FINAL): Q_2(5k, deployed) -> Q_3 on demo + all four rollout
# sets (100 autonomous eps). DEADLINE MODE 5k steps. Curve: 25 -> 40 -> 75 -> 80.
#SBATCH -J q_ic_q3
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

DEMO=VarunGiridhar3/KT_insert_card_into_wallet_20260808_merged
BCQ=VarunGiridhar3/rollout_insertcard_bcq_r0
BC=VarunGiridhar3/rollout_insertcard_bc_r0
R1=VarunGiridhar3/rollout_insertcard_bcq_q1_r1
R2=VarunGiridhar3/rollout_insertcard_bcq_q2_r2
DATA_ROOT=/storage/scratch1/6/vgiridhar6/lerobot_cache
BASE_CKPT="${BASE_CKPT:-/storage/scratch1/6/vgiridhar6/ic_q3_warmstart/pretrained_model}"

STEPS="${STEPS:-5000}"
JOB="${JOB:-qf_insertcard_h32_q3}"
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
  --policy.optimizer_lr=3e-5 \
  --policy.optimizer_lr_backbone=1e-5 \
  --policy.lr_scheduler=cosine_decay_with_warmup \
  --policy.lr_warmup_steps=250 \
  --policy.lr_decay_steps="$STEPS" \
  --policy.lr_decay_min=1e-6 \
  --dataset.repo_ids="[$DEMO, $BCQ, $BC, $R1, $R2]" \
  --dataset.root="$DATA_ROOT" \
  --dataset.video_backend=pyav \
  --batch_size="${BATCH:-32}" \
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
