#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=64G
#SBATCH -q inferno
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:h200:2
#SBATCH -p gpu-h200
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vgiridhar6@gatech.edu

# Q-function DDP training on H200 (2 GPUs, bsz=48).
#
# Two dataset modes controlled by USE_PLAY (default: false):
#   false — BC only: HuggingFaceVLA/libero (q5 bucket, terminal bonus 1.0)
#   true  — BC + play: adds 4 LIBERO play splits (play bucket, terminal bonus 0.0)
#
# Submit from repo root with ./logs/ present (mkdir -p logs):
#   sbatch scripts/train_q_libero_ddp.sh                        # BC only
#   sbatch --export=ALL,USE_PLAY=true scripts/train_q_libero_ddp.sh  # BC + play
#
# Resume from a checkpoint:
#   sbatch --export=ALL,RESUME_CKPT=/path/to/checkpoint scripts/train_q_libero_ddp.sh

USE_PLAY=${USE_PLAY:-false}
RESUME_CKPT=${RESUME_CKPT:-}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-lerobot-q}"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

export WANDB_DIR="$HOME/scratch/wandb"
export WANDB_CACHE_DIR="$HOME/scratch/wandb/cache"
export WANDB_ARTIFACT_DIR="$HOME/scratch/wandb/artifacts"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR"

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}" || exit 1

MASTER_PORT=$(( 20000 + ${SLURM_JOB_ID:-$RANDOM} % 10000 ))
echo "master port: ${MASTER_PORT}"

source scripts/hang_watchdog.sh

if [ "$USE_PLAY" = "true" ]; then
    JOB_NAME=qf_libero_ddp2_bsz48_bc_plus_play_h200
    REPO_IDS='[HuggingFaceVLA/libero,VarunGiridhar3/libero40_libero_object_play,VarunGiridhar3/libero40_libero_10_play,VarunGiridhar3/libero40_libero_goal_play,VarunGiridhar3/libero40_libero_spatial_play]'
    TERMINAL_BONUSES='{q5: 1.0, play: 0.0}'
    BUCKET_OVERRIDES='{HuggingFaceVLA/libero: q5, VarunGiridhar3/libero40_libero_object_play: play, VarunGiridhar3/libero40_libero_10_play: play, VarunGiridhar3/libero40_libero_goal_play: play, VarunGiridhar3/libero40_libero_spatial_play: play}'
else
    JOB_NAME=qf_libero_ddp2_bsz48_bc_h200
    REPO_IDS='[HuggingFaceVLA/libero]'
    TERMINAL_BONUSES='{q5: 1.0}'
    BUCKET_OVERRIDES='{HuggingFaceVLA/libero: q5}'
fi

run_with_hang_watchdog accelerate launch \
    --num_processes=2 \
    --mixed_precision=bf16 \
    --multi_gpu \
    --main_process_port=${MASTER_PORT} \
    $(which lerobot-train) \
    --job_name=${JOB_NAME} \
    --policy.type=q_function \
    --policy.push_to_hub=false \
    --policy.dino_model_name=facebook/dinov2-large \
    --policy.dim_model=1024 \
    --policy.n_heads=16 \
    --policy.dim_feedforward=4096 \
    --policy.n_decoder_layers=18 \
    --policy.image_resize_h=224 \
    --policy.image_resize_w=224 \
    --policy.reward_mode=sparse \
    --policy.step_reward=0.0 \
    --policy.terminal_bonuses="${TERMINAL_BONUSES}" \
    --policy.bucket_overrides="${BUCKET_OVERRIDES}" \
    --policy.v_min=-0.01 \
    --policy.v_max=1.01 \
    --policy.hl_gauss_sigma=0.0075 \
    --policy.use_text_conditioning=true \
    --policy.text_encoder_model=google/t5-v1_1-base \
    --policy.language_key=task \
    --policy.h=32 \
    --policy.gamma=0.99 \
    --policy.target_tau=0.005 \
    --policy.optimizer_lr=3e-4 \
    --policy.optimizer_lr_backbone=9e-5 \
    --policy.optimizer_weight_decay=1e-4 \
    --policy.lr_scheduler=cosine_decay_with_warmup \
    --policy.lr_warmup_steps=2000 \
    --policy.lr_decay_steps=40000 \
    --policy.lr_decay_min=1e-6 \
    --dataset.repo_ids="${REPO_IDS}" \
    --dataset.root="$HOME/scratch/hf_cache/lerobot" \
    --batch_size=48 \
    --steps=40000 \
    --log_freq=50 \
    --save_freq=1000 \
    --eval_freq=2000 \
    --test_split_ratio=0.1 \
    --test_freq=50 \
    --test_n_batches=1 \
    --num_workers=6 \
    --cudnn_deterministic=false \
    ${RESUME_CKPT:+--resume_ckpt="${RESUME_CKPT}"} \
    --wandb.enable=true \
    --wandb.project=awm \
    --wandb.disable_artifact=true
exit $?
