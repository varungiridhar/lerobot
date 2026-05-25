#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=80G
#SBATCH -q embers
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH -p gpu-h100
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vgiridhar6@gatech.edu


# Multi-GPU (DDP) Q-function training on RoboTwin data. Mirrors
# scripts/train_q_libero_ddp_multi_bc.sh, adapted for the RoboTwin multicam
# dataset (3 separate cameras) produced by:
#     scripts/convert_robotwin_to_lerobot.py --mode multicam
#
# RoboTwin demos are all expert successes, so reward_mode=sparse with a single
# `q5` bucket (terminal_bonus 1.0, step_reward 0.0) gives every episode terminal
# a +1 reward — equivalent to all_success, but (unlike all_success) it supports
# the stratified train/test split that the Q-visualization eval needs.
#
# Submit from the worktree root so the SBATCH relative log path resolves there:
#     mkdir -p logs && sbatch scripts/train_q_robotwin_ddp.sh
# Override:  STEPS=10000 NUM_GPUS=1 BATCH_SIZE=8 sbatch scripts/train_q_robotwin_ddp.sh
#
# Notes:
#   * --dataset.video_backend=pyav — torchcodec cannot load FFmpeg in the
#     lerobot-robotwin env (libstdc++/libtbb ABI clash from the SAPIEN deps).
#   * --output_dir is on project storage — the ~5 GB Q-function checkpoints
#     overflow the quota-capped home filesystem.
#   * embers QOS is preemptible; for long runs see the *_resume.sh pattern.

WORKTREE=/storage/home/hcoda1/6/vgiridhar6/forks/lerobot-qfunction-robotwin
DATASET_ROOT=/storage/home/hcoda1/6/vgiridhar6/r-agarg35-0/robotwin/dataset/robotwin2.0_multicam
OUTPUT_DIR="${OUTPUT_DIR:-/storage/home/hcoda1/6/vgiridhar6/r-agarg35-0/robotwin/outputs/train/qf_robotwin_ddp}"
NUM_GPUS="${NUM_GPUS:-2}"
STEPS="${STEPS:-40000}"
BATCH_SIZE="${BATCH_SIZE:-12}"

module load anaconda3/2022.05.0.1
module load cuda/12.6.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-/storage/project/r-agarg35-0/vgiridhar6/.conda/envs/lerobot-robotwin}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/storage/home/hcoda1/6/vgiridhar6/r-agarg35-0/hf_cache
export TMPDIR=/storage/project/r-agarg35-0/vgiridhar6/robotwin/tmp
# Use this worktree's source without repointing the shared env's editable install.
export PYTHONPATH="$WORKTREE/src:${PYTHONPATH:-}"

# Keep wandb's run dir + artifact cache on scratch — home and the project
# filesystem are quota-capped.
export WANDB_DIR="$HOME/scratch/wandb"
export WANDB_CACHE_DIR="$HOME/scratch/wandb/cache"
export WANDB_ARTIFACT_DIR="$HOME/scratch/wandb/artifacts"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR"

# Under sbatch, ${BASH_SOURCE[0]} points at the staged /var/spool copy, so fall
# back to $SLURM_SUBMIT_DIR (the dir sbatch was invoked from) for relative paths.
cd "${SLURM_SUBMIT_DIR:-$WORKTREE}" || exit 1

# Unique torch-distributed rendezvous port from the SLURM job id, so two
# accelerate jobs on the same node don't collide on the default port 29500.
MASTER_PORT=$(( 20000 + ${SLURM_JOB_ID:-$RANDOM} % 10000 ))
echo "master port: ${MASTER_PORT}"

# Hang detector: py-spy-dumps every rank + dataloader worker if no training
# step is logged for HANG_TIMEOUT seconds, then tears the job down so a
# DDP/dataloader stall fails fast. Dumps land in logs/hang_dumps_<jobid>/.
source scripts/hang_watchdog.sh

run_with_hang_watchdog accelerate launch \
    --num_processes="${NUM_GPUS}" \
    --mixed_precision=bf16 \
    --multi_gpu \
    --main_process_port="${MASTER_PORT}" \
    "$(which lerobot-train)" \
    --job_name=qf_robotwin_ddp_multicam \
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
    --policy.step_reward=0.0 \
    --policy.terminal_bonuses='{q5: 1.0}' \
    --policy.bucket_overrides='{local/robotwin2.0_multicam: q5}' \
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
    --policy.lr_decay_steps="${STEPS}" \
    --policy.lr_decay_min=1e-6 \
    --dataset.repo_id=local/robotwin2.0_multicam \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.video_backend=pyav \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --log_freq=50 \
    --save_freq=5000 \
    --eval_freq=2000 \
    --test_split_ratio=0.1 \
    --test_freq=50 \
    --test_n_batches=1 \
    --num_workers=4 \
    --cudnn_deterministic=false \
    --output_dir="${OUTPUT_DIR}" \
    --wandb.enable=true \
    --wandb.project=awm \
    --wandb.disable_artifact=true
exit $?
