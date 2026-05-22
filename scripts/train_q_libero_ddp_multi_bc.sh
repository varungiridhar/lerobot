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


# Multi-dataset Q-function DDP training: BC (HuggingFaceVLA/libero)
# + 4 LIBERO play splits. Mirrors scripts/train_q_libero_ddp.sh for
# environment + perf hyperparameters; only the dataset list and reward
# scheme differ.
#
# Submit from the repo root (so SBATCH's relative log path resolves to
# <repo>/logs/) and ensure ./logs/ exists:  mkdir -p logs
#
# Differences from train_q_libero_ddp.sh (BC-only):
#   * --dataset.repo_ids (list) replaces --dataset.repo_id (str). No
#     --dataset.root — each sub-dataset is fetched from the Hub into
#     HF_LEROBOT_HOME (under HF_HOME) on first use.
#   * --policy.reward_mode=sparse (not all_success) so per-bucket
#     terminal bonuses apply.
#   * --policy.terminal_bonuses gives BC (q5) +1.0 at the terminal
#     frame, play 0.0 (no success signal for play episodes).
#   * --policy.bucket_overrides explicitly assigns a bucket to EVERY
#     repo in --dataset.repo_ids (no implicit inference). Missing
#     entries raise at QValueLabelDataset construction time. Bucket
#     names must appear as keys in --policy.terminal_bonuses (and
#     --policy.quality_scalars for time_to_go).
#   * --dataset.video_backend=pyav — the play splits store frames as
#     MP4 videos, decoded at load time. pyav works in the stock
#     `lerobot` env. For faster decoding switch to torchcodec, which
#     requires a system FFmpeg install (libavutil.so.59 etc):
#         conda install -n lerobot -c conda-forge "ffmpeg=7.*"
#     then drop the --dataset.video_backend=pyav line.
#
# HF cache: if your $HOME is quota-capped, export HF_HOME to a roomier
# volume before submitting, e.g.:
#     export HF_HOME=/storage/project/.../.cache/huggingface

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-lerobot-q}"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

# Keep wandb's run dir + artifact cache on scratch. Home and the project
# filesystem are quota-capped; wandb otherwise writes run files next to the
# repo (home) and caches artifacts under ~/.cache (a symlink onto project),
# which leads to "Disk quota exceeded" failures.
export WANDB_DIR="$HOME/scratch/wandb"
export WANDB_CACHE_DIR="$HOME/scratch/wandb/cache"
export WANDB_ARTIFACT_DIR="$HOME/scratch/wandb/artifacts"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR"

# Under sbatch, SLURM stages the script to /var/spool/slurmd/jobN/slurm_script,
# so ${BASH_SOURCE[0]} points THERE rather than the original file path — a plain
# `dirname "${BASH_SOURCE[0]}"/..` would cd to /var/spool/slurmd and any relative
# output paths (outputs/, logs/) would land in that dir and vanish at job end.
# $SLURM_SUBMIT_DIR is set by SLURM to the directory from which sbatch was invoked
# (the repo root if you submitted from there). The BASH_SOURCE fallback covers
# the case where the script is run interactively via `bash scripts/...`.
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}" || exit 1

# Unique torch-distributed rendezvous port, derived from the SLURM job id, so
# two accelerate jobs packed onto the same DGX node don't collide on the
# default port 29500 (EADDRINUSE).
MASTER_PORT=$(( 20000 + ${SLURM_JOB_ID:-$RANDOM} % 10000 ))
echo "master port: ${MASTER_PORT}"

# Hang detector: wraps the training command, py-spy-dumps every rank +
# dataloader worker if no training step is logged for HANG_TIMEOUT seconds
# (default 900), then tears the job down so a DDP/dataloader stall fails fast
# (~20 min) instead of idling ~4 h until the NCCL watchdog. Dumps land in
# logs/hang_dumps_<jobid>/.
source scripts/hang_watchdog.sh

run_with_hang_watchdog accelerate launch \
    --num_processes=2 \
    --mixed_precision=bf16 \
    --multi_gpu \
    --main_process_port=${MASTER_PORT} \
    $(which lerobot-train) \
    --job_name=qf_libero_ddp2_bsz48_bc_h200 \
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
    --policy.terminal_bonuses='{q5: 1.0}' \
    --policy.bucket_overrides='{HuggingFaceVLA/libero: q5}' \
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
    --dataset.repo_ids='[HuggingFaceVLA/libero]' \
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
    --wandb.enable=true \
    --wandb.project=awm \
    --wandb.disable_artifact=true
exit $?
