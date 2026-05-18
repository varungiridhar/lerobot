#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=64G
#SBATCH -q embers
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:h200:4
#SBATCH -p gpu-h200
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

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
conda activate "${CONDA_ENV:-lerobot}"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    --multi_gpu \
    $(which lerobot-train) \
    --job_name=qf_libero_ddp4_bsz16_bc_plus_play \
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
    --policy.terminal_bonuses='{q5: 1.0, play: 0.0}' \
    --policy.bucket_overrides='{HuggingFaceVLA/libero: q5, VarunGiridhar3/libero40_libero_object_play: play, VarunGiridhar3/libero40_libero_10_play: play, VarunGiridhar3/libero40_libero_goal_play: play, VarunGiridhar3/libero40_libero_spatial_play: play}' \
    --policy.v_min=-0.01 \
    --policy.v_max=1.01 \
    --policy.hl_gauss_sigma=0.0075 \
    --policy.use_text_conditioning=true \
    --policy.text_encoder_model=google/t5-v1_1-base \
    --policy.language_key=task \
    --policy.h=32 \
    --policy.gamma=0.99 \
    --policy.target_tau=0.005 \
    --policy.optimizer_lr=1.4e-4 \
    --policy.optimizer_lr_backbone=4.2e-5 \
    --policy.optimizer_weight_decay=1e-4 \
    --policy.lr_scheduler=cosine_decay_with_warmup \
    --policy.lr_warmup_steps=5000 \
    --policy.lr_decay_steps=100000 \
    --policy.lr_decay_min=1e-6 \
    --dataset.repo_ids='[HuggingFaceVLA/libero,VarunGiridhar3/libero40_libero_object_play,VarunGiridhar3/libero40_libero_10_play,VarunGiridhar3/libero40_libero_goal_play,VarunGiridhar3/libero40_libero_spatial_play]' \
    --batch_size=16 \
    --steps=100000 \
    --log_freq=50 \
    --save_freq=1000 \
    --eval_freq=1000 \
    --num_workers=6 \
    --cudnn_deterministic=false \
    --wandb.enable=true \
    --wandb.project=awm
