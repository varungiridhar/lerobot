#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-gpu=128G
#SBATCH -q embers
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -o /storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/logs/%j.out
#SBATCH -e /storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/logs/%j.err

# Activate conda environment
source /storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
# Arrow cache: redirect away from the 20GB-capped home dir
export HF_DATASETS_CACHE=/storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/.cache/huggingface/datasets
# Override accelerate default config (which has num_processes=4, MULTI_GPU) for single-GPU jobs
export ACCELERATE_NUM_PROCESSES=1

# Run training with accelerate for multi-GPU
cd /storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/lerobot

DATA_ROOT="/storage/home/hcoda1/7/igeorgiev3/shared/lerobot-data-2"

lerobot-train \
    --job_name=qf_libero_bsz8_rtx6k_try_viz \
    --policy.type=q_function \
    --policy.push_to_hub=false \
    --policy.dino_model_name=facebook/dinov2-large \
    --policy.dim_model=1024 \
    --policy.n_heads=16 \
    --policy.dim_feedforward=4096 \
    --policy.n_decoder_layers=18 \
    --policy.image_resize_h=224 \
    --policy.image_resize_w=224 \
    --policy.reward_mode=all_success \
    --policy.terminal_bonus_uniform=1.0 \
    --policy.step_reward=0.0 \
    --policy.v_min=-0.01 \
    --policy.v_max=1.01 \
    --policy.hl_gauss_sigma=0.0075 \
    --policy.use_text_conditioning=true \
    --policy.text_encoder_model=google/t5-v1_1-base \
    --policy.language_key=task \
    --policy.h=32 \
    --policy.gamma=0.99 \
    --policy.target_tau=0.005 \
    --policy.optimizer_lr=1e-4 \
    --policy.optimizer_lr_backbone=3e-5 \
    --policy.optimizer_weight_decay=1e-4 \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --dataset.root="${DATA_ROOT}/HuggingFaceVLA/libero" \
    --batch_size=8 \
    --steps=500 \
    --log_freq=10 \
    --save_freq=500 \
    --eval_freq=100 \
    --num_workers=4 \
    --cudnn_deterministic=false \
    --wandb.enable=true \
    --wandb.project=awm