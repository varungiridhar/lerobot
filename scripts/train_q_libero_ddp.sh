#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=32G
#SBATCH -q embers
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:RTX_6000:4
#SBATCH -o /storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/logs/%j.out
#SBATCH -e /storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/logs/%j.err

# Activate conda environment
source /storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export HF_DATASETS_CACHE=/storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/.cache/huggingface/datasets

cd /storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/lerobot

DATA_ROOT="/storage/home/hcoda1/7/igeorgiev3/shared/lerobot-data-2"

accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    --multi_gpu \
    $(which lerobot-train) \
    --job_name=qf_libero_ddp4_bsz8 \
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
    --policy.lr_scheduler=cosine_decay_with_warmup \
    --policy.lr_warmup_steps=5000 \
    --policy.lr_decay_steps=100000 \
    --policy.lr_decay_min=1e-6 \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --dataset.root="${DATA_ROOT}/HuggingFaceVLA/libero" \
    --batch_size=8 \
    --steps=100000 \
    --log_freq=50 \
    --save_freq=10000 \
    --eval_freq=0 \
    --num_workers=6 \
    --cudnn_deterministic=false \
    --wandb.enable=true \
    --wandb.project=awm
