#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=64G
#SBATCH -q embers
#SBATCH -t 0:30:00
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

source /storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export HF_DATASETS_CACHE=/storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/.cache/huggingface/datasets
export ACCELERATE_NUM_PROCESSES=1

cd /storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/lerobot

python scripts/proto_q_vis.py
