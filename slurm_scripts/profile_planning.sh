#!/bin/bash
# Q-planning latency microbenchmark (CoRL rebuttal) — cluster mechanics from
# ~/.claude/skills/slurm/header.sbatch; L40s on inferno per corl_reviews/rules.md.
#
#   sbatch slurm_scripts/profile_planning.sh python scripts/profile_planning.py --benchmark ...
#
# ---- per-job knobs ----
#SBATCH -J profile_planning
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH -N1 --gres=gpu:l40s:1
#SBATCH -p gpu-l40s
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G
#SBATCH -t4:00:00
#SBATCH -q inferno
#SBATCH --output=slurm_out/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vgiridhar6@gatech.edu

set -euo pipefail

# ---- cluster env: caches/tmp on home storage to cut restart downtime ----
export TMPDIR="$HOME/wandb_tmp"
export PYTHONWARNINGS="ignore::UserWarning:torchvision.io._video_deprecation_warning"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"
export TRITON_CACHE_DIR="$HOME/.cache/triton"
export CUDA_CACHE_PATH="$HOME/.cache/nv"
export CUDA_CACHE_MAXSIZE=4294967296          # 4 GB
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

# ---- repo env (mirrors scripts/eval_fastwam_q_*.sh) ----
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
# Shared cache holds Wan2.2-TI2V-5B + dinov2-large + t5-v1_1-base + yuanty/fastwam.
export HF_HOME=/storage/project/r-agarg35-0/shared/huggingface_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running: $*"
nvidia-smi -L

# ---- RUN line ----
"$@"
