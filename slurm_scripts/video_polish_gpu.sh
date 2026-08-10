#!/bin/bash
# GPU PACE Phoenix job for SAM2 wrist-camera segmentation (account gts-agarg35).
# CLUSTER MECHANICS ONLY — the workload command is passed as args.
#
#   mkdir -p slurm_out slurm_scripts
#   sbatch slurm_scripts/video_polish_gpu.sh <python script and args...>
#
# ---- per-job knobs ----
#SBATCH -J wrist_sam2
#SBATCH --account=gts-agarg35
#SBATCH -N1 --gres=gpu:RTX_6000:1
#SBATCH --mem-per-gpu=32G
#SBATCH -t2:00:00
#SBATCH -q embers
#SBATCH --output=slurm_out/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vgiridhar6@gatech.edu

set -euo pipefail

export TMPDIR="$HOME/wandb_tmp"
mkdir -p "$TMPDIR"
export HF_HOME=/storage/scratch1/6/vgiridhar6/hf
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LAMA_MODEL=/storage/project/r-agarg35-0/vgiridhar6/models/lama/big-lama.pt

PY=/storage/project/r-agarg35-0/vgiridhar6/envs/sam2env/bin/python
export PATH="/storage/home/hcoda1/6/vgiridhar6/.conda/envs/lerobot-qplanning/bin:$PATH"

echo "Running: $PY $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" "$@"
