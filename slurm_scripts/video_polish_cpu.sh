#!/bin/bash
# CPU-only PACE Phoenix job for rollout-video processing (account gts-agarg35).
# CLUSTER MECHANICS ONLY — the workload command is passed as args.
#
#   mkdir -p slurm_out slurm_scripts
#   sbatch slurm_scripts/video_polish_cpu.sh <python script and args...>
#
# Jobs run in the submission dir — submit from the repo root.
#
# ---- per-job knobs ----
#SBATCH -J video_polish
#SBATCH --account=gts-agarg35
#SBATCH -N1 -n1 --cpus-per-task=8
#SBATCH -p cpu-small
#SBATCH --mem=32G
#SBATCH -t2:00:00
#SBATCH -q embers
#SBATCH --output=slurm_out/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vgiridhar6@gatech.edu

set -euo pipefail

export TMPDIR="$HOME/wandb_tmp"
mkdir -p "$TMPDIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENCV_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME=/storage/scratch1/6/vgiridhar6/hf

PY=/storage/home/hcoda1/6/vgiridhar6/.conda/envs/lerobot-qplanning/bin/python
export PATH="/storage/home/hcoda1/6/vgiridhar6/.conda/envs/lerobot-qplanning/bin:$PATH"

echo "Running: $PY $*"
"$PY" "$@"
