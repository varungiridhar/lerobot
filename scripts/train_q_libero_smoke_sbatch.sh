#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH -q embers
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH -p gpu-h200
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# Sbatch wrapper for the single-GPU smoke test. Body is in train_q_libero_smoke.sh;
# this file only adds SBATCH headers + invokes it, so config stays in one place.
#
# Submit from the repo root (so SBATCH's relative log path resolves to
# <repo>/logs/) and ensure ./logs/ exists:  mkdir -p logs
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_q_libero_smoke.sh"
