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
#
# Under sbatch, SLURM stages this script to /var/spool/slurmd/jobN/slurm_script
# (without the sibling train_q_libero_smoke.sh). Resolve the inner script's path
# via $SLURM_SUBMIT_DIR — the dir from which sbatch was invoked — with a
# BASH_SOURCE fallback for the case where this wrapper is run interactively.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec bash "${REPO_ROOT}/scripts/train_q_libero_smoke.sh"
