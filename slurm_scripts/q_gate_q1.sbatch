#!/bin/bash
# Phase-3 round-1 gate for Q1: (1) G2 on held-out demo episodes — criteria (b)/(c)
# must not degrade vs Q0; (2) rollout ranking — real failures below successes,
# compared head-to-head against Q0 on the same episodes.
#
#   sbatch slurm_scripts/q_gate_q1.sbatch
#   CKPT=.../005000/pretrained_model sbatch slurm_scripts/q_gate_q1.sbatch
#
#SBATCH -J q_gate_q1
#SBATCH --account=gts-agarg35
#SBATCH -N1 --gres=gpu:L40S:1
#SBATCH -p gpu-l40s
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G
#SBATCH -t4:00:00
#SBATCH -q embers
#SBATCH --requeue
#SBATCH --output=slurm_out/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vgiridhar6@gatech.edu

set -euo pipefail
REPO=/storage/project/r-agarg35-0/vgiridhar6/lerobot
cd "$REPO"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot-q
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/storage/scratch1/6/vgiridhar6/hf
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

CKPT="${CKPT:-$REPO/outputs/train/qf_stackcups_h32_q1/checkpoints/010000/pretrained_model}"
REF="${REF:-$REPO/outputs/train/qf_stackcups_h32_negft/checkpoints/010000/pretrained_model}"

echo "=== G2 (demo heldout) on $CKPT ==="
python scripts/q_gate2_real.py --q_ckpt "$CKPT"

echo "=== rollout ranking: Q1 vs Q0 ==="
python scripts/q_gate_rollout_rank.py --q_ckpt "$CKPT" --ref_ckpt "$REF"
