#!/bin/bash
# G2 gate: Q ranking probes + Q-vs-time on the held-out real episodes.
#   sbatch --export=ALL,Q_CKPT=<pretrained_model dir> slurm_scripts/q_gate2_real.sbatch
#SBATCH -J q_gate2
#SBATCH --account=gts-agarg35
#SBATCH -N1 --gres=gpu:RTX_6000:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=32G
#SBATCH -t2:00:00
#SBATCH -q embers
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
# Must match the training run (see train_q_stackcups.sbatch) so the reconstructed
# holdout split is identical.
export PYTHONHASHSEED=0

: "${Q_CKPT:?set Q_CKPT=<pretrained_model dir>}"

python scripts/q_gate2_real.py \
  --q_ckpt "$Q_CKPT" \
  --dataset_root /storage/scratch1/6/vgiridhar6/lerobot_qplanning_real/VarunGiridhar3__KT_stack_cups_20260807_211827_right \
  --device cuda
