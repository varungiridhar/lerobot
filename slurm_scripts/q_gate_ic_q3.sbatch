#!/bin/bash
# Task-2 round-1 gate: G2 on the merged insert-card demo + rollout ranking
# (both r0 sets), reference = deployed Q_0 (25k).
#SBATCH -J q_gate_ic_q3
#SBATCH --account=gts-agarg35
#SBATCH -N1 --gres=gpu:RTX_6000:1
#SBATCH -p gpu-rtx6000
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
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
CKPT=$REPO/outputs/train/qf_insertcard_h32_q3/checkpoints/002500/pretrained_model
REF=$REPO/outputs/train/qf_insertcard_h32_q2/checkpoints/005000/pretrained_model
DEMO_ROOT=/storage/scratch1/6/vgiridhar6/lerobot_cache/VarunGiridhar3/KT_insert_card_into_wallet_20260808_merged
echo "=== G2 (demo heldout) ==="
python scripts/q_gate2_real.py --q_ckpt "$CKPT" \
  --repo_id VarunGiridhar3/KT_insert_card_into_wallet_20260808_merged --dataset_root "$DEMO_ROOT"
echo "=== rollout ranking Q1 vs Q0(25k) ==="
python scripts/q_gate_rollout_rank.py --q_ckpt "$CKPT" --ref_ckpt "$REF" \
  --repos VarunGiridhar3/rollout_insertcard_bcq_r0 VarunGiridhar3/rollout_insertcard_bc_r0 VarunGiridhar3/rollout_insertcard_bcq_q1_r1 VarunGiridhar3/rollout_insertcard_bcq_q2_r2
