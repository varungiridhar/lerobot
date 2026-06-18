#!/bin/bash
#SBATCH -J q_align
#SBATCH --account=gts-agarg35-ideasci23_dgx
#SBATCH -N1 --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=4 --mem-per-gpu=64G
#SBATCH -q embers
#SBATCH -t 1:00:00
#SBATCH --output=slurm_out/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vgiridhar6@gatech.edu

set -uo pipefail
export PATH="$HOME/.conda/envs/${CONDA_ENV:-lerobot}/bin:$PATH"
export HF_HOME=/storage/scratch1/6/vgiridhar6/hf
export HF_HUB_CACHE=/storage/project/r-agarg35-0/vgiridhar6/hf_cache
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"

python scripts/debug_q_success_align.py \
    --q-ckpt "${Q_CKPT:?}" \
    --fastwam-ckpt "${FASTWAM_CKPT:-/storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint}" \
    --dataset-config "${DATASET_CONFIG:?}" \
    --n-frames "${N_FRAMES:-40}"
