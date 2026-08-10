#!/bin/bash
# Stage-0 Q-probe battery (rebuttal forensics). Submit from the repo root:
#   sbatch slurm_scripts/q_probe_battery.sbatch
#
#SBATCH -J q_probe_battery
#SBATCH --account=gts-agarg35
#SBATCH -N1 --gres=gpu:L40S:1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=4
#SBATCH -t4:00:00
#SBATCH -q embers
#SBATCH --output=slurm_out/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vgiridhar6@gatech.edu

set -euo pipefail

export TMPDIR="$HOME/wandb_tmp"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"
export TRITON_CACHE_DIR="$HOME/.cache/triton"
export CUDA_CACHE_PATH="$HOME/.cache/nv"
export CUDA_CACHE_MAXSIZE=4294967296
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
# Writable caches: scratch HF_HOME for datasets locks; user hub cache has dinov2+t5.
# (The shared r-agarg35-0/shared cache is read-only — datasets FileLock fails there.)
export HF_HOME=/storage/scratch1/6/vgiridhar6/hf
export HF_HUB_CACHE=/storage/project/r-agarg35-0/vgiridhar6/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"

TRAIN_ROOT=/storage/scratch1/6/vgiridhar6/lerobot/outputs/train/2026-05-19
BCPLAY=$TRAIN_ROOT/23-54-24_qf_libero_ddp2_bsz48_bc_plus_play_h200/checkpoints
BCONLY=$TRAIN_ROOT/23-54-42_qf_libero_ddp2_bsz48_bc_h200/checkpoints
OUT_DIR=${OUT_DIR:-$HOME/scratch/qplanning_rebuttal/probe/stage0a}
# Always probe against the BC+play dataset config so the play-bucket probes
# (cross_bucket, the play test episodes) exist regardless of which ckpt is scored.
DATASET_CONFIG=${DATASET_CONFIG:-$BCPLAY/last/pretrained_model/train_config.json}

# CKPT_ARGS: caller-supplied "--ckpt label=path ..." string. When unset, default
# to the Stage-0 baseline trio. Always append the BC-only baseline for reference.
if [[ -z "${CKPT_ARGS:-}" ]]; then
    CKPT_ARGS="--ckpt bc12k=$BCONLY/last/pretrained_model --ckpt bcplay12k=$BCPLAY/012000/pretrained_model --ckpt bcplay25k=$BCPLAY/last/pretrained_model"
fi

~/.conda/envs/lerobot/bin/python scripts/q_probe_battery.py \
    $CKPT_ARGS \
    --dataset-config "$DATASET_CONFIG" \
    --out-dir "$OUT_DIR" \
    --episodes-per-bucket "${EPISODES_PER_BUCKET:-12}" \
    --stride "${STRIDE:-8}" \
    --max-frames-per-ep "${MAX_FRAMES_PER_EP:-40}" \
    --frames-per-fwd "${FRAMES_PER_FWD:-2}" \
    --seed 0
