#!/bin/bash
# Launch the controlled self-improvement × negatives matrix.
#
# All arms start from the SAME BC-only Q checkpoint, use the SAME seed schedule
# (so per-iteration collect/pc_success is apples-to-apples across arms), and
# differ ONLY in the negative-Q paradigm applied during the in-loop Q finetune:
#
#   L0  baseline loop          neg_margin_weight=0   (Ignat's loop as-is)
#   L1  loop + tube negatives   tube sigmas only      (user-preferred paradigm)
#   L2  loop + tube+swap+trev    full action-discriminative negatives
#
# Each arm is one job running N_ITERATIONS iterations in a single process
# (FastWAM offloaded to CPU during finetune so it fits on a 44GB L40s).
# Job names are uniquely prefixed siNEG-<ARM> (other agents share this username).
#
# Usage:  bash scripts/launch_si_neg_matrix.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SB=slurm_scripts/si_neg_loop.sbatch
COMMON="N_ITERATIONS=6,N_EPISODES=30,FINETUNE_STEPS=200,BATCH_SIZE=16,N_SAMPLES=16,N_ELITES=16,DIFFUSION_STEPS=3,EVAL_N_EPISODES=0,SEED=42,WANDB_PROJECT=awm"
ROOT=$HOME/scratch/qplanning_rebuttal/si

submit () {  # arm  extra_exports
  local arm="$1"; local extra="$2"
  sbatch -J "siNEG-${arm}" \
    --export="ALL,ARM=${arm},${COMMON},OUTPUT_DIR=${ROOT}/${arm}_libero_10,WANDB_RUN_NAME=siNEG_${arm}_libero_10,${extra}" \
    "$SB"
}

echo "== L0 baseline (no negatives) =="
submit L0 "NEG_MARGIN_WEIGHT=0.0"

# NB: do NOT pass NEG_TUBE_SIGMAS via --export — its value "1.0,2.0" contains a
# comma, which sbatch --export treats as a variable separator. The sbatch + the
# python arg both default to "1.0,2.0", which is exactly what we want.
echo "== L1 tube-only negatives =="
submit L1 "NEG_MARGIN_WEIGHT=0.25,NEG_USE_SWAP=0,NEG_USE_TEMPORAL=0"

echo "== L2 full negatives (tube+swap+trev) =="
submit L2 "NEG_MARGIN_WEIGHT=0.25,NEG_USE_SWAP=1,NEG_USE_TEMPORAL=1"

echo
echo "Submitted. Track with:  squeue -u $USER -o '%.10i %.16j %.10T %.8M %.20R' | grep siNEG"
