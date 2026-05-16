#!/bin/bash
# Roll out BC and visualize Q-value estimates side-by-side per chunk-boundary step.
# Usage:
#   sbatch compute_rtx6000_mimicgen.sh \
#       bash experiments/mg_dataset_v1/run_visualize_q.sh <task_slug> [n_rollouts] [mode] [q_variant]
# where <task_slug> ∈ {square, threading, coffee}
#       mode ∈ {prod, local, debug}
#       q_variant suffix on the Q output dir (e.g. "_h10", or "" for the h=20 baseline).
set -euxo pipefail

TASK_SLUG="${1:?usage: $0 <square|threading|coffee> [n_rollouts] [mode] [q_variant]}"
N_ROLLOUTS="${2:-10}"
MODE="${3:-prod}"  # "prod" | "local" | "debug"
Q_VARIANT="${4:-_h10}"  # default to the h=10 Q now that it's trained; pass "" to use h=20.
case "$TASK_SLUG" in
    square|threading|coffee) ;;
    *) echo "unknown task: $TASK_SLUG"; exit 1 ;;
esac
case "$MODE" in
    prod|local|debug) ;;
    *) echo "unknown mode: $MODE (prod|local|debug)"; exit 1 ;;
esac

LEROBOT_ROOT="/storage/home/hcoda1/6/vgiridhar6/forks/lerobot"
BC_CKPT="${LEROBOT_ROOT}/outputs/train/mimicgen_${TASK_SLUG}_d0_act_simple/checkpoints/100000/pretrained_model"
Q_CKPT="${LEROBOT_ROOT}/outputs/train/mimicgen_${TASK_SLUG}_d0_q_function${Q_VARIANT}/checkpoints/last/pretrained_model"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${LEROBOT_ROOT}/outputs/eval/mimicgen_${TASK_SLUG}_d0_q_function${Q_VARIANT}/${RUN_TAG}"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::UserWarning:torchvision.io._video_deprecation_warning"
PYBIN="/storage/home/hcoda1/6/vgiridhar6/.conda/envs/lerobot-mimicgen/bin/python"
CONDA_LIB="/storage/project/r-agarg35-0/vgiridhar6/.conda/envs/lerobot-mimicgen/lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${CONDA_LIB}/python3.10/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

cd "${LEROBOT_ROOT}"

case "$MODE" in
    prod)  EXTRA_FLAGS=( --wandb_enable ) ;;  # local mp4+json + wandb video upload
    local) EXTRA_FLAGS=( ) ;;                  # local mp4+json only, no wandb
    debug) EXTRA_FLAGS=( --debug ) ;;          # print summary only, no files, no wandb
esac

"${PYBIN}" -u experiments/mg_dataset_v1/visualize_q_rollout.py \
    --task "${TASK_SLUG}" \
    --bc_checkpoint "${BC_CKPT}" \
    --q_checkpoint "${Q_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_rollouts "${N_ROLLOUTS}" \
    "${EXTRA_FLAGS[@]}"
