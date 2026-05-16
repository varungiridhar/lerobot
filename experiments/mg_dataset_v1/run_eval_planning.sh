#!/bin/bash
# Roll out ACT-Simple with Q-scored MPPI/CEM planning and record per-chunk traces.
#
# Usage:
#   sbatch compute_rtx6000_mimicgen.sh \
#       bash experiments/mg_dataset_v1/run_eval_planning.sh <task_slug> [n_rollouts] [mode] [planner]
# where:
#   task_slug ∈ {square, threading, coffee}
#   mode      ∈ {prod, local, debug}
#   planner   ∈ {mppi, cem, baseline}    (default mppi; baseline = no planning A/B)
set -euxo pipefail

TASK_SLUG="${1:?usage: $0 <square|threading|coffee> [n_rollouts] [mode] [planner]}"
N_ROLLOUTS="${2:-10}"
MODE="${3:-prod}"
PLANNER="${4:-mppi}"
Q_VARIANT="${Q_VARIANT:-_h10}"  # env var override; default to retrained h=10 Q.

case "$TASK_SLUG" in
    square|threading|coffee) ;;
    *) echo "unknown task: $TASK_SLUG"; exit 1 ;;
esac
case "$MODE" in
    prod|local|debug) ;;
    *) echo "unknown mode: $MODE (prod|local|debug)"; exit 1 ;;
esac
case "$PLANNER" in
    mppi|cem|argmax|baseline) ;;
    *) echo "unknown planner: $PLANNER (mppi|cem|argmax|baseline)"; exit 1 ;;
esac

LEROBOT_ROOT="/storage/home/hcoda1/6/vgiridhar6/forks/lerobot"
BC_CKPT="${LEROBOT_ROOT}/outputs/train/mimicgen_${TASK_SLUG}_d0_act_simple/checkpoints/100000/pretrained_model"
Q_CKPT="${LEROBOT_ROOT}/outputs/train/mimicgen_${TASK_SLUG}_d0_q_function${Q_VARIANT}/checkpoints/last/pretrained_model"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${LEROBOT_ROOT}/outputs/eval/mimicgen_${TASK_SLUG}_d0_q_function${Q_VARIANT}/planning_${PLANNER}_${RUN_TAG}"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::UserWarning:torchvision.io._video_deprecation_warning"
# cuBLAS deterministic workspace — required when --deterministic is passed. Setting
# it here too (vs only inside Python) is safer: matmul kernels may be touched by
# import-time code, which would error after Python flips on use_deterministic_algorithms.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
PYBIN="/storage/home/hcoda1/6/vgiridhar6/.conda/envs/lerobot-mimicgen/bin/python"
CONDA_LIB="/storage/project/r-agarg35-0/vgiridhar6/.conda/envs/lerobot-mimicgen/lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${CONDA_LIB}/python3.10/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

cd "${LEROBOT_ROOT}"

EXTRA_FLAGS=()
case "$MODE" in
    prod)  EXTRA_FLAGS+=( --wandb_enable ) ;;
    local) ;;
    debug) EXTRA_FLAGS+=( --debug ) ;;
esac

# Optional bit-exact reproducibility (slower).
[[ "${DETERMINISTIC:-0}" == "1" ]] && EXTRA_FLAGS+=( --deterministic )
# Probe-debug: single rollout with per-step action+state dump.
[[ "${PROBE_DEBUG:-0}" == "1" ]] && EXTRA_FLAGS+=( --probe_debug )
# Suppress env.reset(seed=...) — for determinism probing.
[[ "${NO_ENV_SEED:-0}" == "1" ]] && EXTRA_FLAGS+=( --no_env_seed )

if [[ "$PLANNER" == "baseline" ]]; then
    EXTRA_FLAGS+=( --baseline )
else
    EXTRA_FLAGS+=( --planner_type "$PLANNER" )
    [[ -n "${NOISE_STD:-}" ]]   && EXTRA_FLAGS+=( --noise_std "${NOISE_STD}" )
    [[ -n "${N_SAMPLES:-}" ]]   && EXTRA_FLAGS+=( --n_samples "${N_SAMPLES}" )
    [[ -n "${TEMPERATURE:-}" ]] && EXTRA_FLAGS+=( --temperature "${TEMPERATURE}" )
    [[ -n "${N_ELITES:-}" ]]    && EXTRA_FLAGS+=( --n_elites "${N_ELITES}" )
    [[ -n "${N_ITERS:-}" ]]     && EXTRA_FLAGS+=( --n_iters "${N_ITERS}" )
    [[ -n "${PLANNER_SEED:-}" ]] && EXTRA_FLAGS+=( --planner_seed "${PLANNER_SEED}" )
    [[ -n "${CLIP_TO:-}" ]]      && EXTRA_FLAGS+=( --clip_to "${CLIP_TO}" )
    [[ -n "${SMOOTH_SIGMA_T:-}" ]] && EXTRA_FLAGS+=( --noise_smooth_sigma_t "${SMOOTH_SIGMA_T}" )
    # Encode the noise into the output dir tag so sweeps don't overwrite each other.
    OUTPUT_DIR="${OUTPUT_DIR}_n${N_SAMPLES:-64}_std${NOISE_STD:-0.3}"
fi

"${PYBIN}" -u experiments/mg_dataset_v1/eval_planning.py \
    --task "${TASK_SLUG}" \
    --bc_checkpoint "${BC_CKPT}" \
    --q_checkpoint "${Q_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_rollouts "${N_ROLLOUTS}" \
    "${EXTRA_FLAGS[@]}"
