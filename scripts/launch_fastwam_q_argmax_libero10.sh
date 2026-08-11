#!/bin/bash
# Submit the two paper-style LIBERO-10 exact-best-of-64 configurations.
# This creates two arrays (steps=10 and steps=20), one element per task.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
EVAL_SCRIPT="$SCRIPT_DIR/eval_fastwam_q_libero.sh"
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
EVAL_PROTOCOL=${EVAL_PROTOCOL:-distinct700}
case "$EVAL_PROTOCOL" in
    distinct700) EPISODE_LENGTH=700 ;;
    fixed0_520) EPISODE_LENGTH=520 ;;
    *)
        echo "EVAL_PROTOCOL must be distinct700 or fixed0_520; got '$EVAL_PROTOCOL'." >&2
        exit 2
        ;;
esac
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/outputs/eval/rebuttal_fastwam_q_bestof64_libero10_${EVAL_PROTOCOL}_$RUN_TAG}
MAX_CONCURRENT_TASKS=${MAX_CONCURRENT_TASKS:-10}
DRY_RUN=${DRY_RUN:-0}
GPU_PROFILE=${GPU_PROFILE:-l40s}

case "$GPU_PROFILE" in
    l40s)
        GPU_DESC="L40S"
        SBATCH_GPU_ARGS=(
            --account=gts-agarg35-ideas_l40s
            --partition=gpu-l40s
            --gres=gpu:l40s:1
        )
        ;;
    a100_40)
        GPU_DESC="A100 40GB"
        SBATCH_GPU_ARGS=(
            --account=gts-agarg35
            --partition=gpu-a100
            --gres=gpu:a100:1
            --constraint=A100-40GB
        )
        ;;
    *)
        echo "GPU_PROFILE must be l40s or a100_40; got '$GPU_PROFILE'." >&2
        exit 2
        ;;
esac

if [[ ! "$MAX_CONCURRENT_TASKS" =~ ^[1-9][0-9]*$ ]] || (( MAX_CONCURRENT_TASKS > 10 )); then
    echo "MAX_CONCURRENT_TASKS must be an integer in [1, 10]." >&2
    exit 2
fi
if [[ ! -x "$EVAL_SCRIPT" ]]; then
    echo "Eval script is not executable: $EVAL_SCRIPT" >&2
    exit 2
fi
if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 2
fi

echo "LIBERO-10 rebuttal matrix"
echo "  configurations: best-of-64 at 10 and 20 diffusion steps"
echo "  evaluation:     10 tasks x 50 episodes = 500 episodes/config"
echo "  protocol:       $EVAL_PROTOCOL (horizon=$EPISODE_LENGTH)"
echo "  resources:      one $GPU_DESC and eval batch_size=1 per array element"
echo "  output root:    $OUTPUT_ROOT"

if (( DRY_RUN == 0 )); then
    mkdir -p "$REPO_ROOT/slurm_out"
fi

for steps in 10 20; do
    case "$steps" in
        10) TIME_LIMIT=02:00:00 ;;
        20) TIME_LIMIT=02:30:00 ;;
    esac
    CMD=(
        sbatch
        --parsable
        "${SBATCH_GPU_ARGS[@]}"
        --time="$TIME_LIMIT"
        --job-name="fwq_b64_s${steps}"
        --array="0-9%${MAX_CONCURRENT_TASKS}"
        --export="ALL,REPO_ROOT=$REPO_ROOT,DIFFUSION_STEPS=$steps,N_SAMPLES=64,N_EPISODES=50,SEED=42,EVAL_PROTOCOL=$EVAL_PROTOCOL,EPISODE_LENGTH=$EPISODE_LENGTH,SMOKE_TEST=0,OUTPUT_ROOT=$OUTPUT_ROOT"
        "$EVAL_SCRIPT"
    )
    if (( DRY_RUN == 1 )); then
        printf 'DRY RUN: '
        printf '%q ' "${CMD[@]}"
        printf '\n'
    else
        job_id=$("${CMD[@]}")
        echo "submitted steps=$steps as job array $job_id"
    fi
done
