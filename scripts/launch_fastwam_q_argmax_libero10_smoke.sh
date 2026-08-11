#!/bin/bash
# Submit one task-0 episode for each step setting before releasing the full arrays.

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
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/outputs/eval/smoke_fastwam_q_bestof64_libero10_${EVAL_PROTOCOL}_$RUN_TAG}
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

if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 2
fi
if (( DRY_RUN == 0 )); then
    mkdir -p "$REPO_ROOT/slurm_out"
fi

echo "FastWAM standard-Q best-of-64 GPU smoke (task 0, one episode/config)"
echo "  protocol: $EVAL_PROTOCOL (horizon=$EPISODE_LENGTH)"
echo "  resources: $GPU_DESC, eval batch_size=1"
echo "  output root: $OUTPUT_ROOT"

for steps in 10 20; do
    CMD=(
        sbatch
        --parsable
        "${SBATCH_GPU_ARGS[@]}"
        --time=01:00:00
        --job-name="fwq_smoke_s${steps}"
        --export="ALL,REPO_ROOT=$REPO_ROOT,TASK_ID=0,DIFFUSION_STEPS=$steps,N_SAMPLES=64,N_EPISODES=1,SEED=42,EVAL_PROTOCOL=$EVAL_PROTOCOL,EPISODE_LENGTH=$EPISODE_LENGTH,SMOKE_TEST=1,OUTPUT_ROOT=$OUTPUT_ROOT"
        "$EVAL_SCRIPT"
    )
    if (( DRY_RUN == 1 )); then
        printf 'DRY RUN: '
        printf '%q ' "${CMD[@]}"
        printf '\n'
    else
        job_id=$("${CMD[@]}")
        echo "submitted smoke steps=$steps as job $job_id"
    fi
done
