#!/bin/bash
# Submit one held-out paper-style LIBERO-10 evaluation for a self-improvement Q checkpoint.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
EVAL_SCRIPT="$SCRIPT_DIR/eval_fastwam_q_libero.sh"

Q_CKPT=${Q_CKPT:?set Q_CKPT to the Q checkpoint under evaluation}
LABEL=${LABEL:?set LABEL to q0, q1, q2, or q3}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT for this held-out checkpoint evaluation}
HELDOUT_SEED=${HELDOUT_SEED:-100042}
MAX_CONCURRENT_TASKS=${MAX_CONCURRENT_TASKS:-10}
DEPENDENCY_JOB_ID=${DEPENDENCY_JOB_ID:-}
SI_QOS=${SI_QOS:-embers}

if [[ ! "$LABEL" =~ ^q[0-9]+$ ]]; then
    echo "LABEL must have the form q<N>; got '$LABEL'." >&2
    exit 2
fi
if [[ ! "$HELDOUT_SEED" =~ ^[0-9]+$ ]]; then
    echo "HELDOUT_SEED must be a non-negative integer; got '$HELDOUT_SEED'." >&2
    exit 2
fi
if [[ ! "$MAX_CONCURRENT_TASKS" =~ ^[1-9][0-9]*$ ]] || (( MAX_CONCURRENT_TASKS > 10 )); then
    echo "MAX_CONCURRENT_TASKS must be in [1, 10]." >&2
    exit 2
fi
if [[ -n "$DEPENDENCY_JOB_ID" && ! "$DEPENDENCY_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "DEPENDENCY_JOB_ID must be a numeric Slurm job ID; got '$DEPENDENCY_JOB_ID'." >&2
    exit 2
fi

# A dependency may produce the Q checkpoint after this array is submitted. The
# evaluation script performs the same checkpoint checks again when it starts.
required_paths=("$EVAL_SCRIPT")
if [[ -z "$DEPENDENCY_JOB_ID" ]]; then
    required_paths+=("$Q_CKPT/config.json" "$Q_CKPT/model.safetensors")
fi
for required_path in "${required_paths[@]}"; do
    if [[ ! -r "$required_path" ]]; then
        echo "Held-out eval input is not readable: $required_path" >&2
        exit 2
    fi
done

mkdir -p "$REPO_ROOT/slurm_out"

dependency_args=()
if [[ -n "$DEPENDENCY_JOB_ID" ]]; then
    dependency_args+=(--dependency="afterok:$DEPENDENCY_JOB_ID")
fi

job_id=$(sbatch \
    --parsable \
    --account=gts-agarg35 \
    --qos="$SI_QOS" \
    --partition=gpu-a100 \
    --gres=gpu:a100:1 \
    --constraint=A100-40GB \
    --time=02:00:00 \
    "${dependency_args[@]}" \
    --job-name="fwq_si_${LABEL}_s10" \
    --array="0-9%${MAX_CONCURRENT_TASKS}" \
    --export="ALL,REPO_ROOT=$REPO_ROOT,Q_CKPT=$Q_CKPT,DIFFUSION_STEPS=10,N_SAMPLES=64,N_EPISODES=50,SEED=$HELDOUT_SEED,EVAL_PROTOCOL=fixed0_520,EPISODE_LENGTH=520,SMOKE_TEST=0,OUTPUT_ROOT=$OUTPUT_ROOT" \
    "$EVAL_SCRIPT")

echo "$job_id"
