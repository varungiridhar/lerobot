#!/bin/bash
# Traditional value-based DAWR for FastWAM. No Q function and no planning.
#
# Submit the requested one-seed baseline:
#   sbatch scripts/run_dawr_self_improvement.sh
#SBATCH -J dawr_fastwam
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH -p gpu-h200
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=192G
#SBATCH -q embers
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:H200:1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

set -Eeuo pipefail

LOCAL_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$LOCAL_SCRIPT_DIR/.." && pwd)}}
SCRIPT_DIR="$REPO_ROOT/scripts"
PYTHON_BIN=${PYTHON_BIN:-/storage/project/r-agarg35-0/akhandelwal79/conda-envs/lerobot-libero-new/bin/python}
LIBERO_ROOT=${LIBERO_ROOT:-/storage/project/r-agarg35-0/akhandelwal79/LIBERO}

FASTWAM_CKPT=${FASTWAM_CKPT:-/storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint}
VALUE_CKPT=${VALUE_CKPT:-}
VALUE_ENCODER_MODEL=${VALUE_ENCODER_MODEL:-facebook/dinov2-large}
WAN_WEIGHTS=${WAN_WEIGHTS:-/storage/project/r-agarg35-0/shared/awm/fastwam_wan22_weights}
MODEL_CACHE=${MODEL_CACHE:-/storage/project/r-agarg35-0/vgiridhar6/hf_cache}

TASK=${TASK:-libero_10}
EPISODE_LENGTH=${EPISODE_LENGTH:-520}
ITERATION=${ITERATION:-0}
MAX_ITERATIONS=${MAX_ITERATIONS:-5}
N_EPISODES=${N_EPISODES:-100}
BUFFER_SIZE=${BUFFER_SIZE:-3000}
COLLECTION_TASK_ID=${COLLECTION_TASK_ID:-}
TRACE_ROLLOUTS=${TRACE_ROLLOUTS:-0}
GAMMA=${GAMMA:-0.99}
TD_LAMBDA=${TD_LAMBDA:-0.95}
BETA=${BETA:-10.0}
MAX_ADV_WEIGHT=${MAX_ADV_WEIGHT:-100.0}
MIN_ADVANTAGE_STD=${MIN_ADVANTAGE_STD:-0.001}

VALUE_STEPS=${VALUE_STEPS:-500}
VALUE_LR=${VALUE_LR:-1e-3}
VALUE_WEIGHT_DECAY=${VALUE_WEIGHT_DECAY:-1e-4}
VALUE_BATCH_SIZE=${VALUE_BATCH_SIZE:-256}
VALUE_ENCODE_BATCH_SIZE=${VALUE_ENCODE_BATCH_SIZE:-16}
VALUE_GRAD_CLIP_NORM=${VALUE_GRAD_CLIP_NORM:-10.0}
CRITIC_WARMUP_ITERATIONS=${CRITIC_WARMUP_ITERATIONS:-2}

ACTOR_STEPS=${ACTOR_STEPS:-200}
ACTOR_LR=${ACTOR_LR:-1e-5}
ACTOR_BATCH_SIZE=${ACTOR_BATCH_SIZE:-1}
ACTOR_GRAD_CLIP_NORM=${ACTOR_GRAD_CLIP_NORM:-1.0}
NUM_WORKERS=${NUM_WORKERS:-4}

EVAL_N_EPISODES=${EVAL_N_EPISODES:-20}
SEED=${SEED:-42}
WANDB_PROJECT=${WANDB_PROJECT:-awm}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
PARTITION=${PARTITION:-gpu-h200}
QOS=${QOS:-embers}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_DIR=${OUTPUT_DIR:-outputs/self_improvement_dawr/${TIMESTAMP}_traditional_dawr_${TASK}_seed${SEED}}
HF_HOME=${HF_HOME:-/storage/project/r-agarg35-0/akhandelwal79/hf}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}

if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
fi

for required_path in \
    "$PYTHON_BIN" \
    "$LIBERO_ROOT/libero" \
    "$FASTWAM_CKPT/config.json" \
    "$FASTWAM_CKPT/model.safetensors" \
    "$FASTWAM_CKPT/policy_preprocessor.json" \
    "$HF_DATASETS_CACHE" \
    "$WAN_WEIGHTS/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors" \
    "$WAN_WEIGHTS/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors" \
    "$WAN_WEIGHTS/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json" \
    "$MODEL_CACHE/models--facebook--dinov2-large/snapshots"; do
    if [[ ! -r "$required_path" ]]; then
        echo "Required launch input is not readable: $required_path" >&2
        exit 2
    fi
done
if [[ -n "$VALUE_CKPT" && ! -r "$VALUE_CKPT" ]]; then
    echo "VALUE_CKPT is not readable: $VALUE_CKPT" >&2
    exit 2
fi
if [[ "$ITERATION" -gt 0 && -z "$VALUE_CKPT" ]]; then
    echo "Resumed DAWR iterations require VALUE_CKPT from the preceding iteration." >&2
    exit 2
fi

export PYTHONPATH="$REPO_ROOT/src:$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME
export HF_DATASETS_CACHE
export HF_HUB_CACHE="$MODEL_CACHE"
export TRANSFORMERS_CACHE="$MODEL_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_MODEL_BASE_PATH="$WAN_WEIGHTS"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "$TRACE_ROLLOUTS" == 1 ]]; then
    export PYTHONFAULTHANDLER=1
    export TORCH_SHOW_CPP_STACKTRACES=1
    export CUDA_LAUNCH_BLOCKING=1
    export MALLOC_CHECK_=3
fi
export WANDB_DIR=${WANDB_DIR:-/storage/project/r-agarg35-0/akhandelwal79/wandb/dawr}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/storage/project/r-agarg35-0/akhandelwal79/wandb-cache/dawr}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$HF_HOME/matplotlib}
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$MPLCONFIGDIR" "$REPO_ROOT/logs" "$OUTPUT_DIR"

cd "$REPO_ROOT"

"$PYTHON_BIN" - "$FASTWAM_CKPT" "$VALUE_ENCODER_MODEL" <<'PY'
import json
import sys
from pathlib import Path

import libero.libero  # noqa: F401
import torch
from transformers import AutoConfig

from lerobot.policies.fastwam.dawr_dataset import exponential_advantage_weights
from lerobot.policies.fastwam.dawr_value import DAWRValueCritic, td_lambda_returns

fastwam_dir = Path(sys.argv[1])
encoder_model = sys.argv[2]
fastwam_cfg = json.loads((fastwam_dir / "config.json").read_text())
if fastwam_cfg.get("type") != "fastwam" or fastwam_cfg.get("chunk_size") != 32:
    raise RuntimeError("Traditional DAWR requires the FastWAM checkpoint with chunk_size=32")
AutoConfig.from_pretrained(encoder_model, local_files_only=True)

# Exercise value targets and batch-size-one-safe actor weighting without loading
# either large model. There is intentionally no Q checkpoint in this contract.
advantage, returns = td_lambda_returns(
    torch.tensor([0.0, 1.0]),
    torch.tensor([False, True]),
    torch.tensor([0, 0]),
    torch.zeros(2),
)
if not torch.allclose(returns, torch.tensor([0.9405, 1.0]), atol=1e-4):
    raise RuntimeError(f"Unexpected TD(lambda) preflight target: {returns}")
weights, _ = exponential_advantage_weights(advantage)
if torch.allclose(weights, torch.ones_like(weights)):
    raise RuntimeError("DAWR weighting unexpectedly collapsed to unit weights")
DAWRValueCritic(vision_dim=8, n_cameras=2, state_dim=8, n_tasks=10)
print("Traditional FastWAM DAWR preflight OK: no Q function, no planner")
PY

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
    echo "Preflight-only validation passed; no rollout or training was started."
    exit 0
fi

echo "=== Traditional FastWAM DAWR ==="
echo "  Q FUNCTION:       none"
echo "  PLANNING:         disabled"
echo "  ITERATION:        $ITERATION / $MAX_ITERATIONS"
echo "  FASTWAM_CKPT:     $FASTWAM_CKPT"
echo "  VALUE_CKPT:       ${VALUE_CKPT:-fresh random value head}"
echo "  EPISODE_LENGTH:   $EPISODE_LENGTH"
echo "  COLLECTION:       $N_EPISODES direct-actor episodes"
echo "  TASK FILTER:      ${COLLECTION_TASK_ID:-all tasks}"
echo "  ROLLOUT TRACE:    $TRACE_ROLLOUTS"
echo "  TD(lambda):       gamma=$GAMMA lambda=$TD_LAMBDA"
echo "  DAWR:             beta=$BETA max_weight=$MAX_ADV_WEIGHT buffer=$BUFFER_SIZE"
echo "  VALUE:            steps=$VALUE_STEPS lr=$VALUE_LR batch=$VALUE_BATCH_SIZE"
echo "  ACTOR:            steps=$ACTOR_STEPS lr=$ACTOR_LR batch=$ACTOR_BATCH_SIZE"
echo "  SCHEDULER:        partition=$PARTITION qos=$QOS"
echo "  OUTPUT_DIR:       $OUTPUT_DIR"
echo "================================="

ARGS=(
    scripts/self_improvement_dawr_loop.py
    --fastwam_ckpt "$FASTWAM_CKPT"
    --task "$TASK"
    --episode_length "$EPISODE_LENGTH"
    --n_iterations 1
    --start_iteration "$ITERATION"
    --n_episodes "$N_EPISODES"
    --buffer_size "$BUFFER_SIZE"
    --gamma "$GAMMA"
    --td_lambda "$TD_LAMBDA"
    --beta "$BETA"
    --max_adv_weight "$MAX_ADV_WEIGHT"
    --min_advantage_std "$MIN_ADVANTAGE_STD"
    --value_encoder_model "$VALUE_ENCODER_MODEL"
    --value_steps "$VALUE_STEPS"
    --value_lr "$VALUE_LR"
    --value_weight_decay "$VALUE_WEIGHT_DECAY"
    --value_batch_size "$VALUE_BATCH_SIZE"
    --value_encode_batch_size "$VALUE_ENCODE_BATCH_SIZE"
    --value_grad_clip_norm "$VALUE_GRAD_CLIP_NORM"
    --critic_warmup_iterations "$CRITIC_WARMUP_ITERATIONS"
    --actor_steps "$ACTOR_STEPS"
    --actor_lr "$ACTOR_LR"
    --actor_batch_size "$ACTOR_BATCH_SIZE"
    --actor_grad_clip_norm "$ACTOR_GRAD_CLIP_NORM"
    --num_workers "$NUM_WORKERS"
    --eval_n_episodes "$EVAL_N_EPISODES"
    --output_dir "$OUTPUT_DIR"
    --seed "$SEED"
    --wandb_project "$WANDB_PROJECT"
)
if [[ -n "$VALUE_CKPT" ]]; then
    ARGS+=(--value_ckpt "$VALUE_CKPT")
fi
if [[ -n "$WANDB_ENTITY" ]]; then
    ARGS+=(--wandb_entity "$WANDB_ENTITY")
fi
if [[ -n "$WANDB_RUN_NAME" ]]; then
    ARGS+=(--wandb_run_name "$WANDB_RUN_NAME")
fi
if [[ -n "$COLLECTION_TASK_ID" ]]; then
    ARGS+=(--collection_task_id "$COLLECTION_TASK_ID")
fi
if [[ "$TRACE_ROLLOUTS" == 1 ]]; then
    ARGS+=(--trace_rollouts)
fi

set +e
"$PYTHON_BIN" "${ARGS[@]}" 2>&1 | tee -a "$OUTPUT_DIR/log.txt"
EXIT_CODE=${PIPESTATUS[0]}
set -e

NEXT_ITERATION=$((ITERATION + 1))
if [[ "$EXIT_CODE" -eq 0 && "$NEXT_ITERATION" -lt "$MAX_ITERATIONS" ]]; then
    ITER_NAME=$(printf '%03d' "$ITERATION")
    PRODUCED_FASTWAM="$OUTPUT_DIR/iter_${ITER_NAME}/fastwam_checkpoint"
    NEXT_VALUE_CKPT="$OUTPUT_DIR/iter_${ITER_NAME}/value_checkpoint.pt"
    if [[ -d "$PRODUCED_FASTWAM" ]]; then
        NEXT_FASTWAM_CKPT="$PRODUCED_FASTWAM"
    else
        # The actor is intentionally unchanged during critic warmup.
        NEXT_FASTWAM_CKPT="$FASTWAM_CKPT"
    fi
    if [[ ! -r "$NEXT_VALUE_CKPT" ]]; then
        echo "Iteration succeeded but did not produce a value checkpoint." >&2
        exit 3
    fi
    echo "Chaining traditional DAWR iteration $NEXT_ITERATION"
    sbatch \
        -p "$PARTITION" \
        --qos="$QOS" \
        --dependency="afterok:$SLURM_JOB_ID" \
        --export=ALL,ITERATION="$NEXT_ITERATION",FASTWAM_CKPT="$NEXT_FASTWAM_CKPT",VALUE_CKPT="$NEXT_VALUE_CKPT",OUTPUT_DIR="$OUTPUT_DIR",PARTITION="$PARTITION",QOS="$QOS",MAX_ITERATIONS="$MAX_ITERATIONS",N_EPISODES="$N_EPISODES",BUFFER_SIZE="$BUFFER_SIZE",COLLECTION_TASK_ID="$COLLECTION_TASK_ID",TRACE_ROLLOUTS="$TRACE_ROLLOUTS",GAMMA="$GAMMA",TD_LAMBDA="$TD_LAMBDA",BETA="$BETA",MAX_ADV_WEIGHT="$MAX_ADV_WEIGHT",MIN_ADVANTAGE_STD="$MIN_ADVANTAGE_STD",VALUE_ENCODER_MODEL="$VALUE_ENCODER_MODEL",VALUE_STEPS="$VALUE_STEPS",VALUE_LR="$VALUE_LR",VALUE_WEIGHT_DECAY="$VALUE_WEIGHT_DECAY",VALUE_BATCH_SIZE="$VALUE_BATCH_SIZE",VALUE_ENCODE_BATCH_SIZE="$VALUE_ENCODE_BATCH_SIZE",VALUE_GRAD_CLIP_NORM="$VALUE_GRAD_CLIP_NORM",CRITIC_WARMUP_ITERATIONS="$CRITIC_WARMUP_ITERATIONS",ACTOR_STEPS="$ACTOR_STEPS",ACTOR_LR="$ACTOR_LR",ACTOR_BATCH_SIZE="$ACTOR_BATCH_SIZE",ACTOR_GRAD_CLIP_NORM="$ACTOR_GRAD_CLIP_NORM",NUM_WORKERS="$NUM_WORKERS",EVAL_N_EPISODES="$EVAL_N_EPISODES",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_RUN_NAME="$WANDB_RUN_NAME",TASK="$TASK",EPISODE_LENGTH="$EPISODE_LENGTH",SEED="$SEED",PYTHON_BIN="$PYTHON_BIN",LIBERO_ROOT="$LIBERO_ROOT",WAN_WEIGHTS="$WAN_WEIGHTS",MODEL_CACHE="$MODEL_CACHE",REPO_ROOT="$REPO_ROOT" \
        "$SCRIPT_DIR/run_dawr_self_improvement.sh"
fi

exit "$EXIT_CODE"
