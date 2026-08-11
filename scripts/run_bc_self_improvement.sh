#!/bin/bash
# Success-only FastWAM self-training on LIBERO, without Q or planning.
#
# Submit:
#   sbatch scripts/run_bc_self_improvement.sh
#
# The job chains one iteration at a time. Only FASTWAM_CKPT advances when a
# new success-filtered BC checkpoint is produced.
#SBATCH -J bc_si_loop
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH -p gpu-l40s
# One synchronous LIBERO environment and two loader workers fit comfortably
# within four CPUs; the L40S nodes provide four CPUs per GPU.
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=192G
#SBATCH -q embers
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

set -Eeuo pipefail

LOCAL_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$LOCAL_SCRIPT_DIR/.." && pwd)}}
SCRIPT_DIR="$REPO_ROOT/scripts"
PYTHON_BIN=${PYTHON_BIN:-/storage/project/r-agarg35-0/akhandelwal79/conda-envs/lerobot-libero-new/bin/python}
LIBERO_ROOT=${LIBERO_ROOT:-/storage/project/r-agarg35-0/akhandelwal79/LIBERO}

FASTWAM_CKPT=${FASTWAM_CKPT:-/storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint}
ORIGINAL_DATASET_REPO_ID=${ORIGINAL_DATASET_REPO_ID:-HuggingFaceVLA/libero}
ORIGINAL_DATASET_ROOT=${ORIGINAL_DATASET_ROOT:-/storage/project/r-agarg35-0/shared/lerobot-data-2/HuggingFaceVLA/libero}
WAN_WEIGHTS=${WAN_WEIGHTS:-/storage/project/r-agarg35-0/shared/awm/fastwam_wan22_weights}
TASK=${TASK:-libero_10}
EPISODE_LENGTH=${EPISODE_LENGTH:-520}
ITERATION=${ITERATION:-0}
MAX_ITERATIONS=${MAX_ITERATIONS:-5}
N_EPISODES=${N_EPISODES:-100}
FINETUNE_STEPS=${FINETUNE_STEPS:-200}
FINETUNE_LR=${FINETUNE_LR:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-1}
ONLINE_FRACTION=${ONLINE_FRACTION:-0.5}
GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-1.0}
NUM_WORKERS=${NUM_WORKERS:-2}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-20}
# Use distinct variable names so the already-running chain's legacy
# EVAL_N_EPISODES=50 export cannot keep overriding intermediate rounds.
INTERMEDIATE_EVAL_N_EPISODES=${INTERMEDIATE_EVAL_N_EPISODES:-20}
FINAL_EVAL_N_EPISODES=${FINAL_EVAL_N_EPISODES:-50}
if (( ITERATION == MAX_ITERATIONS - 1 )); then
    CURRENT_EVAL_N_EPISODES=$FINAL_EVAL_N_EPISODES
else
    CURRENT_EVAL_N_EPISODES=$INTERMEDIATE_EVAL_N_EPISODES
fi
SEED=${SEED:-42}
WANDB_PROJECT=${WANDB_PROJECT:-awm}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
ACCOUNT=${ACCOUNT:-gts-agarg35}
PARTITION=${PARTITION:-gpu-l40s}
QOS=${QOS:-embers}
GRES=${GRES:-gpu:l40s:1}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_DIR=${OUTPUT_DIR:-outputs/self_improvement_bc/${TIMESTAMP}_bc_si_${TASK}_n${N_EPISODES}_s${FINETUNE_STEPS}}
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
    "$ORIGINAL_DATASET_ROOT/meta/info.json" \
    "$HF_DATASETS_CACHE" \
    "$WAN_WEIGHTS/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors" \
    "$WAN_WEIGHTS/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors" \
    "$WAN_WEIGHTS/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json"; do
    if [[ ! -r "$required_path" ]]; then
        echo "Required launch input is not readable: $required_path" >&2
        exit 2
    fi
done

export PYTHONPATH="$REPO_ROOT/src:$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME
export HF_DATASETS_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_MODEL_BASE_PATH="$WAN_WEIGHTS"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR=${WANDB_DIR:-/storage/project/r-agarg35-0/akhandelwal79/wandb/bc_self_improvement}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/storage/project/r-agarg35-0/akhandelwal79/wandb-cache/bc_self_improvement}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$HF_HOME/matplotlib}
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$MPLCONFIGDIR" "$REPO_ROOT/logs" "$OUTPUT_DIR"

cd "$REPO_ROOT"

# Validate the plain FastWAM and real replay-data contract before allocating
# the large policy.
"$PYTHON_BIN" - "$FASTWAM_CKPT" "$ORIGINAL_DATASET_ROOT" "$ORIGINAL_DATASET_REPO_ID" <<'PY'
import json
import sys
from pathlib import Path

import libero.libero  # noqa: F401
import sentencepiece  # noqa: F401

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.fastwam.online_bc_dataset import load_original_success_dataset

fastwam_dir = Path(sys.argv[1])
original_data = Path(sys.argv[2])
original_repo_id = sys.argv[3]
fastwam_cfg = json.loads((fastwam_dir / "config.json").read_text())
if fastwam_cfg.get("chunk_size") != 32:
    raise RuntimeError(f"FastWAM chunk_size must be 32, got {fastwam_cfg.get('chunk_size')}")
if not (original_data / "meta" / "info.json").is_file():
    raise RuntimeError(f"Original LeRobot dataset is incomplete: {original_data}")

# Exercise one real original demonstration through the memory-mapped replay
# path before allocating the large policy.  This catches missing Arrow cache,
# task metadata, action deltas, image resize, and gripper conversion failures.
policy_cfg = PreTrainedConfig.from_pretrained(fastwam_dir)
metadata = LeRobotDatasetMetadata(original_repo_id, root=original_data)
first_task = metadata.episodes[0]["tasks"][0]
replay = load_original_success_dataset(
    original_repo_id,
    original_data,
    policy_cfg,
    task_descriptions=[first_task],
)
sample = replay[0]
if sample["action"].shape != (32, 7):
    raise RuntimeError(f"Expected FastWAM action target (32, 7), got {tuple(sample['action'].shape)}")
if any(tuple(sample[key].shape) != (3, 224, 224) for key in policy_cfg.image_features):
    raise RuntimeError("Original replay cameras do not match FastWAM's per-camera 224x224 contract")
gripper = sample["action"][..., -1]
if gripper.min() < 0 or gripper.max() > 1:
    raise RuntimeError("Original replay gripper conversion did not produce FastWAM [0, 1] targets")
print(f"Plain FastWAM success-only BC preflight OK; real replay sample task={first_task!r}")
PY

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
    echo "Preflight-only validation passed; no rollout or training was started."
    exit 0
fi

echo "=== Plain FastWAM success-only self-training ==="
echo "  ITERATION:      $ITERATION / $MAX_ITERATIONS"
echo "  FASTWAM_CKPT:   $FASTWAM_CKPT"
echo "  TASK:           $TASK"
echo "  ENV WINDOW:     $EPISODE_LENGTH steps"
echo "  COLLECTION:     $N_EPISODES plain FastWAM episodes, inference_steps=$NUM_INFERENCE_STEPS"
echo "  BC FINETUNE:    steps=$FINETUNE_STEPS lr=$FINETUNE_LR batch=$BATCH_SIZE online=$ONLINE_FRACTION"
echo "  HELD-OUT EVAL:  $CURRENT_EVAL_N_EPISODES episodes/task (intermediate=$INTERMEDIATE_EVAL_N_EPISODES final=$FINAL_EVAL_N_EPISODES)"
echo "  OUTPUT_DIR:     $OUTPUT_DIR"
echo "============================================="

ARGS=(
    scripts/self_improvement_bc_loop.py
    --fastwam_ckpt "$FASTWAM_CKPT" \
    --original_dataset_repo_id "$ORIGINAL_DATASET_REPO_ID" \
    --original_dataset_root "$ORIGINAL_DATASET_ROOT" \
    --task "$TASK" \
    --episode_length "$EPISODE_LENGTH" \
    --n_iterations 1 \
    --start_iteration "$ITERATION" \
    --n_episodes "$N_EPISODES" \
    --finetune_steps "$FINETUNE_STEPS" \
    --finetune_lr "$FINETUNE_LR" \
    --batch_size "$BATCH_SIZE" \
    --online_fraction "$ONLINE_FRACTION" \
    --grad_clip_norm "$GRAD_CLIP_NORM" \
    --num_workers "$NUM_WORKERS" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --eval_n_episodes "$CURRENT_EVAL_N_EPISODES" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --wandb_project "$WANDB_PROJECT"
)
if [[ -n "$WANDB_ENTITY" ]]; then
    ARGS+=(--wandb_entity "$WANDB_ENTITY")
fi
if [[ -n "$WANDB_RUN_NAME" ]]; then
    ARGS+=(--wandb_run_name "$WANDB_RUN_NAME")
fi
set +e
"$PYTHON_BIN" "${ARGS[@]}" 2>&1 | tee -a "$OUTPUT_DIR/log.txt"

EXIT_CODE=${PIPESTATUS[0]}
set -e
NEXT_ITERATION=$((ITERATION + 1))
if [ "$EXIT_CODE" -eq 0 ] && [ "$NEXT_ITERATION" -lt "$MAX_ITERATIONS" ]; then
    PRODUCED_CKPT="$OUTPUT_DIR/iter_$(printf '%03d' "$ITERATION")/fastwam_checkpoint"
    if [ -d "$PRODUCED_CKPT" ]; then
        NEXT_FASTWAM_CKPT=$PRODUCED_CKPT
    else
        NEXT_FASTWAM_CKPT=$FASTWAM_CKPT
        echo "No successful-data checkpoint produced; carrying forward $FASTWAM_CKPT"
    fi
    echo "Chaining iteration $NEXT_ITERATION with plain FastWAM=$NEXT_FASTWAM_CKPT"
    sbatch \
        -A "$ACCOUNT" \
        -p "$PARTITION" \
        -q "$QOS" \
        --gres="$GRES" \
        --dependency="afterok:$SLURM_JOB_ID" \
        --export=ALL,ITERATION="$NEXT_ITERATION",FASTWAM_CKPT="$NEXT_FASTWAM_CKPT",OUTPUT_DIR="$OUTPUT_DIR",ACCOUNT="$ACCOUNT",PARTITION="$PARTITION",QOS="$QOS",GRES="$GRES",MAX_ITERATIONS="$MAX_ITERATIONS",N_EPISODES="$N_EPISODES",FINETUNE_STEPS="$FINETUNE_STEPS",FINETUNE_LR="$FINETUNE_LR",BATCH_SIZE="$BATCH_SIZE",ONLINE_FRACTION="$ONLINE_FRACTION",GRAD_CLIP_NORM="$GRAD_CLIP_NORM",NUM_WORKERS="$NUM_WORKERS",NUM_INFERENCE_STEPS="$NUM_INFERENCE_STEPS",INTERMEDIATE_EVAL_N_EPISODES="$INTERMEDIATE_EVAL_N_EPISODES",FINAL_EVAL_N_EPISODES="$FINAL_EVAL_N_EPISODES",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_RUN_NAME="$WANDB_RUN_NAME",ORIGINAL_DATASET_REPO_ID="$ORIGINAL_DATASET_REPO_ID",ORIGINAL_DATASET_ROOT="$ORIGINAL_DATASET_ROOT",TASK="$TASK",EPISODE_LENGTH="$EPISODE_LENGTH",SEED="$SEED",PYTHON_BIN="$PYTHON_BIN",LIBERO_ROOT="$LIBERO_ROOT",WAN_WEIGHTS="$WAN_WEIGHTS",REPO_ROOT="$REPO_ROOT" \
        "$SCRIPT_DIR/run_bc_self_improvement.sh"
fi

exit "$EXIT_CODE"
