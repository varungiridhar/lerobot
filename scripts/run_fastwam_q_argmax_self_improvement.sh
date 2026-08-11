#!/bin/bash
# One paper-protocol LIBERO-10 Q self-improvement iteration. On success, submit
# the next iteration until MAX_ITERATIONS is reached.
#
# Collection overrides the historical planner only: exact best-of-64 diffusion
# argmax with 10 diffusion steps. Q training retains every collected trajectory,
# including failures, and otherwise follows the handoff defaults.
#SBATCH --job-name=fwq_si_s10
#SBATCH --account=gts-agarg35
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=64G
#SBATCH --qos=embers
#SBATCH --time=03:00:00
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --output=slurm_out/%x-%j.out
#SBATCH --error=slurm_out/%x-%j.err

set -Eeuo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
    REPO_ROOT=$(cd -- "$REPO_ROOT" && pwd)
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT=$(cd -- "$SLURM_SUBMIT_DIR" && pwd)
else
    SOURCE_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    REPO_ROOT=$(cd -- "$SOURCE_SCRIPT_DIR/.." && pwd)
fi
SCRIPT_DIR="$REPO_ROOT/scripts"
PYTHON_BIN=${PYTHON_BIN:-/storage/project/r-agarg35-0/akhandelwal79/conda-envs/lerobot-libero-new/bin/python}
LIBERO_ROOT=${LIBERO_ROOT:-/storage/project/r-agarg35-0/akhandelwal79/LIBERO}

FASTWAM_CKPT=${FASTWAM_CKPT:-/storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint}
INITIAL_Q_CKPT=${INITIAL_Q_CKPT:-/storage/project/r-agarg35-0/shared/qplanning_rebuttal_handoff/libero/checkpoints/libero_q_bc_h200_012000/pretrained_model}
Q_CKPT=${Q_CKPT:-$INITIAL_Q_CKPT}
ORIGINAL_DATASET_REPO_ID=HuggingFaceVLA/libero
ORIGINAL_DATASET_ROOT=/storage/project/r-agarg35-0/shared/lerobot-data-2
WAN_WEIGHTS=/storage/project/r-agarg35-0/shared/awm/fastwam_wan22_weights
MODEL_CACHE=/storage/project/r-agarg35-0/vgiridhar6/hf_cache

ITERATION=${ITERATION:-0}
MAX_ITERATIONS=${MAX_ITERATIONS:-3}
SI_QOS=${SI_QOS:-embers}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/outputs/self_improvement/fastwam_q_argmax64_s10_libero10_fixed0_520_$RUN_TAG}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

# Paper/handoff self-improvement schedule, with the user-requested planner.
N_EPISODES=${N_EPISODES:-100}
FINETUNE_STEPS=${FINETUNE_STEPS:-200}
FINETUNE_LR=1e-5
BATCH_SIZE=48
ONLINE_FRACTION=0.5
GRAD_CLIP_NORM=10.0
PLANNER_TYPE=bc_diffusion_argmax
N_SAMPLES=64
DIFFUSION_STEPS=10
EPISODE_LENGTH=520
SEED=${SEED:-42}
TASK_ID=${TASK_ID:-}
COLLECTION_ONLY=${COLLECTION_ONLY:-0}
COLLECTION_START_SEED=${COLLECTION_START_SEED:-}
ROLLOUT_DIAGNOSTICS=${ROLLOUT_DIAGNOSTICS:-0}

if [[ ! "$MAX_ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_ITERATIONS must be a positive integer; got '$MAX_ITERATIONS'." >&2
    exit 2
fi
if [[ ! "$ITERATION" =~ ^[0-9]+$ ]] || (( ITERATION >= MAX_ITERATIONS )); then
    echo "ITERATION must be in [0, $((MAX_ITERATIONS - 1))]; got '$ITERATION'." >&2
    exit 2
fi
if [[ "$PREFLIGHT_ONLY" != 0 && "$PREFLIGHT_ONLY" != 1 ]]; then
    echo "PREFLIGHT_ONLY must be 0 or 1; got '$PREFLIGHT_ONLY'." >&2
    exit 2
fi
for boolean_name in COLLECTION_ONLY ROLLOUT_DIAGNOSTICS; do
    boolean_value=${!boolean_name}
    if [[ "$boolean_value" != 0 && "$boolean_value" != 1 ]]; then
        echo "$boolean_name must be 0 or 1; got '$boolean_value'." >&2
        exit 2
    fi
done

for required_path in \
    "$PYTHON_BIN" \
    "$FASTWAM_CKPT/config.json" \
    "$FASTWAM_CKPT/model.safetensors" \
    "$Q_CKPT/config.json" \
    "$Q_CKPT/model.safetensors" \
    "$Q_CKPT/policy_preprocessor.json" \
    "$Q_CKPT/policy_preprocessor_step_3_normalizer_processor.safetensors" \
    "$ORIGINAL_DATASET_ROOT/$ORIGINAL_DATASET_REPO_ID/meta/info.json" \
    "$WAN_WEIGHTS/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors" \
    "$WAN_WEIGHTS/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors" \
    "$MODEL_CACHE/models--facebook--dinov2-large/snapshots" \
    "$MODEL_CACHE/models--google--t5-v1_1-base/snapshots"; do
    if [[ ! -r "$required_path" ]]; then
        echo "Required input is not readable: $required_path" >&2
        exit 2
    fi
done

export PYTHONPATH="$REPO_ROOT/src:$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/storage/project/r-agarg35-0/akhandelwal79/hf}
export HF_HUB_CACHE="$MODEL_CACHE"
export TRANSFORMERS_CACHE="$MODEL_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_MODEL_BASE_PATH="$WAN_WEIGHTS"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR=${MPLCONFIGDIR:-$HF_HOME/matplotlib}

cd "$REPO_ROOT"
mkdir -p slurm_out "$OUTPUT_ROOT" "$MPLCONFIGDIR"

echo "=== FastWAM + Q exact-best-of-64 self-improvement ==="
echo "iteration=$ITERATION/$((MAX_ITERATIONS - 1)) task=libero_10 horizon=$EPISODE_LENGTH init_state=0-repeated"
echo "collect=$N_EPISODES replay=all-successes-and-failures planner=$PLANNER_TYPE samples=$N_SAMPLES diffusion_steps=$DIFFUSION_STEPS"
echo "finetune_steps=$FINETUNE_STEPS lr=$FINETUNE_LR batch=$BATCH_SIZE online_fraction=$ONLINE_FRACTION"
echo "seed=$SEED q=$Q_CKPT output=$OUTPUT_ROOT"
echo "qos=$SI_QOS"
echo "task_id=${TASK_ID:-all} collection_only=$COLLECTION_ONLY collection_start_seed=${COLLECTION_START_SEED:-derived} rollout_diagnostics=$ROLLOUT_DIAGNOSTICS"

"$PYTHON_BIN" - "$FASTWAM_CKPT" "$Q_CKPT" "$ORIGINAL_DATASET_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import libero.libero  # noqa: F401
import sentencepiece  # noqa: F401
from lerobot.policies.fastwam.planning import FastWAMPlanner  # noqa: F401
from lerobot.policies.q_function.online_dataset import OnlineQDataset  # noqa: F401
from lerobot.policies.q_function.processor_q_function import _q_batch_to_transition  # noqa: F401
from transformers import AutoConfig, T5Tokenizer

fastwam_dir, q_dir, dataset_root = map(Path, sys.argv[1:])
fastwam_cfg = json.loads((fastwam_dir / "config.json").read_text())
q_cfg = json.loads((q_dir / "config.json").read_text())
if fastwam_cfg.get("chunk_size") != 32 or fastwam_cfg.get("n_action_steps") != 10:
    raise RuntimeError(
        f"Expected FastWAM chunk_size=32/n_action_steps=10, got "
        f"{fastwam_cfg.get('chunk_size')}/{fastwam_cfg.get('n_action_steps')}"
    )
if q_cfg.get("type") != "q_function" or q_cfg.get("h") != 32:
    raise RuntimeError(f"Expected Q-function h=32, got type={q_cfg.get('type')} h={q_cfg.get('h')}")
if q_cfg.get("camera_keys") != ["observation.images.image", "observation.images.image2"]:
    raise RuntimeError(f"Unexpected LIBERO Q camera keys: {q_cfg.get('camera_keys')}")
AutoConfig.from_pretrained(q_cfg["dino_model_name"], local_files_only=True)
T5Tokenizer.from_pretrained(q_cfg["text_encoder_model"], local_files_only=True)
if not (dataset_root / "HuggingFaceVLA/libero/meta/info.json").is_file():
    raise RuntimeError("Original LIBERO Q dataset metadata is missing")
print("self-improvement preflight imports/config/cache OK")
PY

if (( PREFLIGHT_ONLY == 1 )); then
    echo "Preflight-only validation passed; no rollout or training was started."
    exit 0
fi

FINAL_ITER_DIR="$OUTPUT_ROOT/iter_$(printf '%03d' "$ITERATION")"
if [[ -s "$FINAL_ITER_DIR/metrics.json" && -s "$FINAL_ITER_DIR/q_checkpoint/model.safetensors" ]]; then
    echo "Iteration $ITERATION is already complete; retaining $FINAL_ITER_DIR."
    RC=0
else
    EXTRA_ARGS=()
    if [[ -n "$TASK_ID" ]]; then
        EXTRA_ARGS+=(--task_ids "$TASK_ID")
    fi
    if (( COLLECTION_ONLY == 1 )); then
        EXTRA_ARGS+=(--collection_only)
    fi
    if [[ -n "$COLLECTION_START_SEED" ]]; then
        EXTRA_ARGS+=(--collection_start_seed "$COLLECTION_START_SEED")
    fi
    if (( ROLLOUT_DIAGNOSTICS == 1 )); then
        EXTRA_ARGS+=(--rollout_diagnostics)
    fi
    set +e
    "$PYTHON_BIN" scripts/self_improvement_loop.py \
        --fastwam_ckpt "$FASTWAM_CKPT" \
        --q_ckpt "$Q_CKPT" \
        --original_dataset_repo_id "$ORIGINAL_DATASET_REPO_ID" \
        --original_dataset_root "$ORIGINAL_DATASET_ROOT" \
        --task libero_10 \
        --episode_length "$EPISODE_LENGTH" \
        --n_iterations 1 \
        --start_iteration "$ITERATION" \
        --n_episodes "$N_EPISODES" \
        --no-successful_only \
        --finetune_steps "$FINETUNE_STEPS" \
        --finetune_lr "$FINETUNE_LR" \
        --batch_size "$BATCH_SIZE" \
        --online_fraction "$ONLINE_FRACTION" \
        --grad_clip_norm "$GRAD_CLIP_NORM" \
        --planner_type "$PLANNER_TYPE" \
        --n_samples "$N_SAMPLES" \
        --diffusion_steps "$DIFFUSION_STEPS" \
        --eval_n_episodes 0 \
        --output_dir "$OUTPUT_ROOT" \
        --seed "$SEED" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee -a "$OUTPUT_ROOT/iteration_${ITERATION}.log"
    RC=${PIPESTATUS[0]}
    set -e
fi

if (( RC != 0 )); then
    exit "$RC"
fi
if (( COLLECTION_ONLY == 1 )); then
    if [[ ! -s "$FINAL_ITER_DIR/metrics.json" ]]; then
        echo "Collection-only probe exited successfully but metrics are missing: $FINAL_ITER_DIR" >&2
        exit 3
    fi
    echo "Collection-only probe completed; no Q checkpoint or chained job was produced."
    exit 0
fi
if [[ ! -s "$FINAL_ITER_DIR/metrics.json" || ! -s "$FINAL_ITER_DIR/q_checkpoint/model.safetensors" ]]; then
    echo "Iteration exited successfully but required outputs are missing: $FINAL_ITER_DIR" >&2
    exit 3
fi

NEXT_ITERATION=$((ITERATION + 1))
HELDOUT_LABEL="q${NEXT_ITERATION}"
HELDOUT_ROOT="$OUTPUT_ROOT/heldout_eval/$HELDOUT_LABEL"
HELDOUT_JOB_FILE="$OUTPUT_ROOT/heldout_eval/${HELDOUT_LABEL}.jobid"
mkdir -p "$OUTPUT_ROOT/heldout_eval"
if [[ ! -s "$HELDOUT_JOB_FILE" ]]; then
    heldout_job_id=$(Q_CKPT="$FINAL_ITER_DIR/q_checkpoint" \
        LABEL="$HELDOUT_LABEL" \
        OUTPUT_ROOT="$HELDOUT_ROOT" \
        SI_QOS="$SI_QOS" \
        "$SCRIPT_DIR/launch_fastwam_q_si_heldout_eval.sh")
    printf '%s\n' "$heldout_job_id" > "$HELDOUT_JOB_FILE"
    echo "submitted held-out $HELDOUT_LABEL evaluation as array $heldout_job_id"
else
    echo "held-out $HELDOUT_LABEL evaluation already recorded as job $(<"$HELDOUT_JOB_FILE")"
fi

if (( NEXT_ITERATION < MAX_ITERATIONS )); then
    NEXT_Q_CKPT="$FINAL_ITER_DIR/q_checkpoint"
    next_job_id=$(sbatch \
        --parsable \
        --job-name="fwq_si_s10_i${NEXT_ITERATION}" \
        --qos="$SI_QOS" \
        --dependency="afterok:$SLURM_JOB_ID" \
        --export="ALL,ITERATION=$NEXT_ITERATION,MAX_ITERATIONS=$MAX_ITERATIONS,SI_QOS=$SI_QOS,Q_CKPT=$NEXT_Q_CKPT,OUTPUT_ROOT=$OUTPUT_ROOT,RUN_TAG=$RUN_TAG" \
        "$SCRIPT_DIR/run_fastwam_q_argmax_self_improvement.sh")
    echo "submitted iteration $NEXT_ITERATION as dependent job $next_job_id"
fi

exit 0
