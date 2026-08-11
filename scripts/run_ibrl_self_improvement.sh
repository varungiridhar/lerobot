#!/bin/bash
# Released-protocol-style primitive-action TD3-IBRL: frozen FastWAM proposal
# versus a trainable TD3 proposal. Each array element trains one independent
# task agent with action-level proposal selection and bootstrapping (H=1).
#
# Submit the default one-seed LIBERO-10 baseline (one agent per task):
#   sbatch scripts/run_ibrl_self_improvement.sh
# Paper reporting should repeat the task array for each declared seed, e.g.:
#   sbatch --export=ALL,SEED=42 --array=0-9 scripts/run_ibrl_self_improvement.sh
#   sbatch --export=ALL,SEED=43 --array=0-9 scripts/run_ibrl_self_improvement.sh
#   sbatch --export=ALL,SEED=44 --array=0-9 scripts/run_ibrl_self_improvement.sh
# Override --array=0 for a single-task smoke test.
#SBATCH -J ibrl_h1_fastwam
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH -p gpu-h200
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=96G
#SBATCH -q embers
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:H200:1
#SBATCH --exclude=atl1-1-03-020-11-0,atl1-1-01-009-16-0
#SBATCH --requeue
#SBATCH --array=0-9%10
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err

set -Eeuo pipefail

LOCAL_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -z "${REPO_ROOT:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" \
        && -d "$SLURM_SUBMIT_DIR/src/lerobot" \
        && -r "$SLURM_SUBMIT_DIR/scripts/self_improvement_ibrl_td3_loop.py" ]]; then
        REPO_ROOT=$SLURM_SUBMIT_DIR
    else
        REPO_ROOT=$(cd -- "$LOCAL_SCRIPT_DIR/.." && pwd)
    fi
fi
PYTHON_BIN=${PYTHON_BIN:-/storage/project/r-agarg35-0/akhandelwal79/conda-envs/lerobot-libero-new/bin/python}
LIBERO_ROOT=${LIBERO_ROOT:-/storage/project/r-agarg35-0/akhandelwal79/LIBERO}
FASTWAM_CKPT=${FASTWAM_CKPT:-/storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint}
VISION_ENCODER_MODEL=${VISION_ENCODER_MODEL:-facebook/dinov2-large}
WAN_WEIGHTS=${WAN_WEIGHTS:-/storage/project/r-agarg35-0/shared/awm/fastwam_wan22_weights}
MODEL_CACHE=${MODEL_CACHE:-/storage/project/r-agarg35-0/vgiridhar6/hf_cache}
DEMO_DATASET_REPO_ID=${DEMO_DATASET_REPO_ID:-HuggingFaceVLA/libero}
DEMO_DATASET_ROOT=${DEMO_DATASET_ROOT:-/storage/project/r-agarg35-0/shared/lerobot-data-2/HuggingFaceVLA/libero}

TASK=${TASK:-libero_10}
COLLECTION_TASK_ID=${COLLECTION_TASK_ID-${SLURM_ARRAY_TASK_ID:-0}}
EPISODE_LENGTH=${EPISODE_LENGTH:-520}
N_EPISODES=${N_EPISODES:-100}
EVAL_N_EPISODES=${EVAL_N_EPISODES:-50}
ACTION_HORIZON=${ACTION_HORIZON:-1}
BUFFER_SIZE=${BUFFER_SIZE:-100000}
BATCH_SIZE=${BATCH_SIZE:-256}
N_DEMO_EPISODES=${N_DEMO_EPISODES:-10}
BC_WARMUP_EPISODES=${BC_WARMUP_EPISODES:-40}
UPDATE_EVERY_ENV_STEPS=${UPDATE_EVERY_ENV_STEPS:-2}
# The released IBRL protocol starts hybrid interaction without offline learner
# pretraining. Keep zero for the primary baseline; nonzero is a labeled ablation.
LEARNER_WARMSTART_UPDATES=${LEARNER_WARMSTART_UPDATES:-0}

GAMMA=${GAMMA:-0.99}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
ACTOR_DROPOUT=${ACTOR_DROPOUT:-0.5}
ACTOR_LR=${ACTOR_LR:-1e-4}
CRITIC_LR=${CRITIC_LR:-1e-4}
TAU=${TAU:-0.01}
POLICY_DELAY=${POLICY_DELAY:-1}
EXPLORATION_NOISE=${EXPLORATION_NOISE:-0.1}
TARGET_NOISE=${TARGET_NOISE:-0.1}
TARGET_NOISE_CLIP=${TARGET_NOISE_CLIP:-0.3}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-10.0}

CHECKPOINT_EVERY_EPISODES=${CHECKPOINT_EVERY_EPISODES:-10}
SEED=${SEED:-42}
EVAL_SEED=${EVAL_SEED:-100000}
WANDB_PROJECT=${WANDB_PROJECT:-awm}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
PHASE=${PHASE:-all}
ONLINE_EPISODE_END=${ONLINE_EPISODE_END:-}
RESUME_STATE=${RESUME_STATE:-}
RUNTIME_RETRIES=${RUNTIME_RETRIES:-5}
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_ID=${RUN_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$TIMESTAMP}}}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/self_improvement_ibrl/${RUN_ID}_td3_ibrl_h${ACTION_HORIZON}_${TASK}_seed${SEED}}
OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/task${COLLECTION_TASK_ID:-all}}
HF_HOME=${HF_HOME:-/storage/project/r-agarg35-0/akhandelwal79/hf}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
ENCODER_CACHE_NAME=${VISION_ENCODER_MODEL//\//--}

if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
fi

for required_path in \
    "$PYTHON_BIN" \
    "$LIBERO_ROOT/libero" \
    "$FASTWAM_CKPT/config.json" \
    "$FASTWAM_CKPT/model.safetensors" \
    "$FASTWAM_CKPT/policy_preprocessor.json" \
    "$FASTWAM_CKPT/policy_postprocessor.json" \
    "$DEMO_DATASET_ROOT/meta/info.json" \
    "$DEMO_DATASET_ROOT/meta/tasks.parquet" \
    "$DEMO_DATASET_ROOT/data" \
    "$WAN_WEIGHTS/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors" \
    "$WAN_WEIGHTS/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors" \
    "$WAN_WEIGHTS/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json" \
    "$MODEL_CACHE/models--$ENCODER_CACHE_NAME/snapshots"; do
    if [[ ! -r "$required_path" ]]; then
        echo "Required launch input is not readable: $required_path" >&2
        exit 2
    fi
done

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
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR=${WANDB_DIR:-/storage/project/r-agarg35-0/akhandelwal79/wandb/ibrl}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/storage/project/r-agarg35-0/akhandelwal79/wandb-cache/ibrl}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$HF_HOME/matplotlib}
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$MPLCONFIGDIR" "$REPO_ROOT/logs" "$OUTPUT_DIR"

cd "$REPO_ROOT"

"$PYTHON_BIN" - "$FASTWAM_CKPT" "$VISION_ENCODER_MODEL" "$ACTION_HORIZON" <<'PY'
import json
import sys
from pathlib import Path

import libero.libero  # noqa: F401
import torch
from transformers import AutoConfig

from lerobot.policies.fastwam.ibrl_td3 import IBRLBatch, IBRLTD3

fastwam_dir = Path(sys.argv[1])
encoder_model = sys.argv[2]
action_horizon = int(sys.argv[3])
config = json.loads((fastwam_dir / "config.json").read_text())
if config.get("type") != "fastwam":
    raise RuntimeError("TD3-IBRL requires a FastWAM checkpoint")
if not 1 <= action_horizon <= int(config["chunk_size"]):
    raise RuntimeError(
        f"ACTION_HORIZON must be in [1, {config['chunk_size']}], got {action_horizon}"
    )
AutoConfig.from_pretrained(encoder_model, local_files_only=True)

agent = IBRLTD3(observation_dim=8, action_dim=action_horizon * int(config["action_dim"]), hidden_dim=16)
batch_size = 2
batch = IBRLBatch(
    observations=torch.randn(batch_size, 8),
    actions=torch.zeros(batch_size, agent.action_dim),
    rewards=torch.zeros(batch_size, 1),
    next_observations=torch.randn(batch_size, 8),
    dones=torch.zeros(batch_size, 1),
    discounts=torch.full((batch_size, 1), 0.99**action_horizon),
    next_bc_actions=torch.zeros(batch_size, agent.action_dim),
)
metrics = agent.update(batch)
if not torch.isfinite(torch.tensor(metrics["critic_loss"])):
    raise RuntimeError("TD3-IBRL preflight produced a non-finite critic loss")
print(f"TD3-IBRL preflight OK: H={action_horizon}, flattened action dim={agent.action_dim}")
PY

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
    echo "Preflight-only validation passed; no rollout or training was started."
    exit 0
fi

echo "=== FastWAM TD3-IBRL ==="
echo "  FROZEN BC:        $FASTWAM_CKPT"
echo "  RL POLICY:        deterministic TD3 actor"
echo "  ACTION CHOICE:    max over min target-Q for BC and RL proposals"
echo "  TD TARGET:        max over next BC and target-RL proposals"
echo "  ACTION_HORIZON:   $ACTION_HORIZON (1 gives paper action-level mechanics)"
echo "  PHASE:            $PHASE${ONLINE_EPISODE_END:+ (through online episode $ONLINE_EPISODE_END)}"
echo "  TASK:             $TASK / ${COLLECTION_TASK_ID:-all task ids}"
echo "  DEMO PREFILL:      $N_DEMO_EPISODES successful expert episodes from $DEMO_DATASET_REPO_ID"
echo "  BC WARMUP:         $BC_WARMUP_EPISODES complete frozen-BC episodes (no updates)"
echo "  ONLINE TRAIN:      $N_EPISODES hybrid episodes, max $EPISODE_LENGTH env steps"
echo "  FINAL EVAL:       $EVAL_N_EPISODES paired episodes, seed $EVAL_SEED, zero exploration"
echo "  REPLAY:           $BUFFER_SIZE decisions, batch $BATCH_SIZE"
echo "  UPDATE CADENCE:   one learner update / $UPDATE_EVERY_ENV_STEPS executed env steps"
echo "  ACTOR DROPOUT:    $ACTOR_DROPOUT (training/target backup only)"
echo "  PRE-RELEASE FIT:  $LEARNER_WARMSTART_UPDATES updates (0 is released protocol)"
echo "  OUTPUT_DIR:       $OUTPUT_DIR"
echo "=========================="

ARGS=(
    scripts/self_improvement_ibrl_td3_loop.py
    --fastwam_ckpt "$FASTWAM_CKPT"
    --task "$TASK"
    --episode_length "$EPISODE_LENGTH"
    --n_episodes "$N_EPISODES"
    --eval_n_episodes "$EVAL_N_EPISODES"
    --action_horizon "$ACTION_HORIZON"
    --phase "$PHASE"
    --vision_encoder_model "$VISION_ENCODER_MODEL"
    --hidden_dim "$HIDDEN_DIM"
    --actor_dropout "$ACTOR_DROPOUT"
    --buffer_size "$BUFFER_SIZE"
    --batch_size "$BATCH_SIZE"
    --demo_dataset_repo_id "$DEMO_DATASET_REPO_ID"
    --demo_dataset_root "$DEMO_DATASET_ROOT"
    --n_demo_episodes "$N_DEMO_EPISODES"
    --bc_warmup_episodes "$BC_WARMUP_EPISODES"
    --update_every_env_steps "$UPDATE_EVERY_ENV_STEPS"
    --learner_warmstart_updates "$LEARNER_WARMSTART_UPDATES"
    --gamma "$GAMMA"
    --actor_lr "$ACTOR_LR"
    --critic_lr "$CRITIC_LR"
    --tau "$TAU"
    --policy_delay "$POLICY_DELAY"
    --exploration_noise "$EXPLORATION_NOISE"
    --target_noise "$TARGET_NOISE"
    --target_noise_clip "$TARGET_NOISE_CLIP"
    --max_grad_norm "$MAX_GRAD_NORM"
    --checkpoint_every_episodes "$CHECKPOINT_EVERY_EPISODES"
    --output_dir "$OUTPUT_DIR"
    --seed "$SEED"
    --eval_seed "$EVAL_SEED"
    --wandb_project "$WANDB_PROJECT"
)
if [[ -n "$ONLINE_EPISODE_END" ]]; then
    ARGS+=(--online_episode_end "$ONLINE_EPISODE_END")
fi
if [[ -n "$RESUME_STATE" ]]; then
    ARGS+=(--resume_state "$RESUME_STATE")
fi
if [[ -n "$COLLECTION_TASK_ID" ]]; then
    ARGS+=(--collection_task_id "$COLLECTION_TASK_ID")
fi
if [[ -n "$WANDB_ENTITY" ]]; then
    ARGS+=(--wandb_entity "$WANDB_ENTITY")
fi
if [[ -n "$WANDB_RUN_NAME" ]]; then
    ARGS+=(--wandb_run_name "$WANDB_RUN_NAME")
fi

attempt=0
while true; do
    set +e
    "$PYTHON_BIN" "${ARGS[@]}"
    status=$?
    set -e
    if (( status == 0 )); then
        exit 0
    fi
    attempt=$((attempt + 1))
    if (( status != 134 || attempt > RUNTIME_RETRIES )); then
        echo "IBRL worker failed with status $status after $attempt attempt(s)." >&2
        exit "$status"
    fi
    echo "Native renderer aborted; resuming phase from its durable checkpoint " \
        "(retry $attempt/$RUNTIME_RETRIES)." >&2
    sleep 10
done
