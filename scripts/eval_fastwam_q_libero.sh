#!/bin/bash
# One paper-style LIBERO-10 task for the FastWAM + standard-Q best-of-N ablation.
# Submit through launch_fastwam_q_argmax_libero10.sh; do not run this directly
# without TASK_ID, DIFFUSION_STEPS, and OUTPUT_ROOT.
#SBATCH --job-name=fwq_b64
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=64G
#SBATCH --qos=embers
#SBATCH --time=8:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm_out/%x-%A_%a.out
#SBATCH --error=slurm_out/%x-%A_%a.err

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-/storage/project/r-agarg35-0/akhandelwal79/conda-envs/lerobot-libero-new/bin/python}
LIBERO_ROOT=${LIBERO_ROOT:-/storage/project/r-agarg35-0/akhandelwal79/LIBERO}

FASTWAM_CKPT=${FASTWAM_CKPT:-/storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint}
Q_CKPT=${Q_CKPT:-/storage/project/r-agarg35-0/shared/qplanning_rebuttal_handoff/libero/checkpoints/libero_q_bc_h200_012000/pretrained_model}
WAN_WEIGHTS=${WAN_WEIGHTS:-/storage/project/r-agarg35-0/shared/awm/fastwam_wan22_weights}
MODEL_CACHE=${MODEL_CACHE:-/storage/project/r-agarg35-0/vgiridhar6/hf_cache}

LIBERO_TASK=libero_10
TASK_ID=${TASK_ID:-${SLURM_ARRAY_TASK_ID:-}}
DIFFUSION_STEPS=${DIFFUSION_STEPS:?set DIFFUSION_STEPS=10 or 20}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT}
N_EPISODES=${N_EPISODES:-50}
N_SAMPLES=${N_SAMPLES:-64}
SEED=${SEED:-42}
EVAL_PROTOCOL=${EVAL_PROTOCOL:-distinct700}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
SMOKE_TEST=${SMOKE_TEST:-0}

case "$EVAL_PROTOCOL" in
    distinct700)
        EXPECTED_EPISODE_LENGTH=700
        STRIDE_INIT_STATES=true
        INIT_STATES_DESC=0-49
        ;;
    fixed0_520)
        EXPECTED_EPISODE_LENGTH=520
        STRIDE_INIT_STATES=false
        INIT_STATES_DESC="0 repeated"
        ;;
    *)
        echo "EVAL_PROTOCOL must be distinct700 or fixed0_520; got '$EVAL_PROTOCOL'." >&2
        exit 2
        ;;
esac
EPISODE_LENGTH=${EPISODE_LENGTH:-$EXPECTED_EPISODE_LENGTH}

if [[ ! "$TASK_ID" =~ ^[0-9]+$ ]] || (( TASK_ID < 0 || TASK_ID > 9 )); then
    echo "TASK_ID must be an integer in [0, 9]; got '${TASK_ID:-unset}'." >&2
    exit 2
fi
if [[ "$DIFFUSION_STEPS" != 10 && "$DIFFUSION_STEPS" != 20 ]]; then
    echo "DIFFUSION_STEPS must be 10 or 20; got '$DIFFUSION_STEPS'." >&2
    exit 2
fi
if [[ "$N_SAMPLES" != 64 ]]; then
    echo "This rebuttal ablation is fixed at best-of-64; got N_SAMPLES='$N_SAMPLES'." >&2
    exit 2
fi
if [[ "$SMOKE_TEST" == 0 && "$N_EPISODES" != 50 ]]; then
    echo "This paper-style launcher requires 50 episodes/task; got N_EPISODES='$N_EPISODES'." >&2
    exit 2
fi
if [[ "$SMOKE_TEST" == 1 && "$N_EPISODES" != 1 ]]; then
    echo "SMOKE_TEST=1 requires N_EPISODES=1; got '$N_EPISODES'." >&2
    exit 2
fi
if [[ "$EPISODE_LENGTH" != "$EXPECTED_EPISODE_LENGTH" ]]; then
    echo "EVAL_PROTOCOL=$EVAL_PROTOCOL requires horizon $EXPECTED_EPISODE_LENGTH; got EPISODE_LENGTH='$EPISODE_LENGTH'." >&2
    exit 2
fi
if [[ "$PREFLIGHT_ONLY" != 0 && "$PREFLIGHT_ONLY" != 1 ]]; then
    echo "PREFLIGHT_ONLY must be 0 or 1; got '$PREFLIGHT_ONLY'." >&2
    exit 2
fi
if [[ "$SMOKE_TEST" != 0 && "$SMOKE_TEST" != 1 ]]; then
    echo "SMOKE_TEST must be 0 or 1; got '$SMOKE_TEST'." >&2
    exit 2
fi

for required_path in \
    "$PYTHON_BIN" \
    "$FASTWAM_CKPT/config.json" \
    "$FASTWAM_CKPT/model.safetensors" \
    "$FASTWAM_CKPT/policy_preprocessor.json" \
    "$Q_CKPT/config.json" \
    "$Q_CKPT/model.safetensors" \
    "$Q_CKPT/policy_preprocessor.json" \
    "$Q_CKPT/policy_preprocessor_step_3_normalizer_processor.safetensors" \
    "$WAN_WEIGHTS/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors" \
    "$WAN_WEIGHTS/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors" \
    "$WAN_WEIGHTS/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json" \
    "$MODEL_CACHE/models--facebook--dinov2-large/snapshots" \
    "$MODEL_CACHE/models--google--t5-v1_1-base/snapshots"; do
    if [[ ! -r "$required_path" ]]; then
        echo "Required launch input is not readable: $required_path" >&2
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR=${MPLCONFIGDIR:-$HF_HOME/matplotlib}
mkdir -p "$MPLCONFIGDIR"

cd "$REPO_ROOT"

# Cheap checks run before allocating model weights. They catch the two common
# rebuttal-time failures: a wrong Python environment and a mismatched Q horizon.
"$PYTHON_BIN" - "$FASTWAM_CKPT" "$Q_CKPT" <<'PY'
import json
import sys
from pathlib import Path

import libero.libero  # noqa: F401
import sentencepiece  # noqa: F401
import transformers
from transformers import AutoConfig, T5Tokenizer
from lerobot.envs.libero import LiberoEnv  # noqa: F401
from lerobot.policies.fastwam.planning import FastWAMPlanner  # noqa: F401
from lerobot.scripts import lerobot_eval  # noqa: F401

fastwam_dir, q_dir = map(Path, sys.argv[1:])
fastwam_cfg = json.loads((fastwam_dir / "config.json").read_text())
q_cfg = json.loads((q_dir / "config.json").read_text())
if fastwam_cfg.get("chunk_size") != 32:
    raise RuntimeError(f"FastWAM chunk_size must be 32, got {fastwam_cfg.get('chunk_size')}")
if q_cfg.get("h") != 32 or q_cfg.get("type") != "q_function":
    raise RuntimeError(f"Expected a Q-function with h=32, got type={q_cfg.get('type')} h={q_cfg.get('h')}")
AutoConfig.from_pretrained(q_cfg["dino_model_name"], local_files_only=True)
T5Tokenizer.from_pretrained(q_cfg["text_encoder_model"], local_files_only=True)
print(f"preflight imports OK (transformers={transformers.__version__})")
PY

if (( PREFLIGHT_ONLY == 1 )); then
    echo "Preflight-only validation passed; no evaluation was started."
    exit 0
fi

OUTPUT_DIR="$OUTPUT_ROOT/steps${DIFFUSION_STEPS}/task${TASK_ID}"
if [[ -e "$OUTPUT_DIR/eval_info.json" || -e "$OUTPUT_DIR/RUNNING" ]]; then
    echo "Refusing to overwrite an existing or active result: $OUTPUT_DIR" >&2
    exit 3
fi
mkdir -p "$OUTPUT_DIR"
touch "$OUTPUT_DIR/RUNNING"

# shellcheck disable=SC2317  # Called indirectly by the EXIT trap.
mark_result() {
    local rc=$?
    rm -f "$OUTPUT_DIR/RUNNING"
    if (( rc == 0 )) && [[ -s "$OUTPUT_DIR/eval_info.json" ]]; then
        printf '0\n' > "$OUTPUT_DIR/DONE"
        rm -f "$OUTPUT_DIR/FAILED"
    else
        printf '%s\n' "$rc" > "$OUTPUT_DIR/FAILED"
    fi
}
trap mark_result EXIT

echo "=== FastWAM + standard-Q exact best-of-64 ==="
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
echo "task=$LIBERO_TASK/$TASK_ID episodes=$N_EPISODES init_states=$INIT_STATES_DESC horizon=$EPISODE_LENGTH protocol=$EVAL_PROTOCOL"
echo "planner=bc_diffusion_argmax samples=$N_SAMPLES diffusion_steps=$DIFFUSION_STEPS eval_batch=1"
echo "seed=$SEED output=$OUTPUT_DIR"
echo "python=$PYTHON_BIN"
echo "FastWAM=$FASTWAM_CKPT"
echo "Q=$Q_CKPT"

COMMON_ARGS=(
    --policy.path="$FASTWAM_CKPT"
    --policy.device=cuda
    --policy.use_amp=false
    --env.type=libero
    --env.task="$LIBERO_TASK"
    --env.task_ids="[$TASK_ID]"
    --env.init_states=true
    --env.stride_init_states="$STRIDE_INIT_STATES"
    --env.episode_length="$EPISODE_LENGTH"
    --env.observation_height=224
    --env.observation_width=224
    --env.max_parallel_tasks=1
    --eval.batch_size=1
    --eval.n_episodes="$N_EPISODES"
    --eval.max_episodes_rendered=0
    --policy.num_inference_steps="$DIFFUSION_STEPS"
    --policy.use_planning=true
    --policy.planning.q_checkpoint_path="$Q_CKPT"
    --policy.planning.planner_type=bc_diffusion_argmax
    --policy.planning.n_samples="$N_SAMPLES"
    --policy.planning.num_diffusion_steps="$DIFFUSION_STEPS"
    --policy.planning.context_noise_std=0.0
    --output_dir="$OUTPUT_DIR"
    --seed="$SEED"
)

set +e
printf 'N\n' | "$PYTHON_BIN" -m lerobot.scripts.lerobot_eval "${COMMON_ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/log.txt"
RC=${PIPESTATUS[1]}
set -e
exit "$RC"
