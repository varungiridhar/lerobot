#!/bin/bash
# Policy-only, success-filtered FastWAM self-improvement on the 47 evaluable
# RoboTwin tasks. Launch the orchestrator with bash, not sbatch:
#
#   bash scripts/run_robotwin_bc_self_improvement.sh
#
# Every round fans collection over configurable H100 shards, requires every completion
# manifests and exactly 100 rollouts, finetunes FastWAM on a 50/50 mixture of
# growing successful online data and the full RoboTwin 2.0 replay, then submits
# a sharded H100 policy-only eval (20 episodes per task). No Q function or
# planner is constructed anywhere in this experiment.

# These directives are fallbacks for MODE=collect/finetune/eval when manually
# submitted. The orchestrator supplies the same routing explicitly.
#SBATCH -J rtw_bc_si
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH -p gpu-h100
#SBATCH -q inferno
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=128G
#SBATCH -t 8:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

set -Eeuo pipefail

LOCAL_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$LOCAL_SCRIPT_DIR/.." && pwd)}}
PYTHON_BIN=${PYTHON_BIN:-/storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/miniconda3/envs/lerobot/bin/python}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-/storage/project/r-agarg35-0/vgiridhar6/robotwin/RoboTwin}
CUROBO_SRC=${CUROBO_SRC:-$ROBOTWIN_ROOT/envs/curobo/src}
FASTWAM_CKPT=${FASTWAM_CKPT:-/storage/project/r-agarg35-0/shared/fastwam/hf_checkpoint_robotwin}
ORIGINAL_DATASET_REPO_ID=${ORIGINAL_DATASET_REPO_ID:-local/robotwin2.0}
ORIGINAL_DATASET_ROOT=${ORIGINAL_DATASET_ROOT:-/storage/project/r-agarg35-0/shared/robotwin2.0}
WAN_WEIGHTS=${WAN_WEIGHTS:-/storage/project/r-agarg35-0/shared/awm/fastwam_wan22_weights}

MODE=${MODE:-orchestrate}
SHARD=${SHARD:-}
ITERATION=${ITERATION:-0}
MAX_ITERATIONS=${MAX_ITERATIONS:-5}
N_EPISODES=${N_EPISODES:-100}
COLLECT_SHARDS=${COLLECT_SHARDS:-4}
EVAL_SHARDS=${EVAL_SHARDS:-4}
EVAL_N_EPISODES=${EVAL_N_EPISODES:-20}
COLLECT_TIME=${COLLECT_TIME:-8:00:00}
FINETUNE_TIME=${FINETUNE_TIME:-8:00:00}
EVAL_TIME=${EVAL_TIME:-8:00:00}
FINETUNE_STEPS=${FINETUNE_STEPS:-200}
FINETUNE_LR=${FINETUNE_LR:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-1}
ONLINE_FRACTION=${ONLINE_FRACTION:-0.5}
GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-1.0}
NUM_WORKERS=${NUM_WORKERS:-2}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-10}
ONLINE_DATASET_FPS=${ONLINE_DATASET_FPS:-50}
SEED=${SEED:-42}
WANDB_PROJECT=${WANDB_PROJECT:-awm}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-robotwin_bc_filtered_sft_5x100}

ACCOUNT=${ACCOUNT:-gts-agarg35}
PARTITION=${PARTITION:-gpu-h100}
QOS=${QOS:-inferno}
GRES=${GRES:-gpu:h100:1}
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/self_improvement_bc_robotwin/${TIMESTAMP}_47tasks_5x100}
DRY_RUN=${DRY_RUN:-0}

# The three excluded tasks crash inside RoboTwin's learned-policy success check
# because task-only fields are initialized by the scripted expert but not reset().
TASKS=${TASKS:-adjust_bottle,beat_block_hammer,blocks_ranking_rgb,blocks_ranking_size,click_alarmclock,click_bell,dump_bin_bigbin,grab_roller,handover_block,handover_mic,hanging_mug,lift_pot,move_can_pot,move_pillbottle_pad,move_playingcard_away,move_stapler_pad,open_microwave,pick_diverse_bottles,pick_dual_bottles,place_a2b_left,place_a2b_right,place_bread_basket,place_bread_skillet,place_burger_fries,place_can_basket,place_cans_plasticbox,place_container_plate,place_dual_shoes,place_empty_cup,place_fan,place_mouse_pad,place_object_basket,place_object_stand,place_phone_stand,place_shoe,press_stapler,put_bottles_dustbin,rotate_qrcode,scan_object,shake_bottle,shake_bottle_horizontally,stack_blocks_three,stack_blocks_two,stack_bowls_three,stack_bowls_two,stamp_seal,turn_switch}

HF_HOME=${HF_HOME:-/storage/project/r-agarg35-0/shared/huggingface_cache}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
WANDB_DIR=${WANDB_DIR:-/storage/project/r-agarg35-0/akhandelwal79/wandb/robotwin_bc_self_improvement}
WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/storage/project/r-agarg35-0/akhandelwal79/wandb-cache/robotwin_bc_self_improvement}
JOB_TMP=${JOB_TMP:-/storage/project/r-agarg35-0/akhandelwal79/tmp/robotwin-bc-${SLURM_JOB_ID:-orchestrator}}

export REPO_ROOT PYTHON_BIN ROBOTWIN_ROOT CUROBO_SRC FASTWAM_CKPT
export ORIGINAL_DATASET_REPO_ID ORIGINAL_DATASET_ROOT WAN_WEIGHTS
export MODE SHARD ITERATION MAX_ITERATIONS N_EPISODES COLLECT_SHARDS EVAL_SHARDS
export EVAL_N_EPISODES COLLECT_TIME FINETUNE_TIME EVAL_TIME
export FINETUNE_STEPS FINETUNE_LR BATCH_SIZE ONLINE_FRACTION
export GRAD_CLIP_NORM NUM_WORKERS NUM_INFERENCE_STEPS ONLINE_DATASET_FPS SEED
export WANDB_PROJECT WANDB_ENTITY WANDB_RUN_NAME ACCOUNT PARTITION QOS GRES OUTPUT_DIR TASKS
export HF_HOME HF_DATASETS_CACHE WANDB_DIR WANDB_CACHE_DIR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_MODEL_BASE_PATH="$WAN_WEIGHTS"
export DIFFSYNTH_SKIP_DOWNLOAD=true MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_ROOT/src:$CUROBO_SRC:$ROBOTWIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$JOB_TMP"

mkdir -p "$REPO_ROOT/logs" "$OUTPUT_DIR" "$TMPDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"
cd "$REPO_ROOT"

task_count() {
    "$PYTHON_BIN" -c 'import os; print(len([x for x in os.environ["TASKS"].split(",") if x]))'
}

static_preflight() {
    for required_path in \
        "$PYTHON_BIN" \
        "$ROBOTWIN_ROOT/task_config/demo_randomized.yml" \
        "$CUROBO_SRC" \
        "$FASTWAM_CKPT/config.json" \
        "$FASTWAM_CKPT/model.safetensors" \
        "$FASTWAM_CKPT/policy_preprocessor.json" \
        "$ORIGINAL_DATASET_ROOT/meta/info.json"; do
        if [[ ! -r "$required_path" ]]; then
            echo "Required launch input is not readable: $required_path" >&2
            exit 2
        fi
    done
    local n_tasks
    n_tasks=$(task_count)
    if [[ "$n_tasks" -ne 47 ]]; then
        echo "Expected exactly 47 RoboTwin tasks, found $n_tasks" >&2
        exit 2
    fi
    "$PYTHON_BIN" scripts/check_robotwin_timing_contract.py \
        --robotwin_root "$ROBOTWIN_ROOT" \
        --dataset_root "$ORIGINAL_DATASET_ROOT" >/dev/null
    "$PYTHON_BIN" - "$ORIGINAL_DATASET_ROOT" <<'PY'
import json
import sys
from pathlib import Path

info = json.loads((Path(sys.argv[1]) / "meta" / "info.json").read_text())
expected = {
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
}
if info.get("total_episodes") != 27500 or info.get("total_frames") != 6075103:
    raise RuntimeError("The full 27,500-episode RoboTwin replay dataset is required")
if not expected.issubset(info["features"]):
    raise RuntimeError(f"Full replay is missing RoboTwin camera streams: {sorted(expected)}")
PY
}

submit_iteration() {
    local iteration=$1
    local shard_ids=()
    local shard_id
    local job_id
    echo "Submitting iteration $iteration: $N_EPISODES rollouts / 47 tasks / $COLLECT_SHARDS H100 shards"
    for ((shard_id=0; shard_id<COLLECT_SHARDS; shard_id++)); do
        if [[ "$DRY_RUN" == 1 ]]; then
            echo "  [dry-run] collect shard $shard_id/$COLLECT_SHARDS"
            continue
        fi
        job_id=$(sbatch --parsable \
            -A "$ACCOUNT" -p "$PARTITION" -q "$QOS" --gres="$GRES" \
            --cpus-per-gpu=8 --mem-per-gpu=128G -t "$COLLECT_TIME" \
            -J "rtbc_c${iteration}_${shard_id}" -o logs/%j.out -e logs/%j.err \
            --export=ALL,MODE=collect,SHARD="$shard_id/$COLLECT_SHARDS",ITERATION="$iteration",WANDB_PROJECT= \
            "$REPO_ROOT/scripts/run_robotwin_bc_self_improvement.sh")
        shard_ids+=("$job_id")
        echo "  collect shard $shard_id/$COLLECT_SHARDS -> $job_id"
    done
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "  [dry-run] finetune after all $COLLECT_SHARDS shard jobs terminate"
        return
    fi

    local dependency
    dependency=$(IFS=:; echo "${shard_ids[*]}")
    job_id=$(sbatch --parsable \
        -A "$ACCOUNT" -p "$PARTITION" -q "$QOS" --gres="$GRES" \
        --cpus-per-gpu=8 --mem-per-gpu=192G -t "$FINETUNE_TIME" \
        -J "rtbc_ft${iteration}" -o logs/%j.out -e logs/%j.err \
        --dependency="afterany:$dependency" \
        --export=ALL,MODE=finetune,SHARD=,ITERATION="$iteration" \
        "$REPO_ROOT/scripts/run_robotwin_bc_self_improvement.sh")
    echo "  finetune -> $job_id (afterany barrier; code requires all $COLLECT_SHARDS complete manifests)"
}

submit_evaluation() {
    local iteration=$1
    local shard_id
    local job_id
    echo "Submitting 47-task x 20-episode policy-only eval for iteration $iteration"
    for ((shard_id=0; shard_id<EVAL_SHARDS; shard_id++)); do
        job_id=$(sbatch --parsable \
            -A "$ACCOUNT" -p "$PARTITION" -q "$QOS" --gres="$GRES" \
            --cpus-per-gpu=8 --mem-per-gpu=128G -t "$EVAL_TIME" \
            -J "rtbc_e${iteration}_${shard_id}" -o logs/%j.out -e logs/%j.err \
            --export=ALL,MODE=eval,SHARD="$shard_id/$EVAL_SHARDS",ITERATION="$iteration",WANDB_PROJECT= \
            "$REPO_ROOT/scripts/run_robotwin_bc_self_improvement.sh")
        echo "  eval shard $shard_id/$EVAL_SHARDS -> $job_id"
    done
}

BC_ARGS=(
    scripts/self_improvement_bc_loop.py
    --env robotwin
    --fastwam_ckpt "$FASTWAM_CKPT"
    --robotwin_root "$ROBOTWIN_ROOT"
    --robotwin_task_config demo_randomized
    --robotwin_instruction_type unseen
    --original_dataset_repo_id "$ORIGINAL_DATASET_REPO_ID"
    --original_dataset_root "$ORIGINAL_DATASET_ROOT"
    --no-replay_same_tasks_only
    --task "$TASKS"
    --online_dataset_fps "$ONLINE_DATASET_FPS"
    --n_iterations 1
    --experiment_iterations "$MAX_ITERATIONS"
    --start_iteration "$ITERATION"
    --n_episodes "$N_EPISODES"
    --finetune_steps "$FINETUNE_STEPS"
    --finetune_lr "$FINETUNE_LR"
    --batch_size "$BATCH_SIZE"
    --online_fraction "$ONLINE_FRACTION"
    --grad_clip_norm "$GRAD_CLIP_NORM"
    --num_workers "$NUM_WORKERS"
    --num_inference_steps "$NUM_INFERENCE_STEPS"
    --eval_n_episodes 0
    --output_dir "$OUTPUT_DIR"
    --seed "$SEED"
)

case "$MODE" in
    orchestrate)
        static_preflight
        echo "=== RoboTwin filtered-SFT experiment ==="
        echo "  output:       $OUTPUT_DIR"
        echo "  rounds:       $MAX_ITERATIONS"
        echo "  collection:   $N_EPISODES rollouts over $(task_count) tasks on $COLLECT_SHARDS H100s"
        echo "  finetune:     $FINETUNE_STEPS steps, lr=$FINETUNE_LR, batch=$BATCH_SIZE, online=$ONLINE_FRACTION"
        echo "  eval:         $EVAL_N_EPISODES episodes/task over $(task_count) tasks on $EVAL_SHARDS H100s"
        echo "  Q/planning:   disabled"
        submit_iteration "$ITERATION"
        ;;

    collect)
        echo "=== collection iteration $ITERATION shard $SHARD ==="
        set +e
        "$PYTHON_BIN" "${BC_ARGS[@]}" --collect_only --task_shard "$SHARD" \
            2>&1 | tee -a "$OUTPUT_DIR/log_collect_iter$(printf '%03d' "$ITERATION")_shard${SHARD%%/*}.txt"
        exit_code=${PIPESTATUS[0]}
        set -e
        exit "$exit_code"
        ;;

    finetune)
        echo "=== finetune iteration $ITERATION ==="
        FT_ARGS=("${BC_ARGS[@]}" --skip_collect --expected_collect_shards "$COLLECT_SHARDS")
        if [[ -n "$WANDB_PROJECT" ]]; then
            FT_ARGS+=(--wandb_project "$WANDB_PROJECT")
        fi
        if [[ -n "$WANDB_ENTITY" ]]; then
            FT_ARGS+=(--wandb_entity "$WANDB_ENTITY")
        fi
        if [[ -n "$WANDB_RUN_NAME" ]]; then
            FT_ARGS+=(--wandb_run_name "${WANDB_RUN_NAME}_iter$(printf '%03d' "$ITERATION")")
        fi
        set +e
        "$PYTHON_BIN" "${FT_ARGS[@]}" 2>&1 | tee -a "$OUTPUT_DIR/log_finetune.txt"
        exit_code=${PIPESTATUS[0]}
        set -e
        if [[ "$exit_code" -ne 0 ]]; then
            exit "$exit_code"
        fi
        produced_ckpt="$OUTPUT_DIR/iter_$(printf '%03d' "$ITERATION")/fastwam_checkpoint"
        if [[ ! -d "$produced_ckpt" ]]; then
            echo "Expected FastWAM checkpoint was not produced: $produced_ckpt" >&2
            exit 1
        fi
        FASTWAM_CKPT="$produced_ckpt"
        export FASTWAM_CKPT
        submit_evaluation "$ITERATION"
        next_iteration=$((ITERATION + 1))
        if [[ "$next_iteration" -lt "$MAX_ITERATIONS" ]]; then
            submit_iteration "$next_iteration"
        fi
        ;;

    eval)
        echo "=== eval iteration $ITERATION shard $SHARD ==="
        checkpoint="$OUTPUT_DIR/iter_$(printf '%03d' "$ITERATION")/fastwam_checkpoint"
        eval_root="$OUTPUT_DIR/iter_$(printf '%03d' "$ITERATION")/eval_${EVAL_N_EPISODES}ep"
        mkdir -p "$eval_root"
        IFS=',' read -r -a all_tasks <<< "$TASKS"
        IFS='/' read -r shard_index n_shards <<< "$SHARD"
        failures=0
        for task_index in "${!all_tasks[@]}"; do
            if (( task_index % n_shards != shard_index )); then
                continue
            fi
            task=${all_tasks[$task_index]}
            task_output="$eval_root/$task"
            if [[ -f "$task_output/eval_info.json" ]]; then
                echo "Skipping completed task $task"
                continue
            fi
            mkdir -p "$task_output"
            echo ">>> $task (20 held-out episodes, seed $SEED)"
            set +e
            "$PYTHON_BIN" -m lerobot.scripts.lerobot_eval \
                --policy.path="$checkpoint" \
                --policy.device=cuda \
                --policy.num_inference_steps="$NUM_INFERENCE_STEPS" \
                --policy.use_planning=false \
                --env.type=robotwin \
                --env.task="$task" \
                --env.robotwin_root="$ROBOTWIN_ROOT" \
                --env.task_config=demo_randomized \
                --env.instruction_type=unseen \
                --eval.batch_size=1 \
                --eval.n_episodes="$EVAL_N_EPISODES" \
                --eval.max_episodes_rendered=0 \
                --output_dir="$task_output" \
                --seed="$SEED" 2>&1 | tee "$task_output/log.txt"
            task_exit=${PIPESTATUS[0]}
            set -e
            if [[ "$task_exit" -ne 0 ]]; then
                echo "Task $task failed with exit $task_exit" >&2
                failures=$((failures + 1))
            fi
        done
        "$PYTHON_BIN" scripts/summarize_robotwin_bc_eval.py \
            --eval_root "$eval_root" --tasks "$TASKS" \
            --episodes "$EVAL_N_EPISODES" --shard "$SHARD" || failures=$((failures + 1))
        if [[ "$failures" -ne 0 ]]; then
            echo "Eval shard $SHARD finished with $failures failure(s)" >&2
            exit 1
        fi
        ;;

    *)
        echo "Unknown MODE=$MODE (expected orchestrate, collect, finetune, or eval)" >&2
        exit 2
        ;;
esac
