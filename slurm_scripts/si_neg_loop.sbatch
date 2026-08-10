#!/bin/bash
# Self-improvement loop with controlled negative-Q paradigms.
# One job runs N_ITERATIONS iterations in a single process (amortizes the ~90s
# FastWAM load + dataset load). Per-iteration Q checkpoints are saved under
# OUTPUT_DIR/iter_NNN/q_checkpoint, so a preempted job can be resumed by
# resubmitting with Q_CKPT=<last good iter ckpt> ITERATION=<next>.
#
# Override everything via `sbatch --export=ALL,KEY=VAL,... slurm_scripts/si_neg_loop.sbatch`.
# Negatives: NEG_MARGIN_WEIGHT=0 => baseline loop; >0 => add tube/swap/trev margins.
#
# Job names are uniquely prefixed siNEG-<ARM> to avoid collision with other
# agents sharing this username.
#SBATCH -J siNEG
#SBATCH -A gts-agarg35-ideas_l40s
#SBATCH -N1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=64G
#SBATCH -q embers
#SBATCH -t 8:00:00
#SBATCH -p gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH -o slurm_out/%x-%j.out
#SBATCH -e slurm_out/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vgiridhar6@gatech.edu

set -uo pipefail

# ---- Robust environment (conda activate is unreliable in non-interactive sbatch) ----
export PATH="$HOME/.conda/envs/${CONDA_ENV:-lerobot}/bin:$PATH"
export HF_HOME=/storage/scratch1/6/vgiridhar6/hf
export HF_HUB_CACHE=/storage/project/r-agarg35-0/vgiridhar6/hf_cache
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR="$HOME/scratch/wandb"
export WANDB_CACHE_DIR="$HOME/scratch/wandb/cache"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" slurm_out

# ---- Paths / checkpoints ----
FASTWAM_CKPT=${FASTWAM_CKPT:-/storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint}
Q_CKPT=${Q_CKPT:-/storage/scratch1/6/vgiridhar6/lerobot/outputs/train/2026-05-19/23-54-42_qf_libero_ddp2_bsz48_bc_h200/checkpoints/last/pretrained_model}
ORIGINAL_DATASET_REPO_ID=${ORIGINAL_DATASET_REPO_ID:-HuggingFaceVLA/libero}
ORIGINAL_DATASET_ROOT=${ORIGINAL_DATASET_ROOT:-/storage/project/r-agarg35-0/shared/lerobot-data-2}
TASK=${TASK:-libero_10}

# ---- Loop schedule ----
ITERATION=${ITERATION:-0}
N_ITERATIONS=${N_ITERATIONS:-6}
N_EPISODES=${N_EPISODES:-30}
FINETUNE_STEPS=${FINETUNE_STEPS:-200}
FINETUNE_LR=${FINETUNE_LR:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-32}
ONLINE_FRACTION=${ONLINE_FRACTION:-0.5}
GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-10.0}
PLANNER_TYPE=${PLANNER_TYPE:-bc_diffusion_mppi}
N_SAMPLES=${N_SAMPLES:-16}
N_ELITES=${N_ELITES:-16}
DIFFUSION_STEPS=${DIFFUSION_STEPS:-3}
EVAL_N_EPISODES=${EVAL_N_EPISODES:-0}
SEED=${SEED:-42}

# ---- Negative-Q paradigm (applied during in-loop finetune) ----
NEG_MARGIN_WEIGHT=${NEG_MARGIN_WEIGHT:-0.0}
NEG_USE_SWAP=${NEG_USE_SWAP:-1}        # 1/0
NEG_USE_TEMPORAL=${NEG_USE_TEMPORAL:-1}
NEG_TUBE_SIGMAS=${NEG_TUBE_SIGMAS:-1.0,2.0}

# ---- Bookkeeping / wandb ----
ARM=${ARM:-L0}
WANDB_PROJECT=${WANDB_PROJECT:-awm}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-siNEG_${ARM}_${TASK}}
OUTPUT_DIR=${OUTPUT_DIR:-$HOME/scratch/qplanning_rebuttal/si/${ARM}_${TASK}}

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}" || exit 1
mkdir -p "$OUTPUT_DIR"

# ---- Assemble negatives flags ----
NEG_ARGS=()
if [ "$NEG_MARGIN_WEIGHT" != "0" ] && [ "$NEG_MARGIN_WEIGHT" != "0.0" ]; then
    NEG_ARGS+=(--neg_margin_weight "$NEG_MARGIN_WEIGHT" --neg_tube_sigmas "$NEG_TUBE_SIGMAS")
    if [ "$NEG_USE_SWAP" = "1" ]; then NEG_ARGS+=(--neg_use_swap); else NEG_ARGS+=(--no-neg_use_swap); fi
    if [ "$NEG_USE_TEMPORAL" = "1" ]; then NEG_ARGS+=(--neg_use_temporal); else NEG_ARGS+=(--no-neg_use_temporal); fi
fi

echo "=== siNEG self-improvement loop: ARM=$ARM ==="
echo "  Q_CKPT:          $Q_CKPT"
echo "  TASK:            $TASK    iterations=$N_ITERATIONS (start=$ITERATION)"
echo "  collect:         n_episodes=$N_EPISODES  planner=$PLANNER_TYPE  n_samples=$N_SAMPLES  diff_steps=$DIFFUSION_STEPS"
echo "  finetune:        steps=$FINETUNE_STEPS  lr=$FINETUNE_LR  bsz=$BATCH_SIZE  online_frac=$ONLINE_FRACTION"
echo "  negatives:       margin_weight=$NEG_MARGIN_WEIGHT  swap=$NEG_USE_SWAP  temporal=$NEG_USE_TEMPORAL  tube=$NEG_TUBE_SIGMAS"
echo "  NEG_ARGS:        ${NEG_ARGS[*]:-<none>}"
echo "  OUTPUT_DIR:      $OUTPUT_DIR"
echo "  node:            $(hostname)   gpu: $CUDA_VISIBLE_DEVICES"
echo "================================================"

python scripts/self_improvement_loop.py \
    --fastwam_ckpt "$FASTWAM_CKPT" \
    --q_ckpt "$Q_CKPT" \
    --original_dataset_repo_id "$ORIGINAL_DATASET_REPO_ID" \
    --original_dataset_root "$ORIGINAL_DATASET_ROOT" \
    --task "$TASK" \
    --n_iterations "$N_ITERATIONS" \
    --start_iteration "$ITERATION" \
    --n_episodes "$N_EPISODES" \
    --finetune_steps "$FINETUNE_STEPS" \
    --finetune_lr "$FINETUNE_LR" \
    --batch_size "$BATCH_SIZE" \
    --online_fraction "$ONLINE_FRACTION" \
    --grad_clip_norm "$GRAD_CLIP_NORM" \
    --planner_type "$PLANNER_TYPE" \
    --n_samples "$N_SAMPLES" \
    --n_elites "$N_ELITES" \
    --diffusion_steps "$DIFFUSION_STEPS" \
    --eval_n_episodes "$EVAL_N_EPISODES" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --wandb_project "$WANDB_PROJECT" \
    ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"} \
    --wandb_run_name "$WANDB_RUN_NAME" \
    "${NEG_ARGS[@]}" \
    2>&1 | tee -a "$OUTPUT_DIR/log.txt"

exit ${PIPESTATUS[0]}
