#!/bin/bash
#SBATCH -J si_loop
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=64G
#SBATCH -q embers
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:H200:1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# ---- Configurable args (override via --export on sbatch) ----
FASTWAM_CKPT=${FASTWAM_CKPT:-/storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint}
Q_CKPT=${Q_CKPT:-/storage/scratch1/6/vgiridhar6/lerobot/outputs/train/2026-05-19/23-54-42_qf_libero_ddp2_bsz48_bc_h200/checkpoints/last/pretrained_model}
ORIGINAL_DATASET_REPO_ID=${ORIGINAL_DATASET_REPO_ID:-HuggingFaceVLA/libero}
ORIGINAL_DATASET_ROOT=${ORIGINAL_DATASET_ROOT:-/storage/project/r-agarg35-0/shared/lerobot-data-2}
TASK=${TASK:-libero_10}
ITERATION=${ITERATION:-0}           # which iteration this job runs (auto-incremented on chain)
MAX_ITERATIONS=${MAX_ITERATIONS:-10} # total iterations across all chained jobs
N_EPISODES=${N_EPISODES:-100}
FINETUNE_STEPS=${FINETUNE_STEPS:-200}
FINETUNE_LR=${FINETUNE_LR:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-48}
ONLINE_FRACTION=${ONLINE_FRACTION:-0.5}
GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-10.0}
PLANNER_TYPE=${PLANNER_TYPE:-bc_diffusion_mppi}
N_SAMPLES=${N_SAMPLES:-64}
N_ELITES=${N_ELITES:-16}
DIFFUSION_STEPS=${DIFFUSION_STEPS:-3}
NOISE_SMOOTH_SIGMA_T=${NOISE_SMOOTH_SIGMA_T:-}
EVAL_N_EPISODES=${EVAL_N_EPISODES:-0}
SEED=${SEED:-42}
WANDB_PROJECT=${WANDB_PROJECT:-awm}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
PARTITION=${PARTITION:-gpu-h200}   # passed through on chain; override at initial submit
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_DIR=${OUTPUT_DIR:-outputs/self_improvement/${TIMESTAMP}_si_${TASK}_n${N_EPISODES}_s${FINETUNE_STEPS}}

# ---- Environment ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot

export HF_HOME=/storage/project/r-agarg35-0/shared/huggingface_cache
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Keep wandb files on scratch to avoid home-dir quota failures.
export WANDB_DIR="$HOME/scratch/wandb"
export WANDB_CACHE_DIR="$HOME/scratch/wandb/cache"
export WANDB_ARTIFACT_DIR="$HOME/scratch/wandb/artifacts"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR"

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}" || exit 1
mkdir -p logs "$OUTPUT_DIR"

echo "=== FastWAM + Q self-improvement loop ==="
echo "  ITERATION:      $ITERATION / $MAX_ITERATIONS"
echo "  FASTWAM_CKPT:   $FASTWAM_CKPT"
echo "  Q_CKPT:         $Q_CKPT"
echo "  TASK:           $TASK"
echo "  N_EPISODES:     $N_EPISODES  (this iteration)"
echo "  FINETUNE_STEPS: $FINETUNE_STEPS  lr=$FINETUNE_LR  bsz=$BATCH_SIZE  online_frac=$ONLINE_FRACTION"
echo "  PLANNER:        $PLANNER_TYPE  n_samples=$N_SAMPLES  n_elites=$N_ELITES  diffusion_steps=$DIFFUSION_STEPS"
echo "  WANDB_PROJECT:  $WANDB_PROJECT"
echo "  OUTPUT_DIR:     $OUTPUT_DIR"
echo "=========================================="

python scripts/self_improvement_loop.py \
    --fastwam_ckpt "$FASTWAM_CKPT" \
    --q_ckpt "$Q_CKPT" \
    --original_dataset_repo_id "$ORIGINAL_DATASET_REPO_ID" \
    --original_dataset_root "$ORIGINAL_DATASET_ROOT" \
    --task "$TASK" \
    --n_iterations 1 \
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
    ${WANDB_RUN_NAME:+--wandb_run_name "$WANDB_RUN_NAME"} \
    ${NOISE_SMOOTH_SIGMA_T:+--noise_smooth_sigma_t "$NOISE_SMOOTH_SIGMA_T"} \
    2>&1 | tee -a "$OUTPUT_DIR/log.txt"

EXIT_CODE=${PIPESTATUS[0]}

# ---- Auto-chain next iteration if this one succeeded ----
NEXT_ITERATION=$(( ITERATION + 1 ))
if [ $EXIT_CODE -eq 0 ] && [ $NEXT_ITERATION -lt $MAX_ITERATIONS ]; then
    NEXT_Q_CKPT="${OUTPUT_DIR}/iter_$(printf '%03d' $ITERATION)/q_checkpoint"
    echo "Chaining iteration $NEXT_ITERATION (Q ckpt: $NEXT_Q_CKPT) ..."
    sbatch \
        -p "$PARTITION" \
        --dependency=afterok:$SLURM_JOB_ID \
        --export=ALL,ITERATION=$NEXT_ITERATION,Q_CKPT=$NEXT_Q_CKPT,OUTPUT_DIR=$OUTPUT_DIR,PARTITION=$PARTITION,MAX_ITERATIONS=$MAX_ITERATIONS,N_EPISODES=$N_EPISODES,FINETUNE_STEPS=$FINETUNE_STEPS,FINETUNE_LR=$FINETUNE_LR,BATCH_SIZE=$BATCH_SIZE,GRAD_CLIP_NORM=$GRAD_CLIP_NORM,PLANNER_TYPE=$PLANNER_TYPE,N_SAMPLES=$N_SAMPLES,N_ELITES=$N_ELITES,DIFFUSION_STEPS=$DIFFUSION_STEPS,NOISE_SMOOTH_SIGMA_T=$NOISE_SMOOTH_SIGMA_T,WANDB_PROJECT=$WANDB_PROJECT,WANDB_ENTITY=$WANDB_ENTITY,WANDB_RUN_NAME=$WANDB_RUN_NAME \
        scripts/run_self_improvement.sh
fi

exit $EXIT_CODE
