#!/bin/bash
#SBATCH -J si_loop
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G
#SBATCH -q embers
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:L40s:1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# ---- Configurable args (override via --export on sbatch) ----
FASTWAM_CKPT=${FASTWAM_CKPT:-/storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint}
Q_CKPT=${Q_CKPT:-/storage/scratch1/6/vgiridhar6/lerobot/outputs/train/2026-05-19/23-54-42_qf_libero_ddp2_bsz48_bc_h200/checkpoints/last/pretrained_model}
ORIGINAL_DATASET_REPO_ID=${ORIGINAL_DATASET_REPO_ID:-HuggingFaceVLA/libero}
ORIGINAL_DATASET_ROOT=${ORIGINAL_DATASET_ROOT:-/storage/project/r-agarg35-0/shared/lerobot-data-2}
TASK=${TASK:-libero_10}
N_ITERATIONS=${N_ITERATIONS:-5}
N_EPISODES=${N_EPISODES:-20}
FINETUNE_STEPS=${FINETUNE_STEPS:-200}
FINETUNE_LR=${FINETUNE_LR:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-32}
ONLINE_FRACTION=${ONLINE_FRACTION:-0.5}
PLANNER_TYPE=${PLANNER_TYPE:-bc_diffusion_mppi}
N_SAMPLES=${N_SAMPLES:-16}
N_ELITES=${N_ELITES:-16}
DIFFUSION_STEPS=${DIFFUSION_STEPS:-3}
SEED=${SEED:-42}
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_DIR=${OUTPUT_DIR:-outputs/self_improvement/${TIMESTAMP}_si_${TASK}_n${N_EPISODES}_s${FINETUNE_STEPS}}

# ---- Environment ----
export HF_HOME=/storage/project/r-agarg35-0/shared/huggingface_cache
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd $SLURM_SUBMIT_DIR
mkdir -p logs "$OUTPUT_DIR"

echo "=== FastWAM + Q self-improvement loop ==="
echo "  FASTWAM_CKPT:   $FASTWAM_CKPT"
echo "  Q_CKPT:         $Q_CKPT"
echo "  TASK:           $TASK"
echo "  N_ITERATIONS:   $N_ITERATIONS"
echo "  N_EPISODES:     $N_EPISODES  (per iteration)"
echo "  FINETUNE_STEPS: $FINETUNE_STEPS  lr=$FINETUNE_LR  bsz=$BATCH_SIZE  online_frac=$ONLINE_FRACTION"
echo "  PLANNER:        $PLANNER_TYPE  n_samples=$N_SAMPLES  n_elites=$N_ELITES  diffusion_steps=$DIFFUSION_STEPS"
echo "  OUTPUT_DIR:     $OUTPUT_DIR"
echo "=========================================="

python scripts/self_improvement_loop.py \
    --fastwam_ckpt "$FASTWAM_CKPT" \
    --q_ckpt "$Q_CKPT" \
    --original_dataset_repo_id "$ORIGINAL_DATASET_REPO_ID" \
    --original_dataset_root "$ORIGINAL_DATASET_ROOT" \
    --task "$TASK" \
    --n_iterations "$N_ITERATIONS" \
    --n_episodes "$N_EPISODES" \
    --finetune_steps "$FINETUNE_STEPS" \
    --finetune_lr "$FINETUNE_LR" \
    --batch_size "$BATCH_SIZE" \
    --online_fraction "$ONLINE_FRACTION" \
    --planner_type "$PLANNER_TYPE" \
    --n_samples "$N_SAMPLES" \
    --n_elites "$N_ELITES" \
    --diffusion_steps "$DIFFUSION_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    2>&1 | tee "$OUTPUT_DIR/log.txt"
