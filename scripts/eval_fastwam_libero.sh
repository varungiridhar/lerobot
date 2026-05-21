#!/bin/bash
#SBATCH -J fastwam_eval
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G
#SBATCH -q embers
#SBATCH -t 6:00:00
#SBATCH --gres=gpu:L40s:1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# ---- Configurable args (override via --export on sbatch) ----
FASTWAM_CKPT=${FASTWAM_CKPT:-/storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint}
LIBERO_TASK=${LIBERO_TASK:-libero_10}
N_EPISODES=${N_EPISODES:-20}
NUM_INFER_STEPS=${NUM_INFER_STEPS:-10}
SEED=${SEED:-42}
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_DIR=${OUTPUT_DIR:-outputs/eval/${TIMESTAMP}_fastwam_${LIBERO_TASK}_l40s}

# ---- Environment ----
export HF_HOME=/storage/project/r-agarg35-0/shared/huggingface_cache
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd $SLURM_SUBMIT_DIR

echo "=== FastWAM BC eval ==="
echo "  FASTWAM_CKPT:  $FASTWAM_CKPT"
echo "  LIBERO_TASK:   $LIBERO_TASK"
echo "  N_EPISODES:    $N_EPISODES"
echo "  OUTPUT_DIR:    $OUTPUT_DIR"
echo "======================="

mkdir -p "$OUTPUT_DIR"

echo N | lerobot-eval \
    --policy.path="$FASTWAM_CKPT" \
    --policy.device=cuda \
    --env.type=libero \
    --env.task="$LIBERO_TASK" \
    --env.observation_height=224 \
    --env.observation_width=224 \
    --eval.batch_size=1 \
    --eval.n_episodes="$N_EPISODES" \
    --policy.num_inference_steps="$NUM_INFER_STEPS" \
    --output_dir="$OUTPUT_DIR" \
    --seed="$SEED" 2>&1 | tee "$OUTPUT_DIR/log.txt"
