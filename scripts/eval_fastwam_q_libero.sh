#!/bin/bash
#SBATCH -J fastwam_q_eval
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G
#SBATCH -q inferno
#SBATCH -t 6:00:00
#SBATCH --gres=gpu:L40s:1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# ---- Configurable args (override via --export on sbatch) ----
FASTWAM_CKPT=${FASTWAM_CKPT:-/storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint}
Q_CKPT=${Q_CKPT:-/storage/scratch1/6/vgiridhar6/lerobot/outputs/train/2026-05-19/23-54-42_qf_libero_ddp2_bsz48_bc_h200/checkpoints/last/pretrained_model}
LIBERO_TASK=${LIBERO_TASK:-libero_10}
N_EPISODES=${N_EPISODES:-20}
NOISE_STD=${NOISE_STD:-0.3}
N_SAMPLES=${N_SAMPLES:-64}
PLANNER=${PLANNER:-mppi}
TEMPERATURE=${TEMPERATURE:-1.0}
N_ITERS=${N_ITERS:-1}
NUM_INFER_STEPS=${NUM_INFER_STEPS:-10}
SEED=${SEED:-42}
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_DIR=${OUTPUT_DIR:-outputs/eval/${TIMESTAMP}_fastwam_q_${LIBERO_TASK}_${PLANNER}_std${NOISE_STD}_n${N_SAMPLES}_l40s}

# ---- Environment ----
export HF_HOME=/storage/project/r-agarg35-0/shared/huggingface_cache
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd $SLURM_SUBMIT_DIR

echo "=== FastWAM + Q-planning eval ==="
echo "  FASTWAM_CKPT:  $FASTWAM_CKPT"
echo "  Q_CKPT:        $Q_CKPT"
echo "  LIBERO_TASK:   $LIBERO_TASK"
echo "  N_EPISODES:    $N_EPISODES"
echo "  PLANNER:       $PLANNER  noise_std=$NOISE_STD  n_samples=$N_SAMPLES  n_iters=$N_ITERS  temperature=$TEMPERATURE"
echo "  OUTPUT_DIR:    $OUTPUT_DIR"
echo "================================="

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
    --policy.use_planning=true \
    --policy.planning.q_checkpoint_path="$Q_CKPT" \
    --policy.planning.planner_type="$PLANNER" \
    --policy.planning.n_samples="$N_SAMPLES" \
    --policy.planning.n_iters="$N_ITERS" \
    --policy.planning.noise_std="$NOISE_STD" \
    --policy.planning.temperature="$TEMPERATURE" \
    --output_dir="$OUTPUT_DIR" \
    --seed="$SEED" 2>&1 | tee "$OUTPUT_DIR/log.txt"
