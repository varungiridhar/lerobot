#!/bin/bash
#SBATCH -J fastwam_q_eval
#SBATCH -A gts-agarg35-ideas_l40s
#SBATCH -N1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G
#SBATCH -q embers
#SBATCH -t 6:00:00
#SBATCH -p gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH -o slurm_out/%x-%j.out
#SBATCH -e slurm_out/%x-%j.err

# ---- Configurable args (override via --export on sbatch) ----
FASTWAM_CKPT=${FASTWAM_CKPT:-/storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint}
Q_CKPT=${Q_CKPT:-/storage/scratch1/6/vgiridhar6/lerobot/outputs/train/2026-05-19/23-54-42_qf_libero_ddp2_bsz48_bc_h200/checkpoints/last/pretrained_model}
LIBERO_TASK=${LIBERO_TASK:-libero_10}
N_EPISODES=${N_EPISODES:-20}
NOISE_STD=${NOISE_STD:-0.3}
N_SAMPLES=${N_SAMPLES:-64}
# bc_diffusion_mppi with 3 diffusion steps is the default: it matched or beat every
# noise-based variant on both LIBERO-10 and RoboTwin (95.0% vs 90.5% for noise-perturb
# mppi and 90.0% for the no-Q baseline on libero_10).
PLANNER=${PLANNER:-bc_diffusion_mppi}
TEMPERATURE=${TEMPERATURE:-1.0}
# bc_diffusion_* planners are single-pass, so N_ITERS only affects the `mppi` planner.
N_ITERS=${N_ITERS:-1}
N_ELITES=${N_ELITES:-16}  # 0 = use all N_SAMPLES as elites
# Per-dim noise: comma-separated floats, length=action_dim. If empty, uses scalar NOISE_STD.
NOISE_STD_PER_DIM=${NOISE_STD_PER_DIM:-}
# Gripper flip probability (0=disabled). Only used when NOISE_STD_PER_DIM is set.
P_FLIP_GRIPPER=${P_FLIP_GRIPPER:-0.0}
NUM_INFER_STEPS=${NUM_INFER_STEPS:-10}
# Number of diffusion steps for bc_diffusion_* planners (overrides NUM_INFER_STEPS for sampling).
# Fewer steps → more diverse candidates. Leave empty to use NUM_INFER_STEPS.
DIFFUSION_STEPS=${DIFFUSION_STEPS:-3}
# Per-sample context noise std for bc_diffusion_* planners (0=disabled).
CONTEXT_NOISE_STD=${CONTEXT_NOISE_STD:-0.0}
# Gaussian smoothing sigma along time axis for MPPI noise (0=IID, 2=smooth). Empty=disabled.
NOISE_SMOOTH_SIGMA_T=${NOISE_SMOOTH_SIGMA_T:-}
# Set USE_PLANNING=false for baseline BC-only eval (skips Q-function entirely).
USE_PLANNING=${USE_PLANNING:-true}
SEED=${SEED:-42}
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_DIR=${OUTPUT_DIR:-outputs/eval/${TIMESTAMP}_fastwam_q_${LIBERO_TASK}_${PLANNER}_std${NOISE_STD}_n${N_SAMPLES}_l40s}

# ---- Environment ----
# PATH-prepend (conda activate is unreliable in non-interactive sbatch shells).
export PATH="$HOME/.conda/envs/${CONDA_ENV:-lerobot}/bin:$PATH"
export HF_HOME=/storage/scratch1/6/vgiridhar6/hf
export HF_HUB_CACHE=/storage/project/r-agarg35-0/vgiridhar6/hf_cache
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_out

echo "=== FastWAM + Q-planning eval ==="
echo "  FASTWAM_CKPT:  $FASTWAM_CKPT"
echo "  Q_CKPT:        $Q_CKPT"
echo "  LIBERO_TASK:   $LIBERO_TASK"
echo "  N_EPISODES:    $N_EPISODES"
echo "  PLANNER:       $PLANNER  noise_std=$NOISE_STD  per_dim=${NOISE_STD_PER_DIM:-none}  p_flip_gripper=$P_FLIP_GRIPPER  n_samples=$N_SAMPLES  n_iters=$N_ITERS  n_elites=$N_ELITES  temperature=$TEMPERATURE  diffusion_steps=${DIFFUSION_STEPS:-default}  use_planning=$USE_PLANNING"
echo "  OUTPUT_DIR:    $OUTPUT_DIR"
echo "================================="

mkdir -p "$OUTPUT_DIR"

if [ "$USE_PLANNING" = "false" ]; then
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
        --policy.use_planning=false \
        --output_dir="$OUTPUT_DIR" \
        --seed="$SEED" 2>&1 | tee "$OUTPUT_DIR/log.txt"
else
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
        --policy.planning.n_elites="$N_ELITES" \
        --policy.planning.noise_std="$NOISE_STD" \
        ${NOISE_STD_PER_DIM:+--policy.planning.noise_std_per_dim="[$NOISE_STD_PER_DIM]"} \
        --policy.planning.p_flip_gripper="$P_FLIP_GRIPPER" \
        --policy.planning.temperature="$TEMPERATURE" \
        ${DIFFUSION_STEPS:+--policy.planning.num_diffusion_steps="$DIFFUSION_STEPS"} \
        --policy.planning.context_noise_std="$CONTEXT_NOISE_STD" \
        ${NOISE_SMOOTH_SIGMA_T:+--policy.planning.noise_smooth_sigma_t="$NOISE_SMOOTH_SIGMA_T"} \
        --output_dir="$OUTPUT_DIR" \
        --seed="$SEED" 2>&1 | tee "$OUTPUT_DIR/log.txt"
fi
EVAL_STATUS=${PIPESTATUS[0]}
echo "=== lerobot-eval exit: ${EVAL_STATUS} ==="
exit "${EVAL_STATUS}"
