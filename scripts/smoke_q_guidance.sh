#!/bin/bash
#SBATCH -J q_guid_smoke
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G
#SBATCH -q embers
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:L40s:1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# Smoke test for Q-gradient guidance (PlanningConfig.q_guidance_scale).
#
# Two stages:
#   1. Scale sweep via scripts/debug_q_guidance.py — confirms dQ/d(action) flows
#      through the bc_post -> q_pre -> Q round-trip and actually moves the plan.
#      If a normalizer detached anywhere, every scale would report d(base)=0.
#   2. One-episode RoboTwin eval with guidance on, to exercise the full eval path.
#
#   sbatch scripts/smoke_q_guidance.sh

set -uo pipefail

REPO=/storage/project/r-agarg35-0/igeorgiev3/lerobot
FASTWAM_CKPT=/storage/project/r-agarg35-0/shared/fastwam/hf_checkpoint_robotwin
Q_CKPT=/storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/q_checkpoints_backup/qf_robotwin_ddp_20260526_220319/015000/pretrained_model
ROBOTWIN_ROOT=/storage/project/r-agarg35-0/vgiridhar6/robotwin/RoboTwin
CUROBO_SRC=${ROBOTWIN_ROOT}/envs/curobo/src

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot

export PYTHONPATH="${CUROBO_SRC}:${ROBOTWIN_ROOT}:${PYTHONPATH:-}"
export HF_HOME=/storage/project/r-agarg35-0/shared/huggingface_cache
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$REPO" || exit 1

echo "=============================================="
echo "Stage 1: guidance scale sweep (gradient flow)"
echo "=============================================="
python scripts/debug_q_guidance.py \
    --fastwam-ckpt "$FASTWAM_CKPT" --q-ckpt "$Q_CKPT" \
    --steps 1,3,5 --scales 0,0.05,0.1,0.3
echo "stage 1 exit: $?"

echo "=============================================="
echo "Stage 2: 1-episode eval with guidance enabled"
echo "=============================================="
OUTDIR="${REPO}/outputs/eval/smoke_q_guidance"
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
echo N | lerobot-eval \
    --policy.path="$FASTWAM_CKPT" \
    --policy.device=cuda \
    --env.type=robotwin \
    --env.task=adjust_bottle \
    --env.robotwin_root="$ROBOTWIN_ROOT" \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --policy.num_inference_steps=10 \
    --policy.use_planning=true \
    --policy.planning.q_checkpoint_path="$Q_CKPT" \
    --policy.planning.planner_type=bc_diffusion_mppi \
    --policy.planning.num_diffusion_steps=3 \
    --policy.planning.n_samples=32 \
    --policy.planning.n_iters=3 \
    --policy.planning.n_elites=8 \
    --policy.planning.noise_std=0.3 \
    --policy.planning.p_flip_gripper=0.0 \
    --policy.planning.temperature=1.0 \
    --policy.planning.q_guidance_scale=0.1 \
    --output_dir="$OUTDIR" \
    --seed=42 2>&1 | tee "${OUTDIR}/log.txt"
echo "stage 2 exit: ${PIPESTATUS[0]}"
echo "=== smoke test done ==="
