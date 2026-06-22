#!/bin/bash
# RoboTwin self-improvement loop with controlled negative-Q paradigms.
# Mirrors si_neg_loop.sbatch but for the RoboTwin (3-camera, bimanual) env:
# lerobot-robotwin conda env + curobo on PYTHONPATH + this worktree's src.
#SBATCH -J robotwin-si
#SBATCH -A gts-agarg35-ideas_l40s
#SBATCH -N1 --cpus-per-gpu=6 --mem-per-gpu=64G
#SBATCH -q embers -t 8:00:00
#SBATCH -p gpu-l40s --gres=gpu:l40s:1
#SBATCH -o slurm_out/%x-%j.out -e slurm_out/%x-%j.err
set -uo pipefail

WT=/storage/home/hcoda1/6/vgiridhar6/forks/lerobot/.claude/worktrees/agent-a293816e2275037d3
B=/storage/cedar/cedar0/cedarp-agarg35-0/robotwin
PY=/storage/home/hcoda1/6/vgiridhar6/.conda/envs/lerobot-robotwin/bin/python
export PYTHONPATH="$WT/src:$B/RoboTwin/envs/curobo/src:$B/RoboTwin:${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST="8.9" MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME=/storage/scratch1/6/vgiridhar6/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR="$HOME/scratch/wandb" WANDB_CACHE_DIR="$HOME/scratch/wandb/cache"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WT/slurm_out"
cd "$WT"

# paths
FASTWAM_CKPT=${FASTWAM_CKPT:-$B/checkpoints/fastwam_robotwin_lerobot}
Q_CKPT=${Q_CKPT:-$B/outputs/train/qf_robotwin_ddp_20260522_203525/checkpoints/last/pretrained_model}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-$B/RoboTwin}
DATASET_REPO_ID=${DATASET_REPO_ID:-robotwin2.0_multicam}
DATASET_ROOT=${DATASET_ROOT:-$B/dataset}
TASK=${TASK:-beat_block_hammer}
# loop schedule
ITERATION=${ITERATION:-0}; N_ITERATIONS=${N_ITERATIONS:-3}
N_EPISODES=${N_EPISODES:-20}; FINETUNE_STEPS=${FINETUNE_STEPS:-200}
FINETUNE_LR=${FINETUNE_LR:-1e-5}; BATCH_SIZE=${BATCH_SIZE:-16}
ONLINE_FRACTION=${ONLINE_FRACTION:-0.5}; GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-10.0}
PLANNER_TYPE=${PLANNER_TYPE:-bc_diffusion_mppi}
N_SAMPLES=${N_SAMPLES:-16}; N_ELITES=${N_ELITES:-8}; DIFFUSION_STEPS=${DIFFUSION_STEPS:-3}
EVAL_N_EPISODES=${EVAL_N_EPISODES:-0}; SEED=${SEED:-42}
# negatives
NEG_MARGIN_WEIGHT=${NEG_MARGIN_WEIGHT:-0.0}
NEG_USE_SWAP=${NEG_USE_SWAP:-1}; NEG_USE_TEMPORAL=${NEG_USE_TEMPORAL:-1}
# bookkeeping
ARM=${ARM:-L0}
WANDB_PROJECT=${WANDB_PROJECT:-awm}; WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-robotwin_${ARM}_${TASK}}
OUTPUT_DIR=${OUTPUT_DIR:-$HOME/scratch/qplanning_rebuttal/robotwin/${ARM}_${TASK}}
mkdir -p "$OUTPUT_DIR"

NEG_ARGS=()
if [ "$NEG_MARGIN_WEIGHT" != "0" ] && [ "$NEG_MARGIN_WEIGHT" != "0.0" ]; then
    NEG_ARGS+=(--neg_margin_weight "$NEG_MARGIN_WEIGHT")
    if [ "$NEG_USE_SWAP" = "1" ]; then NEG_ARGS+=(--neg_use_swap); else NEG_ARGS+=(--no-neg_use_swap); fi
    if [ "$NEG_USE_TEMPORAL" = "1" ]; then NEG_ARGS+=(--neg_use_temporal); else NEG_ARGS+=(--no-neg_use_temporal); fi
fi

echo "=== robotwin-si ARM=$ARM task=$TASK iters=$N_ITERATIONS (start=$ITERATION) ==="
echo "  lerobot from: $($PY -c 'import lerobot;print(lerobot.__file__)' 2>/dev/null)"
echo "  Q=$Q_CKPT"
echo "  neg=$NEG_MARGIN_WEIGHT swap=$NEG_USE_SWAP temporal=$NEG_USE_TEMPORAL  planner=$PLANNER_TYPE n=$N_SAMPLES"

$PY scripts/self_improvement_loop.py \
  --env_type robotwin --robotwin_root "$ROBOTWIN_ROOT" \
  --fastwam_ckpt "$FASTWAM_CKPT" --q_ckpt "$Q_CKPT" \
  --original_dataset_repo_id "$DATASET_REPO_ID" --original_dataset_root "$DATASET_ROOT" \
  --task "$TASK" --n_iterations "$N_ITERATIONS" --start_iteration "$ITERATION" \
  --n_episodes "$N_EPISODES" --finetune_steps "$FINETUNE_STEPS" --finetune_lr "$FINETUNE_LR" \
  --batch_size "$BATCH_SIZE" --online_fraction "$ONLINE_FRACTION" --grad_clip_norm "$GRAD_CLIP_NORM" \
  --planner_type "$PLANNER_TYPE" --n_samples "$N_SAMPLES" --n_elites "$N_ELITES" \
  --diffusion_steps "$DIFFUSION_STEPS" --eval_n_episodes "$EVAL_N_EPISODES" \
  --output_dir "$OUTPUT_DIR" --seed "$SEED" \
  --wandb_project "$WANDB_PROJECT" ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"} \
  --wandb_run_name "$WANDB_RUN_NAME" \
  "${NEG_ARGS[@]}" 2>&1 | tee -a "$OUTPUT_DIR/log.txt"
exit ${PIPESTATUS[0]}
