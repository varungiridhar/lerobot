#!/bin/bash
#SBATCH -J robotwin-sanity
#SBATCH -A gts-agarg35-ideas_l40s
#SBATCH -N1 --cpus-per-gpu=6 --mem-per-gpu=64G
#SBATCH -q embers -t 1:00:00
#SBATCH -p gpu-l40s --gres=gpu:l40s:1
#SBATCH -o slurm_out/%x-%j.out -e slurm_out/%x-%j.err
set -uo pipefail
WT=/storage/home/hcoda1/6/vgiridhar6/forks/lerobot/.claude/worktrees/agent-a293816e2275037d3
B=/storage/cedar/cedar0/cedarp-agarg35-0/robotwin
PY=/storage/home/hcoda1/6/vgiridhar6/.conda/envs/lerobot-robotwin/bin/python
export PYTHONPATH="$WT/src:$B/RoboTwin/envs/curobo/src:$B/RoboTwin:${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST="8.9" MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME=/storage/scratch1/6/vgiridhar6/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$WT"; mkdir -p slurm_out
FASTWAM=$B/checkpoints/fastwam_robotwin_lerobot
Q=$B/outputs/train/qf_robotwin_ddp_20260522_203525/checkpoints/last/pretrained_model
TASK=${TASK:-beat_block_hammer}
OUT=${OUT:-$HOME/scratch/qplanning_rebuttal/robotwin/sanity}
rm -rf "$OUT"; mkdir -p "$OUT"
echo "lerobot from: $($PY -c 'import lerobot;print(lerobot.__file__)' 2>/dev/null)"
echo "=== RoboTwin Q-planning sanity: task=$TASK ==="
$PY -m lerobot.scripts.lerobot_eval \
  --policy.path="$FASTWAM" --policy.device=cuda \
  --env.type=robotwin --env.task="$TASK" --env.robotwin_root="$B/RoboTwin" \
  --eval.batch_size=1 --eval.n_episodes=2 \
  --policy.num_inference_steps=10 \
  --policy.use_planning=true \
  --policy.planning.q_checkpoint_path="$Q" \
  --policy.planning.planner_type=bc_diffusion_mppi \
  --policy.planning.n_samples=16 --policy.planning.n_elites=8 \
  --policy.planning.num_diffusion_steps=3 --policy.planning.temperature=1.0 \
  --output_dir="$OUT" 2>&1 | tee "$OUT/log.txt"
echo "EXIT=${PIPESTATUS[0]}"
