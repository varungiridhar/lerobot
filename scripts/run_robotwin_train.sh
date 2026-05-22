#!/bin/bash
# Train (or smoke-train) the FastWAM policy on RoboTwin data.
#
# The released RoboTwin checkpoint is a ~6B-param model (5B video DiT + 1B action
# DiT). To fit a single l40s (48 GB) the smoke recipe FREEZES the video DiT and
# enables gradient checkpointing — only the action DiT + proprio encoder train.
# Full-scale training (unfreezing the video DiT) needs a multi-GPU allocation.
#
# Submit:  sbatch scripts/run_robotwin_train.sh
# Override:  STEPS=2000 sbatch scripts/run_robotwin_train.sh
#SBATCH --job-name=robotwin-train
#SBATCH --account=gts-agarg35
#SBATCH -N1
#SBATCH --gres=gpu:L40s:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=80G
#SBATCH -q embers
#SBATCH -t 2:00:00
#SBATCH --output=slurm_out/Report-%j.out
set -uxo pipefail

WORKTREE=/storage/home/hcoda1/6/vgiridhar6/forks/lerobot-robotwin
DATASET_ROOT=/storage/home/hcoda1/6/vgiridhar6/r-agarg35-0/robotwin/dataset/robotwin2.0_concat
export ROBOTWIN_ROOT=/storage/home/hcoda1/6/vgiridhar6/r-agarg35-0/robotwin/RoboTwin
export HF_HOME=/storage/home/hcoda1/6/vgiridhar6/r-agarg35-0/hf_cache
export TMPDIR=/storage/project/r-agarg35-0/vgiridhar6/robotwin/tmp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

module load anaconda3/2022.05.0.1
module load cuda/12.6.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /storage/project/r-agarg35-0/vgiridhar6/.conda/envs/lerobot-robotwin

cd "$WORKTREE"
lerobot-train \
    --dataset.repo_id=local/robotwin2.0_concat \
    --dataset.root="$DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --policy.type=fastwam \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --policy.n_obs_steps=1 \
    --policy.chunk_size=32 \
    --policy.n_action_steps=24 \
    --policy.num_cameras=1 \
    --policy.image_size="[384, 320]" \
    --policy.action_dim=14 \
    --policy.state_dim=14 \
    --policy.max_action_dim=14 \
    --policy.max_state_dim=14 \
    --policy.video_dit_action_conditioned=false \
    --policy.freeze_video_dit=true \
    --policy.video_dit_use_gradient_checkpointing=true \
    --policy.action_dit_use_gradient_checkpointing=true \
    --policy.mot_checkpoint_mixed_attn=true \
    --policy.dtype=bfloat16 \
    --batch_size=1 \
    --steps="${STEPS:-10}" \
    --log_freq=1 \
    --save_freq="${SAVE_FREQ:-10}" \
    --eval_freq=0 \
    --num_workers=2 \
    --output_dir="${OUTPUT_DIR:-/storage/home/hcoda1/6/vgiridhar6/r-agarg35-0/robotwin/outputs/train/robotwin_fastwam_smoke}" \
    --job_name=robotwin_fastwam_smoke \
    --wandb.enable=false
echo "TRAIN DONE"
