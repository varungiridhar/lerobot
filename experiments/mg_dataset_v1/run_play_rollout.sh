#!/bin/bash
# Run play-data rollouts for one task using its early BC checkpoint.
#
# Usage:
#   sbatch scripts/compute_rtx6000_mimicgen.sh \
#       bash experiments/mg_dataset_v1/run_play_rollout.sh <task> <ckpt_dir> [n_episodes]
#
# <task>     ∈ {Square_D0, Threading_D0, Coffee_D0}
# <ckpt_dir> path to a saved BC checkpoint (config.json + model.safetensors inside)
set -euxo pipefail

TASK="${1:?usage: $0 <task> <ckpt_dir> [n_episodes]}"
CKPT_DIR="${2:?usage: $0 <task> <ckpt_dir> [n_episodes]}"
N_EPS="${3:-200}"

# Map MG task name → dataset directory name (lowercase, no _D0 suffix).
case "$TASK" in
    Square_D0)    SHORT="square" ;;
    Threading_D0) SHORT="threading" ;;
    Coffee_D0)    SHORT="coffee" ;;
    *) echo "unknown task: $TASK"; exit 1 ;;
esac

OUTPUT_DIR="/storage/home/hcoda1/6/vgiridhar6/shared/mimicgen/lerobot_datasets/v1_q5_q3jitter_play/${SHORT}/play"

# Use lerobot-mimicgen env (has lerobot policy + processors). The play rollout
# does NOT touch mimicgen.scripts.* so robomimic-v0.3 is not required.
PYBIN="/storage/home/hcoda1/6/vgiridhar6/.conda/envs/lerobot-mimicgen/bin/python"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

cd /storage/home/hcoda1/6/vgiridhar6/forks/lerobot
"${PYBIN}" -u experiments/mg_dataset_v1/play_rollout.py \
    --task "${TASK}" \
    --ckpt "${CKPT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_episodes "${N_EPS}" \
    --start_seed 100000 \
    --episode_length 400
