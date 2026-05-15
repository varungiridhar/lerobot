#!/bin/bash
set -euxo pipefail
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
PYBIN="/storage/home/hcoda1/6/vgiridhar6/.conda/envs/lerobot-mimicgen/bin/python"
CONDA_LIB="/storage/project/r-agarg35-0/vgiridhar6/.conda/envs/lerobot-mimicgen/lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${CONDA_LIB}/python3.10/site-packages/av.libs:${LD_LIBRARY_PATH:-}"
cd /storage/home/hcoda1/6/vgiridhar6/forks/lerobot

"${PYBIN}" -u -m lerobot.scripts.lerobot_eval \
    --policy.path=outputs/train/mimicgen_threading_d0_act_simple/checkpoints/100000/pretrained_model \
    --env.type=mimicgen \
    --env.task=Threading_D0 \
    --env.init_states_path=/storage/home/hcoda1/6/vgiridhar6/shared/mimicgen/lerobot_datasets/v1_q5_q3jitter_play/threading/bc_input/lerobot_ds/meta/init_states.pt \
    --eval.batch_size=1 \
    --eval.n_episodes=2 \
    --policy.device=cuda \
    --policy.use_q_planning=true \
    --policy.q_checkpoint_path=outputs/train/mimicgen_threading_d0_q_function_h10/checkpoints/last/pretrained_model \
    --policy.q_planning.planner_type=mppi \
    --policy.q_planning.n_samples=64 \
    --policy.q_planning.noise_std=0.1 \
    --policy.q_planning.temperature=5.0 \
    --policy.q_planning.noise_smooth_sigma_t=1.5 \
    --policy.q_planning.seed=42 \
    --seed=0 \
    --output_dir=/tmp/q_planning_smoke
