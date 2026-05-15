#!/bin/bash
# Args: <mode: baseline|planning> <n_episodes> <output_tag>
set -euxo pipefail
MODE="${1:?usage: $0 <baseline|planning> [n_episodes=100] <output_tag>}"
N_EPS="${2:-100}"
TAG="${3:?output_tag required}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
PYBIN="/storage/home/hcoda1/6/vgiridhar6/.conda/envs/lerobot-mimicgen/bin/python"
CONDA_LIB="/storage/project/r-agarg35-0/vgiridhar6/.conda/envs/lerobot-mimicgen/lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${CONDA_LIB}/python3.10/site-packages/av.libs:${LD_LIBRARY_PATH:-}"
cd /storage/home/hcoda1/6/vgiridhar6/forks/lerobot

EXTRA_FLAGS=()
if [[ "$MODE" == "planning" ]]; then
    EXTRA_FLAGS+=(
        --policy.use_planning=true
        --policy.planning.q_checkpoint_path=outputs/train/mimicgen_threading_d0_q_function_h10/checkpoints/last/pretrained_model
        --policy.planning.planner_type="${PLANNER_TYPE:-mppi}"
        --policy.planning.n_samples="${N_SAMPLES:-64}"
        --policy.planning.noise_std="${NOISE_STD:-0.1}"
        --policy.planning.temperature="${TEMPERATURE:-5.0}"
        --policy.planning.seed="${PLANNER_SEED:-42}"
    )
    [[ -n "${SMOOTH_SIGMA_T:-}" ]] && EXTRA_FLAGS+=( --policy.planning.noise_smooth_sigma_t="${SMOOTH_SIGMA_T}" )
    [[ -n "${N_ELITES:-}" ]] && EXTRA_FLAGS+=( --policy.planning.n_elites="${N_ELITES}" )
    [[ -n "${N_ITERS:-}" ]] && EXTRA_FLAGS+=( --policy.planning.n_iters="${N_ITERS}" )
    [[ -n "${CLIP_TO:-}" ]] && EXTRA_FLAGS+=( --policy.planning.clip_to="${CLIP_TO}" )
fi

OUTDIR=/storage/home/hcoda1/6/vgiridhar6/forks/lerobot/outputs/eval/q_planning_lerobot_eval/${TAG}
"${PYBIN}" -u -m lerobot.scripts.lerobot_eval \
    --policy.path=outputs/train/mimicgen_threading_d0_act_simple/checkpoints/100000/pretrained_model \
    --env.type=mimicgen \
    --env.task=Threading_D0 \
    --env.init_states_path=/storage/home/hcoda1/6/vgiridhar6/shared/mimicgen/lerobot_datasets/v1_q5_q3jitter_play/threading/bc_input/lerobot_ds/meta/init_states.pt \
    --env.render_height="${RENDER_SIZE:-256}" \
    --env.render_width="${RENDER_SIZE:-256}" \
    --eval.batch_size=1 \
    --eval.n_episodes="$N_EPS" \
    --eval.max_episodes_rendered="${MAX_RENDERED:-10}" \
    --policy.device=cuda \
    --seed="${EVAL_SEED:-0}" \
    --output_dir="$OUTDIR" \
    "${EXTRA_FLAGS[@]}"
