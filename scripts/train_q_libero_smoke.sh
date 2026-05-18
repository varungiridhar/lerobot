#!/bin/bash
# Smoke test for the multi-dataset Q-function run. Designed to be launched
# interactively on a single GPU, not via sbatch (see train_q_libero_smoke_sbatch.sh
# for the SBATCH wrapper). Mirrors train_q_libero_ddp_multi.sh but:
#   * single-process (no DDP, no --multi_gpu)
#   * short run (50 steps) — just enough to confirm dataloader + reward labels +
#     a handful of forward/backward passes
#   * wandb + checkpointing disabled
#   * writes to outputs/debug/ (wiped at the start of every run)
#
# What this verifies before you submit the DDP job:
#   1. draccus parses --dataset.repo_ids='[...]' and --policy.bucket_overrides='{...}'
#   2. all 5 sub-datasets resolve and pull from the Hub (first run only) into
#      HF_LEROBOT_HOME (defaults to $HF_HOME/lerobot)
#   3. MultiLeRobotDataset's fps + feature-key strict-equality check passes
#      (or fails with a clear message naming the divergent repo)
#   4. QValueLabelDataset assigns every sub-dataset to its bucket via the
#      explicit --policy.bucket_overrides map; missing entries would raise here
#   5. A few training steps run end-to-end with reward_mode=sparse
#
# Override the conda env via the CONDA_ENV env var if you keep a different one,
# e.g. one with conda-forge ffmpeg installed so torchcodec can replace pyav:
#     CONDA_ENV=lerobot-q bash scripts/train_q_libero_smoke.sh

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_DIR}/outputs/debug/qf_libero_smoke"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-lerobot}"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

cd "${REPO_DIR}"

# Sanity: make sure we're picking up the editable source tree, not a stale install.
# If this import fails, the active env's installed lerobot is masking the source —
# reinstall with `python -m pip install -e .` from this repo inside ${CONDA_ENV}.
python -c "from lerobot.policies.q_function.q_value_labels import resolve_bucket; print('resolve_bucket OK')"

# Clean previous debug output so re-runs don't trip resume logic / mix artefacts.
echo "Removing previous debug output dir: ${OUT_DIR}"
rm -rf "${OUT_DIR}"

# Single process (no --multi_gpu, --num_processes=1) but still via accelerate
# launch so the codepath through Accelerator() matches the DDP version.
accelerate launch \
    --num_processes=1 \
    --mixed_precision=bf16 \
    $(which lerobot-train) \
    --output_dir="${OUT_DIR}" \
    --job_name=qf_libero_smoke \
    --policy.type=q_function \
    --policy.push_to_hub=false \
    --policy.dino_model_name=facebook/dinov2-large \
    --policy.dim_model=1024 \
    --policy.n_heads=16 \
    --policy.dim_feedforward=4096 \
    --policy.n_decoder_layers=18 \
    --policy.image_resize_h=224 \
    --policy.image_resize_w=224 \
    --policy.reward_mode=sparse \
    --policy.step_reward=0.0 \
    --policy.terminal_bonuses='{q5: 1.0, play: 0.0}' \
    --policy.bucket_overrides='{HuggingFaceVLA/libero: q5, VarunGiridhar3/libero40_libero_object_play: play, VarunGiridhar3/libero40_libero_10_play: play, VarunGiridhar3/libero40_libero_goal_play: play, VarunGiridhar3/libero40_libero_spatial_play: play}' \
    --policy.v_min=-0.01 \
    --policy.v_max=1.01 \
    --policy.hl_gauss_sigma=0.0075 \
    --policy.use_text_conditioning=true \
    --policy.text_encoder_model=google/t5-v1_1-base \
    --policy.language_key=task \
    --policy.h=32 \
    --policy.gamma=0.99 \
    --policy.target_tau=0.005 \
    --policy.optimizer_lr=1e-4 \
    --policy.optimizer_lr_backbone=3e-5 \
    --policy.optimizer_weight_decay=1e-4 \
    --dataset.repo_ids='[HuggingFaceVLA/libero,VarunGiridhar3/libero40_libero_object_play,VarunGiridhar3/libero40_libero_10_play,VarunGiridhar3/libero40_libero_goal_play,VarunGiridhar3/libero40_libero_spatial_play]' \
    --dataset.video_backend=pyav \
    --batch_size=8 \
    --steps=50 \
    --log_freq=10 \
    --save_freq=999999 \
    --save_checkpoint=false \
    --eval_freq=0 \
    --num_workers=2 \
    --cudnn_deterministic=false \
    --wandb.enable=false

echo
echo "Smoke test finished. If you got here, all 5 verification points passed."
echo "Bucket counts logged near the start of the run show how each repo was classified."
