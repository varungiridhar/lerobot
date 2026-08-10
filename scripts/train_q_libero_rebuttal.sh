#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=64G
#SBATCH -q embers
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:h200:2
#SBATCH -p gpu-h200
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vgiridhar6@gatech.edu

# Stage-2 rebuttal Q-function retrains: state-conditioned negatives.
# Mirrors train_q_libero_ddp_multi_bc_and_play.sh; adds the ranking-margin
# negative losses, bucket-balanced sampling, and frozen-BC action stats.
#
# Three run variants, selected via VARIANT={r1,r1s,r2} at submit time:
#   r1   BC data only + tube negatives (smoothed sigma-scaled siblings, margin loss)
#   r1s  r1 + swap (wrong-chunk) + time-reversed negatives
#   r2   r1s recipe + 4 play datasets, bucket-balanced 2:1, frozen BC action stats
#
# Submit from the repo root (mkdir -p logs), embers QOS + babysit auto-resume:
#   sbatch --export=ALL,VARIANT=r1,CONDA_ENV=lerobot scripts/train_q_libero_rebuttal.sh
# Resume (new unique wandb id per segment):
#   sbatch --export=ALL,VARIANT=r1,CONDA_ENV=lerobot,RESUME=1,WANDB_RUN_ID=<id> scripts/train_q_libero_rebuttal.sh

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-lerobot}"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
# Writable caches: scratch HF_HOME for datasets locks; user hub cache has dinov2+t5
# (the r-agarg35-0/shared cache is read-only — datasets FileLock fails there).
export HF_HOME=${HF_HOME:-/storage/scratch1/6/vgiridhar6/hf}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/storage/project/r-agarg35-0/vgiridhar6/hf_cache}
export WANDB_DIR="$HOME/scratch/wandb"
export WANDB_CACHE_DIR="$HOME/scratch/wandb/cache"
export WANDB_ARTIFACT_DIR="$HOME/scratch/wandb/artifacts"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR"

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}" || exit 1

VARIANT=${VARIANT:?set VARIANT=r1|r1s|r2}
STEPS=${STEPS:-12000}
BATCH_SIZE=${BATCH_SIZE:-48}
EVAL_FREQ=${EVAL_FREQ:-2000}
SAVE_FREQ=${SAVE_FREQ:-1000}
JOB_NAME="qf_rebuttal_${VARIANT}_h32_bsz${BATCH_SIZE}"
OUTPUT_DIR="$HOME/scratch/qplanning_rebuttal/train/${JOB_NAME}"

# SMOKE=1: tiny single-GPU run into outputs/debug/ exercising every new code
# path (negatives margin, per-bucket metrics, balanced sampling, frozen stats,
# reworked q_vis at eval_freq).
if [[ -n "${SMOKE:-}" ]]; then
    STEPS=120; BATCH_SIZE=8; EVAL_FREQ=50; SAVE_FREQ=60
    JOB_NAME="qf_rebuttal_smoke_${VARIANT}"
    OUTPUT_DIR="outputs/debug/qf_rebuttal_smoke_${VARIANT}"
fi

NUM_GPUS=${SLURM_GPUS_ON_NODE:-2}
LAUNCH=(accelerate launch --num_processes="$NUM_GPUS" --mixed_precision=bf16 --main_process_port=placeholder)
if (( NUM_GPUS > 1 )); then
    LAUNCH+=(--multi_gpu)
fi

BC_REPO="HuggingFaceVLA/libero"
case "$VARIANT" in
  r1)
    REPO_IDS="[$BC_REPO]"
    BUCKETS="{$BC_REPO: q5}"
    BONUSES="{q5: 1.0}"
    NEG_SWAP=false; NEG_TEMPORAL=false
    EXTRA_ARGS=()
    ;;
  r1s)
    REPO_IDS="[$BC_REPO]"
    BUCKETS="{$BC_REPO: q5}"
    BONUSES="{q5: 1.0}"
    NEG_SWAP=true; NEG_TEMPORAL=true
    EXTRA_ARGS=()
    ;;
  r2)
    REPO_IDS="[$BC_REPO,VarunGiridhar3/libero40_libero_object_play,VarunGiridhar3/libero40_libero_10_play,VarunGiridhar3/libero40_libero_goal_play,VarunGiridhar3/libero40_libero_spatial_play]"
    BUCKETS="{$BC_REPO: q5, VarunGiridhar3/libero40_libero_object_play: play, VarunGiridhar3/libero40_libero_10_play: play, VarunGiridhar3/libero40_libero_goal_play: play, VarunGiridhar3/libero40_libero_spatial_play: play}"
    BONUSES="{q5: 1.0, play: 0.0}"
    NEG_SWAP=true; NEG_TEMPORAL=true
    EXTRA_ARGS=(
      "--policy.bucket_sample_weights={q5: 2.0}"
      "--policy.action_stats_repo_id=$BC_REPO"
    )
    ;;
  *) echo "unknown VARIANT=$VARIANT"; exit 1 ;;
esac

MASTER_PORT=$(( 20000 + ${SLURM_JOB_ID:-$RANDOM} % 10000 ))
LAUNCH=("${LAUNCH[@]/placeholder/$MASTER_PORT}")
echo "variant=$VARIANT  steps=$STEPS  bsz=$BATCH_SIZE  gpus=$NUM_GPUS  output=$OUTPUT_DIR  master_port=$MASTER_PORT"

source scripts/hang_watchdog.sh

if [[ -n "${RESUME:-}" ]]; then
    # lerobot appends a timestamp suffix to output_dir when the base path already
    # exists, so checkpoints may live in "<OUTPUT_DIR>" OR "<OUTPUT_DIR>_<ts>".
    # Pick the newest matching dir that actually has checkpoints/last.
    RESUME_DIR=""
    for d in $(ls -dt "${OUTPUT_DIR}" "${OUTPUT_DIR}"_* 2>/dev/null); do
        if [[ -e "${d}/checkpoints/last/pretrained_model/train_config.json" ]]; then
            RESUME_DIR="$d"; break
        fi
    done
    if [[ -z "$RESUME_DIR" ]]; then
        echo "RESUME: no checkpoints/last found under ${OUTPUT_DIR}* — aborting"; exit 1
    fi
    echo "RESUME: resuming from $RESUME_DIR"
    # VIDEO_BACKEND override: play-dataset MP4s can trip torchcodec
    # ("Could not push packet to decoder"); pyav is more tolerant.
    RESUME_VB=()
    if [[ -n "${VIDEO_BACKEND:-}" ]]; then RESUME_VB=("--dataset.video_backend=$VIDEO_BACKEND"); fi
    run_with_hang_watchdog "${LAUNCH[@]}" \
        $(which lerobot-train) \
        --config_path="$RESUME_DIR/checkpoints/last/pretrained_model/train_config.json" \
        --resume=true \
        "${RESUME_VB[@]}" \
        --wandb.run_id="${WANDB_RUN_ID:?set WANDB_RUN_ID for resume segments}"
    exit $?
fi

run_with_hang_watchdog "${LAUNCH[@]}" \
    $(which lerobot-train) \
    --job_name="$JOB_NAME" \
    --output_dir="$OUTPUT_DIR" \
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
    --policy.terminal_bonuses="$BONUSES" \
    --policy.bucket_overrides="$BUCKETS" \
    --policy.v_min=-0.01 \
    --policy.v_max=1.01 \
    --policy.hl_gauss_sigma=0.0075 \
    --policy.use_text_conditioning=true \
    --policy.text_encoder_model=google/t5-v1_1-base \
    --policy.language_key=task \
    --policy.h=32 \
    --policy.gamma=0.99 \
    --policy.target_tau=0.005 \
    --policy.neg_margin_weight=${NEG_MARGIN_WEIGHT:-0.25} \
    --policy.neg_margin_delta=0.1 \
    --policy.neg_tube_sigmas='[1.0, 2.0]' \
    --policy.neg_tube_smooth_sigma_t=2.0 \
    --policy.neg_use_swap=$NEG_SWAP \
    --policy.neg_use_temporal=$NEG_TEMPORAL \
    --policy.optimizer_lr=3e-4 \
    --policy.optimizer_lr_backbone=9e-5 \
    --policy.optimizer_weight_decay=1e-4 \
    --policy.lr_scheduler=cosine_decay_with_warmup \
    --policy.lr_warmup_steps=2000 \
    --policy.lr_decay_steps=$STEPS \
    --policy.lr_decay_min=1e-6 \
    --dataset.repo_ids="$REPO_IDS" \
    --dataset.root="$HOME/scratch/hf_cache/lerobot" \
    --dataset.video_backend=${VIDEO_BACKEND:-torchcodec} \
    --batch_size=$BATCH_SIZE \
    --steps=$STEPS \
    --log_freq=50 \
    --save_freq=$SAVE_FREQ \
    --eval_freq=$EVAL_FREQ \
    --test_split_ratio=0.1 \
    --test_freq=50 \
    --test_n_batches=1 \
    --num_workers=${NUM_WORKERS:-6} \
    --cudnn_deterministic=false \
    --wandb.enable=true \
    --wandb.entity=pair-diffusion \
    --wandb.project=awm \
    --wandb.disable_artifact=true \
    "${EXTRA_ARGS[@]}"
exit $?
