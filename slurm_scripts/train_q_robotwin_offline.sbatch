#!/bin/bash
# OFFLINE RoboTwin Q training (NO rollouts, NO sim) with optional tube negatives.
# Continues from the existing offline RoboTwin Q (= the paper's offline Q) for STEPS
# more steps on the multicam demos only. Two arms: NEG_MARGIN_WEIGHT=0 (base control)
# vs 0.25 (tube). Answers: does offline tube beat offline-base / BC, on headroom tasks?
#
# Offline training needs only the dataset (lazy video decode) — no sapien/curobo — so
# it runs in the plain `lerobot` env (qplanning code w/ negatives) and can use H100.
#SBATCH -J rtwoff
#SBATCH -A gts-agarg35-ideasci23_dgx
#SBATCH -N1 --cpus-per-gpu=8 --mem-per-gpu=96G
#SBATCH -q inferno -t 6:00:00
#SBATCH -p gpu-h100 --gres=gpu:h100:1
#SBATCH -o slurm_out/%x-%j.out -e slurm_out/%x-%j.err
set -uo pipefail

export PATH="$HOME/.conda/envs/${CONDA_ENV:-lerobot}/bin:$PATH"
export PYTHONPATH="$HOME/forks/qplanning/src:${PYTHONPATH:-}"   # ensure negatives code
export HF_HOME=/storage/scratch1/6/vgiridhar6/hf
export HF_HUB_CACHE=/storage/project/r-agarg35-0/vgiridhar6/hf_cache
export TMPDIR=/storage/scratch1/6/vgiridhar6/tmp
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p slurm_out "$TMPDIR"

ST=/storage/scratch1/6/vgiridhar6/robotwin_staged
Q_CKPT=${Q_CKPT:-$ST/q_ckpt}                 # continue from the existing offline Q
DATASET_ROOT=${DATASET_ROOT:-$ST/dataset/robotwin2.0_multicam}
ARM=${ARM:-tube}
NEG_MARGIN_WEIGHT=${NEG_MARGIN_WEIGHT:-0.25}
STEPS=${STEPS:-6000}
BATCH=${BATCH:-12}
OUTPUT_DIR=${OUTPUT_DIR:-$HOME/scratch/qplanning_rebuttal/robotwin_offline/${ARM}}
rm -rf "$OUTPUT_DIR"; mkdir -p "$(dirname "$OUTPUT_DIR")"

NEG_ARGS=()
if [ "$NEG_MARGIN_WEIGHT" != "0" ] && [ "$NEG_MARGIN_WEIGHT" != "0.0" ]; then
  # tube-only: swap + temporal off, tube sigmas (1,2) smoothed (config defaults)
  NEG_ARGS+=(--policy.neg_margin_weight="$NEG_MARGIN_WEIGHT"
             --policy.neg_use_swap=false --policy.neg_use_temporal=false)
else
  NEG_ARGS+=(--policy.neg_margin_weight=0.0)
fi

echo "=== offline RoboTwin Q: ARM=$ARM neg=$NEG_MARGIN_WEIGHT steps=$STEPS batch=$BATCH ==="
echo "  continue from: $Q_CKPT"
echo "  lerobot from:  $(python -c 'import lerobot;print(lerobot.__file__)' 2>/dev/null)"

accelerate launch --num_processes=1 --mixed_precision=bf16 "$(which lerobot-train)" \
    --job_name="qf_robotwin_offline_${ARM}" \
    --policy.path="$Q_CKPT" \
    "${NEG_ARGS[@]}" \
    --dataset.repo_id=local/robotwin2.0_multicam \
    --dataset.root="$DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --batch_size="$BATCH" \
    --steps="$STEPS" \
    --log_freq=10 \
    --save_freq="$STEPS" \
    --eval_freq=999999 \
    --test_split_ratio=0.05 \
    --num_workers=6 \
    --cudnn_deterministic=false \
    --output_dir="$OUTPUT_DIR" \
    --wandb.enable=false 2>&1 | tee -a "$OUTPUT_DIR/../${ARM}_train.log"
echo "TRAIN DONE arm=$ARM exit=${PIPESTATUS[0]}"
