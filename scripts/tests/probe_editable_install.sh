#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=64G
#SBATCH -q embers
#SBATCH -t 0:30:00
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH -p gpu-rtx6000
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# Quick probe: confirm the editable `pip install -e .` of this tree took, so
# the production scripts (which call `$(which lerobot-train)` with NO
# PYTHONPATH override) now resolve to this tree and accept the new CLI flags
# (--test_split_ratio, --test_freq, --test_n_batches) that previously failed
# with "unrecognized arguments".
#
# Deliberately does NOT set PYTHONPATH — the whole point is to test the
# install on its own. Runs a 4-step Q smoke train (dinov2-small, no env, so
# no slow sim rollout) with the split enabled and wandb ON (offline mode, no
# auth/cloud run needed) so the WandBLogger.log_dict(mode="test") path — the
# one that crashed the production jobs — is actually exercised.
#
#     mkdir -p logs && sbatch scripts/tests/probe_editable_install.sh

set -uo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-lerobot-q}"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}" || exit 1

echo "=== resolved lerobot (no PYTHONPATH override) ==="
python -c "import lerobot; print('lerobot:', lerobot.__file__)"
LEROBOT_PATH=$(python -c "import lerobot; print(lerobot.__file__)")

OUT="outputs/probe_editable_install/job_${SLURM_JOB_ID:-local}"
LOG="${OUT}.log"
rm -rf "${OUT}"
mkdir -p "$(dirname "${LOG}")"

# All four flags below are the ones the stale install rejected.
"$(which lerobot-train)" \
    --job_name=probe_editable_install \
    --output_dir="${OUT}" \
    --policy.type=q_function \
    --policy.push_to_hub=false \
    --policy.dino_model_name=facebook/dinov2-small \
    --policy.dim_model=384 \
    --policy.n_heads=6 \
    --policy.dim_feedforward=1536 \
    --policy.n_decoder_layers=4 \
    --policy.image_resize_h=224 \
    --policy.image_resize_w=224 \
    --policy.reward_mode=sparse \
    --policy.step_reward=0.0 \
    --policy.terminal_bonuses='{q5: 1.0, play: 0.0}' \
    --policy.bucket_overrides='{HuggingFaceVLA/libero: q5, VarunGiridhar3/libero40_libero_object_play: play, VarunGiridhar3/libero40_libero_10_play: play, VarunGiridhar3/libero40_libero_goal_play: play, VarunGiridhar3/libero40_libero_spatial_play: play}' \
    --policy.v_min=-0.01 \
    --policy.v_max=1.01 \
    --policy.hl_gauss_sigma=0.05 \
    --policy.use_text_conditioning=true \
    --policy.text_encoder_model=google/t5-v1_1-base \
    --policy.language_key=task \
    --policy.h=25 \
    --policy.gamma=0.99 \
    --policy.target_tau=0.005 \
    --policy.optimizer_lr=3e-4 \
    --policy.optimizer_lr_backbone=9e-5 \
    --policy.optimizer_weight_decay=1e-4 \
    --policy.lr_scheduler=cosine_decay_with_warmup \
    --policy.lr_warmup_steps=2 \
    --policy.lr_decay_steps=4 \
    --policy.lr_decay_min=1e-6 \
    --dataset.repo_ids='[HuggingFaceVLA/libero,VarunGiridhar3/libero40_libero_object_play,VarunGiridhar3/libero40_libero_10_play,VarunGiridhar3/libero40_libero_goal_play,VarunGiridhar3/libero40_libero_spatial_play]' \
    --dataset.root="$HOME/scratch/hf_cache/lerobot" \
    --test_split_ratio=0.1 \
    --test_freq=2 \
    --test_n_batches=1 \
    --batch_size=1 \
    --steps=4 \
    --log_freq=2 \
    --save_freq=1000 \
    --eval_freq=0 \
    --num_workers=1 \
    --cudnn_deterministic=false \
    --wandb.enable=true \
    --wandb.mode=offline \
    --wandb.project=awm \
    2>&1 | tee "${LOG}"
RC=${PIPESTATUS[0]}

echo
echo "================================================================"
echo "PROBE ASSERTIONS"
echo "================================================================"
fail=0

case "${LEROBOT_PATH}" in
    *forks/delete_me/lerobot/src/lerobot/*)
        echo "  [OK] lerobot resolves to this tree" ;;
    *)
        echo "  [FAIL] lerobot resolves elsewhere: ${LEROBOT_PATH}"; fail=1 ;;
esac

if grep -q "unrecognized arguments" "${LOG}"; then
    echo "  [FAIL] 'unrecognized arguments' — stale package still in use"
    fail=1
else
    echo "  [OK] no 'unrecognized arguments' error"
fi

if [ "${RC}" -ne 0 ]; then
    echo "  [FAIL] lerobot-train exited ${RC}"
    fail=1
else
    echo "  [OK] lerobot-train exited 0"
fi

qsplit=$(grep -c "\[q-split\] repo=" "${LOG}" || true)
if [ "${qsplit}" -ge 5 ]; then
    echo "  [OK] ${qsplit} '[q-split] repo=' lines emitted"
else
    echo "  [FAIL] expected >=5 '[q-split] repo=' lines, got ${qsplit}"; fail=1
fi

tmetrics=$(grep -c "Test metrics @ step" "${LOG}" || true)
if [ "${tmetrics}" -ge 1 ]; then
    echo "  [OK] ${tmetrics} 'Test metrics @ step' line(s) emitted"
    grep "Test metrics @ step" "${LOG}" | head -n1 | sed 's/^/  sample: /'
else
    echo "  [FAIL] no 'Test metrics @ step' line"; fail=1
fi

if grep -q "End of training" "${LOG}"; then
    echo "  [OK] reached 'End of training'"
else
    echo "  [FAIL] did not reach 'End of training'"; fail=1
fi

echo
if [ "${fail}" -eq 0 ]; then
    echo "PROBE PASSED — production scripts will work without a PYTHONPATH override."
    exit 0
fi
echo "PROBE FAILED — inspect ${LOG}"
exit 1
