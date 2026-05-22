#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=64G
#SBATCH -q embers
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH -p gpu-rtx6000
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# Integration test for the train-test-split + test-metric-logging refactor.
#
#   * Phase 1 — Python static checks (signatures, config fields, determinism).
#   * Phase 2 — Q smoke train with --test_split_ratio=0.0. Confirms the
#               feature is fully off when disabled (no "Test metrics" line,
#               no "[q-split]" line) and the prior code path is intact.
#   * Phase 3 — Q smoke train with --test_split_ratio=0.1 --test_freq=5
#               --test_n_batches=1 --steps=20 (no BC, no env). Confirms:
#                 * "[q-split] repo=... train_eps=... test_eps=..." prints per repo.
#                 * "Test metrics @ step N: loss=..., td_ce_loss=..., q_pred_mean=..."
#                   prints at every test_freq step.
#                 * No NaN.
#   * Phase 4 — Q smoke train with split + --eval_freq=10. Confirms the
#               test-set Q-value visualization fires: side-by-side MP4s
#               (step*_test_ep*.mp4) are written for held-out test episodes.
#
# Submit from the repo root:
#     mkdir -p logs
#     sbatch scripts/tests/test_q_test_split_sbatch.sh
#
# The conda env's `lerobot` is installed from a sibling fork (forks/qplanning).
# We override with PYTHONPATH so the freshly edited modules in this tree are
# what gets imported.

set -uo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-lerobot-q}"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}" || exit 1

REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
echo "PYTHONPATH=${PYTHONPATH}"
python -c "import lerobot; print('using lerobot from:', lerobot.__file__)"

LEROBOT_TRAIN="$(python -c 'import lerobot.scripts.lerobot_train as m; print(m.__file__)')"
echo "lerobot-train module: ${LEROBOT_TRAIN}"

OUT_BASE="${OUT_BASE:-outputs/q_test_split_tests/job_${SLURM_JOB_ID:-local}}"
mkdir -p "${OUT_BASE}"

PHASE_RESULTS=()
record() {
    PHASE_RESULTS+=("$1: $2")
}

run_phase() {
    local name="$1"; shift
    echo
    echo "================================================================"
    echo "PHASE ${name}"
    echo "================================================================"
    if "$@"; then
        echo "PHASE ${name}: PASS"
        record "${name}" "PASS"
        return 0
    else
        local rc=$?
        echo "PHASE ${name}: FAIL (exit ${rc})"
        record "${name}" "FAIL (exit ${rc})"
        return 0  # full report
    fi
}

# ----- Phase 1: Python static checks --------------------------------------------

phase1_python_checks() {
    python scripts/tests/test_q_test_split.py
}

# ----- Shared Q training args (small enough to run on a single rtx6000) --------

Q_BASE_ARGS=(
    --policy.type=q_function
    --policy.push_to_hub=false
    --policy.dino_model_name=facebook/dinov2-small
    --policy.dim_model=384
    --policy.n_heads=6
    --policy.dim_feedforward=1536
    --policy.n_decoder_layers=4
    --policy.image_resize_h=224
    --policy.image_resize_w=224
    --policy.reward_mode=sparse
    --policy.step_reward=0.0
    --policy.terminal_bonuses='{q5: 1.0, play: 0.0}'
    --policy.bucket_overrides='{HuggingFaceVLA/libero: q5, VarunGiridhar3/libero40_libero_object_play: play, VarunGiridhar3/libero40_libero_10_play: play, VarunGiridhar3/libero40_libero_goal_play: play, VarunGiridhar3/libero40_libero_spatial_play: play}'
    --policy.v_min=-0.01
    --policy.v_max=1.01
    --policy.hl_gauss_sigma=0.05
    --policy.use_text_conditioning=true
    --policy.text_encoder_model=google/t5-v1_1-base
    --policy.language_key=task
    --policy.h=25
    --policy.gamma=0.99
    --policy.target_tau=0.005
    --policy.optimizer_lr=3e-4
    --policy.optimizer_lr_backbone=9e-5
    --policy.optimizer_weight_decay=1e-4
    --policy.lr_scheduler=cosine_decay_with_warmup
    --policy.lr_warmup_steps=2
    --policy.lr_decay_steps=20
    --policy.lr_decay_min=1e-6
    --dataset.repo_ids='[HuggingFaceVLA/libero,VarunGiridhar3/libero40_libero_object_play,VarunGiridhar3/libero40_libero_10_play,VarunGiridhar3/libero40_libero_goal_play,VarunGiridhar3/libero40_libero_spatial_play]'
    --dataset.root="$HOME/scratch/hf_cache/lerobot"
    --batch_size=1
    --steps=20
    --log_freq=5
    --save_freq=1000
    --num_workers=1
    --cudnn_deterministic=false
    --wandb.enable=false
    --wandb.project=awm
)

# ----- Phase 2: split disabled (regression check) -------------------------------

phase2_split_disabled() {
    local outdir="${OUT_BASE}/phase2_split_disabled"
    local log="${OUT_BASE}/phase2.log"
    rm -rf "${outdir}"
    "$(which lerobot-train)" \
        --job_name=test_q_split_2 \
        --output_dir="${outdir}" \
        --test_split_ratio=0.0 \
        --test_freq=0 \
        "${Q_BASE_ARGS[@]}" \
        2>&1 | tee "${log}"
    local rc=${PIPESTATUS[0]}
    echo
    echo "Phase 2 assertions:"
    if [ "${rc}" -ne 0 ]; then
        echo "  [FAIL] lerobot-train exited ${rc}"
        return 1
    fi
    if grep -q "Test metrics @ step" "${log}"; then
        echo "  [FAIL] 'Test metrics @ step' fired despite test_split_ratio=0.0"
        return 1
    fi
    if grep -q "\[q-split\]" "${log}"; then
        echo "  [FAIL] '[q-split]' log line fired despite holdout disabled"
        return 1
    fi
    if ! grep -q "End of training" "${log}"; then
        echo "  [FAIL] training did not reach end-of-training"
        return 1
    fi
    echo "  [OK] no '[q-split]' / 'Test metrics' lines and training completed"
    return 0
}

# ----- Phase 3: split enabled, no env -------------------------------------------

phase3_split_enabled_no_env() {
    local outdir="${OUT_BASE}/phase3_split_enabled"
    local log="${OUT_BASE}/phase3.log"
    rm -rf "${outdir}"
    "$(which lerobot-train)" \
        --job_name=test_q_split_3 \
        --output_dir="${outdir}" \
        --test_split_ratio=0.1 \
        --test_freq=5 \
        --test_n_batches=1 \
        "${Q_BASE_ARGS[@]}" \
        2>&1 | tee "${log}"
    local rc=${PIPESTATUS[0]}
    echo
    echo "Phase 3 assertions:"
    if [ "${rc}" -ne 0 ]; then
        echo "  [FAIL] lerobot-train exited ${rc}"
        return 1
    fi
    # Expect a [q-split] line per repo (5 repos in the multi-data setup).
    local qsplit_count
    qsplit_count=$(grep -c "\[q-split\] repo=" "${log}" || true)
    if [ "${qsplit_count}" -lt 5 ]; then
        echo "  [FAIL] expected >=5 '[q-split] repo=' lines, got ${qsplit_count}"
        return 1
    fi
    echo "  [OK] ${qsplit_count} '[q-split] repo=' lines emitted"
    # Test dataloader bootstrap log
    if ! grep -q "Test dataloader enabled" "${log}"; then
        echo "  [FAIL] 'Test dataloader enabled' log line not emitted"
        return 1
    fi
    echo "  [OK] 'Test dataloader enabled' line present"
    # At test_freq=5 over steps=20 we expect ~4 logs (steps 5, 10, 15, 20).
    local test_log_count
    test_log_count=$(grep -c "Test metrics @ step" "${log}" || true)
    if [ "${test_log_count}" -lt 2 ]; then
        echo "  [FAIL] expected >=2 'Test metrics @ step' lines, got ${test_log_count}"
        return 1
    fi
    echo "  [OK] ${test_log_count} 'Test metrics @ step' lines emitted"
    # Spot-check that the first 'Test metrics' line contains a few expected keys.
    local first_line
    first_line=$(grep "Test metrics @ step" "${log}" | head -n1 || true)
    echo "  Sample: ${first_line}"
    for key in "loss=" "td_ce_loss=" "q_pred_mean=" "td_abs_error_mean="; do
        if ! echo "${first_line}" | grep -q "${key}"; then
            echo "  [FAIL] sample test metrics line missing '${key}'"
            return 1
        fi
    done
    echo "  [OK] sample line carries loss, td_ce_loss, q_pred_mean, td_abs_error_mean"
    # NaN check.
    if grep -E "Test metrics @ step.*=(nan|NaN|NAN)" "${log}" >/dev/null; then
        echo "  [FAIL] NaN observed in a Test metrics line"
        return 1
    fi
    echo "  [OK] no NaN in test metrics"
    return 0
}

# ----- Phase 4: test-set Q-value visualization ----------------------------------

phase4_test_set_qvis() {
    local outdir="${OUT_BASE}/phase4_test_qvis"
    local log="${OUT_BASE}/phase4.log"
    rm -rf "${outdir}"
    # q_vis writes to the global outputs/q_vis/ fallback when wandb is off;
    # clear it so the post-run count only reflects this phase.
    rm -rf outputs/q_vis

    "$(which lerobot-train)" \
        --job_name=test_q_split_4 \
        --output_dir="${outdir}" \
        --test_split_ratio=0.1 \
        --test_freq=10 \
        --test_n_batches=1 \
        --eval_freq=10 \
        "${Q_BASE_ARGS[@]}" \
        2>&1 | tee "${log}"
    local rc=${PIPESTATUS[0]}
    echo
    echo "Phase 4 assertions:"
    if [ "${rc}" -ne 0 ]; then
        echo "  [FAIL] lerobot-train exited ${rc}"
        return 1
    fi
    if ! grep -q "End of training" "${log}"; then
        echo "  [FAIL] training did not reach end-of-training"
        return 1
    fi
    echo "  [OK] reached end-of-training"
    # eval_freq=10 over steps=20 → test-set Q-vis fires at steps 10 and 20.
    # wandb off → videos land in the global outputs/q_vis/ fallback as
    # step<NNNNNN>_test_ep<NNNN>.mp4.
    local qvis_count
    qvis_count=$(find outputs/q_vis -name "step*_test_ep*.mp4" 2>/dev/null | wc -l)
    if [ "${qvis_count}" -lt 1 ]; then
        echo "  [FAIL] no test-set Q-vis MP4 (step*_test_ep*.mp4) under outputs/q_vis/"
        return 1
    fi
    echo "  [OK] ${qvis_count} test-set Q-vis MP4(s) written under outputs/q_vis/"
    return 0
}

# ----- driver -------------------------------------------------------------------

run_phase "1_python_checks" phase1_python_checks
run_phase "2_split_disabled" phase2_split_disabled
run_phase "3_split_enabled_no_env" phase3_split_enabled_no_env
run_phase "4_test_set_qvis" phase4_test_set_qvis

echo
echo "================================================================"
echo "SUMMARY"
echo "================================================================"
for r in "${PHASE_RESULTS[@]}"; do
    echo "  ${r}"
done

if printf '%s\n' "${PHASE_RESULTS[@]}" | grep -q FAIL; then
    exit 1
fi
echo "ALL PHASES PASSED."
exit 0
