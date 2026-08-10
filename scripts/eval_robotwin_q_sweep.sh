#!/bin/bash
# RoboTwin Q-checkpoint sweep: bc_diffusion_mppi with s3/s5 across Q checkpoints.
#
# Goal: find which Q-function training checkpoint plans best. Planner settings are
# held fixed at the values that produced the existing bcdiff results, so the only
# variable is the Q checkpoint.
#
# Output dirs:  outputs/eval/robotwin_bcdiff_s{3,5}_ck{015,030,045}k/<task>/
#
# Tasks that already hold a valid eval_info.json are skipped, both at submit time
# (so no empty jobs are queued) and at run time (so a preempted embers job can be
# resubmitted verbatim and pick up where it stopped).
#
# Usage:
#   ./scripts/eval_robotwin_q_sweep.sh              # submit
#   DRY_RUN=1 ./scripts/eval_robotwin_q_sweep.sh    # show plan, submit nothing
set -euo pipefail

REPO=/storage/project/r-agarg35-0/igeorgiev3/lerobot
WRAPPER_DIR="${REPO}/tmp_robotwin_wrappers"
mkdir -p "${REPO}/logs" "$WRAPPER_DIR"

FASTWAM_CKPT=/storage/project/r-agarg35-0/shared/fastwam/hf_checkpoint_robotwin
# User-owned backup copy, so the sweep stays reproducible even if the original
# training outputs are cleaned up (5k and 20k were already lost that way).
Q_CKPT_ROOT=/storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/q_checkpoints_backup/qf_robotwin_ddp_20260526_220319
ROBOTWIN_ROOT=/storage/project/r-agarg35-0/vgiridhar6/robotwin/RoboTwin
CUROBO_SRC=${ROBOTWIN_ROOT}/envs/curobo/src

DRY_RUN=${DRY_RUN:-0}
N_EPISODES=${N_EPISODES:-20}
SEED=${SEED:-42}
# SLURM routing. Defaults keep the original embers behaviour; the self-improvement
# loop overrides these to run evals on the paid inferno QoS for faster scheduling.
QOS=${QOS:-embers}
ACCOUNT=${ACCOUNT:-gts-agarg35}
# Set Q_CKPT_OVERRIDE + COND_OVERRIDE to evaluate one arbitrary checkpoint (e.g. a
# self-improvement iteration output) instead of sweeping the backup checkpoint root.
Q_CKPT_OVERRIDE=${Q_CKPT_OVERRIDE:-}
COND_OVERRIDE=${COND_OVERRIDE:-}
# Tasks per SLURM job. Smaller = shorter jobs, which survive embers preemption more
# often; the cap is the embers 50-queued-job limit.
BATCH_SIZE=${BATCH_SIZE:-10}

ALL_TASKS=(
    adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size
    click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block
    handover_mic hanging_mug lift_pot move_can_pot move_pillbottle_pad
    move_playingcard_away move_stapler_pad open_laptop open_microwave
    pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right
    place_bread_basket place_bread_skillet place_burger_fries place_can_basket
    place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup
    place_fan place_mouse_pad place_object_basket place_object_scale
    place_object_stand place_phone_stand place_shoe press_stapler
    put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object
    shake_bottle shake_bottle_horizontally stack_blocks_three stack_blocks_two
    stack_bowls_three stack_bowls_two stamp_seal turn_switch
)

# "<diffusion_steps>:<checkpoint_step>" — planner config otherwise fixed.
# Override to submit a subset, e.g.  SWEEP_SPEC="3:015000 5:045000" ./eval_robotwin_q_sweep.sh
# Submitting a condition whose tasks are already in flight would double-run them
# (the skip check only sees finished tasks), so narrow this when jobs are queued.
if [ -n "${SWEEP_SPEC:-}" ]; then
    read -r -a SWEEP <<< "$SWEEP_SPEC"
else
    SWEEP=(
        "3:015000" "3:030000" "3:045000"
        "5:015000" "5:030000" "5:045000"
    )
fi

echo "RoboTwin Q-checkpoint sweep"
echo "  tasks/condition: ${#ALL_TASKS[@]}   episodes/task: ${N_EPISODES}"
echo "  conditions:      ${#SWEEP[@]}"
echo ""

ALL_JOB_IDS=()
TOTAL_PENDING=0

for SPEC in "${SWEEP[@]}"; do
    STEPS="${SPEC%%:*}"
    CKPT="${SPEC##*:}"
    if [ -n "$Q_CKPT_OVERRIDE" ]; then
        Q_CKPT="$Q_CKPT_OVERRIDE"
        COND="${COND_OVERRIDE:-bcdiff_s${STEPS}_custom}"
    else
        CKLABEL="ck$(printf '%03d' $(( 10#$CKPT / 1000 )))k"
        COND="bcdiff_s${STEPS}_${CKLABEL}"
        Q_CKPT="${Q_CKPT_ROOT}/${CKPT}/pretrained_model"
    fi

    if [ ! -d "$Q_CKPT" ]; then
        echo "  !! ${COND}: missing Q checkpoint ${Q_CKPT} — skipping condition"
        continue
    fi

    # Only tasks without a completed eval_info.json still need running.
    PENDING=()
    for TASK in "${ALL_TASKS[@]}"; do
        [ -s "${REPO}/outputs/eval/robotwin_${COND}/${TASK}/eval_info.json" ] || PENDING+=("$TASK")
    done

    DONE=$(( ${#ALL_TASKS[@]} - ${#PENDING[@]} ))
    if [ ${#PENDING[@]} -eq 0 ]; then
        echo "  ${COND}: all ${#ALL_TASKS[@]} tasks complete — nothing to submit"
        continue
    fi
    echo "  ${COND}: ${DONE} done, ${#PENDING[@]} to run"
    TOTAL_PENDING=$(( TOTAL_PENDING + ${#PENDING[@]} ))

    N_PENDING=${#PENDING[@]}
    N_BATCHES=$(( (N_PENDING + BATCH_SIZE - 1) / BATCH_SIZE ))

    for (( b=0; b<N_BATCHES; b++ )); do
        START=$(( b * BATCH_SIZE ))
        LEN=$BATCH_SIZE
        (( START + LEN > N_PENDING )) && LEN=$(( N_PENDING - START ))
        BATCH_TASKS=("${PENDING[@]:$START:$LEN}")
        BATCH_LABEL=$(printf "%02d" $b)
        WRAPPER="${WRAPPER_DIR}/wrap_${COND}_batch${BATCH_LABEL}.sh"

        TASK_LOOP=""
        for TASK in "${BATCH_TASKS[@]}"; do
            OUTDIR="${REPO}/outputs/eval/robotwin_${COND}/${TASK}"
            TASK_LOOP+="
    if [ -s '${OUTDIR}/eval_info.json' ]; then
        echo '>>> skip ${TASK} (already complete)'
    else
        echo '>>> ${TASK}'
        mkdir -p '${OUTDIR}'
        echo N | lerobot-eval \\
            --policy.path=\"\$FASTWAM_CKPT\" \\
            --policy.device=cuda \\
            --env.type=robotwin \\
            --env.task=${TASK} \\
            --env.robotwin_root=\"\$ROBOTWIN_ROOT\" \\
            --eval.batch_size=1 \\
            --eval.n_episodes=${N_EPISODES} \\
            --policy.num_inference_steps=10 \\
            --policy.use_planning=true \\
            --policy.planning.q_checkpoint_path=\"\$Q_CKPT\" \\
            --policy.planning.planner_type=bc_diffusion_mppi \\
            --policy.planning.num_diffusion_steps=${STEPS} \\
            --policy.planning.n_samples=32 \\
            --policy.planning.n_iters=3 \\
            --policy.planning.n_elites=8 \\
            --policy.planning.noise_std=0.3 \\
            --policy.planning.p_flip_gripper=0.0 \\
            --policy.planning.temperature=1.0 \\
            --output_dir='${OUTDIR}' \\
            --seed=${SEED} 2>&1 | tee '${OUTDIR}/log.txt'
        echo '>>> Done ${TASK}'
    fi"
        done

        cat > "$WRAPPER" <<WRAP
#!/bin/bash
#SBATCH -J rt_${COND}_b${BATCH_LABEL}
#SBATCH -A ${ACCOUNT}
#SBATCH -N1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G
#SBATCH -q ${QOS}
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:L40s:1
#SBATCH -o ${REPO}/logs/%j.out
#SBATCH -e ${REPO}/logs/%j.err

FASTWAM_CKPT=${FASTWAM_CKPT}
Q_CKPT=${Q_CKPT}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT}
CUROBO_SRC=${CUROBO_SRC}

export PYTHONPATH="\${CUROBO_SRC}:\${ROBOTWIN_ROOT}:\${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST="8.9"
export HF_HOME=/storage/project/r-agarg35-0/shared/huggingface_cache
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd ${REPO}

echo "=== ${COND} batch ${BATCH_LABEL}: ${BATCH_TASKS[*]} ==="
echo "=== Q: \${Q_CKPT} ==="
${TASK_LOOP}
echo "=== Batch ${BATCH_LABEL} complete ==="
WRAP
        chmod +x "$WRAPPER"

        if [ "$DRY_RUN" = "1" ]; then
            echo "    [dry-run] ${COND} batch${BATCH_LABEL} (${#BATCH_TASKS[@]} tasks): ${BATCH_TASKS[*]}"
        else
            JOB_ID=$(sbatch --parsable "$WRAPPER")
            ALL_JOB_IDS+=("$JOB_ID")
            echo "    [${COND} batch${BATCH_LABEL}] job ${JOB_ID} (${#BATCH_TASKS[@]} tasks)"
        fi
    done
done

echo ""
echo "Task-runs pending: ${TOTAL_PENDING}"
if [ "$DRY_RUN" = "1" ]; then
    echo "Dry run — nothing submitted."
else
    echo "Submitted ${#ALL_JOB_IDS[@]} jobs: ${ALL_JOB_IDS[*]:-none}"
    echo "Monitor:  squeue -u \$USER"
    echo "Summarize: python scripts/summarize_eval.py 'outputs/eval/robotwin_bcdiff_s*_ck*' --table"
fi
