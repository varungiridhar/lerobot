#!/bin/bash
# Submit RoboTwin eval suite: baseline + MPPI std={0.1,0.2,0.3}
# 50 tasks split into 5 batches of 10 → 4 conditions × 5 batches = 20 jobs total
# Each job runs tasks sequentially; ~10 tasks × ~20 min = ~3.5h, well within 8h limit.
set -euo pipefail

REPO=/storage/project/r-agarg35-0/igeorgiev3/lerobot
WRAPPER_DIR="${REPO}/tmp_robotwin_wrappers"
mkdir -p "${REPO}/logs" "$WRAPPER_DIR"

FASTWAM_CKPT=/storage/project/r-agarg35-0/shared/fastwam/hf_checkpoint_robotwin
Q_CKPT=/storage/project/r-agarg35-0/vgiridhar6/robotwin/outputs/train/qf_robotwin_ddp_20260526_220319/checkpoints/005000/pretrained_model
ROBOTWIN_ROOT=/storage/project/r-agarg35-0/vgiridhar6/robotwin/RoboTwin
CUROBO_SRC=${ROBOTWIN_ROOT}/envs/curobo/src

# All 50 RoboTwin tasks
ALL_TASKS=(
    adjust_bottle
    beat_block_hammer
    blocks_ranking_rgb
    blocks_ranking_size
    click_alarmclock
    click_bell
    dump_bin_bigbin
    grab_roller
    handover_block
    handover_mic
    hanging_mug
    lift_pot
    move_can_pot
    move_pillbottle_pad
    move_playingcard_away
    move_stapler_pad
    open_laptop
    open_microwave
    pick_diverse_bottles
    pick_dual_bottles
    place_a2b_left
    place_a2b_right
    place_bread_basket
    place_bread_skillet
    place_burger_fries
    place_can_basket
    place_cans_plasticbox
    place_container_plate
    place_dual_shoes
    place_empty_cup
    place_fan
    place_mouse_pad
    place_object_basket
    place_object_scale
    place_object_stand
    place_phone_stand
    place_shoe
    press_stapler
    put_bottles_dustbin
    put_object_cabinet
    rotate_qrcode
    scan_object
    shake_bottle
    shake_bottle_horizontally
    stack_blocks_three
    stack_blocks_two
    stack_bowls_three
    stack_bowls_two
    stamp_seal
    turn_switch
)

N_EPISODES=20
SEED=42
BATCH_SIZE=10  # tasks per job

# Conditions: name, USE_PLANNING, NOISE_STD, NOISE_STD_PER_DIM
declare -A COND_USE_PLANNING COND_NOISE_STD COND_NOISE_PER_DIM

COND_USE_PLANNING["baseline"]=false
COND_NOISE_STD["baseline"]=0.0
COND_NOISE_PER_DIM["baseline"]=""

COND_USE_PLANNING["mppi_std03"]=true
COND_NOISE_STD["mppi_std03"]=0.3
COND_NOISE_PER_DIM["mppi_std03"]="0.3,0.3,0.3,0.3,0.3,0.3,0.0,0.3,0.3,0.3,0.3,0.3,0.3,0.0"

COND_USE_PLANNING["mppi_std02"]=true
COND_NOISE_STD["mppi_std02"]=0.2
COND_NOISE_PER_DIM["mppi_std02"]="0.2,0.2,0.2,0.2,0.2,0.2,0.0,0.2,0.2,0.2,0.2,0.2,0.2,0.0"

COND_USE_PLANNING["mppi_std01"]=true
COND_NOISE_STD["mppi_std01"]=0.1
COND_NOISE_PER_DIM["mppi_std01"]="0.1,0.1,0.1,0.1,0.1,0.1,0.0,0.1,0.1,0.1,0.1,0.1,0.1,0.0"

CONDITIONS=(baseline mppi_std03 mppi_std02 mppi_std01)
N_TASKS=${#ALL_TASKS[@]}
N_BATCHES=$(( (N_TASKS + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "Submitting RoboTwin eval suite"
echo "  ${#CONDITIONS[@]} conditions × ${N_BATCHES} batches × ${BATCH_SIZE} tasks = 20 jobs"
echo "  Total tasks: ${N_TASKS}"
echo ""

ALL_JOB_IDS=()

for COND in "${CONDITIONS[@]}"; do
    USE_PLANNING="${COND_USE_PLANNING[$COND]}"
    NOISE_STD="${COND_NOISE_STD[$COND]}"
    NOISE_PER_DIM="${COND_NOISE_PER_DIM[$COND]}"

    for (( b=0; b<N_BATCHES; b++ )); do
        START=$(( b * BATCH_SIZE ))
        END=$(( START + BATCH_SIZE ))
        (( END > N_TASKS )) && END=$N_TASKS
        BATCH_TASKS=("${ALL_TASKS[@]:$START:$((END - START))}")
        BATCH_LABEL=$(printf "%02d" $b)

        WRAPPER="${WRAPPER_DIR}/wrap_${COND}_batch${BATCH_LABEL}.sh"

        # Build task loop body
        TASK_LOOP=""
        for TASK in "${BATCH_TASKS[@]}"; do
            OUTDIR="${REPO}/outputs/eval/robotwin_${COND}_${TASK}"
            if [ "$USE_PLANNING" = "false" ]; then
                TASK_LOOP+="
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
        --policy.use_planning=false \\
        --output_dir='${OUTDIR}' \\
        --seed=${SEED} 2>&1 | tee '${OUTDIR}/log.txt'
    echo '>>> Done ${TASK}'"
            else
                PER_DIM_FLAG=""
                [ -n "$NOISE_PER_DIM" ] && PER_DIM_FLAG="--policy.planning.noise_std_per_dim=\"[${NOISE_PER_DIM}]\""
                TASK_LOOP+="
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
        --policy.planning.planner_type=mppi \\
        --policy.planning.n_samples=32 \\
        --policy.planning.n_iters=3 \\
        --policy.planning.n_elites=8 \\
        --policy.planning.noise_std=${NOISE_STD} \\
        ${PER_DIM_FLAG} \\
        --policy.planning.p_flip_gripper=0.1 \\
        --policy.planning.temperature=1.0 \\
        --policy.planning.noise_smooth_sigma_t=2.0 \\
        --output_dir='${OUTDIR}' \\
        --seed=${SEED} 2>&1 | tee '${OUTDIR}/log.txt'
    echo '>>> Done ${TASK}'"
            fi
        done

        cat > "$WRAPPER" <<WRAP
#!/bin/bash
#SBATCH -J rt_${COND}_b${BATCH_LABEL}
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G
#SBATCH -q embers
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
${TASK_LOOP}
echo "=== Batch ${BATCH_LABEL} complete ==="
WRAP
        chmod +x "$WRAPPER"

        JOB_ID=$(sbatch --parsable "$WRAPPER")
        ALL_JOB_IDS+=("$JOB_ID")
        echo "  [${COND} batch${BATCH_LABEL}] job $JOB_ID  tasks: ${BATCH_TASKS[*]}"
    done
    echo ""
done

echo "=== All ${#ALL_JOB_IDS[@]} jobs submitted ==="
echo "Job IDs: ${ALL_JOB_IDS[*]}"
echo ""
echo "Monitor: squeue -u $USER"
