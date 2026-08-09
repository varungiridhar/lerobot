#!/bin/bash
#SBATCH -J si_rt
#SBATCH -A gts-agarg35-ideasci23_dgx
#SBATCH -N1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=64G
#SBATCH -q inferno
#SBATCH -t 8:00:00
#SBATCH -p gpu-h200
# Generic gres: the ideasci23_dgx account cannot request the typed gpu:h200:1, and
# the gpu-h200 partition holds only H200 nodes, so gpu:1 there is still an H200.
#SBATCH --gres=gpu:1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# RoboTwin self-improvement loop — mirrors scripts/run_self_improvement.sh (LIBERO).
#
# Each job runs ONE iteration: collect on-policy episodes with FastWAM + Q planning,
# fine-tune Q on a mix of those and the original demos, then chain the next iteration
# via --dependency=afterok. After every Q fine-tune it also submits a full 50-task
# RoboTwin eval of the new checkpoint (separate L40s jobs, so training is not blocked).
#
# Runs on inferno (paid, non-preemptable). That matters here: the previous LIBERO loop
# died permanently at iteration 6 because its afterok dependency was preempted on
# embers, and afterok treats preemption as failure.
#
#   sbatch scripts/run_self_improvement_robotwin.sh
#   DRY_RUN=1 bash scripts/run_self_improvement_robotwin.sh   # print plan, submit nothing

# ---- Configurable args (override via --export on sbatch) ----
FASTWAM_CKPT=${FASTWAM_CKPT:-/storage/project/r-agarg35-0/shared/fastwam/hf_checkpoint_robotwin}
# Q checkpoint 45k: best of the 15k/30k/45k sweep (84.0% vs 82.8% no-Q baseline).
Q_CKPT=${Q_CKPT:-/storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/q_checkpoints_backup/qf_robotwin_ddp_20260526_220319/045000/pretrained_model}
# Full RoboTwin 2.0 dataset the Q function was trained on: 27500 episodes / 6.1M
# frames, with the 3 camera keys Q expects (cam_high, cam_left_wrist,
# cam_right_wrist). The path recorded in the checkpoint's train_config.json pointed
# into vgiridhar6's scratch and has been purged; this shared copy replaces it.
# Do NOT substitute qplanning_rebuttal_handoff/data/robotwin2.0_multicam — that is a
# 300-episode subset, ~1% of the data, and would silently gut the offline half of
# the finetune mix.
ORIGINAL_DATASET_REPO_ID=${ORIGINAL_DATASET_REPO_ID:-local/robotwin2.0}
ORIGINAL_DATASET_ROOT=${ORIGINAL_DATASET_ROOT:-/storage/project/r-agarg35-0/shared/robotwin2.0}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-/storage/project/r-agarg35-0/vgiridhar6/robotwin/RoboTwin}

ITERATION=${ITERATION:-0}
MAX_ITERATIONS=${MAX_ITERATIONS:-10}
N_EPISODES=${N_EPISODES:-100}          # on-policy episodes collected per iteration
FINETUNE_STEPS=${FINETUNE_STEPS:-200}
FINETUNE_LR=${FINETUNE_LR:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-48}
ONLINE_FRACTION=${ONLINE_FRACTION:-0.5}
GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-10.0}
PLANNER_TYPE=${PLANNER_TYPE:-bc_diffusion_mppi}
N_SAMPLES=${N_SAMPLES:-32}
N_ELITES=${N_ELITES:-8}
DIFFUSION_STEPS=${DIFFUSION_STEPS:-3}
SEED=${SEED:-42}
WANDB_PROJECT=${WANDB_PROJECT:-awm}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-si_robotwin_ck045k_s3}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/self_improvement/si_robotwin_ck045k_s3}

# SLURM routing: H200 for the training iteration, L40s for the 50-task evals.
TRAIN_ACCOUNT=${TRAIN_ACCOUNT:-gts-agarg35-ideasci23_dgx}
TRAIN_PARTITION=${TRAIN_PARTITION:-gpu-h200}
EVAL_ACCOUNT=${EVAL_ACCOUNT:-gts-agarg35-ideas_l40s}
QOS=${QOS:-inferno}
# Merged eval+collection. Each task runs N_EVAL_PER_TASK held-out episodes (seeds fixed
# across iterations, never trained on -> the comparable metric) followed by
# N_TRAIN_PER_TASK episodes (advancing seeds, saved for finetuning). One rollout pass
# serves both purposes, so the separate 50-task eval sweep is no longer needed.
N_EVAL_PER_TASK=${N_EVAL_PER_TASK:-10}
N_TRAIN_PER_TASK=${N_TRAIN_PER_TASK:-10}
# Separate post-finetune eval sweep. Now off by default: the next iteration's merged
# rollout evaluates this checkpoint on held-out seeds anyway, so running it too would
# duplicate ~1000 episodes per iteration.
EVAL_AFTER_TRAIN=${EVAL_AFTER_TRAIN:-0}
EVAL_EPISODES=${EVAL_EPISODES:-20}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-3}   # tasks per eval job

DRY_RUN=${DRY_RUN:-0}

# All 50 RoboTwin tasks — on-policy collection spreads N_EPISODES across them.
TASKS="adjust_bottle,beat_block_hammer,blocks_ranking_rgb,blocks_ranking_size,\
click_alarmclock,click_bell,dump_bin_bigbin,grab_roller,handover_block,handover_mic,\
hanging_mug,lift_pot,move_can_pot,move_pillbottle_pad,move_playingcard_away,\
move_stapler_pad,open_laptop,open_microwave,pick_diverse_bottles,pick_dual_bottles,\
place_a2b_left,place_a2b_right,place_bread_basket,place_bread_skillet,place_burger_fries,\
place_can_basket,place_cans_plasticbox,place_container_plate,place_dual_shoes,\
place_empty_cup,place_fan,place_mouse_pad,place_object_basket,place_object_scale,\
place_object_stand,place_phone_stand,place_shoe,press_stapler,put_bottles_dustbin,\
put_object_cabinet,rotate_qrcode,scan_object,shake_bottle,shake_bottle_horizontally,\
stack_blocks_three,stack_blocks_two,stack_bowls_three,stack_bowls_two,stamp_seal,turn_switch"
TASKS=${TASKS_OVERRIDE:-$TASKS}

# ---- Environment ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot

export PYTHONPATH="${ROBOTWIN_ROOT}/envs/curobo/src:${ROBOTWIN_ROOT}:${PYTHONPATH:-}"
export HF_HOME=/storage/project/r-agarg35-0/shared/huggingface_cache
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export WANDB_DIR="$HOME/scratch/wandb"
export WANDB_CACHE_DIR="$HOME/scratch/wandb/cache"
export WANDB_ARTIFACT_DIR="$HOME/scratch/wandb/artifacts"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR"

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}" || exit 1
mkdir -p logs "$OUTPUT_DIR"

echo "=== RoboTwin FastWAM + Q self-improvement ==="
echo "  ITERATION:      $ITERATION / $MAX_ITERATIONS"
echo "  Q_CKPT:         $Q_CKPT"
echo "  DATASET:        $ORIGINAL_DATASET_REPO_ID @ $ORIGINAL_DATASET_ROOT"
echo "  N_EPISODES:     $N_EPISODES across 50 tasks"
echo "  PLANNER:        $PLANNER_TYPE  steps=$DIFFUSION_STEPS  n_samples=$N_SAMPLES  n_elites=$N_ELITES"
echo "  EVAL_AFTER:     $EVAL_AFTER_TRAIN  (${EVAL_EPISODES} eps x 50 tasks on $EVAL_ACCOUNT)"
echo "  OUTPUT_DIR:     $OUTPUT_DIR"
echo "============================================="

# ── Sharded collection ────────────────────────────────────────────────────────
# Collection is single-threaded SAPIEN physics (measured: ~1 of 6 cores busy, GPU
# mostly idle between chunk boundaries), so it parallelises across nodes almost
# linearly — 47 tasks x ~4 min = ~3.2 h sequential vs ~25 min on 8 nodes.
# COLLECT_SHARDS=1 (default) keeps the original single-job behaviour.
COLLECT_SHARDS=${COLLECT_SHARDS:-1}
MODE=${MODE:-orchestrate}
# Collection shards run on L40s, which is where the spare capacity is: measured
# 2026-08-07, gpu-l40s had 10 of 48 GPUs free while gpu-h100 (0/32) and gpu-h200
# (0/88) were fully allocated. Fanning out onto a saturated partition serialises the
# shards and throws away the point of sharding. Training stays on H100/H200.
COLLECT_ACCOUNT=${COLLECT_ACCOUNT:-$EVAL_ACCOUNT}
COLLECT_PARTITION=${COLLECT_PARTITION:-gpu-l40s}
COLLECT_GRES=${COLLECT_GRES:-gpu:l40s:1}

if [ "$COLLECT_SHARDS" -gt 1 ] && [ "$MODE" = "orchestrate" ]; then
    echo "Fanning out collection over ${COLLECT_SHARDS} shards ..."
    SHARD_IDS=()
    for (( k=0; k<COLLECT_SHARDS; k++ )); do
        if [ "$DRY_RUN" = "1" ]; then
            echo "  [dry-run] shard ${k}/${COLLECT_SHARDS} on ${COLLECT_ACCOUNT}/${COLLECT_PARTITION}"
            continue
        fi
        JID=$(sbatch --parsable \
            -A "$COLLECT_ACCOUNT" -q "$QOS" -p "$COLLECT_PARTITION" --gres="$COLLECT_GRES" \
            --cpus-per-gpu=4 --mem-per-gpu=64G -t 4:00:00 -J "si_rt_c${k}" \
            -o logs/%j.out -e logs/%j.err \
            --export=ALL,MODE=collect,SHARD="${k}/${COLLECT_SHARDS}",ITERATION=$ITERATION,Q_CKPT=$Q_CKPT,OUTPUT_DIR=$OUTPUT_DIR,N_EPISODES=$N_EPISODES,PLANNER_TYPE=$PLANNER_TYPE,N_SAMPLES=$N_SAMPLES,N_ELITES=$N_ELITES,DIFFUSION_STEPS=$DIFFUSION_STEPS,SEED=$SEED,WANDB_PROJECT=,COLLECT_SHARDS=$COLLECT_SHARDS,N_EVAL_PER_TASK=$N_EVAL_PER_TASK,N_TRAIN_PER_TASK=$N_TRAIN_PER_TASK \
            "$0")
        SHARD_IDS+=("$JID")
        echo "  shard ${k}/${COLLECT_SHARDS} -> job ${JID}"
    done
    if [ "$DRY_RUN" = "1" ]; then
        echo "  [dry-run] finetune on ${TRAIN_ACCOUNT}/${TRAIN_PARTITION} after ${COLLECT_SHARDS} shards"
        exit 0
    fi
    DEP=$(IFS=:; echo "${SHARD_IDS[*]}")
    # afterANY, not afterok: with afterok a single failed shard strands the finetune in
    # DependencyNeverSatisfied forever and silently kills the whole chain (this bit us
    # twice — once for 70 days). afterany releases the finetune regardless, and the
    # finetune then checks for itself that enough shards produced data.
    FT=$(sbatch --parsable \
        -A "$TRAIN_ACCOUNT" -q "$QOS" -p "$TRAIN_PARTITION" --gres=gpu:1 \
        --cpus-per-gpu=6 --mem-per-gpu=64G -t 8:00:00 -J "si_rt_ft" \
        -o logs/%j.out -e logs/%j.err \
        --dependency=afterany:$DEP \
        --export=ALL,MODE=finetune,ITERATION=$ITERATION,Q_CKPT=$Q_CKPT,OUTPUT_DIR=$OUTPUT_DIR,MAX_ITERATIONS=$MAX_ITERATIONS,N_EPISODES=$N_EPISODES,FINETUNE_STEPS=$FINETUNE_STEPS,FINETUNE_LR=$FINETUNE_LR,BATCH_SIZE=$BATCH_SIZE,ONLINE_FRACTION=$ONLINE_FRACTION,GRAD_CLIP_NORM=$GRAD_CLIP_NORM,PLANNER_TYPE=$PLANNER_TYPE,N_SAMPLES=$N_SAMPLES,N_ELITES=$N_ELITES,DIFFUSION_STEPS=$DIFFUSION_STEPS,SEED=$SEED,WANDB_PROJECT=$WANDB_PROJECT,WANDB_ENTITY=$WANDB_ENTITY,WANDB_RUN_NAME=$WANDB_RUN_NAME,TRAIN_ACCOUNT=$TRAIN_ACCOUNT,TRAIN_PARTITION=$TRAIN_PARTITION,EVAL_ACCOUNT=$EVAL_ACCOUNT,QOS=$QOS,EVAL_AFTER_TRAIN=$EVAL_AFTER_TRAIN,EVAL_EPISODES=$EVAL_EPISODES,EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE,COLLECT_SHARDS=$COLLECT_SHARDS,COLLECT_ACCOUNT=$COLLECT_ACCOUNT,COLLECT_PARTITION=$COLLECT_PARTITION,COLLECT_GRES=$COLLECT_GRES \
        "$0")
    echo "  finetune -> job ${FT} (after ${#SHARD_IDS[@]} shards)"
    exit 0
fi

# MODE=collect  -> one shard, stop before finetune
# MODE=finetune -> skip collection, train on the shards' episodes
if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] would run iteration $ITERATION (mode=$MODE) then submit eval + chain."
    exit 0
fi

# The finetune runs after `afterany`, so verify the shards actually produced data
# rather than trusting the barrier. Training on a partial or empty collection would
# silently corrupt the run — better to fail loudly and leave the chain stopped.
if [ "$MODE" = "finetune" ]; then
    N_SHARD_DIRS=$(ls -d "${OUTPUT_DIR}"/iter_$(printf '%03d' "$ITERATION")_shard*/online_episodes 2>/dev/null | wc -l)
    MIN_SHARDS=${MIN_SHARDS:-$(( (COLLECT_SHARDS + 1) / 2 ))}
    echo "Collected shard datasets: ${N_SHARD_DIRS}/${COLLECT_SHARDS} (need >= ${MIN_SHARDS})"
    if [ "$N_SHARD_DIRS" -lt "$MIN_SHARDS" ]; then
        echo "!! too few shards produced data — refusing to finetune on a partial collection"
        exit 1
    fi
    if [ "$N_SHARD_DIRS" -lt "$COLLECT_SHARDS" ]; then
        echo "!! WARNING: $(( COLLECT_SHARDS - N_SHARD_DIRS )) shard(s) failed; this iteration trains on fewer tasks"
    fi
fi

# Pool the shards' held-out eval results into a single per-iteration figure. Each shard
# scored its own ~6 tasks; this is the number to compare against the pre-SI baseline.
if [ "$MODE" = "finetune" ]; then
    python - "$OUTPUT_DIR" "$ITERATION" <<'PYAGG'
import json, sys, glob, math
out, it = sys.argv[1], int(sys.argv[2])
files = sorted(glob.glob(f"{out}/iter_{it:03d}_shard*/heldout_eval.json"))
n = s = 0
for f in files:
    d = json.load(open(f))
    k = d.get("n_eval", 0)
    p = d.get("eval_pc_success")
    if k and p is not None and not math.isnan(p):
        n += k; s += p * k / 100.0
if n:
    pooled = 100.0 * s / n
    print(f"HELD-OUT EVAL (iteration {it}, pooled over {len(files)} shards): "
          f"{s:.0f}/{n} = {pooled:.1f}%")
    json.dump({"iteration": it, "heldout_pc_success": pooled, "n_heldout": n,
               "n_shards": len(files)},
              open(f"{out}/iter_{it:03d}_heldout.json", "w"), indent=2)
else:
    print(f"No held-out eval results found for iteration {it} "
          f"(expected if this iteration predates the merged eval stage).")
PYAGG
fi

MODE_FLAGS=""
case "$MODE" in
    collect)  MODE_FLAGS="--collect_only --task_shard ${SHARD} \
        --n_eval_per_task ${N_EVAL_PER_TASK} --n_train_per_task ${N_TRAIN_PER_TASK}" ;;
    finetune) MODE_FLAGS="--skip_collect" ;;
esac

python scripts/self_improvement_loop.py \
    ${MODE_FLAGS} \
    --fastwam_ckpt "$FASTWAM_CKPT" \
    --q_ckpt "$Q_CKPT" \
    --env_type robotwin \
    --robotwin_root "$ROBOTWIN_ROOT" \
    --original_dataset_repo_id "$ORIGINAL_DATASET_REPO_ID" \
    --original_dataset_root "$ORIGINAL_DATASET_ROOT" \
    --task "$TASKS" \
    --n_iterations 1 \
    --start_iteration "$ITERATION" \
    --n_episodes "$N_EPISODES" \
    --finetune_steps "$FINETUNE_STEPS" \
    --finetune_lr "$FINETUNE_LR" \
    --batch_size "$BATCH_SIZE" \
    --online_fraction "$ONLINE_FRACTION" \
    --grad_clip_norm "$GRAD_CLIP_NORM" \
    --planner_type "$PLANNER_TYPE" \
    --n_samples "$N_SAMPLES" \
    --n_elites "$N_ELITES" \
    --diffusion_steps "$DIFFUSION_STEPS" \
    --eval_n_episodes 0 \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --wandb_project "$WANDB_PROJECT" \
    ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"} \
    ${WANDB_RUN_NAME:+--wandb_run_name "${WANDB_RUN_NAME}_iter$(printf '%03d' "$ITERATION")"} \
    2>&1 | tee -a "$OUTPUT_DIR/log.txt"

EXIT_CODE=${PIPESTATUS[0]}
ITER_LABEL=$(printf '%03d' "$ITERATION")

# A collect shard produces no Q checkpoint — the finetune job does the eval + chain.
if [ "$MODE" = "collect" ]; then
    echo "=== shard ${SHARD} of iteration ${ITERATION} done (exit ${EXIT_CODE}) ==="
    exit $EXIT_CODE
fi
NEW_Q_CKPT="${OUTPUT_DIR}/iter_${ITER_LABEL}/q_checkpoint"

# ---- Full 50-task eval of the freshly fine-tuned Q ----
# Submitted as independent L40s jobs so the (H200) training chain is not blocked
# waiting on ~1000 eval episodes.
if [ $EXIT_CODE -eq 0 ] && [ "$EVAL_AFTER_TRAIN" = "1" ] && [ -d "$NEW_Q_CKPT" ]; then
    echo "Submitting 50-task eval for iteration ${ITERATION} (Q: ${NEW_Q_CKPT}) ..."
    QOS="$QOS" ACCOUNT="$EVAL_ACCOUNT" \
    Q_CKPT_OVERRIDE="$NEW_Q_CKPT" \
    COND_OVERRIDE="si_rt_iter${ITER_LABEL}" \
    SWEEP_SPEC="${DIFFUSION_STEPS}:000000" \
    N_EPISODES="$EVAL_EPISODES" BATCH_SIZE="$EVAL_BATCH_SIZE" \
        ./scripts/eval_robotwin_q_sweep.sh
elif [ $EXIT_CODE -eq 0 ] && [ "$EVAL_AFTER_TRAIN" = "1" ]; then
    echo "!! expected Q checkpoint not found at ${NEW_Q_CKPT} — skipping eval"
fi

# ---- Chain next iteration ----
NEXT_ITERATION=$(( ITERATION + 1 ))
if [ $EXIT_CODE -eq 0 ] && [ $NEXT_ITERATION -lt $MAX_ITERATIONS ]; then
    echo "Chaining iteration $NEXT_ITERATION (Q ckpt: $NEW_Q_CKPT) ..."
    sbatch \
        -A "$TRAIN_ACCOUNT" -q "$QOS" -p "$TRAIN_PARTITION" \
        --dependency=afterok:$SLURM_JOB_ID \
        --export=ALL,MODE=orchestrate,SHARD=,COLLECT_SHARDS=$COLLECT_SHARDS,ITERATION=$NEXT_ITERATION,Q_CKPT=$NEW_Q_CKPT,OUTPUT_DIR=$OUTPUT_DIR,MAX_ITERATIONS=$MAX_ITERATIONS,N_EPISODES=$N_EPISODES,FINETUNE_STEPS=$FINETUNE_STEPS,FINETUNE_LR=$FINETUNE_LR,BATCH_SIZE=$BATCH_SIZE,ONLINE_FRACTION=$ONLINE_FRACTION,GRAD_CLIP_NORM=$GRAD_CLIP_NORM,PLANNER_TYPE=$PLANNER_TYPE,N_SAMPLES=$N_SAMPLES,N_ELITES=$N_ELITES,DIFFUSION_STEPS=$DIFFUSION_STEPS,SEED=$SEED,WANDB_PROJECT=$WANDB_PROJECT,WANDB_ENTITY=$WANDB_ENTITY,WANDB_RUN_NAME=$WANDB_RUN_NAME,TRAIN_ACCOUNT=$TRAIN_ACCOUNT,TRAIN_PARTITION=$TRAIN_PARTITION,EVAL_ACCOUNT=$EVAL_ACCOUNT,QOS=$QOS,EVAL_AFTER_TRAIN=$EVAL_AFTER_TRAIN,EVAL_EPISODES=$EVAL_EPISODES,EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE \
        scripts/run_self_improvement_robotwin.sh
fi

exit $EXIT_CODE
