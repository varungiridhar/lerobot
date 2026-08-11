#!/bin/bash
# Submit one correlated Slurm array per resumable H=1 IBRL phase.
#
# Every successor uses aftercorr, so task k advances only after task k from the
# preceding array succeeds. All phases share OUTPUT_ROOT/taskK/resume_state.pt.
# This script submits jobs; use DRY_RUN=1 to inspect the exact dependency chain.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
WORKER=${WORKER:-$SCRIPT_DIR/run_ibrl_self_improvement.sh}
ARRAY_SPEC=${ARRAY_SPEC:-0-9%10}
SEED=${SEED:-42}
N_EPISODES=${N_EPISODES:-100}
ONLINE_SHARD_ENDS=${ONLINE_SHARD_ENDS:-"25 50 75 100"}
RUN_ID=${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/outputs/self_improvement_ibrl/${RUN_ID}_td3_ibrl_h1_libero_10_seed${SEED}}
DRY_RUN=${DRY_RUN:-0}

# These worker-only variables are intentionally derived independently inside
# every array element. Inheriting stale values through --export=ALL could make
# all ten tasks write one directory or resume one task's state.
unset OUTPUT_DIR COLLECTION_TASK_ID PHASE ONLINE_EPISODE_END RESUME_STATE

if [[ ! -r "$WORKER" ]]; then
    echo "IBRL worker is not readable: $WORKER" >&2
    exit 2
fi

previous_end=0
for endpoint in $ONLINE_SHARD_ENDS; do
    if [[ ! "$endpoint" =~ ^[0-9]+$ ]] || (( endpoint <= previous_end )); then
        echo "ONLINE_SHARD_ENDS must be strictly increasing integers: $ONLINE_SHARD_ENDS" >&2
        exit 2
    fi
    previous_end=$endpoint
done
if (( previous_end != N_EPISODES )); then
    echo "Last online shard endpoint ($previous_end) must equal N_EPISODES ($N_EPISODES)." >&2
    exit 2
fi

submit_phase() {
    local phase=$1
    local dependency=$2
    local online_end=${3:-}
    local export_vars="ALL,RUN_ID=$RUN_ID,OUTPUT_ROOT=$OUTPUT_ROOT,SEED=$SEED,N_EPISODES=$N_EPISODES,ACTION_HORIZON=1,PHASE=$phase"
    local command=(sbatch --parsable --array="$ARRAY_SPEC")
    if [[ -n "$dependency" ]]; then
        command+=(--dependency="aftercorr:$dependency")
    fi
    if [[ -n "$online_end" ]]; then
        export_vars+=",ONLINE_EPISODE_END=$online_end"
    fi
    command+=(--export="$export_vars" "$WORKER")

    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'DRY RUN:' >&2
        printf ' %q' "${command[@]}" >&2
        printf '\n' >&2
        # Derive a stable placeholder locally: command substitution runs this
        # function in a subshell, so a mutable global counter would reset.
        local placeholder=900000
        case "$phase" in
            online) placeholder=$((900000 + online_end)) ;;
            reference_eval) placeholder=900900 ;;
            hybrid_eval) placeholder=901000 ;;
        esac
        printf '%d\n' "$placeholder"
        return
    fi

    local submission
    submission=$("${command[@]}")
    local job_id=${submission%%;*}
    if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
        echo "Could not parse Slurm job id from: $submission" >&2
        exit 1
    fi
    echo "Submitted $phase${online_end:+ through episode $online_end}: $job_id" >&2
    printf '%s\n' "$job_id"
}

warmup_job=$(submit_phase warmup "")
previous_job=$warmup_job
for endpoint in $ONLINE_SHARD_ENDS; do
    online_job=$(submit_phase online "$previous_job" "$endpoint")
    previous_job=$online_job
done
reference_job=$(submit_phase reference_eval "$previous_job")
hybrid_job=$(submit_phase hybrid_eval "$reference_job")

echo "IBRL H=1 pipeline submitted." >&2
echo "  run id:       $RUN_ID" >&2
echo "  output root:  $OUTPUT_ROOT" >&2
echo "  warmup array: $warmup_job" >&2
echo "  final array:  $hybrid_job" >&2
echo "  final status: squeue -j $hybrid_job" >&2
