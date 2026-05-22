#!/bin/bash
#SBATCH -A gts-agarg35-ideasci23_dgx
#SBATCH -N1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=64G
#SBATCH -q inferno
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH -p gpu-h100
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vgiridhar6@gatech.edu


# Resume the multi-dataset Q-function DDP run (BC only) from its last
# checkpoint. Same GPU layout as scripts/train_q_libero_ddp_multi_bc.sh.
#
# How resume works here:
#   * --config_path points at the train_config.json inside the last
#     checkpoint. lerobot reloads the ENTIRE original config from it
#     (datasets, policy, hyperparameters, output_dir, steps), so nothing
#     else needs re-specifying — CLI flags below only override.
#   * --resume=true loads optimizer / scheduler / RNG / step state and
#     continues in the SAME output_dir (new checkpoints append: 003000, ...).
#   * --wandb.run_id is set to a FRESH, unique id (job name + timestamp).
#     If we inherited the original run id, wandb would reject the resumed
#     run: the original already logged steps past the checkpoint, and
#     re-logging those steps violates wandb's monotonic-step rule. A new
#     id starts a clean wandb run that continues from the checkpoint step.
#     (Requires the WandBLogger resume="allow" change; see rl/wandb_utils.py.)
#
# Submit from the repo root (so logs/ resolves):
#     mkdir -p logs && sbatch scripts/train_q_libero_ddp_multi_bc_resume.sh
#
# Re-runnable: resume writes back into the same output_dir, so submitting
# this again picks up wherever the previous resume left off.
#
# Override the checkpoint dir if needed (e.g. to resume a different run):
#     CKPT_DIR=outputs/train/.../<run_dir> sbatch scripts/train_q_libero_ddp_multi_bc_resume.sh

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-lerobot-q}"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

# Keep wandb's run dir + artifact cache on scratch (home/project are quota-capped).
export WANDB_DIR="$HOME/scratch/wandb"
export WANDB_CACHE_DIR="$HOME/scratch/wandb/cache"
export WANDB_ARTIFACT_DIR="$HOME/scratch/wandb/artifacts"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR"

# $SLURM_SUBMIT_DIR is the dir sbatch was invoked from (the repo root). The
# BASH_SOURCE fallback covers running the script directly with `bash`.
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}" || exit 1

# --- run to resume -----------------------------------------------------------
JOB_NAME="qf_libero_ddp2_bsz48_bc_h200"
CKPT_DIR="${CKPT_DIR:-outputs/train/2026-05-19/23-54-42_qf_libero_ddp2_bsz48_bc_h200}"
# Resume from the latest good checkpoint of the previous run (step 6000). Bump
# CKPT_STEP (or set CKPT_STEP=last) when resuming again from a later checkpoint.
CKPT_STEP="${CKPT_STEP:-006000}"
CONFIG_PATH="${CKPT_DIR}/checkpoints/${CKPT_STEP}/pretrained_model/train_config.json"

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: checkpoint config not found: ${CONFIG_PATH}" >&2
    echo "       Set CKPT_DIR to the run's output directory and resubmit." >&2
    exit 1
fi

# Unique wandb run id = job name + submit timestamp (generated here, in-script).
WANDB_RUN_ID="${JOB_NAME}_$(date +%Y%m%d_%H%M%S)"
echo "Resuming from : ${CONFIG_PATH}"
echo "wandb run id  : ${WANDB_RUN_ID}"

# Unique torch-distributed rendezvous port, derived from the SLURM job id, so
# two accelerate jobs packed onto the same DGX node don't collide on the
# default port 29500 (EADDRINUSE).
MASTER_PORT=$(( 20000 + ${SLURM_JOB_ID:-$RANDOM} % 10000 ))
echo "master port   : ${MASTER_PORT}"

# Hang detector: py-spy-dumps every rank + dataloader worker if no training
# step is logged for HANG_TIMEOUT seconds (default 900), then tears the job
# down so a stall fails fast. Dumps land in logs/hang_dumps_<jobid>/.
source scripts/hang_watchdog.sh

run_with_hang_watchdog accelerate launch \
    --num_processes=2 \
    --mixed_precision=bf16 \
    --multi_gpu \
    --main_process_port=${MASTER_PORT} \
    $(which lerobot-train) \
    --config_path="${CONFIG_PATH}" \
    --resume=true \
    --wandb.run_id="${WANDB_RUN_ID}"
exit $?
