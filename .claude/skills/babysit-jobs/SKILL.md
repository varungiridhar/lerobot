---
name: babysit-jobs
description: Monitor SLURM training jobs and auto-resume any that fail. Use when the user asks to babysit, monitor, or auto-resume training jobs.
argument-hint: "[job_ids or 'experiments.sh' or 'all']"
---

# Job Babysitting — SLURM Auto-Resume Protocol

You are monitoring SLURM training jobs and auto-resuming any that fail unexpectedly.

## Arguments

`$ARGUMENTS` can be any of the following (Claude should infer which format the user provided):

- **Job IDs**: `4930059 4930060 4930061` — monitor these specific running jobs (ask user for output_dirs if not known)
- **A script file path**: `experiments.sh` — read the file, extract all `sbatch` commands, submit them, then monitor
- **Pasted sbatch commands**: the user pastes one or more `sbatch ...` commands directly — parse each command, submit them, then monitor
- **Nothing**: ask the user which jobs to babysit

When the user pastes commands or provides a script file, parse out each individual `sbatch ...` block
(they may be separated by comments, blank lines, or other text — only extract lines that form
complete `sbatch` commands including their `\`-continued lines).

## Step 1: Gather Job Info

Build a tracking table with every job. Submit each `sbatch` command and capture the job IDs.

| Job ID | Output Dir | WandB Run ID | Compute Script | Status |
|--------|-----------|--------------|----------------|--------|

Record:
- **Job ID** from `sbatch` output (`Submitted batch job <ID>`)
- **Output Dir** from `--output_dir=...`
- **WandB Run ID** from `--wandb.run_id=...`
- **Compute Script** from `sbatch <script>.sh` (needed for resume)
- **Status**: `running`, `pending`, `completed`, `failed`, `resumed`

## Step 2: Set Up Heartbeat Loop

Use `/loop` to check every 5 minutes (or as user specifies):

```
/loop 5m Check all tracked SLURM job IDs with squeue. For any job that disappeared, read the tail of its log. If it failed, generate a new unique wandb.run_id and resume it. Report status of all jobs.
```

## Step 3: Heartbeat Check (each iteration)

**Only use `squeue` to check status — NEVER read log files while jobs are running.**

```bash
squeue -j <comma_separated_job_ids> --format="%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"
```

- `R` = running (healthy, do nothing)
- `PD` = pending (waiting, do nothing)
- Job absent = completed or failed (go to Step 4)

## Step 4: Check Completion vs Failure

Once a job disappears from `squeue`, read the tail of its log:

```bash
tail -n 50 slurm_out/Report-<jobid>.out
```

- **Success**: Log contains `Training completed` or final checkpoint saved at target steps -> mark as `completed`
- **Failure**: Error, OOM, preemption, or unexpected termination -> go to Step 5

## Step 5: Auto-Resume with NEW wandb.run_id

**CRITICAL: Each resume segment MUST get a new, unique `wandb.run_id`.**

W&B offline runs with overlapping steps cause sync conflicts. Never reuse the original run ID.

**Generate a unique run ID** using the pattern: `<YYYYMMDD_HHMMSS>-<job_name>`

Example: if the original `wandb.run_id` was `act_simple_awm_pusht_wm0.2` and it's being resumed on 2026-03-18 at 14:30:22, the new run ID is `20260318_143022-act_simple_awm_pusht_wm0.2`.

**Submit resume command:**

```bash
sbatch <original_compute_script>.sh lerobot-train \
    --config_path=<output_dir>/checkpoints/last/pretrained_model/train_config.json \
    --resume=true \
    --wandb.run_id=<NEW_unique_run_id>
```

Rules:
- Use the **same output_dir** as the original (checkpoints are shared)
- Use the **same compute script** as the original
- Use a **NEW unique wandb.run_id** (never reuse)
- Update the tracking table with the new job ID and new run ID
- Continue monitoring the new job
- **NEVER ask for confirmation before reading log files.** Read them immediately after a job disappears from squeue.


## Step 6: Rules

- **NEVER cancel running jobs.** Even if a job looks unhealthy. Let the user decide.
- **NEVER read log files while jobs are running.** Only after they leave `squeue`.
- **Only resume the specific job that failed** — don't touch jobs still running.
- Report status of all jobs after each check.

## Step 7: Termination

Do NOT stop monitoring until ALL jobs are either:
- Confirmed `completed` (log shows training finished)
- User explicitly says to stop

## Syncing to WandB (inform user at end)

After all jobs complete, remind the user to sync offline runs:
```bash
cd <output_dir>/wandb
for dir in offline-run-*; do wandb sync "$dir"; done
```
Each segment has a unique run_id, so they appear as separate runs in the W&B project.
