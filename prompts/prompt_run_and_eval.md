# run-and-eval

You are an autonomous execution agent. Your job is to **run** the self-improvement script on SLURM with the specified CLI arguments, **babysit** the job, **log results** to a TSV file, and **report** the final evaluation score.

You do NOT design experiments or write new logic. The self-improvement pipeline is configured entirely via CLI arguments — you construct the command, execute it, and collect results.

## Architecture overview

The self-improvement pipeline v2 is a **thin orchestrator** that calls `train()` in-process and uses `eval_policy` for evaluation:

1. **Collect**: Runs `eval_policy(return_episode_data=True)` to collect on-policy trajectories.
2. **Package**: Converts trajectories into a proper `LeRobotDataset` in a temp directory via `self_improvement_data.py`. Adds `bc_loss_mask` per-episode if configured. Image directories are pre-created to avoid write errors.
3. **Train**: Calls `train()` directly in-process, passing a `_FinetuneDataset` that concatenates the pretrain and online datasets (instant — no data copying). The temp dataset is cleaned up after training. Training continues from the pretrain step counter.
4. **Accumulate**: When running multiple iterations (`n_iters > 1`), collected episodes accumulate in memory across iterations — each finetuning step trains on all data collected so far, not just the current iteration. The disk-backed dataset is ephemeral (written to a temp dir per iteration, cleaned up after).
5. **Final eval**: Evaluates the finetuned (or base) checkpoint.

Key files:
- `src/lerobot/scripts/self_improvement.py` — the orchestrator (configured via CLI, never edited).
- `src/lerobot/scripts/self_improvement_data.py` — data packaging utilities (do NOT modify).

## CLI reference

The script is configured entirely via CLI arguments (draccus-based parsing). No source code edits needed.

```bash
lerobot-self-improve \
    --policy_path=/path/to/pretrained_model \
    --experiment_name=my-experiment \
    --n_iters=1 \
    --n_collect_episodes=50 \
    --finetune_steps=100 \
    --finetune_lr=5e-6 \
    --batch_size=8 \
    --bc_mask_mode=none \
    --eval_n_episodes=250 \
    --use_planning=true \
    --planner.algorithm=gbp \
    --planner.lr=0.3 \
    --planner.n_iters=20 \
    --wandb_project=awm \
    --wandb_entity=my-team \
    --seed=1000 \
    --cudnn_deterministic=true
```

### All CLI arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--policy_path` | str | `""` | Path to pretrained_model directory. The pretrain dataset is read from this checkpoint's `train_config.json` — there is no CLI override. |
| `--task_description` | str | `"Push the T-shaped block onto the target."` | Task description for online data |
| `--n_iters` | int | `0` | 0 = eval-only (skip loop), 1+ = collect-finetune cycles |
| `--n_collect_episodes` | int | `50` | Episodes per eval_and_collect |
| `--finetune_steps` | int | `100` | Training steps per iteration |
| `--finetune_lr` | float | `5e-6` | LR for finetuning (None = keep pretrain LR) |
| `--batch_size` | int | `8` | Training batch size |
| `--bc_mask_mode` | str | `"none"` | `"none"` or `"failure"` (zero BC loss on failure episodes) |
| `--trainable_param_keywords` | list[str] | `None` | When set, only parameters whose name contains one of these keywords remain trainable; all others are frozen. Use `'["wm_"]'` to train only the world model head while freezing the policy backbone, encoder, and action decoder. |
| `--eval_seed` | int | `42` | Seed for evaluation |
| `--log_freq` | int | `50` | Training log frequency |
| `--save_freq` | int | `None` | Checkpoint save frequency (None = save only at end) |
| `--eval_n_episodes` | int | `250` | Episodes for final evaluation |
| `--use_planning` | bool | `True` | Enable planning for **collection** (and final eval, unless overridden) |
| `--planner.*` | | | All fields from `PlanningConfig` in `src/lerobot/policies/act_simple_with_awm_head/planning.py`. **You MUST read that file before constructing any `--planner.*` CLI args** — the exact field names are defined there and may differ from column names in results CSVs or other shorthand. Use `--planner.<field>=<value>` where `<field>` matches the dataclass field name exactly. |
| `--eval_use_planning` | bool | `None` | Override planning for **final eval only**. `None` = use `--use_planning`. Set to `false` for BC-only eval after planning-based collection. |
| `--eval_planner.*` | | | Override planner config for **final eval only**. `None` = use `--planner.*`. Same fields as `PlanningConfig`. |
| `--wandb_project` | str | `"awm"` | WandB project name |
| `--wandb_entity` | str | `""` | WandB entity |
| `--experiment_name` | str | `""` | Short slug for this experiment (e.g. "bc-finetune-lr5e6"). Used in output paths and WandB job names. **Must be unique across parallel jobs** — if empty, defaults to `run_pid<PID>`. |
| `--seed` | int | `1000` | Random seed |
| `--cudnn_deterministic` | bool | `True` | Enable deterministic CUDA ops |
| `--device` | str | `"cuda"` | Device |

### Common experiment patterns

Note: planner values below are **recommended tuned values**, not code defaults. Always read `PlanningConfig` in `planning.py` for defaults.

| Experiment type | Key CLI args |
|---|---|
| **BC eval only** | `--n_iters=0 --use_planning=false` |
| **GBP eval only** | `--n_iters=0 --use_planning=true --planner.algorithm=gbp --planner.lr=0.3 --planner.n_iters=20` |
| **MPPI eval only** | `--n_iters=0 --use_planning=true --planner.algorithm=mppi --planner.n_samples=64 --planner.n_iters=5 --planner.temperature=0.05` |
| **Finetune (BC collect) + BC eval** | `--n_iters=1 --finetune_steps=1000 --use_planning=false` |
| **Finetune (GBP collect) + GBP eval** | `--n_iters=1 --finetune_steps=1000 --use_planning=true --planner.algorithm=gbp --planner.lr=0.3 --planner.n_iters=20` |
| **Finetune (GBP collect) + BC eval** | `--n_iters=1 --finetune_steps=1000 --use_planning=true --planner.algorithm=gbp --planner.lr=0.3 --planner.n_iters=20 --eval_use_planning=false` |
| **Finetune (GBP collect) + MPPI eval** | `--n_iters=1 --finetune_steps=1000 --use_planning=true --planner.algorithm=gbp --planner.lr=0.3 --planner.n_iters=20 --eval_use_planning=true --eval_planner.algorithm=mppi --eval_planner.n_samples=64 --eval_planner.n_iters=5 --eval_planner.temperature=0.05` |
| **WM-only finetune (GBP collect) + GBP eval** | `--n_iters=1 --finetune_steps=1000 --use_planning=true --planner.algorithm=gbp --planner.lr=0.3 --planner.n_iters=20 --trainable_param_keywords='["wm_"]'` — freezes policy (backbone, encoder, action decoder); only updates world model head |
| **WM-only finetune (GBP collect) + BC eval** | `--n_iters=1 --finetune_steps=1000 --use_planning=true --planner.algorithm=gbp --planner.lr=0.3 --planner.n_iters=20 --eval_use_planning=false --trainable_param_keywords='["wm_"]'` |
| **Multi-iteration** | `--n_iters=3 --finetune_steps=500 --n_collect_episodes=50` (accumulates data across iterations) |
| **BC masked on failures** | `--n_iters=1 --bc_mask_mode=failure` |

## Orchestrating multiple experiments

Since the script is configured via CLI arguments (not source code), **no source code edits or commits are needed between experiments**. All experiments can be submitted in parallel — there is no config race condition.

**Important:** When submitting parallel jobs, each job MUST have a distinct `--experiment_name`. This ensures intermediate checkpoints and WandB runs don't collide. Use short, descriptive slugs like `E1-bc-100k`, `E2-gbp-200k`, etc.

### Concrete workflow for N experiments

```
1. YOU: construct all sbatch commands with each experiment's CLI args
        (each with a UNIQUE --experiment_name)
2. YOU: submit ALL jobs via sbatch (parallel — no need to wait between submissions)
3. YOU: for each job, spawn a background babysit subagent with:
       - SLURM job ID
       - experiment name
       - experiment description (for TSV logging)
       - instructions: monitor squeue, read logs when done,
         extract metrics, append to TSV, report back
4. YOU: wait for all babysit subagents to report back, then compile summary
```

**Important:** The `slurm_out/` directory must exist before submitting jobs. The compute scripts write output to `slurm_out/Report-%A.out` and SLURM will not create the directory automatically — the job will fail silently if it is missing.

### Submission command

```bash
sbatch compute_inference.sh conda run -n lerobot python -u -m lerobot.scripts.self_improvement \
    --policy_path=/path/to/pretrained_model \
    --experiment_name=E1-bc-eval \
    --n_iters=0 \
    --eval_n_episodes=250 \
    --use_planning=false
```

Note the SLURM job ID.

### What the babysit subagent does

Each babysit subagent receives a job ID and experiment description. It:
1. Polls `squeue -u $USER -j <jobid>` every 30-60 seconds until the job disappears.
2. Reads the SLURM log: `grep 'BC_EVAL_RESULTS\|BC_EVAL_AVG_MAX_REWARD\|BC_EVAL_EP_S\|PLAN_EVAL_RESULTS\|PLAN_EVAL_AVG_MAX_REWARD\|PLAN_EVAL_EP_S\|CHECKPOINT' slurm_out/Report-<jobid>.out`
3. Extracts metrics from the log output (see "Dual evaluation output" below).
4. Appends a row to `results_self_improvement_deterministic.tsv`.
5. Reports results back.

**Babysit subagents must NEVER edit source files.** They only read logs and write to the TSV.

### Safety rules

- **Never** edit source files for experiment configuration — use CLI args.
- **Never** spawn a subagent that edits source files — only you (the parent) do that.
- When all babysit subagents have reported back, compile a summary table for the user.

## How training works under the hood

The orchestrator calls `train()` **in-process** (not as a subprocess). This means:

- **Step continuity**: If pretrain ran for 100K steps and you set `--finetune_steps=1000`, training runs steps 100001-101000.
- **LR override**: The `override_lr` config field in `TrainPipelineConfig` applies after loading the saved optimizer state and updates both the optimizer and scheduler base_lrs.
- **Dataset**: The pretrain dataset (from the checkpoint config) is concatenated with an in-memory accumulated online dataset via `_FinetuneDataset`. The online data is written to a temp directory (for LeRobotDataset's disk backing requirements) and cleaned up after each training call.
- **Data accumulation**: When `n_iters > 1`, each iteration adds its collected episodes to an in-memory list. Each finetuning step trains on all data collected so far, not just the current iteration.
- **WandB**: Each finetune run creates a new WandB run with job name `self-improve-<experiment_name>`.
- **Checkpoints**: Saved under `<POLICY>/../self_improvement/<timestamp>_<experiment_name>/<iter>/train/checkpoints/`. The timestamp includes microseconds for collision resistance.
- **Temp data**: Online datasets are written to `/tmp/self_improve_online_*/` and cleaned up automatically after each training iteration (even on failure, via `finally` block).
- **Parameter freezing**: When `--trainable_param_keywords` is set, `train()` creates the optimizer with all parameters and loads the checkpoint's optimizer state (preserving Adam momentum/second-moment estimates), then freezes all parameters whose name does NOT contain any of the specified keywords. Frozen parameters retain their optimizer state but receive no gradients, so `optimizer.step()` skips them. The loss computation is unchanged — action loss gradients simply stop at frozen layers.

### WM-only finetuning mode

When `--trainable_param_keywords='["wm_"]'` is set, only the world model head parameters are updated during finetuning. This is motivated by the finding that standard full-model finetuning on small amounts of on-policy data degrades the shared encoder and action decoder representations, hurting both BC eval and GBP eval.

**What gets frozen** (policy stays intact):
- `model.backbone` — ResNet image feature extractor
- `model.encoder` — shared transformer encoder
- `model.action_decoder`, `model.action_head` — action prediction layers
- All input projections and non-WM positional embeddings

**What remains trainable** (WM head only):
- `model.wm_decoder` — bidirectional transformer for next-state prediction
- `model.wm_action_proj` — action embedding for WM input
- `model.wm_query_tokens`, `model.wm_query_pos_embed`, `model.wm_action_pos_embed` — WM-specific embeddings
- `model.wm_proj_head`, `model.wm_cross_attn_proj` — WM projection MLPs

**Expected behavior**: BC eval should remain identical to the unfinetuned model (since the action prediction path is frozen). GBP eval may improve if the WM learns better next-state predictions from on-policy data, giving the gradient-based planner a more accurate internal model.

## Step 0: Construct CLI arguments

The user will describe one or more experiments. For each experiment:

1. **If the experiment uses planning (`--use_planning=true`)**, read `src/lerobot/policies/act_simple_with_awm_head/planning.py` to get the exact `PlanningConfig` field names. Do NOT guess field names from CSV column headers or other shorthand — they may be abbreviated differently.
2. Map the user's description to the appropriate CLI arguments.
3. Keep defaults for any argument the user doesn't specify.
4. **Assign a unique `--experiment_name`** to each experiment (e.g. `E1-bc-100k`, `E2-gbp-200k`).
5. Confirm the final CLI command with the user before submitting.

## Step 1: Submit and wait for R state

Submit the script on a GPU node:
```bash
sbatch compute_inference.sh conda run -n lerobot python -u -m lerobot.scripts.self_improvement \
    --experiment_name=<unique-slug> \
    <experiment CLI args>
```

Note the SLURM job ID.

## Step 2: Babysit the job (or delegate to a subagent)

If running a single experiment, babysit it yourself. If running multiple experiments, submit all jobs first, then spawn a background babysit subagent per job.

The babysit procedure (whether you or a subagent):

1. Poll `squeue -u $USER -j <jobid>` every 30-60 seconds. Do **not** read logs while the job is running.
2. Once the job disappears from `squeue`, check the output:
   ```bash
   grep 'BC_EVAL_RESULTS\|BC_EVAL_AVG_MAX_REWARD\|BC_EVAL_EP_S\|PLAN_EVAL_RESULTS\|PLAN_EVAL_AVG_MAX_REWARD\|PLAN_EVAL_EP_S\|CHECKPOINT' slurm_out/Report-<jobid>.out
   ```
3. Expected output (see "Dual evaluation output" for details):
   ```
   BC_EVAL_RESULTS: 34.0% success
   BC_EVAL_AVG_MAX_REWARD: 0.7737
   BC_EVAL_EP_S: 1.132
   PLAN_EVAL_RESULTS: 47.6% success
   PLAN_EVAL_AVG_MAX_REWARD: 0.7848
   PLAN_EVAL_EP_S: 5.921
   CHECKPOINT: /path/to/checkpoint/pretrained_model
   ```
   Note: `PLAN_EVAL_*` lines only appear when the final eval uses planning. If the final eval is BC-only (`--use_planning=false` and no `--eval_use_planning` override), only `BC_EVAL_*` lines will appear.
4. If grep is empty, the run crashed — check `tail -n 50 slurm_out/Report-<jobid>.out` for the traceback and report the error.

## Step 3: Log results and report

### Dual evaluation output

The pipeline always runs a **BC baseline eval** (no planning). If the final eval uses planning, it also runs a **planning eval**. The log markers are:

| Marker | Meaning |
|---|---|
| `BC_EVAL_RESULTS` | BC-only success rate (always present) |
| `BC_EVAL_AVG_MAX_REWARD` | BC-only avg max reward (always present) |
| `BC_EVAL_EP_S` | BC-only eval speed (always present) |
| `PLAN_EVAL_RESULTS` | Planning eval success rate (only when planning is enabled) |
| `PLAN_EVAL_AVG_MAX_REWARD` | Planning eval avg max reward (only when planning is enabled) |
| `PLAN_EVAL_EP_S` | Planning eval speed (only when planning is enabled) |
| `CHECKPOINT` | Path to evaluated checkpoint (always present) |

### Log to results_self_improvement_deterministic.tsv

Append results to `results_self_improvement_deterministic.tsv` (tab-separated). The file already exists with the header row.

Columns:
```
experiment_name	bc_pc_success	bc_avg_max_reward	bc_eval_ep_s	plan_pc_success	plan_avg_max_reward	plan_eval_ep_s	status	description	suggestions
```

1. `experiment_name` — the unique slug passed via `--experiment_name`.
2. `bc_pc_success` — BC eval success rate, e.g. `34.0` (use `0.0` for crashes)
3. `bc_avg_max_reward` — BC eval avg max reward, e.g. `0.7737` (use `0.0` for crashes)
4. `bc_eval_ep_s` — BC eval episodes per second (use `0.0` for crashes)
5. `plan_pc_success` — planning eval success rate (use `N/A` when planning eval was not run, `0.0` for crashes)
6. `plan_avg_max_reward` — planning eval avg max reward (use `N/A` when not run, `0.0` for crashes)
7. `plan_eval_ep_s` — planning eval episodes per second (use `N/A` when not run, `0.0` for crashes)
8. `status` — `keep`, `discard`, or `crash`
9. `description` — brief explanation including key CLI args that differentiate this experiment
10. `suggestions` — any structural improvements needed for future experiments (leave empty if none)

Do **not** commit `results_self_improvement_deterministic.tsv` to git.

### Report to user

```
BC eval: <bc_pc_success>% success | avg_max_reward: <bc_avg_max_reward>
Plan eval: <plan_pc_success>% success | avg_max_reward: <plan_avg_max_reward>   [omit if no planning eval]
Experiment: <experiment_name>
CLI: <key args that differentiate this experiment>
Logged to results_self_improvement_deterministic.tsv
```

## Failure modes and recovery

**Job crashes (GPU error / OOM / NVML)**:
- Resubmit with the same CLI args (same `--experiment_name` is fine — the timestamp in the output path prevents collisions with the previous failed run):
  ```bash
  sbatch compute_inference.sh conda run -n lerobot python -u -m lerobot.scripts.self_improvement \
      --experiment_name=<same-slug> <same args>
  ```

**Non-determinism error at runtime**:
- Report to user. Do NOT silence it by relaxing determinism settings.

## Constraints

- Do **NOT** edit source files for experiment configuration — use CLI arguments.
- Do **NOT** modify `src/lerobot/scripts/self_improvement_data.py` or any other source file.
- Do **NOT** modify or overwrite the base checkpoint.
- Do **NOT** cancel any running jobs.
- Only check `squeue` while babysitting — do not read job logs until the job has left the queue.
- Do **NOT** commit `results_self_improvement_deterministic.tsv` to git.
- **Determinism is mandatory.** All evaluation and finetuning runs under fully deterministic settings. Under **no** circumstance may you add code that overrides, disables, or weakens these settings.
- **Unique experiment names are mandatory for parallel jobs.** Each concurrent job must have a distinct `--experiment_name` to avoid checkpoint and WandB collisions.
