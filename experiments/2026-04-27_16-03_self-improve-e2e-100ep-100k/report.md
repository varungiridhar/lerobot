# E2E Self-Improvement Finetuning on 100ep / 100K Checkpoint

## Original prompt

> Using @prompt_run_and_eval.md to run and evaluate self improvement experiment. I pretrained a model on very low data setting: @/storage/home/hcoda1/6/vgiridhar6/forks/lerobot/outputs/act_simple_awm_pusht_wm1.0_l2norm_100ep/checkpoints/100000/pretrained_model. Even with a low data regime, this checkpoint gets ~15% success, which is not bad for pushT. Though, the impetus for training in a low data regime was to see if we can boost performance with GBP data colelction + finetuning on this data. I would like to finetune e2e, which means all parameters are finetunined on on-policy data.
>
> Please see a previous experiments:
>
> -  @/storage/home/hcoda1/6/vgiridhar6/forks/lerobot/experiments/2026-04-09_self-improve-wm-only-90k - finding best planning parameters relative to BC baseline and then finetuning world model only on on-policy data
> - @/storage/home/hcoda1/6/vgiridhar6/forks/lerobot/experiments/2026-04-19_19-14_self-improve-e2e-90k - using the previous experiment's best GBP parameters to do on-policy e2e finetunine (we finetune everything, masking BC episodes with 0 mask)
>
> The aim of _this_ experiment is to first find the best GBP parameters, much like how "@/storage/home/hcoda1/6/vgiridhar6/forks/lerobot/experiments/2026-04-09_self-improve-wm-only-90k" does it, then to take the best GBP parameters to explore best e2e finetuning parameters (use @/storage/home/hcoda1/6/vgiridhar6/forks/lerobot/experiments/2026-04-19_19-14_self-improv  e-e2e-90k for inspiration on axes to explore when sweeping finetuning settings). To name it out, the variables you should focus on exploring with e2e finetuning is the number of on-policy episodes we append to the pretrain dataset, the number of finetuning steps, and the number of iterations we do this for. Make sure your experiment builds off results you get at earlier stages. In other words, serialize your research process, and iterate on previous findings.
>
> In summary, find best GBP parameters with the "@/storage/home/hcoda1/6/vgiridhar6/forks/lerobot/outputs  /act_simple_awm_pusht_wm1.0_l2norm_100ep/checkpoints/100000/pretrained_model" checkpoint, use this GBP parameters to then explore self-improvement via e2e finetuning of the model with on-policy data.

## Research question

For the `act_simple_awm_pusht_wm1.0_l2norm_100ep` checkpoint at step 100000 — a low-data base trained on 100 expert episodes (vs 50ep used in the prior 50ep/90k experiments and 200ep used in the original `truly_deterministic` setting) — that achieves ~15% BC success on PushT:

1. **Stage 1**: What is the best GBP planning configuration on this base, relative to the BC baseline?
2. **Stage 2**: Using Stage 1's best GBP config for on-policy data collection, does **end-to-end finetuning** (all parameters trainable) lift success rates above the unfinetuned BC and GBP baselines? Sweep axes: `n_collect_episodes`, `finetune_steps`, `n_iters`.

The 100ep base sits between the prior 50ep base (BC ≈ 4.8%, where neither WM-only nor e2e finetuning helped) and the 200ep `truly_deterministic` base (BC ≈ 34%, GBP ≈ 47%). At ~15% BC success we have meaningful headroom *and* a non-trivial planner signal — both prerequisites for self-improvement to work.

## Experiment plan

### Strategy

Two stages, each adaptive on the prior stage's results.

- **Stage 1** characterizes the new base. BC baseline + a GBP sweep centered on the prior optima (G7 from `experiments/2026-04-04_planning-hparam-sweep/data/gbp_results.csv` and G6 from `experiments/2026-04-09_self-improve-wm-only-90k`). The user's 100ep/100k base is intermediate between the 200ep/100k checkpoint (BC=34%, GBP G7=+13.6pp) and the 50ep/90k checkpoint (BC=4.8%, GBP G6=+0.4pp), so the prior optima are reasonable starting points but the exact best (lr, n_iters) may shift.

- **Stage 2** uses Stage 1's best GBP for collection and runs an e2e finetuning sweep on (`n_collect_episodes`, `finetune_steps`, `n_iters`). Coarse single-iter first; multi-iter and zoom adaptively after.

### Fixed across all stages

- Base policy: `outputs/act_simple_awm_pusht_wm1.0_l2norm_100ep/checkpoints/100000/pretrained_model`
- `--eval_n_episodes=250`
- `--seed=1000 --cudnn_deterministic=true`
- `--wandb_project=awm --wandb_entity=pair-diffusion`
- `--batch_size=32` (matches pretrain batch_size from `train_config.json`; orchestrator default is 8 which would shrink the batch)
- Compute: `compute_rtx6000.sh`
- Branch: `self-improvement-v2`

### Stage 1 — GBP characterization

**Stage 1A — coarse 4×4 grid (17 jobs, parallel):**
- E0: BC eval-only (`--n_iters=0 --use_planning=false`)
- G-lr{0.2,0.3,0.4,0.5}-ni{5,10,15,20}: 16 GBP eval-only jobs at `action_cost_coef=0.1`, `lr_decay=1.0`, `convergence_tol=1e-3`. Centered on G7 (lr=0.3, n_iters=10).

**Stage 1B — zoom on best cell (~4-6 jobs, parallel, adaptive):**
- Finer grid around Stage 1A best (lr, n_iters) and small acc sweep `acc_cost_coef ∈ {0.05, 0.15, 0.2}`.

### Stage 2 — E2E finetuning sweep (Stage 1 best GBP for collection)

Fixed for Stage 2:
- `--use_planning=true` with Stage 1 best GBP for collection
- No `--eval_planner.*` override → final eval = BC eval (always) + GBP eval at the same Stage 1 best config
- `--bc_mask_mode=failure`
- `--finetune_lr=null` (preserve pretrain optimizer state)
- `--batch_size=32`

**Stage 2A — coarse single-iter sweep (9 jobs, parallel):**
- `n_collect_episodes ∈ {10, 50, 100}` × `finetune_steps ∈ {500, 2000, 10000}`
- `n_iters=1`

The prior 50ep/90k e2e found `n_collect=10` was the only safe regime and `n_collect=50` collapsed BC. The 100ep/100k base has 3× higher BC and a stronger planner, so on-policy data quality is much better; n_collect=50 and n_collect=100 are worth retesting.

**Stage 2B — adaptive zoom + multi-iter (~6-9 jobs, parallel):**
- Multi-iter on Stage 2A's best `(n_collect, ft_steps)`: `n_iters ∈ {2, 3}` × 2-3 ft variants
- Plus 1-2 zoom cells around the best Stage 2A configuration

### Budget

| Stage | Jobs | Cumulative |
|---|---|---|
| 1A | 17 | 17 |
| 1B | ≤6 | ≤23 |
| 2A | 9 | ≤32 |
| 2B | ≤9 | ≤41 |

Each batch ≤30 concurrent (SLURM cap). Stages run sequentially; sub-stages within a stage in parallel. Total ceiling ≈40 jobs.

### Stopping criteria

- **Stage 1**: Stop after Stage 1A if a clear best cell emerges with statistically meaningful improvement over BC baseline (>3pp at 250 episodes, well above ~1.4pp binomial σ at 5%-15% success); Stage 1B only if landscape is ambiguous or to fine-tune the best cell.
- **Stage 2**: Stop after Stage 2B if the best cell (a) clears both BC and GBP baselines, or (b) two consecutive sub-stages show no improvement. If e2e cannot beat the GBP baseline despite the higher-quality on-policy data, the conclusion mirrors the 50ep result (e2e is degrading).

## Methodology

_(To be written after execution completes.)_

## Results

_(To be written after execution completes.)_

## Key findings

_(To be written after execution completes.)_

## Conclusions

_(To be written after execution completes.)_

## Stopping rationale

_(To be written after execution completes.)_
