# FastWAM + Q-planning — experiment context

Handoff notes for an agent picking up work on this repo. Written 2026-08-07.
Repo: `/storage/project/r-agarg35-0/igeorgiev3/lerobot`, branch `qplanning`.

Everything below is measured on this cluster with these checkpoints. Where a number
is uncertain or came from a small sample, it says so — please keep that habit.

---

## 1. What the project is

FastWAM is a video-world-model VLA policy. On top of it we run **Q-planning at eval
time**: sample N candidate action chunks from the diffusion prior, score each with a
trained Q-function, and aggregate. Two benchmarks: **LIBERO** (4 suites x 10 tasks)
and **RoboTwin 2.0** (50 bimanual tasks).

Planner variants (`PlanningConfig.planner_type`):
- `bc_diffusion_mppi` — N independent diffusion runs, Q-weighted mean of top-K. **Default.**
- `bc_diffusion_argmax` — same sampling, take the single best.
- `mppi` / `cem` / `argmax` — run diffusion once, perturb with Gaussian noise, score.

`bc_diffusion_*` is single-pass: **`n_iters` is ignored** for it, only the noise-based
`mppi` branch loops. Several configs in the repo set `n_iters=3` misleadingly.

---

## 2. Headline results

### RoboTwin — 50 tasks x 20 episodes, matched on the 47 tasks every condition has

| condition | succ% | vs baseline | ep_len |
|---|---|---|---|
| bcdiff s5, Q 45k | 84.0 | +1.3 | 278.7 |
| bcdiff s2, Q 15k | 83.4 | +0.6 | 285.0 |
| bcdiff s3, Q 15k | 83.4 | +0.6 | 276.5 |
| bcdiff s5, Q 15k | 83.0 | +0.2 | 276.8 |
| **baseline (no Q)** | **82.8** | — | 278.5 |
| bcdiff s5, Q 30k | 82.2 | -0.5 | 278.7 |
| bcdiff s3, Q 30k | 81.8 | -1.0 | 289.7 |
| bcdiff s3 + Q-guidance 0.02 | 80.1 | -2.7 | 294.6 |
| bcdiff s1, Q 15k | 78.9 | -3.8 | 300.0 |
| bcdiff s3 + Q-guidance 0.05 | 65.5 | -17.2 | 369.1 |

Binomial SE on 940 episodes is ~1.2pp, so **nothing between 81.8 and 84.0 is
distinguishable from baseline**. The honest summary: on RoboTwin, no Q configuration
beats no-Q by a measurable margin. Only the guidance conditions and s1 separate.

### LIBERO-10 — 10 tasks x 20 episodes

| method | succ | rate% | vs baseline |
|---|---|---|---|
| bc_diffusion_mppi, 3 steps | 190/200 | 95.0 | +5.0 |
| bc_diffusion_mppi, 5 steps | 183/200 | 91.5 | +1.5 |
| mppi (noise-perturb) | 181/200 | 90.5 | +0.5 |
| **baseline (no Q)** | 180/200 | 90.0 | — |
| mppi + smoothing sigma_t=8 | 148/200 | 74.0 | -16.0 |

**The suites disagree.** Q clearly helps on LIBERO-10 (+5.0pp) and does nothing
measurable on RoboTwin (+0.6pp). Note the two are not sample-matched: LIBERO uses
n_samples=64/n_elites=16, RoboTwin 32/8. Some of the gap could be sampling budget —
a 64-sample RoboTwin run has never been done and would be worth doing.

Temporal smoothing of MPPI noise is monotonically harmful on LIBERO-10:
sigma_t = 2 / 4 / 8 -> -1.1 / -7.8 / -18.9 pp (matched 90-episode subset). No sweet spot.

### Q-gradient guidance — implemented, works mechanically, hurts

Lives on branch **`q-guidance`** (commit 5b1a57a), deliberately not merged. It steers
diffusion along dQ/d(clean action estimate) instead of ranking after the fact. It does
what it says: candidate diversity 4.4x at scale 0.02, 7.7x at 0.05, and q_max +39%.
Task success still drops (-2.7pp / -17.2pp above). Gradient ascent walks off the BC
manifold into actions Q overrates; sample-then-rank is safe precisely because it can
only pick among on-manifold samples.

**Methodological warning worth internalising:** the guidance trust region initially
looked safe (5/5 success at both 0.02 and 0.05) because it was measured on
`adjust_bottle`, where the baseline is also 5/5. A task at ceiling can only detect
harm, and under-detects it. Do not generalise an operating range from a saturated task.

---

## 3. Traps that have already cost time

1. **Three RoboTwin tasks are broken** and crash when driven by a learned policy:
   `open_laptop`, `place_object_scale`, `put_object_cabinet`. Cause: their
   `check_success()` reads `self.arm_tag`, which is only assigned inside `play_once()`
   (the scripted demo). They are absent from every eval condition. Eval tolerates this
   (one process per task); anything running all 50 in one process must catch per-task
   exceptions.

2. **Checkpoint / dataset paths in `train_config.json` point into other users' scratch
   and have been purged.** Copy anything you depend on to
   `~/r-agarg35-0/q_checkpoints_backup/` first. Already lost this way: Q 5k and 20k.

3. **There are two RoboTwin datasets and one is a decoy.**
   - Correct: `/storage/project/r-agarg35-0/shared/robotwin2.0` — 27,500 episodes, 6.1M frames.
   - Decoy: `.../qplanning_rebuttal_handoff/data/robotwin2.0_multicam` — 300 episodes (~1%).
   Both have the right camera keys, so structure checks do not distinguish them. Check
   `meta/info.json total_episodes`.

4. **`save_episodes_lerobot` built its schema from a hardcoded LIBERO camera list.**
   Fixed — it now derives features from `camera_keys`. If you add a third env, pass its
   keys through; the env's raw camera names (RoboTwin: `head_camera`/`left_camera`/
   `right_camera`) differ from the Q function's (`cam_high`/`cam_left_wrist`/
   `cam_right_wrist`) and need the rename map in `collect_episodes`.

5. **`afterok` + preemption = permanently stranded job.** A preempted predecessor
   leaves its successor in `DependencyNeverSatisfied` forever. This killed the LIBERO
   self-improvement loop at iteration 6 for ~70 days. Use `inferno` for chained
   pipelines, and prefer `afterany` + an explicit data check over `afterok`.

6. **`tee` masks failures.** Several scripts reported exit 0 while the underlying
   `lerobot-eval` died at argument parsing. Use `${PIPESTATUS[0]}`.

---

## 4. Cluster routing (verified 2026-08-07)

| Account | Reaches |
|---|---|
| `gts-agarg35` | everything, incl. typed `--gres=gpu:h200:1` on `gpu-h200` |
| `gts-agarg35-ideas_l40s` | `gpu-l40s` |
| `gts-agarg35-ideasci23_dgx` | **routes to `gpu-h100`, not `gpu-h200`**; cannot request typed `gpu:h200:1` |

- QoS: `embers` free/preemptable, 50 queued, 8 h wall. `inferno` paid, non-preemptable,
  500 queued, no wall cap.
- Typed `--gres=gpu:<type>:1` needs the matching `#SBATCH -p gpu-<type>`. Names lowercase.
- Free-GPU counts move a lot. On 2026-08-07: l40s 10/48 free, h100 0/32, h200 0/88.
  **Check before fanning out** — a wide fan-out onto a saturated partition serialises.
  `scontrol show node -o` + parse `CfgTRES`/`AllocTRES`; node state (`mix`) is misleading.
- Dry-run routing: `sbatch --test-only -p <part> -q <qos> -A <acct> --gres=gpu:1 -t 1:00:00 --wrap="true"`

---

## 5. Key paths

```
Checkpoints
  FastWAM RoboTwin  /storage/project/r-agarg35-0/shared/fastwam/hf_checkpoint_robotwin
  FastWAM LIBERO    /storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint
  Q RoboTwin        ~/r-agarg35-0/q_checkpoints_backup/qf_robotwin_ddp_20260526_220319/{015000,030000,045000}
  Q LIBERO          ~/r-agarg35-0/q_checkpoints_backup/qf_libero_ddp2_20260519_235442/012000
Data
  RoboTwin 2.0      /storage/project/r-agarg35-0/shared/robotwin2.0        (27.5k episodes — use this)
  RoboTwin repo     /storage/project/r-agarg35-0/vgiridhar6/robotwin/RoboTwin
Results
  outputs/eval/robotwin_<cond>/<task>/     RoboTwin, one dir per task
  outputs/eval/bench_bcdiff_s3_libero_10/  LIBERO, one dir per suite
  outputs/eval/robotwin_overview.csv       cross-condition table
```

## 6. Scripts

| Script | Purpose |
|---|---|
| `scripts/eval_fastwam_q_robotwin.sh` | RoboTwin eval suite (baseline + bcdiff s3) |
| `scripts/eval_fastwam_q_libero.sh` | LIBERO eval; all knobs via `--export` |
| `scripts/eval_robotwin_q_sweep.sh` | Sharded 50-task sweep; `Q_CKPT_OVERRIDE`/`COND_OVERRIDE`/`QOS`/`ACCOUNT`. Skips finished tasks, so resubmitting resumes |
| `scripts/summarize_eval.py` | Per-experiment `summary.csv` + cross-condition table. Caches ffprobe frame counts |
| `scripts/export_eval_csv.py` | Per-episode CSV; recovers preempted runs by inferring success from frame count |
| `scripts/self_improvement_loop.py` | SI loop. Supports both suites via `--env_type`; `--task_shard K/N`, `--collect_only`, `--skip_collect`; `--planner_type dsrl` for the DSRL arm |
| `scripts/run_self_improvement_robotwin.sh` | RoboTwin SI driver: sharded collect -> finetune -> eval -> chain |
| `scripts/run_dsrl_si_libero.sh` | DSRL baseline on LIBERO-10 SI. Single resumable job, no `afterok` chain |
| `scripts/debug_dsrl_latent.py` | DSRL preflight: noise-injection fidelity, latent steerability, tiling cost |

`summarize_eval.py` and `eval_robotwin_q_sweep.sh` are **untracked** — commit them if
you rely on them.

---

## 7. Currently running (as of 2026-08-07)

RoboTwin self-improvement, iteration 0, from Q 45k with 3-step bcdiff.
Output: `outputs/self_improvement/si_robotwin_ck045k_s3/`.

- 8 collection shards (~6 tasks each), then finetune on H100, then a 50-task eval, then
  chain to iteration 1. 10 iterations planned.
- Data layout per iteration: `iter_NNN_shardKK/online_episodes` (episodes),
  `iter_NNN/q_checkpoint`, `iter_NNN/metrics.json`, plus appended `loop_summary.jsonl`.
  `OnlineQDataset` globs `iter_*/online_episodes`, so it is a **growing buffer** — later
  iterations train on all earlier episodes too.
- Sharding is exactly equivalent to sequential collection (verified: identical
  (task, seed) sets), because seeds derive from each task's index in the full list.

**The number that decides whether this is worth continuing:** the pre-SI score of this
exact config is **82.6%** (`outputs/eval/robotwin_bcdiff_s3_ck045k`). That is
effectively iteration -1. If iteration 0's eval does not clear it by more than ~1.2pp,
the loop is not doing anything and hyperparameter ablations will not rescue it.

---

## 7b. DSRL baseline — implemented 2026-08-07, NOT YET RUN

Latent-noise steering (Wagenmaker et al., arXiv 2506.15799) as a baseline for Q-Planning.
Freezes FastWAM, learns a policy over the flow sampler's **initial noise** instead of
ranking its outputs. Code: `src/lerobot/policies/dsrl/`.

Why it matters for the paper: that paper's Fig. 8 steers pi0 on LIBERO from ~20% to ~100%
and reports that **V-GPS — value-guided re-ranking of sampled actions, i.e. our offline
Q-Planning — fails to improve pi0 at all**. A reviewer who knows it will ask why
sample-and-rank should work for us. Our RoboTwin numbers (nothing distinguishable from
baseline) sit close to their V-GPS result.

Instantiation is DSRL-NA (their Algorithm 1) with our Q as the action-space critic Q^A, so
both arms share an identical critic and fine-tune schedule. Only two things are new:
Q^W(s,w) <- Q^A(s, pi_dp(s,w)) distilled over prior draws, and an actor maximising Q^W.

Three deliberate deviations, all documented in `modeling_dsrl.py`:
- Q^A keeps our bootstrap (next chunk from the buffer) instead of `a' ~ pi_dp(s', pi^W(s'))`.
  **Consequence: Q^A stays ~Q^{pi_BC}, so the loop does one step of policy improvement per
  iteration, not full policy iteration.** This is the main thing that would make our DSRL
  weaker than the paper's, and it is the first thing to fix if the arm underperforms.
- Deterministic actor + exploration noise, not SAC with entropy (no latent TD => no
  principled temperature).
- Batched updates between rounds, not interleaved per env step.

`tiled` mode (default) follows their pi0 recipe: learn one per-timestep latent, repeat it
across the 32-step chunk (7-d instead of 224-d). This is a restriction of the search space
and iteration 0 pays for it — `debug_dsrl_latent.py` measures the cost before committing
GPU hours. `--dsrl_full_latent` switches to the unrestricted 224-d latent.

Cost profile is favourable: collection uses the same 16 decodes/step as the Q-Planning arm
(1 executed + 15 prior draws recorded for distillation), but **deployment is 1 decode per
step instead of 16** — no per-step search at all. The latent fit needs no diffusion runs,
since every `pi_dp(s,w)` was already computed and stored during collection.

## 8. Open questions, roughly in order of value

1. **Is this Q function informative at all?** Three independent attempts (checkpoint
   sweep, diffusion-step sweep, gradient guidance) have failed to extract value on
   RoboTwin. Run `scripts/debug_q_values_dataset_replay.py` on 15k/30k/45k and check
   whether Q rises monotonically toward the terminal state on known-good demos. ~1 GPU-h,
   and it gates everything else. **This has been recommended repeatedly and never run.**
2. **Is the LIBERO/RoboTwin gap a sampling-budget artefact?** Run RoboTwin at
   n_samples=64 / n_elites=16 to match LIBERO.
3. **Does SI help?** Compare iteration-N evals against 82.6%.
4. **SI x negative-Q paradigms.** `--neg_margin_weight`, `--neg_tube_sigmas` etc. are
   wired into `finetune_q` but off by default. Negatives directly target the failure
   mode guidance exposed (Q overrating off-manifold actions).
5. **Guidance on LIBERO.** Scale sweeps 0.01/0.02/0.05/0.10 were built and cancelled
   before finishing. LIBERO is the suite where Q demonstrably helps, so it is the fair
   test of whether guidance can ever help.
6. **Does DSRL beat Q-Planning on matched data?** (Sec. 7b.) Run `debug_dsrl_latent.py`
   first — if Q over independently drawn latents barely spreads, there is nothing to
   steer and the arm cannot work regardless of training. Compare against
   `si_libero_h200_gc10_n100_v2`: collect 93 / 91 / 95 / 95 / 93 / 97 % over iterations
   0-5, ~45 min/iteration on an H200.

---

## 9. Working notes

- Always compare on a **matched task subset**. Preemption and broken tasks leave
  conditions with different coverage, and unmatched rankings are selection-biased.
  `summarize_eval.py` emits NaN for missing tasks so they drop out cleanly.
- For per-task deltas prefer a paired test (Wilcoxon over tasks) — much more sensitive
  than comparing pooled rates.
- RoboTwin logs a benign `UnStableError: Objects is unstable in seed(N)` and re-seeds.
  It is caught internally. Do not grep for bare `Traceback` as a failure signal.
- Episode length is not in `eval_info.json`; it is derived from video frame counts.
  RoboTwin videos cover all episodes, LIBERO renders only 100 of 200.
