# STATUS — read this first

Orientation for anyone (human or agent) picking up a CoRL rebuttal task.
`corl_reviews.md` = the reviews. `conversations.md` = team plan (informal, some parts stale).
`rules.md` = compute + checkpoint paths. `qplanning_paper.pdf` = **the submitted paper, not the current method.**

## Reviewer ID map

`conversations.md` tags weaknesses with `[r1,r2,r3,m]`. The mapping is:

| tag | reviewer | score | stance |
|---|---|---|---|
| `r1` | **AzZT** | 6 — Accept | Positive. Asked about the Gaussian-envelope exploration limit; "real robot experiments would be awesome." |
| `r2` | **ZGT5** | 3 — Weak reject | Asked the latency questions. Wants Best-of-N + filtered SFT baselines, newer benchmarks, MPPI-bootstrap justification. |
| `r3` | **svm5** | 2 — Reject | Novelty + missing baselines only. |
| `m` | **AC rV3G** | borderline | Metareview; five Key Issues for Rebuttal. |

Note: the latency ask appears in **both** the AC metareview and ZGT5's Q2. The "real robot would be awesome"
line is AzZT's, not ZGT5's — `conversations.md` line 24 tags it `[r2]`, which is wrong.

## The method changed after submission — this matters

The paper describes **temporal-smoothed MPPI**: one BC chunk, perturbed with temporally-correlated
Gaussian noise, `N=64` candidates × `T=3` MPPI iterations = **192 Q-evaluations** per planning step.
That is the number the AC and ZGT5 are asking about.

**The rebuttal will instead ship `bc_diffusion_mppi`:** draw `N` on-manifold BC chunks directly from the
diffusion head at a low step count (3 steps on LIBERO, 5 on RoboTwin), score all `N` with Q, aggregate
with a single softmax-weighted mean over the top-K. **No noise perturbation, no MPPI iterations, no
temporal smoothing.** See `conversations.md` §"Ignat on reproducing robotwin results" for why.

Consequence: the cost profile inverts. Old = 1 denoising run + 192 Q-evals. New = `N`-sample denoising
+ `N` Q-evals. Any latency claim must cover **both** so the reviewers' "192" is actually answered.

Temporal smoothing is being dropped from the paper and the team has agreed to disclose the change.

## Real-robot demo

**Varun is running real-world demos** for the rebuttal. `conversations.md` line 24 ("no time; reviewers
don't care that much") is superseded — do not repeat that reasoning anywhere near a rebuttal draft.
Latency analysis and the real-robot demo are separate, complementary deliverables.

## Numbers a latency/feasibility argument needs

Both from this repo, both **simulated benchmarks** (real-world control rates are Varun's to supply):

All values below are the **resolved** config, read out of the eval logs of the two runs Ignat named as
the current results (`bench_bcdiff_s3_libero_10`, `robotwin_bcdiff_s3_ck045k`) — not defaults, not
guesses. Those logs dump the full merged config; re-read them if anything here looks off.

| | LIBERO | RoboTwin |
|---|---|---|
| env control rate | **30 Hz** | **25 Hz** |
| BC checkpoint | `/storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint` | `/storage/project/r-agarg35-0/shared/fastwam/hf_checkpoint_robotwin` |
| `chunk_size` | 32 | 32 |
| `n_action_steps` (replan interval) | **10** | **24** |
| `num_inference_steps` (BC policy, resolved) | **10** | **10** |
| ⇒ wall-clock budget per planning step | **333 ms** | **960 ms** |
| action dim | 7 | 14 |
| cameras | 2 (agentview + wrist, 224×224) | 1 stacked (384×320) |

The replan budget is the deadline: the planner emits a chunk, `n_action_steps` of it are executed at the
env rate, then it replans. Beat that budget and the method is real-time at the benchmark's own rate.

**Denoising steps — settled.** `scripts/eval_fastwam_q_libero.sh` sets `NUM_INFER_STEPS=10`, which
overrides the LIBERO checkpoint's `config.json` value of 20; the RoboTwin script hardcodes
`--policy.num_inference_steps=10`. So the BC baseline is **10 steps on both benchmarks**. The paper's
line-130 claim that 5 steps "halves latency" is relative to that 10.

## The exact shipping planner config

Read from the same two logs. Note these differ between benchmarks:

| | LIBERO | RoboTwin |
|---|---|---|
| `planner_type` | `bc_diffusion_mppi` | `bc_diffusion_mppi` |
| `n_samples` (N) | **64** | **32** |
| `n_elites` (K) | 16 | 8 |
| `num_diffusion_steps` (candidate sampling) | **3** | **3** |
| `temperature` | 1.0 | 1.0 |
| `n_iters` | 1 | 3 — **inert, see below** |
| Q checkpoint | libero `…23-54-42_qf_libero_ddp2…/last` | `…qf_robotwin_ddp_20260526_220319/045000` |

Two traps:

1. **`n_iters` is ignored by the `bc_diffusion_*` branch.** `plan_chunk_fastwam` samples N candidates
   once, scores once, aggregates once, and returns — it never loops. RoboTwin's `n_iters=3` therefore
   costs nothing and does nothing. RoboTwin runs **32** Q-evals per planning step, not 96.
2. **`conversations.md` labels the best RoboTwin row "5 steps"**, but the run directory Ignat pointed
   at is `robotwin_bcdiff_s3_ck045k` and its log says `num_diffusion_steps=3`. Either the table label
   or the directory pointer is wrong. Time **3** steps as the primary and 5 as a secondary; flag the
   discrepancy rather than silently picking one.

## Where the planner lives

- `src/lerobot/policies/fastwam/planning.py::plan_chunk_fastwam` — all planner branches
  (`bc_diffusion_mppi`, `bc_diffusion_argmax`, `mppi`, `argmax`, `cem`)
- `src/lerobot/policies/act_simple/planning.py` — `PlanningConfig` (defaults: `n_samples=64`,
  `n_iters=3`, `n_elites=16`, `temperature=1.0`), `_sample_noise`, `_score_candidates`,
  `_score_candidates_fast`
- `src/lerobot/policies/fastwam/modeling_fastwam.py::predict_n_action_chunks` — N-sample BC draw;
  passes `num_samples=N` into `infer_action`, so the denoising loop is likely batched rather than
  looped N times. **Verify by measurement.**
- Q-function: `src/lerobot/policies/q_function/` — DINOv2-large (~307M) + T5-v1.1-base (~250M) frozen
  encoders, 18-layer d=1024 transformer decoder (~302M), 101-bin HL-Gauss head.
  `encode_obs_context()` runs the encoders **once** per planning step; `_score_candidates_fast()` then
  expands that context to N and runs only the decoder batched. ZGT5 asked about exactly this.
- Eval entry points: `scripts/eval_fastwam_q_libero.sh`, `scripts/eval_fastwam_q_robotwin.sh`

There is **no timing instrumentation anywhere** in the planning path — no `perf_counter`, no
`cuda.synchronize`. It has to be added.
