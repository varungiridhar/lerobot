# Real-robot Q-planning + self-improvement — report (YAM, stack-cups)

Status: **Phase 0 in progress** (training running; gates pending). This file is the living
report; every number lands here as it is produced. Sections marked ⏳ await results.

## Setup

**Task**: single right arm of the bimanual YAM (`bi_yam_follower`) picks up one cup and
stacks it in the other. Human judges success; the leLab eval UI records outcomes.

**Source dataset**: `VarunGiridhar3/KT_stack_cups_20260807_211827` — LeRobot v3.0,
114 episodes / 22,314 frames @ 30 fps, 14-dim bimanual action + state, cameras
`left_wrist` / `right_wrist` / `top` (480×640, AV1). Per-episode outcomes in
`meta/episode_labels.json`: **101 success / 13 failure** (failures are short aborts,
34–174 frames; successes 159–268 frames, mean 208 ≈ 6.9 s). Collected **kinesthetically**
(handle on the arm): verified `action[t] ≈ state[t]` (mean gap 0.0036 rad at lag 0,
growing with lag) — same-timestep convention, no leader-follower offset.

**Derived dataset** (all training uses this): `VarunGiridhar3/KT_stack_cups_20260807_211827_right`
— right arm only (dims 7–13 → 7-dim `action`/`observation.state`, gripper last),
`left_wrist` dropped, videos byte-exact copies, per-dim stats (incl. quantiles) sliced
exactly, `episode_labels.json` carried. Derivation: `scripts/derive_right_arm_dataset.py`
(parameterized by dataset id — rerun as-is for the second task).

**Kinesthetic training convention**: chunks trained **as-is** (no temporal shift). With
chunked execution the one-step offset is an eval-time choice (execute from chunk index 1);
decided at G1/deployment. Rationale: shifting labels would touch 4 configs + every
consumer for a ≤33 ms/replan gain.

## Models

| | BC primary | BC fallback | Q-function |
|---|---|---|---|
| arch | ABC-DiT **Base** (768/12/12, 282M trainable; DINOv3-B/16 + frozen CLIP; rectified flow) | lerobot `diffusion` (resnet18 ×2 cams, U-Net, DDIM 100 train steps) | DINOv2-large + T5-v1.1-base (frozen) + 18×1024 decoder, HL-Gauss 101 bins on [-0.01, 1.01] |
| chunk / h | chunk_size **32**, n_action_steps 10 | horizon **32**, n_obs_steps 1 (usable chunk 32) | h = **32** (== BC chunk; loader-enforced) |
| key config | QUANTILES norm (STATE+ACTION), fp32 (Turing), grad ckpt, lr 3e-5 (YAM-recipe), image transforms on | MIN_MAX norm, resize (240,320) crop 0.95, lr 1e-4, batch 64 | cams `[right_wrist, top]`, text on ("Stack cups"), γ 0.99, τ 0.005, lr 2e-4/6e-5, batch 32 |
| data | success episodes only (101), test split 0.1 seed 1000 → 10 val eps [3,17,22,40,54,75,82,100,104,106] | same | all 114 episodes, stratified holdout 0.1 seed 42 (10 success + 1 failure val), `PYTHONHASHSEED=0` pinned |
| steps | 30k | 60k | 30k |

**Reward definition (Q)**: sparse; r = 0 at every in-episode step; the episode's terminal
frame adds a bonus of **1.0 if the episode is labeled success, 0.0 if failure** (per-episode
buckets from `meta/episode_labels.json` via the new `bucket_overrides` sentinel
`episode_labels`; success→`q5`, failure→`play`). Discount γ = 0.99; TD target
`Σ γ^i r_i + γ^h · bootstrap · Q_target(s_{t+h}, a_{t+h:t+2h})`. Failed episodes are real
zero-return negatives; synthetic negatives ON (margin 0.25: swap + tube(1,2)σ smoothed,
gripper held + time-reversal — the L2 recipe).

**Planner (shipping config family)**: `bc_diffusion_mppi` — N on-manifold chunks in ONE
batched low-step draw from the BC head, one batched Q scoring pass (obs encoded once),
softmax-weighted mean of top-K. Ported as a self-contained module; the BC-norm → raw
(`bc_post`) → Q-norm (`q_pre`) round-trip uses each checkpoint's saved processors, and
feeding Q preprocessor-normalized images is a **loud error** (the G2 silent-failure class).

## Phase 0 — training jobs (PACE)

| job | what | resource | status |
|---|---|---|---|
| 11743665 | ABC-DiT Base full (30k) | RTX6000 embers, auto-requeue+resume | ⏳ running |
| 11743666 | diffusion fallback full (60k) | RTX6000 embers | ⏳ 50k/60k (resume 11751793) |
| 11749949 / 11749950 | bc25 / bc50 subset ABC-DiT | RTX6000 embers | ⏳ running |
| 11742406 / 11743613 | Q probe (L40S inferno / H100 inferno race) | | ⏳ queued |
| ⏳ | Q full (30k, h=32) | winner of the probe race | pending probe |
| 11743331 / 11743330 | BC probes (30 steps) | | ✅ passed |

Probe-driven fixes upstreamed to the training branch: episode-subset sampler used absolute
frame indices (IndexError; fixed + validated), `PYTHONHASHSEED` salting made the Q holdout
split irreproducible across resumes (pinned to 0).

## Phase 0 gates ⏳

- **G1** BC replay (`bc_gate1_replay.py`): per-horizon-step MAE on the 10 held-out demos,
  gripper accuracy, kinesthetic first-step check. ⏳
- **G2** Q ranking (`q_gate2_real.py`): rank-acc/AUROC of true vs {swap, reversed,
  shuffled, hold, tube×3, grip-flip} on held-out successes+failure; Q-vs-time (Spearman)
  must rise in successes, not the failure.
  **Round 1 (base Q 30k & 25k ckpts): FAIL on criterion (a), PASS (b) and (c).**
  The value signal is excellent — every held-out success episode's Q ramps ~0.1→0.96
  with Spearman ρ ≥ 0.90 (median 0.995); the failure episode ends at 0.13; tube
  negatives ranked 0.91–0.97, "freeze" chunks 0.86. But action discrimination is weak:
  frame-matched cross-episode swaps 0.48, reversed 0.64, shuffled 0.55 → the base Q
  ≈ V(s) (image-based progress) with sensitivity only to off-manifold actions.
  **Response (per the gate's mandate, no robot time spent):** two new targeted
  negatives added to the pipeline — fraction-matched cross-episode swap (`mswap`) and
  temporal-shift chunks (`shift8/16`, "right actions, wrong time" free from the 2h
  window) — plus margin 0.25→0.5, δ 0.1→0.15; Q finetuned 10k steps from the 30k base
  (job 11747688).
  **Round 2 (negft 10k & 7.5k): (a) unchanged 0.63 FAIL; (b) ρ 0.994 PASS; (c) PASS.**
  Held-out mswap stays at chance; shift negatives *invert* on held-out data (0.08–0.12
  — Q prefers later-looking chunks; the TD progress signal dominates the margin loss).
  Interpretation: with 101 single-task kinesthetic demos and terminal-only reward, Q is
  a near-perfect progress estimator with off-manifold action rejection (tubes ≈1.0,
  freeze 0.86) but no same-manifold chunk discrimination. Note the sim Q that delivered
  planner lift also trained without margin loss — bad-draw rejection, not fine-grained
  ranking, is the plausible lift mechanism. **Ship Q**: `qf_stackcups_h32_negft_010000`
  (uploaded; gates b/c intact, best trev).
- **G3** planner smoke (real BC100-10k + negft-Q): mechanics PASS (round-trip, attach,
  batched N=32 draw, scoring); candidate diversity real (cross-sample std 0.08); **but
  Q-spread over the 32 BC candidates ≈ 0.002** on an in-distribution frame — planning
  is a no-op where the strong BC's draws are all good. 40-frame spread sweep running;
  the decisive version runs against **bc25** (weak BC ⇒ variable draws ⇒ Q's bad-draw
  rejection can bite). ✅ mechanics.
  **40-frame Q-spread sweeps (N=32, s3)** — the pre-registered offline lift predictor:
  BC100-10k: median 0.010, p90 0.086, 12% of frames > 0.05. **BC-5k (the 40%-success
  deployment BC): median 0.017, p90 0.167, 32% of frames > 0.05** — Q separates the
  weak BC's candidates at a third of frames. Prediction on record: BC-5k+Q should
  beat BC-5k's 40%; if it doesn't, candidate spread was not the bottleneck.
  [Confirmed on-robot: 40% → 65%.]
  **bc25 (25 demos, 15k steps, job 11753286): FLAT — median 0.0027, p90 0.0096, 0%
  of frames > 0.05.** Prediction on record: BC25+Q ≈ BC25 — a bc25 robot comparison
  is NOT worth robot time. Mechanism reading: candidate spread tracks
  UNDERTRAINING (sampler entropy), not demo-set size — bc25 at 15k steps is
  converged/overfit on its 25 demos and draws consistently, while the deployed
  BC-5k's spread came from being undertrained (5k of 30k).
  **Completed grid (demo count × training progress), all sweeps N=32 s3 40 frames,
  sorted by epochs over the training set:**

  | BC | steps | ≈epochs | spread median | p90 | % frames > 0.05 |
  |---|---|---|---|---|---|
  | 100 demos | 5k — deployed | 8 | 0.017 | 0.167 | 32% |
  | 50 demos | 5k (job 11753826) | 15 | 0.0130 | 0.0983 | 18% |
  | 100 demos | 10k — BC100 ref | 16 | 0.010 | 0.086 | 12% |
  | 25 demos | 5k (job 11753293) | 31 | 0.0081 | 0.0338 | 8% |
  | 50 demos | 20k (job 11753825) | 62 | 0.0030 | 0.0748 | 12% |
  | 25 demos | 15k (job 11753286) | 92 | 0.0027 | 0.0096 | 0% |

  **Spread median is monotone in epochs over the training set** (distance from
  convergence) — across BOTH axes, six checkpoints, no exceptions. The actionable
  tail (% frames > 0.05) broadly follows (32→18→12→8→0 along the epoch ordering)
  with one wrinkle: converged bc50 retains a small heavy tail (12%, max 0.16 —
  a few states where 50 demos still disagree). Q-planning's bad-draw-rejection
  lift engages where the BC sampler is still high-entropy; demo count matters
  mainly through how fast the policy converges. Caveat (constant across all
  sweeps): the predictor is measured on in-distribution demo frames.
- **G4** latency: **315 ms median vs 333 ms budget** (95%) on L40S at N=32, 5 denoise
  steps (BC draw 109 ms + Q encode/score 206 ms); the deployment 5090 will be faster.
  ✅ (cai-1 confirmation at session time).
- **Plan revision (headroom + G3 findings, 2026-08-08)**: BC100 succeeds "almost every
  time" on-robot (operator report) ⇒ no headroom, and Q-spread is flat under BC100.
  **Phase 1 primary comparison becomes bc25 vs bc25+Q** (25-demo BC, training as jobs
  11749949/11749950 with bc50), with BC100 kept as a reference condition.

## Deployment deliverable (Phase 0d) — built, pending on-rig validation

- `lerobot_collectedai` branch **`qplanning/yam-eval`** (off `feat/yam-real-world-integration`):
  abc_dit cherry-picked from main; q_function inference port; planner module;
  `qplanning_sync` engine (planner attach **before** hardware connect + full-path warmup,
  raw-obs feed to Q, 14→7 state slice, 7→14 action padding with measured left-arm hold).
- `leLab` branch **`qplanning/yam-operator-eval`** (off `feat/yam-operator`): request
  fields (default-off), session-start hardware guard against the daily stack, separate
  launchers (unit `lelab-qplanning-server`, port 8010), `QPLANNING_DEPLOY.md`.
- Two adversarially-verified review findings fixed pre-deployment (cold-planner
  CAN-watchdog risk; session-start exclusivity race).

## Phase 1 — BC vs BC+Q — **RESULT (2026-08-08, operator-reported)**

| condition | success | config |
|---|---|---|
| BC-only | **8/20 (40%)** | ABC-DiT Base ckpt 005000 |
| BC + Q-planning | **13/20 (65%)** | same BC + `qf_stackcups_h32_negft_010000`, bc_diffusion_mppi N=32 K=8 s5 |
| Filtered BC (reviewer baseline) | **50%** (operator-reported 2026-08-09, standard 20-seed protocol) | `abcdit_stackcups_right_filteredbc_r1_005000` — same BC finetuned on demo + its own 12 rollout successes |

**Same rollout budget, two uses**: feeding the BC rollouts back as filtered BC buys
+10 points (40→50); feeding the identical episodes into Q-planning buys +25
(40→65). Q-planning beats the filtered-BC baseline by 15 points — with the caveat
(recorded above) that filtered BC's +10 partly reflects extra training steps on the
undertrained deployment checkpoint, which if anything flatters the baseline.

**+25 points over 20 eval seeds per condition**, matching the pre-registered offline
prediction (Q-spread p90 0.167 / 32% of frames > 0.05 under this BC). Notes: the
100-demo BC ("succeeds almost every time") had no headroom, so the deployed BC is the
5k checkpoint; BC100 remains the reference ceiling. Rollout datasets:
`rollout_abcdit_stackcups_right_005000_bc_r0` (control) and
`..._bcq_negft010k_r0` (planning), both with `episode_labels.json`.
⏳ to backfill: placement notes, on-robot planning latency (debug JSONL), stamped ids.

Original protocol (kept for the formal writeup): ≥20 episodes/condition, matched +
randomized placements, interleaved, fixed checkpoints; headroom lever exercised
(weaker 5k checkpoint after BC100 showed ~no headroom).

## Phase 2 — planning-scheme comparison ⏳

Offline-first grid: {argmax, mppi} × N ∈ {8,16,32} × steps ∈ {3,5,10}; top 2–3 to robot.

## Phase 3 — self-improvement (Q-only; BC frozen) — **ACTIVE, round 1 in flight**

BC frozen = ABC-DiT 005000. Q_0 = `qf_stackcups_h32_negft_010000`. Round-1 input: the
two r0 rollout datasets above (~40 autonomous labeled episodes). Per-round contract:
verify labels → finetune mix (rollout successes positive, rollout failures REAL
negatives, demo fraction ≥50%, held-out demo slice) → finetune Q only → gate: G2(b)/(c)
must not degrade + rank rollout failures below successes → publish Q_1 → operator
re-rolls (same protocol incl. BC-only drift control). Headline curve: success vs round.
A second task's dataset is being collected in parallel (prompt reruns end-to-end).

### Round-1 data (verified 2026-08-08)

| dataset | episodes | labels | notes |
|---|---|---|---|
| `rollout_..._bc_r0_20260808_165837` | 30 | **12/30 success (40%)** | BC-only condition; operator-reported 8/20 — the dataset carries 30 scored episodes at the same 40% rate; all 30 used |
| `rollout_..._bcq_negft010k_r0_20260808_194142` | 20 | **13/20 success (65%)** | matches the report exactly |

Verification (both): every episode labeled, lengths plausible (successes ≈ demo
lengths; failures shorter or timeout), left arm static (hold-pose padding worked,
std ≤ 0.001 rad), right-arm |action−state| ≈ 0.01–0.03 rad — i.e. **real commanded
actions** (deployment distribution), not kinesthetic identity; task string exactly
"Stack cups". Derived right-arm versions (14→7 slice + rig `head`→dataset `top`
camera rename, new `--rename_camera`/`--source_root` flags on the derive script):
`VarunGiridhar3/rollout_stackcups_bc_r0_right` / `..._bcq_r0_right` (pushed, private).

### Round-1 finetune (job 11752552 probe → full)

Multi-repo `MultiLeRobotDataset` mix: demo (68.9% of frames — ≥50% contract met) +
both rollout sets; per-repo `episode_labels` buckets; ACTION normalization stats
pinned to the demo repo (`action_stats_repo_id`) so Q1's action space == Q0's;
stratified per-(repo,bucket) holdout seed 42 → held-out demo AND rollout episodes.
Warm start from Q_0, negatives recipe inherited unchanged (the ONLY change vs Q_0 is
the rollout data), lr 3e-5 / backbone 1e-5, 10k steps, H100 inferno
(`train_q_stackcups_q1.sbatch`). CPU pre-flight validated the full wiring incl.
terminal rewards (rollout failure terminal frame → 0.0; success → 1.0).
Gate runner ready: `q_gate_q1.sbatch` = G2(demo, b/c non-degradation) +
`scripts/q_gate_rollout_rank.py` (terminal-window Q AUROC success-vs-failure on
rollout episodes, Q1 head-to-head vs Q_0; pre-registered PASS = AUROC ≥ 0.8 ∧
≥ Q_0's ∧ held-out failures below held-out success median).

**Round-1 result (2026-08-09, gate job 11753156): PASS → Q_1 published.**
Training: warm start Q_0, 10k steps; held-out failure/success value separation
widened from 0.18/0.38 to 0.08/0.44 during the finetune.
G2 (demo heldout): (b) ρ 0.9905 (Q_0: 0.994 — not degraded), (c) failure terminal
0.127 vs success median 0.964 PASS; (a) unchanged at Q_0's accepted 0.63.
Rollout ranking: **Q_1 AUROC 1.000** on all 50 rollout episodes (terminal-Q median:
failures 0.000, successes 0.918) AND on the held-out subset (2s/3f, small n);
Q_0 reference: 0.909 all / 0.833 heldout (failure median 0.326) — the finetune
crushed real-failure values to ~0 without touching the demo-side value function.
Honest caveat: 45/50 rollout episodes were in Q_1's training mix; the held-out-only
AUROC (1.000, n=5) and the Q_0 comparison carry the generalization claim.
**Q_1 deploy ref: `VarunGiridhar3/qf_stackcups_h32_q1_010000`** (private; staged
rsync copy at `scratch/hf_upload/qf_stackcups_h32_q1_010000`). BC unchanged at
`abcdit_stackcups_right_005000`.

**Round-1 ON-ROBOT RESULT (2026-08-09, operator-reported): BC+Q_1 = 15/20 (75%).**
The self-improvement curve: **40% (BC) → 65% (BC+Q_0) → 75% (BC+Q_1)** — +10 points
from one Q-only round on 50 autonomous episodes; BC frozen throughout. Rollouts:
`VarunGiridhar3/rollout_stackcups_bcq_q1_r1` (15/20 labels verified; derived
`..._bcq_q1_r1_right`). Protocol deviation (operator's call, on record): the r1
BC-only drift control was skipped — the 40% baseline reference is r0's.
Filtered-BC eval rollouts also captured:
`rollout_..._filteredbc_r1_005000_r0_20260809_000332` (10/20, verified; derived
`rollout_stackcups_fbc_r0_right`).

**Round 2 (2026-08-09): BC+Q_2 = 17/20 (85%).** Q_2 = finetune of Q_1 on demo +
all three loop rollout sets (70 eps; filtered-BC eval data EXCLUDED — user decision,
pure loop); deployed at the 7.5k-of-10k checkpoint under deadline pressure
(`VarunGiridhar3/qf_stackcups_h32_q2_007500`); 10k + gate-vs-Q_1 ran after
(job 11768820). Rollouts: `VarunGiridhar3/rollout_stackcups_bcq_q2_r2` (17/20
verified; derived `..._r2_right`).

**The self-improvement curve: 40% → 65% → 75% → 85%** (BC frozen throughout;
Q-only updates on autonomous experience; +45 points total, +20 from two
self-improvement rounds on ~90 autonomous episodes).

Q_2(10k) gate (job 11768820, post-hoc to the deployed 7.5k): **PASS** — G2(b)
ρ 0.9922 / (c) intact ((a) at the accepted 0.65 chance level), rollout ranking
AUROC 1.000 all + 1.000 held-out (now 70 episodes incl. r1), vs Q_1 reference
0.994/0.938 — each generation cleanly ranks the newest rollout set that its
predecessor saw only at inference time.

**Round 3 (final, in flight)**: Q_3 = finetune of deployed Q_2(7.5k) on demo +
all FOUR rollout generations (~53% demo), 5k steps deadline mode, 5-repo mix
(job 11768848, `train_q_stackcups_q3.sbatch`, warm-start `q3_warmstart`).
Trained (full 5k) + published: `VarunGiridhar3/qf_stackcups_h32_q3_005000`;
iter-3 eval dataset requested as `rollout_stackcups_bcq_q3_r3`. ⏳ on-robot.

### Task-2 self-improvement round 1 + filtered BC (2026-08-09, deadline night)

r0 on-robot: **BC-10k 5/20 (25%) → BC+Q(25k) 8/20 (40%)** — the lift replicates
on the bimanual task. Datasets (verified; head→top renamed, kept 14-dim):
`rollout_..._bc010000_bcq_q025k_r0_20260809_153704` / `..._bc010000_bc_r0_20260809_151300`
(cache `rollout_insertcard_{bcq,bc}_r0`).
**Filtered BC (task 2): 30%** — finetune of BC-10k on demo (43) + its own 5
rollout successes (`abcdit_insertcard_filteredbc_r1_002500`, dataset
`insertcard_filteredbc_r1`, G1 skipped under deadline; rollouts
`rollout_insertcard_filteredbc_r1_r0`). Same pattern as task 1 at lower absolute
level: filtered SFT +5 vs Q-planning +15 on the identical rollout budget —
consistent with the mechanism (draw-level veto beats trajectory-level SFT where
precision binds).
**Q iter-1**: finetune of deployed Q_0(25k) on demo (49.6%) + both r0 sets, 5k
deadline mode → `VarunGiridhar3/qf_insertcard_h32_q1_005000`.
**Iter-1 on-robot: 15/20 (75%)** — dataset `rollout_insertcard_bc010000_bcq_q1_r1`
(40 eps: first 20 env-mis-set (8/20, excluded from the metric, kept for training),
latter 20 = the metric run).
**Q iter-2**: Q_1 → Q_2 on demo + all three rollout sets (60 eps; demo fraction
~34%, below the 50% contract — rollouts outgrew the demo set; G2 gate guards the
value function instead) → `VarunGiridhar3/qf_insertcard_h32_q2_005000`.
**Iter-2 on-robot: 16/20 (80%)** — dataset
`rollout_insertcard_bc010000_bcq_q2_r2_eval` (verified).

**Task-2 self-improvement curve: 25% → 40% → 75% → 80%** (BC frozen at 010000;
Q-only updates; +55 points total, +40 from two rounds on 100 autonomous episodes).

**Q iter-3 (final)**: Q_2 → Q_3 on demo + all four rollout sets (job 11777481,
5k; deployed at the 2.5k checkpoint under deadline —
`VarunGiridhar3/qf_insertcard_h32_q3_002500`; H100 queue starved 4h overnight,
resolved via a three-arm GPU race, one arm exposing that PACE A100s come in
40/80GB flavors — `--constraint=A100-80GB` required). Gate (job 11780943): same
shape as rounds 1–2 — (r1)/(r2) PASS, (r3) held-out small-n FAIL (the documented
long-episode confusion; not predictive of on-robot lift per rounds 1–2).
Iter-3 eval dataset requested as `rollout_insertcard_bc010000_bcq_q3_r3`. ⏳ on-robot.
**Task-2 Q_1 gate (job 11770322): rollout ranking FAIL on criterion (r3)** —
recorded honestly; deployment had already proceeded under deadline, on-robot
result is the decisive test. Details: G2 value function healthy (ρ 0.984,
failure/success terminals 0.05/0.98; (a) at the accepted chance level).
Rollout AUROC on ALL episodes 0.986 vs Q_0's 0.613 — the finetune fixed exactly
what it targeted (Q_0 scored real failures at 0.73!). But on the 5 HELD-OUT
rollout episodes the separation does not generalize: held-out failures score
0.58–0.74, one held-out success 0.05 — all five are near-timeout-length
episodes (T 417–491 ≈ 14–16 s), where a barely-succeeded terminal window and a
grinding-timeout terminal window look alike. Reading: 40 rollout episodes of a
hard bimanual task is thin for generalizable outcome prediction (stack-cups Q_1
generalized with 50); the deployment-relevant mechanism (candidate-chunk
ranking from in-distribution states) is not directly the failed criterion.
Fix path for any round 2: more episodes per round.

**Task-2 Q_2 gate (job 11774413): same shape as round 1 — FAIL on (r3) only.**
Value function healthy (ρ 0.990, terminals 0.04/0.98); all-episode AUROC 0.956
over 80 episodes (≥ Q_1's 0.793); held-out AUROC 0.350 (n=9). The confusions are
again concentrated in near-timeout-length episodes (held-out failures at T
417–491 scoring 0.67–0.95) — across two rounds, the consistent pattern is that
barely-succeeded and grinding-timeout terminal windows are visually alike on
this task; a limitations line for the paper. Round-1 precedent: this criterion
did not predict on-robot lift (iter-1 delivered 75% despite the same flag);
iter-2 on-robot is the decisive test.

**Stack-cups Q_3 gate (job 11770321): rollout ranking PASS** (AUROC ≥ 0.8, ≥
deployed-Q_2's, held-out sanity ✓; G2 b/c intact, (a) accepted) — the iter-3
deploy is formally gated.

Ops notes: draccus dict-override staging reused (`ic_q1_warmstart`, `q3_warmstart`);
double-merged demo had stale episodes self-pointers that broke merge_datasets
(repaired in place — only the merge tool follows them); wandb >64-char tag crash
fixed in `rl/wandb_utils.py`; task-2 BC full ladder 5k–30k published.

### Filtered BC arm (reviewer-requested; task #9, job 11752698)

"Feed the positive BC episodes back into the BC policy": ABC-DiT finetuned from the
deployed 005000 checkpoint on demo successes (101) + the 12 bc_r0 rollout successes,
5k steps lr 3e-5 (`bc_abcdit_filteredbc.sbatch`). Dataset
`VarunGiridhar3/stackcups_filteredbc_r1` = demo_right + AV1-transcoded rollout set
(the rig records H264; the merge concatenates video files so codecs must match —
frame-exact libsvtav1 transcode at the demo's settings, boundary decode verified
identical). **Caveat on record**: the deployed BC is undertrained (5k of 30k), so
part of any filtered-BC gain is extra training steps rather than feedback data;
BC100 is the ceiling reference, and the honest comparison vs the Q loop holds the
rollout data budget fixed (same bc_r0 episodes feed Q_1). Varun evals on-robot.

**Trained + gated + published (2026-08-08)**: 5k finetune steps (test loss 0.118,
action MSE 0.048 — base-run range). G1 replay on the 10 base-split held-out demos:
**PASS** (jMAE@8 0.0159, MAE/motion 0.437 < 0.5, gripper acc ≥ 0.98 at all
horizons); planner smoke with Q_0 also passed. Deploy ref:
**`VarunGiridhar3/abcdit_stackcups_right_filteredbc_r1_005000`** (private,
load_vision_weights=false; staged rsync copy at
`scratch/hf_upload/abcdit_stackcups_right_filteredbc_r1_005000`).

## Task 2 — insert card into wallet (BIMANUAL) — training launched 2026-08-08

Source: `VarunGiridhar3/KT_insert_card_into_wallet_20260808_{230253,233122}` (42 + 9
eps, same rig/recording stack: AV1, camera names `left_wrist`/`right_wrist`/`top`,
kinesthetic |act−st| ≈ 0.0014). **Both arms move** (left joint ranges up to 2.0 rad,
full gripper travel) ⇒ no right-arm derivation: 14-dim action/state, all 3 cameras;
the qplanning engine passes 14-dim policies through unchanged (by design).
Merged (labels sidecar carried, card_b episodes at 42+):
`VarunGiridhar3/KT_insert_card_into_wallet_20260808_merged` — 51 eps / 16,550 frames,
**43 success / 8 failure**, episodes longer than task 1 (mean 328 frames ≈ 11 s).

Convention (user-confirmed): BC trains on the 43 successes only — human-collected
failures never enter BC training (and "filtered BC" refers only to the
autonomous-rollout baseline); Q trains on all 51 episodes with the 8 failures as
real zero-return negatives via `episode_labels`.

Training chains: BC = ABC-DiT Base, task-1 recipe, 3 cams, 30k steps (probe 11753873
→ full 11753874, RTX6000 embers). Q = base L2 negatives recipe (the one that produced
the shipped task-1 value function), h=32, 3 cams, text on, 30k steps (probe 11753875
→ full 11753877, H100 inferno). Gate adaptations pending: G2's gripper-dim assumption
(14-dim has grippers at dims 6 and 13, not just last).

Step-count note (2026-08-09): 30k steps ≈ 68 epochs on 43 episodes — over-provisioned
vs task 1 (recipe copied for protocol identity); deployment candidates harvested
mid-run instead of retraining. G1 replay ranking (bimanual-generalized gate —
gripper dims now derived from action names, [6, 13]):

| ckpt (≈epochs) | jMAE@8 | @31 | grip acc min | MAE8/motion8 |
|---|---|---|---|---|
| 5k (11) | 0.0390 | 0.0612 | 0.969 | 1.164 |
| 10k (23) | 0.0341 | 0.0556 | 0.965 | 1.017 |
| 15k (34) | 0.0313 | 0.0558 | 0.969 | 0.935 |

Kinesthetic identity holds (15k: 0.022 joints / 0.015 gripper); the motion-ratio
"FAIL" is the task-1 criterion artifact amplified — insertion demos move so slowly
that 8-step true motion (~0.03 rad) matches the error scale. Absolute MAE still
improving at 15k → wait for the 20k/25k/30k tail (resume leg 11758306) and re-rank
before publishing deployment candidates.

## Checkpoint / dataset / job registry

| artifact | id / path |
|---|---|
| derived dataset | `VarunGiridhar3/KT_stack_cups_20260807_211827_right` |
| rollout datasets (r0, source) | `VarunGiridhar3/rollout_abcdit_stackcups_right_005000_bc_r0_20260808_165837`, `VarunGiridhar3/rollout_VarunGiridhar3_abcdit_stackcups_right_005000_bcq_negft010k_r0_20260808_194142`, filtered-BC eval `VarunGiridhar3/rollout_VarunGiridhar3_abcdit_stackcups_right_filteredbc_r1_005000_r0_20260809_000332` (50%) |
| rollout datasets (r1, requested) | `VarunGiridhar3/rollout_stackcups_bcq_q1_r1` (BC+Q_1), `VarunGiridhar3/rollout_stackcups_bc_r1` (drift control) — stamps appended at record time |
| rollout datasets (derived right-arm) | `VarunGiridhar3/rollout_stackcups_bc_r0_right`, `VarunGiridhar3/rollout_stackcups_bcq_r0_right` (pushed; local under `lerobot_cache/VarunGiridhar3/`) |
| filtered-BC dataset | `VarunGiridhar3/stackcups_filteredbc_r1` (local merge: demo + AV1 rollout; eps 114–143 = bc_r0) |
| BC ABC-DiT run | `qplanning-train worktree outputs/train/bc_abcdit_stackcups_base` ✅ (deployed ckpt 005000; BC100 ref 010000) |
| BC subset runs | bc25 job 11749949 ⏳, bc50 job 11749950 ⏳ |
| BC diffusion run | `outputs/train/bc_diffusion_stackcups_h32` (final leg 11751793) ⏳ |
| Q_0 run | `PACE outputs/train/qf_stackcups_h32{,_negft}` ✅ → `VarunGiridhar3/qf_stackcups_h32_negft_010000` |
| Q_1 round-1 run | `PACE outputs/train/qf_stackcups_h32_q1` (chain: probe 11753154 → full 11753155 → gate 11753156; first probe 11752552 failed on a draccus quirk — dict-typed `--policy.X` CLI overrides break under `--policy.path`, so the 3-repo `bucket_overrides` + `action_stats_repo_id` are baked into the warm-start copy at `scratch/q1_warmstart/pretrained_model`) ⏳ |
| filtered-BC run | worktree `outputs/train/bc_abcdit_filteredbc_r1` (probe 11752678 ✅, full 11752698) ⏳ |
| eval branches | `qplanning/yam-eval` (fork), `qplanning/yam-operator-eval` (leLab) — pushed, rig-validated |

## Rebuttal paragraph (~150 words) ⏳

Drafted after Phase 1 numbers exist. The honest curve goes in regardless of direction.
