# Task prompt — Real-robot Q-planning + self-improvement (YAM, stack-cups)

Hand this to the downstream agent verbatim. Supports AzZT's "real robot experiments would be
awesome" and the AC's real-robot-viability discussion. The human (Varun, first author) operates
the robot; the agent trains, integrates, validates, and analyzes.

---

Task: produce the real-robot result for the Q-Planning CoRL rebuttal: (1) BC vs BC+Q on a real
single-arm cup-stacking task, (2) a planning-scheme comparison, (3) a Q-only self-improvement
loop over autonomous rollouts. You never run the robot; every robot interaction is a handoff to
the human. Robot episodes are the scarcest resource in this project — spend cluster hours to
save robot minutes.

READ FIRST: `corl_reviews/STATUS.md` (method context; the shipping planner is
`bc_diffusion_mppi`), `corl_reviews/latency_profile.md` (sim latency study — Phase 2 mirrors
its comparison), `corl_reviews/rules.md`. The planner code map in STATUS.md §"Where the planner
lives" applies here too.

DATA (verified 2026-08-07 — the dataset is ground truth for shapes/names; re-verify, don't
trust this prompt over `meta/info.json`):
- `VarunGiridhar3/KT_stack_cups_20260807_211827` — LeRobot **v3.0**, 114 episodes, 22 314
  frames @ 30 fps (~6.5 s/episode). Task: pick up one cup, stack it in the other. A second
  task's dataset will arrive later — parameterize everything by dataset id.
- Features: `action` and `observation.state` are **bimanual 14-dim** with named dims
  (left_joint_1..6, left_gripper = dims 0–6; right_joint_1..6, right_gripper = dims 7–13);
  cameras `left_wrist`, `right_wrist`, `top`, each 480×640 AV1 video.
- **Use only the right arm** (dims 7–13) and cameras `right_wrist` + `top`. Drop left-arm
  dims and `left_wrist` entirely. After slicing, the gripper is the last dim (index 6) —
  the planners' `gripper_dim=-1` default is then correct.
- `meta/episode_labels.json` holds per-episode `{outcome, success}` — the collection UI
  writes this. Not all demos are successes (e.g. episode 10 is a failure): split on this
  label; failures are real negatives for Q, never BC training data.
- Decide early: derive a single-arm dataset once (`scripts/lerobot_edit_dataset.py`, new repo
  id under the `VarunGiridhar3/` namespace) vs. slicing at load time. A derived dataset keeps
  every downstream config 7-dim-simple and is recommended; whichever you pick, use it
  consistently for BC and Q.

REPOS AND WHERE WORK HAPPENS:
- **This repo** (PACE): Q-function (`src/lerobot/policies/q_function/` — DINOv2-large + T5
  + 18-layer d=1024 decoder, HL-Gauss 101 bins), planners
  (`src/lerobot/policies/fastwam/planning.py`, `act_simple/planning.py`), Q training
  (`scripts/train_q_libero_ddp.sh` as template; reward/label pipeline under
  `experiments/mg_dataset_v1/q_dataset.py`), finetune-with-negatives machinery
  (`scripts/self_improvement_loop.py` — its finetune internals are reusable, its env-rollout
  collection is sim-only and is NOT), probes (`scripts/q_probe_battery.py`,
  `scripts/debug_q_values_dataset_replay.py`).
- **lerobot_collectedai** local copy: `/storage/project/r-agarg35-0/vgiridhar6/lerobot_collectedai`
  @ `feat/yam-real-world-integration` (codegraph initialized). ABC DiT was implemented in
  PR #21 (github.com/collected-ai/lerobot_collectedai/pull/21). FIRST verify whether that PR
  is merged into this branch and which policy dir it is (`multi_task_dit` and
  `diffusion_discrete_rtc` are the candidates on disk) — read the PR diff, don't guess.
- **leLab** (rollout UI) local copy: `/storage/project/r-agarg35-0/vgiridhar6/leLab`
  @ `feat/yam-operator` (codegraph initialized). Deployment host is **cai-1 (RTX 5090)**;
  the rollout path is leLab UI → lerobot_collectedai inference — NOT `lerobot-eval`. Search
  both repos for existing qplanning/eval hooks before building anything.
- Work-repo hygiene (from the robot owner's own instructions): create dedicated branches off
  the two `feat/` branches for eval mode, plus separate Desktop start/stop launchers
  (`~/Desktop/*` as reference) — Maya and Sasha collect data on the working stack daily and
  must never be affected. Do not push to their mainline branches; keep qplanning code on the
  dedicated branches only. Machine credentials are supplied by the human out-of-band — never
  write them into files, commits, or logs. Publish checkpoints/datasets under personal
  namespaces (`VarunGiridhar3/…` on HF), not the org's.

ARCHITECTURE (decided with the human — don't relitigate):
- BC = **ABC DiT** primary; lerobot `diffusion` policy as fallback if ABC DiT fights the
  planner integration. Both are trained per-task on ~100 demos; small models, quick runs.
- Self-improvement updates **Q only; BC stays frozen** after Phase 1. This is the point of
  the demo: improvement past a fixed BC policy attributable to the Q-function alone.

INTEGRATION CONTRACT the BC policy must satisfy (this is the main engineering lift; mirror
`fastwam`'s planner surface, found via codegraph):
  1. `predict_n_action_chunks(batch, N, num_inference_steps=…)` — N diverse chunks in ONE
     batched denoise (replicate conditioning across the batch dim, independent initial noise).
     lerobot diffusion's `predict_action_chunk(batch, noise=…)` already accepts injected
     noise — extend, don't rewrite. Verify batching empirically (N=1 vs N=32 wall-clock).
  2. `attach_planner` + `use_planning`/`planning` config fields, and standard processor
     pipelines so the planner's normalization round-trip works (BC-norm → raw via `bc_post`
     → Q-norm via `q_pre`). Getting the stats/pipelines wrong fails silently — see gate G2.
  3. Q horizon `h` must equal the scored chunk length exactly (loader enforces it). Pick the
     BC chunk size first (diffusion default horizon 16; U-Net needs a multiple of 8), then
     train Q with matching `h`. Replan interval `n_action_steps` sets the latency budget:
     budget_ms = n_action_steps / 30 fps × 1000 (e.g. 8 steps → 267 ms; 16 → 533 ms).
- Port the planner into the deployment path as a small self-contained module on the dedicated
  branch (the `bc_diffusion_*` branches of `plan_chunk_fastwam` + `_score_candidates_fast` +
  `encode_obs_context` are the only pieces needed at run time).
- Deployment gotchas (all must be handled on the eval branch): the robot runtime speaks
  14-dim bimanual actions — pad the policy's 7 with a left-arm hold pose (read the hold
  values from a real observation, not zeros); slice `observation.state` 14→7 to match
  training; keep obs preprocessing bit-identical to training (resize, channel order, [0,1]);
  eval mode must still write `episode_labels.json` — the self-improvement loop depends on it.

Q-FUNCTION SPECIFICS:
- `camera_keys=[observation.images.right_wrist, observation.images.top]` (DINOv2 resizes to
  224² internally; 480×640 input is fine). Language: single task → keep the constant task
  string (T5 caches it) or disable text; don't spend time here.
- Rewards for real data derive from `episode_labels.json` success + episode termination —
  adapt the q_dataset pipeline; document the exact reward definition in the report.
- With ~100 mostly-success demos, synthetic negatives (`neg_use_swap`, `neg_tube_sigmas`,
  `neg_use_temporal`) carry the contrastive signal initially; demo failures add real ones.
  Expect to sweep: negatives mix, `h`-consistent chunk size, LR, steps. Single GPU is enough.

COMPUTE (SLURM; see the slurm skill for mechanics — babysit every job):
- BC policy: RTX6000, account `gts-agarg35`, qos `embers` (preemptible — checkpoint + resume).
- Q function: L40s, account `gts-agarg35-ideas_l40s`, qos `inferno`; if the L40s pool is
  clogged, H100, account `gts-agarg35-ideasci23_dgx`, qos `inferno`.
- Latency: adapt `scripts/profile_planning.py` for the new stack; measure on cluster first,
  re-measure on cai-1's 5090 before any robot session. Only configs inside the replan budget
  go to the robot.
- W&B: log **offline** (`WANDB_MODE=offline`), entity `vgiridhar6`, project `awm` — set
  entity/project explicitly in every training config so runs sync to the right place if the
  human later runs `wandb sync`. This applies to all BC and Q runs, on both clusters' queues.

PHASES — ordered; each ends at a human gate. Never spend robot time on an unvalidated config.

Phase 0 — plumbing and offline validation (no robot).
  a. Verify dataset facts above; build the single-arm dataset; resolve the ABC DiT PR state.
  b. Train first BC + first Q; wire the planner.
  c. Offline gates before any rollout:
     G1: BC replay sanity — open-loop chunks on held-out demo observations track the demo.
     G2: Q ranking — on held-out episodes, Q(demo action) beats Q(perturbed/swapped/reversed)
         (reuse the probe scripts; rank-acc well above 0.5), and Q rises over time within
         successful episodes but not failed ones. This gate catches the normalization
         round-trip bug class; do not proceed past it on vibes.
     G3: planner smoke test on dataset observations — N candidates show real diversity and a
         nonzero Q-spread; argmax pick is sane to the eye.
     G4: latency — planning step fits the replan budget on the 5090 (estimate from cluster
         numbers, confirm on cai-1 once integration lands).
  d. Deliver the eval branches (lerobot_collectedai + leLab) with deploy instructions and the
     separate Desktop launchers.

Phase 1 — first result: BC vs BC+Q.
  Pre-register the protocol before the session: ≥20 episodes per condition, randomized +
  matched initial cup placements (same set of placements for both conditions), conditions
  interleaved within one session, fixed checkpoints, success = cup stacked stably (human
  judges, UI records). Report exact counts (x/N) and planning latency observed on-robot.
  ⚠ Headroom check: if BC-only ≥ ~90%, there is no room to demonstrate Q lift — stop and
  agree a lever with the human (train BC on a 25/50-demo subset, fewer BC steps, or harder
  placements) before burning more robot episodes.

Phase 2 — planning-scheme comparison (mirror the sim study, `latency_profile.md`).
  Offline-first: score schemes on latency + candidate diversity + offline Q-ranking; take
  only the top 2–3 to the robot at ~10–15 episodes each against matched placements.
  Grid: `bc_diffusion_argmax` vs `bc_diffusion_mppi` (top-K softmax), N ∈ {8, 16, 32},
  denoise steps ∈ {3, 5, BC-default}; noise-perturbation MPPI only as a stretch goal (it
  needs per-dim noise calibration in the new action space — skip unless time allows).
  Pick ONE shipping config; freeze it for Phase 3.

Phase 3 — self-improvement loop (repeat 2–3 rounds or until plateau).
  Contract per round k:
   1. Human: ≥20 autonomous rollouts with the frozen BC + current Q_k, uploads the rollout
      dataset (UI writes `episode_labels.json`), posts the dataset id.
   2. Agent: verify labels present and plausible; build the finetune mix — rollout successes
      as positives, rollout failures as REAL negatives, blended with original demo data
      (guard against forgetting: keep a demo fraction ≥ ~50% and a held-out demo slice for
      G2 re-checks); finetune Q only (small LR, e.g. 1e-5–3e-5, limited steps; log the
      negatives/rank-acc metrics the finetune tooling already emits).
   3. Gate: re-run G2 on held-out demos (must not degrade) + rank rollout failures below
      successes; if the gate fails, adjust (LR/steps/mix) instead of shipping a worse Q.
   4. Publish Q_{k+1} + exact deploy command; human rolls out again (same protocol,
      including a BC-only control every round — with BC frozen its rate should be stable;
      it doubles as a drift check on the rig).
  Headline curve: success rate vs round for BC-only (flat) and BC+Q_k (hopefully rising).

REPORT: `corl_reviews/realrobot_selfimprove_report.md` — per-phase tables (exact episode
counts, all conditions incl. the ones that lost), the self-improvement curve, observed
on-robot planning latency vs budget, the reward definition, all checkpoint/dataset ids and
job ids, and a ~150-word rebuttal paragraph. Report the true numbers: if Q-planning does not
beat BC, or a round of self-improvement regresses, that goes in the report as measured —
the rebuttal needs the honest curve, not a favourable one.

OPEN ITEMS the human will supply out-of-band: cai-1 access, rollout scheduling, the second
task's dataset id (rerun this prompt end-to-end for it when it lands).
