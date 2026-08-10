---
name: qplanning-realrobot-state
description: "Live state pointers for the CoRL real-robot Q-planning task (checkpoints, branches, datasets, self-improvement curves)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 22ed5b4f-f441-4c62-98c4-c15b3ea27ab7
  modified: 2026-08-10T16:10:04.447Z
---

CoRL rebuttal real-robot task (corl_reviews/task_realrobot_prompt.md). **The living
report is `corl_reviews/realrobot_selfimprove_report.md` — read it first.**

**Headline curves (BC frozen, Q-only self-improvement on autonomous rollouts):**
- Stack cups (single-arm): BC 40% → +Q0 65% → Q1 75% → Q2 85% → Q3 ⏳; filtered-BC 50% (r2 ⏳)
- Insert card (bimanual, 14-dim, 3 cams): BC 25% → +Q0 40% → Q1 75% → Q2 80% → Q3 ⏳; filtered-BC 30% (r2 ⏳)
- Same rollout budget: filtered SFT +5..10 vs Q-planning +15..25 per round.

Key artifacts (all VarunGiridhar3/, private; staged rsync copies scratch/hf_upload/):
- BCs: abcdit_stackcups_right_005000 (frozen t1), abcdit_insertcard_010000 (frozen t2,
  full ladder 5k–30k up), filtered-BC r1 both tasks (r2 training).
- Qs: qf_stackcups_h32_{negft_010000,q1_010000,q2_007500,q3_005000};
  qf_insertcard_h32_{025000,q1_005000,q2_005000,q3_002500}. Chain = warm start the
  DEPLOYED previous gen; per-round bucket_overrides+action-stats-pin BAKED into
  scratch/{q,ic_q}*_warmstart copies (draccus dict CLI override broken under --policy.path).
- Datasets: KT_stack_cups_..._right (101s/13f), KT_insert_card_..._merged (43s/8f, has
  repaired episodes self-pointers — merge_datasets output needs meta/episodes
  chunk/file_index reset to 0 before re-merging); rollout sets rollout_{stackcups,insertcard}_*
  in lerobot_cache/VarunGiridhar3/ (rig recordings: rename head→top; bimanual kept 14-dim;
  task-1 sliced _right; AV1 transcode needed only for merge_datasets, not multi-repo Q).
- Eval branches: fork qplanning/yam-eval + leLab qplanning/yam-operator-eval (pushed;
  incl. camera-staleness safety fix: read_latest_tolerant, eval abort-continue,
  unconditional _hold_before_powerdown). cai-1 clones ~/lerobot_qplanning, ~/leLab-qplanning.
  UI: `<BC ref>@root +q <Q ref>@root`.
- Gates: q_gate2_real + q_gate_rollout_rank (gripper dims from action names; works 7/14-dim).
  Known pattern: (a) chance-level swap = accepted; task-2 (r3) held-out small-n FAIL every
  round (long-episode terminal-window confusion) — never predicted on-robot lift.
- Spread grid: candidate spread monotone in epochs-over-dataset; lift needs high-entropy BC.
- SLURM: H100 inferno (dgx acct) primary for Q; A100 needs --constraint=A100-80GB (40GB
  variant OOMs 3-cam batch-32); L40S inferno chronically jammed; embers RTX6000 instant.

Next: final eval numbers (Q3 both tasks + FBC2 both tasks) → close report → ~150-word
rebuttal paragraph. Latency backfill + Phase-2 planner grid remain optional extras.
