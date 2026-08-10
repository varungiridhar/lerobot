# Phase 1 protocol — BC vs BC+Q on real stack-cups (pre-registered)

Pre-registered BEFORE the robot session; deviations get logged in the report, not
silently absorbed. Operator: Varun. Robot minutes are the scarce resource — the session
plan below is ~2 h including resets.

## Conditions (both through the SAME `qplanning_sync` engine — identical padding/state
## path; they differ only in planning)

| | A: BC-only | B: BC+Q |
|---|---|---|
| request | `use_planning=false` | `use_planning=true`, `planner_type=bc_diffusion_mppi` |
| BC checkpoint | ⬜ fixed before session: `______` (step ____) | same checkpoint |
| Q checkpoint | — | ⬜ fixed: `______` (step ____) |
| planner params | — | ⬜ from G3/G4: N=___, K=___, steps=___, T=1.0, seed=42 |

Checkpoint + planner params frozen after gates G1–G4; write them here before the session.

## Episodes and placements

- **n = 20 per condition** (40 total). One extra pair allowed for a voided episode
  (hardware fault, operator error — void reasons logged; policy failures are NEVER voided).
- **Placement set**: 20 predefined cup-pair placements, defined once before the session
  (tape-marked grid on the table; photograph the marked table for the report). Spread:
  ~12 nominal (both cups in the demo distribution's core region), ~5 edge-of-distribution,
  ~3 hard (near workspace limits / cups close together). Each placement is used
  **exactly once per condition** — matched pairs.
- **Interleaving**: run placements in fixed order 1..20, conditions in ABBA blocks
  (1:A 1:B, 2:B 2:A, 3:A 3:B, …) within one session — cancels rig/lighting drift.

## Execution parameters

- `fps=30`, `n_action_steps=10` (replan every 333 ms), `max_rollout_s=30` (timeout ⇒
  failure; demos are ~7 s), `record_rollouts=true`,
  `dataset_repo_id=rollout_stackcups_p1` (auto-stamped), same task string "Stack cups".
- Arms re-home between episodes (standard eval flow); left arm holds captured pose.

## Success criterion

Cup stacked in/on the other cup, gripper released, stack stable for ~3 s untouched.
Human judges; top-button = success, grip-button = failure (UI writes
`episode_labels.json`). Borderline (cup lands but rocks off after release) = failure.

## Recorded per episode (mostly automatic)

Outcome (UI), placement id (operator log sheet), condition, wall-clock duration,
planning-step latency (from the debug JSONL telemetry; report median + p95 on-robot),
q_spread when planning (planner logs).

## Interim headroom check (MANDATORY)

After the first 5 ABBA blocks (10 episodes, 5 per condition): if BC-only is ≥ ~90 %
(5/5), STOP — there is no room to demonstrate Q lift. Agree a lever with the agent
before burning more episodes: (a) retrain BC on a 25/50-demo subset, (b) use an earlier
BC checkpoint, or (c) re-weight the placement set toward the hard region (then restart
the count with the new protocol noted).

## Report rows produced

`x/N` per condition (overall + per placement-difficulty tier), matched-pair win/loss/tie
table, observed latency vs the 333 ms budget, all checkpoint/dataset ids. Honest numbers
regardless of direction.
