---
name: wandb-logging-convention
description: "W&B runs must be offline, entity vgiridhar6, project awm"
metadata: 
  node_type: memory
  type: project
  originSessionId: e11f16d4-1d38-4820-913f-2547b2cb644a
  modified: 2026-08-08T03:25:03.502Z
---

For training runs in this repo (BC policies, Q functions), Weights & Biases logging must be
**offline** (`WANDB_MODE=offline`) with entity `vgiridhar6` and project `awm` set explicitly
in the run config.

**Why:** compute nodes log offline; runs are synced later with `wandb sync`, and explicit
entity/project ensures they land in the right workspace (stated by Varun, 2026-08-07, while
setting up the real-robot self-improvement task).

**How to apply:** set `WANDB_MODE=offline` in job scripts and pass entity/project in the
training config (e.g. `--wandb.entity=vgiridhar6 --wandb.project=awm` or the repo's config
equivalent) for every submitted training job; include it when composing prompts for
downstream agents that will launch training.
