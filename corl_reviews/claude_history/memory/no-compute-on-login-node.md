---
name: no-compute-on-login-node
description: Never run processing/compute scripts on the PACE login node — submit them as SLURM jobs under account gts-agarg35
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 20bffbde-75d5-40b5-b2fb-0692144115f8
  modified: 2026-08-10T00:51:01.172Z
---

Do not run processing scripts (video/image processing, model inference, data crunching)
directly on the login node. Submit them as SLURM jobs under account `gts-agarg35`. Ask the
user if GPU allocation is needed rather than assuming CPU-only.

**Why:** the login node is shared, and the user's cgroup there is capped at 4 GB (already
~3.4 GB used by other processes), so heavy jobs get SIGKILLed anyway — but the real reason
is cluster etiquette, which the user enforces.

**How to apply:** write an sbatch script, `sbatch` it, then poll with `squeue`/the output
file instead of running the work inline in Bash. Quick metadata reads (`ls`, small JSON,
a few `ffmpeg` single-frame extracts) are fine inline; anything that loops over many frames
or loads a model goes to a job. See [[qplanning-realrobot-state]].
