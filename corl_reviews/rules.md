## Compute 

This remains the main bottleneck in most cases. As such, strictly follow the following SLURM compute parameters to get compute fastest:

Use L40s gts-agarg35-ideas_l40s / H100s gts-agarg35-ideasci23_dgx / H200 gts-agarg35, all on qos on inferno. Use `sqme` output to gauge which GPU is in least demand, and use that. You should not require DDP training, so stick with single GPU.

## Rebuttal instructions

https://www.corl.org/contributions/instruction-for-rebuttal

## Pretrained Q functions and policies

### Just give me a working config

**LIBERO**
- BC policy (`--policy.path`): `/storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint`
- Q (`Q_CKPT`): `/storage/scratch1/6/vgiridhar6/lerobot/outputs/train/2026-05-19/23-54-42_qf_libero_ddp2_bsz48_bc_h200/checkpoints/last/pretrained_model`
- Launcher: `scripts/eval_fastwam_q_libero.sh` (both paths are its defaults)

**RoboTwin**
- BC policy: `/storage/project/r-agarg35-0/shared/fastwam/hf_checkpoint_robotwin` (24 G)
- Q (the committed 45k checkpoint): `/storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/q_checkpoints_backup/qf_robotwin_ddp_20260526_220319/045000/pretrained_model`
  - group-readable mirror: `/storage/project/r-agarg35-0/shared/qplanning_rebuttal_handoff/checkpoints/baseq_220319_045000/pretrained_model`
- Launcher: `scripts/eval_fastwam_q_robotwin.sh`

Every eval loads `pretrained_model/` only; `training_state/` is needed just to resume training.

### Full checkpoint inventory

64 G bundle at `/storage/project/r-agarg35-0/shared/qplanning_rebuttal_handoff/`, group-readable.
`analysis/CHECKPOINTS.md` in that bundle lists every RoboTwin Q on disk (run, step, role, path,
base-vs-tube); §2 is the table of what is physically staged. `HANDOFF.md` and `README.md` orient the rest.

Other staged Q's: `baseq_55k`, `baseq_220319_050000`, `recreated5k_005000`, `base174702_005000`,
`origQ_203525_{015000,030000,035000,040000}`, `tube_055000`.

One gap: the paper's exact eval-Q (`220319/005000`) was **deleted**. Two proxies are staged —
`recreated5k_005000` (July recipe-recreation) and `base174702_005000` (a contemporaneous same-recipe
step-5000). `conversations.md` records the decision to move to the **45k** checkpoint regardless.