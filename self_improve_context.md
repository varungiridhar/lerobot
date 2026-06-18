# FastWAM + Q-Function: Integration, Planning, and Self-Improvement Context

## Overview

This document captures the full context of work done integrating Q-function planning with FastWAM
policies in the LeRobot framework, with particular focus on the iterative self-improvement loop.
Intended as a handoff to a new Claude session working on self-improvement.

---

## 1. Repo and Environment

- **Repo**: `/storage/project/r-agarg35-0/igeorgiev3/lerobot` (also at `~/r-agarg35-0/lerobot`)
- **Active branch**: `qplanning`
- **Cluster**: PACE (Georgia Tech), GPU nodes — L40S (48 GB), H200 (80 GB)
- **FastWAM checkpoint (Libero)**: `/storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint`
- **FastWAM checkpoint (RoboTwin)**: `/storage/project/r-agarg35-0/shared/fastwam/hf_checkpoint_robotwin`
- **Q checkpoint (Libero)**: `/storage/scratch1/6/vgiridhar6/lerobot/outputs/train/2026-05-19/23-54-42_qf_libero_ddp2_bsz48_bc_h200/checkpoints/last/pretrained_model`
- **Q checkpoint (RoboTwin, best)**: `/storage/project/r-agarg35-0/vgiridhar6/robotwin/outputs/train/qf_robotwin_ddp_20260522_203525/checkpoints/last/pretrained_model` (40k steps)
- **Python env**: `conda activate lerobot` (at `~/r-agarg35-0/miniconda3/envs/lerobot`)

---

## 2. FastWAM Policy

FastWAM is a 6B diffusion VLA (Wan2.2-TI2V-5B backbone). Key config for RoboTwin checkpoint:
- `chunk_size = 32`, `n_action_steps = 24`, `action_dim = 14`
- `num_inference_steps = 10` (default; planning uses 3 for diversity)
- BF16 inference, ~25-31 GB VRAM
- Takes ~90 sec to load

**Important**: FastWAM for RoboTwin uses a single concatenated [3, 384, 320] image from three cameras
(head 256×320 top, left+right wrists 128×160 each side-by-side bottom). The `RoboTwinProcessorStep`
in `src/lerobot/processor/env_processor.py` handles this concat at eval time AND exposes individual
cameras under Q-function keys:
- `observation.images.image` → concatenated FastWAM input
- `observation.images.cam_high` → head camera (for Q function)
- `observation.images.cam_left_wrist` → left wrist (for Q function)
- `observation.images.cam_right_wrist` → right wrist (for Q function)

---

## 3. Q-Function Architecture

- **Model**: `QFunctionPolicy` in `src/lerobot/policies/q_function/modeling_q_function.py`
- **Backbone**: DINOv2-large (image encoder) + T5 (language encoder)
- **Input features (RoboTwin)**: `observation.state` (14-dim), 3 cameras at 256×256, `action`
- **Horizon**: h=32 (matches FastWAM chunk_size)
- **Output**: Distributional Q (categorical bins over [v_min, v_max])
- **Camera keys**: `observation.images.cam_high`, `observation.images.cam_left_wrist`, `observation.images.cam_right_wrist`

**Q preprocessor**: MEAN_STD normalization for images, state, and action — stats from training dataset.
Preprocessor saved at `<q_ckpt>/policy_preprocessor.json`.

**Known issue with eval Q values**: During eval, Q values stay compressed (~0.1–0.2) vs training
dataset evaluation (0.2→1.0 arc). Root cause: mild out-of-distribution shift between training demos
and eval rollouts (different seeds/randomization). The Q IS responding to observations (slow
monotonic increase confirmed), but the range is compressed. This affects planning quality but doesn't
break the pipeline.

---

## 4. Planning Integration

### Key files
- `src/lerobot/policies/fastwam/planning.py` — `FastWAMPlanner`, `plan_chunk_fastwam()`
- `src/lerobot/policies/fastwam/modeling_fastwam.py` — `FastWAMPolicy.select_action()`, `predict_n_action_chunks()`
- `src/lerobot/policies/act_simple/planning.py` — `_score_candidates_fast()`, `PlannerContext`
- `src/lerobot/scripts/lerobot_eval.py` — planner attachment, vis video generation

### Planner types
| Type | Description |
|------|-------------|
| `bc_diffusion_argmax` | N BC diffusion samples, pick argmax Q. N=1 = pure BC (for dry-run/vis) |
| `bc_diffusion_mppi` | N BC diffusion samples, softmax-weighted mean of top-K elites |
| `mppi` | Gaussian noise around BC mean, MPPI weighting |
| `argmax` | Gaussian noise, argmax |
| `cem` | Cross-entropy method |

### Best hyperparams so far (RoboTwin/Libero)
```bash
--policy.planning.planner_type=bc_diffusion_mppi
--policy.planning.n_samples=64
--policy.planning.n_elites=16
--policy.planning.noise_std=0.3
--policy.planning.temperature=1.0
--policy.planning.num_diffusion_steps=3
```

### Dry-run mode (BC execution + Q visualization, no optimization)
```bash
--policy.use_planning=true
--policy.planning.q_checkpoint_path=<Q_CKPT>
--policy.planning.planner_type=bc_diffusion_argmax
--policy.planning.n_samples=1
--policy.planning.num_diffusion_steps=10
```

### Bug fixes applied
- `candidates.dim() == 2` → unsqueeze(0) when N=1 (model squeezes sample dim)
- Camera key exposure in `RoboTwinProcessorStep` (already done)
- `mkdir -p "$TASK_DIR"` before `lerobot-eval | tee` to avoid SIGPIPE silent crash

### Planning visualization
- Generates `eval_episode_N_planning_vis.mp4` per episode
- Left panel: camera feed (now at full env fps with stride-1 frames, not stride-24)
- Right panel: Q-value spotlight plot (updates at chunk boundaries, holds between)
- Fix applied in `src/lerobot/policies/fastwam/planning_vis.py` + `modeling_fastwam.py` (record_step)

---

## 5. Q-Function Training

Q function is trained offline on demonstration datasets. Key training script:
`lerobot-train --policy.type=q_function ...`

Training dataset for RoboTwin: `local/robotwin2.0_multicam`
Training dataset for Libero: `HuggingFaceVLA/libero` (subset: `libero_10`)

**Important**: Use `resolve_delta_timestamps(cfg, ds_meta)` to compute action delta timestamps.
`cfg.action_delta_indices = list(range(0, 2*h))` = 64 entries for h=32.

The Q function uses **distributional RL** (categorical bins). Reward is terminal (1.0 at final
successful step, 0.0 otherwise). Bootstrap at t+h.

---

## 6. Self-Improvement Loop

### Concept
Run Q-planning eval → collect on-policy successful rollouts → fine-tune Q on combined
(original demos + online successes) → repeat. Goal: tighten Q estimates toward the policy's actual
distribution rather than the original BC demos.

### Implementation
- **Script**: `scripts/self_improvement_loop.py`
- **SLURM job**: `scripts/run_self_improvement.sh`

### Data flow
```
for iteration i:
  1. COLLECT  eval_policy() with Q-planning
             → filter: keep only successful episodes
             → save as episodes_iter{i}_{ep}.pt

  2. DATASET  OnlineQDataset(episodes.pt)
             → same __getitem__ as QValueLabelDataset
             → reward: 1.0 at last frame, 0.0 otherwise
             → bilinear upsample from 224→256 to match dataset

  3. FINETUNE ConcatDataset([original_q_dataset, online_dataset])
             → collate_shared_keys (filters to online dataset's keys only)
             → AdamW, constant LR ~1e-5
             → saves updated Q checkpoint

  4. UPDATE   reload Q weights into existing planner ctx (in-place)
```

### Key implementation details

**`OnlineQDataset`** (`src/lerobot/policies/q_function/online_dataset.py`):
- Wraps `.pt` episode files
- Episode file format:
  ```python
  {
      "obs": {
          "observation.images.image": Tensor[T, 3, H, W],  # float32 [0,1], 224x224
          "observation.state": Tensor[T, state_dim],
      },
      "action": Tensor[T, action_dim],   # raw env-space (post-postprocessor)
      "success": bool,
      "task": str,
  }
  ```
- **Critical**: `target_image_size=256` — bilinear upsample 224→256 to match original dataset
- Uses `F.interpolate(..., mode="bilinear", align_corners=False)` per frame

**`_load_original_q_dataset`** in `scripts/self_improvement_loop.py`:
- Use `resolve_delta_timestamps(cfg, ds_meta)` — NOT a manual loop over action indices
- `h=int(cfg.h)` — cast explicitly, avoids NameError
- Camera key mapping: online data uses `observation.images.image` (single concat); original dataset
  uses `observation.images.cam_high` etc. This mismatch is handled by `collate_shared_keys`

**`finetune_q`** in `scripts/self_improvement_loop.py`:
- Custom `collate_shared_keys` collate function:
  ```python
  online_keys = set(online_dataset[0].keys())
  def collate_shared_keys(batch):
      filtered = [{k: v for k, v in item.items() if k in online_keys} for item in batch]
      return default_collate(filtered)
  ```
  This is necessary because `ConcatDataset` mixes original dataset (many keys: timestamps,
  frame_index, episode_index, is_pad flags) with online dataset (only 8 essential keys).
- 50/50 split: half batch from original, half from online
- Uses `ctx.q_pre(batch)` before `q_policy.forward(batch)["loss"]`

**Q preprocessing in finetune**: Must call `ctx.q_pre` (the Q preprocessor) on every batch before
the forward pass. The Q preprocessor normalizes images (MEAN_STD) and actions.

**Image size handling**: The Q preprocessor expects 256×256 images. Online episodes have 224×224
images (from env). `OnlineQDataset` upsamples to 256×256. The upsample is done in `__getitem__`:
```python
if self.target_image_size is not None:
    s = self.target_image_size
    frame_t = F.interpolate(frame_t.unsqueeze(0), size=(s,s), mode="bilinear", align_corners=False).squeeze(0)
```

### Bugs fixed during development
1. **NameError `h`**: `resolve_delta_timestamps` removed the `h` variable from scope; fixed with `h=int(cfg.h)` in the `QValueLabelDataset` constructor call
2. **Image size mismatch** `[2,3,224,224] vs [2,3,256,256]`: Rollout images are 224×224 (env), original dataset is 256×256. Fixed with `target_image_size=256` in `OnlineQDataset`
3. **Action shape mismatch** `(7,)` vs `(64, 7)`: "action" not in `cfg.input_features`, so the manual delta_timestamps loop never set it. Fixed by using `resolve_delta_timestamps`
4. **Key mismatch collation error**: `ConcatDataset` has original (many keys) + online (8 keys). Fixed with `collate_shared_keys`

### Validated results
Prototype run on H200 (3 iterations, n_episodes=5, finetune_steps=50):
- ~2 min per iteration (eval + finetune)
- 80-100% success rate per iteration
- Loss ~2.4-2.6 (typical for distributional Q)
- Q checkpoint reloads cleanly after each iteration

### CLI prototype command
```bash
python scripts/self_improvement_loop.py \
    --fastwam_ckpt /storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint \
    --q_ckpt /storage/scratch1/6/vgiridhar6/lerobot/outputs/train/2026-05-19/23-54-42_qf_libero_ddp2_bsz48_bc_h200/checkpoints/last/pretrained_model \
    --original_dataset_repo_id HuggingFaceVLA/libero \
    --task libero_10 \
    --n_iterations 3 \
    --n_episodes 5 \
    --finetune_steps 50 \
    --finetune_lr 1e-5 \
    --batch_size 8 \
    --output_dir /storage/home/hcoda1/7/igeorgiev3/scratch/si_proto
```

### SLURM script
`scripts/run_self_improvement.sh` — H200, 8h. Auto-chains next iteration via `sbatch --dependency=afterok`.
Override params via `--export` flags on sbatch.

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--fastwam_ckpt` | (required) | Path to FastWAM checkpoint |
| `--q_ckpt` | (required) | Path to initial Q checkpoint |
| `--original_dataset_repo_id` | `HuggingFaceVLA/libero` | HF repo ID of the original training dataset |
| `--original_dataset_root` | `/storage/project/r-agarg35-0/shared/lerobot-data-2` | Local root for the dataset |
| `--task` | `libero_10` | LIBERO task split |
| `--n_iterations` | 5 | Number of self-improvement iterations |
| `--n_episodes` | 20 | Episodes to collect per iteration |
| `--finetune_steps` | 200 | Q fine-tuning gradient steps per iteration |
| `--finetune_lr` | 1e-5 | AdamW learning rate |
| `--batch_size` | 32 | Training batch size |
| `--online_fraction` | 0.5 | Fraction of each batch from online data |
| `--planner_type` | `bc_diffusion_mppi` | Planner for collection |
| `--n_samples` | 16 | Number of planning candidates |
| `--n_elites` | 16 | Number of elite candidates (MPPI) |
| `--diffusion_steps` | 3 | Diffusion steps for `bc_diffusion_*` planner |
| `--output_dir` | (required) | Where to save checkpoints and logs |
| `--seed` | 42 | Random seed |

### Output structure

```
output_dir/
  loop_summary.jsonl          # one JSON line per iteration
  iter_000/
    episodes/
      ep_0000.pt              # successful episode tensors
      ep_0001.pt
      ...
    q_checkpoint/
      config.json
      model.safetensors       # fine-tuned Q weights
    metrics.json              # per-iteration metrics
  iter_001/
    ...
```

Episode `.pt` file keys: `observation.images.image` (T,3,224,224), `observation.state` (T,state_dim), `action` (T,action_dim), `next.success` (T,), `task` (str).

`loop_summary.jsonl` fields per line: `iteration`, `n_collected`, `n_online_frames`, `pc_success`, `finetune_loss`, `elapsed_s`.

### Memory budget (one L40S, 46 GB)

| Component | VRAM |
|-----------|------|
| FastWAM (BF16, N=16 samples) | ~25 GB |
| Q-function (FP32 + AdamW) | ~4 GB |
| Batch (bsz=32, 256×256 images) | ~2 GB |
| **Total** | ~31 GB |

Use `--n_samples 16 --batch_size 32` on L40S. For H200 (80 GB) increase to `--n_samples 64 --batch_size 64`.

---

## 7. RoboTwin Evaluation (10-task benchmark)

### Tasks
beat_block_hammer, stack_blocks_two, place_shoe, handover_block, move_can_pot,
click_alarmclock, place_bread_basket, grab_roller, open_laptop*, pick_dual_bottles

*open_laptop has a RoboTwin env bug (`AttributeError: 'open_laptop' object has no attribute 'arm_tag'`).

### Eval scripts
- `scripts/run_robotwin_bc_10tasks.sh` — BC dry-run (bc_diffusion_argmax N=1, loads Q for vis)
- `scripts/run_robotwin_qplan_10tasks.sh` — Q-planning (bc_diffusion_mppi N=64, top-16)

### CRITICAL: Run sequentially, not in parallel
FastWAM takes ~25-31 GB. Two concurrent instances OOM. Always use `&&` to chain:
```bash
bash run_bc.sh && bash run_qplan.sh
```

### Results so far (in progress, 2026-05-25)
BC baseline (dry-run): 8/9 completable tasks at 90-100%
Q-planning: still running

---

## 8. Open Questions / Next Steps for Self-Improvement

1. **Q value distribution shift**: Q values at eval time are compressed (0.1–0.2 range) vs training
   (0.2–1.0). Self-improvement should help here — online rollouts match the eval distribution.

2. **Libero vs RoboTwin**: Self-improvement loop was validated on Libero. RoboTwin uses multi-camera
   inputs (cam_high, cam_left_wrist, cam_right_wrist) which complicates the `OnlineQDataset`
   — it needs to store and replay individual camera images, not just the concatenated FastWAM frame.

3. **Scaling**: Current prototype: 5 eps/iter. For meaningful improvement: 20-50 eps/iter.
   Each Libero episode ~30s, so 20 eps ≈ 10 min collect + 2 min finetune per iteration.

4. **Success rate filtering**: Only successful episodes used for fine-tuning. With BC success rates
   of 90-100% on easy tasks, there's plenty of signal. Harder tasks may need to lower the threshold
   or use partial successes.

5. **Potential improvement**: Use `_q_batch_to_transition` instead of `batch_to_transition` when
   building `q_pre` for the fine-tuning pipeline, to ensure Q reward keys pass through correctly.
   Currently using `batch_to_transition` which is fine for planning but was flagged for training.
