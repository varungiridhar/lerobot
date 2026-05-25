# Q-Function Self-Improvement Loop

Iteratively collects on-policy trajectories via FastWAM + Q-planning, fine-tunes
the Q-function on a 50/50 mix of original training data and successful rollouts,
then repeats. Runs in a single process on one GPU.

## Files

| File | Purpose |
|------|---------|
| `scripts/self_improvement_loop.py` | Main loop script |
| `scripts/run_self_improvement.sh` | SLURM job script |
| `src/lerobot/policies/q_function/online_dataset.py` | Dataset wrapper for collected `.pt` episodes |

## How it works

```
for iteration i:
  1. COLLECT   run FastWAM + Q-planning for n_episodes tasks
               → save successful episodes as iter_{i}/episodes/ep_*.pt
  2. DATASET   OnlineQDataset wraps the .pt files
               → same __getitem__ format as QValueLabelDataset
               → reward: 1.0 at terminal frame, 0.0 elsewhere
  3. FINETUNE  ConcatDataset(original_libero + online)
               → WeightedRandomSampler for 50/50 batch split
               → AdamW, constant LR, grad clip 1.0
               → q_policy.update() (Polyak) after each step
  4. UPDATE    Q weights updated in-place inside planner.ctx.q_policy
               → planner uses new weights on next iteration automatically
```

Only successful episodes are kept (unambiguous terminal reward signal).

## Quick start (interactive node)

```bash
cd /storage/project/r-agarg35-0/igeorgiev3/lerobot

export HF_HOME=/storage/project/r-agarg35-0/shared/huggingface_cache
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

python scripts/self_improvement_loop.py \
    --fastwam_ckpt /storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint \
    --q_ckpt /storage/scratch1/6/vgiridhar6/lerobot/outputs/train/2026-05-19/23-54-42_qf_libero_ddp2_bsz48_bc_h200/checkpoints/last/pretrained_model \
    --task libero_10 \
    --n_iterations 3 \
    --n_episodes 5 \
    --finetune_steps 50 \
    --output_dir /storage/home/hcoda1/7/igeorgiev3/scratch/si_test
```

## SLURM job

```bash
# Default run (20 eps/iter, 200 fine-tune steps, 5 iterations)
sbatch scripts/run_self_improvement.sh

# Override parameters
sbatch --export=ALL,N_ITERATIONS=10,N_EPISODES=30,FINETUNE_STEPS=500 \
    scripts/run_self_improvement.sh

# Use a different Q checkpoint (e.g. from a previous SI run)
sbatch --export=ALL,Q_CKPT=/path/to/iter_004/q_checkpoint \
    scripts/run_self_improvement.sh
```

## CLI arguments

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
| `--planner_type` | `bc_diffusion_mppi` | Planner for collection (`bc_diffusion_mppi` or `mppi`) |
| `--n_samples` | 16 | Number of planning candidates |
| `--n_elites` | 16 | Number of elite candidates (MPPI) |
| `--diffusion_steps` | 3 | Diffusion steps for `bc_diffusion_*` planner |
| `--output_dir` | (required) | Where to save checkpoints and logs |
| `--seed` | 42 | Random seed |

## Output structure

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

### Episode `.pt` file format

```python
{
    "observation.images.image":  Tensor[T, 3, 224, 224],  # float32 [0,1], pre-flip
    "observation.images.image2": Tensor[T, 3, 224, 224],
    "action":                    Tensor[T, 7],             # raw env-space actions
    "next.success":              Tensor[T],                # bool
    "task":                      str,
}
```

### `loop_summary.jsonl` fields

```json
{
  "iteration": 0,
  "n_collected": 5,
  "n_online_frames": 1287,
  "pc_success": 100.0,
  "finetune_loss": 2.41,
  "elapsed_s": 119.2
}
```

## Memory budget (one L40S, 46 GB)

| Component | VRAM |
|-----------|------|
| FastWAM (BF16, N=16 samples) | ~25 GB |
| Q-function (FP32 + AdamW) | ~4 GB |
| Batch (bsz=32, 256×256 images) | ~2 GB |
| **Total** | ~31 GB |

Use `--n_samples 16 --batch_size 32` on L40S. For H200 (140 GB) you can
increase to `--n_samples 64 --batch_size 64`.

## Implementation notes

- **Image resize**: Rollout images are 224×224 (env resolution). `OnlineQDataset`
  upsamples them to 256×256 (dataset resolution) before the Q preprocessor
  scales them back to 224×224. This keeps both datasets format-compatible.
- **Action window**: Uses `resolve_delta_timestamps(cfg, ds_meta)` to load 2h=64
  consecutive action steps from the original dataset, matching the Q forward's
  `action: (B, 2h, A)` expectation.
- **Key filtering**: Original dataset has extra keys (`observation.state`, `*_is_pad`,
  etc.) that online data lacks. A custom `collate_fn` keeps only the keys present
  in the online dataset (the superset of what Q forward actually needs).
- **Q reward keys**: `_q_batch_to_transition` (not `batch_to_transition`) is used
  as the pipeline's `to_transition` so that `q_reward_chunk_first`,
  `q_reward_pad_first`, `q_bootstrap_valid`, and `q_bucket_index` survive
  preprocessing.
- **In-place weight update**: `q_policy` is held by reference inside
  `planner.ctx.q_policy`. Fine-tuning updates it in-place so the planner
  automatically picks up new weights on the next iteration.
