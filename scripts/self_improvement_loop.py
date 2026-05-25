#!/usr/bin/env python
"""Self-improvement loop for FastWAM + Q-function planning.

Collects on-policy rollouts with Q-planning, fine-tunes the Q-function on
successful episodes combined with the original training data, then repeats.
Runs in a single process on one GPU.

Usage:
    python scripts/self_improvement_loop.py \
        --fastwam_ckpt /storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint \
        --q_ckpt <path/to/q/checkpoint> \
        --task libero_10 \
        --n_iterations 3 \
        --n_episodes 5 \
        --finetune_steps 50 \
        --output_dir /storage/home/hcoda1/7/igeorgiev3/scratch/si_proto
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Env / GPU setup ───────────────────────────────────────────────────────────

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ─────────────────────────────────────────────────────────────────────────────
# Episode collection
# ─────────────────────────────────────────────────────────────────────────────


def _run_episode(
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    seed: int | None = None,
) -> dict | None:
    """Run one episode and return a data dict if successful, else None.

    Returns dict with keys: action (T,A), observation.images.image (T,3,H,W),
    observation.images.image2 (T,3,H,W), next.success (T,), task (str).
    """
    from copy import deepcopy

    import numpy as np
    from lerobot.envs.utils import add_envs_task, preprocess_observation
    from lerobot.utils.constants import ACTION, DONE

    policy.reset()
    observation, info = env.reset(seed=[seed] if seed is not None else None)

    all_obs_tensors: list[dict] = []
    all_actions: list = []
    all_successes: list = []
    done = np.array([False])
    max_steps = env.call("_max_episode_steps")[0]
    step = 0

    while not np.all(done) and step < max_steps:
        obs_tensor = preprocess_observation(observation)  # NCHW float32, unflipped
        # Save only tensor-valued image keys (skip robot_state dict).
        obs_saved = {
            k: deepcopy(v) for k, v in obs_tensor.items()
            if k.startswith("observation.images.") and isinstance(v, torch.Tensor)
        }
        all_obs_tensors.append(obs_saved)

        obs_tensor = add_envs_task(env, obs_tensor)
        obs_tensor = env_preprocessor(obs_tensor)
        obs_tensor = preprocessor(obs_tensor)

        with torch.no_grad():
            action = policy.select_action(obs_tensor)
        action = postprocessor(action)
        action_trans = env_postprocessor({ACTION: action})
        action_np = action_trans[ACTION].cpu().numpy()

        observation, reward, terminated, truncated, step_info = env.step(action_np)
        if "final_info" in step_info:
            final_info = step_info["final_info"]
            successes = final_info["is_success"].tolist()
        else:
            successes = [False]

        done = terminated | truncated | done
        all_actions.append(torch.from_numpy(action_np))
        all_successes.append(successes[0])
        step += 1

    ep_success = any(all_successes)

    if not ep_success:
        return None

    T = len(all_actions)
    action_tensor = torch.cat(all_actions, dim=0)  # (T, A)

    ep_dict: dict = {"action": action_tensor, "task": ""}

    # Stack images for each camera key.
    cam_keys = set()
    for obs in all_obs_tensors:
        cam_keys.update(obs.keys())
    for key in cam_keys:
        frames = [obs[key][0] for obs in all_obs_tensors if key in obs]  # [T x (3,H,W)]
        if len(frames) == T:
            ep_dict[key] = torch.stack(frames, dim=0)  # (T, 3, H, W)

    # Task string.
    if hasattr(env, "envs") and len(env.envs) > 0:
        task_attr = getattr(env.envs[0], "task_description", None) or getattr(env.envs[0], "task", None)
        if task_attr and isinstance(task_attr, str):
            ep_dict["task"] = task_attr

    ep_dict["next.success"] = torch.tensor(all_successes, dtype=torch.bool)
    return ep_dict


def collect_episodes(
    policy,
    envs,  # nested dict: {task_group: {task_id: VectorEnv}}
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    episodes_dir: Path,
    start_seed: int = 0,
) -> tuple[list[Path], float]:
    """Run episodes across all task envs and save successful ones.

    Distributes n_episodes evenly across tasks. Returns (saved .pt files, success rate).
    """
    task_envs = [(tg, tid, env) for tg, group in envs.items() for tid, env in group.items()]
    n_tasks = len(task_envs)
    eps_per_task = max(1, n_episodes // n_tasks)

    log.info(f"Collecting {n_episodes} episodes across {n_tasks} task envs ({eps_per_task} per task) ...")
    episodes_dir.mkdir(parents=True, exist_ok=True)

    all_files: list[Path] = []
    all_successes: list[bool] = []
    total_eps = 0
    file_idx = 0

    for tg, tid, env in task_envs:
        if total_eps >= n_episodes:
            break
        task_success = 0
        for ep_i in range(eps_per_task):
            seed = start_seed + total_eps
            ep_dict = _run_episode(
                env=env, policy=policy,
                env_preprocessor=env_preprocessor, env_postprocessor=env_postprocessor,
                preprocessor=preprocessor, postprocessor=postprocessor, seed=seed,
            )
            success = ep_dict is not None
            all_successes.append(success)
            total_eps += 1
            if success:
                path = episodes_dir / f"ep_{file_idx:04d}.pt"
                torch.save(ep_dict, str(path))
                all_files.append(path)
                file_idx += 1
                task_success += 1
        log.info(f"  Task {tg}/{tid}: {task_success}/{eps_per_task} success")

    pc_success = 100.0 * sum(all_successes) / max(len(all_successes), 1)
    log.info(f"  Total: {len(all_files)} successful eps, overall success={pc_success:.1f}%")
    return all_files, pc_success


# ─────────────────────────────────────────────────────────────────────────────
# Q fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def _load_original_q_dataset(
    repo_id: str,
    dataset_root: str,
    q_policy,
) -> "QValueLabelDataset":
    """Load the original LeRobot dataset wrapped as a QValueLabelDataset."""
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.q_function.q_value_labels import QValueLabelDataset

    cfg = q_policy.config

    log.info(f"Loading original dataset {repo_id} from {dataset_root} ...")
    ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(cfg, ds_meta)
    ds = LeRobotDataset(repo_id=repo_id, root=dataset_root, delta_timestamps=delta_timestamps)

    bucket_overrides = dict(cfg.bucket_overrides) if hasattr(cfg, "bucket_overrides") and cfg.bucket_overrides else {repo_id: "q5"}
    terminal_bonuses = dict(cfg.terminal_bonuses) if hasattr(cfg, "terminal_bonuses") and cfg.terminal_bonuses else {"q5": 1.0}
    step_reward = float(getattr(cfg, "step_reward", 0.0))

    q_ds = QValueLabelDataset(
        ds,
        h=int(cfg.h),
        step_reward=step_reward,
        terminal_bonuses=terminal_bonuses,
        reward_mode="sparse",
        bucket_overrides=bucket_overrides,
        load_preencoded=False,
    )
    log.info(f"  Loaded {len(q_ds)} frames from original dataset.")
    return q_ds


def _make_q_train_preprocessor(q_ckpt_path: str):
    """Load Q training preprocessor that preserves Q reward keys."""
    from lerobot.policies.q_function.processor_q_function import (
        _q_batch_to_transition,
        _q_transition_to_batch,
    )
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME

    return PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=q_ckpt_path,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        to_transition=_q_batch_to_transition,
        to_output=_q_transition_to_batch,
    )


def finetune_q(
    q_policy,
    q_pre_train,
    original_q_dataset,
    online_dataset,
    steps: int,
    lr: float,
    batch_size: int,
    online_fraction: float,
    device: torch.device,
    ckpt_dir: Path,
) -> Path:
    """Fine-tune Q on 50/50 mix of original + online data.

    Returns the path to the saved checkpoint directory.
    """
    from torch.optim import AdamW

    q_policy.train()

    n_orig = len(original_q_dataset)
    n_online = len(online_dataset)
    log.info(f"Fine-tuning Q: {n_orig} orig + {n_online} online frames, {steps} steps")

    combined = ConcatDataset([original_q_dataset, online_dataset])

    # Build weights: online_fraction of each batch from online data.
    orig_w = (1.0 - online_fraction) / n_orig
    online_w = online_fraction / n_online
    weights = [orig_w] * n_orig + [online_w] * n_online
    sampler = WeightedRandomSampler(weights, num_samples=steps * batch_size, replacement=True)

    # Online and original datasets have different keys (orig has observation.state, *_is_pad, etc.).
    # Use only the keys that the online dataset provides (superset of what Q forward needs).
    online_keys = set(online_dataset[0].keys())
    def collate_shared_keys(batch):
        from torch.utils.data._utils.collate import default_collate
        filtered = [{k: v for k, v in item.items() if k in online_keys} for item in batch]
        return default_collate(filtered)

    loader = DataLoader(combined, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=2, pin_memory=True, collate_fn=collate_shared_keys)

    optimizer = AdamW(q_policy.get_optim_params(), lr=lr, weight_decay=1e-4)

    losses = []
    data_iter = iter(loader)
    for step in range(steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # Preprocess: normalize action, move to device.
        batch = q_pre_train(batch)

        optimizer.zero_grad()
        loss, loss_dict = q_policy.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_policy.parameters(), 1.0)
        optimizer.step()
        q_policy.update()  # Polyak update target network

        losses.append(loss_dict["td_ce_loss"])
        if (step + 1) % 10 == 0:
            recent = sum(losses[-10:]) / 10
            log.info(f"  step {step+1}/{steps}  loss={recent:.4f}")

    q_policy.eval()

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    q_policy.save_pretrained(str(ckpt_dir))
    log.info(f"  Saved Q checkpoint to {ckpt_dir}")

    mean_loss = sum(losses) / len(losses) if losses else float("nan")
    return ckpt_dir, mean_loss


# ─────────────────────────────────────────────────────────────────────────────
# Setup helpers
# ─────────────────────────────────────────────────────────────────────────────

def _setup_policy_and_env(args, device: torch.device):
    """Build FastWAM policy, Q planner, env, and all processors."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
    from lerobot.policies.fastwam.planning import FastWAMPlanner
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.random_utils import set_seed

    set_seed(args.seed)

    env_cfg = LiberoEnv(task=args.task, observation_height=224, observation_width=224)
    log.info(f"Creating LIBERO env: {args.task}")
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)

    # PreTrainedConfig.from_pretrained dispatches on the 'type' field and returns FastWAMConfig.
    log.info("Loading FastWAM policy ...")
    policy_cfg = PreTrainedConfig.from_pretrained(args.fastwam_ckpt)
    policy_cfg.pretrained_path = args.fastwam_ckpt
    policy_cfg.device = str(device)

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    preprocessor_overrides = {"device_processor": {"device": str(device)}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.fastwam_ckpt,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    # Build Q planner config and attach planner.
    from lerobot.policies.act_simple.planning import PlanningConfig

    planning_cfg = PlanningConfig(
        q_checkpoint_path=args.q_ckpt,
        planner_type=args.planner_type,
        n_samples=args.n_samples,
        n_elites=args.n_elites,
        noise_std=0.3,
        temperature=1.0,
    )
    if args.diffusion_steps:
        planning_cfg.num_diffusion_steps = args.diffusion_steps

    planner = FastWAMPlanner.from_checkpoints(
        cfg=planning_cfg,
        bc_post=postprocessor,
        bc_chunk_size=int(policy_cfg.chunk_size),
        device=device,
    )
    policy.attach_planner(planner)
    log.info("FastWAM + Q planner ready.")

    return policy, planner, envs, preprocessor, postprocessor, env_preprocessor, env_postprocessor


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def self_improvement_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "loop_summary.jsonl"

    # Set up FastWAM + env + preprocessors
    policy, planner, envs, preprocessor, postprocessor, env_preprocessor, env_postprocessor = (
        _setup_policy_and_env(args, device)
    )

    # Q policy reference (lives inside planner.ctx)
    q_policy = planner.ctx.q_policy

    # Q training preprocessor (with Q-key passthrough)
    q_pre_train = _make_q_train_preprocessor(args.q_ckpt)

    # Original training dataset (loaded once, reused every iteration)
    original_q_ds = _load_original_q_dataset(
        repo_id=args.original_dataset_repo_id,
        dataset_root=args.original_dataset_root,
        q_policy=q_policy,
    )

    start_seed = args.seed

    for iteration in range(args.n_iterations):
        log.info(f"\n{'='*60}\nIteration {iteration}\n{'='*60}")
        iter_dir = output_dir / f"iter_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        # ── 1. Collect episodes ──────────────────────────────────────────────
        episodes_dir = iter_dir / "episodes"
        episode_files, pc_success = collect_episodes(
            policy=policy,
            envs=envs,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=args.n_episodes,
            episodes_dir=episodes_dir,
            start_seed=start_seed,
        )
        start_seed += args.n_episodes

        if not episode_files:
            log.warning("No successful episodes collected — skipping fine-tuning this iteration.")
            metrics = {
                "iteration": iteration,
                "n_collected": 0,
                "pc_success": pc_success,
                "finetune_loss": None,
                "elapsed_s": time.time() - t0,
            }
            with open(summary_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")
            continue

        # ── 2. Build online dataset ──────────────────────────────────────────
        from lerobot.policies.q_function.online_dataset import OnlineQDataset

        # Q was trained on images at the dataset's native resolution (e.g. 256×256).
        # Rollout images are at the env resolution (e.g. 224×224). Upsample to match
        # the original training dataset so the Q preprocessor normalizes consistently.
        q_cfg = q_policy.config
        orig_ds_sample = original_q_ds[0]
        orig_img_size = orig_ds_sample[list(q_cfg.camera_keys)[0]].shape[-1]  # e.g. 256
        online_ds = OnlineQDataset(
            episode_files=episode_files,
            h=int(q_cfg.h),
            terminal_bonus=1.0,
            camera_keys=tuple(q_cfg.camera_keys),
            target_image_size=orig_img_size,
        )
        log.info(f"Online dataset: {len(online_ds)} frames from {len(episode_files)} episodes.")

        # ── 3. Fine-tune Q ───────────────────────────────────────────────────
        ckpt_dir = iter_dir / "q_checkpoint"
        _, mean_loss = finetune_q(
            q_policy=q_policy,
            q_pre_train=q_pre_train,
            original_q_dataset=original_q_ds,
            online_dataset=online_ds,
            steps=args.finetune_steps,
            lr=args.finetune_lr,
            batch_size=args.batch_size,
            online_fraction=args.online_fraction,
            device=device,
            ckpt_dir=ckpt_dir,
        )

        # ── 4. Q weights updated in-place — planner.ctx.q_policy is already updated ──
        log.info("Q weights updated in-place (planner will use new weights on next call).")

        metrics = {
            "iteration": iteration,
            "n_collected": len(episode_files),
            "n_online_frames": len(online_ds),
            "pc_success": pc_success,
            "finetune_loss": mean_loss,
            "elapsed_s": time.time() - t0,
        }
        with open(summary_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        (iter_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        log.info(f"Iteration {iteration} done: loss={mean_loss:.4f}, elapsed={metrics['elapsed_s']:.1f}s")

    log.info(f"\nSelf-improvement loop complete. Summary: {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FastWAM + Q self-improvement loop")
    p.add_argument("--fastwam_ckpt", required=True)
    p.add_argument("--q_ckpt", required=True)
    p.add_argument("--original_dataset_repo_id", default="HuggingFaceVLA/libero")
    p.add_argument("--original_dataset_root", default="/storage/project/r-agarg35-0/shared/lerobot-data-2")
    p.add_argument("--task", default="libero_10")
    p.add_argument("--n_iterations", type=int, default=5)
    p.add_argument("--n_episodes", type=int, default=20)
    p.add_argument("--finetune_steps", type=int, default=200)
    p.add_argument("--finetune_lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--online_fraction", type=float, default=0.5)
    p.add_argument("--planner_type", default="bc_diffusion_mppi")
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--n_elites", type=int, default=16)
    p.add_argument("--diffusion_steps", type=int, default=3)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    self_improvement_loop(args)
