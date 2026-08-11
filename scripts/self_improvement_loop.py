#!/usr/bin/env python
"""Self-improvement loop for FastWAM + Q-function planning.

Collects on-policy rollouts with Q-planning, fine-tunes the Q-function on all
collected episodes combined with the original training data, then repeats.
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
import faulthandler
import gc
import json
import logging
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Native MuJoCo / EGL failures otherwise terminate the process with SIGABRT and
# leave no Python-side location in the SLURM log.
faulthandler.enable(all_threads=True)

# ── Env / GPU setup ───────────────────────────────────────────────────────────

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Redirect HuggingFace cache away from the 20 GB home-dir quota.
os.environ.setdefault("HF_HOME", "/storage/project/r-agarg35-0/shared/huggingface_cache")


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
    rollout_diagnostics: bool = False,
) -> dict:
    """Run one episode and return a data dict regardless of success.

    Returns dict with keys: action (T,A), observation.images.* (T,3,H,W),
    next.success (T,), success (bool), task (str).
    """
    from copy import deepcopy

    import numpy as np
    from lerobot.envs.utils import add_envs_task, preprocess_observation
    from lerobot.utils.constants import ACTION

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

        if rollout_diagnostics:
            log.info("rollout seed=%s step=%d policy.select_action begin", seed, step)
        policy_started = time.monotonic()
        with torch.no_grad():
            action = policy.select_action(obs_tensor)
        policy_elapsed = time.monotonic() - policy_started
        action = postprocessor(action)
        action_trans = env_postprocessor({ACTION: action})
        action_np = action_trans[ACTION].cpu().numpy()
        if not np.isfinite(action_np).all():
            raise FloatingPointError(f"Non-finite action at seed={seed} step={step}: {action_np!r}")

        if rollout_diagnostics:
            log.info(
                "rollout seed=%s step=%d env.step begin policy_s=%.3f action_min=%.5f action_max=%.5f",
                seed,
                step,
                policy_elapsed,
                float(action_np.min()),
                float(action_np.max()),
            )
        env_step_started = time.monotonic()
        observation, reward, terminated, truncated, step_info = env.step(action_np)
        env_step_elapsed = time.monotonic() - env_step_started
        if rollout_diagnostics:
            log.info(
                "rollout seed=%s step=%d env.step done env_s=%.3f terminated=%s truncated=%s",
                seed,
                step,
                env_step_elapsed,
                bool(np.any(terminated)),
                bool(np.any(truncated)),
            )
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
    T = len(all_actions)
    action_tensor = torch.cat(all_actions, dim=0)  # (T, A)

    ep_dict: dict = {"action": action_tensor, "task": "", "success": ep_success}

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
    env_cfg,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    episodes_dir: Path,
    start_seed: int = 0,
    successful_only: bool = False,
    rollout_diagnostics: bool = False,
) -> tuple[float, int]:
    """Run episodes across all task envs and save them as a LeRobotDataset.

    Distributes n_episodes evenly across tasks. Saves to episodes_dir in LeRobot
    format (parquet + PNG images) for lazy per-frame loading. When
    ``successful_only`` is true, failed collection episodes are counted in the
    collection metric but omitted from Q replay. Standard Q self-improvement
    leaves this false so failures provide zero-return targets.

    Returns ``(collection_success_rate, n_replay_episodes)``.
    """
    from lerobot.envs.factory import make_env
    from lerobot.envs.libero import _get_suite, _select_task_ids
    from lerobot.policies.q_function.online_dataset import save_episodes_lerobot

    suite_names = [suite.strip() for suite in str(env_cfg.task).split(",") if suite.strip()]
    task_specs = []
    for suite_name in suite_names:
        suite = _get_suite(suite_name)
        task_specs.extend(
            (suite_name, task_id)
            for task_id in _select_task_ids(len(suite.tasks), env_cfg.task_ids)
        )

    n_tasks = len(task_specs)
    if n_episodes < n_tasks or n_episodes % n_tasks != 0:
        raise ValueError(
            f"n_episodes={n_episodes} must be a positive multiple of the {n_tasks} task envs "
            "so every task receives the same collection budget."
        )
    eps_per_task = n_episodes // n_tasks

    log.info(f"Collecting {n_episodes} episodes across {n_tasks} task envs ({eps_per_task} per task) ...")

    all_episode_dicts: list[dict] = []
    all_successes: list[bool] = []
    total_eps = 0

    for tg, tid in task_specs:
        if total_eps >= n_episodes:
            break
        # LIBERO collection is sequential, so keeping one EGL context per task
        # alive at once is unnecessary and has caused native read_pixels aborts.
        # Create only the task being collected, then close it before moving on.
        task_cfg = replace(env_cfg, task=tg, task_ids=[tid])
        log.info("Creating isolated LIBERO env for task %s/%d", tg, tid)
        task_envs = make_env(task_cfg, n_envs=1, use_async_envs=False)
        env = task_envs[tg][tid]
        try:
            task_success = 0
            for _ep_i in range(eps_per_task):
                seed = start_seed + total_eps
                log.info(
                    "Starting rollout %d/%d task=%s/%d seed=%d",
                    total_eps + 1,
                    n_episodes,
                    tg,
                    tid,
                    seed,
                )
                ep_dict = _run_episode(
                    env=env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    seed=seed,
                    rollout_diagnostics=rollout_diagnostics,
                )
                success = ep_dict["success"]
                all_successes.append(success)
                if success or not successful_only:
                    all_episode_dicts.append(ep_dict)
                total_eps += 1
                if success:
                    task_success += 1
            log.info(f"  Task {tg}/{tid}: {task_success}/{eps_per_task} success")
        finally:
            env.close()
            del env
            del task_envs
            gc.collect()
            log.info("Closed isolated LIBERO env for task %s/%d", tg, tid)

    if not all_episode_dicts:
        raise RuntimeError(
            "Collection produced no replay episodes. With successful_only=true, "
            "this means every rollout failed."
        )
    action_dim = int(all_episode_dicts[0]["action"].shape[1])
    save_episodes_lerobot(all_episode_dicts, episodes_dir, action_dim=action_dim)

    n_success = sum(all_successes)
    pc_success = 100.0 * n_success / max(len(all_successes), 1)
    log.info(
        f"  Total: {len(all_successes)} collected ({n_success} success, "
        f"{len(all_successes)-n_success} failure), retained={len(all_episode_dicts)}, "
        f"success={pc_success:.1f}%"
    )
    return pc_success, len(all_episode_dicts)


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
    original_q_ckpt_path: str = "",
    grad_clip_norm: float = 10.0,
    global_step: int = 0,
    use_wandb: bool = False,
) -> tuple[Path, float, int]:
    """Fine-tune Q on 50/50 mix of original + online data.

    Backbone LR is scaled by the same ratio used during original training
    (optimizer_lr_backbone / optimizer_lr), applied to the fine-tune LR.

    Returns (checkpoint_dir, mean_loss, updated_global_step).
    """
    from torch.optim import AdamW

    q_policy.train()

    n_orig = len(original_q_dataset)
    n_online = len(online_dataset)
    log.info(f"Fine-tuning Q: {n_orig} orig + {n_online} online frames, {steps} steps, lr={lr:.2e}")

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

    loader = DataLoader(combined, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=4, pin_memory=True, collate_fn=collate_shared_keys)

    # Build param groups preserving the backbone:head LR ratio from original training.
    # During training: head=optimizer_lr, backbone=optimizer_lr_backbone.
    # get_optim_params() returns [{head_params}, {backbone_params, lr=cfg.optimizer_lr_backbone}].
    # Without explicit override, AdamW uses the top-level lr for the head but the hardcoded
    # cfg.optimizer_lr_backbone (e.g. 9e-5) for the backbone — making backbone 9x faster than
    # the head at lr=1e-5. Fix: scale both groups by the same factor relative to training LRs.
    q_cfg = q_policy.config
    backbone_ratio = q_cfg.optimizer_lr_backbone / q_cfg.optimizer_lr
    param_groups = q_policy.get_optim_params()
    param_groups[0]["lr"] = lr                       # head
    param_groups[1]["lr"] = lr * backbone_ratio       # backbone at same ratio as training (0.3x)
    optimizer = AdamW(param_groups, weight_decay=1e-4)

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
        grad_norm = torch.nn.utils.clip_grad_norm_(q_policy.parameters(), grad_clip_norm)
        optimizer.step()
        q_policy.update()  # Polyak update target network

        step_loss = loss_dict["td_ce_loss"]
        losses.append(step_loss)
        global_step += 1

        if (step + 1) % 10 == 0:
            recent = sum(losses[-10:]) / 10
            # Surface the negatives signal so L0/L1/L2 are attributable: margin_loss
            # (0 when negatives off) + per-negative rank accuracy from the forward.
            margin = loss_dict.get("margin_loss", 0.0)
            rank_bits = " ".join(
                f"{k.split('/')[0].replace('neg_','')}={v:.2f}"
                for k, v in loss_dict.items() if k.startswith("neg_") and k.endswith("/rank_acc")
            )
            log.info(
                f"  step {step+1}/{steps}  td_ce={recent:.4f}  margin={margin:.4f}  "
                f"grad_norm={grad_norm:.3f}  {rank_bits}"
            )
            if use_wandb:
                import wandb
                wlog = {"finetune/loss": recent, "finetune/grad_norm": float(grad_norm), "finetune/lr": lr}
                # Pass through all scalar diagnostics from the forward (margin_loss,
                # neg_*/q_mean, neg_*/rank_acc, bucket_*/...) under finetune/.
                for k, v in loss_dict.items():
                    if isinstance(v, (int, float)):
                        wlog[f"finetune/{k}"] = float(v)
                wandb.log(wlog, step=global_step)

    q_policy.eval()

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    q_policy.save_pretrained(str(ckpt_dir))

    # Copy preprocessor/postprocessor files from the source Q checkpoint so the
    # saved checkpoint is self-contained and can be loaded by from_checkpoints().
    import shutil
    for fname in Path(original_q_ckpt_path).glob("policy_pre*"):
        shutil.copy2(fname, ckpt_dir / fname.name)
    for fname in Path(original_q_ckpt_path).glob("policy_post*"):
        shutil.copy2(fname, ckpt_dir / fname.name)
    log.info(f"  Saved Q checkpoint to {ckpt_dir}")

    mean_loss = sum(losses) / len(losses) if losses else float("nan")
    return ckpt_dir, mean_loss, global_step


# ─────────────────────────────────────────────────────────────────────────────
# Setup helpers
# ─────────────────────────────────────────────────────────────────────────────

def _setup_policy_and_env(args, device: torch.device):
    """Build FastWAM policy, Q planner, env config, and all processors."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
    from lerobot.policies.fastwam.planning import FastWAMPlanner
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.random_utils import set_seed

    set_seed(args.seed)

    env_cfg = LiberoEnv(
        task=args.task,
        observation_height=224,
        observation_width=224,
        episode_length=args.episode_length,
        stride_init_states=False,
        task_ids=args.task_ids,
    )
    log.info(
        "Creating LIBERO env: %s (horizon=%d, init_state=0 repeated)",
        args.task,
        args.episode_length,
    )
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
    if args.noise_smooth_sigma_t is not None:
        planning_cfg.noise_smooth_sigma_t = args.noise_smooth_sigma_t

    planner = FastWAMPlanner.from_checkpoints(
        cfg=planning_cfg,
        bc_post=postprocessor,
        bc_chunk_size=int(policy_cfg.chunk_size),
        device=device,
    )
    policy.attach_planner(planner)
    log.info("FastWAM + Q planner ready.")

    return policy, planner, env_cfg, preprocessor, postprocessor, env_preprocessor, env_postprocessor


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def self_improvement_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "loop_summary.jsonl"

    # ── Wandb ────────────────────────────────────────────────────────────────
    use_wandb = bool(args.wandb_project)
    if use_wandb:
        import wandb as _wandb
        run_name = args.wandb_run_name or f"{output_dir.name}"
        _wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=run_name,
            config=vars(args),
            dir=str(output_dir),
        )
        log.info(f"wandb run: {_wandb.run.get_url()}")

    # ── Set up FastWAM + env + preprocessors ─────────────────────────────────
    policy, planner, env_cfg, preprocessor, postprocessor, env_preprocessor, env_postprocessor = (
        _setup_policy_and_env(args, device)
    )

    # Q policy reference (lives inside planner.ctx)
    q_policy = planner.ctx.q_policy

    # ── Negative-Q paradigm override (controlled toggle) ─────────────────────
    # The in-loop finetune calls q_policy.forward(batch), which adds the
    # ranking-margin loss on batch-internal synthetic negatives whenever
    # config.neg_margin_weight > 0 (tube/swap/trev; see modeling_q_function
    # ._negatives_margin_loss). Overriding the loaded Q's config here lets every
    # arm start from the SAME Q checkpoint and differ only in the negatives
    # applied during self-improvement finetuning. Online frames are bucket
    # "q5" (q_bucket_index=0) so they are eligible negatives anchors.
    q_policy.config.neg_margin_weight = float(args.neg_margin_weight)
    if args.neg_margin_weight > 0:
        q_policy.config.neg_margin_delta = float(args.neg_margin_delta)
        q_policy.config.neg_use_swap = bool(args.neg_use_swap)
        q_policy.config.neg_use_temporal = bool(args.neg_use_temporal)
        q_policy.config.neg_tube_smooth_sigma_t = float(args.neg_tube_smooth_sigma_t)
        if args.neg_tube_sigmas is not None:
            sigmas = tuple(float(s) for s in str(args.neg_tube_sigmas).split(",") if s.strip())
            q_policy.config.neg_tube_sigmas = sigmas
        log.info(
            "Negatives ON: margin_weight=%.3f delta=%.3f tube_sigmas=%s smooth_t=%.1f "
            "swap=%s temporal=%s buckets=%s",
            q_policy.config.neg_margin_weight, q_policy.config.neg_margin_delta,
            q_policy.config.neg_tube_sigmas, q_policy.config.neg_tube_smooth_sigma_t,
            q_policy.config.neg_use_swap, q_policy.config.neg_use_temporal,
            q_policy.config.neg_buckets,
        )
    else:
        log.info("Negatives OFF (baseline loop): neg_margin_weight=0")

    q_pre_train = None
    original_q_ds = None
    if not args.collection_only:
        # Q training preprocessor (with Q-key passthrough)
        q_pre_train = _make_q_train_preprocessor(args.q_ckpt)

        # Original training dataset (loaded once, reused every iteration)
        original_q_ds = _load_original_q_dataset(
            repo_id=args.original_dataset_repo_id,
            dataset_root=args.original_dataset_root,
            q_policy=q_policy,
        )

    start_seed = (
        args.collection_start_seed
        if args.collection_start_seed is not None
        else args.seed + args.start_iteration * args.n_episodes
    )
    global_step = args.start_iteration * args.finetune_steps

    for i in range(args.n_iterations):
        iteration = args.start_iteration + i
        log.info(f"\n{'='*60}\nIteration {iteration}\n{'='*60}")
        final_iter_dir = output_dir / f"iter_{iteration:03d}"
        iter_dir = output_dir / f"iter_{iteration:03d}_incomplete"
        if final_iter_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite completed iteration directory: {final_iter_dir}"
            )
        if iter_dir.exists():
            log.warning("Removing incomplete iteration from an earlier interrupted attempt: %s", iter_dir)
            shutil.rmtree(iter_dir)
        iter_dir.mkdir(parents=True, exist_ok=False)
        t0 = time.time()

        # ── 1. Collect episodes ──────────────────────────────────────────────
        episodes_dir = iter_dir / "online_episodes"
        pc_success, n_replay_episodes = collect_episodes(
            policy=policy,
            env_cfg=env_cfg,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=args.n_episodes,
            episodes_dir=episodes_dir,
            start_seed=start_seed,
            successful_only=args.successful_only,
            rollout_diagnostics=args.rollout_diagnostics,
        )
        start_seed += args.n_episodes

        if args.collection_only:
            metrics = {
                "iteration": iteration,
                "collection_only": True,
                "n_collected": args.n_episodes,
                "n_replay_episodes": n_replay_episodes,
                "pc_success": pc_success,
                "collection_seed_start": start_seed - args.n_episodes,
                "collection_seed_end": start_seed - 1,
                "task_ids": args.task_ids,
                "episode_length": args.episode_length,
                "planner_type": args.planner_type,
                "n_samples": args.n_samples,
                "diffusion_steps": args.diffusion_steps,
                "elapsed_s": time.time() - t0,
            }
            (iter_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            iter_dir.rename(final_iter_dir)
            with open(summary_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")
            log.info(
                "Collection-only probe complete: collect=%.1f%% elapsed=%.1fs",
                pc_success,
                metrics["elapsed_s"],
            )
            break

        # ── 2. Build online dataset (growing buffer across all iterations) ──────
        from lerobot.policies.q_function.online_dataset import OnlineQDataset

        q_cfg = q_policy.config
        online_ds = OnlineQDataset(
            output_dir=output_dir,
            h=int(q_cfg.h),
            terminal_bonus=1.0,
            camera_keys=tuple(q_cfg.camera_keys),
        )
        log.info(f"Online dataset: {len(online_ds)} frames across {iteration + 1} iterations.")

        # ── 3. Fine-tune Q ───────────────────────────────────────────────────
        # FastWAM (~25 GB) is idle during finetune; offload it to CPU so the
        # ~1B-param Q + AdamW + DINOv2 activations fit on a 44 GB L40s. FastWAM
        # is reloaded to GPU before the next iteration's collection. q_policy is
        # reached via a plain attribute (planner._planner.ctx.q_policy), NOT a
        # registered submodule of `policy`, so policy.to("cpu") never moves it.
        offloaded = device.type == "cuda"
        if offloaded:
            policy.to("cpu")
            torch.cuda.empty_cache()
            log.info("Offloaded FastWAM to CPU for finetune (frees VRAM for Q training).")

        ckpt_dir = iter_dir / "q_checkpoint"
        _, mean_loss, global_step = finetune_q(
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
            original_q_ckpt_path=args.q_ckpt,
            grad_clip_norm=args.grad_clip_norm,
            global_step=global_step,
            use_wandb=use_wandb,
        )

        # ── 4. Q weights updated in-place — planner.ctx.q_policy is already updated ──
        if offloaded:
            torch.cuda.empty_cache()
            policy.to(device)  # reload FastWAM for the next iteration's collection / eval
            log.info("Reloaded FastWAM to GPU.")
        log.info("Q weights updated in-place (planner will use new weights on next call).")

        # ── 5. Eval with video clips ─────────────────────────────────────────
        eval_pc_success = None
        if args.eval_n_episodes > 0:
            from lerobot.envs.factory import make_env
            from lerobot.scripts.lerobot_eval import eval_policy_all

            log.info(f"Running eval ({args.eval_n_episodes} episodes) for clips ...")
            videos_dir = iter_dir / "eval_videos"
            eval_envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
            try:
                with torch.no_grad():
                    eval_info = eval_policy_all(
                        envs=eval_envs,
                        policy=policy,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=args.eval_n_episodes,
                        max_episodes_rendered=min(args.eval_n_episodes, 4),
                        videos_dir=videos_dir,
                        start_seed=args.seed + 100_000 + iteration * 1000,
                    )
            finally:
                for task_group in eval_envs.values():
                    for eval_env in task_group.values():
                        eval_env.close()
            overall = eval_info["overall"]
            eval_pc_success = overall.get("pc_success", None)
            log.info(f"  Eval: pc_success={eval_pc_success:.1f}%")

            if use_wandb:
                import wandb as _wandb
                log_dict = {
                    "collect/pc_success": pc_success,
                    "collect/n_collected": args.n_episodes,
                    "finetune/mean_loss": mean_loss,
                    "eval/pc_success": eval_pc_success,
                }
                if "avg_sum_reward" in overall:
                    log_dict["eval/avg_sum_reward"] = overall["avg_sum_reward"]
                _wandb.log(log_dict, step=global_step)
                for vp in overall.get("video_paths", [])[:2]:
                    try:
                        _wandb.log({"eval/video": _wandb.Video(vp, fps=10, format="mp4")}, step=global_step)
                    except Exception as e:
                        log.warning(f"Failed to log video {vp}: {e}")
        elif use_wandb:
            import wandb as _wandb
            _wandb.log(
                {"collect/pc_success": pc_success, "collect/n_collected": args.n_episodes, "finetune/mean_loss": mean_loss},
                step=global_step,
            )

        metrics = {
            "iteration": iteration,
            "n_collected": args.n_episodes,
            "n_replay_episodes": n_replay_episodes,
            "n_online_frames": len(online_ds),
            "pc_success": pc_success,
            "eval_pc_success": eval_pc_success,
            "finetune_loss": mean_loss,
            "collection_seed_start": start_seed - args.n_episodes,
            "collection_seed_end": start_seed - 1,
            "episode_length": args.episode_length,
            "init_state": 0,
            "successful_only": args.successful_only,
            "planner_type": args.planner_type,
            "n_samples": args.n_samples,
            "diffusion_steps": args.diffusion_steps,
            "elapsed_s": time.time() - t0,
        }
        (iter_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        iter_dir.rename(final_iter_dir)
        with open(summary_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        log.info(
            f"Iteration {iteration} done: collect={pc_success:.1f}% eval={eval_pc_success} "
            f"loss={mean_loss:.4f} elapsed={metrics['elapsed_s']:.1f}s"
        )

    if use_wandb:
        import wandb as _wandb
        _wandb.finish()

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
    p.add_argument(
        "--task_ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional LIBERO task IDs to collect; defaults to every task in the suite.",
    )
    p.add_argument("--episode_length", type=int, default=520)
    p.add_argument("--n_iterations", type=int, default=5)
    p.add_argument("--n_episodes", type=int, default=20)
    p.add_argument(
        "--collection_only",
        action="store_true",
        help="Collect and serialize rollouts, then exit without Q finetuning or evaluation.",
    )
    p.add_argument(
        "--collection_start_seed",
        type=int,
        default=None,
        help="Override the derived first rollout seed (used for targeted crash probes).",
    )
    p.add_argument(
        "--rollout_diagnostics",
        action="store_true",
        help="Log policy and environment timing immediately before and after every rollout step.",
    )
    p.add_argument(
        "--successful_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional ablation that drops failed rollouts; standard Q replay retains all rollouts.",
    )
    p.add_argument("--finetune_steps", type=int, default=200)
    p.add_argument("--finetune_lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--online_fraction", type=float, default=0.5)
    p.add_argument("--grad_clip_norm", type=float, default=10.0)
    # ── Negative-Q paradigms (applied during the in-loop Q finetune) ──────────
    # neg_margin_weight=0 → baseline loop (on-policy success/failure terminal
    # rewards only). >0 → add ranking-margin negatives on top.
    p.add_argument("--neg_margin_weight", type=float, default=0.0,
                   help="Weight on the ranking-margin negatives loss in finetune (0 = off).")
    p.add_argument("--neg_margin_delta", type=float, default=0.1)
    p.add_argument("--neg_tube_sigmas", type=str, default="1.0,2.0",
                   help="Comma-separated tube perturbation sigmas (smoothed per-dim noise).")
    p.add_argument("--neg_tube_smooth_sigma_t", type=float, default=2.0)
    p.add_argument("--neg_use_swap", action=argparse.BooleanOptionalAction, default=True,
                   help="Cross-batch wrong-chunk (roll) negatives.")
    p.add_argument("--neg_use_temporal", action=argparse.BooleanOptionalAction, default=True,
                   help="Time-reversed-chunk negatives.")
    p.add_argument("--planner_type", default="bc_diffusion_mppi")
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--n_elites", type=int, default=16)
    p.add_argument("--noise_smooth_sigma_t", type=float, default=None)
    p.add_argument("--diffusion_steps", type=int, default=3)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start_iteration", type=int, default=0,
                   help="Iteration index to start from (for chained jobs). iter dirs are named iter_{start_iteration + i}.")
    # eval
    p.add_argument("--eval_n_episodes", type=int, default=5,
                   help="Episodes to run for eval clips after each finetune (0 to skip).")
    # wandb
    p.add_argument("--wandb_project", default="",
                   help="W&B project name. Leave empty to disable wandb.")
    p.add_argument("--wandb_entity", default="",
                   help="W&B entity (team/user). Defaults to your default entity.")
    p.add_argument("--wandb_run_name", default="",
                   help="W&B run name. Defaults to si_<task>_n<n_episodes>_s<finetune_steps>.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    self_improvement_loop(args)
