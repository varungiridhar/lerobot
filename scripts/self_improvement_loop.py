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
# Redirect HuggingFace cache away from the 20 GB home-dir quota.
os.environ.setdefault("HF_HOME", "/storage/project/r-agarg35-0/shared/huggingface_cache")

# Training seeds start this far past the held-out eval seeds. Large enough that the two
# blocks can never overlap for any plausible task count or iteration index, which is what
# keeps the benchmark uncontaminated.
EVAL_SEED_STRIDE = 1_000_000


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

    # DSRL records one latent per planning step and needs to know which env step it was
    # taken at, so the record can be matched to a saved dataset frame later.
    planner = getattr(policy, "_planner", None)
    latent_planner = planner if hasattr(planner, "pop_episode_latents") else None

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

        if latent_planner is not None:
            latent_planner.note_env_step(step)

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
    if latent_planner is not None:
        ep_dict["latent_records"] = latent_planner.pop_episode_latents()
    return ep_dict


def _offline_image_shape(args) -> "tuple[int, int, int] | None":
    """(H, W, C) of the offline dataset's images, or None if it cannot be read."""
    import json
    from pathlib import Path as _P
    info = _P(args.original_dataset_root) / "meta" / "info.json"
    if not info.exists():
        return None
    feats = json.loads(info.read_text()).get("features", {})
    for k, v in feats.items():
        if k.startswith("observation.images.") and "shape" in v:
            return tuple(v["shape"])
    return None


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
    env_factory=None,
    shard: "tuple[int, int] | None" = None,
    camera_key_map: "dict[str, str] | None" = None,
    save_camera_keys: "tuple[str, ...] | None" = None,
    save_image_shape: "tuple[int, int, int] | None" = None,
    latent_pool_path: "Path | None" = None,
    n_eval_per_task: int = 0,
    n_train_per_task: "int | None" = None,
    eval_seed_base: int = 42,
) -> dict:
    """Run episodes across all task envs and save them as a LeRobotDataset.

    ``shard``: optional ``(k, n)`` — collect only tasks where ``index % n == k``.
    Lets N SLURM jobs collect disjoint task subsets in parallel, the same way the
    RoboTwin eval suite shards. Each shard writes its own dataset dir; the
    ``iter_*/online_episodes`` glob in OnlineQDataset reunites them at finetune time.

    ``env_factory``: optional ``(task_name) -> envs_dict`` callable. When given,
    ``envs`` is ignored and each task's env is built immediately before its episodes
    and closed straight after. RoboTwin needs this — holding all 50 SAPIEN sims
    open at once exhausts the job's memory, whereas a LIBERO suite's 10 envs fit.

    ``n_eval_per_task`` / ``n_train_per_task``: episodes per task, run in ONE pass.
    Held-out eval episodes use seeds fixed across iterations and are never saved, so the
    success rate stays a clean cross-iteration metric. Training episodes use advancing
    seeds and are the only ones written to disk. This replaces running a separate eval
    sweep over the same policy: one rollout serves both purposes.

    When ``n_eval_per_task`` is 0 the function behaves as before, splitting
    ``n_episodes`` evenly across tasks and saving all of them.

    Saves to episodes_dir in LeRobot format (parquet + PNG images) for lazy per-frame
    loading. Returns {"eval_pc_success", "train_pc_success", "n_eval", "n_train"};
    eval_pc_success is NaN when no held-out episodes were requested.
    """
    from lerobot.policies.q_function.online_dataset import save_episodes_lerobot

    if env_factory is not None:
        # (task_group, task_id, None) — the env is built inside the loop below.
        all_task_envs = [(t, 0, None) for t in env_factory.tasks]
    else:
        all_task_envs = [(tg, tid, env) for tg, group in envs.items() for tid, env in group.items()]

    # eps_per_task is computed over ALL tasks, then the shard takes its slice, so a
    # sharded run collects exactly the same episodes as an unsharded one.
    n_tasks = len(all_task_envs)
    eps_per_task = max(1, n_episodes // n_tasks)
    if n_train_per_task is None:
        n_train_per_task = eps_per_task if n_eval_per_task == 0 else eps_per_task
    per_task = n_eval_per_task + n_train_per_task
    if shard is not None:
        k, n = shard
        task_envs = [(tg, tid, e) for i, (tg, tid, e) in enumerate(all_task_envs) if i % n == k]
        # Seeds are derived from the task's index in the FULL list (below), not from a
        # running counter, so every shard reproduces the seeds it would have used.
        shard_index = {tg: i for i, (tg, _, _) in enumerate(all_task_envs) if i % n == k}
        log.info(f"Shard {k}/{n}: {len(task_envs)} of {n_tasks} tasks")
    else:
        task_envs = all_task_envs
        shard_index = {tg: i for i, (tg, _, _) in enumerate(all_task_envs)}

    log.info(f"Rollout over {len(task_envs)} tasks: {n_eval_per_task} held-out eval + "
             f"{n_train_per_task} training episodes each ({per_task * len(task_envs)} total)")

    all_episode_dicts: list[dict] = []
    all_successes: list[bool] = []
    eval_successes: list[bool] = []
    failed_tasks: list[str] = []
    total_eps = 0

    for tg, tid, env in task_envs:
        # The legacy n_episodes cap applies only in legacy mode. With explicit
        # per-task counts the caller has already said exactly how many episodes to
        # run, and applying the cap here would silently truncate the rollout.
        if shard is None and n_eval_per_task == 0 and total_eps >= n_episodes:
            break
        built_here = False
        task_success = 0
        eval_task_success = 0
        # A crash in one task must not discard every episode collected so far. Some
        # RoboTwin tasks raise from their own check_success() when driven by a learned
        # policy (e.g. open_laptop reads self.arm_tag, which only the scripted demo
        # play_once() ever assigns), and each eval task normally runs in its own
        # process, so this only became fatal once all 50 shared one.
        try:
            if env is None:
                env = env_factory(tg)  # returns a VectorEnv for this task
                built_here = True
            # Two blocks of episodes per task in ONE rollout pass.
            #
            #   held-out : seeds fixed across every iteration -> the comparable metric.
            #              NEVER saved for training, so the benchmark stays clean.
            #   training : seeds advance each iteration -> fresh states, saved to disk.
            #
            # Training on the held-out seeds would let the loop fit the very states its
            # score is computed on, and "iteration N improved" would be partly
            # memorisation with no way to separate it out.
            for _ep_i in range(n_eval_per_task + n_train_per_task):
                is_eval = _ep_i < n_eval_per_task
                if is_eval:
                    # Independent of start_seed, hence identical every iteration.
                    seed = eval_seed_base + shard_index[tg] * n_eval_per_task + _ep_i
                else:
                    # Offset past the held-out block so the two never collide.
                    j = _ep_i - n_eval_per_task
                    seed = (start_seed + EVAL_SEED_STRIDE
                            + shard_index[tg] * n_train_per_task + j)
                ep_dict = _run_episode(
                    env=env, policy=policy,
                    env_preprocessor=env_preprocessor, env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor, postprocessor=postprocessor, seed=seed,
                )
                success = ep_dict["success"]
                if is_eval:
                    eval_successes.append(success)
                    eval_task_success += int(success)
                else:
                    all_successes.append(success)
                    all_episode_dicts.append(ep_dict)
                    task_success += int(success)
                total_eps += 1
            parts = []
            if n_eval_per_task:
                parts.append(f"eval {eval_task_success}/{n_eval_per_task}")
            if n_train_per_task:
                parts.append(f"train {task_success}/{n_train_per_task}")
            log.info(f"  Task {tg}/{tid}: " + "  ".join(parts))
        except Exception as e:
            failed_tasks.append(tg)
            log.warning(f"  Task {tg}/{tid}: SKIPPED after {task_success} episodes — "
                        f"{type(e).__name__}: {e}")
        finally:
            if built_here and env is not None:
                try:
                    env.close()
                except Exception as e:  # a close failure must not abort collection
                    log.warning(f"  failed to close env for {tg}: {e}")

    if failed_tasks:
        # Surfaced explicitly: a silently shrinking task set would look like a real
        # change in success rate across iterations.
        log.warning(f"  {len(failed_tasks)} task(s) skipped due to env errors: {', '.join(failed_tasks)}")
    if not all_episode_dicts:
        raise RuntimeError("collect_episodes gathered 0 episodes — every task errored")

    # The env emits its own camera names (RoboTwin: head_camera/left_camera/
    # right_camera); the Q function indexes by its own (cam_high/cam_left_wrist/
    # cam_right_wrist). Rename before saving so OnlineQDataset can read the result.
    if camera_key_map:
        for ep in all_episode_dicts:
            for src, dst in camera_key_map.items():
                if src in ep:
                    ep[dst] = ep.pop(src)

    action_dim = int(all_episode_dicts[0]["action"].shape[1]) if all_episode_dicts else 7
    save_kwargs = {"action_dim": action_dim}
    if save_camera_keys:
        save_kwargs["camera_keys"] = save_camera_keys
    if save_image_shape:
        save_kwargs["image_shape"] = save_image_shape
    save_episodes_lerobot(all_episode_dicts, episodes_dir, **save_kwargs)

    # DSRL only: the per-planning-step latent pool, ordered to match the episodes just
    # written (save_episodes_lerobot consumes all_episode_dicts in this same order).
    if latent_pool_path is not None:
        from lerobot.policies.dsrl.planning_dsrl import save_latent_pool

        planner = getattr(policy, "_planner", None)
        img_hw = getattr(planner, "img_hw", None)
        if img_hw is None:
            raise RuntimeError(
                "DSRL collection produced no planner image size — the planner was never called."
            )
        save_latent_pool(
            [ep.get("latent_records", []) for ep in all_episode_dicts],
            latent_pool_path,
            img_hw=img_hw,
        )

    n_success = sum(all_successes)
    pc_success = 100.0 * n_success / max(len(all_successes), 1)
    eval_pc = (100.0 * sum(eval_successes) / len(eval_successes)) if eval_successes else float("nan")
    # Success over EVERY episode rolled out this iteration (held-out + training).
    # Legitimate as a performance estimate: the training episodes are rolled out with
    # the current Q *before* it is finetuned on them, so there is no leakage at
    # measurement time. It is the tighter point estimate (2x the episodes); the
    # held-out rate remains the metric to trend, because its seeds are identical every
    # iteration and so seed variance cancels in a paired comparison.
    _all = eval_successes + all_successes
    overall_pc = (100.0 * sum(_all) / len(_all)) if _all else float("nan")
    log.info(f"  Training set: {len(all_episode_dicts)} episodes saved, {pc_success:.1f}% success")
    if eval_successes:
        log.info(f"  HELD-OUT EVAL: {sum(eval_successes)}/{len(eval_successes)} = {eval_pc:.1f}% "
                 f"(fixed seeds, never trained on)")
    if _all:
        log.info(f"  OVERALL (all {len(_all)} episodes, held-out + training): {overall_pc:.1f}%")
    return {"eval_pc_success": eval_pc, "train_pc_success": pc_success,
            "overall_pc_success": overall_pc,
            "n_eval": len(eval_successes), "n_train": len(all_episode_dicts),
            "n_overall": len(_all)}


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
    """Build FastWAM policy, Q planner, env, and all processors."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv, RoboTwinEnv
    from lerobot.utils.constants import OBS_IMAGES
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
    from lerobot.policies.fastwam.planning import FastWAMPlanner
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.random_utils import set_seed

    set_seed(args.seed)

    if args.env_type == "robotwin":
        # A LIBERO suite name ("libero_10") expands to all its tasks in one make_env
        # call; RoboTwin has no suite concept, so build one env per task and merge.
        # The collection loop below already iterates over every (group, id, env).
        tasks = [t.strip() for t in args.task.split(",") if t.strip()]
        if not tasks:
            raise ValueError("--task must list at least one RoboTwin task")
        if not args.robotwin_root:
            raise ValueError("--robotwin_root is required when --env_type robotwin")
        log.info(f"RoboTwin: {len(tasks)} tasks, envs built lazily per task")

        class _RoboTwinEnvFactory:
            """Builds one task's env on demand; collection closes it when done."""

            def __init__(self, task_list, robotwin_root):
                self.tasks = task_list
                self._root = robotwin_root

            def __call__(self, task):
                cfg = RoboTwinEnv(task=task, robotwin_root=self._root)
                group = make_env(cfg, n_envs=1, use_async_envs=False)
                # make_env returns {task_group: {task_id: VectorEnv}}; unwrap to the env.
                inner = next(iter(group.values()))
                return next(iter(inner.values()))

        env_factory = _RoboTwinEnvFactory(tasks, args.robotwin_root)
        # _run_episode captures observations before the env preprocessor runs, so the
        # saved keys are RoboTwin's raw camera names. Map them onto the Q function's.
        env_factory.camera_key_map = {
            f"{OBS_IMAGES}.head_camera": f"{OBS_IMAGES}.cam_high",
            f"{OBS_IMAGES}.left_camera": f"{OBS_IMAGES}.cam_left_wrist",
            f"{OBS_IMAGES}.right_camera": f"{OBS_IMAGES}.cam_right_wrist",
        }
        # One env up front: make_policy and the env processors need a concrete cfg,
        # and they depend on the env type rather than the individual task.
        env_cfg = RoboTwinEnv(task=tasks[0], robotwin_root=args.robotwin_root)
        envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    else:
        env_factory = None
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

    # "dsrl" is not one of PlanningConfig's action-space planners; it only borrows the
    # config for the Q checkpoint path and the denoising-step count.
    planning_cfg = PlanningConfig(
        q_checkpoint_path=args.q_ckpt,
        planner_type="bc_diffusion_mppi" if args.planner_type == "dsrl" else args.planner_type,
        n_samples=args.n_samples,
        n_elites=args.n_elites,
        noise_std=0.3,
        temperature=1.0,
    )
    if args.diffusion_steps:
        planning_cfg.num_diffusion_steps = args.diffusion_steps
    if args.noise_smooth_sigma_t is not None:
        planning_cfg.noise_smooth_sigma_t = args.noise_smooth_sigma_t

    if args.planner_type == "dsrl":
        from lerobot.policies.dsrl.planning_dsrl import DSRLPlanner

        # Resume: a chained job restarts the process, so pick up the newest saved agent
        # rather than starting from an untrained one (which would silently revert to BC).
        agent_path = None
        prior = sorted(Path(args.output_dir).glob("iter_*/dsrl_agent.pt"))
        if prior:
            agent_path = prior[-1]
        # n_pool + the executed sample = n_samples decodes per planning step, i.e. the same
        # diffusion budget per step as the Q-Planning arm it is being compared against.
        planner = DSRLPlanner.from_checkpoints(
            cfg=planning_cfg,
            bc_post=postprocessor,
            bc_chunk_size=int(policy_cfg.chunk_size),
            device=device,
            bc_action_dim=int(policy_cfg.action_dim),
            dsrl_cfg_overrides={
                "tiled": not args.dsrl_full_latent,
                "b_w": args.dsrl_b_w,
                "hidden": args.dsrl_hidden,
                "n_critics": args.dsrl_n_critics,
                "explore_std": args.dsrl_explore_std,
            },
            n_pool=max(0, args.n_samples - 1),
            agent_path=agent_path,
        )
        log.info("FastWAM + DSRL latent-steering planner ready.")
    else:
        planner = FastWAMPlanner.from_checkpoints(
            cfg=planning_cfg,
            bc_post=postprocessor,
            bc_chunk_size=int(policy_cfg.chunk_size),
            device=device,
        )
        log.info("FastWAM + Q planner ready.")
    policy.attach_planner(planner)

    return (policy, planner, envs, preprocessor, postprocessor,
            env_preprocessor, env_postprocessor, env_factory)


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

    # ── Set up policy + env, unless this job only finetunes ──────────────────
    # A --skip_collect job runs no rollouts, so FastWAM (6B params) and the sim env
    # are pure overhead: loading FastWAM costs ~4 min, ~12 GiB of VRAM, and leaves the
    # allocator fragmented. Offloading it to CPU afterwards was not enough — an 80 GiB
    # H100 still OOM'd once Q's params, grads, Adam state, target network and batch-48
    # DINOv2 activations were stacked on top. Load only the Q function.
    if args.skip_collect:
        from lerobot.policies.q_function.modeling_q_function import QFunctionPolicy
        log.info("--skip_collect: loading Q only (no FastWAM, no env).")
        q_policy = QFunctionPolicy.from_pretrained(args.q_ckpt).to(device).eval()
        policy = planner = envs = None
        preprocessor = postprocessor = None
        env_preprocessor = env_postprocessor = env_factory = None
    else:
        (policy, planner, envs, preprocessor, postprocessor, env_preprocessor,
         env_postprocessor, env_factory) = _setup_policy_and_env(args, device)
        # Q policy reference (lives inside planner.ctx)
        q_policy = planner.ctx.q_policy
    is_dsrl = args.planner_type == "dsrl"

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

    # Q training preprocessor (with Q-key passthrough)
    q_pre_train = _make_q_train_preprocessor(args.q_ckpt)

    # Original training dataset (loaded once, reused every iteration)
    original_q_ds = _load_original_q_dataset(
        repo_id=args.original_dataset_repo_id,
        dataset_root=args.original_dataset_root,
        q_policy=q_policy,
    )

    start_seed = args.seed + args.start_iteration * args.n_episodes
    global_step = args.start_iteration * args.finetune_steps

    for i in range(args.n_iterations):
        iteration = args.start_iteration + i
        log.info(f"\n{'='*60}\nIteration {iteration}\n{'='*60}")
        iter_dir = output_dir / f"iter_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        # ── 1. Collect episodes ──────────────────────────────────────────────
        pc_success = float("nan")
        heldout_pc = float("nan")
        n_heldout = 0
        overall_pc = float("nan")
        n_overall = 0
        if not args.skip_collect:
            shard = None
            if args.task_shard:
                k, n = (int(x) for x in args.task_shard.split("/"))
                if not 0 <= k < n:
                    raise ValueError(f"--task_shard K/N needs 0 <= K < N, got {args.task_shard}")
                shard = (k, n)
            # Shards write to sibling iter_<i>_shard<k> dirs. OnlineQDataset globs
            # "iter_*/online_episodes", so the finetune step reunites them with no
            # merge step — but only if the shard dir keeps that exact leaf name.
            iter_out = output_dir / f"iter_{iteration:03d}_shard{shard[0]:02d}" if shard else iter_dir
            iter_out.mkdir(parents=True, exist_ok=True)
            episodes_dir = iter_out / "online_episodes"
            stats = collect_episodes(
                policy=policy,
                envs=envs,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=args.n_episodes,
                episodes_dir=episodes_dir,
                env_factory=env_factory,
                start_seed=start_seed,
                shard=shard,
                # Both suites drive off the Q function's own camera keys: LIBERO's
                # happen to equal online_dataset's module default, RoboTwin's do not.
                # The rename map is set by the env factory and is None for LIBERO,
                # whose raw env keys already match.
                camera_key_map=getattr(env_factory, "camera_key_map", None),
                save_camera_keys=tuple(q_policy.config.camera_keys),
                # Match the offline demos' stored resolution exactly. The env renders
                # at a lower resolution than the demos were recorded at (RoboTwin:
                # D435 320x240 vs Large_D435 640x480), and finetune_q batches the two
                # datasets together, so a mismatch fails collate. Q resizes everything
                # to 224 internally, so this only affects storage, not what Q sees.
                save_image_shape=_offline_image_shape(args),
                latent_pool_path=(iter_out / "latent_pool.npz") if is_dsrl else None,
                n_eval_per_task=args.n_eval_per_task,
                n_train_per_task=args.n_train_per_task,
                eval_seed_base=args.eval_seed_base,
            )
            pc_success = stats["train_pc_success"]
            heldout_pc = stats["eval_pc_success"]
            n_heldout = stats["n_eval"]
            overall_pc = stats["overall_pc_success"]
            n_overall = stats["n_overall"]
            if args.collect_only:
                log.info(f"--collect_only: wrote {episodes_dir} "
                         f"({stats['n_train']} train eps, {pc_success:.1f}% success; "
                         f"{stats['n_eval']} held-out eval eps, {heldout_pc:.1f}%). Done.")
                # Held-out results live beside the episodes so the finetune job can
                # aggregate them across shards without re-running anything.
                (iter_out / "heldout_eval.json").write_text(json.dumps(stats, indent=2))
                return
        else:
            log.info("--skip_collect: finetuning on previously collected episodes.")
        start_seed += args.n_episodes

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
        offloaded = device.type == "cuda" and policy is not None
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

        # ── 3b. DSRL: distil the updated Q^A into Q^W, then fit the latent actor ──────
        # Runs while FastWAM is still offloaded: every pi_dp(s, w) decode this needs was
        # already done during collection, so only Q^A and two small MLPs are resident.
        dsrl_metrics = None
        if is_dsrl:
            from lerobot.policies.dsrl.latent_dataset import build_dsrl_training_data

            log.info("DSRL: rebuilding latent targets against the updated Q ...")
            t_dsrl = time.time()
            feats, w_pool, q_pool = build_dsrl_training_data(
                output_dir=output_dir,
                ctx=planner.ctx,
                device=device,
                batch_states=args.dsrl_target_batch,
                max_states=args.dsrl_max_states,
            )
            planner.agent.to(device)
            dsrl_metrics = planner.agent.fit(
                feats=feats, w_pool=w_pool, q_pool=q_pool,
                steps=args.dsrl_steps, batch_size=args.dsrl_batch_size,
            )
            dsrl_metrics["elapsed_s"] = time.time() - t_dsrl
            planner.agent.save(iter_dir / "dsrl_agent.pt")
            log.info(
                "DSRL: actor_Q=%.4f vs prior mean %.4f / prior best-of-%d %.4f  "
                "(critic_mse=%.5f, %d states, %.0fs)",
                dsrl_metrics["actor_q"], dsrl_metrics["q_prior_mean"],
                int(dsrl_metrics["n_pool"]), dsrl_metrics["q_prior_max_mean"],
                dsrl_metrics["critic_mse"], int(dsrl_metrics["n_states"]),
                dsrl_metrics["elapsed_s"],
            )
            if use_wandb:
                import wandb as _wandb
                _wandb.log({f"dsrl/{k}": v for k, v in dsrl_metrics.items()}, step=global_step)

        # ── 4. Q weights updated in-place — planner.ctx.q_policy is already updated ──
        if offloaded:
            torch.cuda.empty_cache()
            policy.to(device)  # reload FastWAM for the next iteration's collection / eval
            log.info("Reloaded FastWAM to GPU.")
        log.info("Q weights updated in-place (planner will use new weights on next call).")

        # ── 5. Eval with video clips ─────────────────────────────────────────
        eval_pc_success = None
        if args.eval_n_episodes > 0:
            from lerobot.scripts.lerobot_eval import eval_policy_all

            log.info(f"Running eval ({args.eval_n_episodes} episodes) for clips ...")
            videos_dir = iter_dir / "eval_videos"
            # Eval is the deployed policy: deterministic actor, no exploration noise, and
            # no prior pool — one decode per planning step instead of n_samples.
            if is_dsrl:
                planner.set_mode("eval")
            with torch.no_grad():
                eval_info = eval_policy_all(
                    envs=envs,
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
            if is_dsrl:
                planner.set_mode("collect")
            overall = eval_info["overall"]
            eval_pc_success = overall.get("pc_success", None)
            log.info(f"  Eval: pc_success={eval_pc_success:.1f}%")

            if use_wandb:
                import wandb as _wandb
                log_dict = {
                    "heldout/pc_success": heldout_pc,
                    "heldout/n_episodes": n_heldout,
                    "overall/pc_success": overall_pc,
                    "overall/n_episodes": n_overall,
                    "collect/train_pc_success": pc_success,
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
                {"heldout/pc_success": heldout_pc, "heldout/n_episodes": n_heldout,
                 "overall/pc_success": overall_pc, "overall/n_episodes": n_overall,
                 "collect/train_pc_success": pc_success, "collect/pc_success": pc_success,
                 "collect/n_collected": args.n_episodes, "finetune/mean_loss": mean_loss},
                step=global_step,
            )

        metrics = {
            "iteration": iteration,
            "n_collected": args.n_episodes,
            "n_online_frames": len(online_ds),
            # THE headline number: success on held-out episodes whose seeds are fixed
            # across iterations and never trained on. This is the only figure that is
            # comparable between iterations — compare it against the pre-SI baseline.
            "heldout_pc_success": heldout_pc,
            "n_heldout": n_heldout,
            # All episodes this iteration. Tighter estimate of current performance;
            # use heldout_pc_success for the cross-iteration trend.
            "overall_pc_success": overall_pc,
            "n_overall": n_overall,
            # Success on the TRAINING episodes. Their seeds advance every iteration, so
            # this moves with task difficulty as well as policy quality: useful for
            # spotting collapse, not valid as a trend.
            "train_pc_success": pc_success,
            # Deprecated alias kept so older readers of loop_summary.jsonl keep working.
            "pc_success": pc_success,
            "eval_pc_success": eval_pc_success,
            "finetune_loss": mean_loss,
            "elapsed_s": time.time() - t0,
        }
        if dsrl_metrics is not None:
            metrics["dsrl"] = dsrl_metrics
        with open(summary_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        (iter_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        log.info(
            f"Iteration {iteration} done: OVERALL={overall_pc:.1f}% (n={n_overall}) "
            f"HELD-OUT={heldout_pc:.1f}% (n={n_heldout}) train={pc_success:.1f}% "
            f"loss={mean_loss:.4f} "
            f"elapsed={metrics['elapsed_s']:.1f}s"
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
    p.add_argument("--task", default="libero_10",
                   help="LIBERO suite name, or a comma-separated RoboTwin task list.")
    p.add_argument("--env_type", default="libero", choices=["libero", "robotwin"])
    # Sharded collection: N jobs each run `--task_shard k/N --collect_only`, writing to
    # iter_<i>_shard<k>/online_episodes; a final `--skip_collect` job globs them all and
    # finetunes. Mirrors how the RoboTwin eval suite splits tasks across jobs.
    p.add_argument("--task_shard", default=None, metavar="K/N",
                   help="Collect only tasks where index %% N == K (e.g. 0/8).")
    # Merged eval+collection: each task runs n_eval held-out episodes (seeds fixed
    # across iterations, never saved) followed by n_train episodes (advancing seeds,
    # saved for finetuning). One rollout pass yields both the metric and the data.
    p.add_argument("--n_eval_per_task", type=int, default=0,
                   help="Held-out episodes per task; 0 disables the merged eval.")
    p.add_argument("--n_train_per_task", type=int, default=None,
                   help="Training episodes per task (default: n_episodes // n_tasks).")
    p.add_argument("--eval_seed_base", type=int, default=42,
                   help="Seed base for held-out episodes; keep fixed across iterations.")
    p.add_argument("--collect_only", action="store_true",
                   help="Collect episodes and exit before finetuning.")
    p.add_argument("--skip_collect", action="store_true",
                   help="Skip collection; finetune on already-collected iter_*/online_episodes.")
    p.add_argument("--robotwin_root", default=None,
                   help="Path to the RoboTwin repo clone (required when --env_type robotwin).")
    p.add_argument("--n_iterations", type=int, default=5)
    p.add_argument("--n_episodes", type=int, default=20)
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
    p.add_argument("--planner_type", default="bc_diffusion_mppi",
                   help="Action-space planner, or 'dsrl' for latent-noise steering "
                        "(Wagenmaker et al. 2025) as a baseline.")
    # ── DSRL (only used when --planner_type dsrl) ─────────────────────────────
    # Defaults follow Table 11 of the DSRL paper (its pi0-on-LIBERO column).
    p.add_argument("--dsrl_b_w", type=float, default=1.0,
                   help="Latent action magnitude: w is bounded to [-b_w, b_w]. Lower = more "
                        "conservative (nearer the centre of the sampler's noise prior).")
    p.add_argument("--dsrl_full_latent", action="store_true",
                   help="Learn over the full (h*A)-dim latent instead of a per-timestep "
                        "latent tiled across the chunk (the paper's large-chunk recipe).")
    p.add_argument("--dsrl_hidden", type=int, default=128)
    p.add_argument("--dsrl_n_critics", type=int, default=10)
    p.add_argument("--dsrl_explore_std", type=float, default=0.2,
                   help="Gaussian exploration noise on the latent action during collection.")
    p.add_argument("--dsrl_steps", type=int, default=5000,
                   help="Latent actor/critic gradient steps per iteration (MLPs on cached "
                        "features — seconds, not minutes).")
    p.add_argument("--dsrl_batch_size", type=int, default=256)
    p.add_argument("--dsrl_max_states", type=int, default=20000,
                   help="Cap on states used to rebuild latent targets; bounds the DINOv2 cost.")
    p.add_argument("--dsrl_target_batch", type=int, default=8,
                   help="States per forward pass when rebuilding latent targets.")
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
