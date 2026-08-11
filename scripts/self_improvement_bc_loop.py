#!/usr/bin/env python
"""Success-filtered self-training of the FastWAM behavior policy.

Each iteration collects LIBERO or RoboTwin rollouts directly from FastWAM,
without a Q function, planner, candidate ranking, or reward-conditioned action
selection.
Only environment-verified successful episodes enter the online training buffer.
The FastWAM video expert, VAE, and text encoder are frozen, while the action
expert and proprio encoder are updated with FastWAM's flow-matching BC loss.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from lerobot.policies.fastwam.online_bc_dataset import (
    ONLINE_BC_DATASET_FPS,
    ROBOTWIN_ONLINE_BC_DATASET_FPS,
    SuccessfulEpisodeWriter,
    libero_gripper_to_fastwam,
    load_growing_online_success_dataset,
    load_original_success_dataset,
)
from lerobot.utils.constants import ACTION, OBS_STATE


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", "/storage/project/r-agarg35-0/shared/huggingface_cache")


def configure_behavior_finetune(policy, *, gradient_checkpointing: bool = True) -> list[torch.nn.Parameter]:
    """Train FastWAM's behavior path while freezing its video-generative path."""
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

    action_expert = policy.model.action_expert
    for parameter in action_expert.parameters():
        parameter.requires_grad_(True)

    # The standard FastWAM finetuning recipe trains the proprio projection with
    # the action expert.  It is part of the behavior policy, not the video DiT.
    if policy.model.proprio_encoder is not None:
        for parameter in policy.model.proprio_encoder.parameters():
            parameter.requires_grad_(True)

    policy.config.freeze_video_dit = True
    policy.config.loss_lambda_video = 0.0
    policy.config.action_dit_use_gradient_checkpointing = bool(gradient_checkpointing)
    policy.config.mot_checkpoint_mixed_attn = bool(gradient_checkpointing)
    policy.model.loss_lambda_video = 0.0
    policy.model.action_expert.use_gradient_checkpointing = bool(gradient_checkpointing)
    policy.model.mot.mot_checkpoint_mixed_attn = bool(gradient_checkpointing)

    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("FastWAM behavior finetuning produced no trainable parameters.")
    return trainable


def _task_description(env) -> str:
    if hasattr(env, "envs") and len(env.envs) > 0:
        task = getattr(env.envs[0], "task_description", None) or getattr(env.envs[0], "task", None)
        if isinstance(task, str):
            return task
    return ""


def task_descriptions_from_envs(envs) -> tuple[str, ...]:
    descriptions = []
    for group in envs.values():
        for env in group.values():
            task = _task_description(env)
            if task:
                descriptions.append(task)
    return tuple(descriptions)


def executed_action_to_training(
    env_type: str,
    policy_action: torch.Tensor,
    env_action: torch.Tensor,
) -> torch.Tensor:
    """Return executed actions in the checkpoint's behavior-training convention."""
    executed_env_action = env_action.detach().cpu()
    if env_type == "robotwin":
        # RoboTwin FastWAM is trained and executed directly in 14-D qpos space.
        return executed_env_action.clone()
    if env_type == "libero":
        # Preserve the unnormalized arm command and reconstruct the binarized
        # gripper decision that LIBERO actually executed.
        executed_bc_action = policy_action.detach().cpu().clone()
        executed_bc_action[..., -1] = libero_gripper_to_fastwam(executed_env_action)[..., -1]
        return executed_bc_action
    raise ValueError(f"Unsupported self-improvement environment: {env_type!r}")


def _run_episode(
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    camera_keys: tuple[str, ...],
    seed: int,
    env_type: str = "libero",
) -> dict[str, Any]:
    """Collect one rollout in FastWAM's training convention.

    Images and proprioception are captured after environment preprocessing but
    before policy normalization. Actions are captured after policy
    unnormalization. LIBERO's executed gripper decision is converted back to
    FastWAM's 0/1 training convention; RoboTwin's executed 14-D qpos target is
    stored unchanged.
    """
    import numpy as np
    from lerobot.envs.utils import add_envs_task, preprocess_observation

    policy.reset()
    observation, _info = env.reset(seed=[seed])

    observations: list[dict[str, torch.Tensor]] = []
    actions: list[torch.Tensor] = []
    successes: list[bool] = []
    done = np.array([False])
    max_steps = env.call("_max_episode_steps")[0]

    while not np.all(done) and len(actions) < max_steps:
        batch = preprocess_observation(observation)
        batch = add_envs_task(env, batch)
        batch = env_preprocessor(batch)

        required = (*camera_keys, OBS_STATE)
        missing = [key for key in required if key not in batch or not isinstance(batch[key], torch.Tensor)]
        if missing:
            raise KeyError(f"Processed {env_type} observation is missing FastWAM BC keys: {missing}")
        observations.append({key: batch[key][0].detach().cpu().clone() for key in required})

        policy_batch = preprocessor(batch)
        with torch.no_grad():
            normalized_action = policy.select_action(policy_batch)
        bc_action = postprocessor(normalized_action)
        env_action = env_postprocessor({ACTION: bc_action})[ACTION]

        # Store the action that was actually executed. RoboTwin's env
        # postprocessor is identity. LIBERO binarizes/remaps its gripper, so
        # map only that channel back to FastWAM's close=0/open=1 convention.
        executed_bc_action = executed_action_to_training(env_type, bc_action, env_action)
        actions.append(executed_bc_action)

        observation, _reward, terminated, truncated, step_info = env.step(env_action.cpu().numpy())
        if "final_info" in step_info:
            step_success = bool(step_info["final_info"]["is_success"].tolist()[0])
        else:
            step_success = False
        successes.append(step_success)
        done = terminated | truncated | done

    n_frames = len(actions)
    episode: dict[str, Any] = {
        ACTION: torch.cat(actions, dim=0),
        OBS_STATE: torch.stack([obs[OBS_STATE] for obs in observations], dim=0),
        "task": _task_description(env),
        "success": any(successes),
    }
    for key in camera_keys:
        episode[key] = torch.stack([obs[key] for obs in observations], dim=0)
    if episode[ACTION].shape[0] != n_frames:
        raise RuntimeError("Collected action sequence has an unexpected batch dimension.")
    return episode


def collect_successful_episodes(
    policy,
    envs,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    writer: SuccessfulEpisodeWriter,
    camera_keys: tuple[str, ...],
    start_seed: int,
    env_type: str = "libero",
    env_factory=None,
    shard: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Collect exactly ``n_episodes`` and stream only successes to ``writer``."""
    if env_factory is not None:
        task_envs = [(task, 0, None) for task in env_factory.tasks]
    else:
        task_envs = [
            (group_name, task_id, env)
            for group_name, group in envs.items()
            for task_id, env in group.items()
        ]
    if not task_envs:
        raise ValueError("No task environments were created.")

    base, remainder = divmod(n_episodes, len(task_envs))
    counts = [base + (idx < remainder) for idx in range(len(task_envs))]
    seed_offsets = []
    offset = 0
    for count in counts:
        seed_offsets.append(offset)
        offset += count
    if shard is not None:
        shard_index, n_shards = shard
        if not 0 <= shard_index < n_shards:
            raise ValueError(f"Invalid collection shard {shard_index}/{n_shards}")
        selected = [idx for idx in range(len(task_envs)) if idx % n_shards == shard_index]
    else:
        selected = list(range(len(task_envs)))

    per_task: dict[str, dict[str, int]] = {}
    total = 0
    total_success = 0
    failed_tasks: dict[str, str] = {}

    for task_index in selected:
        group_name, task_id, env = task_envs[task_index]
        count = counts[task_index]
        task_success = 0
        total_before_task = total
        built_here = False
        try:
            if env is None:
                env = env_factory(group_name)
                built_here = True
            for episode_index in range(count):
                episode = _run_episode(
                    env,
                    policy,
                    env_preprocessor,
                    env_postprocessor,
                    preprocessor,
                    postprocessor,
                    camera_keys,
                    seed=start_seed + seed_offsets[task_index] + episode_index,
                    env_type=env_type,
                )
                total += 1
                if episode["success"]:
                    writer.add_episode(episode)
                    task_success += 1
                    total_success += 1
        except Exception as error:
            failed_tasks[str(group_name)] = f"{type(error).__name__}: {error}"
            log.exception("Collection failed for task %s", group_name)
        finally:
            if built_here and env is not None:
                try:
                    env.close()
                except Exception:
                    log.exception("Failed to close environment for task %s", group_name)
        task_name = f"{group_name}/{task_id}"
        task_collected = total - total_before_task
        per_task[task_name] = {"requested": count, "collected": task_collected, "successful": task_success}
        log.info("  %s: %d/%d successful", task_name, task_success, task_collected)

    writer.finalize()
    success_rate = 100.0 * total_success / max(total, 1)
    log.info("Collection: %d/%d successful (%.1f%%)", total_success, total, success_rate)
    return {
        "n_collected": total,
        "n_success": total_success,
        "pc_success": success_rate,
        "per_task": per_task,
        "failed_tasks": failed_tasks,
        "complete": not failed_tasks and total == sum(counts[idx] for idx in selected),
    }


def _make_weighted_loader(
    original_dataset: Dataset | None,
    online_dataset: Dataset,
    *,
    online_fraction: float,
    steps: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    if not 0.0 < online_fraction <= 1.0:
        raise ValueError(f"online_fraction must be in (0, 1], got {online_fraction}")
    if len(online_dataset) == 0:
        raise ValueError("Online successful dataset is empty.")

    if original_dataset is None or online_fraction == 1.0:
        dataset: Dataset = online_dataset
        weights = [1.0 / len(online_dataset)] * len(online_dataset)
    else:
        if len(original_dataset) == 0:
            raise ValueError("Original replay dataset is empty.")
        dataset = ConcatDataset([original_dataset, online_dataset])
        original_weight = (1.0 - online_fraction) / len(original_dataset)
        online_weight = online_fraction / len(online_dataset)
        weights = [original_weight] * len(original_dataset) + [online_weight] * len(online_dataset)

    sampler = WeightedRandomSampler(weights, num_samples=steps * batch_size, replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def finetune_bc_policy(
    policy,
    preprocessor,
    original_dataset: Dataset | None,
    online_dataset: Dataset,
    *,
    steps: int,
    lr: float,
    batch_size: int,
    online_fraction: float,
    grad_clip_norm: float,
    num_workers: int,
    global_step: int,
    use_wandb: bool,
) -> tuple[float, int]:
    """Train FastWAM's behavior path on successful demonstration/online data."""
    trainable = configure_behavior_finetune(policy)
    n_trainable = sum(parameter.numel() for parameter in trainable)
    n_total = sum(parameter.numel() for parameter in policy.parameters())
    log.info(
        "BC finetune: %d original + %d online frames, %.2fB/%.2fB trainable params, %d steps",
        len(original_dataset) if original_dataset is not None else 0,
        len(online_dataset),
        n_trainable / 1e9,
        n_total / 1e9,
        steps,
    )

    loader = _make_weighted_loader(
        original_dataset,
        online_dataset,
        online_fraction=online_fraction,
        steps=steps,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=next(policy.parameters()).device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=lr,
        betas=tuple(policy.config.optimizer_betas),
        eps=float(policy.config.optimizer_eps),
        weight_decay=float(policy.config.optimizer_weight_decay),
    )

    policy.train()
    losses: list[float] = []
    for local_step, batch in enumerate(loader, start=1):
        batch = preprocessor(batch)
        optimizer.zero_grad(set_to_none=True)
        loss, loss_dict = policy.forward(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
        optimizer.step()

        action_loss = float(loss_dict["loss_action"])
        losses.append(action_loss)
        global_step += 1
        if local_step == 1 or local_step % 10 == 0 or local_step == steps:
            recent = sum(losses[-10:]) / min(len(losses), 10)
            log.info(
                "  step %d/%d action_loss=%.5f grad_norm=%.3f",
                local_step,
                steps,
                recent,
                float(grad_norm),
            )
            if use_wandb:
                import wandb

                wandb.log(
                    {
                        "finetune/action_loss": recent,
                        "finetune/grad_norm": float(grad_norm),
                        "finetune/lr": lr,
                    },
                    step=global_step,
                )

    policy.eval()
    mean_loss = sum(losses) / len(losses) if losses else float("nan")
    return mean_loss, global_step


def save_fastwam_checkpoint(policy, preprocessor, postprocessor, checkpoint_dir: Path) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    policy.save_pretrained(checkpoint_dir)
    preprocessor.save_pretrained(checkpoint_dir)
    postprocessor.save_pretrained(checkpoint_dir)
    log.info("Saved updated FastWAM checkpoint to %s", checkpoint_dir)
    return checkpoint_dir


def _setup_policy_and_env(args, device: torch.device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv, RoboTwinEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.random_utils import set_seed

    set_seed(args.seed)
    env_factory = None
    if args.env == "libero":
        env_cfg = LiberoEnv(
            task=args.task,
            episode_length=args.episode_length or 520,
            observation_height=224,
            observation_width=224,
        )
        envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    elif args.env == "robotwin":
        if not args.robotwin_root:
            raise ValueError("--robotwin_root is required when --env=robotwin")
        tasks = [task.strip() for task in args.task.split(",") if task.strip()]
        if not tasks:
            raise ValueError("--task must list at least one RoboTwin task")
        env_cfg = RoboTwinEnv(
            task=tasks[0],
            episode_length=args.episode_length,
            robotwin_root=args.robotwin_root,
            task_config=args.robotwin_task_config,
            instruction_type=args.robotwin_instruction_type,
        )

        class _RoboTwinEnvFactory:
            def __init__(self, task_list: list[str]):
                self.tasks = task_list

            def __call__(self, task: str):
                task_cfg = RoboTwinEnv(
                    task=task,
                    episode_length=args.episode_length,
                    robotwin_root=args.robotwin_root,
                    task_config=args.robotwin_task_config,
                    instruction_type=args.robotwin_instruction_type,
                )
                task_envs = make_env(task_cfg, n_envs=1, use_async_envs=False)
                return next(iter(next(iter(task_envs.values())).values()))

        env_factory = _RoboTwinEnvFactory(tasks)
        # RoboTwin environments are expensive SAPIEN instances. They are built
        # lazily and closed task-by-task during collection.
        envs = None
    else:
        raise ValueError(f"Unsupported self-improvement environment: {args.env!r}")

    policy_cfg = PreTrainedConfig.from_pretrained(args.fastwam_ckpt)
    if policy_cfg.type != "fastwam":
        raise ValueError(f"Expected a FastWAM checkpoint, got policy type {policy_cfg.type!r}")
    policy_cfg.pretrained_path = args.fastwam_ckpt
    policy_cfg.device = str(device)
    policy_cfg.num_inference_steps = args.num_inference_steps
    policy_cfg.freeze_video_dit = True
    policy_cfg.loss_lambda_video = 0.0
    policy_cfg.action_dit_use_gradient_checkpointing = True
    policy_cfg.mot_checkpoint_mixed_attn = True

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.fastwam_ckpt,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    return policy, envs, preprocessor, postprocessor, env_preprocessor, env_postprocessor, env_factory


def _eval_once(
    envs,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    *,
    n_episodes: int,
    start_seed: int,
    videos_dir: Path,
) -> float:
    from lerobot.scripts.lerobot_eval import eval_policy_all

    with torch.no_grad():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=n_episodes,
            max_episodes_rendered=min(n_episodes, 4),
            videos_dir=videos_dir,
            start_seed=start_seed,
        )
    return float(info["overall"]["pc_success"])


def _parse_shard(value: str) -> tuple[int, int] | None:
    if not value:
        return None
    try:
        shard_index, n_shards = (int(part) for part in value.split("/", maxsplit=1))
    except (TypeError, ValueError) as error:
        raise ValueError(f"--task_shard must have form K/N, got {value!r}") from error
    if n_shards <= 0 or not 0 <= shard_index < n_shards:
        raise ValueError(f"Invalid task shard {value!r}")
    return shard_index, n_shards


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def _load_collection_manifests(
    output_dir: Path,
    iteration: int,
    *,
    expected_shards: int,
    expected_rollouts: int,
) -> dict[str, Any]:
    pattern = f"iter_{iteration:03d}_shard*/collection_manifest.json"
    paths = sorted(output_dir.glob(pattern))
    if len(paths) != expected_shards:
        raise RuntimeError(
            f"Expected {expected_shards} collection manifests for iteration {iteration}, "
            f"found {len(paths)}: {[str(path) for path in paths]}"
        )
    manifests = [json.loads(path.read_text()) for path in paths]
    incomplete = [str(path) for path, manifest in zip(paths, manifests, strict=True) if not manifest.get("complete")]
    if incomplete:
        raise RuntimeError(f"Incomplete collection shards: {incomplete}")
    total_collected = sum(int(manifest["n_collected"]) for manifest in manifests)
    if total_collected != expected_rollouts:
        raise RuntimeError(
            f"Iteration {iteration} collected {total_collected} episodes across all shards; "
            f"expected exactly {expected_rollouts}."
        )
    per_task: dict[str, dict[str, int]] = {}
    for manifest in manifests:
        overlap = set(per_task).intersection(manifest["per_task"])
        if overlap:
            raise RuntimeError(f"Tasks were duplicated across collection shards: {sorted(overlap)}")
        per_task.update(manifest["per_task"])
    return {
        "n_collected": total_collected,
        "n_success": sum(int(manifest["n_success"]) for manifest in manifests),
        "pc_success": 100.0
        * sum(int(manifest["n_success"]) for manifest in manifests)
        / max(total_collected, 1),
        "per_task": per_task,
        "failed_tasks": {},
        "complete": True,
        "collection_manifests": [str(path) for path in paths],
    }


def self_improvement_loop(args) -> None:
    if args.env not in {"libero", "robotwin"}:
        raise ValueError(f"env must be 'libero' or 'robotwin', got {args.env!r}")
    if args.online_fraction is None:
        args.online_fraction = 0.5
    if args.n_iterations <= 0:
        raise ValueError("n_iterations must be positive.")
    if args.n_episodes <= 0:
        raise ValueError("n_episodes must be positive.")
    if args.finetune_steps < 0:
        raise ValueError("finetune_steps cannot be negative.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    if args.num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive.")
    if args.episode_length is not None and args.episode_length <= 0:
        raise ValueError("episode_length must be positive.")
    if args.online_dataset_fps is not None and args.online_dataset_fps <= 0:
        raise ValueError("online_dataset_fps must be positive.")
    if not 0.0 < args.online_fraction <= 1.0:
        raise ValueError("online_fraction must be in (0, 1].")
    if args.replay_same_tasks_only is None:
        # RoboTwin demonstration language contains many paraphrases, so exact
        # task-string filtering is unsafe without an explicit task-id map.
        args.replay_same_tasks_only = args.env == "libero"
    if args.collect_only and args.skip_collect:
        raise ValueError("--collect_only and --skip_collect are mutually exclusive")
    shard = _parse_shard(args.task_shard)
    if shard is not None and not args.collect_only:
        raise ValueError("--task_shard is only valid with --collect_only")
    if args.skip_collect and args.expected_collect_shards <= 0:
        raise ValueError("--expected_collect_shards must be positive with --skip_collect")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "loop_summary.jsonl"

    use_wandb = bool(args.wandb_project) and not args.collect_only
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or output_dir.name,
            config=vars(args),
            dir=str(output_dir),
        )

    (
        policy,
        envs,
        preprocessor,
        postprocessor,
        env_preprocessor,
        env_postprocessor,
        env_factory,
    ) = _setup_policy_and_env(args, device)
    camera_keys = tuple(policy.config.image_features)
    task_descriptions = task_descriptions_from_envs(envs) if envs is not None else ()
    validate_gripper_targets = args.env == "libero"
    online_dataset_fps = args.online_dataset_fps
    if online_dataset_fps is None:
        online_dataset_fps = (
            ROBOTWIN_ONLINE_BC_DATASET_FPS if args.env == "robotwin" else ONLINE_BC_DATASET_FPS
        )

    original_dataset = None
    if args.online_fraction < 1.0 and not args.collect_only:
        original_dataset = load_original_success_dataset(
            args.original_dataset_repo_id,
            args.original_dataset_root,
            policy.config,
            task_descriptions=task_descriptions if args.replay_same_tasks_only else None,
            libero_action_convention=args.env == "libero",
            validate_gripper_targets=validate_gripper_targets,
        )
        log.info("Original successful replay: %d frames", len(original_dataset))

    start_seed = args.seed + args.start_iteration * args.n_episodes
    global_step = args.start_iteration * args.finetune_steps
    for offset in range(args.n_iterations):
        iteration = args.start_iteration + offset
        iter_name = f"iter_{iteration:03d}"
        if args.collect_only and shard is not None:
            iter_name += f"_shard{shard[0]:02d}"
        iter_dir = output_dir / iter_name
        iter_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        log.info("%s\nIteration %d\n%s", "=" * 60, iteration, "=" * 60)

        if args.skip_collect:
            collection = _load_collection_manifests(
                output_dir,
                iteration,
                expected_shards=args.expected_collect_shards,
                expected_rollouts=args.n_episodes,
            )
            log.info(
                "Validated %d/%d collected episodes across %d shards",
                collection["n_collected"],
                args.n_episodes,
                args.expected_collect_shards,
            )
        else:
            writer = SuccessfulEpisodeWriter(
                iter_dir / "successful_episodes",
                camera_keys=camera_keys,
                image_size=tuple(policy.config.image_size),
                action_dim=int(policy.config.action_dim),
                state_dim=int(policy.config.state_dim),
                fps=online_dataset_fps,
            )
            collection = collect_successful_episodes(
                policy,
                envs,
                env_preprocessor,
                env_postprocessor,
                preprocessor,
                postprocessor,
                args.n_episodes,
                writer,
                camera_keys,
                start_seed,
                env_type=args.env,
                env_factory=env_factory,
                shard=shard,
            )
        start_seed += args.n_episodes

        if args.collect_only:
            manifest = {
                "algorithm": "success_filtered_behavior_policy_sft",
                "uses_q_function": False,
                "uses_planning": False,
                "iteration": iteration,
                "task_shard": args.task_shard or None,
                "n_rollouts_global": args.n_episodes,
                **collection,
                "successful_dataset": str(iter_dir / "successful_episodes")
                if (iter_dir / "successful_episodes").is_dir()
                else None,
                "elapsed_s": time.time() - started,
            }
            _write_json_atomic(iter_dir / "collection_manifest.json", manifest)
            if not collection["complete"]:
                raise RuntimeError(
                    f"Collection shard {args.task_shard or 'unsharded'} was incomplete; "
                    "the manifest was preserved for diagnosis."
                )
            log.info("Collection-only shard complete: %s", iter_dir)
            continue
        if not collection["complete"]:
            raise RuntimeError(f"Collection was incomplete: {collection['failed_tasks']}")

        online_dataset = load_growing_online_success_dataset(
            output_dir,
            policy.config,
            validate_gripper_targets=validate_gripper_targets,
        )
        mean_loss = None
        checkpoint = None
        if online_dataset is None or args.finetune_steps == 0:
            reason = "no successful online episodes" if online_dataset is None else "finetune_steps=0"
            log.warning("Skipping BC finetune: %s", reason)
        else:
            mean_loss, global_step = finetune_bc_policy(
                policy,
                preprocessor,
                original_dataset,
                online_dataset,
                steps=args.finetune_steps,
                lr=args.finetune_lr,
                batch_size=args.batch_size,
                online_fraction=args.online_fraction,
                grad_clip_norm=args.grad_clip_norm,
                num_workers=args.num_workers,
                global_step=global_step,
                use_wandb=use_wandb,
            )
            checkpoint = save_fastwam_checkpoint(
                policy,
                preprocessor,
                postprocessor,
                iter_dir / "fastwam_checkpoint",
            )

        policy_eval = None
        if args.eval_n_episodes > 0:
            if envs is None:
                raise RuntimeError(
                    "Multi-task RoboTwin evaluation must use the external sharded evaluator; "
                    "set --eval_n_episodes 0 in this process."
                )
            policy_eval = _eval_once(
                envs,
                policy,
                env_preprocessor,
                env_postprocessor,
                preprocessor,
                postprocessor,
                n_episodes=args.eval_n_episodes,
                # Reuse the same held-out initial-state seeds every iteration
                # so checkpoint-to-checkpoint success rates are comparable.
                start_seed=args.seed + 100_000,
                videos_dir=iter_dir / "eval" / "videos",
            )
            log.info("Eval: FastWAM policy=%.1f%%", policy_eval)

        metrics = {
            "algorithm": "success_filtered_behavior_policy_sft",
            "uses_q_function": False,
            "uses_planning": False,
            "collection_protocol": "ungated_sequential_seeds",
            "iteration": iteration,
            "env": args.env,
            "task": args.task,
            "n_rollouts_requested": args.n_episodes,
            "n_rounds_requested": args.experiment_iterations or args.n_iterations,
            "finetune_steps": args.finetune_steps,
            "finetune_lr": args.finetune_lr,
            "online_fraction": args.online_fraction,
            "original_replay_task": args.original_replay_task or None,
            **collection,
            "n_online_success_frames": len(online_dataset) if online_dataset is not None else 0,
            "finetune_action_loss": mean_loss,
            "policy_eval_pc_success": policy_eval,
            "fastwam_checkpoint": str(checkpoint) if checkpoint is not None else None,
            "elapsed_s": time.time() - started,
        }
        _write_json_atomic(iter_dir / "metrics.json", metrics)
        with summary_path.open("a") as summary_file:
            summary_file.write(json.dumps(metrics) + "\n")
        if use_wandb:
            import wandb

            wandb_metrics = {
                "collect/pc_success": collection["pc_success"],
                "collect/n_success": collection["n_success"],
            }
            if mean_loss is not None:
                wandb_metrics["finetune/mean_action_loss"] = mean_loss
            if policy_eval is not None:
                wandb_metrics["eval/policy_pc_success"] = policy_eval
            wandb.log(wandb_metrics, step=global_step)

    if use_wandb:
        import wandb

        wandb.finish()
    log.info("BC self-improvement complete: %s", summary_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Success-only FastWAM self-training without Q-planning")
    parser.add_argument("--env", choices=("libero", "robotwin"), default="libero")
    parser.add_argument("--fastwam_ckpt", required=True)
    parser.add_argument("--original_dataset_repo_id", default="HuggingFaceVLA/libero")
    parser.add_argument(
        "--original_dataset_root",
        default=(
            "/storage/project/r-agarg35-0/shared/lerobot-data-2/"
            "HuggingFaceVLA/libero"
        ),
    )
    parser.add_argument(
        "--original_replay_task",
        default="",
        help="Optional provenance label for the original replay dataset's benchmark task.",
    )
    parser.add_argument("--task", default="libero_10")
    parser.add_argument(
        "--episode_length",
        type=int,
        default=None,
        help="Maximum environment steps. Defaults to 520 for LIBERO and RoboTwin's per-task limit.",
    )
    parser.add_argument(
        "--robotwin_root",
        default=os.environ.get("ROBOTWIN_ROOT", ""),
        help="Path to the RoboTwin checkout; required for --env=robotwin.",
    )
    parser.add_argument(
        "--robotwin_task_config",
        choices=("demo_randomized", "demo_clean"),
        default="demo_randomized",
    )
    parser.add_argument("--robotwin_instruction_type", default="unseen")
    parser.add_argument(
        "--online_dataset_fps",
        type=float,
        default=None,
        help=(
            "Frame-index/video FPS stored in collected LeRobot metadata "
            "(default: LIBERO=10, RoboTwin=50); this is not RoboTwin's physical control rate."
        ),
    )
    parser.add_argument("--n_iterations", type=int, default=5)
    parser.add_argument("--start_iteration", type=int, default=0)
    parser.add_argument(
        "--experiment_iterations",
        type=int,
        default=0,
        help="Total rounds in a chained experiment (metrics provenance only).",
    )
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument(
        "--collect_only",
        action="store_true",
        help="Collect one (optionally sharded) round and stop before loading replay or finetuning.",
    )
    parser.add_argument(
        "--skip_collect",
        action="store_true",
        help="Validate completed collection manifests, then finetune without collecting.",
    )
    parser.add_argument(
        "--task_shard",
        default="",
        help="Collection task shard in K/N form; allocation remains global and totals n_episodes.",
    )
    parser.add_argument(
        "--expected_collect_shards",
        type=int,
        default=1,
        help="Number of complete shard manifests required by --skip_collect.",
    )
    parser.add_argument("--finetune_steps", type=int, default=200)
    parser.add_argument("--finetune_lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--online_fraction",
        type=float,
        default=None,
        help="Online-success sampling fraction (default: 0.5 for both environments).",
    )
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--replay_same_tasks_only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Restrict original replay by exact task text (default: LIBERO on, RoboTwin off).",
    )
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--eval_n_episodes", type=int, default=5)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_run_name", default="")
    return parser.parse_args()


if __name__ == "__main__":
    self_improvement_loop(parse_args())
