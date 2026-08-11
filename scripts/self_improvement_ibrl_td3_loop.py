#!/usr/bin/env python
"""Primitive-action IBRL with a frozen FastWAM behavior policy and TD3.

At each decision, frozen FastWAM and the trainable deterministic actor propose
actions in the same LIBERO action space.  Target twin-Q functions score both,
and the higher conservative value is executed.  The same two-proposal maximum
is used in the Bellman target, which is the defining IBRL mechanism.

The default action horizon is one, matching the paper's action-level proposal
and backup equations.  Larger explicit horizons remain available as an SMDP
macro-action ablation.  The released pixel-IBRL replay protocol is mapped explicitly:
ten expert episodes prefill one unified replay, forty complete frozen-BC
episodes are collected without updates, and only then does hybrid interaction
and online learning begin. Long experiments can be split into deterministic,
restartable warmup, online, reference-evaluation, and hybrid-evaluation phases.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.policies.fastwam.dawr_value import FrozenDINOEncoder
from lerobot.policies.fastwam.ibrl_td3 import (
    EncodedReplayBuffer,
    IBRLTD3,
    freeze_behavior_policy,
)
from lerobot.utils.constants import ACTION, OBS_STATE


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
log = logging.getLogger(__name__)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass(frozen=True)
class ProposalContext:
    observation: torch.Tensor
    bc_action: torch.Tensor


RESUME_STATE_VERSION = 1
PHASE_CHOICES = ("all", "warmup", "online", "reference_eval", "hybrid_eval")


def _empty_collection_progress(phase: str) -> dict[str, Any]:
    return {
        "phase": phase,
        "n_episodes": 0,
        "n_success": 0,
        "return_sum": 0.0,
        "env_steps": 0,
        "decisions": 0,
        "bc_selected": 0,
        "rl_selected": 0,
        "updates": 0,
        "episodes": [],
    }


def _collection_summary(progress: dict[str, Any]) -> dict[str, Any]:
    n_episodes = int(progress["n_episodes"])
    decisions = int(progress["decisions"])
    return {
        "phase": progress["phase"],
        "n_episodes": n_episodes,
        "n_success": int(progress["n_success"]),
        "pc_success": (
            None if n_episodes == 0 else 100.0 * int(progress["n_success"]) / n_episodes
        ),
        "mean_return": (
            None if n_episodes == 0 else float(progress["return_sum"]) / n_episodes
        ),
        "env_steps": int(progress["env_steps"]),
        "decisions": decisions,
        "bc_selected": int(progress["bc_selected"]),
        "rl_selected": int(progress["rl_selected"]),
        "bc_fraction": int(progress["bc_selected"]) / max(decisions, 1),
        "updates": int(progress["updates"]),
    }


def _merge_evaluation_results(
    previous: dict[str, Any] | None,
    shard: dict[str, Any],
) -> dict[str, Any]:
    """Merge deterministic evaluation shards into the original result schema."""
    if previous is None:
        return shard
    if previous["policy"] != shard["policy"]:
        raise ValueError("Cannot merge evaluation shards from different policies.")
    per_task = {
        task: dict(metrics) for task, metrics in previous["per_task"].items()
    }
    for task, metrics in shard["per_task"].items():
        merged = per_task.setdefault(task, {"episodes": 0, "successes": 0})
        merged["episodes"] += int(metrics["episodes"])
        merged["successes"] += int(metrics["successes"])
    n_episodes = int(previous["n_episodes"]) + int(shard["n_episodes"])
    n_success = int(previous["n_success"]) + int(shard["n_success"])
    bc_selected = int(previous["bc_selected"]) + int(shard["bc_selected"])
    rl_selected = int(previous["rl_selected"]) + int(shard["rl_selected"])
    return {
        **previous,
        "n_episodes": n_episodes,
        "n_success": n_success,
        "pc_success": 100.0 * n_success / max(n_episodes, 1),
        "mean_return": (
            float(previous["mean_return"]) * int(previous["n_episodes"])
            + float(shard["mean_return"]) * int(shard["n_episodes"])
        )
        / max(n_episodes, 1),
        "bc_selected": bc_selected,
        "rl_selected": rl_selected,
        "bc_fraction": bc_selected / max(bc_selected + rl_selected, 1),
        "per_task": per_task,
        "episodes": [*previous["episodes"], *shard["episodes"]],
    }


def _capture_rng_state(replay_generator: torch.Generator) -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "replay_generator": replay_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any], replay_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    replay_generator.set_state(state["replay_generator"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("Resume state contains CUDA RNG state but CUDA is unavailable.")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _continuity_config(args, *, observation_dim: int, action_dim: int) -> dict[str, Any]:
    """Select arguments that must remain identical across resumed phases."""
    ignored = {
        "phase",
        "online_episode_end",
        "resume_state",
        "checkpoint_every_episodes",
        "wandb_project",
        "wandb_entity",
        "wandb_run_name",
    }
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in ignored
    }
    config.update(observation_dim=observation_dim, action_dim=action_dim)
    return config


def _validate_continuity_config(saved: dict[str, Any], current: dict[str, Any]) -> None:
    if saved == current:
        return
    mismatches = {
        key: (saved.get(key), current.get(key))
        for key in sorted(set(saved) | set(current))
        if saved.get(key) != current.get(key)
    }
    raise ValueError(f"Resume configuration does not match this phase invocation: {mismatches}")


def _task_description(env) -> str:
    if hasattr(env, "envs") and len(env.envs) > 0:
        task = getattr(env.envs[0], "task_description", None) or getattr(env.envs[0], "task", None)
        if isinstance(task, str):
            return task
    return ""


def _episode_schedule(task_envs: list[tuple[str, int, Any]], n_episodes: int):
    """Interleave task environments while assigning exactly ``n_episodes``."""
    if n_episodes <= 0:
        return []
    base, remainder = divmod(n_episodes, len(task_envs))
    per_task_counts = [base + int(index < remainder) for index in range(len(task_envs))]
    return [
        task_env
        for round_index in range(max(per_task_counts))
        for task_env, count in zip(task_envs, per_task_counts, strict=True)
        if round_index < count
    ]


def _evaluation_trials(
    task_envs: list[tuple[str, int, Any]],
    *,
    total_episodes: int,
    episode_start: int,
    n_episodes: int,
) -> list[tuple[int, str, int, Any, int]]:
    """Return globally indexed trials whose task-local indices ignore sharding."""
    if episode_start < 0 or n_episodes < 0:
        raise ValueError("Evaluation shard indices cannot be negative.")
    if episode_start + n_episodes > total_episodes:
        raise ValueError("Evaluation shard extends beyond total_episodes.")
    full_schedule = _episode_schedule(task_envs, total_episodes)
    per_task_seen: dict[tuple[str, int], int] = {}
    for group_name, task_id, _ in full_schedule[:episode_start]:
        task_key = (group_name, task_id)
        per_task_seen[task_key] = per_task_seen.get(task_key, 0) + 1
    trials = []
    for local_index, (group_name, task_id, env) in enumerate(
        full_schedule[episode_start : episode_start + n_episodes]
    ):
        task_key = (group_name, task_id)
        task_episode_index = per_task_seen.get(task_key, 0)
        per_task_seen[task_key] = task_episode_index + 1
        trials.append(
            (
                episode_start + local_index,
                group_name,
                task_id,
                env,
                task_episode_index,
            )
        )
    return trials


def _scalar_bool(value: Any) -> bool:
    array = np.asarray(value)
    return bool(array.reshape(-1)[0]) if array.size else False


def _step_success(info: dict[str, Any]) -> bool:
    final_info = info.get("final_info")
    if not isinstance(final_info, dict) and final_info is not None:
        flattened = np.asarray(final_info, dtype=object).reshape(-1)
        final_info = flattened[0] if len(flattened) else None
    if isinstance(final_info, dict) and "is_success" in final_info:
        return _scalar_bool(final_info["is_success"])
    if "is_success" in info:
        return _scalar_bool(info["is_success"])
    return False


def _consume_update_credit(
    carried_env_steps: int,
    executed_env_steps: int,
    update_every_env_steps: int,
) -> tuple[int, int]:
    """Map primitive environment steps to learner updates without losing remainders."""
    if carried_env_steps < 0 or executed_env_steps < 0:
        raise ValueError("Environment-step update credit cannot be negative.")
    if update_every_env_steps <= 0:
        raise ValueError("update_every_env_steps must be positive.")
    return divmod(
        carried_env_steps + executed_env_steps,
        update_every_env_steps,
    )


def _canonicalize_executed_action(
    action: torch.Tensor,
    *,
    executed_steps: int,
    action_horizon: int,
    action_dim: int,
) -> torch.Tensor:
    """Repeat the last executed action over a terminal macro's unused suffix."""
    if not 1 <= executed_steps <= action_horizon:
        raise ValueError("executed_steps must lie in [1, action_horizon].")
    expected = action_horizon * action_dim
    if action.numel() != expected:
        raise ValueError(f"Expected {expected} action values, got {action.numel()}.")
    chunk = action.reshape(-1, action_horizon, action_dim).clone()
    if executed_steps < action_horizon:
        chunk[:, executed_steps:] = chunk[:, executed_steps - 1 : executed_steps]
    return chunk.reshape(action.shape)


def _setup(args, device: torch.device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.random_utils import set_seed

    set_seed(args.seed)
    env_cfg = LiberoEnv(
        task=args.task,
        # A paper-style run trains one agent per task. Construct only that task's
        # environment instead of creating every LIBERO task and filtering later.
        # Besides wasting memory, keeping ten Robosuite EGL contexts alive in one
        # process can invalidate the active context and abort in mjr_readPixels.
        task_ids=(
            None
            if args.collection_task_id is None
            else [int(args.collection_task_id)]
        ),
        episode_length=args.episode_length,
        observation_height=224,
        observation_width=224,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    effective_lengths = {
        int(length)
        for group in envs.values()
        for env in group.values()
        for length in env.call("_max_episode_steps")
    }
    if effective_lengths != {args.episode_length}:
        raise RuntimeError(
            "LIBERO did not apply the requested episode length: "
            f"requested={args.episode_length}, effective={sorted(effective_lengths)}"
        )

    policy_cfg = PreTrainedConfig.from_pretrained(args.fastwam_ckpt)
    if policy_cfg.type != "fastwam":
        raise ValueError(f"Expected a FastWAM checkpoint, got {policy_cfg.type!r}.")
    policy_cfg.pretrained_path = args.fastwam_ckpt
    policy_cfg.device = str(device)
    policy_cfg.freeze_video_dit = True
    policy_cfg.loss_lambda_video = 0.0
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    freeze_behavior_policy(policy)
    policy.attach_planner(None)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.fastwam_ckpt,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=policy_cfg,
    )
    return policy, envs, preprocessor, postprocessor, env_preprocessor, env_postprocessor


@torch.no_grad()
def _make_proposal_context(
    observation,
    env,
    policy,
    vision_encoder: FrozenDINOEncoder,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    *,
    camera_keys: tuple[str, ...],
    task_to_index: dict[str, int],
    action_horizon: int,
    device: torch.device,
) -> ProposalContext:
    from lerobot.envs.utils import add_envs_task, preprocess_observation

    batch = preprocess_observation(observation)
    batch = add_envs_task(env, batch)
    batch = env_preprocessor(batch)
    missing = [
        key
        for key in (*camera_keys, OBS_STATE)
        if key not in batch or not isinstance(batch[key], torch.Tensor)
    ]
    if missing:
        raise KeyError(f"Processed LIBERO observation is missing IBRL keys: {missing}")

    images = torch.stack([batch[key] for key in camera_keys], dim=1)
    vision_features = vision_encoder(images).flatten(1)
    policy_batch = preprocessor(batch)
    normalized_state = policy_batch[OBS_STATE].to(device=device, dtype=torch.float32)
    task = _task_description(env)
    if task not in task_to_index:
        raise KeyError(f"Unknown LIBERO task description: {task!r}")
    task_one_hot = F.one_hot(
        torch.tensor([task_to_index[task]], device=device),
        num_classes=len(task_to_index),
    ).float()
    encoded_observation = torch.cat((vision_features, normalized_state, task_one_hot), dim=-1)

    normalized_chunk = policy.predict_action_chunk(policy_batch)
    if normalized_chunk.ndim != 3 or normalized_chunk.shape[0] != 1:
        raise ValueError(
            "FastWAM must return one action chunk shaped (1, horizon, action_dim); "
            f"got {tuple(normalized_chunk.shape)}."
        )
    if normalized_chunk.shape[1] < action_horizon:
        raise ValueError(
            f"FastWAM returned horizon {normalized_chunk.shape[1]}, shorter than {action_horizon}."
        )
    raw_chunk = postprocessor(normalized_chunk[:, :action_horizon]).float()
    libero_chunk = env_postprocessor({ACTION: raw_chunk})[ACTION].float().clamp(-1.0, 1.0)
    return ProposalContext(
        observation=encoded_observation.detach(),
        bc_action=libero_chunk.reshape(1, -1).to(device=device),
    )


def _execute_chunk(
    env,
    action_chunk: torch.Tensor,
    *,
    env_step: int,
    max_env_steps: int,
    gamma: float,
) -> tuple[Any, float, float, float, int, bool, bool]:
    """Execute a macro-action and return its discounted SMDP transition data."""
    observation = None
    discounted_reward = 0.0
    raw_reward = 0.0
    discount_power = 1.0
    executed = 0
    done = False
    success = False

    for action in action_chunk:
        if env_step + executed >= max_env_steps:
            done = True
            break
        observation, reward, terminated, truncated, info = env.step(action.unsqueeze(0).numpy())
        step_success = _step_success(info)
        step_reward = max(float(np.asarray(reward).reshape(-1)[0]), float(step_success))
        discounted_reward += discount_power * step_reward
        raw_reward += step_reward
        discount_power *= gamma
        executed += 1
        success = success or step_success
        done = bool(np.all(np.asarray(terminated) | np.asarray(truncated)))
        done = done or env_step + executed >= max_env_steps
        if done:
            break

    if observation is None or executed == 0:
        raise RuntimeError("IBRL selected an action chunk but executed no environment action.")
    return observation, discounted_reward, raw_reward, discount_power, executed, done, success


def _save_checkpoint(
    agent: IBRLTD3,
    path: Path,
    *,
    args,
    task_vocab: tuple[str, ...],
    camera_keys: tuple[str, ...],
    action_horizon: int,
    global_decisions: int,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = agent.checkpoint()
    checkpoint.update(
        {
            "algorithm": "chunked_ibrl_td3",
            "task_vocab": task_vocab,
            "camera_keys": camera_keys,
            "vision_encoder_model": args.vision_encoder_model,
            "action_horizon": action_horizon,
            "global_decisions": global_decisions,
            "fastwam_ckpt": args.fastwam_ckpt,
            "run_config": vars(args),
            "contains_replay": False,
            "resume_supported": False,
        }
    )
    torch.save(checkpoint, path)
    return path


def _validate_args(args) -> None:
    if args.n_episodes <= 0 or args.episode_length <= 0:
        raise ValueError("n_episodes and episode_length must be positive.")
    if args.buffer_size <= 0 or args.batch_size <= 0:
        raise ValueError("buffer_size and batch_size must be positive.")
    if args.buffer_size < args.batch_size:
        raise ValueError("buffer_size cannot be smaller than batch_size.")
    if args.n_demo_episodes < 0 or args.bc_warmup_episodes < 0:
        raise ValueError("Demo and BC warmup episode counts cannot be negative.")
    if args.update_every_env_steps <= 0 or args.learner_warmstart_updates < 0:
        raise ValueError("Update cadence must be positive and warm-start updates non-negative.")
    if args.n_demo_episodes > 0 and not args.demo_dataset_root:
        raise ValueError("demo_dataset_root is required when n_demo_episodes is positive.")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1].")
    if args.action_horizon <= 0:
        raise ValueError("action_horizon must be positive.")
    if args.eval_n_episodes < 0:
        raise ValueError("eval_n_episodes cannot be negative.")
    if not 0.0 <= args.actor_dropout < 1.0:
        raise ValueError("actor_dropout must lie in [0, 1).")
    if args.online_episode_end is not None and not 0 <= args.online_episode_end <= args.n_episodes:
        raise ValueError("online_episode_end must lie in [0, n_episodes].")


@torch.no_grad()
def _evaluate_hybrid(
    agent: IBRLTD3,
    task_envs: list[tuple[str, int, Any]],
    policy,
    vision_encoder: FrozenDINOEncoder,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    *,
    camera_keys: tuple[str, ...],
    task_to_index: dict[str, int],
    action_horizon: int,
    n_episodes: int,
    total_episodes: int | None = None,
    episode_start: int = 0,
    episode_length: int,
    gamma: float,
    start_seed: int,
    init_state_offsets: dict[tuple[str, int], int],
    device: torch.device,
    reference_only: bool = False,
) -> dict[str, Any] | None:
    """Evaluate one deterministic shard of the hybrid or frozen reference."""
    if n_episodes == 0:
        return None
    if total_episodes is None:
        total_episodes = episode_start + n_episodes
    trials = _evaluation_trials(
        task_envs,
        total_episodes=total_episodes,
        episode_start=episode_start,
        n_episodes=n_episodes,
    )
    per_task: dict[str, dict[str, int]] = {}
    successes = 0
    bc_selected = 0
    rl_selected = 0
    returns: list[float] = []
    episode_records: list[dict[str, Any]] = []
    was_training = agent.training
    agent.eval()

    for episode_index, group_name, task_id, env, task_seen in trials:
        episode_seed = start_seed + episode_index
        # Per-episode seeding makes evaluation invariant to Slurm sharding and
        # gives the frozen-reference and hybrid trials identical proposal RNG.
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        torch.cuda.manual_seed_all(episode_seed)
        task_key = (group_name, task_id)
        init_state_index = init_state_offsets.get(task_key, 0) + task_seen
        policy.reset()
        policy.attach_planner(None)
        observation, _ = env.reset(
            seed=[episode_seed],
            options={"episode_index_offset": init_state_index},
        )
        max_env_steps = min(int(env.call("_max_episode_steps")[0]), episode_length)
        context = _make_proposal_context(
            observation,
            env,
            policy,
            vision_encoder,
            env_preprocessor,
            env_postprocessor,
            preprocessor,
            postprocessor,
            camera_keys=camera_keys,
            task_to_index=task_to_index,
            action_horizon=action_horizon,
            device=device,
        )
        env_step = 0
        episode_return = 0.0
        episode_success = False
        episode_bc_selected = 0
        episode_rl_selected = 0
        while env_step < max_env_steps:
            if reference_only:
                selected_action = context.bc_action
                bc_selected += 1
                episode_bc_selected += 1
            else:
                selection = agent.select_hybrid_action(context.observation, context.bc_action)
                selected_action = selection.action
                if bool(selection.bc_selected.item()):
                    bc_selected += 1
                    episode_bc_selected += 1
                else:
                    rl_selected += 1
                    episode_rl_selected += 1
            action_chunk = selected_action.reshape(
                action_horizon, int(policy.config.action_dim)
            ).detach().cpu()
            (
                next_raw_observation,
                _,
                raw_reward,
                _,
                executed,
                done,
                success,
            ) = _execute_chunk(
                env,
                action_chunk,
                env_step=env_step,
                max_env_steps=max_env_steps,
                gamma=gamma,
            )
            env_step += executed
            episode_return += raw_reward
            episode_success = episode_success or success
            if done:
                break
            context = _make_proposal_context(
                next_raw_observation,
                env,
                policy,
                vision_encoder,
                env_preprocessor,
                env_postprocessor,
                preprocessor,
                postprocessor,
                camera_keys=camera_keys,
                task_to_index=task_to_index,
                action_horizon=action_horizon,
                device=device,
            )

        successes += int(episode_success)
        returns.append(episode_return)
        task_name = f"{group_name}/{task_id}"
        task_metrics = per_task.setdefault(task_name, {"episodes": 0, "successes": 0})
        task_metrics["episodes"] += 1
        task_metrics["successes"] += int(episode_success)
        episode_records.append(
            {
                "episode": episode_index,
                "task": task_name,
                "seed": episode_seed,
                "init_state_index": init_state_index,
                "success": episode_success,
                "return": episode_return,
                "env_steps": env_step,
                "bc_selected": episode_bc_selected,
                "rl_selected": episode_rl_selected,
            }
        )

    agent.train(was_training)
    return {
        "n_episodes": len(trials),
        "n_success": successes,
        "pc_success": 100.0 * successes / max(len(trials), 1),
        "mean_return": sum(returns) / max(len(returns), 1),
        "bc_selected": bc_selected,
        "rl_selected": rl_selected,
        "bc_fraction": bc_selected / max(bc_selected + rl_selected, 1),
        "per_task": per_task,
        "episodes": episode_records,
        "policy": "frozen_fastwam" if reference_only else "ibrl_hybrid",
        "exploration_noise": 0.0,
        "parameter_updates": 0,
    }


def _paired_evaluation_summary(
    reference: dict[str, Any] | None,
    hybrid: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Summarize paired outcomes on identical eval seeds and initial states."""
    if reference is None or hybrid is None:
        return None
    reference_episodes = reference["episodes"]
    hybrid_episodes = hybrid["episodes"]
    if len(reference_episodes) != len(hybrid_episodes):
        raise ValueError("Reference and hybrid evaluations have different episode counts.")
    wins = ties = losses = 0
    for reference_episode, hybrid_episode in zip(
        reference_episodes, hybrid_episodes, strict=True
    ):
        pairing_keys = ("task", "seed", "init_state_index")
        if any(reference_episode[key] != hybrid_episode[key] for key in pairing_keys):
            raise ValueError("Reference and hybrid evaluations are not episode-paired.")
        delta = int(hybrid_episode["success"]) - int(reference_episode["success"])
        wins += int(delta > 0)
        ties += int(delta == 0)
        losses += int(delta < 0)
    return {
        "n_paired_episodes": len(reference_episodes),
        "hybrid_wins": wins,
        "ties": ties,
        "hybrid_losses": losses,
        "hybrid_minus_reference_success_pp": (
            float(hybrid["pc_success"]) - float(reference["pc_success"])
        ),
    }


def train_ibrl(args) -> dict[str, Any]:
    """Run one restartable IBRL phase, or the complete sequence with ``all``."""
    from lerobot.envs.utils import close_envs

    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("FastWAM IBRL requires a CUDA GPU.")

    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = Path(args.resume_state) if args.resume_state else output_dir / "resume_state.pt"
    episode_log_path = output_dir / "episodes.jsonl"
    invocation_started = time.time()
    invocation_config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    config_path = output_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(invocation_config, indent=2) + "\n")

    use_wandb = bool(args.wandb_project)
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=(args.wandb_run_name or output_dir.name) + f"-{args.phase}",
            config=invocation_config,
            dir=str(output_dir),
        )

    envs = None
    try:
        policy, envs, preprocessor, postprocessor, env_preprocessor, env_postprocessor = _setup(
            args, device
        )
        camera_keys = tuple(policy.config.image_features)
        task_envs = [
            (group_name, int(task_id), env)
            for group_name, group in envs.items()
            for task_id, env in group.items()
        ]
        if args.collection_task_id is not None:
            task_envs = [entry for entry in task_envs if entry[1] == args.collection_task_id]
        if not task_envs:
            raise ValueError("No LIBERO environment matches the requested collection task.")
        task_vocab = tuple(sorted({_task_description(env) for _, _, env in task_envs}))
        if not task_vocab:
            raise ValueError("LIBERO did not expose task descriptions for IBRL conditioning.")
        task_to_index = {task: index for index, task in enumerate(task_vocab)}
        action_horizon = int(args.action_horizon)
        if action_horizon > int(policy.config.chunk_size):
            raise ValueError(
                f"action_horizon={action_horizon} exceeds FastWAM "
                f"chunk_size={policy.config.chunk_size}."
            )

        vision_encoder = FrozenDINOEncoder(
            args.vision_encoder_model,
            device=device,
            dtype=torch.bfloat16,
        )
        observation_dim = (
            len(camera_keys) * int(vision_encoder.output_dim)
            + int(policy.config.state_dim)
            + len(task_vocab)
        )
        action_dim = action_horizon * int(policy.config.action_dim)
        agent = IBRLTD3(
            observation_dim,
            action_dim,
            hidden_dim=args.hidden_dim,
            actor_dropout=args.actor_dropout,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            tau=args.tau,
            policy_delay=args.policy_delay,
            target_noise=args.target_noise,
            target_noise_clip=args.target_noise_clip,
        ).to(device)
        replay = EncodedReplayBuffer(args.buffer_size, observation_dim, action_dim)
        replay_generator = torch.Generator().manual_seed(args.seed)
        continuity_config = _continuity_config(
            args, observation_dim=observation_dim, action_dim=action_dim
        )
        n_collection_tasks = len(task_envs)
        if args.n_demo_episodes > 0 and n_collection_tasks != 1:
            raise ValueError(
                "Paper-style demo prefill requires one TD3 agent per task; pass "
                "--collection_task_id or set --n_demo_episodes 0 for a labeled multitask ablation."
            )

        if resume_path.exists():
            log.info("Restoring phased IBRL state from %s", resume_path)
            saved = torch.load(resume_path, map_location="cpu", weights_only=False)
            if int(saved.get("version", -1)) != RESUME_STATE_VERSION:
                raise ValueError(
                    f"Unsupported resume-state version {saved.get('version')}; "
                    f"expected {RESUME_STATE_VERSION}."
                )
            _validate_continuity_config(saved["continuity_config"], continuity_config)
            if tuple(saved["task_vocab"]) != task_vocab:
                raise ValueError("LIBERO task vocabulary changed since the previous phase.")
            if tuple(saved["camera_keys"]) != camera_keys:
                raise ValueError("FastWAM camera keys changed since the previous phase.")
            agent.load_checkpoint(saved["agent"])
            replay.load_state_dict(saved["replay"])
            trainer = saved["trainer"]
            _restore_rng_state(saved["rng"], replay_generator)
            elapsed_before_invocation = float(trainer.get("elapsed_s", 0.0))

            # The state is authoritative. Drop any JSONL tail written after the
            # most recent atomic checkpoint so a retry cannot duplicate trials.
            episode_records = [
                *trainer["bc_warmup"]["episodes"],
                *trainer["online_training"]["episodes"],
            ]
            with episode_log_path.open("w") as episode_file:
                for record in episode_records:
                    episode_file.write(json.dumps(record) + "\n")
        else:
            demo_prefill: dict[str, Any] | None = None
            if args.n_demo_episodes > 0:
                from lerobot.policies.fastwam.ibrl_demo_replay import (
                    prefill_demo_replay,
                    select_demo_episode_indices,
                )

                demo_task = task_vocab[0]
                episode_indices = select_demo_episode_indices(
                    args.demo_dataset_root,
                    repo_id=args.demo_dataset_repo_id,
                    task_description=demo_task,
                    n_episodes=args.n_demo_episodes,
                )
                log.info(
                    "Prefilling unified replay from %d expert demos for %r: %s",
                    len(episode_indices),
                    demo_task,
                    episode_indices,
                )
                prefill_stats = prefill_demo_replay(
                    replay,
                    dataset_root=args.demo_dataset_root,
                    repo_id=args.demo_dataset_repo_id,
                    episode_indices=episode_indices,
                    task_description=demo_task,
                    action_horizon=action_horizon,
                    gamma=args.gamma,
                    policy=policy,
                    vision_encoder=vision_encoder,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    env_postprocessor=env_postprocessor,
                    camera_keys=camera_keys,
                    task_to_index=task_to_index,
                    device=device,
                )
                demo_prefill = {
                    "dataset_repo_id": args.demo_dataset_repo_id,
                    "dataset_root": str(args.demo_dataset_root),
                    "episode_indices": list(prefill_stats.episode_indices),
                    "n_episodes": len(prefill_stats.episode_indices),
                    "transitions": prefill_stats.transitions,
                    "environment_steps": prefill_stats.environment_steps,
                    "terminal_transitions": prefill_stats.terminal_transitions,
                    "padded_action_steps": prefill_stats.padded_action_steps,
                }
            trainer = {
                "environment_decisions": 0,
                "environment_steps": 0,
                "environment_episodes": 0,
                "update_step_credit": 0,
                "latest_update": None,
                "per_task_episode_index": {},
                "demo_prefill": demo_prefill,
                "bc_warmup": _empty_collection_progress("bc_warmup"),
                "learner_warmstart": {
                    "requested_updates": args.learner_warmstart_updates,
                    "completed_updates": 0,
                    "released_protocol_default": args.learner_warmstart_updates == 0,
                },
                "online_training": _empty_collection_progress("online_train"),
                "reference_evaluation": None,
                "evaluation": None,
                "completed_phases": [],
                "elapsed_s": 0.0,
            }
            elapsed_before_invocation = 0.0

        def save_resume_state(reason: str) -> None:
            trainer["elapsed_s"] = elapsed_before_invocation + time.time() - invocation_started
            state = {
                "version": RESUME_STATE_VERSION,
                "reason": reason,
                "continuity_config": continuity_config,
                "task_vocab": task_vocab,
                "camera_keys": camera_keys,
                "action_horizon": action_horizon,
                "agent": agent.checkpoint(),
                "replay": replay.state_dict(),
                "rng": _capture_rng_state(replay_generator),
                "trainer": trainer,
            }
            _atomic_torch_save(state, resume_path)
            log.info(
                "Saved resumable IBRL state (%s): episodes=%d replay=%d updates=%d -> %s",
                reason,
                trainer["environment_episodes"],
                len(replay),
                agent.total_updates,
                resume_path,
            )

        if not resume_path.exists():
            save_resume_state("demo_prefill_complete")

        log.info(
            "Starting phased primitive-action IBRL-TD3: requested_phase=%s horizon=%d "
            "observation_dim=%d action_dim=%d replay=%d warmup=%d/%d online=%d/%d",
            args.phase,
            action_horizon,
            observation_dim,
            action_dim,
            len(replay),
            trainer["bc_warmup"]["n_episodes"],
            args.bc_warmup_episodes,
            trainer["online_training"]["n_episodes"],
            args.n_episodes,
        )

        def collect_until(
            progress: dict[str, Any],
            target_episodes: int,
            total_phase_episodes: int,
            *,
            hybrid: bool,
            learn: bool,
        ) -> None:
            completed = int(progress["n_episodes"])
            if target_episodes < completed:
                raise ValueError(
                    f"Requested {progress['phase']} endpoint {target_episodes} precedes "
                    f"already completed episode count {completed}."
                )
            full_schedule = _episode_schedule(task_envs, total_phase_episodes)
            schedule = full_schedule[completed:target_episodes]

            for group_name, task_id, env in schedule:
                phase_episode_index = int(progress["n_episodes"])
                episode_index = int(trainer["environment_episodes"])
                seed = args.seed + episode_index
                task_key = (group_name, task_id)
                init_state_index = int(trainer["per_task_episode_index"].get(task_key, 0))
                policy.reset()
                policy.attach_planner(None)
                observation, _ = env.reset(
                    seed=[seed],
                    options={"episode_index_offset": init_state_index},
                )
                max_env_steps = min(
                    int(env.call("_max_episode_steps")[0]), args.episode_length
                )
                context = _make_proposal_context(
                    observation,
                    env,
                    policy,
                    vision_encoder,
                    env_preprocessor,
                    env_postprocessor,
                    preprocessor,
                    postprocessor,
                    camera_keys=camera_keys,
                    task_to_index=task_to_index,
                    action_horizon=action_horizon,
                    device=device,
                )
                env_step = 0
                episode_return = 0.0
                episode_decisions = 0
                episode_bc_selected = 0
                episode_rl_selected = 0
                episode_success = False
                bc_q: float | None = None
                rl_q: float | None = None
                updates_before_episode = agent.total_updates

                while env_step < max_env_steps:
                    if hybrid:
                        selection = agent.select_hybrid_action(
                            context.observation,
                            context.bc_action,
                            exploration_noise=args.exploration_noise,
                        )
                        selected_action = selection.action
                        selected_bc = bool(selection.bc_selected.item())
                        bc_q = float(selection.bc_q.item())
                        rl_q = float(selection.rl_q.item())
                    else:
                        selected_action = context.bc_action
                        selected_bc = True

                    action_chunk = selected_action.reshape(
                        action_horizon, int(policy.config.action_dim)
                    ).detach().cpu()
                    (
                        next_raw_observation,
                        transition_reward,
                        transition_raw_reward,
                        transition_discount,
                        executed,
                        done,
                        success,
                    ) = _execute_chunk(
                        env,
                        action_chunk,
                        env_step=env_step,
                        max_env_steps=max_env_steps,
                        gamma=args.gamma,
                    )
                    env_step += executed
                    trainer["environment_steps"] += executed
                    episode_return += transition_raw_reward
                    episode_success = episode_success or success

                    if done:
                        next_observation = torch.zeros_like(context.observation)
                        next_bc_action = torch.zeros_like(context.bc_action)
                        next_context = None
                    else:
                        next_context = _make_proposal_context(
                            next_raw_observation,
                            env,
                            policy,
                            vision_encoder,
                            env_preprocessor,
                            env_postprocessor,
                            preprocessor,
                            postprocessor,
                            camera_keys=camera_keys,
                            task_to_index=task_to_index,
                            action_horizon=action_horizon,
                            device=device,
                        )
                        next_observation = next_context.observation
                        next_bc_action = next_context.bc_action

                    replay.add(
                        context.observation,
                        _canonicalize_executed_action(
                            selected_action,
                            executed_steps=executed,
                            action_horizon=action_horizon,
                            action_dim=int(policy.config.action_dim),
                        ),
                        transition_reward,
                        next_observation,
                        done,
                        transition_discount,
                        next_bc_action,
                    )
                    trainer["environment_decisions"] += 1
                    episode_decisions += 1
                    if selected_bc:
                        episode_bc_selected += 1
                    else:
                        episode_rl_selected += 1

                    if learn:
                        if len(replay) >= args.batch_size:
                            updates_due, trainer["update_step_credit"] = _consume_update_credit(
                                int(trainer["update_step_credit"]),
                                executed,
                                args.update_every_env_steps,
                            )
                            for _ in range(updates_due):
                                train_batch = replay.sample(
                                    args.batch_size, generator=replay_generator
                                ).to(device)
                                trainer["latest_update"] = agent.update(
                                    train_batch,
                                    max_grad_norm=args.max_grad_norm,
                                )
                        else:
                            trainer["update_step_credit"] += executed
                    if done:
                        break
                    if next_context is None:
                        raise RuntimeError(
                            "Nonterminal IBRL transition has no next proposal context."
                        )
                    context = next_context

                trainer["environment_episodes"] += 1
                trainer["per_task_episode_index"][task_key] = init_state_index + 1
                episode_metrics: dict[str, Any] = {
                    "phase": progress["phase"],
                    "episode": episode_index,
                    "phase_episode": phase_episode_index,
                    "task": f"{group_name}/{task_id}",
                    "seed": seed,
                    "init_state_index": init_state_index,
                    "success": episode_success,
                    "return": episode_return,
                    "env_steps": env_step,
                    "decisions": episode_decisions,
                    "bc_selected": episode_bc_selected,
                    "rl_selected": episode_rl_selected,
                    "bc_fraction": episode_bc_selected / max(episode_decisions, 1),
                    "replay_size": len(replay),
                    "updates": agent.total_updates,
                    "last_bc_q": bc_q,
                    "last_rl_q": rl_q,
                    "latest_update": trainer["latest_update"] if learn else None,
                }
                progress["n_episodes"] += 1
                progress["n_success"] += int(episode_success)
                progress["return_sum"] += episode_return
                progress["env_steps"] += env_step
                progress["decisions"] += episode_decisions
                progress["bc_selected"] += episode_bc_selected
                progress["rl_selected"] += episode_rl_selected
                progress["updates"] += agent.total_updates - updates_before_episode
                progress["episodes"].append(episode_metrics)
                with episode_log_path.open("a") as episode_file:
                    episode_file.write(json.dumps(episode_metrics) + "\n")
                log.info(
                    "%s episode %d/%d %s success=%s return=%.3f steps=%d "
                    "decisions=%d BC/RL=%d/%d replay=%d updates=%d",
                    progress["phase"],
                    progress["n_episodes"],
                    total_phase_episodes,
                    episode_metrics["task"],
                    episode_success,
                    episode_return,
                    env_step,
                    episode_decisions,
                    episode_bc_selected,
                    episode_rl_selected,
                    len(replay),
                    agent.total_updates,
                )
                if use_wandb:
                    import wandb

                    metrics = {
                        f"{progress['phase']}/success": float(episode_success),
                        f"{progress['phase']}/return": episode_return,
                        f"{progress['phase']}/bc_fraction": episode_metrics["bc_fraction"],
                        "replay/size": len(replay),
                    }
                    if learn and trainer["latest_update"] is not None:
                        metrics.update(
                            {
                                f"train/{key}": value
                                for key, value in trainer["latest_update"].items()
                                if value is not None and key != "updates"
                            }
                        )
                    wandb.log(metrics, step=trainer["environment_decisions"])
                if (
                    args.checkpoint_every_episodes > 0
                    and progress["n_episodes"] % args.checkpoint_every_episodes == 0
                ):
                    save_resume_state(
                        f"{progress['phase']}_episode_{progress['n_episodes']:04d}"
                    )

        def finish_warmup() -> None:
            collect_until(
                trainer["bc_warmup"],
                args.bc_warmup_episodes,
                args.bc_warmup_episodes,
                hybrid=False,
                learn=False,
            )
            if trainer["bc_warmup"]["updates"] != 0:
                raise RuntimeError("BC warmup must complete before any learner update.")
            warmstart = trainer["learner_warmstart"]
            remaining_updates = args.learner_warmstart_updates - int(
                warmstart["completed_updates"]
            )
            if remaining_updates < 0:
                raise ValueError("Resume state contains too many learner warm-start updates.")
            if remaining_updates and len(replay) < args.batch_size:
                raise ValueError(
                    "learner_warmstart_updates requires at least batch_size replay transitions."
                )
            for _ in range(remaining_updates):
                train_batch = replay.sample(args.batch_size, generator=replay_generator).to(device)
                trainer["latest_update"] = agent.update(
                    train_batch, max_grad_norm=args.max_grad_norm
                )
                warmstart["completed_updates"] += 1
            if "warmup" not in trainer["completed_phases"]:
                trainer["completed_phases"].append("warmup")
            save_resume_state("warmup_complete")

        def require_complete(phase: str, completed: int, expected: int) -> None:
            if completed != expected:
                raise RuntimeError(
                    f"Phase {phase!r} requires {expected} prior episodes, "
                    f"but state has {completed}."
                )

        def finish_online(target: int) -> None:
            require_complete(
                "online",
                int(trainer["bc_warmup"]["n_episodes"]),
                args.bc_warmup_episodes,
            )
            completed_warmstart = int(
                trainer["learner_warmstart"]["completed_updates"]
            )
            if completed_warmstart != args.learner_warmstart_updates:
                raise RuntimeError("Online phase requires the warm-start barrier to be complete.")
            collect_until(
                trainer["online_training"],
                target,
                args.n_episodes,
                hybrid=True,
                learn=True,
            )
            if target == args.n_episodes and "online" not in trainer["completed_phases"]:
                trainer["completed_phases"].append("online")
            save_resume_state(f"online_episode_{target:04d}_complete")

        def finish_evaluation(*, reference_only: bool) -> None:
            key = "reference_evaluation" if reference_only else "evaluation"
            phase_name = "reference_eval" if reference_only else "hybrid_eval"
            if reference_only:
                require_complete(
                    phase_name,
                    int(trainer["online_training"]["n_episodes"]),
                    args.n_episodes,
                )
            else:
                reference_count = (
                    0
                    if trainer["reference_evaluation"] is None
                    else int(trainer["reference_evaluation"]["n_episodes"])
                )
                require_complete(phase_name, reference_count, args.eval_n_episodes)

            completed = (
                0 if trainer[key] is None else int(trainer[key]["n_episodes"])
            )
            shard_size = (
                args.eval_n_episodes
                if args.checkpoint_every_episodes <= 0
                else args.checkpoint_every_episodes
            )
            while completed < args.eval_n_episodes:
                count = min(shard_size, args.eval_n_episodes - completed)
                shard = _evaluate_hybrid(
                    agent,
                    task_envs,
                    policy,
                    vision_encoder,
                    env_preprocessor,
                    env_postprocessor,
                    preprocessor,
                    postprocessor,
                    camera_keys=camera_keys,
                    task_to_index=task_to_index,
                    action_horizon=action_horizon,
                    n_episodes=count,
                    total_episodes=args.eval_n_episodes,
                    episode_start=completed,
                    episode_length=args.episode_length,
                    gamma=args.gamma,
                    start_seed=args.eval_seed,
                    init_state_offsets=trainer["per_task_episode_index"],
                    device=device,
                    reference_only=reference_only,
                )
                if shard is None:
                    raise RuntimeError("A non-empty evaluation shard returned no metrics.")
                trainer[key] = _merge_evaluation_results(trainer[key], shard)
                completed = int(trainer[key]["n_episodes"])
                save_resume_state(f"{phase_name}_episode_{completed:04d}")
            if phase_name not in trainer["completed_phases"]:
                trainer["completed_phases"].append(phase_name)
            save_resume_state(f"{phase_name}_complete")

        if args.phase in ("all", "warmup"):
            finish_warmup()
        if args.phase in ("all", "online"):
            endpoint = (
                args.n_episodes
                if args.phase == "all" or args.online_episode_end is None
                else args.online_episode_end
            )
            finish_online(endpoint)
        if args.phase in ("all", "reference_eval"):
            finish_evaluation(reference_only=True)
        if args.phase in ("all", "hybrid_eval"):
            finish_evaluation(reference_only=False)

        final_complete = (
            trainer["reference_evaluation"] is not None
            and trainer["evaluation"] is not None
            and int(trainer["reference_evaluation"]["n_episodes"]) == args.eval_n_episodes
            and int(trainer["evaluation"]["n_episodes"]) == args.eval_n_episodes
        )
        if not final_complete:
            status = {
                "phase": args.phase,
                "resume_state": str(resume_path),
                "replay_size": len(replay),
                "updates": agent.total_updates,
                "bc_warmup": _collection_summary(trainer["bc_warmup"]),
                "online_training": _collection_summary(trainer["online_training"]),
                "reference_eval_episodes": (
                    0
                    if trainer["reference_evaluation"] is None
                    else trainer["reference_evaluation"]["n_episodes"]
                ),
                "hybrid_eval_episodes": (
                    0 if trainer["evaluation"] is None else trainer["evaluation"]["n_episodes"]
                ),
                "completed_phases": trainer["completed_phases"],
            }
            (output_dir / "phase_status.json").write_text(json.dumps(status, indent=2) + "\n")
            return status

        bc_warmup = _collection_summary(trainer["bc_warmup"])
        online_training = _collection_summary(trainer["online_training"])
        reference_evaluation = trainer["reference_evaluation"]
        evaluation = trainer["evaluation"]
        paired_evaluation = _paired_evaluation_summary(reference_evaluation, evaluation)
        checkpoint_path = _save_checkpoint(
            agent,
            output_dir / "td3_final.pt",
            args=args,
            task_vocab=task_vocab,
            camera_keys=camera_keys,
            action_horizon=action_horizon,
            global_decisions=trainer["environment_decisions"],
        )
        released_protocol_settings = {
            "expert_demo_episodes": args.n_demo_episodes == 10,
            "separate_bc_warmup_episodes": args.bc_warmup_episodes == 40,
            "no_updates_during_bc_warmup": bc_warmup["updates"] == 0,
            "no_offline_learner_warmstart": args.learner_warmstart_updates == 0,
            "update_every_two_primitive_actions": args.update_every_env_steps == 2,
            "actor_dropout_half": abs(args.actor_dropout - 0.5) < 1e-12,
            "actor_hidden_dim_1024": args.hidden_dim == 1024,
            "constant_exploration_std_point_one": abs(args.exploration_noise - 0.1) < 1e-12,
            "eval_episodes_50": args.eval_n_episodes == 50,
            "single_task_agent": n_collection_tasks == 1,
            "hard_target_q_chooser": True,
            "unified_uniform_replay": True,
        }
        summary = {
            "algorithm": "ibrl_td3",
            "protocol": "released_pixel_ibrl_mapped_to_fastwam_td3",
            "protocol_sources": {
                "paper": "https://www.roboticsproceedings.org/rss20/p056.pdf",
                "released_pixel_config": (
                    "https://github.com/hengyuan-hu/ibrl/blob/main/release/cfgs/"
                    "robomimic_rl/can_ibrl.yaml"
                ),
                "released_training_loop": (
                    "https://github.com/hengyuan-hu/ibrl/blob/main/train_rl.py"
                ),
                "released_q_agent": (
                    "https://github.com/hengyuan-hu/ibrl/blob/main/rl/q_agent.py"
                ),
            },
            "ibrl_mechanism_faithful": True,
            "paper_primitive_action_setting": action_horizon == 1,
            "paper_exact_reproduction": False,
            "paper_exact_reproduction_reason": (
                "A TD3 backbone and frozen DINO features replace the paper's "
                "end-to-end visual Q agent."
            ),
            "released_protocol_settings": released_protocol_settings,
            "released_protocol_defaults_match": all(released_protocol_settings.values()),
            "action_level_proposal_backup_at_action_horizon_one": True,
            "bc_policy_frozen": True,
            "uses_target_q_for_action_selection": True,
            "uses_bc_and_target_rl_in_bootstrap": True,
            "bc_proposals_cached_in_replay": True,
            "phased_resume_state_version": RESUME_STATE_VERSION,
            "completed_phases": trainer["completed_phases"],
            "demo_prefill": trainer["demo_prefill"],
            "action_horizon": action_horizon,
            "bc_warmup": bc_warmup,
            "learner_warmstart": trainer["learner_warmstart"],
            "online_training": online_training,
            "n_episodes": online_training["n_episodes"],
            "n_success": online_training["n_success"],
            "collection_pc_success": online_training["pc_success"],
            "reference_warmup_pc_success": bc_warmup["pc_success"],
            "eval_pc_success": evaluation["pc_success"],
            "evaluation": evaluation,
            "reference_evaluation": reference_evaluation,
            "paired_evaluation": paired_evaluation,
            "reference_eval_pc_success": reference_evaluation["pc_success"],
            "env_steps": trainer["environment_steps"],
            "decisions": trainer["environment_decisions"],
            "bc_selected": online_training["bc_selected"],
            "rl_selected": online_training["rl_selected"],
            "bc_fraction": online_training["bc_fraction"],
            "replay_size": len(replay),
            "updates": agent.total_updates,
            "unspent_update_env_steps": trainer["update_step_credit"],
            "checkpoint": str(checkpoint_path),
            "resume_state": str(resume_path),
            "elapsed_s": elapsed_before_invocation + time.time() - invocation_started,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        log.info(
            "IBRL complete: warmup_ref=%s online=%s eval_hybrid=%s eval_ref=%s "
            "BC/RL=%d/%d checkpoint=%s",
            summary["reference_warmup_pc_success"],
            summary["collection_pc_success"],
            summary["eval_pc_success"],
            summary["reference_eval_pc_success"],
            online_training["bc_selected"],
            online_training["rl_selected"],
            checkpoint_path,
        )
        if use_wandb:
            import wandb

            wandb.log(
                {
                    "eval/pc_success": evaluation["pc_success"],
                    "eval/mean_return": evaluation["mean_return"],
                    "eval/bc_fraction": evaluation["bc_fraction"],
                    "eval_reference/pc_success": reference_evaluation["pc_success"],
                    "eval_reference/mean_return": reference_evaluation["mean_return"],
                },
                step=trainer["environment_decisions"],
            )
        return summary
    finally:
        if envs is not None:
            close_envs(envs)
        if use_wandb:
            import wandb

            wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="FastWAM + TD3 imitation-bootstrapped RL")
    parser.add_argument("--fastwam_ckpt", required=True)
    parser.add_argument("--task", default="libero_10")
    parser.add_argument("--episode_length", type=int, default=520)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--eval_n_episodes", type=int, default=50)
    parser.add_argument(
        "--collection_task_id",
        type=int,
        default=None,
        help=(
            "Optional single LIBERO task id. Otherwise train one task-conditioned "
            "agent on all tasks."
        ),
    )
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=1,
        help=(
            "Actions per Q decision. Defaults to 1 for the paper's action-level "
            "proposal and backup equations; larger values are explicit SMDP ablations."
        ),
    )
    parser.add_argument(
        "--phase",
        choices=PHASE_CHOICES,
        default="all",
        help="Run the full experiment or one deterministic resumable phase.",
    )
    parser.add_argument(
        "--online_episode_end",
        type=int,
        default=None,
        help=(
            "Cumulative online-episode endpoint for --phase online. Repeated jobs can "
            "advance the same state to 25, 50, 75, then 100 episodes."
        ),
    )
    parser.add_argument(
        "--resume_state",
        type=Path,
        default=None,
        help="Shared phase checkpoint (default: OUTPUT_DIR/resume_state.pt).",
    )

    parser.add_argument("--vision_encoder_model", default="facebook/dinov2-large")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--actor_dropout", type=float, default=0.5)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--demo_dataset_repo_id", default="HuggingFaceVLA/libero")
    parser.add_argument("--demo_dataset_root", type=Path, default=None)
    parser.add_argument("--n_demo_episodes", type=int, default=10)
    parser.add_argument("--bc_warmup_episodes", type=int, default=40)
    parser.add_argument(
        "--update_every_env_steps",
        type=int,
        default=2,
        help="Run one learner update for this many executed primitive environment actions.",
    )
    parser.add_argument(
        "--learner_warmstart_updates",
        type=int,
        default=0,
        help=(
            "Optional learner-only updates after replay warmup and before hybrid rollout. "
            "The released IBRL protocol uses 0; nonzero is a conservative ablation."
        ),
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--policy_delay", type=int, default=2)
    parser.add_argument("--exploration_noise", type=float, default=0.1)
    parser.add_argument("--target_noise", type=float, default=0.1)
    parser.add_argument("--target_noise_clip", type=float, default=0.3)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)

    parser.add_argument("--checkpoint_every_episodes", type=int, default=10)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_seed", type=int, default=100_000)
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_run_name", default="")
    return parser.parse_args()


if __name__ == "__main__":
    train_ibrl(parse_args())
