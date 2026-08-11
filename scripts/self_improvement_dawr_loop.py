#!/usr/bin/env python
"""Traditional diffusion advantage-weighted regression for FastWAM.

This baseline is deliberately independent of qplanning and IBRL:

1. The stochastic FastWAM actor acts directly, without a planner.
2. A newly initialized state-value head learns TD(lambda) returns from online
   success and failure trajectories.
3. FastWAM is updated with clipped exponential advantage-weighted
   flow-matching regression.

The value critic uses frozen generic DINOv2 image features for practical pixel
RL, but loads no Q-function architecture, checkpoint, targets, or replay.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import os
import resource
import signal
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from lerobot.envs.utils import close_envs
from lerobot.policies.fastwam.dawr_dataset import (
    DAWR_REWARD,
    DAWR_TERMINAL,
    DAWR_TRAJECTORY_INDEX,
    DAWR_WEIGHT,
    DAWRDecisionWriter,
    LossWeightedDataset,
    exponential_advantage_weights,
    libero_action_to_fastwam,
    load_growing_dawr_dataset,
)
from lerobot.policies.fastwam.dawr_value import (
    DAWRValueCritic,
    EncodedDAWRReplay,
    FrozenDINOEncoder,
    td_lambda_returns,
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
os.environ.setdefault("HF_HOME", "/storage/project/r-agarg35-0/shared/huggingface_cache")


def _enable_fatal_tracing() -> None:
    """Emit Python stacks on fatal native signals and on an explicit SIGUSR1."""
    faulthandler.enable(file=sys.stderr, all_threads=True)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)


def _trace_runtime(label: str) -> None:
    """Synchronize CUDA and report process/GPU memory at a rollout boundary."""
    cuda_stats = "cuda=unavailable"
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gib = 1024**3
        cuda_stats = (
            f"cuda_device={torch.cuda.current_device()} "
            f"allocated_gib={torch.cuda.memory_allocated() / gib:.3f} "
            f"reserved_gib={torch.cuda.memory_reserved() / gib:.3f} "
            f"max_allocated_gib={torch.cuda.max_memory_allocated() / gib:.3f} "
            f"max_reserved_gib={torch.cuda.max_memory_reserved() / gib:.3f}"
        )
    # Linux reports ru_maxrss in KiB. This is the high-water mark, not current RSS.
    max_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    log.info("[TRACE] %s max_rss_gib=%.3f %s", label, max_rss_gib, cuda_stats)


def _trace_tensor(name: str, tensor: torch.Tensor) -> None:
    """Fail at the first non-finite tensor and otherwise log its numeric range."""
    detached = tensor.detach()
    finite = torch.isfinite(detached)
    if not bool(finite.all().item()):
        bad_indices = (~finite).nonzero(as_tuple=False)[:8].cpu().tolist()
        raise FloatingPointError(f"{name} contains non-finite values at {bad_indices}")
    log.info(
        "[TRACE] tensor=%s shape=%s dtype=%s device=%s min=%.7g max=%.7g mean=%.7g",
        name,
        tuple(detached.shape),
        detached.dtype,
        detached.device,
        float(detached.min().item()),
        float(detached.max().item()),
        float(detached.float().mean().item()),
    )


def _task_description(env) -> str:
    if hasattr(env, "envs") and len(env.envs) > 0:
        task = getattr(env.envs[0], "task_description", None) or getattr(env.envs[0], "task", None)
        if isinstance(task, str):
            return task
    return ""


def task_descriptions_from_envs(envs) -> tuple[str, ...]:
    descriptions = {
        task
        for group in envs.values()
        for env in group.values()
        if (task := _task_description(env))
    }
    return tuple(sorted(descriptions))


def configure_actor_finetune(policy, *, gradient_checkpointing: bool = True) -> list[torch.nn.Parameter]:
    """Train FastWAM's action/proprio path while freezing its video path."""
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in policy.model.action_expert.parameters():
        parameter.requires_grad_(True)
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
        raise RuntimeError("FastWAM DAWR finetuning produced no trainable actor parameters.")
    return trainable


def _run_episode(
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    *,
    camera_keys: tuple[str, ...],
    seed: int,
    episode_index: int = 0,
    trace_rollouts: bool = False,
) -> dict[str, Any]:
    """Run one direct FastWAM trajectory in the multi-step decision MDP."""
    import numpy as np

    from lerobot.envs.utils import add_envs_task, preprocess_observation

    trace_prefix = f"episode={episode_index} seed={seed}"
    if trace_rollouts:
        log.info("[TRACE] %s policy.reset begin", trace_prefix)
        _trace_runtime(f"{trace_prefix} before_policy_reset")
    policy.reset()
    policy.attach_planner(None)
    if trace_rollouts:
        log.info("[TRACE] %s env.reset begin", trace_prefix)
    observation, reset_info = env.reset(seed=[seed])
    if trace_rollouts:
        log.info("[TRACE] %s env.reset done info=%r", trace_prefix, reset_info)
        _trace_runtime(f"{trace_prefix} after_env_reset")
    done = np.array([False])
    max_env_steps = int(env.call("_max_episode_steps")[0])
    env_step = 0
    successes: list[bool] = []
    decisions: list[dict[str, Any]] = []
    decision_index = 0

    while not np.all(done) and env_step < max_env_steps:
        decision_prefix = f"{trace_prefix} decision={decision_index} env_step={env_step}"
        if trace_rollouts:
            log.info("[TRACE] %s observation preprocessing begin", decision_prefix)
            _trace_runtime(f"{decision_prefix} decision_begin")
        batch = preprocess_observation(observation)
        batch = add_envs_task(env, batch)
        batch = env_preprocessor(batch)
        required = (*camera_keys, OBS_STATE)
        missing = [
            key
            for key in required
            if key not in batch or not isinstance(batch[key], torch.Tensor)
        ]
        if missing:
            raise KeyError(f"Processed LIBERO observation is missing DAWR keys: {missing}")
        if trace_rollouts:
            for key in required:
                _trace_tensor(f"{decision_prefix} processed[{key}]", batch[key])

        decision: dict[str, Any] = {
            OBS_STATE: batch[OBS_STATE][0].detach().cpu().clone(),
        }
        for key in camera_keys:
            decision[key] = batch[key][0].detach().cpu().clone()

        policy_batch = preprocessor(batch)
        if trace_rollouts:
            for key, value in policy_batch.items():
                if isinstance(value, torch.Tensor):
                    _trace_tensor(f"{decision_prefix} policy_batch[{key}]", value)
            log.info("[TRACE] %s predict_action_chunk begin", decision_prefix)
        with torch.no_grad():
            normalized_chunk = policy.predict_action_chunk(policy_batch)
        if trace_rollouts:
            _trace_runtime(f"{decision_prefix} after_predict_action_chunk")
            _trace_tensor(f"{decision_prefix} normalized_chunk_gpu", normalized_chunk)
        normalized_chunk = normalized_chunk[0].detach().cpu().float()
        horizon, action_dim = normalized_chunk.shape
        raw_chunk = postprocessor(normalized_chunk).detach().cpu().float()
        libero_chunk = env_postprocessor({ACTION: raw_chunk})[ACTION].detach().cpu().float()
        if trace_rollouts:
            _trace_tensor(f"{decision_prefix} normalized_chunk", normalized_chunk)
            _trace_tensor(f"{decision_prefix} raw_chunk", raw_chunk)
            _trace_tensor(f"{decision_prefix} libero_chunk", libero_chunk)

        max_actions = min(int(policy.config.n_action_steps), horizon, max_env_steps - env_step)
        executed_fastwam: list[torch.Tensor] = []
        decision_reward = 0.0
        for action_index in range(max_actions):
            if np.all(done):
                break
            env_action = libero_chunk[action_index].unsqueeze(0)
            actor_action = raw_chunk[action_index].clone()
            actor_action[-1] = libero_action_to_fastwam(env_action)[0, -1]
            executed_fastwam.append(actor_action)

            if trace_rollouts:
                log.info(
                    "[TRACE] %s action_index=%d env.step begin action=%s",
                    decision_prefix,
                    action_index,
                    env_action[0].tolist(),
                )
            observation, reward, terminated, truncated, step_info = env.step(env_action.numpy())
            step_reward = float(np.asarray(reward).reshape(-1)[0])
            if "final_info" in step_info:
                step_success = bool(step_info["final_info"]["is_success"].tolist()[0])
            else:
                step_success = False
            # Use the benchmark's verified binary success as the sparse RL
            # reward even if a wrapper omits it from the numeric reward field.
            decision_reward += max(step_reward, float(step_success))
            successes.append(step_success)
            done = terminated | truncated | done
            env_step += 1
            if trace_rollouts:
                log.info(
                    "[TRACE] %s action_index=%d env.step done reward=%.7g "
                    "terminated=%s truncated=%s success=%s new_env_step=%d",
                    decision_prefix,
                    action_index,
                    step_reward,
                    np.asarray(terminated).tolist(),
                    np.asarray(truncated).tolist(),
                    step_success,
                    env_step,
                )

        if not executed_fastwam:
            raise RuntimeError("FastWAM decision produced no executable actions.")
        executed = torch.stack(executed_fastwam)
        n_executed = len(executed)
        # Only executed actions contribute to the loss. Keep the actor's own
        # sampled tail as unsupervised diffusion context instead of fabricating
        # repeated actions; the mask below excludes that tail from regression.
        padded_action = raw_chunk.clone()
        padded_action[:n_executed] = executed
        action_is_pad = torch.arange(horizon) >= n_executed
        decision.update(
            {
                ACTION: padded_action,
                f"{ACTION}_is_pad": action_is_pad,
                DAWR_REWARD: decision_reward,
                DAWR_TERMINAL: bool(np.all(done) or env_step >= max_env_steps),
            }
        )
        decisions.append(decision)
        if trace_rollouts:
            log.info(
                "[TRACE] %s decision done executed=%d decision_reward=%.7g done=%s",
                decision_prefix,
                n_executed,
                decision_reward,
                np.asarray(done).tolist(),
            )
            _trace_runtime(f"{decision_prefix} decision_done")
        decision_index += 1

    if not decisions:
        raise RuntimeError("LIBERO episode ended before the actor made a decision.")
    # Treat the finite LIBERO horizon as an episodic boundary. This prevents
    # TD(lambda) returns from leaking across independently reset episodes.
    decisions[-1][DAWR_TERMINAL] = True
    if trace_rollouts:
        log.info(
            "[TRACE] %s episode done decisions=%d env_steps=%d success=%s return=%.7g",
            trace_prefix,
            len(decisions),
            env_step,
            any(successes),
            sum(float(decision[DAWR_REWARD]) for decision in decisions),
        )
        _trace_runtime(f"{trace_prefix} episode_done")
    return {
        "decisions": decisions,
        "task": _task_description(env),
        "success": any(successes),
        "n_env_steps": env_step,
        "episode_return": sum(float(decision[DAWR_REWARD]) for decision in decisions),
    }


def collect_dawr_episodes(
    policy,
    envs,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    *,
    n_episodes: int,
    writer: DAWRDecisionWriter,
    camera_keys: tuple[str, ...],
    start_seed: int,
    collection_task_id: int | None = None,
    trace_rollouts: bool = False,
) -> dict[str, Any]:
    """Collect exactly ``n_episodes`` with the unplanned actor."""
    task_envs = [
        (group_name, task_id, env)
        for group_name, group in envs.items()
        for task_id, env in group.items()
    ]
    if collection_task_id is not None:
        task_envs = [entry for entry in task_envs if int(entry[1]) == collection_task_id]
        log.info(
            "Diagnostic collection filter active: task_id=%d (%d environment(s))",
            collection_task_id,
            len(task_envs),
        )
    if not task_envs:
        raise ValueError(
            "No task environments were created"
            + (f" for collection_task_id={collection_task_id}." if collection_task_id is not None else ".")
        )
    base, remainder = divmod(n_episodes, len(task_envs))
    counts = [base + (index < remainder) for index in range(len(task_envs))]
    total = 0
    n_success = 0
    episode_returns: list[float] = []
    per_task: dict[str, dict[str, int]] = {}

    for (group_name, task_id, env), count in zip(task_envs, counts, strict=True):
        task_success = 0
        for _ in range(count):
            log.info(
                "Collection episode %d/%d begin: %s/%s seed=%d",
                total + 1,
                n_episodes,
                group_name,
                task_id,
                start_seed + total,
            )
            episode = _run_episode(
                env,
                policy,
                env_preprocessor,
                env_postprocessor,
                preprocessor,
                postprocessor,
                camera_keys=camera_keys,
                seed=start_seed + total,
                episode_index=total,
                trace_rollouts=trace_rollouts,
            )
            log.info(
                "Collection episode %d rollout done: decisions=%d env_steps=%d success=%s; "
                "writer.add_episode begin",
                total + 1,
                len(episode["decisions"]),
                episode["n_env_steps"],
                episode["success"],
            )
            writer.add_episode(
                episode["decisions"],
                task=episode["task"],
                success=episode["success"],
            )
            log.info("Collection episode %d writer.add_episode done", total + 1)
            total += 1
            episode_returns.append(float(episode["episode_return"]))
            if episode["success"]:
                n_success += 1
                task_success += 1
        per_task[f"{group_name}/{task_id}"] = {"collected": count, "successful": task_success}
        log.info("  %s/%s: %d/%d successful", group_name, task_id, task_success, count)

    log.info("Collection writer.finalize begin")
    writer.finalize()
    log.info("Collection writer.finalize done")
    success_rate = 100.0 * n_success / max(total, 1)
    log.info(
        "Direct actor collection: %d episodes, %d decisions, %d successes (%.1f%%)",
        total,
        writer.num_decisions,
        n_success,
        success_rate,
    )
    return {
        "n_collected": total,
        "n_decisions_collected": writer.num_decisions,
        "n_success": n_success,
        "pc_success": success_rate,
        "mean_episode_return": sum(episode_returns) / max(len(episode_returns), 1),
        "per_task": per_task,
    }


@torch.no_grad()
def encode_value_replay(
    dataset: Dataset,
    encoder: FrozenDINOEncoder,
    *,
    camera_keys: tuple[str, ...],
    task_to_index: dict[str, int],
    batch_size: int,
    num_workers: int,
) -> EncodedDAWRReplay:
    """Encode all replay observations once; value optimization then stays cheap."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    vision_features: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    task_indices: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    terminals: list[torch.Tensor] = []
    trajectory_indices: list[torch.Tensor] = []

    encoder.eval()
    for batch_index, batch in enumerate(loader, start=1):
        images = torch.stack([batch[key] for key in camera_keys], dim=1)
        vision_features.append(encoder(images).cpu())
        states.append(batch[OBS_STATE].float().cpu())
        unknown_tasks = [task for task in batch["task"] if task not in task_to_index]
        if unknown_tasks:
            raise ValueError(f"Value replay contains unknown tasks: {sorted(set(unknown_tasks))}")
        task_indices.append(
            torch.tensor([task_to_index[task] for task in batch["task"]], dtype=torch.long)
        )
        rewards.append(batch[DAWR_REWARD].float().reshape(-1).cpu())
        terminals.append(batch[DAWR_TERMINAL].bool().reshape(-1).cpu())
        trajectory_indices.append(batch[DAWR_TRAJECTORY_INDEX].long().reshape(-1).cpu())
        if batch_index % 20 == 0 or batch_index == len(loader):
            log.info("  Visual replay encoding: %d/%d batches", batch_index, len(loader))

    return EncodedDAWRReplay(
        vision_features=torch.cat(vision_features),
        state=torch.cat(states),
        task_index=torch.cat(task_indices),
        reward=torch.cat(rewards),
        terminal=torch.cat(terminals),
        trajectory_index=torch.cat(trajectory_indices),
    )


@torch.no_grad()
def predict_replay_values(
    critic: DAWRValueCritic,
    replay: EncodedDAWRReplay,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    critic.eval()
    values: list[torch.Tensor] = []
    for start in range(0, len(replay), batch_size):
        stop = min(start + batch_size, len(replay))
        values.append(
            critic(
                replay.vision_features[start:stop].to(device, non_blocking=True),
                replay.state[start:stop].to(device, non_blocking=True),
                replay.task_index[start:stop].to(device, non_blocking=True),
            ).cpu()
        )
    return torch.cat(values)


def finetune_value_critic(
    critic: DAWRValueCritic,
    replay: EncodedDAWRReplay,
    targets: torch.Tensor,
    *,
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    global_step: int,
    use_wandb: bool,
) -> tuple[float, int]:
    """Fit the fresh value head to frozen TD(lambda) targets."""
    critic.train()
    optimizer = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=weight_decay)
    losses: list[float] = []
    for local_step in range(1, steps + 1):
        indices = torch.randint(len(replay), (batch_size,))
        prediction = critic(
            replay.vision_features[indices].to(device, non_blocking=True),
            replay.state[indices].to(device, non_blocking=True),
            replay.task_index[indices].to(device, non_blocking=True),
        )
        target = targets[indices].to(device, non_blocking=True)
        loss = torch.nn.functional.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip_norm)
        optimizer.step()

        losses.append(float(loss.detach()))
        global_step += 1
        if local_step == 1 or local_step % 25 == 0 or local_step == steps:
            window = min(25, len(losses))
            recent = sum(losses[-window:]) / window
            log.info(
                "  value %d/%d mse=%.6f grad=%.3f",
                local_step,
                steps,
                recent,
                float(grad_norm),
            )
            if use_wandb:
                import wandb

                wandb.log(
                    {
                        "value/mse": recent,
                        "value/grad_norm": float(grad_norm),
                        "value/lr": lr,
                    },
                    step=global_step,
                )
    critic.eval()
    return sum(losses) / len(losses), global_step


def compute_actor_weights(
    critic: DAWRValueCritic,
    replay: EncodedDAWRReplay,
    *,
    device: torch.device,
    value_batch_size: int,
    gamma: float,
    td_lambda: float,
    beta: float,
    max_weight: float,
    min_advantage_std: float,
    normalize_weight_mean: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Recompute post-update TD(lambda) advantages and freeze actor weights."""
    values = predict_replay_values(
        critic,
        replay,
        device=device,
        batch_size=value_batch_size,
    )
    advantages, returns = td_lambda_returns(
        replay.reward,
        replay.terminal,
        replay.trajectory_index,
        values,
        gamma=gamma,
        td_lambda=td_lambda,
    )
    advantage_std = float(advantages.std(unbiased=False))
    if advantage_std < min_advantage_std:
        raise RuntimeError(
            "TD(lambda) advantages are too flat for defensible DAWR standardization: "
            f"std={advantage_std:.6g} < {min_advantage_std:.6g}."
        )
    weights, stats = exponential_advantage_weights(
        advantages,
        beta=beta,
        max_weight=max_weight,
        standardize=True,
    )
    stats.update(
        {
            "value_mean": float(values.mean()),
            "value_std": float(values.std(unbiased=False)),
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std(unbiased=False)),
            "positive_advantage_fraction": float((advantages > 0).float().mean()),
            "raw_weight_mean": float(weights.mean()),
            "raw_weight_max": float(weights.max()),
        }
    )
    target_variance = returns.var(unbiased=False)
    if float(target_variance) > 1e-8:
        explained_variance = 1.0 - (returns - values).var(unbiased=False) / target_variance
        stats["value_explained_variance"] = float(explained_variance)
    else:
        stats["value_explained_variance"] = 0.0

    if normalize_weight_mean:
        weights = weights / weights.mean().clamp(min=1e-8)
    stats.update(
        {
            "actor_weight_mean": float(weights.mean()),
            "actor_weight_min": float(weights.min()),
            "actor_weight_max": float(weights.max()),
        }
    )
    log.info(
        "DAWR: A=%.4f±%.4f value_EV=%.3f raw_w=%.2f actor_w=[%.3g, %.3g] ESS=%.1f/%d",
        stats["advantage_mean"],
        stats["advantage_std"],
        stats["value_explained_variance"],
        stats["raw_weight_mean"],
        stats["actor_weight_min"],
        stats["actor_weight_max"],
        stats["effective_sample_size"],
        len(weights),
    )
    if stats["effective_sample_fraction"] < 0.05:
        log.warning(
            "DAWR effective sample size is below 5%% (%.3f).",
            stats["effective_sample_fraction"],
        )
    return weights, stats


def finetune_actor(
    policy,
    preprocessor,
    online_dataset: Dataset,
    online_weights: torch.Tensor,
    *,
    steps: int,
    lr: float,
    batch_size: int,
    grad_clip_norm: float,
    num_workers: int,
    global_step: int,
    use_wandb: bool,
) -> tuple[float, int]:
    """Run advantage-weighted FastWAM flow-matching updates."""
    trainable = configure_actor_finetune(policy)
    weighted_dataset = LossWeightedDataset(online_dataset, online_weights)
    sampler = WeightedRandomSampler(
        [1.0 / len(weighted_dataset)] * len(weighted_dataset),
        num_samples=steps * batch_size,
        replacement=True,
    )
    loader = DataLoader(
        weighted_dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
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
        sample_weights = batch.pop(DAWR_WEIGHT)
        batch = preprocessor(batch)
        optimizer.zero_grad(set_to_none=True)
        loss, loss_dict = policy.forward(batch, sample_weights=sample_weights)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
        optimizer.step()

        losses.append(float(loss_dict["loss_action"]))
        global_step += 1
        if local_step == 1 or local_step % 10 == 0 or local_step == steps:
            window = min(10, len(losses))
            recent = sum(losses[-window:]) / window
            log.info(
                "  actor %d/%d weighted_loss=%.5f weight=%.3f grad=%.3f",
                local_step,
                steps,
                recent,
                float(sample_weights.float().mean()),
                float(grad_norm),
            )
            if use_wandb:
                import wandb

                wandb.log(
                    {
                        "actor/weighted_action_loss": recent,
                        "actor/sample_weight": float(sample_weights.float().mean()),
                        "actor/grad_norm": float(grad_norm),
                        "actor/lr": lr,
                    },
                    step=global_step,
                )
    policy.eval()
    return sum(losses) / len(losses), global_step


def save_fastwam_checkpoint(policy, preprocessor, postprocessor, checkpoint_dir: Path) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    policy.save_pretrained(checkpoint_dir)
    preprocessor.save_pretrained(checkpoint_dir)
    postprocessor.save_pretrained(checkpoint_dir)
    log.info("Saved FastWAM checkpoint to %s", checkpoint_dir)
    return checkpoint_dir


def save_value_checkpoint(
    critic: DAWRValueCritic,
    path: Path,
    *,
    task_vocab: tuple[str, ...],
    encoder_model: str,
    iteration: int,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": critic.state_dict(),
            "config": critic.checkpoint_config(),
            "task_vocab": list(task_vocab),
            "encoder_model": encoder_model,
            "iteration": iteration,
        },
        path,
    )
    log.info("Saved fresh DAWR value critic to %s", path)
    return path


def make_value_critic(
    args,
    *,
    task_vocab: tuple[str, ...],
    n_cameras: int,
    state_dim: int,
) -> DAWRValueCritic:
    from transformers import AutoConfig

    vision_config = AutoConfig.from_pretrained(
        args.value_encoder_model,
        local_files_only=True,
    )
    expected_config = {
        "vision_dim": int(vision_config.hidden_size),
        "n_cameras": n_cameras,
        "state_dim": state_dim,
        "n_tasks": len(task_vocab),
        "hidden_dim": args.value_hidden_dim,
        "dropout": args.value_dropout,
    }
    critic = DAWRValueCritic(**expected_config)
    if args.value_ckpt:
        checkpoint = torch.load(args.value_ckpt, map_location="cpu", weights_only=False)
        if tuple(checkpoint["task_vocab"]) != task_vocab:
            raise ValueError("Value checkpoint task vocabulary does not match the current environment.")
        if checkpoint["encoder_model"] != args.value_encoder_model:
            raise ValueError("Value checkpoint uses a different frozen visual encoder.")
        if checkpoint["config"] != critic.checkpoint_config():
            raise ValueError("Value checkpoint architecture does not match the requested critic.")
        critic.load_state_dict(checkpoint["model_state_dict"])
        log.info("Loaded DAWR value critic from %s", args.value_ckpt)
    else:
        log.info("Initialized a fresh DAWR state-value critic (no Q checkpoint).")
    return critic


def _setup_actor_and_env(args, device: torch.device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.random_utils import set_seed

    set_seed(args.seed)
    env_cfg = LiberoEnv(
        task=args.task,
        episode_length=args.episode_length,
        observation_height=224,
        observation_width=224,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    effective_episode_lengths = {
        int(length)
        for group in envs.values()
        for env in group.values()
        for length in env.call("_max_episode_steps")
    }
    if effective_episode_lengths != {args.episode_length}:
        raise RuntimeError(
            "LIBERO did not apply the requested DAWR episode length: "
            f"requested={args.episode_length}, effective={sorted(effective_episode_lengths)}"
        )
    log.info("Using a maximum LIBERO episode length of %d steps.", args.episode_length)
    policy_cfg = PreTrainedConfig.from_pretrained(args.fastwam_ckpt)
    if policy_cfg.type != "fastwam":
        raise ValueError(f"Expected FastWAM, got policy type {policy_cfg.type!r}")
    policy_cfg.pretrained_path = args.fastwam_ckpt
    policy_cfg.device = str(device)
    policy_cfg.freeze_video_dit = True
    policy_cfg.loss_lambda_video = 0.0
    policy_cfg.action_dit_use_gradient_checkpointing = True
    policy_cfg.mot_checkpoint_mixed_attn = True
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg).eval()
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


def _eval_actor(
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

    policy.attach_planner(None)
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


def self_improvement_loop(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("FastWAM DAWR requires a CUDA GPU.")
    if args.n_iterations <= 0 or args.n_episodes <= 0:
        raise ValueError("n_iterations and n_episodes must be positive.")
    if args.actor_steps < 0 or args.value_steps <= 0:
        raise ValueError("actor_steps must be non-negative and value_steps must be positive.")
    if args.actor_batch_size <= 0 or args.value_batch_size <= 0:
        raise ValueError("Actor and value batch sizes must be positive.")
    if args.episode_length <= 0:
        raise ValueError("episode_length must be positive.")
    if not 0.0 <= args.gamma <= 1.0 or not 0.0 <= args.td_lambda <= 1.0:
        raise ValueError("gamma and td_lambda must lie in [0, 1].")
    if args.critic_warmup_iterations < 0:
        raise ValueError("critic_warmup_iterations cannot be negative.")
    if args.start_iteration > 0 and not args.value_ckpt:
        raise ValueError("Resumed DAWR runs require --value_ckpt from the prior iteration.")

    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "loop_summary.jsonl"
    use_wandb = bool(args.wandb_project)
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or output_dir.name,
            config=vars(args),
            dir=str(output_dir),
        )

    policy, envs, preprocessor, postprocessor, env_preprocessor, env_postprocessor = (
        _setup_actor_and_env(args, device)
    )
    camera_keys = tuple(policy.config.image_features)
    task_vocab = task_descriptions_from_envs(envs)
    if not task_vocab:
        raise ValueError("LIBERO environments did not expose task descriptions for the value critic.")
    task_to_index = {task: index for index, task in enumerate(task_vocab)}
    value_critic = make_value_critic(
        args,
        task_vocab=task_vocab,
        n_cameras=len(camera_keys),
        state_dim=int(policy.config.state_dim),
    )
    start_seed = args.seed + args.start_iteration * args.n_episodes
    actor_global_step = args.start_iteration * args.actor_steps
    value_global_step = args.start_iteration * args.value_steps

    for offset in range(args.n_iterations):
        iteration = args.start_iteration + offset
        iter_dir = output_dir / f"iter_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        log.info("%s\nTraditional DAWR iteration %d\n%s", "=" * 60, iteration, "=" * 60)
        policy.to(device)
        policy.eval()
        policy.attach_planner(None)

        writer = DAWRDecisionWriter(
            iter_dir / "dawr_actor_episodes",
            camera_keys=camera_keys,
            image_size=tuple(policy.config.image_size),
            action_dim=int(policy.config.action_dim),
            state_dim=int(policy.config.state_dim),
            horizon=int(policy.config.chunk_size),
        )
        collection = collect_dawr_episodes(
            policy,
            envs,
            env_preprocessor,
            env_postprocessor,
            preprocessor,
            postprocessor,
            n_episodes=args.n_episodes,
            writer=writer,
            camera_keys=camera_keys,
            start_seed=start_seed,
            collection_task_id=args.collection_task_id,
            trace_rollouts=args.trace_rollouts,
        )
        start_seed += args.n_episodes
        online_dataset = load_growing_dawr_dataset(
            output_dir,
            camera_keys=camera_keys,
            buffer_size=args.buffer_size,
        )
        if online_dataset is None:
            raise RuntimeError("Collection finished without creating DAWR replay.")

        policy.to("cpu")
        torch.cuda.empty_cache()
        vision_encoder = FrozenDINOEncoder(
            args.value_encoder_model,
            device=device,
            dtype=torch.bfloat16,
        )
        encoded_replay = encode_value_replay(
            online_dataset,
            vision_encoder,
            camera_keys=camera_keys,
            task_to_index=task_to_index,
            batch_size=args.value_encode_batch_size,
            num_workers=args.num_workers,
        )
        del vision_encoder
        torch.cuda.empty_cache()

        value_critic.to(device)
        initial_values = predict_replay_values(
            value_critic,
            encoded_replay,
            device=device,
            batch_size=args.value_batch_size,
        )
        _, value_targets = td_lambda_returns(
            encoded_replay.reward,
            encoded_replay.terminal,
            encoded_replay.trajectory_index,
            initial_values,
            gamma=args.gamma,
            td_lambda=args.td_lambda,
        )
        # Value and actor updates share one monotonically increasing W&B step
        # axis. Without this handoff, the actor's smaller independent counter
        # can move backwards after the value phase and W&B drops its metrics.
        value_global_step = max(value_global_step, actor_global_step)
        value_loss, value_global_step = finetune_value_critic(
            value_critic,
            encoded_replay,
            value_targets,
            device=device,
            steps=args.value_steps,
            batch_size=args.value_batch_size,
            lr=args.value_lr,
            weight_decay=args.value_weight_decay,
            grad_clip_norm=args.value_grad_clip_norm,
            global_step=value_global_step,
            use_wandb=use_wandb,
        )
        actor_warmup = iteration < args.critic_warmup_iterations
        actor_weights, advantage_stats = compute_actor_weights(
            value_critic,
            encoded_replay,
            device=device,
            value_batch_size=args.value_batch_size,
            gamma=args.gamma,
            td_lambda=args.td_lambda,
            beta=args.beta,
            max_weight=args.max_adv_weight,
            min_advantage_std=(
                0.0 if actor_warmup or args.actor_steps == 0 else args.min_advantage_std
            ),
            normalize_weight_mean=args.normalize_weight_mean,
        )
        value_checkpoint = save_value_checkpoint(
            value_critic.to("cpu"),
            iter_dir / "value_checkpoint.pt",
            task_vocab=task_vocab,
            encoder_model=args.value_encoder_model,
            iteration=iteration,
        )
        torch.cuda.empty_cache()

        actor_loss = None
        fastwam_checkpoint = None
        if actor_warmup:
            log.info(
                "Skipping actor update during value warmup (%d/%d).",
                iteration + 1,
                args.critic_warmup_iterations,
            )
        elif args.actor_steps > 0:
            policy.to(device)
            actor_global_step = max(actor_global_step, value_global_step)
            actor_loss, actor_global_step = finetune_actor(
                policy,
                preprocessor,
                online_dataset,
                actor_weights,
                steps=args.actor_steps,
                lr=args.actor_lr,
                batch_size=args.actor_batch_size,
                grad_clip_norm=args.actor_grad_clip_norm,
                num_workers=args.num_workers,
                global_step=actor_global_step,
                use_wandb=use_wandb,
            )
            fastwam_checkpoint = save_fastwam_checkpoint(
                policy,
                preprocessor,
                postprocessor,
                iter_dir / "fastwam_checkpoint",
            )

        eval_success = None
        if args.eval_n_episodes > 0:
            policy.to(device)
            eval_success = _eval_actor(
                envs,
                policy,
                env_preprocessor,
                env_postprocessor,
                preprocessor,
                postprocessor,
                n_episodes=args.eval_n_episodes,
                start_seed=args.seed + 100_000 + iteration * 1000,
                videos_dir=iter_dir / "eval_videos",
            )
            log.info("Direct actor eval: %.1f%%", eval_success)

        metrics: dict[str, Any] = {
            "algorithm": "traditional_dawr",
            "uses_q_function": False,
            "uses_planning": False,
            "reward": "verified_binary_success",
            "iteration": iteration,
            **collection,
            "n_online_decisions": len(online_dataset),
            "value_loss": value_loss,
            "actor_warmup": actor_warmup,
            "actor_weighted_loss": actor_loss,
            "actor_eval_pc_success": eval_success,
            "value_checkpoint": str(value_checkpoint),
            "fastwam_checkpoint": (
                str(fastwam_checkpoint) if fastwam_checkpoint is not None else args.fastwam_ckpt
            ),
            "advantage": advantage_stats,
            "elapsed_s": time.time() - started,
        }
        (iter_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        with summary_path.open("a") as summary_file:
            summary_file.write(json.dumps(metrics) + "\n")
        if use_wandb:
            import wandb

            wandb_metrics = {
                "collect/pc_success": collection["pc_success"],
                "collect/mean_episode_return": collection["mean_episode_return"],
                "replay/decisions": len(online_dataset),
                "value/mean_loss": value_loss,
                **{f"advantage/{key}": value for key, value in advantage_stats.items()},
            }
            if actor_loss is not None:
                wandb_metrics["actor/mean_weighted_loss"] = actor_loss
            if eval_success is not None:
                wandb_metrics["eval/actor_pc_success"] = eval_success
            wandb.log(wandb_metrics, step=max(actor_global_step, value_global_step))

        log.info(
            "Iteration %d done: collect=%.1f%% eval=%s value=%.5f actor=%s elapsed=%.1fs",
            iteration,
            collection["pc_success"],
            eval_success,
            value_loss,
            actor_loss,
            metrics["elapsed_s"],
        )

    # Destroy MuJoCo / EGL contexts while the EGL display is still active.
    # Leaving this to interpreter shutdown lets EGL's atexit handler terminate
    # the display before Robosuite context destructors run, producing a wall of
    # misleading EGL_NOT_INITIALIZED tracebacks after a successful job.
    close_envs(envs)
    if use_wandb:
        import wandb

        wandb.finish()
    log.info("Traditional DAWR complete: %s", summary_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Traditional value-based DAWR for FastWAM")
    parser.add_argument("--fastwam_ckpt", required=True)
    parser.add_argument("--value_ckpt", default="")
    parser.add_argument("--task", default="libero_10")
    parser.add_argument(
        "--episode_length",
        type=int,
        default=520,
        help="Maximum LIBERO environment steps per episode.",
    )
    parser.add_argument("--n_iterations", type=int, default=5)
    parser.add_argument("--start_iteration", type=int, default=0)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--buffer_size", type=int, default=3000)
    parser.add_argument(
        "--collection_task_id",
        type=int,
        default=None,
        help="Diagnostic-only filter that collects from one LIBERO task id.",
    )
    parser.add_argument(
        "--trace_rollouts",
        action="store_true",
        help="Synchronize CUDA and log every rollout/reset/inference/environment boundary.",
    )

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--td_lambda", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--max_adv_weight", type=float, default=100.0)
    parser.add_argument("--min_advantage_std", type=float, default=1e-3)
    parser.add_argument(
        "--normalize_weight_mean",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional stability ablation; canonical DAWR uses unnormalized clipped weights.",
    )

    parser.add_argument("--value_encoder_model", default="facebook/dinov2-large")
    parser.add_argument("--value_hidden_dim", type=int, default=512)
    parser.add_argument("--value_dropout", type=float, default=0.1)
    parser.add_argument("--value_steps", type=int, default=500)
    parser.add_argument("--value_lr", type=float, default=1e-3)
    parser.add_argument("--value_weight_decay", type=float, default=1e-4)
    parser.add_argument("--value_batch_size", type=int, default=256)
    parser.add_argument("--value_encode_batch_size", type=int, default=16)
    parser.add_argument("--value_grad_clip_norm", type=float, default=10.0)
    parser.add_argument("--critic_warmup_iterations", type=int, default=2)

    parser.add_argument("--actor_steps", type=int, default=200)
    parser.add_argument("--actor_lr", type=float, default=1e-5)
    parser.add_argument("--actor_batch_size", type=int, default=1)
    parser.add_argument("--actor_grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--eval_n_episodes", type=int, default=20)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_run_name", default="")
    return parser.parse_args()


if __name__ == "__main__":
    _enable_fatal_tracing()
    self_improvement_loop(parse_args())
