"""Paper-style demonstration replay prefill for chunked FastWAM IBRL.

The released IBRL recipe seeds replay with the same demonstrations used to
train the frozen behavior policy.  This module adapts the local
``HuggingFaceVLA/libero`` demonstrations to the H-step SMDP replay used by the
FastWAM TD3 baseline without changing the online training loop.

Two action conventions meet here and must remain distinct:

* The original dataset stores actions in executed LIBERO space, including a
  binary gripper in ``{-1, +1}``.  Those expert actions are inserted directly
  into TD3 replay.
* Frozen FastWAM predicts in its training convention, including a gripper in
  ``[0, 1]``.  Cached next-state BC proposals therefore pass through both the
  policy postprocessor and the LIBERO environment postprocessor.

Each demonstration is split into non-overlapping H-step chunks.  A short
terminal chunk repeats its final expert action only to fill the fixed critic
input; the repeated suffix is never executed.  Sparse success reward is placed
on the final real action, so the terminal macro reward is ``gamma ** (k - 1)``
for a k-step tail and its SMDP discount is ``gamma ** k``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.fastwam.ibrl_td3 import EncodedReplayBuffer
from lerobot.utils.constants import ACTION, OBS_STATE


@dataclass(frozen=True)
class DemoMacroTransition:
    """One fixed-width expert macro-action and its exact SMDP labels."""

    episode_index: int
    start_frame: int
    executed_steps: int
    action: Tensor
    reward: float
    discount: float
    done: bool
    next_frame: int | None


@dataclass(frozen=True)
class DemoPrefillStats:
    """Auditable description of the demonstration replay inserted in memory."""

    episode_indices: tuple[int, ...]
    transitions: int
    environment_steps: int
    terminal_transitions: int
    padded_action_steps: int


def _episode_task_names(episode: dict[str, Any]) -> tuple[str, ...]:
    tasks = episode.get("tasks", ())
    if isinstance(tasks, str):
        return (tasks,)
    return tuple(str(task) for task in tasks)


def select_demo_episode_indices_from_metadata(
    episodes: Iterable[dict[str, Any]],
    *,
    task_description: str,
    n_episodes: int,
) -> tuple[int, ...]:
    """Select the lowest-index demos for one exact task, without cherry-picking."""
    if not task_description:
        raise ValueError("task_description must be non-empty.")
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive.")

    matches = sorted(
        int(episode["episode_index"])
        for episode in episodes
        if task_description in _episode_task_names(episode)
    )
    if len(matches) < n_episodes:
        raise ValueError(
            f"Dataset contains {len(matches)} demonstrations for {task_description!r}; "
            f"requested {n_episodes}."
        )
    return tuple(matches[:n_episodes])


def select_demo_episode_indices(
    dataset_root: str | Path,
    *,
    repo_id: str,
    task_description: str,
    n_episodes: int,
) -> tuple[int, ...]:
    """Read only dataset metadata and deterministically select task demos."""
    metadata = LeRobotDatasetMetadata(
        repo_id=repo_id,
        root=Path(dataset_root),
        force_cache_sync=False,
    )
    return select_demo_episode_indices_from_metadata(
        (metadata.episodes[index] for index in range(len(metadata.episodes))),
        task_description=task_description,
        n_episodes=n_episodes,
    )


def build_demo_macro_transitions(
    actions: Tensor,
    *,
    episode_index: int,
    action_horizon: int,
    gamma: float,
    terminal_reward: float = 1.0,
) -> tuple[DemoMacroTransition, ...]:
    """Convert one successful frame-level demo into stride-H SMDP transitions.

    Dataset rows have the standard ``(observation_t, action_t)`` alignment and
    do not include a post-terminal observation.  Consequently, a nonterminal
    chunk beginning at ``t`` uses row ``t + H`` as its next observation, while
    the terminal chunk uses the replay buffer's zero sentinel.
    """
    actions = torch.as_tensor(actions).detach().cpu().float()
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] == 0:
        raise ValueError(
            "actions must be a non-empty tensor shaped (episode_steps, action_dim)."
        )
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive.")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1].")
    if terminal_reward < 0.0:
        raise ValueError("terminal_reward cannot be negative.")

    episode_steps = int(actions.shape[0])
    transitions: list[DemoMacroTransition] = []
    for start in range(0, episode_steps, action_horizon):
        stop = min(start + action_horizon, episode_steps)
        executed_steps = stop - start
        chunk = actions[start:stop]
        if executed_steps < action_horizon:
            # Match LeRobot's post-episode delta padding while keeping every
            # fixed-width action coordinate on the expert-action manifold.
            padding = chunk[-1:].expand(action_horizon - executed_steps, -1)
            chunk = torch.cat((chunk, padding), dim=0)

        done = stop == episode_steps
        reward = terminal_reward * gamma ** (executed_steps - 1) if done else 0.0
        transitions.append(
            DemoMacroTransition(
                episode_index=int(episode_index),
                start_frame=start,
                executed_steps=executed_steps,
                action=chunk.contiguous(),
                reward=float(reward),
                discount=float(gamma**executed_steps),
                done=done,
                next_frame=None if done else stop,
            )
        )
    return tuple(transitions)


def _absolute_to_relative_indices(dataset: LeRobotDataset) -> dict[int, int]:
    indices = dataset.hf_dataset.select_columns(["index"])["index"]
    return {
        int(index.item() if isinstance(index, Tensor) else index): relative
        for relative, index in enumerate(indices)
    }


def _episode_actions(
    dataset: LeRobotDataset,
    *,
    episode_index: int,
    absolute_to_relative: dict[int, int],
) -> Tensor:
    episode = dataset.meta.episodes[episode_index]
    absolute_indices = range(
        int(episode["dataset_from_index"]),
        int(episode["dataset_to_index"]),
    )
    relative_indices = [absolute_to_relative[index] for index in absolute_indices]
    action_rows = dataset.hf_dataset.select_columns([ACTION])[relative_indices][ACTION]
    return torch.stack(
        [row if isinstance(row, Tensor) else torch.as_tensor(row) for row in action_rows]
    ).float()


def _demo_observation_batch(
    item: dict[str, Any],
    *,
    camera_keys: tuple[str, ...],
    image_size: tuple[int, int],
    expected_task: str,
) -> dict[str, Any]:
    missing = [key for key in (*camera_keys, OBS_STATE, "task") if key not in item]
    if missing:
        raise KeyError(f"Demonstration observation is missing IBRL keys: {missing}")
    if item["task"] != expected_task:
        raise ValueError(
            f"Selected demonstration task {item['task']!r} does not match "
            f"environment task {expected_task!r}."
        )

    batch: dict[str, Any] = {
        OBS_STATE: item[OBS_STATE].detach().float().reshape(1, -1),
        "task": [expected_task],
    }
    for key in camera_keys:
        # HuggingFaceVLA/libero already stores the 180-degree-corrected camera
        # convention used for FastWAM training. Resize only; applying the online
        # LiberoProcessorStep here would flip each demonstration image twice.
        image = item[key].detach().float().unsqueeze(0)
        if tuple(image.shape[-2:]) != image_size:
            image = F.interpolate(
                image,
                size=image_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        batch[key] = image
    return batch


@torch.no_grad()
def _encode_demo_observation(
    batch: dict[str, Any],
    *,
    vision_encoder,
    preprocessor,
    camera_keys: tuple[str, ...],
    task_to_index: dict[str, int],
    device: torch.device,
) -> tuple[Tensor, dict[str, Any]]:
    images = torch.stack([batch[key] for key in camera_keys], dim=1)
    vision_features = vision_encoder(images).flatten(1)
    policy_batch = preprocessor(batch)
    normalized_state = policy_batch[OBS_STATE].to(device=device, dtype=torch.float32)
    task = batch["task"][0]
    if task not in task_to_index:
        raise KeyError(f"Unknown IBRL task description in demonstrations: {task!r}")
    task_one_hot = F.one_hot(
        torch.tensor([task_to_index[task]], device=device),
        num_classes=len(task_to_index),
    ).float()
    encoded = torch.cat((vision_features, normalized_state, task_one_hot), dim=-1)
    return encoded.detach(), policy_batch


@torch.no_grad()
def _frozen_bc_proposal(
    policy_batch: dict[str, Any],
    *,
    policy,
    postprocessor,
    env_postprocessor,
    action_horizon: int,
    device: torch.device,
) -> Tensor:
    normalized_chunk = policy.predict_action_chunk(policy_batch)
    if normalized_chunk.ndim != 3 or normalized_chunk.shape[0] != 1:
        raise ValueError(
            "FastWAM demo proposal must be shaped (1, chunk_size, action_dim); "
            f"got {tuple(normalized_chunk.shape)}."
        )
    if normalized_chunk.shape[1] < action_horizon:
        raise ValueError(
            f"FastWAM returned {normalized_chunk.shape[1]} actions, fewer than "
            f"action_horizon={action_horizon}."
        )
    raw_chunk = postprocessor(normalized_chunk[:, :action_horizon]).float()
    libero_chunk = env_postprocessor({ACTION: raw_chunk})[ACTION].float().clamp(-1.0, 1.0)
    return libero_chunk.reshape(1, -1).to(device=device)


@torch.no_grad()
def prefill_demo_replay(
    replay: EncodedReplayBuffer,
    *,
    dataset_root: str | Path,
    repo_id: str,
    episode_indices: tuple[int, ...],
    task_description: str,
    action_horizon: int,
    gamma: float,
    policy,
    vision_encoder,
    preprocessor,
    postprocessor,
    env_postprocessor,
    camera_keys: tuple[str, ...],
    task_to_index: dict[str, int],
    device: torch.device,
) -> DemoPrefillStats:
    """Encode selected demonstrations and append them to ``replay``.

    Frozen visual/state encodings are computed once per macro state.  The
    frozen FastWAM proposal at every nonterminal next state is also computed
    once and cached in the corresponding replay row, preserving the IBRL
    Bellman target without rerunning the large policy during TD3 updates.
    """
    if not episode_indices:
        raise ValueError("episode_indices must contain at least one demonstration.")
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError("episode_indices cannot contain duplicates.")
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive.")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1].")
    if not camera_keys:
        raise ValueError("camera_keys must contain at least one camera.")
    if task_description not in task_to_index:
        raise KeyError(f"Task {task_description!r} is absent from task_to_index.")

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=Path(dataset_root),
        episodes=list(episode_indices),
        force_cache_sync=False,
    )
    if len(replay) != 0:
        raise ValueError("Demonstration prefill requires an empty replay buffer.")
    required_transitions = 0
    for episode_index in episode_indices:
        if episode_index < 0 or episode_index >= len(dataset.meta.episodes):
            raise IndexError(f"Demonstration episode_index={episode_index} is out of range.")
        episode = dataset.meta.episodes[episode_index]
        if task_description not in _episode_task_names(episode):
            raise ValueError(
                f"Demonstration episode {episode_index} does not belong to "
                f"task {task_description!r}."
            )
        episode_length = int(episode["length"])
        required_transitions += (episode_length + action_horizon - 1) // action_horizon
    if required_transitions > replay.capacity:
        raise ValueError(
            f"Demo prefill needs {required_transitions} replay rows but capacity is "
            f"{replay.capacity}; refusing to silently evict demonstrations during prefill."
        )

    absolute_to_relative = _absolute_to_relative_indices(dataset)
    image_size = tuple(int(value) for value in policy.config.image_size)
    action_dim = int(policy.config.action_dim)

    total_transitions = 0
    total_steps = 0
    terminal_transitions = 0
    padded_action_steps = 0

    for episode_index in episode_indices:
        policy.reset()
        actions = _episode_actions(
            dataset,
            episode_index=episode_index,
            absolute_to_relative=absolute_to_relative,
        )
        if actions.shape[1] != action_dim:
            raise ValueError(
                f"Demo action_dim={actions.shape[1]} does not match FastWAM action_dim={action_dim}."
            )
        # The original corpus is in executed LIBERO space.  Preserve it exactly;
        # in particular, do not apply FastWAM's [0,1] gripper postprocessor.
        if not torch.all((actions[:, -1] == -1) | (actions[:, -1] == 1)):
            unique = torch.unique(actions[:, -1]).tolist()
            raise ValueError(
                "Original LIBERO demonstrations must have binary {-1,+1} gripper actions; "
                f"observed {unique}."
            )
        macros = build_demo_macro_transitions(
            actions,
            episode_index=episode_index,
            action_horizon=action_horizon,
            gamma=gamma,
        )
        episode = dataset.meta.episodes[episode_index]
        absolute_start = int(episode["dataset_from_index"])

        encoded_by_frame: dict[int, Tensor] = {}
        bc_by_frame: dict[int, Tensor] = {}
        for macro_index, macro in enumerate(macros):
            relative_index = absolute_to_relative[absolute_start + macro.start_frame]
            batch = _demo_observation_batch(
                dataset[relative_index],
                camera_keys=camera_keys,
                image_size=image_size,
                expected_task=task_description,
            )
            encoded, policy_batch = _encode_demo_observation(
                batch,
                vision_encoder=vision_encoder,
                preprocessor=preprocessor,
                camera_keys=camera_keys,
                task_to_index=task_to_index,
                device=device,
            )
            encoded_by_frame[macro.start_frame] = encoded
            # Only inbound next states need cached BC proposals.  The first
            # macro state's proposal is never referenced by a replay target.
            if macro_index > 0:
                bc_by_frame[macro.start_frame] = _frozen_bc_proposal(
                    policy_batch,
                    policy=policy,
                    postprocessor=postprocessor,
                    env_postprocessor=env_postprocessor,
                    action_horizon=action_horizon,
                    device=device,
                )

        for macro in macros:
            observation = encoded_by_frame[macro.start_frame]
            if macro.done:
                next_observation = torch.zeros_like(observation)
                next_bc_action = torch.zeros(
                    1,
                    action_horizon * action_dim,
                    device=device,
                    dtype=torch.float32,
                )
            else:
                if macro.next_frame is None:
                    raise RuntimeError("A nonterminal demo macro is missing its next frame.")
                next_observation = encoded_by_frame[macro.next_frame]
                next_bc_action = bc_by_frame[macro.next_frame]

            replay.add(
                observation,
                macro.action.reshape(1, -1).to(device=device).clamp(-1.0, 1.0),
                macro.reward,
                next_observation,
                macro.done,
                macro.discount,
                next_bc_action,
            )
            total_transitions += 1
            total_steps += macro.executed_steps
            terminal_transitions += int(macro.done)
            padded_action_steps += action_horizon - macro.executed_steps

    return DemoPrefillStats(
        episode_indices=tuple(int(index) for index in episode_indices),
        transitions=total_transitions,
        environment_steps=total_steps,
        terminal_transitions=terminal_transitions,
        padded_action_steps=padded_action_steps,
    )
