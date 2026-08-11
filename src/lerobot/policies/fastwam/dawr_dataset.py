"""Trajectory replay and loss-weight utilities for traditional FastWAM DAWR."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE


DAWR_DATASET_REPO_ID = "online_fastwam_dawr_decisions"
DAWR_DATASET_FPS = 10.0
DAWR_REWARD = "dawr_reward"
DAWR_TERMINAL = "dawr_terminal"
DAWR_TRAJECTORY_INDEX = "dawr_trajectory_index"
DAWR_WEIGHT = "dawr_weight"


def fastwam_action_to_libero(action: torch.Tensor) -> torch.Tensor:
    """Map raw FastWAM actions to the exact gripper convention LIBERO executes."""
    converted = action.clone()
    converted[..., -1] = torch.sign(1.0 - 2.0 * converted[..., -1])
    return converted


def libero_action_to_fastwam(action: torch.Tensor) -> torch.Tensor:
    """Map executed LIBERO actions to FastWAM's close=0/open=1 convention."""
    converted = action.clone()
    converted[..., -1] = (1.0 - converted[..., -1]) / 2.0
    return converted


def _resize_image(image: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    if tuple(image.shape[-2:]) == tuple(image_size):
        return image
    return resize(
        image,
        size=list(image_size),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


def _features(
    camera_keys: tuple[str, ...],
    image_size: tuple[int, int],
    action_dim: int,
    state_dim: int,
    horizon: int,
) -> dict[str, dict[str, Any]]:
    height, width = image_size
    features: dict[str, dict[str, Any]] = {
        key: {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }
        for key in camera_keys
    }
    features.update(
        {
            ACTION: {"dtype": "float32", "shape": (horizon, action_dim), "names": None},
            f"{ACTION}_is_pad": {"dtype": "bool", "shape": (horizon,), "names": None},
            OBS_STATE: {"dtype": "float32", "shape": (state_dim,), "names": None},
            DAWR_REWARD: {"dtype": "float32", "shape": (1,), "names": None},
            DAWR_TERMINAL: {"dtype": "bool", "shape": (1,), "names": None},
            "episode_success": {"dtype": "bool", "shape": (1,), "names": None},
        }
    )
    return features


class DAWRDecisionWriter:
    """Write one replay frame per multi-environment-step policy decision."""

    def __init__(
        self,
        root: Path,
        *,
        camera_keys: tuple[str, ...],
        image_size: tuple[int, int],
        action_dim: int,
        state_dim: int,
        horizon: int,
        fps: float = DAWR_DATASET_FPS,
    ) -> None:
        self.root = Path(root)
        self.camera_keys = camera_keys
        self.image_size = tuple(image_size)
        self.horizon = int(horizon)
        self.dataset: LeRobotDataset | None = None
        self.num_episodes = 0
        self.num_decisions = 0
        self._feature_spec = _features(
            camera_keys,
            self.image_size,
            action_dim,
            state_dim,
            self.horizon,
        )
        self.fps = float(fps)

    def _ensure_dataset(self) -> LeRobotDataset:
        if self.dataset is None:
            if self.root.exists():
                raise FileExistsError(f"Refusing to overwrite DAWR replay at {self.root}")
            self.root.parent.mkdir(parents=True, exist_ok=True)
            self.dataset = LeRobotDataset.create(
                repo_id=DAWR_DATASET_REPO_ID,
                fps=self.fps,
                features=self._feature_spec,
                root=self.root,
                use_videos=False,
            )
        return self.dataset

    def add_episode(
        self,
        decisions: list[dict[str, Any]],
        *,
        task: str,
        success: bool,
    ) -> None:
        if not decisions:
            raise ValueError("A DAWR episode must contain at least one policy decision.")
        ds = self._ensure_dataset()

        for decision_index, decision in enumerate(decisions):
            action = decision[ACTION].detach().cpu().float()
            action_is_pad = decision[f"{ACTION}_is_pad"].detach().cpu().bool()
            if tuple(action.shape[:1]) != (self.horizon,):
                raise ValueError(
                    f"Expected a horizon-{self.horizon} action target, got {tuple(action.shape)}"
                )
            if tuple(action_is_pad.shape) != (self.horizon,):
                raise ValueError(
                    f"Expected a horizon-{self.horizon} action mask, got {tuple(action_is_pad.shape)}"
                )
            is_last = decision_index == len(decisions) - 1
            if bool(decision[DAWR_TERMINAL]) != is_last:
                raise ValueError("Exactly the final decision in each DAWR episode must be terminal.")

            frame: dict[str, Any] = {
                "task": task,
                ACTION: action.numpy(),
                f"{ACTION}_is_pad": action_is_pad.numpy(),
                OBS_STATE: decision[OBS_STATE].detach().cpu().float().numpy(),
                DAWR_REWARD: torch.tensor(
                    [float(decision[DAWR_REWARD])], dtype=torch.float32
                ).numpy(),
                DAWR_TERMINAL: torch.tensor([is_last], dtype=torch.bool).numpy(),
                "episode_success": torch.tensor([success], dtype=torch.bool).numpy(),
            }
            for key in self.camera_keys:
                if key not in decision:
                    raise KeyError(f"DAWR decision is missing camera {key!r}")
                image = _resize_image(
                    decision[key].detach().cpu().float(), self.image_size
                )
                frame[key] = (image.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            ds.add_frame(frame)

        ds.save_episode()
        self.num_episodes += 1
        self.num_decisions += len(decisions)

    def finalize(self) -> None:
        if self.dataset is not None:
            self.dataset.finalize()


class DAWRDecisionDataset(Dataset):
    """Expose decision trajectories for the value critic and actor."""

    def __init__(
        self,
        dataset: LeRobotDataset,
        camera_keys: tuple[str, ...],
        *,
        trajectory_offset: int,
    ) -> None:
        self.dataset = dataset
        self.camera_keys = camera_keys
        self.trajectory_offset = int(trajectory_offset)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        required = (
            *self.camera_keys,
            OBS_STATE,
            ACTION,
            f"{ACTION}_is_pad",
            DAWR_REWARD,
            DAWR_TERMINAL,
            "episode_index",
        )
        missing = [key for key in required if key not in item]
        if missing:
            raise KeyError(f"DAWR replay sample is missing keys: {missing}")
        out: dict[str, Any] = {
            ACTION: item[ACTION].float(),
            f"{ACTION}_is_pad": item[f"{ACTION}_is_pad"].bool(),
            OBS_STATE: item[OBS_STATE].float(),
            DAWR_REWARD: item[DAWR_REWARD].float(),
            DAWR_TERMINAL: item[DAWR_TERMINAL].bool(),
            DAWR_TRAJECTORY_INDEX: torch.tensor(
                self.trajectory_offset + int(item["episode_index"]), dtype=torch.long
            ),
            "task": item.get("task", ""),
        }
        for key in self.camera_keys:
            out[key] = item[key].float()
        return out


class LossWeightedDataset(Dataset):
    """Attach frozen DAWR weights without modifying the on-disk trajectory replay."""

    def __init__(self, dataset: Dataset, weights: torch.Tensor) -> None:
        self.dataset = dataset
        if weights.ndim != 1 or len(weights) != len(dataset):
            raise ValueError(f"Expected {len(dataset)} scalar weights, got {tuple(weights.shape)}")
        self.weights = weights.detach().cpu().float()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = dict(self.dataset[idx])
        item.pop(DAWR_REWARD, None)
        item.pop(DAWR_TERMINAL, None)
        item.pop(DAWR_TRAJECTORY_INDEX, None)
        item[DAWR_WEIGHT] = self.weights[idx]
        return item


def load_growing_dawr_dataset(
    output_dir: Path,
    *,
    camera_keys: tuple[str, ...],
    buffer_size: int | None = None,
) -> Dataset | None:
    """Load chronological online decisions, optionally retaining a FIFO suffix."""
    datasets: list[Dataset] = []
    trajectory_offset = 0
    for root in sorted(Path(output_dir).glob("iter_*/dawr_actor_episodes")):
        dataset = LeRobotDataset(
            repo_id=DAWR_DATASET_REPO_ID,
            root=root,
            force_cache_sync=False,
        )
        datasets.append(
            DAWRDecisionDataset(
                dataset,
                camera_keys,
                trajectory_offset=trajectory_offset,
            )
        )
        trajectory_offset += len(dataset.meta.episodes)
    if not datasets:
        return None

    combined: Dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    if buffer_size is not None and buffer_size > 0 and len(combined) > buffer_size:
        first = len(combined) - buffer_size
        combined = Subset(combined, range(first, len(combined)))
    return combined


def exponential_advantage_weights(
    advantages: torch.Tensor,
    *,
    beta: float = 10.0,
    max_weight: float = 100.0,
    standardize: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute clipped exponential weights from frozen replay advantages."""
    advantages = advantages.detach().cpu().float().flatten()
    if len(advantages) == 0:
        raise ValueError("Cannot compute DAWR weights for an empty replay buffer.")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError(f"beta must be finite and non-negative, got {beta}")
    if not math.isfinite(max_weight) or max_weight < 1:
        raise ValueError(f"max_weight must be finite and at least 1, got {max_weight}")
    if not torch.isfinite(advantages).all():
        raise ValueError("Advantages contain NaN or infinite values.")

    advantage_mean = advantages.mean()
    advantage_std = advantages.std(unbiased=False)
    if standardize and len(advantages) > 1 and float(advantage_std) > 1e-6:
        scaled_advantages = (advantages - advantage_mean) / advantage_std
    elif standardize:
        scaled_advantages = torch.zeros_like(advantages)
    else:
        scaled_advantages = advantages

    log_weights = (beta * scaled_advantages).clamp(
        min=-50.0,
        max=math.log(max_weight),
    )
    weights = torch.exp(log_weights).clamp(max=max_weight)
    weight_sum = weights.sum()
    ess = weight_sum.square() / weights.square().sum().clamp(min=1e-12)
    stats = {
        "advantage_mean": float(advantage_mean),
        "advantage_std": float(advantage_std),
        "standardized_advantage_mean": float(scaled_advantages.mean()),
        "standardized_advantage_std": float(scaled_advantages.std(unbiased=False)),
        "weight_mean": float(weights.mean()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_cap_fraction": float(
            (weights >= max_weight * (1.0 - 1e-6)).float().mean()
        ),
        "effective_sample_size": float(ess),
        "effective_sample_fraction": float(ess / len(weights)),
    }
    return weights, stats
