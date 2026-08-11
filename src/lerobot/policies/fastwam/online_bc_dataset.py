"""Successful-rollout datasets for FastWAM behavior-policy self-improvement.

The Q self-improvement loop stores data in the Q-function's convention: 256 px
images, no proprioception, and environment-space actions.  Those samples cannot
be fed directly to FastWAM.  This module owns the BC-specific on-policy format
and adapts both it and original demonstrations to FastWAM's training contract.
LIBERO needs a gripper-convention conversion; RoboTwin's 14-D qpos actions are
already in the convention used by its FastWAM checkpoint and stay unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import ACTION, OBS_STATE


ONLINE_BC_DATASET_REPO_ID = "online_successful_bc_episodes"
ONLINE_BC_DATASET_FPS = 10.0
# Preserve the released RoboTwin LeRobot dataset's frame-index and video
# metadata. This is not a physical control frequency: one deployed qpos action
# is executed by a variable-length TOPP trajectory at the simulator's 250 Hz.
ROBOTWIN_ONLINE_BC_DATASET_FPS = 50.0
ROBOTWIN_SOURCE_CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def libero_gripper_to_fastwam(action: torch.Tensor) -> torch.Tensor:
    """Convert LIBERO gripper actions (open=-1, close=+1) to FastWAM (open=1, close=0)."""
    converted = action.clone()
    converted[..., -1] = (1.0 - converted[..., -1]) / 2.0
    return converted


def _resize_image(image: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """Apply the official FastWAM per-camera resize used for LIBERO training."""
    if tuple(image.shape[-2:]) == tuple(image_size):
        return image
    return resize(
        image,
        size=list(image_size),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


class FastWAMBCDataset(Dataset):
    """Expose a LeRobot dataset in the exact format consumed by ``FastWAMPolicy.forward``.

    Args:
        dataset: A LeRobot dataset configured with FastWAM action deltas.
        camera_keys: Camera keys expected by the FastWAM checkpoint.
        image_size: Per-camera FastWAM input size.
        libero_action_convention: Convert the dataset's gripper from LIBERO
            ``[-1, 1]`` to FastWAM ``[0, 1]``.  Original demonstrations need
            this conversion; newly collected BC data is already in FastWAM
            convention because it is captured before the env postprocessor.
        validate_gripper_targets: Check the final action dimension is in
            FastWAM's binary-gripper ``[0, 1]`` range. Disable this for
            RoboTwin, whose actions are two-arm 14-D qpos targets and whose
            grippers are dimensions 6 and 13 rather than a single final
            LIBERO gripper channel.
    """

    def __init__(
        self,
        dataset: Dataset,
        camera_keys: tuple[str, ...],
        image_size: tuple[int, int],
        *,
        libero_action_convention: bool,
        validate_gripper_targets: bool = True,
        robotwin_source_camera_keys: tuple[str, str, str] | None = None,
    ) -> None:
        self.dataset = dataset
        self.camera_keys = camera_keys
        self.image_size = tuple(image_size)
        self.libero_action_convention = libero_action_convention
        self.validate_gripper_targets = validate_gripper_targets
        self.robotwin_source_camera_keys = robotwin_source_camera_keys
        if self.robotwin_source_camera_keys is not None and self.camera_keys != (
            "observation.images.image",
        ):
            raise ValueError(
                "RoboTwin camera concatenation requires the FastWAM checkpoint to expect only "
                "'observation.images.image'."
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        required_cameras = self.robotwin_source_camera_keys or self.camera_keys
        missing = [key for key in (*required_cameras, OBS_STATE, ACTION) if key not in item]
        if missing:
            raise KeyError(f"FastWAM BC sample is missing required keys: {missing}")

        action = item[ACTION].float()
        if self.libero_action_convention:
            action = libero_gripper_to_fastwam(action)
        if self.validate_gripper_targets:
            gripper = action[..., -1]
            if torch.any(gripper < -1e-4) or torch.any(gripper > 1.0001):
                raise ValueError(
                    "FastWAM BC gripper targets must be in [0, 1]; "
                    f"observed range [{gripper.min().item():.4f}, {gripper.max().item():.4f}]"
                )

        out: dict[str, Any] = {
            ACTION: action,
            OBS_STATE: item[OBS_STATE].float(),
            "task": item.get("task", ""),
        }
        if self.robotwin_source_camera_keys is not None:
            from lerobot.envs.robotwin import build_robotwin_image

            head_key, left_key, right_key = self.robotwin_source_camera_keys
            image = build_robotwin_image(
                item[head_key].float(),
                item[left_key].float(),
                item[right_key].float(),
            )
            if tuple(image.shape[-2:]) != self.image_size:
                image = _resize_image(image, self.image_size)
            out[self.camera_keys[0]] = image
        else:
            for key in self.camera_keys:
                out[key] = _resize_image(item[key].float(), self.image_size)

        action_pad_key = f"{ACTION}_is_pad"
        if action_pad_key in item:
            out[action_pad_key] = item[action_pad_key].bool()
        else:
            out[action_pad_key] = torch.zeros(action.shape[0], dtype=torch.bool)
        return out


def _online_features(
    camera_keys: tuple[str, ...],
    image_size: tuple[int, int],
    action_dim: int,
    state_dim: int,
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
    features[ACTION] = {"dtype": "float32", "shape": (action_dim,), "names": None}
    features[OBS_STATE] = {"dtype": "float32", "shape": (state_dim,), "names": None}
    features["episode_success"] = {"dtype": "bool", "shape": (1,), "names": None}
    return features


class SuccessfulEpisodeWriter:
    """Stream successful episodes to disk without retaining the full collection in RAM."""

    def __init__(
        self,
        root: Path,
        camera_keys: tuple[str, ...],
        image_size: tuple[int, int],
        action_dim: int,
        state_dim: int,
        fps: float = ONLINE_BC_DATASET_FPS,
    ) -> None:
        self.root = Path(root)
        self.camera_keys = camera_keys
        self.image_size = tuple(image_size)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.fps = float(fps)
        self.dataset: LeRobotDataset | None = None
        self.num_episodes = 0
        self.num_frames = 0

    def _ensure_dataset(self) -> LeRobotDataset:
        if self.dataset is None:
            if self.root.exists():
                raise FileExistsError(f"Refusing to overwrite existing online BC dataset: {self.root}")
            self.root.parent.mkdir(parents=True, exist_ok=True)
            self.dataset = LeRobotDataset.create(
                repo_id=ONLINE_BC_DATASET_REPO_ID,
                fps=self.fps,
                features=_online_features(
                    self.camera_keys,
                    self.image_size,
                    self.action_dim,
                    self.state_dim,
                ),
                root=self.root,
                use_videos=False,
            )
        return self.dataset

    def add_episode(self, episode: dict[str, Any]) -> None:
        if not bool(episode.get("success", False)):
            raise ValueError("SuccessfulEpisodeWriter only accepts successful episodes.")
        ds = self._ensure_dataset()
        action = episode[ACTION]
        state = episode[OBS_STATE]
        n_frames = int(action.shape[0])
        if state.shape[0] != n_frames:
            raise ValueError(f"State/action length mismatch: {state.shape[0]} vs {n_frames}")

        for key in self.camera_keys:
            if key not in episode or episode[key].shape[0] != n_frames:
                raise ValueError(f"Missing or misaligned camera sequence {key!r}")

        task = str(episode.get("task", ""))
        for frame_idx in range(n_frames):
            frame: dict[str, Any] = {
                "task": task,
                ACTION: action[frame_idx].detach().cpu().numpy().astype("float32"),
                OBS_STATE: state[frame_idx].detach().cpu().numpy().astype("float32"),
                "episode_success": torch.tensor([True], dtype=torch.bool).numpy(),
            }
            for key in self.camera_keys:
                image = _resize_image(episode[key][frame_idx].detach().cpu(), self.image_size)
                frame[key] = (image.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            ds.add_frame(frame)
        ds.save_episode()
        self.num_episodes += 1
        self.num_frames += n_frames

    def finalize(self) -> None:
        if self.dataset is not None:
            self.dataset.finalize()


def _make_lerobot_dataset(
    repo_id: str,
    root: str | Path,
    policy_config,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    delta_timestamps = resolve_delta_timestamps(policy_config, metadata)
    return LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        force_cache_sync=False,
    )


def _episode_ids_for_tasks(metadata: LeRobotDatasetMetadata, task_descriptions: Iterable[str]) -> list[int]:
    wanted = {str(task).strip().lower() for task in task_descriptions}
    selected: list[int] = []
    for episode in metadata.episodes:
        episode_tasks = episode.get("tasks", [])
        if isinstance(episode_tasks, str):
            episode_tasks = [episode_tasks]
        if any(str(task).strip().lower() in wanted for task in episode_tasks):
            selected.append(int(episode["episode_index"]))
    return selected


def _frame_ids_for_episodes(
    metadata: LeRobotDatasetMetadata,
    episode_ids: Iterable[int],
) -> list[int]:
    """Return global frame indices without materializing an episode-filtered Arrow table."""
    wanted = {int(episode_id) for episode_id in episode_ids}
    frame_ids: list[int] = []
    found: set[int] = set()
    for episode in metadata.episodes:
        episode_id = int(episode["episode_index"])
        if episode_id not in wanted:
            continue
        found.add(episode_id)
        frame_ids.extend(
            range(
                int(episode["dataset_from_index"]),
                int(episode["dataset_to_index"]),
            )
        )
    missing = wanted - found
    if missing:
        raise ValueError(f"Episode metadata is missing requested episode ids: {sorted(missing)}")
    return frame_ids


def load_original_success_dataset(
    repo_id: str,
    root: str | Path,
    policy_config,
    task_descriptions: Iterable[str] | None = None,
    *,
    libero_action_convention: bool = True,
    validate_gripper_targets: bool = True,
) -> Dataset:
    """Load successful demonstrations, optionally restricted to the collection tasks.

    The full dataset is opened through Hugging Face's memory-mapped Arrow cache.
    Task restriction is applied as a lightweight frame-index ``Subset``.  Passing
    hundreds of episode ids to ``LeRobotDataset`` instead asks PyArrow to build a
    new filtered table containing all selected inline images, which causes a
    large transient host-memory spike for the 33 GB LIBERO dataset.
    """
    metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    policy_camera_keys = tuple(policy_config.image_features)
    metadata_keys = set(metadata.features)
    robotwin_source_camera_keys = None
    if not set(policy_camera_keys).issubset(metadata_keys):
        if policy_camera_keys == ("observation.images.image",) and set(
            ROBOTWIN_SOURCE_CAMERA_KEYS
        ).issubset(metadata_keys):
            robotwin_source_camera_keys = ROBOTWIN_SOURCE_CAMERA_KEYS
        else:
            raise KeyError(
                f"Dataset {repo_id!r} does not provide the policy cameras {policy_camera_keys}; "
                f"available cameras are {tuple(metadata.camera_keys)}"
            )
    selected_frames = None
    if task_descriptions is not None:
        episodes = _episode_ids_for_tasks(metadata, task_descriptions)
        if not episodes:
            raise ValueError(f"No episodes in {repo_id!r} match the collection task descriptions.")
        selected_frames = _frame_ids_for_episodes(metadata, episodes)

    # Deliberately load the full, cached Arrow dataset.  This stays memory mapped
    # and also keeps LeRobotDataset's absolute delta-timestamp indexing intact.
    dataset = _make_lerobot_dataset(repo_id, root, policy_config, episodes=None)
    wrapped = FastWAMBCDataset(
        dataset,
        camera_keys=policy_camera_keys,
        image_size=tuple(policy_config.image_size),
        libero_action_convention=libero_action_convention,
        validate_gripper_targets=validate_gripper_targets,
        robotwin_source_camera_keys=robotwin_source_camera_keys,
    )
    if selected_frames is None:
        return wrapped
    return Subset(wrapped, selected_frames)


def load_growing_online_success_dataset(
    output_dir: Path,
    policy_config,
    *,
    validate_gripper_targets: bool = True,
) -> Dataset | None:
    """Load every completed ``iter_*/successful_episodes`` dataset as one growing buffer."""
    datasets: list[Dataset] = []
    for root in sorted(Path(output_dir).glob("iter_*/successful_episodes")):
        dataset = _make_lerobot_dataset(
            ONLINE_BC_DATASET_REPO_ID,
            root,
            policy_config,
        )
        datasets.append(
            FastWAMBCDataset(
                dataset,
                camera_keys=tuple(policy_config.image_features),
                image_size=tuple(policy_config.image_size),
                libero_action_convention=False,
                validate_gripper_targets=validate_gripper_targets,
            )
        )
    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)
