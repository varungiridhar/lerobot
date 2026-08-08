"""OnlineQDataset: growing-buffer online dataset for Q-function self-improvement.

Episodes are saved in LeRobot format (parquet + PNG images) — one LeRobotDataset per
iteration under iter_NNN/online_episodes/.  OnlineQDataset loads ALL past iteration
datasets lazily: only metadata (episode boundaries, success flag) is read at
construction time; images and actions are loaded per-frame on demand via
LeRobotDataset.__getitem__, which reads individual PNG files.

This avoids loading full episode tensors into RAM — the old .pt approach required
~430 MB per episode × N episodes in the growing buffer, easily exceeding the
64 GB/GPU SLURM limit.

Episode .pt files (legacy) are no longer read here; save_episodes_lerobot() in
self_improvement_loop.py handles writing.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


_CAMERA_KEYS = ("observation.images.image", "observation.images.image2")

# Image feature shape saved during collection: (H, W, C) uint8, env native resolution.
ONLINE_IMAGE_SHAPE = (256, 256, 3)

# Feature spec for LeRobotDataset.create() — must match save_episodes_lerobot().
ONLINE_DATASET_FEATURES = {
    "observation.images.image": {
        "dtype": "image",
        "shape": ONLINE_IMAGE_SHAPE,
        "names": ["height", "width", "channel"],
    },
    "observation.images.image2": {
        "dtype": "image",
        "shape": ONLINE_IMAGE_SHAPE,
        "names": ["height", "width", "channel"],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": None,
    },
    "episode_success": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
}

ONLINE_DATASET_FPS = 10.0
ONLINE_DATASET_REPO_ID = "online_episodes"


def save_episodes_lerobot(
    episode_dicts: list[dict],
    episodes_dir: Path,
    action_dim: int = 7,
    fps: float = ONLINE_DATASET_FPS,
    camera_keys: tuple[str, ...] = _CAMERA_KEYS,
    image_shape: "tuple[int, int, int] | None" = None,
) -> None:
    """Save a list of episode dicts to a LeRobotDataset at episodes_dir.

    episode_dicts must have keys produced by _run_episode():
        "action":                    (T, A) float32 tensor
        "observation.images.image":  (T, 3, H, W) float32 [0,1] tensor (unflipped)
        "observation.images.image2": (T, 3, H, W) float32 [0,1] tensor (unflipped)
        "task":                      str
        "success":                   bool

    Images are flipped along H and W before saving to match the LiberoProcessorStep
    convention used by the Q-function preprocessor (same as offline training data).
    """
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Build the image features from camera_keys rather than the module-level LIBERO
    # default, otherwise a non-LIBERO env writes frames whose keys do not match the
    # declared schema and LeRobotDataset.add_frame raises "Feature mismatch".
    # Online images MUST be stored at the same resolution as the offline dataset they
    # are batched with: finetune_q concatenates the two, and a DataLoader cannot
    # collate tensors of different sizes ("Trying to resize storage that is not
    # resizable"). Default to the resolution the episodes were actually captured at,
    # which is the env's native size and therefore matches the offline demos.
    if image_shape is None:
        first_img = next(
            (ep[k] for ep in episode_dicts for k in camera_keys if k in ep), None
        )
        image_shape = (
            (int(first_img.shape[-2]), int(first_img.shape[-1]), 3)
            if first_img is not None else ONLINE_IMAGE_SHAPE
        )

    features = {k: v for k, v in ONLINE_DATASET_FEATURES.items() if not k.startswith("observation.images.")}
    features = {k: dict(v) for k, v in features.items()}
    for key in camera_keys:
        features[key] = {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }
    features["action"]["shape"] = (action_dim,)

    episodes_dir.parent.mkdir(parents=True, exist_ok=True)
    ds = LeRobotDataset.create(
        repo_id=ONLINE_DATASET_REPO_ID,
        fps=fps,
        features=features,
        root=str(episodes_dir),
        use_videos=False,
    )

    for ep in episode_dicts:
        T = int(ep["action"].shape[0])
        ep_success = bool(ep.get("success", False))
        task_str = ep.get("task", "")

        for t in range(T):
            frame: dict = {
                "task": task_str,
                "action": ep["action"][t].numpy().astype(np.float32),
                "episode_success": np.array([ep_success], dtype=bool),
            }
            for key in camera_keys:
                if key in ep:
                    img = ep[key][t]  # (3, H, W) float32 [0,1]
                    img = torch.flip(img, dims=[1, 2])  # flip H, W to match offline convention
                    target_h, target_w = image_shape[:2]
                    if img.shape[-2] != target_h or img.shape[-1] != target_w:
                        import torch.nn.functional as F
                        img = F.interpolate(img.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False).squeeze(0)
                    frame[key] = (img * 255).byte().permute(1, 2, 0).numpy()  # (H, W, C) uint8
            ds.add_frame(frame)

        ds.save_episode()

    ds.finalize()


class OnlineQDataset(Dataset):
    """Growing-buffer online Q-function dataset backed by per-iteration LeRobotDatasets.

    At construction, globs output_dir for all iter_*/online_episodes/ directories and
    loads them as LeRobotDataset instances (metadata only — no images in RAM).
    __getitem__ loads individual frames on demand via LeRobotDataset.

    Args:
        output_dir: Self-improvement run directory; globs iter_*/online_episodes/.
        h: Q-function horizon (same as QFunctionConfig.h).
        fps: Dataset FPS — must match ONLINE_DATASET_FPS used at save time.
        terminal_bonus: Reward at the terminal frame of successful episodes.
        camera_keys: Image keys to include.
    """

    def __init__(
        self,
        output_dir: Path,
        h: int,
        fps: float = ONLINE_DATASET_FPS,
        terminal_bonus: float = 1.0,
        camera_keys: tuple[str, ...] = _CAMERA_KEYS,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.h = h
        self.fps = fps
        self.terminal_bonus = terminal_bonus
        self.camera_keys = camera_keys

        delta_timestamps: dict[str, list[float]] = {k: [0.0, h / fps] for k in camera_keys}
        delta_timestamps["action"] = [i / fps for i in range(2 * h)]

        ds_dirs = sorted(output_dir.glob("iter_*/online_episodes"))
        if not ds_dirs:
            raise ValueError(f"No iter_*/online_episodes dirs found under {output_dir}")

        self._sub_datasets: list[LeRobotDataset] = []
        # (ds_idx, local_frame_idx, frame_in_ep, ep_len)
        self._index: list[tuple[int, int, int, int]] = []

        for ds_dir in ds_dirs:
            ds = LeRobotDataset(
                repo_id=ONLINE_DATASET_REPO_ID,
                root=str(ds_dir),          # dataset root = online_episodes/ dir itself
                delta_timestamps=delta_timestamps,
                force_cache_sync=False,    # never fetch from HF Hub
            )
            ds_idx = len(self._sub_datasets)
            self._sub_datasets.append(ds)

            eps_meta = ds.meta.episodes
            for ep_i in range(len(eps_meta)):
                ep = eps_meta[ep_i]
                ep_from = int(ep["dataset_from_index"])
                ep_len  = int(ep["length"])
                for local_idx in range(ep_from, ep_from + ep_len):
                    frame_in_ep = local_idx - ep_from
                    self._index.append((ds_idx, local_idx, frame_in_ep, ep_len))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        ds_idx, local_idx, frame_in_ep, ep_len = self._index[idx]
        item = self._sub_datasets[ds_idx][local_idx]

        h = self.h
        terminal_t = ep_len - 1

        # episode_success is (1,) bool tensor; constant for all frames in the episode.
        ep_success = bool(item["episode_success"])
        ep_terminal_bonus = self.terminal_bonus if ep_success else 0.0

        out: dict = {}

        # Images: (2, 3, H, W) stacked by LeRobotDataset for delta_timestamps [0, h/fps]
        for key in self.camera_keys:
            if key in item:
                out[key] = item[key]

        # Actions: (2h, A) stacked by LeRobotDataset for delta_timestamps [0, ..., (2h-1)/fps]
        out["action"] = item["action"]

        # Task string
        out["task"] = item.get("task", "")

        # Reward labels
        reward_chunk = torch.zeros(h, dtype=torch.float32)
        reward_pad = torch.zeros(h, dtype=torch.bool)
        for i in range(h):
            f = frame_in_ep + i
            if f < ep_len:
                reward_chunk[i] = ep_terminal_bonus if f == terminal_t else 0.0
            else:
                reward_pad[i] = True

        bootstrap_valid = (frame_in_ep + h) < (ep_len - 1)

        out["q_reward_chunk_first"] = reward_chunk
        out["q_reward_pad_first"] = reward_pad
        out["q_bootstrap_valid"] = torch.tensor(bool(bootstrap_valid))
        out["q_bucket_index"] = torch.tensor(0, dtype=torch.long)

        return out
