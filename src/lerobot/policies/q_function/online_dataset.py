"""OnlineQDataset: wraps per-episode .pt files collected during eval rollouts.

Produces the same __getitem__ format as QValueLabelDataset for use in
self-improvement Q fine-tuning. Reward mode is always 'all_success': terminal
frame gets q_reward=1.0, all others get 0.0.

Episode .pt files must contain:
    "observation.images.image":  (T, 3, H, W) float32 [0,1], NOT flipped
    "observation.images.image2": (T, 3, H, W) float32 [0,1], NOT flipped (optional)
    "action":                    (T, A) float32
    "task":                      str
    (other keys from _compile_episode_data are ignored)

Images are flipped along H and W dims to match the LiberoProcessorStep convention
used during original Q training (raw rollout obs are saved pre-flip).

Images are also resized to (target_image_size, target_image_size) to match the
original training dataset resolution (256×256 for LIBERO Q-function). The Q
preprocessor will then resize them to the model's input resolution (224×224).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


_CAMERA_KEYS = ("observation.images.image", "observation.images.image2")


class OnlineQDataset(Dataset):
    """Wrap successful episode .pt files as a Q-function training dataset.

    Args:
        episode_files: Paths to .pt episode files produced by collect_episodes().
        h: Q-function horizon (same as QFunctionConfig.h).
        terminal_bonus: Reward assigned at the last frame of each episode.
        camera_keys: Image keys to include. Defaults to both LIBERO cameras.
        target_image_size: Resize images to this square size to match the
            original training dataset resolution before Q-preprocessor scaling.
            None means no resize (use when env and dataset resolutions match).
    """

    def __init__(
        self,
        episode_files: list[Path],
        h: int,
        terminal_bonus: float = 1.0,
        camera_keys: tuple[str, ...] = _CAMERA_KEYS,
        target_image_size: int | None = 256,
    ):
        self.h = h
        self.terminal_bonus = terminal_bonus
        self.camera_keys = camera_keys
        self.target_image_size = target_image_size

        self._episodes: list[dict] = []
        self._index: list[tuple[int, int]] = []  # (episode_idx, frame_t)

        for path in episode_files:
            ep = torch.load(str(path), map_location="cpu", weights_only=False)
            ep_idx = len(self._episodes)
            T = int(ep["action"].shape[0])
            self._episodes.append(ep)
            for t in range(T):
                self._index.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, t = self._index[idx]
        ep = self._episodes[ep_idx]
        T = int(ep["action"].shape[0])
        terminal_t = T - 1
        h = self.h

        # Bootstrap obs at t+h, clamped to last in-episode frame.
        t_boot = min(t + h, terminal_t)
        out: dict = {}

        # ── Images: (2, 3, H, W) stacks at [t, t+h], flipped to match Q format ──
        for key in self.camera_keys:
            if key not in ep:
                continue
            imgs: Tensor = ep[key]  # (T, 3, H, W) unflipped
            frame_t = torch.flip(imgs[t], dims=[1, 2])      # flip H, W
            frame_boot = torch.flip(imgs[t_boot], dims=[1, 2])
            if self.target_image_size is not None:
                s = self.target_image_size
                frame_t = F.interpolate(frame_t.unsqueeze(0), size=(s, s), mode="bilinear", align_corners=False).squeeze(0)
                frame_boot = F.interpolate(frame_boot.unsqueeze(0), size=(s, s), mode="bilinear", align_corners=False).squeeze(0)
            out[key] = torch.stack([frame_t, frame_boot], dim=0)  # (2, 3, H, W)

        # ── Action window [t : t+2h] with end-of-episode padding ────────────
        chunks = []
        for i in range(2 * h):
            ti = min(t + i, terminal_t)
            chunks.append(ep["action"][ti])
        out["action"] = torch.stack(chunks, dim=0)  # (2h, A)

        # ── Task string ──────────────────────────────────────────────────────
        out["task"] = ep.get("task", "")

        # ── Reward labels (all_success mode) ────────────────────────────────
        reward_chunk = torch.zeros(h, dtype=torch.float32)
        reward_pad = torch.zeros(h, dtype=torch.bool)
        for i in range(h):
            f = t + i
            if f < T:
                reward_chunk[i] = self.terminal_bonus if f == terminal_t else 0.0
            else:
                reward_pad[i] = True

        bootstrap_valid = (t + h) < (T - 1)

        out["q_reward_chunk_first"] = reward_chunk
        out["q_reward_pad_first"] = reward_pad
        out["q_bootstrap_valid"] = torch.tensor(bool(bootstrap_valid))
        out["q_bucket_index"] = torch.tensor(0, dtype=torch.long)

        return out
