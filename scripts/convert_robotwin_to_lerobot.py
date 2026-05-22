#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Re-pack RoboTwin episodes into FastWAM's single-concatenated-image format.

Reads the ``robotwin2.0-fastwam`` dataset's raw v2.1 files directly — per-episode
parquet (14-D state + 14-D action) and the three per-episode camera mp4s
(``cam_high``, ``cam_left_wrist``, ``cam_right_wrist``) — concatenates the
cameras with the shared ``build_robotwin_image`` helper (the exact eval-time
layout), and writes a new LeRobot v3.0 dataset with a single
``observation.images.image`` ``[3, 384, 320]`` feature.

Reading the v2.1 files directly (rather than running the full v2.1->v3.0
migration) lets us re-pack an arbitrary episode subset and skip episodes whose
source videos are missing/corrupt.

Usage:
    python scripts/convert_robotwin_to_lerobot.py \
        --src_root /path/to/robotwin2.0 \
        --episodes 5000-5059 \
        --dst_repo_id local/robotwin2.0_concat \
        --dst_root /path/to/robotwin2.0_concat
"""
import argparse
import json
import shutil
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.robotwin import build_robotwin_image

CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
CHUNK_SIZE = 1000


def decode_video_frames(path: Path) -> list[torch.Tensor]:
    """Decode every frame of an mp4 into a list of (3, H, W) uint8 tensors.

    Uses pyav (self-contained FFmpeg) — torchcodec fails to load FFmpeg in this
    env due to a libstdc++/libtbb ABI clash introduced by the SAPIEN sim deps.
    """
    frames: list[torch.Tensor] = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
            frames.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
    return frames


def parse_episodes(spec: str) -> list[int]:
    """Parse '5000-5059' / '5000,5001' / '5000-5009,5020' into a sorted list."""
    eps: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            eps.update(range(int(lo), int(hi) + 1))
        elif part:
            eps.add(int(part))
    return sorted(eps)


def _ep_paths(src: Path, ep: int) -> tuple[Path, dict[str, Path]]:
    chunk = f"chunk-{ep // CHUNK_SIZE:03d}"
    name = f"episode_{ep:06d}"
    parquet = src / "data" / chunk / f"{name}.parquet"
    videos = {cam: src / "videos" / chunk / f"observation.images.{cam}" / f"{name}.mp4" for cam in CAMERAS}
    return parquet, videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, help="Path to the extracted robotwin2.0 v2.1 dataset")
    ap.add_argument("--episodes", required=True, help="Episode spec, e.g. '5000-5059'")
    ap.add_argument("--dst_repo_id", default="local/robotwin2.0_concat")
    ap.add_argument("--dst_root", required=True)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--no_videos", action="store_true")
    a = ap.parse_args()

    src = Path(a.src_root)
    episodes = parse_episodes(a.episodes)
    print(f"Re-packing {len(episodes)} episodes: {episodes[0]}..{episodes[-1]}")

    # task_index -> task string
    tasks: dict[int, str] = {}
    with open(src / "meta" / "tasks.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            tasks[int(d["task_index"])] = d["task"]

    dst_root = Path(a.dst_root)
    if dst_root.exists():
        if (dst_root / "meta" / "info.json").is_file():
            print(f"Removing existing dataset at {dst_root}")
            shutil.rmtree(dst_root)
        else:
            raise RuntimeError(f"Refusing to delete non-dataset dir {dst_root}")

    img_dtype = "image" if a.no_videos else "video"
    features = {
        "action": {"dtype": "float32", "shape": (14,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (14,), "names": None},
        "observation.images.image": {
            "dtype": img_dtype, "shape": (384, 320, 3), "names": ["height", "width", "channels"]
        },
    }
    dst = LeRobotDataset.create(
        repo_id=a.dst_repo_id, fps=a.fps, features=features, root=dst_root,
        robot_type="aloha", use_videos=not a.no_videos,
    )

    n_done = 0
    for ep in episodes:
        parquet, video_paths = _ep_paths(src, ep)
        if not parquet.is_file() or not all(p.is_file() and p.stat().st_size > 0 for p in video_paths.values()):
            print(f"  skip episode {ep}: missing/empty parquet or video")
            continue

        df = pd.read_parquet(parquet)
        cam_frames = {cam: decode_video_frames(video_paths[cam]) for cam in CAMERAS}
        n_frames = min(len(df), *(len(v) for v in cam_frames.values()))
        if n_frames != len(df):
            print(f"  episode {ep}: parquet rows={len(df)} vs video frames "
                  f"{[len(v) for v in cam_frames.values()]} — using {n_frames}")

        for t in range(n_frames):
            image = build_robotwin_image(
                cam_frames["cam_high"][t],
                cam_frames["cam_left_wrist"][t],
                cam_frames["cam_right_wrist"][t],
            )  # (3, 384, 320) uint8
            dst.add_frame({
                "action": torch.tensor(np.asarray(df["action"].iloc[t]), dtype=torch.float32),
                "observation.state": torch.tensor(
                    np.asarray(df["observation.state"].iloc[t]), dtype=torch.float32
                ),
                "observation.images.image": Image.fromarray(
                    image.permute(1, 2, 0).contiguous().numpy()
                ),
                "task": tasks[int(df["task_index"].iloc[t])],
            })
        dst.save_episode()
        n_done += 1
        if n_done % 20 == 0:
            print(f"  re-packed {n_done}/{len(episodes)} episodes")

    dst.finalize()
    print(f"Done. {dst.meta.total_episodes} episodes, {dst.meta.total_frames} frames at {dst_root}")


if __name__ == "__main__":
    main()
