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
"""Sanity-check the RoboTwin <-> LeRobot integration before a long training run.

Runs four checks and prints ``[ OK ]`` / ``[FAIL]`` for each:
  1. Dataset    — the re-packed dataset loads; a batch has the expected
                  single-image / 14-D state / 14-D action shapes.
  2. Env        — RoboTwinEnv resets + steps; observation shapes are correct.
  3. Parity     — the dataset image and the eval-time RoboTwinProcessorStep
                  image are the same [3,384,320] layout.
  4. Determinism — two reset(seed=0) calls give identical observations.

Usage:
    python scripts/sanity_check_robotwin.py \
        --dataset_repo_id local/robotwin2.0_concat \
        --dataset_root /path/to/robotwin2.0_concat \
        --robotwin_root /path/to/RoboTwin --task beat_block_hammer
"""
import argparse
import sys
import traceback

import torch

_RESULTS: list[tuple[str, bool, str]] = []


def _check(name: str):
    def deco(fn):
        try:
            msg = fn()
            _RESULTS.append((name, True, msg or ""))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            _RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        return fn

    return deco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_repo_id", default="local/robotwin2.0_concat")
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--robotwin_root", required=True)
    ap.add_argument("--task", default="beat_block_hammer")
    a = ap.parse_args()

    state = {}

    @_check("dataset")
    def _dataset():
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(a.dataset_repo_id, root=a.dataset_root)
        feats = ds.meta.features
        assert "observation.images.image" in feats, f"missing image feature; have {list(feats)}"
        frame = ds[0]
        img, st, act = frame["observation.images.image"], frame["observation.state"], frame["action"]
        assert tuple(img.shape) == (3, 384, 320), f"image shape {tuple(img.shape)}"
        assert st.shape[-1] == 14, f"state dim {st.shape}"
        assert act.shape[-1] == 14, f"action dim {act.shape}"
        state["ds_image"] = img
        return f"{ds.meta.total_episodes} eps, image={tuple(img.shape)} state={tuple(st.shape)}"

    @_check("env")
    def _env():
        from lerobot.envs.robotwin import RoboTwinEnv

        env = RoboTwinEnv(task_name=a.task, robotwin_root=a.robotwin_root)
        obs, _ = env.reset(seed=0)
        for cam in ("head_camera", "left_camera", "right_camera"):
            assert obs["pixels"][cam].ndim == 3, f"{cam} not HWC"
        assert obs["agent_pos"].shape == (14,), f"agent_pos {obs['agent_pos'].shape}"
        obs, *_ = env.step(env.action_space.sample())
        env.close()
        return "reset + step OK"

    @_check("parity")
    def _parity():
        from lerobot.envs.robotwin import RoboTwinEnv, build_robotwin_image

        env = RoboTwinEnv(task_name=a.task, robotwin_root=a.robotwin_root)
        obs, _ = env.reset(seed=0)
        env.close()

        def chw(x):
            return torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        img = build_robotwin_image(
            chw(obs["pixels"]["head_camera"]),
            chw(obs["pixels"]["left_camera"]),
            chw(obs["pixels"]["right_camera"]),
        )
        assert tuple(img.shape) == (1, 3, 384, 320), f"env concat {tuple(img.shape)}"
        if "ds_image" in state:
            assert tuple(state["ds_image"].shape) == tuple(img.shape[1:]), "dataset/env layout differ"
        return f"env concat {tuple(img.shape)} == dataset layout"

    @_check("determinism")
    def _determinism():
        import numpy as np

        from lerobot.envs.robotwin import RoboTwinEnv

        e1 = RoboTwinEnv(task_name=a.task, robotwin_root=a.robotwin_root)
        o1, _ = e1.reset(seed=0)
        e1.close()
        e2 = RoboTwinEnv(task_name=a.task, robotwin_root=a.robotwin_root)
        o2, _ = e2.reset(seed=0)
        e2.close()
        same = np.array_equal(o1["pixels"]["head_camera"], o2["pixels"]["head_camera"])
        assert same, "reset(seed=0) not deterministic"
        return "reset(seed=0) reproducible"

    print("\n=== RoboTwin integration sanity check ===")
    for name, ok, msg in _RESULTS:
        print(f"  [{' OK ' if ok else 'FAIL'}] {name:13s} {msg}")
    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print(f"=== {len(_RESULTS) - n_fail}/{len(_RESULTS)} checks passed ===")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
