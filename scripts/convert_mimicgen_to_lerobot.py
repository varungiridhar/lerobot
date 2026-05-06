#!/usr/bin/env python
"""Convert MimicGen HDF5 datasets to LeRobot v3.0 format.

Usage:
    python scripts/convert_mimicgen_to_lerobot.py \
        --hdf5_path /path/to/coffee_d0.hdf5 \
        --repo_id local/mimicgen_coffee_d0 \
        [--root /path/to/output] [--max_episodes 100] [--fps 20] [--save_init_states]
"""

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

MIMICGEN_TASK_DESCRIPTIONS: dict[str, str] = {
    "Coffee_D0": "Pick up the coffee pod and place it in the coffee machine",
    "Coffee_D1": "Pick up the coffee pod and place it in the coffee machine",
    "Coffee_D2": "Pick up the coffee pod and place it in the coffee machine",
    "Stack_D0": "Stack cube A on top of cube B",
    "Stack_D1": "Stack cube A on top of cube B",
    "StackThree_D0": "Stack three cubes on top of each other",
    "StackThree_D1": "Stack three cubes on top of each other",
    "Square_D0": "Place the square nut on the square peg",
    "Square_D1": "Place the square nut on the square peg",
    "Square_D2": "Place the square nut on the square peg",
    "Threading_D0": "Thread the needle through the tripod",
    "Threading_D1": "Thread the needle through the tripod",
    "Threading_D2": "Thread the needle through the tripod",
    "ThreePieceAssembly_D0": "Assemble three pieces together on the base",
    "HammerCleanup_D0": "Pick up the hammer and place it in the drawer",
    "HammerCleanup_D1": "Pick up the hammer and place it in the drawer",
    "MugCleanup_D0": "Pick up the mug and place it in the drawer",
    "MugCleanup_D1": "Pick up the mug and place it in the drawer",
    "NutAssembly_D0": "Assemble the nuts onto the pegs",
    "PickPlace_D0": "Pick objects and place them in the correct bins",
    "Kitchen_D0": "Complete the kitchen task sequence",
    "Kitchen_D1": "Complete the kitchen task sequence",
    "CoffeePreparation_D0": "Prepare coffee from start to finish",
    "CoffeePreparation_D1": "Prepare coffee from start to finish",
}


def _sorted_demo_keys(f):
    return sorted(
        (k for k in f["data"].keys() if k.startswith("demo")),
        key=lambda x: int(x.split("_")[1]),
    )


def _save_init_states(hdf5_path, demo_keys, out_path):
    init_states, model_files = [], []
    with h5py.File(hdf5_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        for dk in demo_keys:
            demo = f[f"data/{dk}"]
            init_states.append(demo["states"][0])
            model_files.append(demo.attrs.get("model_file", ""))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"states": torch.tensor(np.stack(init_states)), "model_files": model_files, "env_args": env_args}, out_path)
    print(f"Saved {len(init_states)} init states to {out_path}")


def convert_mimicgen_to_lerobot(hdf5_path, repo_id, root=None, max_episodes=None, fps=20, use_videos=True, save_init_states=False):
    hdf5_path = Path(hdf5_path)
    with h5py.File(hdf5_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        env_name = env_args["env_name"]
        task_description = MIMICGEN_TASK_DESCRIPTIONS.get(env_name, env_name)
        demo_keys = _sorted_demo_keys(f)
        if max_episodes:
            demo_keys = demo_keys[:max_episodes]
        first_obs = f[f"data/{demo_keys[0]}/obs"]
        has_agentview = "agentview_image" in first_obs
        has_eye_in_hand = "robot0_eye_in_hand_image" in first_obs
        img_shape = tuple(first_obs["agentview_image"].shape[1:]) if has_agentview else None
        action_dim = f[f"data/{demo_keys[0]}/actions"].shape[1]

    print(f"Converting {len(demo_keys)} episodes from {env_name} ({task_description})")

    img_dtype = "video" if use_videos else "image"
    features = {
        "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (8,), "names": ["eef_x", "eef_y", "eef_z", "aa_x", "aa_y", "aa_z", "grip_l", "grip_r"]},
    }
    if has_agentview:
        features["observation.images.image"] = {"dtype": img_dtype, "shape": img_shape, "names": ["height", "width", "channels"]}
    if has_eye_in_hand:
        features["observation.images.image2"] = {"dtype": img_dtype, "shape": img_shape, "names": ["height", "width", "channels"]}

    root_path = Path(root) if root else HF_LEROBOT_HOME / repo_id
    if root_path.exists():
        shutil.rmtree(root_path)

    dataset = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=root_path, robot_type="panda", use_videos=use_videos)

    with h5py.File(hdf5_path, "r") as f:
        for ep_idx, demo_key in enumerate(demo_keys):
            demo = f[f"data/{demo_key}"]
            n_frames = int(demo.attrs["num_samples"])
            actions = demo["actions"][:].astype(np.float32)
            eef_pos = demo["obs/robot0_eef_pos"][:].astype(np.float32)
            eef_quat = demo["obs/robot0_eef_quat"][:].astype(np.float32)
            gripper_qpos = demo["obs/robot0_gripper_qpos"][:].astype(np.float32)
            agentview = demo["obs/agentview_image"][:] if has_agentview else None
            eye_in_hand = demo["obs/robot0_eye_in_hand_image"][:] if has_eye_in_hand else None
            axis_angle = Rotation.from_quat(eef_quat).as_rotvec().astype(np.float32)
            state = np.concatenate([eef_pos, axis_angle, gripper_qpos], axis=-1)

            for t in range(n_frames):
                frame = {"action": actions[t], "observation.state": state[t], "task": task_description}
                if has_agentview:
                    frame["observation.images.image"] = Image.fromarray(agentview[t])
                if has_eye_in_hand:
                    frame["observation.images.image2"] = Image.fromarray(eye_in_hand[t])
                dataset.add_frame(frame)
            dataset.save_episode()
            if (ep_idx + 1) % 50 == 0 or ep_idx == len(demo_keys) - 1:
                print(f"  [{ep_idx + 1}/{len(demo_keys)}] episodes converted")

    dataset.finalize()
    if save_init_states:
        _save_init_states(hdf5_path, demo_keys, root_path / "meta" / "init_states.pt")
    print(f"Done. {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames at {root_path}")
    return dataset


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5_path", required=True)
    p.add_argument("--repo_id", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--no_videos", action="store_true")
    p.add_argument("--save_init_states", action="store_true")
    a = p.parse_args()
    convert_mimicgen_to_lerobot(a.hdf5_path, a.repo_id, a.root, a.max_episodes, a.fps, not a.no_videos, a.save_init_states)
