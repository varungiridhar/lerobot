#!/usr/bin/env python
"""Cheap source/data audit for RoboTwin frame and action timing semantics.

This deliberately avoids constructing SAPIEN. It verifies the exact code paths
used to collect demonstrations and deploy FastWAM, then reports which FPS values
are metadata and which timing is physical.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml


def _compact_source(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return re.sub(r"\s+", "", path.read_text(encoding="utf-8"))


def _require(source: str, needle: str, description: str) -> None:
    if needle not in source:
        raise RuntimeError(f"Could not verify {description}; expected source fragment {needle!r}")


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(f"None of the timing-audit source paths exists: {paths}")


def inspect_timing_contract(robotwin_root: Path, dataset_root: Path, repo_root: Path) -> dict:
    base_task_path = robotwin_root / "envs" / "_base_task.py"
    task_cfg_path = robotwin_root / "task_config" / "demo_randomized.yml"
    baseline_deploy_path = _first_existing(
        robotwin_root
        / "reference"
        / "FastWAM"
        / "experiments"
        / "robotwin"
        / "fastwam_policy"
        / "deploy_policy.py",
        robotwin_root.parent
        / "reference"
        / "FastWAM"
        / "experiments"
        / "robotwin"
        / "fastwam_policy"
        / "deploy_policy.py",
    )
    wrapper_path = repo_root / "src" / "lerobot" / "envs" / "robotwin.py"
    dataset_info_path = dataset_root / "meta" / "info.json"

    base_task = _compact_source(base_task_path)
    baseline_deploy = _compact_source(baseline_deploy_path)
    wrapper = _compact_source(wrapper_path)

    _require(
        base_task,
        'self.scene.set_timestep(kwargs.get("timestep",1/250))',
        "the default 250 Hz SAPIEN physics timestep",
    )
    _require(base_task, "TOPP(left_path,1/250", "left-arm TOPP at the physics timestep")
    _require(base_task, "TOPP(right_path,1/250", "right-arm TOPP at the physics timestep")
    _require(
        base_task,
        "whilenow_left_id<left_n_stepornow_right_id<right_n_step:",
        "variable-length TOPP execution",
    )
    _require(
        baseline_deploy,
        'task_env.take_action(action,action_type="qpos")',
        "released FastWAM's RoboTwin qpos deployment API",
    )
    _require(
        wrapper,
        'self._task_env.take_action(action,action_type="qpos")',
        "the LeRobot wrapper's matching qpos deployment API",
    )
    _require(wrapper, '"render_fps":25', "the wrapper's video render FPS")

    task_cfg = yaml.safe_load(task_cfg_path.read_text(encoding="utf-8"))
    save_freq = int(task_cfg["save_freq"])
    if save_freq <= 0:
        raise RuntimeError(f"Expected a positive demonstration save_freq, got {save_freq}")

    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    dataset_fps = float(dataset_info["fps"])
    physics_hz = 250.0
    configured_demo_sample_hz = physics_hz / save_freq

    return {
        "physics_hz": physics_hz,
        "configured_demo_save_every_physics_steps": save_freq,
        "configured_demo_nominal_sample_hz": configured_demo_sample_hz,
        "lerobot_dataset_metadata_fps": dataset_fps,
        "wrapper_render_video_fps": 25.0,
        "policy_step_semantics": "one variable-duration TOPP qpos transition",
        "baseline_action_api_match": True,
        "one_policy_step_is_one_fixed_dataset_frame_period": False,
        "interpretation": (
            "50 FPS preserves released LeRobot frame indexing/video metadata; 25 FPS controls "
            "rendered-video playback only. Neither is RoboTwin's policy control rate. The wrapper "
            "matches released FastWAM evaluation by calling the same variable-duration qpos API."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin_root", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_timing_contract(args.robotwin_root, args.dataset_root, args.repo_root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
