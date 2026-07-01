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
"""RoboTwin 2.0 environment wrapper for LeRobot.

Wraps RoboTwin's SAPIEN-based bimanual manipulation tasks into the standard
``gymnasium.Env`` interface expected by the LeRobot eval pipeline.

RoboTwin tasks are not gym environments — each task is a Python class with its
own lifecycle (``setup_demo`` -> ``get_obs`` / ``take_action`` loop ->
``close_env``). This wrapper drives that lifecycle behind a standard
reset/step interface so ``lerobot-eval`` can roll out any LeRobot policy.

The observation mirrors the MimicGen wrapper's structure (``pixels`` dict +
flat ``agent_pos``) so the same preprocessing pipeline is reused. The three
raw cameras are exposed separately; the FastWAM-specific concat into a single
[3, 384, 320] frame is done by ``RoboTwinProcessorStep`` (see env_processor.py).
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
import sys
import traceback
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import yaml
from gymnasium import spaces

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTION_DIM = 14  # bimanual qpos: 6 arm joints + 1 gripper, per arm
STATE_DIM = 14
# RoboTwin's default step limit when a task is absent from _eval_step_limit.yml
DEFAULT_MAX_EPISODE_STEPS = 1000
# Number of consecutive seeds to try if setup_demo raises (unstable scene).
RESET_RETRIES = 12
# FastWAM expert-check seed gating (RoboTwin/script/eval_policy.py::eval_policy).
# The seed cursor starts at 100000*(1+seed); a seed is scored only if the scripted
# oracle solves it. Cap the per-episode seed search so a task with no solvable
# seed fails loudly instead of hanging.
SEED_BASE_MULTIPLIER = 100000
MAX_GATE_CANDIDATES = 200


# ---------------------------------------------------------------------------
# Image layout helper — the single source of truth for the FastWAM RoboTwin
# 3-camera concat. Used by both RoboTwinProcessorStep (eval) and the dataset
# re-packer (training) so train/eval see a bit-identical layout.
# ---------------------------------------------------------------------------
def build_robotwin_image(head: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Concatenate the 3 RoboTwin cameras into FastWAM's single [3, 384, 320] frame.

    Layout (matches FastWAM ``deploy_policy.py::_build_robotwin_image_tensor`` and
    ``robot_video_dataset.py`` ``concat_multi_camera="robotwin"``)::

        head  resized to 256h x 320w            -> top
        left  resized to 128h x 160w  \\
        right resized to 128h x 160w  /  side-by-side -> 128h x 320w bottom
        image = vstack(head, bottom)            -> 384h x 320w

    Args:
        head, left, right: RGB tensors in channel-first ``(..., 3, H, W)`` layout
            (an optional leading batch dim is supported). Any dtype/scale is kept.

    Returns:
        Concatenated tensor of shape ``(..., 3, 384, 320)``.
    """

    def _resize(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        squeeze = x.dim() == 3
        if squeeze:
            x = x.unsqueeze(0)
        out = F.interpolate(x.float(), size=(h, w), mode="bilinear", align_corners=False)
        out = out.to(x.dtype)
        return out.squeeze(0) if squeeze else out

    head = _resize(head, 256, 320)
    left = _resize(left, 128, 160)
    right = _resize(right, 128, 160)
    bottom = torch.cat([left, right], dim=-1)  # concat along width -> (..., 3, 128, 320)
    image = torch.cat([head, bottom], dim=-2)  # concat along height -> (..., 3, 384, 320)
    return image


@contextlib.contextmanager
def _robotwin_ctx(path: str | Path):
    """Context for every RoboTwin call: chdir into the RoboTwin root + enable grad.

    - chdir: RoboTwin resolves most paths via its own ROOT_PATH, but a few code
      paths (asset/texture loading) use cwd-relative paths.
    - enable_grad: RoboTwin's curobo motion planner runs an autograd-based Newton
      optimizer (it calls ``.backward()`` internally). ``lerobot-eval`` wraps the
      whole rollout in ``torch.no_grad()``, which would otherwise break curobo
      with "element 0 of tensors does not require grad".
    """
    prev = os.getcwd()
    os.chdir(path)
    try:
        with torch.enable_grad():
            yield
    finally:
        os.chdir(prev)


def _humanize(task_name: str) -> str:
    return task_name.replace("_", " ").strip().capitalize()


# ---------------------------------------------------------------------------
# RoboTwin config (`args` dict) construction — ports eval_policy.py::main
# ---------------------------------------------------------------------------
def _build_robotwin_args(robotwin_root: Path, task_name: str, task_config: str) -> dict[str, Any]:
    """Rebuild the ``args`` dict that ``setup_demo`` consumes.

    Mirrors ``RoboTwin/script/eval_policy.py::main``: loads the task-config yaml,
    resolves the embodiment + camera config, and adds the programmatic keys.
    Embodiment file paths are made absolute so the wrapper is cwd-independent.
    """
    cfg_dir = robotwin_root / "task_config"
    with open(cfg_dir / f"{task_config}.yml", encoding="utf-8") as f:
        args: dict[str, Any] = yaml.safe_load(f)

    with open(cfg_dir / "_embodiment_config.yml", encoding="utf-8") as f:
        embodiment_types = yaml.safe_load(f)
    with open(cfg_dir / "_camera_config.yml", encoding="utf-8") as f:
        camera_config = yaml.safe_load(f)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = "lerobot"
    args["policy_name"] = "lerobot"
    args["eval_mode"] = True
    # lerobot-eval records video via env.render(); disable RoboTwin's own ffmpeg.
    args["eval_video_log"] = False
    args["render_freq"] = 0

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]

    def _embodiment_file(name: str) -> str:
        rel = embodiment_types[name]["file_path"]
        return str((robotwin_root / rel).resolve())

    def _embodiment_config(robot_file: str) -> dict:
        with open(os.path.join(robot_file, "config.yml"), encoding="utf-8") as f:
            return yaml.safe_load(f)

    embodiment_type = args["embodiment"]
    if len(embodiment_type) == 1:
        args["left_robot_file"] = _embodiment_file(embodiment_type[0])
        args["right_robot_file"] = _embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = _embodiment_file(embodiment_type[0])
        args["right_robot_file"] = _embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError(f"embodiment must have 1 or 3 items, got {embodiment_type}")

    args["left_embodiment_config"] = _embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = _embodiment_config(args["right_robot_file"])
    return args


def _resolve_instruction(robotwin_root: Path, task_name: str) -> str:
    """Resolve a natural-language instruction for the task.

    Uses ``description/task_instruction/{task}.json``'s ``full_description``
    (placeholder-free), falling back to a humanized task name. The templated
    ``seen``/``unseen`` variants need per-episode object/arm fills from the
    scripted expert; that faithful path is a future refinement.
    """
    json_path = robotwin_root / "description" / "task_instruction" / f"{task_name}.json"
    if json_path.is_file():
        try:
            import json

            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            desc = data.get("full_description")
            if desc:
                return desc.replace("<", "").replace(">", "").strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to read instruction for %s: %s", task_name, e)
    return _humanize(task_name)


# ---------------------------------------------------------------------------
# Gym environment
# ---------------------------------------------------------------------------
class RoboTwinEnv(gym.Env):
    """Gymnasium wrapper around a single RoboTwin 2.0 task."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        task_name: str,
        robotwin_root: str | Path,
        task_config: str = "demo_randomized",
        instruction_type: str = "unseen",
        instruction: str | None = None,
        max_episode_steps: int | None = None,
        render_mode: str = "rgb_array",
        expert_check: bool = True,
    ):
        super().__init__()
        self.task_name = task_name
        self.robotwin_root = Path(robotwin_root).resolve()
        self.task_config = task_config
        self.instruction_type = instruction_type
        self.render_mode = render_mode
        self.expert_check = expert_check
        self._max_episode_steps_override = max_episode_steps
        # Persistent seed cursor for FastWAM's expert-check gating (set on 1st reset).
        self._now_seed: int | None = None

        if not self.robotwin_root.is_dir():
            raise FileNotFoundError(f"robotwin_root does not exist: {self.robotwin_root}")

        # RoboTwin imports `envs.<task>` and `policy` relative to its own root.
        for p in (str(self.robotwin_root), str(self.robotwin_root / "description" / "utils")):
            if p not in sys.path:
                sys.path.insert(0, p)

        # Instantiate the task class (light — scene setup happens in setup_demo).
        with _robotwin_ctx(self.robotwin_root):
            module = importlib.import_module(f"envs.{task_name}")
            task_cls = getattr(module, task_name)
            self._task_env = task_cls()

        self._args = _build_robotwin_args(self.robotwin_root, task_name, task_config)
        self._fixed_instruction = instruction
        self.task: str = instruction or _resolve_instruction(self.robotwin_root, task_name)

        self._episode_idx = 0
        self._last_obs: dict[str, Any] | None = None
        self._is_setup = False

        # Camera resolutions (head + wrist may differ by camera type).
        cam_cfg_path = self.robotwin_root / "task_config" / "_camera_config.yml"
        with open(cam_cfg_path, encoding="utf-8") as f:
            camera_config = yaml.safe_load(f)
        head_t = self._args["camera"]["head_camera_type"]
        wrist_t = self._args["camera"]["wrist_camera_type"]
        self._head_h, self._head_w = camera_config[head_t]["h"], camera_config[head_t]["w"]
        self._wrist_h, self._wrist_w = camera_config[wrist_t]["h"], camera_config[wrist_t]["w"]

        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        "head_camera": spaces.Box(
                            low=0, high=255, shape=(self._head_h, self._head_w, 3), dtype=np.uint8
                        ),
                        "left_camera": spaces.Box(
                            low=0, high=255, shape=(self._wrist_h, self._wrist_w, 3), dtype=np.uint8
                        ),
                        "right_camera": spaces.Box(
                            low=0, high=255, shape=(self._wrist_h, self._wrist_w, 3), dtype=np.uint8
                        ),
                    }
                ),
                "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float64),
            }
        )
        # qpos targets: arm joints (rad) + gripper (~[0, 1]); wide bounds.
        self.action_space = spaces.Box(low=-3.2, high=3.2, shape=(ACTION_DIM,), dtype=np.float32)

    # -- lifecycle -----------------------------------------------------------
    @property
    def _max_episode_steps(self) -> int:
        """Per-task step limit (lerobot_eval.rollout reads this via env.call())."""
        if self._max_episode_steps_override is not None:
            return int(self._max_episode_steps_override)
        lim = getattr(self._task_env, "step_lim", None)
        return int(lim) if lim else DEFAULT_MAX_EPISODE_STEPS

    def _close_task(self) -> None:
        if self._is_setup:
            with _robotwin_ctx(self.robotwin_root), contextlib.suppress(Exception):
                self._task_env.close_env(clear_cache=True)
            self._is_setup = False

    def reset(self, seed: int | None = None, options: dict | None = None):  # noqa: ARG002
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2**31 - 1))

        self._close_task()

        used_seed = self._gated_setup(seed) if self.expert_check else self._plain_setup(seed)

        self._episode_idx += 1
        obs = self._get_obs()
        info = {"is_success": False, "seed": used_seed, "task": self.task}
        return obs, info

    # -- setup helpers -------------------------------------------------------
    def _setup_seed(self, seed: int) -> None:
        """Build the task scene for `seed` (may raise UnStableError / build errors)."""
        with _robotwin_ctx(self.robotwin_root):
            self._task_env.setup_demo(now_ep_num=self._episode_idx, seed=seed, is_test=True, **self._args)

    def _safe_close(self) -> None:
        with _robotwin_ctx(self.robotwin_root), contextlib.suppress(Exception):
            self._task_env.close_env()

    def _plain_setup(self, seed: int) -> int:
        """Ungated setup: skip past unstable scenes only (no expert-solvability gate)."""
        first_exc: Exception | None = None
        attempt_errors: list[str] = []
        used_seed = seed
        for attempt in range(RESET_RETRIES):
            used_seed = seed + attempt
            try:
                self._setup_seed(used_seed)
                self._is_setup = True
                break
            except Exception as e:  # noqa: BLE001 — UnStableError or scene-build failure
                if first_exc is None:
                    first_exc = e
                attempt_errors.append(f"seed {used_seed}: {type(e).__name__}: {e}")
                logger.warning(
                    "RoboTwin setup_demo failed (attempt %d/%d, seed %d):\n%s",
                    attempt + 1,
                    RESET_RETRIES,
                    used_seed,
                    traceback.format_exc(),
                )
                self._safe_close()
        else:
            raise RuntimeError(
                f"RoboTwin setup_demo failed for task '{self.task_name}' after "
                f"{RESET_RETRIES} attempts (base seed {seed}).\n"
                f"First error: {type(first_exc).__name__}: {first_exc}\n"
                f"All attempts: {attempt_errors}"
            ) from first_exc
        self.task = self._fixed_instruction or _resolve_instruction(self.robotwin_root, self.task_name)
        with contextlib.suppress(Exception):
            self._task_env.set_instruction(instruction=self.task)
        return used_seed

    def _gated_setup(self, seed: int) -> int:
        """FastWAM expert-check seed gating (RoboTwin/script/eval_policy.py::eval_policy).

        Enumerate seeds from a persistent cursor (initialised to ``100000*(1+seed)``
        on the first reset). For each candidate, run the scripted oracle
        (``setup_demo`` -> ``play_once``) and accept the seed only if the oracle
        solves it (``plan_success and check_success()``); otherwise skip it. On
        acceptance, re-setup the scene for the policy rollout, sample a per-episode
        instruction from the oracle's descriptions, and resume the cursor past the
        accepted seed on the next reset (mirrors ``now_seed += 1`` after a rollout).
        """
        if self._now_seed is None:
            self._now_seed = SEED_BASE_MULTIPLIER * (1 + int(seed))

        episode_info: dict | None = None
        accepted: int | None = None
        for _ in range(MAX_GATE_CANDIDATES):
            cand = self._now_seed
            # --- expert check: does the scripted oracle solve this seed? ---
            try:
                self._setup_seed(cand)
                with _robotwin_ctx(self.robotwin_root):
                    episode_info = self._task_env.play_once()
                self._safe_close()
            except Exception:  # noqa: BLE001 — UnStableError or expert-rollout failure
                self._safe_close()
                self._now_seed += 1
                continue
            solved = bool(getattr(self._task_env, "plan_success", False)) and bool(
                self._task_env.check_success()
            )
            if not solved:
                self._now_seed += 1
                continue
            # --- accepted: re-build the same seed for the policy rollout ---
            try:
                self._setup_seed(cand)
            except Exception:  # noqa: BLE001 — passed expert check but failed re-init
                self._safe_close()
                self._now_seed += 1
                continue
            accepted = cand
            self._is_setup = True
            break

        if accepted is None:
            raise RuntimeError(
                f"No expert-solvable seed found for RoboTwin task '{self.task_name}' "
                f"within {MAX_GATE_CANDIDATES} candidates (cursor at {self._now_seed})."
            )

        self._now_seed = accepted + 1  # next reset resumes past this seed (eval_policy L409)
        self._set_episode_instruction(episode_info)
        return accepted

    def _set_episode_instruction(self, episode_info: dict | None) -> None:
        """Sample the per-episode instruction like FastWAM (falls back to static)."""
        if self._fixed_instruction is not None:
            self.task = self._fixed_instruction
        else:
            self.task = _resolve_instruction(self.robotwin_root, self.task_name)
            if episode_info is not None:
                try:
                    from generate_episode_instructions import generate_episode_descriptions

                    # 3rd arg = FastWAM's test_num (eval episode count); it only bounds
                    # the generated description-list length. The exact sampled string is
                    # not bit-reproducible vs FastWAM (global np-RNG state differs), so
                    # it only matters for language-conditioned policies.
                    results = generate_episode_descriptions(self.task_name, [episode_info["info"]], 100)
                    self.task = str(np.random.choice(results[0][self.instruction_type]))
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Faithful instruction sampling failed for %s (%s); using static.",
                        self.task_name,
                        e,
                    )
        with contextlib.suppress(Exception):
            self._task_env.set_instruction(instruction=self.task)

    def step(self, action: np.ndarray):
        if not self._is_setup:
            raise RuntimeError("step() called before reset()")
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] != ACTION_DIM:
            raise ValueError(f"Expected {ACTION_DIM}-D action, got shape {action.shape}")

        with _robotwin_ctx(self.robotwin_root):
            self._task_env.take_action(action, action_type="qpos")

        is_success = bool(getattr(self._task_env, "eval_success", False))
        take_action_cnt = int(getattr(self._task_env, "take_action_cnt", 0))
        terminated = is_success
        truncated = (take_action_cnt >= self._max_episode_steps) and not terminated
        reward = 1.0 if is_success else 0.0

        obs = self._get_obs()
        info = {"is_success": is_success, "task": self.task, "step": take_action_cnt}
        if terminated or truncated:
            info["final_info"] = {"is_success": is_success, "task": self.task}
        return obs, reward, terminated, truncated, info

    # -- observation ---------------------------------------------------------
    def _get_obs(self) -> dict[str, Any]:
        with _robotwin_ctx(self.robotwin_root):
            raw = self._task_env.get_obs()
        cams = raw["observation"]
        pixels = {
            "head_camera": np.ascontiguousarray(cams["head_camera"]["rgb"], dtype=np.uint8),
            "left_camera": np.ascontiguousarray(cams["left_camera"]["rgb"], dtype=np.uint8),
            "right_camera": np.ascontiguousarray(cams["right_camera"]["rgb"], dtype=np.uint8),
        }
        agent_pos = np.asarray(raw["joint_action"]["vector"], dtype=np.float64).reshape(-1)
        obs = {"pixels": pixels, "agent_pos": agent_pos}
        self._last_obs = obs
        return obs

    def render(self) -> np.ndarray:
        if self._last_obs is not None:
            return self._last_obs["pixels"]["head_camera"]
        return np.zeros((self._head_h, self._head_w, 3), dtype=np.uint8)

    def close(self):
        self._close_task()


# ---------------------------------------------------------------------------
# Vectorized env factory
# ---------------------------------------------------------------------------
def create_robotwin_envs(
    task: str,
    n_envs: int,
    robotwin_root: str | Path,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    episode_length: int | None = None,
) -> dict[str, dict[int, Any]]:
    """Create vectorized RoboTwin environments.

    Args:
        task: Comma-separated RoboTwin task names (e.g. ``"beat_block_hammer"``).
        n_envs: Number of parallel envs per task.
        robotwin_root: Path to the cloned RoboTwin repository.
        gym_kwargs: Extra kwargs forwarded to ``RoboTwinEnv`` (task_config, etc.).
        env_cls: Vectorized env class (e.g. ``gym.vector.SyncVectorEnv``).
        episode_length: Optional max-steps override.

    Returns:
        ``{task_name: {0: vec_env}}`` mapping.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be positive; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    if gym_kwargs.get("expert_check", True) and n_envs != 1:
        # FastWAM parity uses a single sequential seed cursor per task. With n_envs>1
        # each sub-env owns an independent cursor (a different seed stream) and
        # early-finishing sub-envs run the oracle gate for discarded episodes, so the
        # scored seed set is not FastWAM's. Require batch_size=1 (planning is bs=1
        # locked anyway); disable gating explicitly to opt out.
        raise ValueError(
            f"RoboTwin expert-check gating requires batch_size=1, got n_envs={n_envs}. "
            "Set --eval.batch_size=1, or pass --env.expert_check=false to disable gating."
        )
    task_names = [t.strip() for t in str(task).split(",") if t.strip()]
    if not task_names:
        raise ValueError("`task` must contain at least one RoboTwin task name.")

    print(f"Creating RoboTwin envs | tasks={task_names} | n_envs(per task)={n_envs}")

    out: dict[str, dict[int, Any]] = defaultdict(dict)
    for tname in task_names:
        fns = [
            partial(
                RoboTwinEnv,
                task_name=tname,
                robotwin_root=robotwin_root,
                max_episode_steps=episode_length,
                **gym_kwargs,
            )
            for _ in range(n_envs)
        ]
        out[tname][0] = env_cls(fns)
        print(f"  Built vec env | task={tname} | n_envs={n_envs}")

    return {t: dict(m) for t, m in out.items()}
