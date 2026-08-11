#!/usr/bin/env python
"""GPU-native stress smoke for LIBERO's offscreen EGL lifecycle.

Run this on a GPU node. It deliberately avoids loading a policy so failures
isolate environment reset, camera rendering, and teardown from FastWAM / TD3.
"""

from __future__ import annotations

import argparse
import os


# These must be selected before importing Robosuite / PyOpenGL.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
from libero.libero import benchmark  # noqa: E402

from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action  # noqa: E402


def _validate_pixels(observation, *, height: int, width: int) -> None:
    pixels = observation["pixels"]
    if set(pixels) != {"image", "image2"}:
        raise RuntimeError(f"Unexpected LIBERO camera keys: {sorted(pixels)}")
    for name, image in pixels.items():
        if image.shape != (height, width, 3):
            raise RuntimeError(f"{name} has shape {image.shape}, expected {(height, width, 3)}")
        if image.dtype != np.uint8:
            raise RuntimeError(f"{name} has dtype {image.dtype}, expected uint8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--resets", type=int, default=100)
    parser.add_argument("--steps-per-reset", type=int, default=32)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resets <= 0 or args.steps_per_reset <= 0:
        raise ValueError("resets and steps-per-reset must be positive")

    suites = benchmark.get_benchmark_dict()
    if args.suite not in suites:
        raise ValueError(f"Unknown LIBERO suite {args.suite!r}")
    suite = suites[args.suite]()
    if not 0 <= args.task_id < len(suite.tasks):
        raise ValueError(f"task-id must lie in [0, {len(suite.tasks) - 1}]")

    env = LiberoEnv(
        task_suite=suite,
        task_id=args.task_id,
        task_suite_name=args.suite,
        episode_length=args.steps_per_reset,
        observation_height=args.height,
        observation_width=args.width,
    )
    action = np.asarray(get_libero_dummy_action(), dtype=np.float32)
    rendered_steps = 0
    try:
        for reset_index in range(args.resets):
            observation, info = env.reset(
                seed=args.seed + reset_index,
                options={"episode_index_offset": reset_index},
            )
            _validate_pixels(observation, height=args.height, width=args.width)
            if info["init_state_id"] != reset_index % len(env._init_states):
                raise RuntimeError("LIBERO reset selected the wrong initial state")

            for _ in range(args.steps_per_reset):
                observation, _, _, _, _ = env.step(action)
                _validate_pixels(observation, height=args.height, width=args.width)
                rendered_steps += 1

            if (reset_index + 1) % 10 == 0:
                print(
                    f"EGL stress progress: resets={reset_index + 1}/{args.resets} "
                    f"camera_steps={rendered_steps}",
                    flush=True,
                )
    finally:
        env.close()

    print(
        f"EGL stress PASS: suite={args.suite} task={args.task_id} "
        f"resets={args.resets} camera_steps={rendered_steps}",
        flush=True,
    )


if __name__ == "__main__":
    main()
