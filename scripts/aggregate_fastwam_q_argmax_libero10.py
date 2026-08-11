#!/usr/bin/env python
"""Strictly aggregate the two task-sharded LIBERO-10 rebuttal configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STEPS = (10, 20)
TASK_IDS = range(10)
EXPECTED_EPISODES_PER_TASK = 50


def _load_task(root: Path, steps: int, task_id: int) -> dict:
    result_path = root / f"steps{steps}" / f"task{task_id}" / "eval_info.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing task result: {result_path}")
    payload = json.loads(result_path.read_text())
    per_task = payload.get("per_task", [])
    if len(per_task) != 1 or per_task[0].get("task_id") != task_id:
        raise ValueError(
            f"{result_path} must contain exactly task_id={task_id}; got "
            f"{[item.get('task_id') for item in per_task]}"
        )
    metrics = per_task[0]["metrics"]
    successes = metrics.get("successes", [])
    if len(successes) != EXPECTED_EPISODES_PER_TASK:
        raise ValueError(
            f"{result_path} has {len(successes)} episodes; "
            f"expected {EXPECTED_EPISODES_PER_TASK}."
        )
    return {
        "task_id": task_id,
        "n_episodes": len(successes),
        "n_success": sum(bool(value) for value in successes),
        "pc_success": 100.0 * sum(bool(value) for value in successes) / len(successes),
        "successes": [bool(value) for value in successes],
    }


def aggregate(root: Path) -> dict:
    configurations = {}
    for steps in STEPS:
        per_task = [_load_task(root, steps, task_id) for task_id in TASK_IDS]
        n_episodes = sum(task["n_episodes"] for task in per_task)
        n_success = sum(task["n_success"] for task in per_task)
        if n_episodes != 500:
            raise ValueError(f"steps={steps} has {n_episodes} episodes; expected 500.")
        configurations[f"steps{steps}"] = {
            "diffusion_steps": steps,
            "n_samples": 64,
            "planner": "bc_diffusion_argmax",
            "n_episodes": n_episodes,
            "n_success": n_success,
            "pc_success": 100.0 * n_success / n_episodes,
            "per_task": per_task,
        }
    return {
        "protocol": {
            "suite": "libero_10",
            "tasks": 10,
            "episodes_per_task": 50,
            "episode_length": 700,
            "eval_batch_size": 1,
            "init_state_ids": list(range(50)),
        },
        "configurations": configurations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="OUTPUT_ROOT passed to the launcher")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Summary JSON path (default: ROOT/summary.json)",
    )
    args = parser.parse_args()
    summary = aggregate(args.root)
    output = args.output or args.root / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    for name, result in summary["configurations"].items():
        print(
            f"{name}: {result['n_success']}/{result['n_episodes']} "
            f"({result['pc_success']:.1f}%)"
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
