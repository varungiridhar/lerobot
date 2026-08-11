#!/usr/bin/env python
"""Validate and summarize a sharded policy-only RoboTwin evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def task_result(path: Path, expected_episodes: int) -> dict:
    info = json.loads(path.read_text())
    per_task = info.get("per_task", [])
    if len(per_task) != 1:
        raise ValueError(f"Expected exactly one task in {path}, found {len(per_task)}")
    record = per_task[0]
    successes = [bool(value) for value in record["metrics"]["successes"]]
    if len(successes) != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} episodes in {path}, found {len(successes)}"
        )
    return {
        "n_episodes": len(successes),
        "n_success": sum(successes),
        "pc_success": 100.0 * sum(successes) / len(successes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--shard", required=True, help="K/N")
    args = parser.parse_args()

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    shard_index, n_shards = (int(value) for value in args.shard.split("/", maxsplit=1))
    selected = [task for index, task in enumerate(tasks) if index % n_shards == shard_index]

    per_task = {}
    missing = []
    invalid = {}
    for task in selected:
        path = args.eval_root / task / "eval_info.json"
        if not path.is_file():
            missing.append(task)
            continue
        try:
            per_task[task] = task_result(path, args.episodes)
        except Exception as error:
            invalid[task] = f"{type(error).__name__}: {error}"

    shard_payload = {
        "shard": args.shard,
        "complete": not missing and not invalid,
        "expected_tasks": selected,
        "per_task": per_task,
        "missing_tasks": missing,
        "invalid_tasks": invalid,
    }
    write_json_atomic(args.eval_root / f"eval_shard{shard_index:02d}.json", shard_payload)
    if missing or invalid:
        raise RuntimeError(f"Incomplete eval shard {args.shard}: missing={missing}, invalid={invalid}")

    shard_paths = [args.eval_root / f"eval_shard{index:02d}.json" for index in range(n_shards)]
    if not all(path.is_file() for path in shard_paths):
        print(f"Eval shard {args.shard} complete; waiting for the other shard manifests.")
        return

    shard_payloads = [json.loads(path.read_text()) for path in shard_paths]
    if not all(payload.get("complete") for payload in shard_payloads):
        print("All eval shard manifests exist, but at least one is incomplete.")
        return

    combined = {}
    for payload in shard_payloads:
        overlap = set(combined).intersection(payload["per_task"])
        if overlap:
            raise RuntimeError(f"Tasks duplicated across eval shards: {sorted(overlap)}")
        combined.update(payload["per_task"])
    if set(combined) != set(tasks):
        raise RuntimeError(
            f"Eval task coverage mismatch: missing={sorted(set(tasks) - set(combined))}, "
            f"extra={sorted(set(combined) - set(tasks))}"
        )

    total_episodes = sum(result["n_episodes"] for result in combined.values())
    total_success = sum(result["n_success"] for result in combined.values())
    summary = {
        "complete": True,
        "n_tasks": len(combined),
        "episodes_per_task": args.episodes,
        "n_episodes": total_episodes,
        "n_success": total_success,
        "pc_success": 100.0 * total_success / total_episodes,
        "per_task": {task: combined[task] for task in tasks},
    }
    write_json_atomic(args.eval_root / "summary.json", summary)
    print(
        f"RoboTwin BC eval complete: {total_success}/{total_episodes} "
        f"({summary['pc_success']:.2f}%) over {len(combined)} tasks"
    )


if __name__ == "__main__":
    main()
