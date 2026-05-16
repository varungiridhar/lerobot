"""Convert the v1 robomimic-format buckets into 9 LeRobotDatasets (3 tasks × 3 buckets).

Why this exists
===============
``lerobot-train`` consumes ``LeRobotDataset`` (Parquet+meta), but our v1
``shared/mimicgen/lerobot_datasets/v1_q5_q3jitter_play/`` root is in
**robomimic HDF5** format under the misleading name. To drive Q training via
the standard LeRobot CLI we materialise a sibling root in proper LeRobotDataset
format::

    /storage/.../shared/mimicgen/lerobot_datasets/v1_lerobot/
        mg_coffee_q5/
        mg_coffee_q3_termjitter/
        mg_coffee_play/
        mg_threading_q5/
        ...                  (9 dirs total, one per (task, bucket))

Source HDF5 picked per bucket
-----------------------------
* q5, q3_termjitter      →  ``<task>/<bucket>/<bucket>/demo.hdf5``
                            (success-only by construction; failed demos in
                             ``demo_failed.hdf5`` are intentionally NOT converted)
* play                   →  ``<task>/play/demo_failed.hdf5``
                            (the only play file: BC succeeded 0/200)

Why ``--no_videos``
-------------------
The Q-function precaches features through a frozen ResNet18 backbone copied
from a trained act_simple policy. If we store mp4 here, decode noise drifts
the cached features away from what the BC backbone actually consumed at
training time. PNG keeps the cache pixel-faithful at ~3GB total — cheap.

Usage
-----
::

    # Sequential (~1.5h total):
    python -u experiments/mg_dataset_v1/convert_v1_to_lerobot.py \\
        --src_root /storage/.../shared/mimicgen/lerobot_datasets/v1_q5_q3jitter_play \\
        --dst_root /storage/.../shared/mimicgen/lerobot_datasets/v1_lerobot

    # One task at a time (parallelisable across SLURM jobs):
    python -u experiments/mg_dataset_v1/convert_v1_to_lerobot.py \\
        --src_root ... --dst_root ... --task coffee
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the converter importable when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from convert_mimicgen_to_lerobot import convert_mimicgen_to_lerobot  # noqa: E402

TASKS = ("square", "threading", "coffee")
BUCKETS = ("q5", "q3_termjitter", "play")


def source_hdf5_for(task: str, bucket: str, src_root: Path) -> Path:
    """Map (task, bucket) to the source HDF5 file we should convert."""
    if bucket == "play":
        return src_root / task / "play" / "demo_failed.hdf5"
    # q5 / q3_termjitter: success demos live one level deeper because of the
    # MimicGen output convention (<bucket>/<bucket>/demo.hdf5).
    return src_root / task / bucket / bucket / "demo.hdf5"


def repo_id_for(task: str, bucket: str) -> str:
    return f"mg_{task}_{bucket}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, type=Path)
    ap.add_argument("--dst_root", required=True, type=Path)
    ap.add_argument("--task", choices=TASKS, default=None,
                    help="restrict to one task; omit to convert all 3")
    ap.add_argument("--bucket", choices=BUCKETS, default=None,
                    help="restrict to one bucket; omit to convert all 3")
    ap.add_argument("--max_episodes", type=int, default=None,
                    help="cap demos per dataset (smoke testing)")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    tasks = (args.task,) if args.task else TASKS
    buckets = (args.bucket,) if args.bucket else BUCKETS

    args.dst_root.mkdir(parents=True, exist_ok=True)

    plan: list[tuple[str, str, Path, str, Path]] = []
    for task in tasks:
        for bucket in buckets:
            src = source_hdf5_for(task, bucket, args.src_root)
            repo_id = repo_id_for(task, bucket)
            dst = args.dst_root / repo_id
            if not src.exists():
                print(f"[skip] missing source HDF5: {src}")
                continue
            plan.append((task, bucket, src, repo_id, dst))

    if not plan:
        raise SystemExit("nothing to convert; check --src_root and --task/--bucket")

    print("plan:")
    for task, bucket, src, repo_id, dst in plan:
        print(f"  {repo_id:30s}  src={src}  dst={dst}")

    t_global = time.time()
    for task, bucket, src, repo_id, dst in plan:
        print(f"\n=== converting {repo_id} ===")
        t0 = time.time()
        convert_mimicgen_to_lerobot(
            hdf5_path=src,
            repo_id=repo_id,
            root=dst,
            max_episodes=args.max_episodes,
            fps=args.fps,
            use_videos=False,         # PNG; see module docstring
            save_init_states=False,   # not needed for Q training
            force=True,               # we own dst_root
        )
        print(f"[{repo_id}] done in {time.time() - t0:.1f}s")

    print(f"\nall done. total elapsed: {time.time() - t_global:.1f}s. dst_root={args.dst_root}")


if __name__ == "__main__":
    main()
