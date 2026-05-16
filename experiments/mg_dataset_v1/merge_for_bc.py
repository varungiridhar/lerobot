"""Merge successful demos from multiple buckets into one HDF5 for BC training.

User spec: "BC will train on all successful episodes from all buckets."

We treat the BC-training input as `<task>/q5/demo.hdf5` ∪ `<task>/q3_termjitter/demo.hdf5`
(per-bucket success files only). Failed demos and play data are excluded.
Resulting file lives at:
  <dataset_root>/<task>/bc_input/demo.hdf5

Each demo group is renamed `demo_<i>` with `i` reassigned globally so
convert_mimicgen_to_lerobot.py can iterate. The original `data` attrs of the
first source HDF5 (env_args, etc.) are copied over so robomimic-aware tools
keep working.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py

DEFAULT_ROOT = Path(
    "/storage/home/hcoda1/6/vgiridhar6/shared/mimicgen/lerobot_datasets/v1_q5_q3jitter_play"
)
DEFAULT_BUCKETS = ("q5", "q3_termjitter")


def _copy_demo(src_grp: h5py.Group, dst_grp: h5py.Group, bucket: str):
    """Deep-copy a single demo group; tag with `bucket` attr for downstream use."""
    src_grp.copy(source=".", dest=dst_grp)
    dst_grp.attrs["bucket"] = bucket


def merge_one_task(task_dir: Path, buckets=DEFAULT_BUCKETS):
    out_dir = task_dir / "bc_input"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "demo.hdf5"
    if out_path.exists():
        out_path.unlink()

    source_files: list[tuple[Path, str]] = []
    for b in buckets:
        # MG output convention: <task>/<bucket>/<bucket>/demo.hdf5
        # play output convention: <task>/play/demo.hdf5 (no double-nesting)
        candidates = [
            task_dir / b / b / "demo.hdf5",      # MG-generated
            task_dir / b / "demo.hdf5",          # play-rollout-generated
        ]
        for c in candidates:
            if c.exists():
                source_files.append((c, b))
                break

    if not source_files:
        raise SystemExit(f"No source success-HDF5s found under {task_dir}")

    n_total = 0
    with h5py.File(out_path, "w") as out:
        # Copy top-level data attrs from the first source file.
        with h5py.File(source_files[0][0], "r") as f0:
            data_grp = out.create_group("data")
            for k, v in f0["data"].attrs.items():
                data_grp.attrs[k] = v

        for src_path, bucket in source_files:
            with h5py.File(src_path, "r") as f:
                # Sort numerically.
                keys = sorted(
                    [k for k in f["data"].keys() if k.startswith("demo")],
                    key=lambda x: int(x.split("_")[1]),
                )
                for k in keys:
                    new_name = f"demo_{n_total}"
                    src_demo = f["data"][k]
                    src_demo.copy(source=".", dest=out["data"], name=new_name)
                    out["data"][new_name].attrs["bucket"] = bucket
                    n_total += 1
            print(f"  merged {len(keys):3d} demos from {src_path}  (bucket={bucket})", flush=True)

        out["data"].attrs["total"] = n_total
    print(f"-> {out_path}  ({n_total} demos total)", flush=True)
    return out_path, n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--task", required=True, help="square | threading | coffee")
    ap.add_argument("--buckets", nargs="+", default=list(DEFAULT_BUCKETS))
    args = ap.parse_args()

    task_dir = args.root / args.task
    if not task_dir.exists():
        raise SystemExit(f"missing task dir: {task_dir}")
    print(f"Merging buckets {args.buckets} for task '{args.task}' ...", flush=True)
    merge_one_task(task_dir, args.buckets)


if __name__ == "__main__":
    main()
