#!/usr/bin/env python
"""Derive a right-arm-only (7-dim) LeRobot v3.0 dataset from a bimanual YAM dataset.

Pipeline (all episodes kept, episode indices unchanged, videos copied byte-exact):
  1. `remove_feature` drops the left_wrist camera (videos of kept cameras are
     shutil.copy'd, never re-encoded).
  2. A slice pass rewrites `action` / `observation.state` from 14 dims to the
     right-arm dims [7:14] everywhere they appear: data parquets, per-episode
     stats inside meta/episodes parquets, meta/stats.json, meta/info.json
     (shape + names). Stale left_wrist stats/video columns are dropped from the
     episodes parquets. Slicing aggregate stats elementwise is exact — no
     recompute needed.
  3. meta/episode_labels.json (leLab success sidecar) is copied over — the Q
     reward pipeline depends on it and no lerobot tool carries it.
  4. Validation: reload through LeRobotDataset, compare sliced vectors against
     the source, decode a frame from each kept camera.
  5. Optional push to the Hub under --output_repo_id.

Parameterized by dataset id so the second task's dataset can be processed
identically:

  conda run -n lerobot python scripts/derive_right_arm_dataset.py \
      --source_repo_id VarunGiridhar3/KT_stack_cups_20260807_211827 --push
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SLICE = slice(7, 14)  # right arm dims of the bimanual layout
DROP_CAMERA = "observation.images.left_wrist"
KEEP_CAMERAS = ("observation.images.right_wrist", "observation.images.top")
SLICED_FEATURES = ("action", "observation.state")


def slice_names(names: list[str]) -> list[str]:
    sliced = names[SLICE]
    assert all(n.startswith("right_") for n in sliced), f"unexpected dim names: {sliced}"
    return sliced


def rename_camera(root: Path, src_key: str, dst_key: str) -> None:
    """Rename a camera feature everywhere it appears (rig name -> dataset name).

    leLab rollout recordings use the rig's camera names (e.g. ``head``); training
    and Q checkpoints use the demo dataset's names (``top``). Applied after
    remove_feature, before the dim slice.
    """
    video_dir = root / "videos" / src_key
    assert video_dir.is_dir(), f"{video_dir} missing"
    video_dir.rename(root / "videos" / dst_key)

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    assert src_key in info["features"] and dst_key not in info["features"]
    info["features"] = {dst_key if k == src_key else k: v for k, v in info["features"].items()}
    info_path.write_text(json.dumps(info, indent=4))

    for path in sorted(root.glob("meta/episodes/chunk-*/file-*.parquet")):
        df = pd.read_parquet(path)
        renames = {c: c.replace(src_key, dst_key) for c in df.columns if src_key in c}
        if renames:
            df = df.rename(columns=renames)
            df.to_parquet(path)

    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    if src_key in stats:
        stats[dst_key] = stats.pop(src_key)
        stats_path.write_text(json.dumps(stats, indent=4))


def rewrite_info(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    assert DROP_CAMERA not in info["features"], "remove_feature should have dropped left_wrist"
    for feat in SLICED_FEATURES:
        entry = info["features"][feat]
        assert entry["shape"] == [14], f"{feat} shape {entry['shape']} != [14]"
        entry["shape"] = [7]
        entry["names"] = slice_names(entry["names"])
    info_path.write_text(json.dumps(info, indent=4))
    return info


def rewrite_data_parquets(root: Path, meta) -> int:
    from lerobot.datasets.dataset_tools import _write_parquet

    n = 0
    for path in sorted(root.glob("data/chunk-*/file-*.parquet")):
        df = pd.read_parquet(path)
        for feat in SLICED_FEATURES:
            df[feat] = df[feat].map(lambda v: np.asarray(v, dtype=np.float32)[SLICE])
        _write_parquet(df, path, meta)
        n += len(df)
    return n


def rewrite_episodes_parquets(root: Path) -> None:
    for path in sorted(root.glob("meta/episodes/chunk-*/file-*.parquet")):
        df = pd.read_parquet(path)
        stale = [c for c in df.columns if DROP_CAMERA in c]
        if stale:
            df = df.drop(columns=stale)
        for col in df.columns:
            if any(col.startswith(f"stats/{feat}/") for feat in SLICED_FEATURES):
                if col.endswith("/count"):
                    continue
                df[col] = df[col].map(lambda v: np.asarray(v)[SLICE])
        df.to_parquet(path)


def rewrite_stats_json(root: Path) -> None:
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    for feat in SLICED_FEATURES:
        for stat, val in stats[feat].items():
            arr = np.asarray(val)
            if arr.ndim >= 1 and arr.shape[0] == 14:
                stats[feat][stat] = arr[SLICE].tolist()
    assert DROP_CAMERA not in stats
    stats_path.write_text(json.dumps(stats, indent=4))


def validate(out_root: Path, out_repo_id: str, src_root: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(out_repo_id, root=out_root)
    assert ds.meta.total_episodes > 0
    src_df = pd.read_parquet(sorted(src_root.glob("data/chunk-*/file-*.parquet"))[0])

    item = ds[0]
    assert item["action"].shape == (7,), item["action"].shape
    assert item["observation.state"].shape == (7,), item["observation.state"].shape
    np.testing.assert_allclose(
        item["action"].numpy(), np.asarray(src_df.iloc[0]["action"], dtype=np.float32)[SLICE], rtol=1e-6
    )
    for cam in KEEP_CAMERAS:
        assert cam in item, f"{cam} missing"
        assert item[cam].shape[-2:] == (480, 640), item[cam].shape
    assert DROP_CAMERA not in item
    assert np.asarray(ds.meta.stats["action"]["mean"]).shape == (7,)
    assert np.asarray(ds.meta.stats["action"]["q01"]).shape == (7,)
    names = ds.meta.features["action"]["names"]
    assert names[-1] == "right_gripper.pos" and len(names) == 7, names
    labels = json.loads((out_root / "meta" / "episode_labels.json").read_text())
    assert len(labels) == ds.meta.total_episodes
    print(f"validation OK: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames, "
          f"{sum(1 for v in labels.values() if v['success'])} successes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_repo_id", required=True)
    parser.add_argument("--source_root", default=None,
                        help="local dataset dir (skips snapshot_download)")
    parser.add_argument("--output_repo_id", default=None,
                        help="default: <source_repo_id>_right")
    parser.add_argument("--work_dir", default="/storage/scratch1/6/vgiridhar6/lerobot_qplanning_real")
    parser.add_argument("--rename_camera", default=None, metavar="SRC=DST",
                        help="e.g. observation.images.head=observation.images.top")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    out_repo_id = args.output_repo_id or f"{args.source_repo_id}_right"
    if args.source_root:
        src_root = Path(args.source_root)
    else:
        from huggingface_hub import snapshot_download

        src_root = Path(snapshot_download(args.source_repo_id, repo_type="dataset"))
    out_root = Path(args.work_dir) / out_repo_id.replace("/", "__")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.parent.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.dataset_tools import remove_feature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    print(f"stage A: remove {DROP_CAMERA}")
    src_ds = LeRobotDataset(args.source_repo_id, root=src_root)
    remove_feature(src_ds, DROP_CAMERA, output_dir=out_root, repo_id=out_repo_id)

    if args.rename_camera:
        src_key, dst_key = args.rename_camera.split("=")
        print(f"stage A': rename camera {src_key} -> {dst_key}")
        rename_camera(out_root, src_key, dst_key)

    print("stage B: slice 14 -> 7")
    rewrite_info(out_root)
    meta = LeRobotDatasetMetadata(out_repo_id, root=out_root)
    n = rewrite_data_parquets(out_root, meta)
    print(f"  rewrote {n} frames")
    rewrite_episodes_parquets(out_root)
    rewrite_stats_json(out_root)
    shutil.copy(src_root / "meta" / "episode_labels.json", out_root / "meta" / "episode_labels.json")

    print("stage C: validate")
    validate(out_root, out_repo_id, src_root)

    if args.push:
        print(f"pushing to {out_repo_id}")
        ds = LeRobotDataset(out_repo_id, root=out_root)
        ds.push_to_hub()
        print("pushed")


if __name__ == "__main__":
    main()
