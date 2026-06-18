#!/usr/bin/env python3
"""Export per-episode eval results to CSV from a lerobot eval output directory.

Usage:
    python scripts/export_eval_csv.py <eval_dir> [--out <path.csv>] [--env {libero,robotwin}]

Columns: task_id, task_name, episode, success, episode_length

For complete runs: success is read from eval_info.json (ground truth).
For preempted/partial runs: success is inferred from episode_length
  - Libero: episodes that hit max_frames (520) are failures; shorter = success
  - RoboTwin: episodes that hit max_frames (1500@50fps) are failures

Episode length is always derived from video frame count (ffprobe).
"""

import argparse
import csv
import json
import os
import subprocess

# Libero-10 task names (task_id → short name)
LIBERO_10_TASK_NAMES = {
    0: "put_alphabet_soup_and_tomato_sauce_in_basket",
    1: "put_cream_cheese_and_butter_in_basket",
    2: "turn_on_stove_and_put_moka_pot",
    3: "put_black_bowl_in_bottom_drawer_and_close",
    4: "put_white_mug_left_plate_yellow_mug_right_plate",
    5: "pick_up_book_and_place_in_caddy",
    6: "put_white_mug_on_plate_and_chocolate_pudding_right",
    7: "put_alphabet_soup_and_cream_cheese_in_basket",
    8: "put_both_moka_pots_on_stove",
    9: "put_yellow_white_mug_in_microwave_and_close",
}

# Max episode frames (failure = hit this limit)
MAX_FRAMES = {
    "libero": 520,    # 600 steps × ~0.87 (some tasks shorter), empirically all failures = 520
    "robotwin": 1500, # 1500 steps @ 50fps (30s episodes)
}


def get_frame_count(video_path: str) -> int | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=15,
        )
        return int(r.stdout.strip())
    except Exception:
        return None


def get_task_name(task_group: str, task_id: int, env: str) -> str:
    if env == "libero" and "libero_10" in task_group:
        return LIBERO_10_TASK_NAMES.get(task_id, f"{task_group}_{task_id}")
    return f"{task_group}_{task_id}"


def export_csv(eval_dir: str, out_path: str, env: str) -> None:
    max_frames = MAX_FRAMES.get(env, 520)
    info_path = os.path.join(eval_dir, "eval_info.json")
    rows = []

    if os.path.exists(info_path):
        # Complete run — ground truth successes from eval_info.json
        # Iterate over successes array (not videos) so all episodes are exported.
        # Episode length is filled from video if available, else None.
        info = json.load(open(info_path))
        for task in info["per_task"]:
            tid = task["task_id"]
            tgroup = task["task_group"]
            task_name = get_task_name(tgroup, tid, env)
            successes = task["metrics"]["successes"]
            vdir = os.path.join(eval_dir, "videos", f"{tgroup}_{tid}")
            # Build video index: episode_idx -> frame_count (only for saved videos)
            video_lengths = {}
            if os.path.exists(vdir):
                eps = sorted([f for f in os.listdir(vdir) if f.endswith(".mp4") and "planning" not in f])
                for ep_file in eps:
                    # filename: eval_episode_N.mp4
                    try:
                        ep_idx = int(ep_file.replace("eval_episode_", "").replace(".mp4", ""))
                    except ValueError:
                        continue
                    video_lengths[ep_idx] = get_frame_count(os.path.join(vdir, ep_file))
            for ep_idx, success in enumerate(successes):
                rows.append({
                    "task_id": tid,
                    "task_name": task_name,
                    "episode": ep_idx,
                    "success": success,
                    "episode_length": video_lengths.get(ep_idx),
                })
    else:
        # Partial / preempted run — infer success from episode length
        videos_dir = os.path.join(eval_dir, "videos")
        if not os.path.exists(videos_dir):
            raise FileNotFoundError(f"No eval_info.json and no videos/ dir in {eval_dir}")
        for task_dir in sorted(os.listdir(videos_dir)):
            vdir = os.path.join(videos_dir, task_dir)
            if not os.path.isdir(vdir):
                continue
            # Parse task_group and task_id from dir name, e.g. libero_10_3 or beat_block_hammer_0
            parts = task_dir.rsplit("_", 1)
            tgroup, tid = parts[0], int(parts[1])
            task_name = get_task_name(tgroup, tid, env)
            eps = sorted([f for f in os.listdir(vdir) if f.endswith(".mp4") and "planning" not in f])
            for ep_idx, ep_file in enumerate(eps):
                fl = get_frame_count(os.path.join(vdir, ep_file))
                success = (fl < max_frames) if fl is not None else None
                rows.append({
                    "task_id": tid,
                    "task_name": task_name,
                    "episode": ep_idx,
                    "success": success,
                    "episode_length": fl,
                })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "task_name", "episode", "success", "episode_length"])
        writer.writeheader()
        writer.writerows(rows)

    n_s = sum(1 for r in rows if r["success"] is True)
    n_total = len(rows)
    print(f"Wrote {n_total} rows → {out_path}  ({n_s}/{n_total} = {n_s/n_total*100:.1f}% success)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dir", help="Path to lerobot eval output directory")
    parser.add_argument("--out", default=None, help="Output CSV path (default: <eval_dir_name>.csv next to eval_dir)")
    parser.add_argument("--env", default="libero", choices=["libero", "robotwin"],
                        help="Environment type (affects task name lookup and max_frames threshold)")
    args = parser.parse_args()

    eval_dir = args.eval_dir.rstrip("/")
    out = args.out or os.path.join(os.path.dirname(eval_dir), os.path.basename(eval_dir) + ".csv")
    export_csv(eval_dir, out, args.env)


if __name__ == "__main__":
    main()
