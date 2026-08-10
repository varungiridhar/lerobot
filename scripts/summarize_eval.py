#!/usr/bin/env python3
"""Summarize eval experiment directories into per-task summary.csv files.

An "experiment dir" holds one subdirectory per task, each a lerobot eval output
directory containing eval_info.json:

    outputs/eval/robotwin_bcdiff_s3/
        adjust_bottle/eval_info.json
        beat_block_hammer/eval_info.json
        ...

Writes <exp_dir>/summary.csv with one row per canonical task plus a final MEAN
row. Tasks with no data are emitted as nan so pandas/numpy skip them.

Usage:
    # One experiment
    python scripts/summarize_eval.py outputs/eval/robotwin_bcdiff_s3

    # Every robotwin experiment, plus a cross-experiment table
    python scripts/summarize_eval.py outputs/eval/robotwin_* --table

    # Table to CSV as well
    python scripts/summarize_eval.py outputs/eval/robotwin_* --table \
        --table-out outputs/eval/robotwin_overview.csv

Episode length is the video frame count (ffprobe), cached in
<exp_dir>/.ep_len_cache.json so reruns are cheap. Note that lerobot saves videos
for only the first N episodes, so length statistics are computed over that subset
while success rates use all episodes.
"""

import argparse
import concurrent.futures
import csv
import glob
import json
import math
import os
import subprocess
import sys

# Canonical RoboTwin 2.0 benchmark task list (50 tasks). Kept in sync with
# ALL_TASKS in scripts/eval_fastwam_q_robotwin.sh.
ROBOTWIN_TASKS = [
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller", "handover_block",
    "handover_mic", "hanging_mug", "lift_pot", "move_can_pot", "move_pillbottle_pad",
    "move_playingcard_away", "move_stapler_pad", "open_laptop", "open_microwave",
    "pick_diverse_bottles", "pick_dual_bottles", "place_a2b_left", "place_a2b_right",
    "place_bread_basket", "place_bread_skillet", "place_burger_fries", "place_can_basket",
    "place_cans_plasticbox", "place_container_plate", "place_dual_shoes", "place_empty_cup",
    "place_fan", "place_mouse_pad", "place_object_basket", "place_object_scale",
    "place_object_stand", "place_phone_stand", "place_shoe", "press_stapler",
    "put_bottles_dustbin", "put_object_cabinet", "rotate_qrcode", "scan_object",
    "shake_bottle", "shake_bottle_horizontally", "stack_blocks_three", "stack_blocks_two",
    "stack_bowls_three", "stack_bowls_two", "stamp_seal", "turn_switch",
]

TASK_LISTS = {"robotwin": ROBOTWIN_TASKS}

CACHE_NAME = ".ep_len_cache.json"

FIELDS = [
    "task", "n_episodes", "n_success", "success_rate",
    "mean_ep_len", "mean_ep_len_success", "mean_ep_len_fail", "n_videos",
]


def get_frame_count(video_path):
    """Frame count from container metadata, falling back to a full packet count."""
    for args in (
        ["-show_entries", "stream=nb_frames"],
        ["-count_packets", "-show_entries", "stream=nb_read_packets"],
    ):
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", *args, "-of", "csv=p=0", video_path],
                capture_output=True, text=True, timeout=30,
            )
            return int(r.stdout.strip())
        except (ValueError, subprocess.SubprocessError, OSError):
            continue
    return None


def episode_videos(task_dir):
    """Map episode index -> video path, scanning disk.

    eval_info.json stores absolute video_paths that go stale whenever a
    directory is moved, so paths are always rebuilt from the filesystem.
    """
    out = {}
    videos_dir = os.path.join(task_dir, "videos")
    if not os.path.isdir(videos_dir):
        return out
    for group in sorted(os.listdir(videos_dir)):
        gdir = os.path.join(videos_dir, group)
        if not os.path.isdir(gdir):
            continue
        for fname in os.listdir(gdir):
            if not fname.endswith(".mp4") or "planning" in fname:
                continue
            try:
                idx = int(fname.replace("eval_episode_", "").replace(".mp4", ""))
            except ValueError:
                continue
            out[idx] = os.path.join(gdir, fname)
    return out


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{x:.{nd}f}"


def collect_task(task_dir):
    """Read one task's eval_info.json -> (successes, {ep_idx: video_path})."""
    info_path = os.path.join(task_dir, "eval_info.json")
    if not os.path.exists(info_path):
        return None
    try:
        info = json.load(open(info_path))
    except (json.JSONDecodeError, OSError):
        return None
    successes = []
    for task in info.get("per_task", []):
        successes.extend(bool(s) for s in task["metrics"].get("successes", []))
    if not successes:
        return None
    return successes, episode_videos(task_dir)


def summarize_exp(exp_dir, task_list, jobs, refresh=False):
    """Write <exp_dir>/summary.csv. Returns an aggregate dict for the table."""
    cache_path = os.path.join(exp_dir, CACHE_NAME)
    cache = {}
    if os.path.exists(cache_path) and not refresh:
        try:
            cache = json.load(open(cache_path))
        except (json.JSONDecodeError, OSError):
            cache = {}

    collected = {}
    for task in sorted(os.listdir(exp_dir)):
        tdir = os.path.join(exp_dir, task)
        if os.path.isdir(tdir):
            got = collect_task(tdir)
            if got:
                collected[task] = got

    # Resolve any uncached frame counts in parallel (ffprobe is I/O bound).
    todo = [
        vp for _, vids in collected.values()
        for vp in vids.values()
        if os.path.relpath(vp, exp_dir) not in cache
    ]
    if todo:
        print(f"  probing {len(todo)} videos ...", end="", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
            for vp, n in zip(todo, ex.map(get_frame_count, todo)):
                cache[os.path.relpath(vp, exp_dir)] = n
        try:
            json.dump(cache, open(cache_path, "w"))
        except OSError:
            pass
        print(" done")

    # Union of canonical tasks and anything extra found on disk.
    tasks = list(task_list) + [t for t in sorted(collected) if t not in task_list]

    rows = []
    for task in tasks:
        if task not in collected:
            rows.append({
                "task": task, "n_episodes": 0, "n_success": 0, "success_rate": float("nan"),
                "mean_ep_len": float("nan"), "mean_ep_len_success": float("nan"),
                "mean_ep_len_fail": float("nan"), "n_videos": 0,
            })
            continue
        successes, vids = collected[task]
        lens, lens_s, lens_f = [], [], []
        for idx, vp in vids.items():
            n = cache.get(os.path.relpath(vp, exp_dir))
            if n is None:
                continue
            lens.append(n)
            if idx < len(successes):
                (lens_s if successes[idx] else lens_f).append(n)
        rows.append({
            "task": task,
            "n_episodes": len(successes),
            "n_success": sum(successes),
            "success_rate": 100.0 * sum(successes) / len(successes),
            "mean_ep_len": mean(lens),
            "mean_ep_len_success": mean(lens_s),
            "mean_ep_len_fail": mean(lens_f),
            "n_videos": len(lens),
        })

    present = [r for r in rows if r["n_episodes"] > 0]
    tot_eps = sum(r["n_episodes"] for r in present)
    tot_succ = sum(r["n_success"] for r in present)
    mean_row = {
        "task": "MEAN",
        "n_episodes": tot_eps,
        "n_success": tot_succ,
        "success_rate": mean([r["success_rate"] for r in present]),
        "mean_ep_len": mean([r["mean_ep_len"] for r in present]),
        "mean_ep_len_success": mean([r["mean_ep_len_success"] for r in present]),
        "mean_ep_len_fail": mean([r["mean_ep_len_fail"] for r in present]),
        "n_videos": sum(r["n_videos"] for r in present),
    }
    rows.append(mean_row)

    out_path = os.path.join(exp_dir, "summary.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({
                k: (fmt(r[k], 1) if k in ("success_rate", "mean_ep_len",
                                          "mean_ep_len_success", "mean_ep_len_fail") else r[k])
                for k in FIELDS
            })

    micro = 100.0 * tot_succ / tot_eps if tot_eps else float("nan")
    print(f"  {os.path.basename(exp_dir):32s} {len(present):2d}/{len(task_list)} tasks  "
          f"macro={fmt(mean_row['success_rate'])}%  micro={fmt(micro)}%  -> {out_path}")

    return {
        "experiment": os.path.basename(exp_dir),
        "n_tasks": len(present),
        "n_episodes": tot_eps,
        "n_success": tot_succ,
        "success_rate_macro": mean_row["success_rate"],
        "success_rate_micro": micro,
        "mean_ep_len": mean_row["mean_ep_len"],
        "mean_ep_len_success": mean_row["mean_ep_len_success"],
        "mean_ep_len_fail": mean_row["mean_ep_len_fail"],
    }


def print_table(aggs, out_path=None):
    hdr = (f"{'experiment':34s} {'tasks':>5s} {'succ/eps':>12s} {'macro%':>7s} "
           f"{'micro%':>7s} {'ep_len':>8s} {'len_ok':>8s} {'len_fail':>9s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for a in aggs:
        print(f"{a['experiment']:34s} {a['n_tasks']:5d} "
              f"{a['n_success']:5d}/{a['n_episodes']:6d} "
              f"{fmt(a['success_rate_macro']):>7s} {fmt(a['success_rate_micro']):>7s} "
              f"{fmt(a['mean_ep_len']):>8s} {fmt(a['mean_ep_len_success']):>8s} "
              f"{fmt(a['mean_ep_len_fail']):>9s}")
    if out_path:
        cols = ["experiment", "n_tasks", "n_episodes", "n_success", "success_rate_macro",
                "success_rate_micro", "mean_ep_len", "mean_ep_len_success", "mean_ep_len_fail"]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for a in aggs:
                w.writerow({c: (fmt(a[c], 1) if isinstance(a[c], float) else a[c]) for c in cols})
        print(f"\nWrote table -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp_dirs", nargs="+", help="Experiment directories (each holds per-task subdirs)")
    p.add_argument("--env", default="robotwin", choices=sorted(TASK_LISTS),
                   help="Canonical task list to pad missing tasks against")
    p.add_argument("--table", action="store_true", help="Print a cross-experiment overview table")
    p.add_argument("--table-out", default=None, help="Also write the overview table to this CSV")
    p.add_argument("--jobs", type=int, default=8, help="Parallel ffprobe workers")
    p.add_argument("--refresh", action="store_true", help="Ignore cached frame counts and re-probe")
    args = p.parse_args()

    task_list = TASK_LISTS[args.env]
    exp_dirs = [d.rstrip("/") for pat in args.exp_dirs for d in sorted(glob.glob(pat))]
    exp_dirs = [d for d in exp_dirs if os.path.isdir(d)]
    if not exp_dirs:
        sys.exit("No experiment directories matched.")

    aggs = []
    for d in exp_dirs:
        agg = summarize_exp(d, task_list, args.jobs, args.refresh)
        if agg and agg["n_episodes"]:
            aggs.append(agg)

    if args.table or args.table_out:
        aggs.sort(key=lambda a: (-a["success_rate_macro"] if not math.isnan(a["success_rate_macro"]) else 0))
        print_table(aggs, args.table_out)


if __name__ == "__main__":
    main()
