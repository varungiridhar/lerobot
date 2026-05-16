"""Build side-by-side composite mp4s for episodes where baseline and planning disagree.

Reads ``eval_info.json`` from two ``lerobot-eval`` runs (baseline + planning) at
matched N. For each episode where ``baseline.successes[i] != planning.successes[i]``,
renders a side-by-side mp4 (baseline left, planning right). Frames are read with
``imageio`` (ffmpeg under the hood, already shipped via the ``imageio_ffmpeg``
package this env uses for matplotlib video output), so no extra dependencies.

Outputs split into two subdirs:
    ``planning_recovers/`` — baseline ✗, planning ✓  (planner saves a failure)
    ``planning_regresses/`` — baseline ✓, planning ✗ (planner breaks a success)

Each composite pads the shorter clip by repeating its last frame (success usually
terminates earlier than failure's truncation, so the success side freezes while
the failure side plays out its full max-steps truncation).

Usage::

    python -u experiments/mg_dataset_v1/composite_planning_wins.py \\
        --baseline_dir outputs/eval/q_planning_lerobot_eval/<baseline_run> \\
        --planning_dir outputs/eval/q_planning_lerobot_eval/<planning_run> \\
        --out_dir     outputs/eval/q_planning_lerobot_eval/<comparison_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def _success_vec(eval_info_path: Path) -> list[bool]:
    info = json.loads(eval_info_path.read_text())
    out: list[bool] = []
    for task in info["per_task"]:
        out.extend(bool(x) for x in task["metrics"]["successes"])
    return out


def _episode_video(run_dir: Path, ep_idx: int) -> Path:
    """Resolve the mp4 path for one episode, regardless of the task_group_id folder name."""
    matches = list(run_dir.glob(f"videos/*/eval_episode_{ep_idx}.mp4"))
    if not matches:
        raise FileNotFoundError(f"no video for ep {ep_idx} under {run_dir / 'videos'}")
    if len(matches) > 1:
        raise RuntimeError(f"multiple videos matched ep {ep_idx}: {matches}")
    return matches[0]


def _read_frames(path: Path) -> tuple[np.ndarray, float]:
    """Returns (frames, fps). frames shape: (T, H, W, 3) uint8."""
    frames = iio.imread(path, plugin="pyav")
    if frames.ndim != 4:
        raise RuntimeError(f"expected (T,H,W,3) from {path}, got {frames.shape}")
    meta = iio.immeta(path, plugin="pyav")
    fps = float(meta.get("fps", 20.0))
    return frames, fps


def _hstack_pad(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pad shorter clip by repeating its last frame; return (T_max, H, W*2, 3)."""
    tl, tr = left.shape[0], right.shape[0]
    if left.shape[1:] != right.shape[1:]:
        raise RuntimeError(f"frame shapes differ: {left.shape[1:]} vs {right.shape[1:]}")
    t_max = max(tl, tr)
    if tl < t_max:
        pad = np.repeat(left[-1:], t_max - tl, axis=0)
        left = np.concatenate([left, pad], axis=0)
    if tr < t_max:
        pad = np.repeat(right[-1:], t_max - tr, axis=0)
        right = np.concatenate([right, pad], axis=0)
    return np.concatenate([left, right], axis=2)  # hstack along width


def _write_composite(out_path: Path, frames: np.ndarray, fps: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Pad width to even (yuv420p requires it) to avoid encoder errors.
    if frames.shape[2] % 2 == 1:
        frames = np.pad(frames, ((0, 0), (0, 0), (0, 1), (0, 0)), mode="edge")
    if frames.shape[1] % 2 == 1:
        frames = np.pad(frames, ((0, 0), (0, 1), (0, 0), (0, 0)), mode="edge")
    iio.imwrite(out_path, frames, fps=fps, codec="h264", plugin="pyav")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", type=Path, required=True,
                    help="lerobot-eval output dir for the baseline run")
    ap.add_argument("--planning_dir", type=Path, required=True,
                    help="lerobot-eval output dir for the planning run")
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    base_succ = _success_vec(args.baseline_dir / "eval_info.json")
    plan_succ = _success_vec(args.planning_dir / "eval_info.json")
    if len(base_succ) != len(plan_succ):
        raise SystemExit(
            f"mismatched N: baseline={len(base_succ)}  planning={len(plan_succ)}"
        )

    winning_eps = [i for i, (b, p) in enumerate(zip(base_succ, plan_succ)) if (not b) and p]
    losing_eps = [i for i, (b, p) in enumerate(zip(base_succ, plan_succ)) if b and (not p)]
    both_win   = [i for i, (b, p) in enumerate(zip(base_succ, plan_succ)) if b and p]
    both_lose  = [i for i, (b, p) in enumerate(zip(base_succ, plan_succ)) if (not b) and (not p)]

    print(f"[composite] N={len(base_succ)}", flush=True)
    print(f"[composite]   baseline ✗, planning ✓ : {len(winning_eps)}   eps={winning_eps}", flush=True)
    print(f"[composite]   baseline ✓, planning ✗ : {len(losing_eps)}    eps={losing_eps}", flush=True)
    print(f"[composite]   both ✓                 : {len(both_win)}", flush=True)
    print(f"[composite]   both ✗                 : {len(both_lose)}", flush=True)

    recovers_dir = args.out_dir / "planning_recovers"
    regresses_dir = args.out_dir / "planning_regresses"
    recovers_dir.mkdir(parents=True, exist_ok=True)
    regresses_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = [
        "# A/B disagreement composites",
        "",
        f"- N = {len(base_succ)}",
        f"- baseline success = {sum(base_succ)}",
        f"- planning success = {sum(plan_succ)}",
        f"- baseline ✗ ∧ planning ✓ (planner_recovers) = **{len(winning_eps)}**  (eps {winning_eps})",
        f"- baseline ✓ ∧ planning ✗ (planner_regresses) = **{len(losing_eps)}**  (eps {losing_eps})",
        f"- both ✓ = {len(both_win)}",
        f"- both ✗ = {len(both_lose)}",
        "",
        "## planning_recovers/  (planner saves a failure)",
        "",
    ]

    def _render(ep: int, out_dir: Path) -> tuple[int, int, str]:
        b_path = _episode_video(args.baseline_dir, ep)
        p_path = _episode_video(args.planning_dir, ep)
        b_frames, fps = _read_frames(b_path)
        p_frames, _ = _read_frames(p_path)
        composite = _hstack_pad(b_frames, p_frames)
        out = out_dir / f"ep_{ep:03d}_baseline_LEFT_planning_RIGHT.mp4"
        _write_composite(out, composite, fps)
        print(
            f"[composite] ep {ep:3d}: baseline {b_frames.shape[0]}f  planning {p_frames.shape[0]}f  → {out.relative_to(args.out_dir)}",
            flush=True,
        )
        return b_frames.shape[0], p_frames.shape[0], out.name

    for ep in winning_eps:
        b_len, p_len, name = _render(ep, recovers_dir)
        manifest_lines.append(
            f"- ep {ep:03d}: baseline len={b_len} (fail), planning len={p_len} (success) → `planning_recovers/{name}`"
        )

    manifest_lines += ["", "## planning_regresses/  (planner breaks a success)", ""]
    for ep in losing_eps:
        b_len, p_len, name = _render(ep, regresses_dir)
        manifest_lines.append(
            f"- ep {ep:03d}: baseline len={b_len} (success), planning len={p_len} (fail) → `planning_regresses/{name}`"
        )

    (args.out_dir / "manifest.md").write_text("\n".join(manifest_lines) + "\n")
    print(
        f"[composite] done. {len(winning_eps)} recovers + {len(losing_eps)} regresses "
        f"+ manifest.md → {args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
