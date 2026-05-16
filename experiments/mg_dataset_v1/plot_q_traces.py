"""Aggregate Q-value traces across all rollouts in a single directory.

Reads ``ep_NN_trace.json`` files written by ``visualize_q_rollout.py`` and
produces a side-by-side comparison panel:
    left  = Q traces from successful rollouts
    right = Q traces from failed (truncated / timed-out) rollouts

Per panel:
    - thin line per episode (low alpha),
    - bold mean line at each chunk-boundary step,
    - ±1σ shaded band.

The two panels share Y-axis bounds so qualitative shape differences are
visible at a glance.

Usage::

    python -u experiments/mg_dataset_v1/plot_q_traces.py \\
        --traces_dir outputs/eval/mimicgen_threading_d0_q_function/20260510_173809 \\
        [--output /path/to/aggregate.png]   # defaults to <traces_dir>/aggregate_q_traces.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402


def _load_traces(traces_dir: Path) -> list[dict]:
    traces = []
    for p in sorted(traces_dir.glob("ep_*_trace.json")):
        with open(p) as f:
            traces.append(json.load(f))
    if not traces:
        raise FileNotFoundError(f"no ep_*_trace.json files under {traces_dir}")
    return traces


def _interp_to_grid(traces: list[dict], grid: np.ndarray) -> np.ndarray:
    """Linearly interpolate each (q_steps, q_values) onto `grid` (NaN beyond ep length)."""
    M = len(traces)
    G = len(grid)
    out = np.full((M, G), np.nan, dtype=np.float32)
    for i, tr in enumerate(traces):
        xs = np.asarray(tr["q_steps"], dtype=np.float32)
        ys = np.asarray(tr["q_values"], dtype=np.float32)
        if len(xs) < 1:
            continue
        # Interpolate within [xs[0], xs[-1]]; leave NaN outside (those grid points
        # are past this episode's end and shouldn't contribute to the band).
        in_range = (grid >= xs[0]) & (grid <= xs[-1])
        out[i, in_range] = np.interp(grid[in_range], xs, ys)
    return out


def _draw_panel(ax, traces: list[dict], title: str, color: str):
    if not traces:
        ax.text(0.5, 0.5, "no episodes", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("sim step")
        ax.set_ylabel("Q(s_t, exec[t:t+h])")
        return None, None
    # Per-episode thin lines (raw x, raw y).
    for tr in traces:
        ax.plot(tr["q_steps"], tr["q_values"], color=color, alpha=0.25, linewidth=1.0)
    # Mean ± 1σ band interpolated onto a shared grid spanning all episodes.
    max_step = max(int(tr["q_steps"][-1]) for tr in traces if tr["q_steps"])
    grid = np.arange(0, max_step + 1, dtype=np.float32)
    stacked = _interp_to_grid(traces, grid)
    # Reduce; keep NaN-tolerant.
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    valid = ~np.isnan(mean)
    ax.plot(grid[valid], mean[valid], color=color, linewidth=2.2, label=f"mean (N={len(traces)})")
    ax.fill_between(grid[valid], (mean - std)[valid], (mean + std)[valid], color=color, alpha=0.18)
    ax.set_title(title)
    ax.set_xlabel("sim step")
    ax.set_ylabel("Q(s_t, exec[t:t+h])")
    ax.legend(loc="lower right")
    return float(np.nanmin(stacked)), float(np.nanmax(stacked))


def make_aggregate_plot(traces_dir: Path, output_path: Path) -> Path:
    traces = _load_traces(traces_dir)
    succ = [t for t in traces if t.get("success")]
    fail = [t for t in traces if not t.get("success")]
    task = traces[0].get("task", "?")
    h = traces[0].get("h", "?")
    stride = traces[0].get("chunk_stride", "?")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    lo_s, hi_s = _draw_panel(axes[0], succ, f"successful  (N={len(succ)})", color="C2")
    lo_f, hi_f = _draw_panel(axes[1], fail, f"failed  (N={len(fail)})", color="C3")

    # Share y-range across panels for fair visual comparison.
    bounds = [v for v in (lo_s, lo_f, hi_s, hi_f) if v is not None]
    if bounds:
        ymin = min(bounds[::2])
        ymax = max(bounds[1::2])
        pad = max(1.0, 0.05 * (ymax - ymin))
        for ax in axes:
            ax.set_ylim(ymin - pad, ymax + pad)

    succ_rate = len(succ) / max(len(traces), 1)
    fig.suptitle(
        f"Q-value traces  |  task={task}  |  rollouts={len(traces)}  "
        f"(success rate {succ_rate:.0%})  |  h={h}  chunk_stride={stride}",
        fontsize=11,
    )
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces_dir", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    out = args.output or (args.traces_dir / "aggregate_q_traces.png")
    path = make_aggregate_plot(args.traces_dir, out)
    print(f"[plot_q_traces] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
