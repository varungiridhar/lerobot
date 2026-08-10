"""Planning visualization: side-by-side eval video + Q-value spotlight animation.

Left panel  — camera frame at each planning step.
Right panel — Q-value vs planning-chunk index:
  - faint blue scatter: all N candidate Q values at every chunk
  - solid line: Q(s, a_selected) for chunks seen so far
  - animated spotlight: at chunk t the candidate dots for t turn solid/bright,
    then revert to faint when t advances

Reuses panel-rendering machinery from q_vis.py.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_VIS_FPS = 5   # planning-rate video — slower than env rate, easier to read


def _obs_frame_to_uint8(img_tensor) -> np.ndarray:
    """Convert a (1, C, H, W) or (C, H, W) float tensor to (H, W, 3) uint8.

    FastWAM uses IDENTITY visual normalization so images arrive in [0, 1].
    """
    t = img_tensor
    if t.dim() == 4:
        t = t[0]                        # (C, H, W)
    t = t.detach().float().cpu().clamp(0, 1)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def make_planning_vis_video(
    chunk_frames: list[np.ndarray],    # one (H, W, 3) uint8 frame per planning chunk (fallback)
    q_candidates: list[np.ndarray],    # one (N,) float array per chunk
    q_selected: list[float],           # one float per chunk (MPPI-weighted mean of Q)
    out_path: str | Path,
    v_min: float = 0.0,
    v_max: float = 1.0,
    fps: int = _VIS_FPS,
    vis_chunks: list[dict] | None = None,  # full chunk dicts with optional "step_frames"
) -> str:
    """Render side-by-side MP4: left=camera, right=animated Q-value spotlight.

    If vis_chunks contains "step_frames" lists, the camera panel advances every env step
    (stride 1) while the Q panel updates only at chunk boundaries. Otherwise falls back
    to one video frame per planning chunk.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio

    n_chunks = len(chunk_frames)
    if n_chunks == 0:
        logger.warning("make_planning_vis_video: no chunks to render, skipping.")
        return str(out_path)

    # Build per-video-frame sequence: (camera_frame, chunk_index)
    # Use step_frames if available for full temporal resolution.
    if vis_chunks is not None and any("step_frames" in c and len(c["step_frames"]) > 1 for c in vis_chunks):
        frame_seq: list[tuple[np.ndarray, int]] = []
        for t, chunk in enumerate(vis_chunks):
            frames = chunk.get("step_frames") or ([chunk["frame"]] if chunk.get("frame") is not None else [])
            for f in frames:
                frame_seq.append((f, t))
    else:
        frame_seq = [(chunk_frames[t], t) for t in range(n_chunks)]

    if not frame_seq:
        logger.warning("make_planning_vis_video: no frames to render, skipping.")
        return str(out_path)

    cam_h, cam_w = frame_seq[0][0].shape[:2]
    dpi = 80
    fig, ax = plt.subplots(1, 1, figsize=(cam_w / dpi, cam_h / dpi), dpi=dpi)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.97, bottom=0.18)

    ax.set_xlim(-0.5, n_chunks - 0.5)
    ax.set_ylim(v_min - 0.02, v_max + 0.02)
    ax.set_xlabel("Planning chunk", fontsize=7)
    ax.set_ylabel("Q(s, a)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.axhline(0, color="k", linewidth=0.5, linestyle=":")

    # Faint background scatter for ALL chunks (plotted once, always visible)
    all_xs = np.concatenate([[c] * len(q_candidates[c]) for c in range(n_chunks)])
    all_qs = np.concatenate(q_candidates)
    ax.scatter(all_xs, all_qs, s=1, alpha=0.12, c="steelblue", linewidths=0, zorder=1)

    # Solid line for selected Q values (revealed incrementally)
    (line_obj,) = ax.plot([], [], color="navy", linewidth=1.5, zorder=3, label="Q(s,a_selected)")
    ax.legend(fontsize=6, loc="upper left")

    fig.canvas.draw()
    plot_w_px, plot_h_px = fig.canvas.get_width_height()

    video_frames: list[np.ndarray] = []
    spotlight = None
    prev_chunk_idx = -1

    for cam_frame, t in frame_seq:
        # Update Q panel only when advancing to a new chunk
        if t != prev_chunk_idx:
            line_obj.set_data(range(t + 1), q_selected[: t + 1])
            if spotlight is not None:
                spotlight.remove()
            cq = q_candidates[t]
            spotlight = ax.scatter(
                [t] * len(cq), cq, s=6, alpha=1.0, c="steelblue", linewidths=0, zorder=5,
            )
            fig.canvas.draw()
            prev_chunk_idx = t

        plot_buf = (
            np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            .reshape(plot_h_px, plot_w_px, 4)[:, :, :3]
            .copy()
        )

        # Match camera height (dpi rounding can cause ±1 px)
        if plot_buf.shape[0] != cam_h:
            ph = plot_buf.shape[0]
            if ph > cam_h:
                plot_buf = plot_buf[:cam_h]
            else:
                plot_buf = np.pad(
                    plot_buf, ((0, cam_h - ph), (0, 0), (0, 0)), constant_values=255
                )

        video_frames.append(np.concatenate([cam_frame, plot_buf], axis=1))

    plt.close(fig)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), video_frames, fps=fps)
    logger.info("Planning vis video saved: %s  (%d frames)", out_path, len(video_frames))
    return str(out_path)
