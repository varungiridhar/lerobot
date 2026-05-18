"""Q-function visualization: per-episode Q(s,a) curves logged as wandb videos.

Called at eval_freq during training. For each of num_episodes episodes sampled
from the dataset, renders a side-by-side MP4:
  left  — camera frame at each timestep
  right — Q-value vs timestep: solid line for true actions, gray scatter for
           random perturbations (like planning samples)
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.q_function.modeling_q_function import QFunctionPolicy

logger = logging.getLogger(__name__)

_NUM_EPISODES = 10
_NUM_PERTURB = 8
_PERTURB_STD = 0.3
_VIS_FPS = 10
_CHUNK_T = 8   # timesteps batched per forward pass (batch = _CHUNK_T * (1 + _NUM_PERTURB))
_STRIDE = 4    # evaluate Q every Nth frame; values between strides are linearly interpolated


def _normalize_actions(actions: Tensor, stats: dict) -> Tensor:
    """MEAN_STD normalize action tensor using dataset stats.

    actions: (..., action_dim)
    stats: {"mean": Tensor or ndarray, "std": Tensor or ndarray}
    """
    mean = torch.as_tensor(stats["mean"]).to(actions.device, dtype=actions.dtype)
    std = torch.as_tensor(stats["std"]).to(actions.device, dtype=actions.dtype) + 1e-8
    return (actions - mean) / std


def _item_to_current_images(item: dict, camera_keys: tuple[str, ...], device: torch.device) -> dict:
    """Extract the t=0 observation from a raw dataset item for each camera key.

    The Q-function dataset loads observations at delta_indices [0, h]. Each
    camera tensor has shape (n_delta, 3, H, W). We take index 0 (current frame).
    Returns {key: Tensor(3, H, W)} on device.
    """
    imgs = {}
    for ck in camera_keys:
        t = item[ck]  # (n_delta, 3, H, W) or (3, H, W)
        if t.dim() == 4:
            t = t[0]  # (3, H, W)
        imgs[ck] = t.to(device, dtype=torch.float32)
    return imgs


def _compute_chunk_q_values(
    policy: "QFunctionPolicy",
    imgs_per_cam: dict[str, Tensor],
    actions_norm: Tensor,
    task_str: str,
) -> Tensor:
    """Single forward pass for a chunk of (timestep, sample) pairs.

    imgs_per_cam: {cam_key: Tensor(B, 3, H, W)} — B = T_chunk * (1 + N)
    actions_norm: Tensor(B, h, action_dim)
    Returns: Tensor(B,) Q-values
    """
    batch: dict = {}
    for ck, imgs in imgs_per_cam.items():
        batch[ck] = imgs
    batch["action"] = actions_norm
    if policy.config.use_text_conditioning:
        batch["task"] = [task_str] * actions_norm.shape[0]
    return policy.predict_value(batch)


def compute_episode_q_values(
    policy: "QFunctionPolicy",
    raw_dataset: "LeRobotDataset",
    ep_idx: int,
    num_perturb: int = _NUM_PERTURB,
    perturb_std: float = _PERTURB_STD,
    stride: int = _STRIDE,
    device: torch.device | None = None,
) -> dict:
    """Compute Q(s_t, a_true) and Q(s_t, a_perturbed) for an episode.

    Frames are collected at every timestep (smooth video). Q-values are computed
    only at every `stride` timesteps and linearly interpolated in between, reducing
    forward passes by stride× while keeping visual continuity.

    Returns dict with:
      frames:    (T, H, W, 3) uint8 numpy — camera frames for video
      q_true:    (T,) float32 numpy
      q_perturb: (T, num_perturb) float32 numpy
      ep_idx:    int
    """
    if device is None:
        device = next(policy.parameters()).device

    ep_meta = raw_dataset.meta.episodes[ep_idx]
    from_idx = int(ep_meta["dataset_from_index"])
    to_idx = int(ep_meta["dataset_to_index"])
    T = to_idx - from_idx
    h = policy.config.h
    action_stats = raw_dataset.meta.stats["action"]
    cam_key_vis = policy.config.camera_keys[0]

    frames_list: list[np.ndarray] = []
    # Only collect imgs/actions at strided timesteps for inference
    strided_ts: list[int] = []
    raw_imgs: dict[str, list[Tensor]] = {ck: [] for ck in policy.config.camera_keys}
    raw_actions: list[Tensor] = []
    task_str = ""

    dummy_img = torch.zeros(3, policy.config.image_resize_h, policy.config.image_resize_w)
    dummy_frame = np.zeros((policy.config.image_resize_h, policy.config.image_resize_w, 3), dtype=np.uint8)

    for t in range(T):
        try:
            item = raw_dataset[from_idx + t]
        except Exception:
            frames_list.append(dummy_frame)
            if t % stride == 0:
                strided_ts.append(t)
                for ck in policy.config.camera_keys:
                    raw_imgs[ck].append(dummy_img)
                raw_actions.append(torch.zeros(h, policy.full_action_dim))
            continue

        img_vis = item[cam_key_vis]
        if img_vis.dim() == 4:
            img_vis = img_vis[0]
        frames_list.append((img_vis.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8))

        if t % stride == 0:
            strided_ts.append(t)
            for ck in policy.config.camera_keys:
                ci = item[ck]
                if ci.dim() == 4:
                    ci = ci[0]
                raw_imgs[ck].append(ci)
            action = item["action"]
            if action.dim() == 2:
                action = action[:h]
            raw_actions.append(action)

        if not task_str:
            ts = item.get("task", "")
            task_str = ts[0] if isinstance(ts, list) else str(ts)

    S = len(strided_ts)
    q_true_s = np.zeros(S, dtype=np.float32)
    q_perturb_s = np.zeros((S, num_perturb), dtype=np.float32)

    # Chunked inference over strided timesteps: batch = _CHUNK_T * (1 + num_perturb)
    for chunk_start in range(0, S, _CHUNK_T):
        chunk_end = min(chunk_start + _CHUNK_T, S)
        ct = chunk_end - chunk_start

        a_true_stack = torch.stack(raw_actions[chunk_start:chunk_end], dim=0)  # (ct, h, D)
        D = a_true_stack.shape[-1]

        noise_p = torch.randn(ct, num_perturb, h, D) * perturb_std
        a_perturb = (a_true_stack.unsqueeze(1) + noise_p).clamp(-3, 3)
        a_all = torch.cat([a_true_stack.unsqueeze(1), a_perturb], dim=1).reshape(ct * (1 + num_perturb), h, D)
        a_norm = _normalize_actions(a_all, action_stats).to(device)

        imgs_chunk: dict[str, Tensor] = {}
        for ck in policy.config.camera_keys:
            img_stack = torch.stack(raw_imgs[ck][chunk_start:chunk_end], dim=0)  # (ct, 3, H, W)
            imgs_chunk[ck] = (
                img_stack.unsqueeze(1)
                .expand(ct, 1 + num_perturb, -1, -1, -1)
                .reshape(ct * (1 + num_perturb), *img_stack.shape[1:])
                .to(device, dtype=torch.float32)
            )

        with torch.no_grad():
            q_vals = _compute_chunk_q_values(policy, imgs_chunk, a_norm, task_str).cpu()

        q_vals_2d = q_vals.reshape(ct, 1 + num_perturb).numpy()
        q_true_s[chunk_start:chunk_end] = q_vals_2d[:, 0]
        q_perturb_s[chunk_start:chunk_end] = q_vals_2d[:, 1:]

    # Interpolate Q-values from strided positions back to all T timesteps
    all_ts = np.arange(T)
    q_true = np.interp(all_ts, strided_ts, q_true_s).astype(np.float32)
    q_perturb_arr = np.stack(
        [np.interp(all_ts, strided_ts, q_perturb_s[:, n]) for n in range(num_perturb)], axis=1
    ).astype(np.float32)

    return {
        "frames": np.array(frames_list),
        "q_true": q_true,
        "q_perturb": q_perturb_arr,
        "ep_idx": ep_idx,
    }


def make_episode_video(
    frames: np.ndarray,
    q_true: np.ndarray,
    q_perturb: np.ndarray,
    ep_idx: int,
    out_path: str,
    v_min: float = -0.05,
    v_max: float = 1.1,
    fps: int = _VIS_FPS,
) -> str:
    """Render a side-by-side MP4: left=raw camera frame, right=Q-value plot.

    frames:    (T, H, W, 3) uint8
    q_true:    (T,) float32
    q_perturb: (T, N) float32
    Returns: out_path
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio

    T = len(frames)
    timesteps = np.arange(T)
    cam_h, cam_w = frames[0].shape[:2]

    # Q-plot panel sized to exactly match camera frame height (no padding)
    dpi = 80
    plot_w_in = cam_w / dpi  # same width as camera
    plot_h_in = cam_h / dpi
    fig, ax_q = plt.subplots(1, 1, figsize=(plot_w_in, plot_h_in), dpi=dpi)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.97, bottom=0.18)

    ax_q.set_xlim(-0.5, T - 0.5)
    ax_q.set_ylim(v_min - 0.02, v_max + 0.02)
    ax_q.set_xlabel("Timestep", fontsize=7)
    ax_q.set_ylabel("Q(s, a)", fontsize=7)
    ax_q.tick_params(labelsize=6)
    ax_q.axhline(0, color="k", linewidth=0.5, linestyle=":")

    T_tile = np.repeat(timesteps, q_perturb.shape[1])
    q_perturb_flat = q_perturb.ravel()
    ax_q.scatter(T_tile, q_perturb_flat, s=1, alpha=0.12, c="steelblue", linewidths=0, zorder=1)
    ax_q.plot(timesteps, q_true, color="navy", linewidth=1.5, zorder=3, label="Q(s,a_true)")
    ax_q.legend(fontsize=6, loc="upper left")

    vline = ax_q.axvline(0, color="red", linestyle="--", linewidth=1.0, alpha=0.8, zorder=4)
    fig.canvas.draw()
    plot_w_px, plot_h_px = fig.canvas.get_width_height()

    video_frames: list[np.ndarray] = []
    for t in range(T):
        vline.set_xdata([t, t])
        fig.canvas.draw()
        plot_buf = (
            np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            .reshape(plot_h_px, plot_w_px, 4)[:, :, :3]
            .copy()
        )
        # Crop/pad plot to exactly cam_h rows if dpi rounding caused 1-px drift
        if plot_buf.shape[0] != cam_h:
            ph = plot_buf.shape[0]
            if ph > cam_h:
                plot_buf = plot_buf[:cam_h]
            else:
                plot_buf = np.pad(plot_buf, ((0, cam_h - ph), (0, 0), (0, 0)), constant_values=255)
        video_frames.append(np.concatenate([frames[t], plot_buf], axis=1))

    plt.close(fig)

    imageio.mimsave(out_path, video_frames, fps=fps)
    return out_path


def log_q_visualizations(
    policy: "QFunctionPolicy",
    raw_dataset: "LeRobotDataset",
    step: int,
    wandb_logger,
    num_episodes: int = _NUM_EPISODES,
    device: torch.device | None = None,
) -> None:
    """Compute Q-value rollouts for num_episodes and log videos to wandb.

    Evenly samples episode indices from the dataset. Safe to call in training
    (restores policy.train() on exit).
    """
    if device is None:
        device = next(policy.parameters()).device

    num_total = getattr(raw_dataset, "num_episodes", None) or getattr(
        raw_dataset.meta, "total_episodes", 0
    )
    if num_total == 0:
        return

    ep_indices = np.linspace(0, num_total - 1, num=min(num_episodes, num_total), dtype=int)
    was_training = policy.training
    policy.eval()

    v_min = policy.config.v_min
    v_max = policy.config.v_max

    try:
        for ep_idx in ep_indices:
            ep_idx = int(ep_idx)
            torch.cuda.empty_cache()
            try:
                data = compute_episode_q_values(policy, raw_dataset, ep_idx, device=device)
            except Exception:
                logger.exception("Q-vis: failed on episode %d", ep_idx)
                continue

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                tmp_path = f.name
            try:
                make_episode_video(
                    frames=data["frames"],
                    q_true=data["q_true"],
                    q_perturb=data["q_perturb"],
                    ep_idx=ep_idx,
                    out_path=tmp_path,
                    v_min=v_min,
                    v_max=v_max,
                )
                if wandb_logger is not None:
                    import wandb
                    wandb_logger._wandb.log(
                        {f"eval/q_ep{ep_idx:04d}": wandb.Video(tmp_path, fps=_VIS_FPS, format="mp4")},
                        step=step,
                    )
                else:
                    # Save locally for prototyping
                    local_dir = Path("outputs/q_vis")
                    local_dir.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy(tmp_path, local_dir / f"step{step:06d}_ep{ep_idx:04d}.mp4")
                    logger.info("Q-vis saved: %s", local_dir / f"step{step:06d}_ep{ep_idx:04d}.mp4")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    finally:
        if was_training:
            policy.train()
