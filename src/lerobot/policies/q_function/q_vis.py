"""Q-function eval visualization on HELD-OUT TEST episodes.

Called at ``eval_freq`` during training. For each sampled test episode — held
out from training by ``QValueLabelDataset``'s per-bucket episode split — renders
a side-by-side MP4:

  left  — camera frame at each timestep
  right — Q-value vs timestep: a navy line for the true (demonstrated) action
           chunk, plus a gray scatter of Q for random perturbations of it.

This is the original dataset-trajectory Q visualization, but pointed at episodes
the Q never trained on — so the curve is a genuine held-out signal, and the
true-vs-perturbed gap shows whether the Q ranks the demonstrator action above
random noise on unseen states.

Actions/images are normalized by running each batch through the Q's own
preprocessor pipeline (the exact path the training loop uses), so Q values here
match what the model sees at train time regardless of normalization mode.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import default_collate

from lerobot.utils.constants import ACTION

if TYPE_CHECKING:
    from lerobot.policies.q_function.modeling_q_function import QFunctionPolicy
    from lerobot.policies.q_function.q_value_labels import QValueLabelDataset
    from lerobot.processor import PolicyProcessorPipeline

logger = logging.getLogger(__name__)

_NUM_EPISODES = 8       # test episodes visualized per eval
_NUM_PERTURB = 8        # random perturbations scored alongside the true chunk
_PERTURB_STD = 0.3      # perturbation noise std (raw action space)
_VIS_FPS = 10
_CHUNK_T = 8            # strided timesteps batched per forward pass
_STRIDE = 4             # evaluate Q every Nth frame; interpolate in between


def _compute_chunk_q_values(
    policy: "QFunctionPolicy",
    preprocessor: "PolicyProcessorPipeline",
    items_chunk: list[dict],
    num_perturb: int,
    perturb_std: float,
    device: torch.device,
) -> torch.Tensor:
    """Score ``ct`` dataset frames, each with its true action chunk + N perturbed.

    ``items_chunk``: list of ``ct`` raw dataset items (from ``dataset[idx]``).
    Returns a ``(ct, 1 + num_perturb)`` CPU tensor of Q-values — column 0 is the
    true chunk, columns 1.. are the perturbations.
    """
    ct = len(items_chunk)
    group = 1 + num_perturb
    lang_key = getattr(policy.config, "language_key", "task")

    # Pull task strings out before collation (string collation is fiddly).
    task_per_item: list[str] = []
    stripped: list[dict] = []
    for it in items_chunk:
        it = dict(it)
        ts = it.pop(lang_key, "")
        task_per_item.append(ts[0] if isinstance(ts, list) else str(ts))
        stripped.append(it)

    collated = default_collate(stripped)

    # Build true + perturbed action chunks in raw action space, then let the
    # preprocessor normalize them (matches training's normalization exactly).
    raw_action = collated[ACTION]                       # (ct, L, A) raw
    L, A = raw_action.shape[1], raw_action.shape[2]
    noise = torch.randn(ct, num_perturb, L, A) * perturb_std
    perturbed = (raw_action.unsqueeze(1) + noise).clamp(-3.0, 3.0)
    a_all = torch.cat([raw_action.unsqueeze(1), perturbed], dim=1)  # (ct, group, L, A)
    a_all = a_all.reshape(ct * group, L, A)

    B = ct * group
    batch: dict = {}
    for k, v in collated.items():
        if k == ACTION:
            continue
        if torch.is_tensor(v):
            batch[k] = (
                v.unsqueeze(1)
                .expand(ct, group, *v.shape[1:])
                .reshape(B, *v.shape[1:])
                .contiguous()
            )
        # non-tensor collated values (rare) are dropped — Q doesn't need them
    batch[ACTION] = a_all
    batch[lang_key] = [task_per_item[i] for i in range(ct) for _ in range(group)]

    batch = preprocessor(batch)
    with torch.no_grad():
        q = policy.predict_value(batch)            # (B,)
    return q.reshape(ct, group).float().cpu()


def compute_episode_q_values(
    policy: "QFunctionPolicy",
    dataset: "QValueLabelDataset",
    preprocessor: "PolicyProcessorPipeline",
    ep_global_id: int,
    num_perturb: int = _NUM_PERTURB,
    perturb_std: float = _PERTURB_STD,
    stride: int = _STRIDE,
    device: torch.device | None = None,
) -> dict:
    """Walk one test episode; compute Q(s_t, a_true) and Q(s_t, a_perturbed).

    ``ep_global_id`` indexes ``dataset._ep_from`` / ``dataset._ep_to`` (the
    wrapper's global episode index). Frames are collected at every timestep for
    a smooth video; Q is computed every ``stride`` timesteps and linearly
    interpolated in between.

    Returns dict with:
      frames:    (T, H, W, 3) uint8
      q_true:    (T,) float32
      q_perturb: (T, num_perturb) float32
      ep_idx:    int
    """
    if device is None:
        device = next(policy.parameters()).device

    from_idx = int(dataset._ep_from[ep_global_id].item())
    to_idx = int(dataset._ep_to[ep_global_id].item())
    T = to_idx - from_idx
    cam_key_vis = policy.config.camera_keys[0]

    frames_list: list[np.ndarray] = []
    strided_ts: list[int] = []
    strided_items: list[dict] = []
    for t in range(T):
        item = dataset[from_idx + t]
        img_vis = item[cam_key_vis]
        if img_vis.dim() == 4:        # (n_delta, 3, H, W) → current frame
            img_vis = img_vis[0]
        frames_list.append(
            (img_vis.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        )
        if t % stride == 0:
            strided_ts.append(t)
            strided_items.append(item)

    S = len(strided_ts)
    q_true_s = np.zeros(S, dtype=np.float32)
    q_perturb_s = np.zeros((S, num_perturb), dtype=np.float32)
    for cs in range(0, S, _CHUNK_T):
        ce = min(cs + _CHUNK_T, S)
        q = _compute_chunk_q_values(
            policy, preprocessor, strided_items[cs:ce], num_perturb, perturb_std, device
        )
        q_true_s[cs:ce] = q[:, 0].numpy()
        q_perturb_s[cs:ce] = q[:, 1:].numpy()

    all_ts = np.arange(T)
    q_true = np.interp(all_ts, strided_ts, q_true_s).astype(np.float32)
    q_perturb = np.stack(
        [np.interp(all_ts, strided_ts, q_perturb_s[:, n]) for n in range(num_perturb)],
        axis=1,
    ).astype(np.float32)

    return {
        "frames": np.asarray(frames_list),
        "q_true": q_true,
        "q_perturb": q_perturb,
        "ep_idx": int(ep_global_id),
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
    """Render a side-by-side MP4: left = camera frame, right = Q-value plot.

    frames:    (T, H, W, 3) uint8
    q_true:    (T,) float32         — navy line
    q_perturb: (T, N) float32       — gray scatter
    """
    import imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(frames)
    if T == 0:
        raise ValueError("make_episode_video called with 0 frames")
    timesteps = np.arange(T)
    cam_h, cam_w = frames[0].shape[:2]

    dpi = 80
    plot_w_in = cam_w / dpi
    plot_h_in = cam_h / dpi
    fig, ax_q = plt.subplots(1, 1, figsize=(plot_w_in, plot_h_in), dpi=dpi)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.92, bottom=0.18)

    ax_q.set_xlim(-0.5, max(0.5, T - 0.5))
    ax_q.set_ylim(v_min - 0.02, v_max + 0.02)
    ax_q.set_xlabel("Timestep", fontsize=7)
    ax_q.set_ylabel("Q(s, a)", fontsize=7)
    ax_q.tick_params(labelsize=6)
    ax_q.axhline(0, color="k", linewidth=0.5, linestyle=":")
    ax_q.set_title(f"test episode {ep_idx}", fontsize=7)

    T_tile = np.repeat(timesteps, q_perturb.shape[1])
    ax_q.scatter(
        T_tile, q_perturb.ravel(), s=1, alpha=0.12, c="steelblue", linewidths=0, zorder=1
    )
    ax_q.plot(timesteps, q_true, color="navy", linewidth=1.5, zorder=3, label="Q(s, a_true)")
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
        if plot_buf.shape[0] != cam_h:
            ph = plot_buf.shape[0]
            if ph > cam_h:
                plot_buf = plot_buf[:cam_h]
            else:
                plot_buf = np.pad(plot_buf, ((0, cam_h - ph), (0, 0), (0, 0)), constant_values=255)
        if plot_buf.shape[1] != cam_w:
            pw = plot_buf.shape[1]
            if pw > cam_w:
                plot_buf = plot_buf[:, :cam_w]
            else:
                plot_buf = np.pad(plot_buf, ((0, 0), (0, cam_w - pw), (0, 0)), constant_values=255)
        video_frames.append(np.concatenate([frames[t], plot_buf], axis=1))

    plt.close(fig)
    imageio.mimsave(out_path, video_frames, fps=fps)
    return out_path


def log_q_test_visualizations(
    policy: "QFunctionPolicy",
    dataset: "QValueLabelDataset",
    preprocessor: "PolicyProcessorPipeline",
    step: int,
    wandb_logger,
    num_episodes: int = _NUM_EPISODES,
    device: torch.device | None = None,
) -> None:
    """Render Q-value videos for a sample of held-out test episodes.

    No-op when the dataset has no test holdout (``test_split_ratio=0``). Safe to
    call during training — restores ``policy.train()`` on exit.
    """
    if device is None:
        device = next(policy.parameters()).device

    test_ids = list(getattr(dataset, "test_episode_ids", []) or [])
    if not test_ids:
        logger.info("Q-vis: no held-out test episodes (test_split_ratio=0?) — skipping")
        return

    n = min(num_episodes, len(test_ids))
    chosen = [int(test_ids[i]) for i in np.linspace(0, len(test_ids) - 1, num=n, dtype=int)]

    was_training = policy.training
    policy.eval()
    v_min = float(policy.config.v_min)
    v_max = float(policy.config.v_max)

    try:
        for ep_id in chosen:
            torch.cuda.empty_cache()
            try:
                data = compute_episode_q_values(policy, dataset, preprocessor, ep_id, device=device)
            except Exception:
                logger.exception("Q-vis: failed on test episode %d", ep_id)
                continue
            if len(data["frames"]) == 0:
                logger.warning("Q-vis: empty test episode %d, skipping", ep_id)
                continue

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                tmp_path = f.name
            try:
                make_episode_video(
                    frames=data["frames"],
                    q_true=data["q_true"],
                    q_perturb=data["q_perturb"],
                    ep_idx=ep_id,
                    out_path=tmp_path,
                    v_min=v_min,
                    v_max=v_max,
                )
                if wandb_logger is not None:
                    import wandb

                    wandb_logger._wandb.log(
                        {f"eval/q_test_ep{ep_id:04d}": wandb.Video(tmp_path, fps=_VIS_FPS, format="mp4")},
                        step=step,
                    )
                else:
                    local_dir = Path("outputs/q_vis")
                    local_dir.mkdir(parents=True, exist_ok=True)
                    import shutil

                    dest = local_dir / f"step{step:06d}_test_ep{ep_id:04d}.mp4"
                    shutil.copy(tmp_path, dest)
                    logger.info("Q-vis saved: %s", dest)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    finally:
        if was_training:
            policy.train()
