"""Q-function eval visualization on HELD-OUT TEST episodes.

Called at ``eval_freq`` during training. For each sampled test episode — held
out from training by ``QValueLabelDataset``'s per-bucket episode split — renders
a side-by-side MP4:

  left  — camera frame at each timestep
  right — Q-value vs timestep, four series:
            navy line    Q(s, a_true) — the demonstrated chunk
            gray scatter Q(s, a_smooth) — planner-matched perturbations:
                         per-dim 1σ (BC bucket stats), temporally smoothed
                         like the MPPI planner, gripper held at true value.
                         Should sit lower-but-close to the navy line.
            orange line  Q(s, a_wrong) — the TRUE chunk of a DIFFERENT held-out
                         episode at the same episode fraction. Smooth, expert,
                         but wrong for this state. THE planning-relevant probe:
                         a Q that can't rank navy above orange cannot steer MPPI.
            light gray   Q(s, a_white) — legacy i.i.d. N(0, 0.3) raw-space noise,
                         kept as an out-of-distribution reference (a play-negative
                         Q collapses this to ~0 via the action-magnitude shortcut;
                         that collapse is an artifact of this probe, which is why
                         it is no longer the headline series).

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

from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset

if TYPE_CHECKING:
    from lerobot.policies.q_function.modeling_q_function import QFunctionPolicy
    from lerobot.policies.q_function.q_value_labels import QValueLabelDataset
    from lerobot.processor import PolicyProcessorPipeline

logger = logging.getLogger(__name__)

_NUM_EPISODES = 8       # test episodes visualized per eval
_NUM_SMOOTH = 6         # planner-matched smoothed perturbations per frame
_NUM_WHITE = 2          # legacy white-noise OOD references per frame
_PERTURB_STD = 0.3      # white-noise std (raw action space, legacy probe)
_SMOOTH_SIGMA_T = 2.0   # temporal smoothing bandwidth, matches the MPPI planner
_GRIP_DIM = -1
_VIS_FPS = 10
_CHUNK_T = 8   # timesteps batched per forward pass
_STRIDE = 4    # evaluate Q every Nth frame; values between strides are linearly interpolated


def _bc_action_sigma(dataset) -> np.ndarray:
    """Per-dim action std of the BC ('q5') bucket, for planner-matched perturbations.

    On a BC+play mixture the aggregated stats are play-inflated on the rotation
    dims (~10x), so we prefer the q5 sub-dataset's own stats; fall back to the
    wrapper-aggregated stats for single-bucket or unlabeled datasets.
    """
    raw = getattr(dataset, "dataset", dataset)
    buckets = getattr(dataset, "_dataset_index_to_bucket", None)
    if buckets is not None and isinstance(raw, MultiLeRobotDataset):
        for ds_idx, bucket in enumerate(buckets):
            if bucket == "q5":
                return np.asarray(raw._datasets[ds_idx].meta.stats["action"]["std"], dtype=np.float64)
    stats = getattr(dataset, "stats", None) or raw.meta.stats
    return np.asarray(stats["action"]["std"], dtype=np.float64)


def _episode_action_pool(dataset, ep_id: int, stride: int) -> dict:
    """Strided true action chunks of one episode, for the wrong-chunk probe.

    Returns {"fracs": (S,) ndarray, "actions": list[(L, A) Tensor]}. Loads full
    items (video decode included) but keeps only the actions.
    """
    from_idx = int(dataset._ep_from[ep_id].item())
    to_idx = int(dataset._ep_to[ep_id].item())
    T = to_idx - from_idx
    fracs, actions = [], []
    for t in range(0, T, stride):
        item = dataset[from_idx + t]
        fracs.append(t / max(1, T - 1))
        actions.append(item[ACTION].clone())
    return {"fracs": np.asarray(fracs), "actions": actions}


def _smoothed_sigma_noise(true: torch.Tensor, sigma_vec: np.ndarray, n: int) -> torch.Tensor:
    """Planner-matched perturbations: per-dim 1σ Gaussian, temporally smoothed,
    gripper held at the true value. Returns (n, L, A) raw-space chunks."""
    from lerobot.policies.act_simple.planning import _smooth_time

    L, A = true.shape
    noise = torch.randn(n, L, A) * torch.as_tensor(sigma_vec, dtype=torch.float32)
    noise = _smooth_time(noise, sigma=_SMOOTH_SIGMA_T)
    noise[..., _GRIP_DIM] = 0.0
    return (true.unsqueeze(0) + noise).clamp(-1.0, 1.0)


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
    preprocessor: "PolicyProcessorPipeline",
    items_chunk: list[dict],
    num_perturb: int,
    perturb_std: float,
    device: torch.device,
    candidates: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score ``ct`` dataset frames, each with its true action chunk + N alternatives.

    ``items_chunk``: list of ``ct`` raw dataset items (from ``dataset[idx]``).
    ``candidates``: optional ``(ct, K, L, A)`` tensor of raw-action-space chunks to
    score alongside the true chunk. When omitted, ``num_perturb`` white-noise
    perturbations of the true chunk (std ``perturb_std``) are generated instead.
    Returns a ``(ct, 1 + K)`` CPU tensor of Q-values — column 0 is the true
    chunk, columns 1.. are the candidates/perturbations.
    """
    ct = len(items_chunk)
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

    # Build true + alternative action chunks in raw action space, then let the
    # preprocessor normalize them (matches training's normalization exactly).
    raw_action = collated[ACTION]                       # (ct, L, A) raw
    L, A = raw_action.shape[1], raw_action.shape[2]
    if candidates is None:
        noise = torch.randn(ct, num_perturb, L, A) * perturb_std
        candidates = (raw_action.unsqueeze(1) + noise).clamp(-3.0, 3.0)
    elif candidates.shape[0] != ct or candidates.shape[2:] != (L, A):
        raise ValueError(
            f"candidates shape {tuple(candidates.shape)} incompatible with actions ({ct}, K, {L}, {A})"
        )
    group = 1 + candidates.shape[1]
    a_all = torch.cat([raw_action.unsqueeze(1), candidates.to(raw_action.dtype)], dim=1)
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
    sigma_vec: np.ndarray,
    ref_pool: dict | None,
    stride: int = _STRIDE,
    device: torch.device | None = None,
) -> dict:
    """Walk one test episode; score the true chunk against the probe battery.

    ``ep_global_id`` indexes ``dataset._ep_from`` / ``dataset._ep_to`` (the
    wrapper's global episode index). Frames are collected at every timestep for
    a smooth video; Q is computed every ``stride`` timesteps and linearly
    interpolated in between.

    ``ref_pool`` (from ``_episode_action_pool`` of a DIFFERENT episode) supplies
    the wrong-chunk probe; when None (single-test-episode edge case) the wrong
    chunk falls back to this episode's own chunk half an episode away in time.

    Returns dict with:
      frames:   (T, H, W, 3) uint8
      q_true:   (T,) float32
      q_smooth: (T, _NUM_SMOOTH) float32  planner-matched perturbations
      q_wrong:  (T,) float32              in-distribution wrong chunk
      q_white:  (T, _NUM_WHITE) float32   legacy white-noise OOD reference
      ep_idx:   int
    """
    if device is None:
        device = next(policy.parameters()).device

    # The QValueLabelDataset wrapper precomputes global episode bounds (works for
    # both single- and multi-dataset); no need to walk sub-datasets here.
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

    own_actions = [it[ACTION] for it in strided_items]

    def _wrong_chunk(idx: int) -> torch.Tensor:
        frac = strided_ts[idx] / max(1, T - 1)
        if ref_pool is not None and len(ref_pool["actions"]) > 0:
            j = int(np.argmin(np.abs(ref_pool["fracs"] - frac)))
            return ref_pool["actions"][j]
        # Fallback: own chunk half an episode away (wrong-time probe).
        return own_actions[(idx + len(own_actions) // 2) % len(own_actions)]

    S = len(strided_ts)
    q_true_s = np.zeros(S, dtype=np.float32)
    q_smooth_s = np.zeros((S, _NUM_SMOOTH), dtype=np.float32)
    q_wrong_s = np.zeros(S, dtype=np.float32)
    q_white_s = np.zeros((S, _NUM_WHITE), dtype=np.float32)
    for cs in range(0, S, _CHUNK_T):
        ce = min(cs + _CHUNK_T, S)
        cand_rows = []
        for k in range(cs, ce):
            true = own_actions[k]
            L, A = true.shape
            smooth = _smoothed_sigma_noise(true, sigma_vec, _NUM_SMOOTH)
            wrong = _wrong_chunk(k).unsqueeze(0)
            white = (true.unsqueeze(0) + torch.randn(_NUM_WHITE, L, A) * _PERTURB_STD).clamp(-3.0, 3.0)
            cand_rows.append(torch.cat([smooth, wrong, white], dim=0))
        q = _compute_chunk_q_values(
            policy, preprocessor, strided_items[cs:ce], 0, 0.0, device,
            candidates=torch.stack(cand_rows),
        )
        q_true_s[cs:ce] = q[:, 0].numpy()
        q_smooth_s[cs:ce] = q[:, 1 : 1 + _NUM_SMOOTH].numpy()
        q_wrong_s[cs:ce] = q[:, 1 + _NUM_SMOOTH].numpy()
        q_white_s[cs:ce] = q[:, 2 + _NUM_SMOOTH :].numpy()

    all_ts = np.arange(T)

    def _interp(col: np.ndarray) -> np.ndarray:
        return np.interp(all_ts, strided_ts, col).astype(np.float32)

    return {
        "frames": np.asarray(frames_list),
        "q_true": _interp(q_true_s),
        "q_smooth": np.stack([_interp(q_smooth_s[:, n]) for n in range(_NUM_SMOOTH)], axis=1),
        "q_wrong": _interp(q_wrong_s),
        "q_white": np.stack([_interp(q_white_s[:, n]) for n in range(_NUM_WHITE)], axis=1),
        "ep_idx": int(ep_global_id),
    }


def make_episode_video(
    frames: np.ndarray,
    q_true: np.ndarray,
    q_smooth: np.ndarray,
    q_wrong: np.ndarray,
    q_white: np.ndarray,
    ep_idx: int,
    out_path: str,
    v_min: float = -0.05,
    v_max: float = 1.1,
    fps: int = _VIS_FPS,
) -> str:
    """Render a side-by-side MP4: left = camera frame, right = Q-value plot.

    frames:   (T, H, W, 3) uint8
    q_true:   (T,) float32          — navy line
    q_smooth: (T, N) float32        — gray scatter (planner-matched noise)
    q_wrong:  (T,) float32          — orange line (wrong-episode chunk)
    q_white:  (T, M) float32        — light gray scatter (legacy OOD reference)
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

    ax_q.scatter(
        np.repeat(timesteps, q_white.shape[1]), q_white.ravel(),
        s=1, alpha=0.10, c="lightgray", linewidths=0, zorder=1, label="white (OOD ref)",
    )
    ax_q.scatter(
        np.repeat(timesteps, q_smooth.shape[1]), q_smooth.ravel(),
        s=1, alpha=0.18, c="steelblue", linewidths=0, zorder=2, label="planner-noise 1σ",
    )
    ax_q.plot(timesteps, q_wrong, color="darkorange", linewidth=1.2, zorder=3, label="wrong-ep chunk")
    ax_q.plot(timesteps, q_true, color="navy", linewidth=1.5, zorder=4, label="Q(s, a_true)")
    ax_q.legend(fontsize=5, loc="upper left", framealpha=0.7)

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

    sigma_vec = _bc_action_sigma(dataset)

    # Wrong-chunk reference pool: one held-out episode NOT among the visualized
    # ones (so the orange series has the same cross-episode meaning everywhere).
    # With a single test episode there is no such episode; compute_episode_q_values
    # then falls back to a wrong-time probe within the episode itself.
    ref_pool = None
    spare = [int(e) for e in test_ids if int(e) not in set(chosen)]
    if spare:
        try:
            ref_pool = _episode_action_pool(dataset, spare[0], stride=_STRIDE)
        except Exception:
            logger.exception("Q-vis: failed to build wrong-chunk pool from episode %d", spare[0])
    elif len(chosen) > 1:
        # All test episodes are being visualized: use the last chosen one as the
        # pool; for that episode itself the in-episode fallback kicks in... except
        # it would compare against its own chunks — drop it from its own pool by
        # passing None for that episode below.
        try:
            ref_pool = _episode_action_pool(dataset, chosen[-1], stride=_STRIDE)
        except Exception:
            logger.exception("Q-vis: failed to build wrong-chunk pool from episode %d", chosen[-1])

    try:
        for ep_id in chosen:
            torch.cuda.empty_cache()
            ep_ref = ref_pool if (spare or ep_id != chosen[-1]) else None
            try:
                data = compute_episode_q_values(
                    policy, dataset, preprocessor, ep_id,
                    sigma_vec=sigma_vec, ref_pool=ep_ref, device=device,
                )
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
                    q_smooth=data["q_smooth"],
                    q_wrong=data["q_wrong"],
                    q_white=data["q_white"],
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
