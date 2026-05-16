"""Visualize Q-function estimates alongside a BC rollout in the eval env.

For each rollout:
  - Roll out the trained act_simple BC policy in the MimicGen env (single env, deterministic
    init via the task's init_states.pt).
  - Record per-step rgb camera frames (high-res render) + the action that was actually executed.
  - After the rollout, at every BC chunk boundary (i.e., every n_action_steps sim steps),
    score Q on the (s_t, executed_actions[t:t+h]) pair — re-encoding the camera images at
    s_t live through the same frozen act_simple backbone that produced the cache.
  - Compose a side-by-side mp4:
      left  = high-res rgb at step t,
      right = scatter+line plot of Q(t') for chunk-boundary steps t' <= t,
              with a vertical marker tracking the current step.
  - Save each rollout's mp4 locally and (optionally) upload to wandb.

Usage::

    python -u experiments/mg_dataset_v1/visualize_q_rollout.py \\
        --task coffee \\
        --bc_checkpoint outputs/train/mimicgen_coffee_d0_act_simple/checkpoints/100000/pretrained_model \\
        --q_checkpoint  outputs/train/mimicgen_coffee_d0_q_function/checkpoints/last/pretrained_model \\
        --output_dir    outputs/eval/mimicgen_coffee_d0_q_function/$(date +%Y%m%d_%H%M%S) \\
        --n_rollouts 10 \\
        [--wandb_enable]

Notes
-----
* The chunk-size mismatch (BC chunk_size=10, Q h=20) is handled by scoring Q at every
  10th sim step on the next 20 EXECUTED actions. This means Q's view of "the policy's chunk"
  is two consecutive BC chunks. Boundary steps where t+h > rollout_length are skipped.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import torch                             # noqa: E402

# System ffmpeg isn't on PATH inside the lerobot-mimicgen conda env; point
# matplotlib at the bundled binary that imageio_ffmpeg ships with.
try:
    import imageio_ffmpeg as _iio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = _iio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

from lerobot.envs.mimicgen import MimicGenEnv  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.policies.act_simple.modeling_act_simple import ACTSimplePolicy  # noqa: E402
from lerobot.policies.act_simple.precache_features import (  # noqa: E402
    load_camera_norm_stats,
    load_policy_backbone,
)
from lerobot.policies.q_function.modeling_q_function import QFunctionPolicy  # noqa: E402
from lerobot.processor import PolicyProcessorPipeline  # noqa: E402
from lerobot.processor.converters import (  # noqa: E402
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import ACTION, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [viz_q] %(message)s")
log = logging.getLogger("viz_q")

TASK_TO_ENV = {"square": "Square_D0", "threading": "Threading_D0", "coffee": "Coffee_D0"}
CAMERA_IN_OBS = {"observation.images.image": "image", "observation.images.image2": "image2"}


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def _load_pre(ckpt_dir: Path) -> PolicyProcessorPipeline:
    """Load a preprocessor (dict→dict). Must pass converters explicitly because
    PolicyProcessorPipeline.from_pretrained defaults them to the dict-based ones,
    which is what we want for preprocessors anyway, but we set them explicitly
    for parity with how make_pre_post_processors loads them."""
    return PolicyProcessorPipeline.from_pretrained(
        str(ckpt_dir), config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        to_transition=batch_to_transition, to_output=transition_to_batch,
    )


def _load_post(ckpt_dir: Path) -> PolicyProcessorPipeline:
    """Load a postprocessor (PolicyAction tensor → PolicyAction tensor). The
    default dict-based converters CANNOT be used here — we must pass the
    action-tensor variants explicitly (mirrors make_pre_post_processors)."""
    return PolicyProcessorPipeline.from_pretrained(
        str(ckpt_dir), config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action,
    )


def load_bc(checkpoint_dir: Path, device: torch.device):
    policy = ACTSimplePolicy.from_pretrained(str(checkpoint_dir)).to(device).eval()
    pre = _load_pre(checkpoint_dir)
    post = _load_post(checkpoint_dir)
    return policy, pre, post


def load_q(checkpoint_dir: Path, device: torch.device):
    policy = QFunctionPolicy.from_pretrained(str(checkpoint_dir)).to(device).eval()
    pre = _load_pre(checkpoint_dir)
    return policy, pre


# ──────────────────────────────────────────────────────────────────────────────
# Rollout
# ──────────────────────────────────────────────────────────────────────────────

def rollout(bc_policy, bc_pre, bc_post, env, max_steps: int, device: torch.device):
    """One BC rollout. Returns (rgb_frames, agent_obs_history, executed_actions, success)."""
    raw_obs, _info = env.reset()
    bc_policy.reset()

    rgb_frames: list[np.ndarray] = []
    agent_obs_history: list[dict] = []   # raw pixels dict per step (for Q-scoring later)
    executed_actions: list[np.ndarray] = []
    success = False

    # Record the initial frame + obs.
    rgb_frames.append(env.render())
    agent_obs_history.append({k: v.copy() for k, v in raw_obs["pixels"].items()})

    for _ in range(max_steps):
        # Build LeRobot-format batch from raw obs.
        batch = preprocess_observation(raw_obs)
        batch["task"] = [env.task_description]
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch = bc_pre(batch)
        action = bc_policy.select_action(batch)             # (1, A) normalized
        action = bc_post(action).cpu().squeeze(0).numpy()   # (A,) raw

        raw_obs, _reward, terminated, truncated, info = env.step(action)
        executed_actions.append(action.copy())
        rgb_frames.append(env.render())
        agent_obs_history.append({k: v.copy() for k, v in raw_obs["pixels"].items()})
        if info.get("is_success"):
            success = True
        if terminated or truncated:
            break

    return rgb_frames, agent_obs_history, np.stack(executed_actions), success


# ──────────────────────────────────────────────────────────────────────────────
# Q scoring at chunk boundaries
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _encode_views(backbone, norm_stats, pixels_dict, device) -> dict[str, torch.Tensor]:
    """Live-encode both cameras for one observation. Returns (1, 512, 3, 3) per cam key."""
    out = {}
    for lerobot_key, cam_in_obs in CAMERA_IN_OBS.items():
        img = pixels_dict[cam_in_obs]
        x = torch.from_numpy(img).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        mean, std = norm_stats[lerobot_key]
        x = (x - mean.to(device).view(1, 3, 1, 1)) / std.to(device).view(1, 3, 1, 1).clamp_min(1e-8)
        feat = backbone(x)["feature_map"]                                # (1, 512, 3, 3)
        out[f"{lerobot_key}_preencoded"] = feat
    return out


@torch.no_grad()
def score_q_at_boundaries(
    q_policy, q_pre, backbone, norm_stats,
    agent_obs_history: list[dict],
    executed_actions: np.ndarray,
    chunk_stride: int,
    h: int,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    """Return (steps, q_values) at chunk-aligned t with t+h <= len(actions)."""
    T = executed_actions.shape[0]
    steps, values = [], []
    for t in range(0, T - h + 1, chunk_stride):
        action_window = executed_actions[t : t + h]                           # (h, A)
        action_t = torch.from_numpy(action_window).unsqueeze(0).to(device)    # (1, h, A) raw
        batch = _encode_views(backbone, norm_stats, agent_obs_history[t], device)
        batch[ACTION] = action_t.float()
        # Bypass the preprocessor's `_q_batch_to_transition`, which expects q_* keys.
        # We just need action normalization here; the preprocessor's NormalizerProcessorStep
        # handles it. Run the pipeline:
        batch = q_pre(batch)
        q = q_policy.predict_value(batch).item()
        steps.append(t)
        values.append(q)
    return steps, values


# ──────────────────────────────────────────────────────────────────────────────
# Video composition
# ──────────────────────────────────────────────────────────────────────────────

def compose_video(rgb_frames: list[np.ndarray], q_steps: list[int], q_values: list[float],
                  success: bool, ep_idx: int, fps: int, output_path: Path) -> None:
    """Side-by-side: left = rgb at step t; right = Q(t') for chunk-boundary t' <= t."""
    T = len(rgb_frames)
    fig, (ax_img, ax_q) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1, 1.6]})
    im = ax_img.imshow(rgb_frames[0])
    ax_img.axis("off")
    ax_img.set_title(f"ep {ep_idx}  success={success}")

    if q_values:
        ymin, ymax = float(min(q_values)), float(max(q_values))
        pad = max(1.0, 0.1 * (ymax - ymin))
        ax_q.set_ylim(ymin - pad, ymax + pad)
    ax_q.set_xlim(0, max(T - 1, 1))
    ax_q.set_xlabel("sim step")
    ax_q.set_ylabel("Q(s_t, exec[t:t+h])")
    ax_q.set_title("Q value (chunk-aligned)")
    ax_q.grid(True, alpha=0.3)
    line, = ax_q.plot([], [], "-o", color="C0", markersize=4, linewidth=1.2)
    marker = ax_q.axvline(0, color="C3", linewidth=0.8)

    q_steps_arr = np.asarray(q_steps)
    q_values_arr = np.asarray(q_values)

    def update(t):
        im.set_data(rgb_frames[t])
        if q_steps:
            mask = q_steps_arr <= t
            line.set_data(q_steps_arr[mask], q_values_arr[mask])
        marker.set_xdata([t, t])
        return im, line, marker

    anim = animation.FuncAnimation(fig, update, frames=T, interval=int(1000 / fps), blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer=animation.FFMpegWriter(fps=fps))
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=tuple(TASK_TO_ENV))
    ap.add_argument("--bc_checkpoint", required=True, type=Path,
                    help="path to act_simple pretrained_model dir")
    ap.add_argument("--q_checkpoint", required=True, type=Path,
                    help="path to q_function pretrained_model dir")
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--n_rollouts", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--init_states_path", type=Path, default=None,
                    help="optional override; defaults to v1 BC dataset's init_states.pt")
    ap.add_argument("--render_size", type=int, default=256)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--wandb_enable", action="store_true")
    ap.add_argument("--debug", action="store_true",
                    help="skip wandb init + file writes (mp4, json trace); just run "
                         "rollout + Q scoring + print summary. Use for quick validation.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[viz_q] device={device} task={args.task} output_dir={args.output_dir}", flush=True)

    # ── Build policies + preprocessors ───────────────────────────────────
    print(f"[viz_q] loading BC from {args.bc_checkpoint}", flush=True)
    bc_policy, bc_pre, bc_post = load_bc(args.bc_checkpoint, device)
    print(f"[viz_q] loading Q from {args.q_checkpoint}", flush=True)
    q_policy, q_pre = load_q(args.q_checkpoint, device)
    h = int(q_policy.config.h)
    chunk_stride = int(bc_policy.config.n_action_steps)
    print(f"[viz_q] Q.h={h}  BC.n_action_steps={chunk_stride} (chunk stride for Q scoring)", flush=True)

    # Live image encoder (the act_simple backbone, frozen, same one the precache used).
    print("[viz_q] loading backbone for live image encoding", flush=True)
    backbone = load_policy_backbone(args.bc_checkpoint, device)
    norm_stats = load_camera_norm_stats(args.bc_checkpoint)

    # ── Build env ────────────────────────────────────────────────────────
    env_name = TASK_TO_ENV[args.task]
    init_states_path = args.init_states_path or Path(
        f"/storage/home/hcoda1/6/vgiridhar6/shared/mimicgen/lerobot_datasets/v1_q5_q3jitter_play"
        f"/{args.task}/bc_input/lerobot_ds/meta/init_states.pt"
    )
    if not init_states_path.exists():
        print(f"[viz_q] WARN init_states_path={init_states_path} missing — rollouts will be random", flush=True)
        init_states_path = None
    env = MimicGenEnv(
        env_name=env_name,
        init_states_path=str(init_states_path) if init_states_path else None,
        render_height=args.render_size,
        render_width=args.render_size,
        max_episode_steps=args.max_steps,
    )

    # ── Wandb (optional; suppressed in debug) ────────────────────────────
    wandb_run = None
    if args.wandb_enable and not args.debug:
        import wandb
        wandb_run = wandb.init(
            project="awm", entity="pair-diffusion", job_type="eval",
            name=f"eval_q_{args.task}_{int(time.time())}",
            config={**vars(args), "h": h, "chunk_stride": chunk_stride},
        )

    # ── Rollouts ─────────────────────────────────────────────────────────
    summary = {"success": [], "q_at_t0": [], "q_at_terminal_step": [], "ep_len": []}
    for ep in range(args.n_rollouts):
        env.action_space.seed(args.seed + ep)
        torch.manual_seed(args.seed + ep)
        np.random.seed(args.seed + ep)

        t0 = time.time()
        rgb_frames, obs_history, actions, success = rollout(
            bc_policy, bc_pre, bc_post, env, args.max_steps, device,
        )
        ep_len = len(rgb_frames)
        print(f"[viz_q] ep {ep}/{args.n_rollouts}  success={success}  len={ep_len}  rollout_s={time.time() - t0:.2f}", flush=True)

        t0 = time.time()
        q_steps, q_values = score_q_at_boundaries(
            q_policy, q_pre, backbone, norm_stats,
            obs_history, actions, chunk_stride=chunk_stride, h=h, device=device,
        )
        print(f"[viz_q]   Q scored at {len(q_steps)} boundaries  score_s={time.time() - t0:.2f}", flush=True)

        if args.debug:
            q_min = float(min(q_values)) if q_values else float("nan")
            q_max = float(max(q_values)) if q_values else float("nan")
            q_mean = float(np.mean(q_values)) if q_values else float("nan")
            head_vals = [f"{v:.3f}" for v in q_values[:5]]
            print(
                f"[viz_q]   [debug] success={success} ep_len={ep_len} q_n={len(q_steps)} "
                f"q_steps[head]={q_steps[:5]} q_values[head]={head_vals} "
                f"q_min={q_min:.3f} q_max={q_max:.3f} q_mean={q_mean:.3f}",
                flush=True,
            )
        else:
            t0 = time.time()
            video_path = args.output_dir / f"ep_{ep:02d}_success={int(success)}.mp4"
            compose_video(rgb_frames, q_steps, q_values, success, ep, args.fps, video_path)
            print(f"[viz_q]   video → {video_path}  compose_s={time.time() - t0:.2f}", flush=True)

            trace_path = args.output_dir / f"ep_{ep:02d}_trace.json"
            import json as _json
            trace_path.write_text(_json.dumps({
                "task": args.task, "ep": ep, "success": bool(success),
                "ep_len": ep_len, "q_steps": q_steps, "q_values": q_values,
                "h": h, "chunk_stride": chunk_stride,
            }, indent=2))

        if wandb_run is not None:
            import wandb
            wandb_run.log({
                f"rollout/video": wandb.Video(str(video_path), fps=args.fps, format="mp4"),
                f"rollout/success": int(success),
                f"rollout/ep_len": ep_len,
                f"rollout/q_at_t0": q_values[0] if q_values else float("nan"),
                f"rollout/q_at_terminal_step": q_values[-1] if q_values else float("nan"),
                "rollout/ep": ep,
            })

        summary["success"].append(int(success))
        summary["q_at_t0"].append(q_values[0] if q_values else float("nan"))
        summary["q_at_terminal_step"].append(q_values[-1] if q_values else float("nan"))
        summary["ep_len"].append(ep_len)

    succ_rate = np.mean(summary["success"]) if summary["success"] else 0.0
    print(
        f"[viz_q] done. success rate {succ_rate:.2f} "
        f"({sum(summary['success'])}/{args.n_rollouts}). artifacts in {args.output_dir}",
        flush=True,
    )

    # Aggregate plot: success vs failure Q traces (skip in debug mode — no json
    # traces were written).
    if not args.debug and any(p.exists() for p in args.output_dir.glob("ep_*_trace.json")):
        try:
            from plot_q_traces import make_aggregate_plot  # local import — same dir
            agg_path = make_aggregate_plot(args.output_dir, args.output_dir / "aggregate_q_traces.png")
            print(f"[viz_q] aggregate plot → {agg_path}", flush=True)
            if wandb_run is not None:
                import wandb
                wandb_run.log({"rollout/aggregate_q_traces": wandb.Image(str(agg_path))})
        except Exception as exc:
            print(f"[viz_q] WARN aggregate plot failed: {exc}", flush=True)

    if wandb_run is not None:
        import wandb
        wandb_run.summary["success_rate"] = float(succ_rate)
        wandb_run.summary["n_rollouts"] = args.n_rollouts
        wandb_run.finish()


if __name__ == "__main__":
    import traceback as _tb
    try:
        sys.exit(main() or 0)
    except Exception:
        # Any uncaught exception otherwise gets swallowed by mujoco/EGL teardown noise.
        print("[viz_q] FATAL uncaught exception:", flush=True)
        _tb.print_exc()
        sys.exit(1)
