"""Roll out ACT-Simple with Q-scored MPPI/CEM planning and record per-chunk traces.

Mirrors ``visualize_q_rollout.py`` but with ``bc_policy.enable_planning(...)`` engaged.
Per-step planning is hidden inside ``select_action``; we additionally hook the BC
chunk boundary to record the Q value of the chosen plan so we can plot a Q-trace
analogous to the visualization script.

Usage::

    python -u experiments/mg_dataset_v1/eval_planning.py \\
        --task threading \\
        --bc_checkpoint outputs/train/mimicgen_threading_d0_act_simple/checkpoints/100000/pretrained_model \\
        --q_checkpoint  outputs/train/mimicgen_threading_d0_q_function_h10/checkpoints/last/pretrained_model \\
        --output_dir    outputs/eval/mimicgen_threading_d0_q_function_h10/planning_$(date +%Y%m%d_%H%M%S) \\
        --n_rollouts 10 \\
        --planner_type mppi --n_samples 64 --noise_std 0.3 --temperature 1.0 \\
        [--baseline] [--wandb_enable] [--debug]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np                       # noqa: E402
import torch                             # noqa: E402

try:
    import imageio_ffmpeg as _iio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = _iio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

from lerobot.envs.mimicgen import MimicGenEnv  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.policies.act_simple.modeling_act_simple import ACTSimplePolicy  # noqa: E402
from lerobot.policies.act_simple.planning import PlannerConfig  # noqa: E402
from lerobot.policies.q_function.modeling_q_function import QFunctionPolicy  # noqa: E402
from lerobot.processor import PolicyProcessorPipeline  # noqa: E402
from lerobot.processor.converters import (  # noqa: E402
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import (  # noqa: E402
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

TASK_TO_ENV = {"square": "Square_D0", "threading": "Threading_D0", "coffee": "Coffee_D0"}


def _load_pre(ckpt_dir: Path) -> PolicyProcessorPipeline:
    return PolicyProcessorPipeline.from_pretrained(
        str(ckpt_dir), config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        to_transition=batch_to_transition, to_output=transition_to_batch,
    )


def _load_post(ckpt_dir: Path) -> PolicyProcessorPipeline:
    return PolicyProcessorPipeline.from_pretrained(
        str(ckpt_dir), config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action,
    )


def load_bc(checkpoint_dir: Path, device: torch.device):
    policy = ACTSimplePolicy.from_pretrained(str(checkpoint_dir)).to(device).eval()
    return policy, _load_pre(checkpoint_dir), _load_post(checkpoint_dir)


def load_q(checkpoint_dir: Path, device: torch.device):
    policy = QFunctionPolicy.from_pretrained(str(checkpoint_dir)).to(device).eval()
    return policy, _load_pre(checkpoint_dir)


def _probe_dump(step: int, action: np.ndarray, raw_obs_robosuite: dict, q_val: float | None,
                bc_chunk: np.ndarray | None) -> None:
    """Per-step deterministic-probe dump. Compact dense fingerprint for diffing."""
    eef_pos = raw_obs_robosuite.get("robot0_eef_pos", np.zeros(3))
    eef_quat = raw_obs_robosuite.get("robot0_eef_quat", np.zeros(4))
    gripper = raw_obs_robosuite.get("robot0_gripper_qpos", np.zeros(2))
    parts = [f"step={step:03d}"]
    parts.append("action=[" + ",".join(f"{v:+.8f}" for v in action) + "]")
    parts.append("eef_pos=[" + ",".join(f"{v:+.8f}" for v in eef_pos) + "]")
    parts.append("eef_quat=[" + ",".join(f"{v:+.8f}" for v in eef_quat) + "]")
    parts.append("gripper=[" + ",".join(f"{v:+.8f}" for v in gripper) + "]")
    for key in sorted(raw_obs_robosuite.keys()):
        if key.startswith(("robot0_", "image", "depth")) or key.endswith(("_image", "_depth")):
            continue
        val = raw_obs_robosuite[key]
        if isinstance(val, np.ndarray) and val.dtype.kind in "fi" and val.size <= 16:
            parts.append(f"{key}=[" + ",".join(f"{v:+.8f}" for v in val.flatten()) + "]")
    if q_val is not None:
        parts.append(f"q_executed={q_val:+.8f}")
    if bc_chunk is not None:
        # First action of the chunk (already in normalized space): summary hash.
        h = float(np.abs(bc_chunk).sum())
        parts.append(f"bc_chunk_norm_abs_sum={h:+.8f}")
    print("[probe] " + " ".join(parts), flush=True)


def rollout(bc_policy, bc_pre, bc_post, env, max_steps: int, device: torch.device,
            q_policy=None, q_pre=None, probe_debug: bool = False,
            reset_seed: int | None = None):
    """One rollout. Returns (rgb_frames, executed_actions, success, q_steps, q_values, q_spreads).

    At each chunk boundary we also score Q on the chunk that BC is about to
    execute (the planned chunk, when planning is on; the deterministic chunk, in
    baseline mode). ``q_spreads`` records (q_min, q_max, q_mean, q_std) across
    candidates from the most recent planner step — empty in baseline mode.
    """
    raw_obs, _info = env.reset(seed=reset_seed) if reset_seed is not None else env.reset()
    bc_policy.reset()

    rgb_frames: list[np.ndarray] = [env.render()]
    executed_actions: list[np.ndarray] = []
    q_steps: list[int] = []
    q_values: list[float] = []
    q_spreads: list[tuple[float, float, float, float] | None] = []
    success = False
    n_action_steps = int(bc_policy.config.n_action_steps)
    h = int(q_policy.config.h) if q_policy is not None else None

    for t in range(max_steps):
        batch = preprocess_observation(raw_obs)
        batch["task"] = [env.task_description]
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch = bc_pre(batch)

        # At chunk boundaries, generate THE chunk once (running the planner if enabled),
        # score Q on it, then prime BC's action queue with that chunk so select_action
        # below pops from it (and does NOT re-invoke the planner). This guarantees the
        # Q value we record matches the chunk BC actually executes.
        at_boundary = (t % n_action_steps == 0)
        if at_boundary:
            chunk = bc_policy.predict_action_chunk(batch)  # (1, chunk_size, A) normalized
            # Prime BC's action queue so select_action just pops.
            bc_policy._action_queue.extend(chunk[:, : n_action_steps].transpose(0, 1))
            chunk_np = chunk.detach().cpu().numpy()  # for probe

            if q_policy is not None:
                chunk_to_score = chunk[:, : h, :]
                chunk_raw = bc_post(chunk_to_score.reshape(-1, chunk_to_score.shape[-1])).reshape(chunk_to_score.shape)
                q_batch = {"action": chunk_raw.to(device)}
                for cam_key in q_policy.config.camera_keys:
                    img = batch[cam_key]
                    if img.dim() == 3:
                        img = img.unsqueeze(0)
                    q_batch[f"{cam_key}_preencoded"] = bc_policy.model.backbone(img)["feature_map"]
                q_batch = q_pre(q_batch)
                q_val = float(q_policy.predict_value(q_batch).item())
                q_steps.append(t)
                q_values.append(q_val)
                # Capture planner internal diagnostics if available (planning mode only).
                spread = getattr(bc_policy, "_last_q_spread", None)
                q_spreads.append(spread)

        action = bc_policy.select_action(batch)
        action = bc_post(action).cpu().squeeze(0).numpy()
        raw_obs, _reward, terminated, truncated, info = env.step(action)
        if probe_debug:
            # Use robosuite's full raw obs for state (richer than wrapper's filtered).
            raw_robosuite = env._env._get_observations()
            _probe_dump(
                step=t,
                action=action,
                raw_obs_robosuite=raw_robosuite,
                q_val=(q_values[-1] if q_values and at_boundary else None),
                bc_chunk=(chunk_np if at_boundary else None),
            )
        executed_actions.append(action.copy())
        rgb_frames.append(env.render())
        if info.get("is_success"):
            success = True
        if terminated or truncated:
            break

    return rgb_frames, np.stack(executed_actions), success, q_steps, q_values, q_spreads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=tuple(TASK_TO_ENV))
    ap.add_argument("--bc_checkpoint", required=True, type=Path)
    ap.add_argument("--q_checkpoint", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--n_rollouts", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--init_states_path", type=Path, default=None)
    ap.add_argument("--render_size", type=int, default=256)
    ap.add_argument("--fps", type=int, default=20)

    # Planning knobs.
    ap.add_argument("--planner_type", choices=("mppi", "cem", "argmax"), default="mppi")
    ap.add_argument("--n_samples", type=int, default=64)
    ap.add_argument("--noise_std", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--n_elites", type=int, default=16)
    ap.add_argument("--n_iters", type=int, default=3)
    ap.add_argument("--clip_to", type=float, default=None)
    ap.add_argument("--planner_seed", type=int, default=None)
    ap.add_argument("--noise_smooth_sigma_t", type=float, default=None,
                    help="If set, low-pass filter sampled noise along the time axis "
                         "with a 1D Gaussian kernel of this bandwidth.")

    ap.add_argument("--baseline", action="store_true",
                    help="Skip enable_planning; run deterministic BC. Useful for A/B.")
    ap.add_argument("--wandb_enable", action="store_true")
    ap.add_argument("--debug", action="store_true",
                    help="Skip wandb + file writes; print summary only.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--deterministic", action="store_true",
                    help="Force bit-exact reproducibility: cudnn.deterministic, "
                         "torch.use_deterministic_algorithms, fixed RNG. Slightly slower.")
    ap.add_argument("--probe_debug", action="store_true",
                    help="Single-rollout determinism probe: print per-step action + env "
                         "state to stdout. Forces n_rollouts=1, disables mp4/json/wandb.")
    ap.add_argument("--no_env_seed", action="store_true",
                    help="Don't pass seed= to env.reset(). For determinism probing only.")
    args = ap.parse_args()
    if args.probe_debug:
        args.n_rollouts = 1
        args.wandb_enable = False
        args.debug = True  # skips mp4 + json writes

    if args.deterministic:
        import os, random
        # cuBLAS: required so torch.use_deterministic_algorithms doesn't error.
        # Must be set BEFORE any CUDA matmul runs — we set it here even though
        # ideally it would be in the launcher (still works because eval_planning
        # hasn't done any matmul yet).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        print(f"[eval_planning] deterministic=True (cudnn det, use_deterministic_algorithms, "
              f"CUBLAS_WORKSPACE_CONFIG=:4096:8)", flush=True)

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_planning] device={device} task={args.task} planner={'baseline' if args.baseline else args.planner_type}", flush=True)
    print(f"[eval_planning] output_dir={args.output_dir}", flush=True)

    print(f"[eval_planning] loading BC from {args.bc_checkpoint}", flush=True)
    bc_policy, bc_pre, bc_post = load_bc(args.bc_checkpoint, device)
    print(f"[eval_planning] loading Q from {args.q_checkpoint}", flush=True)
    q_policy, q_pre = load_q(args.q_checkpoint, device)
    h = int(q_policy.config.h)
    cs = int(bc_policy.config.chunk_size)
    if h != cs:
        raise SystemExit(
            f"[eval_planning] mismatch: BC.chunk_size={cs} but Q.h={h}. "
            "Retrain Q at h={cs} (run_q_train_lerobot.sh with --policy.h={cs})."
        )

    if not args.baseline:
        planner_cfg = PlannerConfig(
            planner_type=args.planner_type,
            n_samples=args.n_samples,
            noise_std=args.noise_std,
            temperature=args.temperature,
            n_elites=args.n_elites,
            n_iters=args.n_iters,
            clip_to=args.clip_to,
            seed=args.planner_seed,
            noise_smooth_sigma_t=args.noise_smooth_sigma_t,
        )
        bc_policy.enable_planning(q_policy=q_policy, q_pre=q_pre, bc_post=bc_post, planner_cfg=planner_cfg)
        print(f"[eval_planning] planner enabled: {planner_cfg}", flush=True)
    else:
        print("[eval_planning] baseline mode — planning disabled", flush=True)

    # Env construction.
    env_name = TASK_TO_ENV[args.task]
    init_states_path = args.init_states_path or Path(
        f"/storage/home/hcoda1/6/vgiridhar6/shared/mimicgen/lerobot_datasets/v1_q5_q3jitter_play"
        f"/{args.task}/bc_input/lerobot_ds/meta/init_states.pt"
    )
    if not init_states_path.exists():
        print(f"[eval_planning] WARN init_states_path={init_states_path} missing — random init", flush=True)
        init_states_path = None
    env = MimicGenEnv(
        env_name=env_name,
        init_states_path=str(init_states_path) if init_states_path else None,
        render_height=args.render_size,
        render_width=args.render_size,
        max_episode_steps=args.max_steps,
    )

    # WandB.
    wandb_run = None
    if args.wandb_enable and not args.debug:
        import wandb
        run_tag = "baseline" if args.baseline else args.planner_type
        wandb_run = wandb.init(
            project="awm", entity="pair-diffusion", job_type="eval",
            name=f"eval_planning_{args.task}_{run_tag}_{int(time.time())}",
            config={**vars(args), "h": h, "chunk_size": cs},
        )

    # Rollouts.
    summary = {"success": [], "ep_len": [], "q_at_t0": [], "q_at_terminal_step": []}
    for ep in range(args.n_rollouts):
        env.action_space.seed(args.seed + ep)
        torch.manual_seed(args.seed + ep)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + ep)
        np.random.seed(args.seed + ep)
        import random as _random
        _random.seed(args.seed + ep)

        t0 = time.time()
        rgb_frames, actions, success, q_steps, q_values, q_spreads = rollout(
            bc_policy, bc_pre, bc_post, env, args.max_steps, device,
            q_policy=q_policy, q_pre=q_pre, probe_debug=args.probe_debug,
            reset_seed=(None if args.no_env_seed else args.seed + ep),
        )
        ep_len = len(rgb_frames)
        print(
            f"[eval_planning] ep {ep}/{args.n_rollouts}  success={success}  len={ep_len}  "
            f"q_n={len(q_steps)}  rollout_s={time.time() - t0:.2f}",
            flush=True,
        )

        if args.debug:
            if q_values:
                qh = [f"{v:.2f}" for v in q_values[:5]]
                qt = [f"{v:.2f}" for v in q_values[-5:]]
                print(f"[eval_planning]   q_head={qh} q_tail={qt}", flush=True)
            valid_spreads = [s for s in q_spreads if s is not None]
            if valid_spreads:
                std_vals = [s[3] for s in valid_spreads]
                rng_vals = [s[1] - s[0] for s in valid_spreads]
                print(
                    f"[eval_planning]   Q-spread across candidates  "
                    f"std_mean={float(np.mean(std_vals)):.2f}  range_mean={float(np.mean(rng_vals)):.2f}  "
                    f"std_max={float(np.max(std_vals)):.2f}",
                    flush=True,
                )
        else:
            video_path = args.output_dir / f"ep_{ep:02d}_success={int(success)}.mp4"
            from visualize_q_rollout import compose_video  # reuse
            compose_video(rgb_frames, q_steps, q_values, success, ep, args.fps, video_path)
            trace_path = args.output_dir / f"ep_{ep:02d}_trace.json"
            trace_path.write_text(json.dumps({
                "task": args.task, "ep": ep, "success": bool(success),
                "ep_len": ep_len, "q_steps": q_steps, "q_values": q_values,
                "q_spreads": q_spreads,
                "h": h, "chunk_stride": int(bc_policy.config.n_action_steps),
                "planner": ("baseline" if args.baseline else args.planner_type),
            }, indent=2))

            if wandb_run is not None:
                import wandb
                wandb_run.log({
                    "rollout/video": wandb.Video(str(video_path), fps=args.fps, format="mp4"),
                    "rollout/success": int(success),
                    "rollout/ep_len": ep_len,
                    "rollout/q_at_t0": q_values[0] if q_values else float("nan"),
                    "rollout/q_at_terminal_step": q_values[-1] if q_values else float("nan"),
                    "rollout/ep": ep,
                })

        summary["success"].append(int(success))
        summary["ep_len"].append(ep_len)
        summary["q_at_t0"].append(q_values[0] if q_values else float("nan"))
        summary["q_at_terminal_step"].append(q_values[-1] if q_values else float("nan"))

    succ_rate = float(np.mean(summary["success"])) if summary["success"] else 0.0
    print(
        f"[eval_planning] done. success rate {succ_rate:.2f} "
        f"({sum(summary['success'])}/{args.n_rollouts}). artifacts in {args.output_dir}",
        flush=True,
    )

    if not args.debug and any(args.output_dir.glob("ep_*_trace.json")):
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from plot_q_traces import make_aggregate_plot
            agg_path = make_aggregate_plot(args.output_dir, args.output_dir / "aggregate_q_traces.png")
            print(f"[eval_planning] aggregate plot → {agg_path}", flush=True)
            if wandb_run is not None:
                import wandb
                wandb_run.log({"rollout/aggregate_q_traces": wandb.Image(str(agg_path))})
        except Exception as exc:
            print(f"[eval_planning] WARN aggregate plot failed: {exc}", flush=True)

    if wandb_run is not None:
        wandb_run.summary["success_rate"] = succ_rate
        wandb_run.summary["n_rollouts"] = args.n_rollouts
        wandb_run.finish()


if __name__ == "__main__":
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        sys.exit(main() or 0)
    except Exception:
        print("[eval_planning] FATAL uncaught exception:", flush=True)
        traceback.print_exc()
        sys.exit(1)
