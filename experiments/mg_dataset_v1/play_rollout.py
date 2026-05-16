#!/usr/bin/env python
"""Roll out a BC checkpoint in MimicGenEnv and dump per-step trajectories
to robomimic-format HDF5 (demo.hdf5 / demo_failed.hdf5).

Output shape mirrors MimicGen's generate_dataset output for q5/q3_termjitter
so downstream BC + Q-function loaders can treat all buckets identically:

  data/
    attrs: env_args (json), bucket="play", total
    demo_<i>/
      attrs: num_samples=T, bucket="play", success=bool
      actions      (T, 7)        float32
      rewards      (T,)          float32   <- env reward; loader may override per bucket
      obs/
        agentview_image           (T, 84, 84, 3)  uint8   <- mujoco-raw orientation
        robot0_eye_in_hand_image  (T, 84, 84, 3)  uint8   <- mujoco-raw orientation
        robot0_eef_pos            (T, 3)          float64
        robot0_eef_quat           (T, 4)          float64
        robot0_gripper_qpos       (T, 2)          float64
        robot0_joint_pos          (T, 7)          float64
        ... and any other low-dim obs that robosuite exposes for the task.

Image arrays are flipped vertically before storage so they match the
mujoco-raw upside-down orientation that MG's demos use — this is critical
because the BC policy expects that orientation at train time.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

# Lerobot policy + processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.constants import ACTION
from lerobot.envs.mimicgen import MimicGenEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


def _load_policy(ckpt_dir: str):
    cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(ckpt_dir)
    policy.eval()
    return policy, cfg


def _flip_images(raw_obs: dict) -> dict:
    """Flip image entries vertically (right-side-up → mujoco-raw)."""
    out = {}
    for k, v in raw_obs.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 3 and v.dtype == np.uint8:
            out[k] = np.ascontiguousarray(v[::-1])
        else:
            out[k] = v
    return out


def _filter_obs_keys(raw_obs: dict) -> dict:
    """Drop keys we never want in the dataset (depth maps, redundant copies)."""
    SKIP = {"frontview_image", "birdview_image"}  # in case extra cameras get rendered
    return {k: v for k, v in raw_obs.items() if k not in SKIP and not k.endswith("_depth")}


def _pack_episode_arrays(raw_obs_seq: list[dict], actions: list[np.ndarray], rewards: list[float]) -> dict:
    T = len(actions)
    assert len(raw_obs_seq) == T, f"obs/action length mismatch {len(raw_obs_seq)} vs {T}"
    keys = sorted(raw_obs_seq[0].keys())
    obs_stacked = {}
    for k in keys:
        try:
            obs_stacked[k] = np.stack([raw_obs_seq[t][k] for t in range(T)])
        except Exception:
            # Skip keys whose contents aren't stack-friendly across time.
            continue
    return {
        "actions": np.stack(actions).astype(np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "obs": obs_stacked,
    }


def _open_for_append(path: Path, env_name: str, bucket: str) -> h5py.File:
    """Open (or create) the per-bucket HDF5 with the standard top-level layout."""
    if path.exists():
        f = h5py.File(path, "a")
        if "data" not in f:
            grp = f.create_group("data")
            grp.attrs["env_args"] = json.dumps({"env_name": env_name, "bucket": bucket})
            grp.attrs["bucket"] = bucket
            grp.attrs["total"] = 0
        return f
    f = h5py.File(path, "w")
    grp = f.create_group("data")
    grp.attrs["env_args"] = json.dumps({"env_name": env_name, "bucket": bucket})
    grp.attrs["bucket"] = bucket
    grp.attrs["total"] = 0
    return f


def _append_demo(f: h5py.File, demo: dict, bucket: str):
    grp = f["data"]
    # Find next free demo index by counting existing demo_<N> groups.
    next_i = sum(1 for k in grp.keys() if k.startswith("demo_"))
    T = len(demo["actions"])
    d = grp.create_group(f"demo_{next_i}")
    d.attrs["num_samples"] = T
    d.attrs["bucket"] = bucket
    d.attrs["success"] = bool(demo["success"])
    d.create_dataset("actions", data=demo["actions"])
    d.create_dataset("rewards", data=demo["rewards"])
    obs_g = d.create_group("obs")
    for k, v in demo["obs"].items():
        if v.dtype == np.uint8 and v.ndim == 4:
            obs_g.create_dataset(k, data=v, compression="gzip", compression_opts=4)
        else:
            obs_g.create_dataset(k, data=v)
    grp.attrs["total"] = int(grp.attrs.get("total", 0)) + T
    f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="MimicGen task name, e.g. Square_D0")
    ap.add_argument("--ckpt", required=True, help="Path to BC checkpoint directory")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_episodes", type=int, default=200)
    ap.add_argument("--start_seed", type=int, default=100000)
    ap.add_argument("--episode_length", type=int, default=400)
    ap.add_argument("--bucket_name", default="play")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.task}] loading policy from {args.ckpt}", flush=True)
    policy, policy_cfg = _load_policy(args.ckpt)
    device = next(policy.parameters()).device
    print(f"[{args.task}] policy.device={device}", flush=True)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    print(f"[{args.task}] building env", flush=True)
    env = MimicGenEnv(
        env_name=args.task,
        camera_name=["agentview", "robot0_eye_in_hand"],
        observation_height=84,
        observation_width=84,
        max_episode_steps=args.episode_length,
        control_freq=20,
    )

    success_path = output_dir / "demo.hdf5"
    failed_path = output_dir / "demo_failed.hdf5"
    # Reset any previous partial files so a rerun starts clean.
    for p in (success_path, failed_path):
        if p.exists():
            p.unlink()
    success_f = _open_for_append(success_path, env_name=args.task, bucket=args.bucket_name)
    failed_f = _open_for_append(failed_path, env_name=args.task, bucket=args.bucket_name)

    n_success = 0
    n_failed = 0
    ep_lengths: list[int] = []

    t_start = time.time()
    for ep in range(args.n_episodes):
        seed = args.start_seed + ep
        gym_obs, _ = env.reset(seed=seed)
        policy.reset()

        raw_obs_seq = []
        actions = []
        rewards = []
        ep_success = False
        steps_taken = 0

        for step in range(args.episode_length):
            # Record raw obs (with image-keys flipped to mujoco-raw, matching MG demos).
            raw_obs = _flip_images(_filter_obs_keys(env._env._get_observations()))
            raw_obs_seq.append(raw_obs)

            # Build batch from gym obs and run policy.
            obs_batch = {
                "pixels": {k: v[None] for k, v in gym_obs["pixels"].items()},
                "agent_pos": gym_obs["agent_pos"][None],
            }
            obs_batch = preprocess_observation(obs_batch)
            obs_batch["task"] = [env.task_description]
            obs_batch = preprocessor(obs_batch)

            with torch.no_grad():
                action_t = policy.select_action(obs_batch)
            # `postprocessor` operates on the bare PolicyAction tensor, not a dict.
            # (lerobot_eval distinguishes this from `env_postprocessor`, which is a
            # transition-dict pipeline; for MimicGen the env-side postprocessor is
            # identity so we skip it.)
            action_t = postprocessor(action_t)
            action = action_t.detach().to("cpu").numpy()[0].astype(np.float32)

            actions.append(action)
            gym_obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            steps_taken += 1
            if info.get("is_success", False):
                ep_success = True
            if terminated or truncated:
                break

        packed = _pack_episode_arrays(raw_obs_seq, actions, rewards)
        packed["success"] = ep_success
        # Append to the per-bucket HDF5 immediately so a job timeout still leaves
        # a valid partial dataset.
        if ep_success:
            _append_demo(success_f, packed, bucket=args.bucket_name)
            n_success += 1
        else:
            _append_demo(failed_f, packed, bucket=args.bucket_name)
            n_failed += 1
        ep_lengths.append(steps_taken)

        elapsed = time.time() - t_start
        print(
            f"[{args.task}] ep {ep+1}/{args.n_episodes}  "
            f"len={steps_taken}  success={ep_success}  "
            f"running_succ={n_success/(ep+1):.2f}  "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )

    success_f.close()
    failed_f.close()
    env.close()

    stats = {
        "task": args.task,
        "ckpt": args.ckpt,
        "bucket": args.bucket_name,
        "n_episodes": args.n_episodes,
        "num_success": n_success,
        "num_failures": n_failed,
        "success_rate": n_success / max(1, n_success + n_failed),
        "ep_length_mean": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
    }
    (output_dir / "important_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
