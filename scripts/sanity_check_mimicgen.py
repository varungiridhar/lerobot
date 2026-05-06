#!/usr/bin/env python
"""Sanity-check the MimicGen <-> LeRobot pipeline without running training.

Performs four checks:

  1. **Dataset batch**   — load one batch from the converted LeRobot dataset
                          and dump shapes / dtypes / value ranges.
  2. **Env obs**         — make the env, reset, and dump observation after
                          ``preprocess_observation``.
  3. **Parity**          — assert keys+per-sample shapes line up between (1)
                          and (2). Catches silent train/eval mismatches.
  4. **Demo replay**     — set the simulator to a recorded init state, replay
                          that demo's recorded actions, and check the env
                          reaches success. Also compares the env's emitted
                          state to the dataset's recorded state at each step.

Example:

    python scripts/sanity_check_mimicgen.py \\
        --repo_id local/mimicgen_coffee_source \\
        --dataset_root /storage/.../mimicgen_coffee_source \\
        --task Coffee_D0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.factory import make_env, make_env_config
from lerobot.envs.utils import preprocess_observation


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def _summarize(t) -> str:
    if isinstance(t, torch.Tensor):
        f = t.float()
        return (
            f"shape={tuple(t.shape)} dtype={t.dtype} "
            f"min={f.min().item():+.3f} max={f.max().item():+.3f} mean={f.mean().item():+.3f}"
        )
    if isinstance(t, np.ndarray):
        return (
            f"shape={t.shape} dtype={t.dtype} "
            f"min={t.min():+.3f} max={t.max():+.3f} mean={t.mean():+.3f}"
        )
    return f"{type(t).__name__}: {t!r}"


def _print_dict(d, indent="  ", prefix="") -> None:
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, dict):
            _print_dict(v, indent=indent, prefix=f"{prefix}{k}/")
        else:
            print(f"{indent}{prefix}{k:<48s} {_summarize(v)}")


def _section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Part 1: dataset batch
# ---------------------------------------------------------------------------

def part1_dataset(repo_id: str, root: str, batch_size: int = 4, video_backend: str = "pyav"):
    _section("PART 1 — dataset batch (what training sees)")
    ds = LeRobotDataset(repo_id=repo_id, root=root, video_backend=video_backend)
    print(f"  episodes={ds.num_episodes}  frames={ds.num_frames}  fps={ds.fps}")
    print(f"  features keys: {sorted(ds.features.keys())}")
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=0)
    batch = next(iter(loader))
    # Drop the string `task` key — torch dataloader returns it as a list, not a tensor
    batch = {k: v for k, v in batch.items() if not isinstance(v, list)}
    _print_dict(batch)
    return ds, batch


# ---------------------------------------------------------------------------
# Part 2: env reset obs
# ---------------------------------------------------------------------------

def part2_env_obs(task: str, init_states_path: str | None, seed: int = 0):
    _section("PART 2 — env reset obs after preprocess_observation (what eval sees)")
    cfg = make_env_config("mimicgen", task=task, init_states_path=init_states_path)
    vec_envs = make_env(cfg, n_envs=1)
    vec_env = vec_envs[task][0]
    obs, _ = vec_env.reset(seed=seed)
    print(f"  raw env obs keys: {sorted(obs.keys())}")
    if isinstance(obs.get("pixels"), dict):
        print(f"  raw pixels sub-keys: {sorted(obs['pixels'].keys())}")
    obs_proc = preprocess_observation(obs)
    _print_dict(obs_proc)
    vec_env.close()
    return obs_proc


# ---------------------------------------------------------------------------
# Part 3: train vs eval parity
# ---------------------------------------------------------------------------

def part3_parity(batch: dict, env_obs: dict) -> bool:
    _section("PART 3 — train vs eval obs parity")
    train_keys = {k for k in batch if k.startswith("observation.")}
    eval_keys = set(env_obs.keys())

    missing = train_keys - eval_keys
    extra = eval_keys - train_keys
    common = train_keys & eval_keys

    ok = not missing and not extra
    if missing:
        print(f"  [WARN] in dataset but missing from env: {sorted(missing)}")
    if extra:
        print(f"  [WARN] in env but missing from dataset: {sorted(extra)}")

    for k in sorted(common):
        b_per = tuple(batch[k].shape[1:])
        e_per = tuple(env_obs[k].shape[1:])
        match = b_per == e_per
        flag = " OK " if match else "FAIL"
        print(f"  [{flag}] {k:<40s} dataset={b_per}  env={e_per}")
        ok = ok and match
    return ok


# ---------------------------------------------------------------------------
# Part 4a: demo replay through the simulator
# ---------------------------------------------------------------------------

def _episode_indices(ds: LeRobotDataset, episode_idx: int) -> list[int]:
    """Return the absolute frame indices of `episode_idx`."""
    ds._ensure_hf_dataset_loaded()
    ep_col = ds.hf_dataset["episode_index"]
    if isinstance(ep_col, torch.Tensor):
        ep_arr = ep_col.numpy()
    else:
        ep_arr = np.asarray(ep_col)
    return np.flatnonzero(ep_arr == episode_idx).tolist()


def _state_from_robosuite(raw_obs: dict) -> np.ndarray:
    """Mirror MimicGenEnv._format_raw_obs state assembly: 8-D flat vector."""
    eef_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float64)
    eef_quat = np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float64)
    gripper_qpos = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float64)
    axis_angle = Rotation.from_quat(eef_quat).as_rotvec()
    return np.concatenate([eef_pos, axis_angle, gripper_qpos]).astype(np.float64)


def part4a_replay(
    ds: LeRobotDataset,
    task: str,
    init_states_path: str | None,
    episode_idx: int,
    state_atol: float = 1e-3,
) -> None:
    _section(f"PART 4a — replay episode {episode_idx} (init_state + recorded actions)")

    indices = _episode_indices(ds, episode_idx)
    if not indices:
        print(f"  ERROR: episode {episode_idx} not present in dataset.")
        return
    actions = np.stack([np.asarray(ds[i]["action"], dtype=np.float32) for i in indices])
    states_recorded = np.stack(
        [np.asarray(ds[i]["observation.state"], dtype=np.float64) for i in indices]
    )
    print(f"  episode length={len(actions)}  action_dim={actions.shape[1]}")

    # Build a single non-vectorized MimicGenEnv just to get the loaded init states
    # and the underlying robosuite env. Then bypass the gym wrapper to set state
    # by index (np_random selection isn't suitable for a deterministic replay).
    from lerobot.envs.mimicgen import MimicGenEnv

    env = MimicGenEnv(env_name=task, init_states_path=init_states_path)
    if env._init_states is None:
        print("  ERROR: env did not load init states (init_states_path missing?).")
        env.close()
        return
    if episode_idx >= len(env._init_states):
        print(f"  ERROR: only {len(env._init_states)} init states available.")
        env.close()
        return

    # Force-set sim state to the requested demo's start, mirroring what
    # MimicGenEnv.reset() does when init_states_path is configured: load the
    # per-demo XML model first, then restore the flattened sim state.
    env._env.reset()
    if env._model_files is not None and episode_idx < len(env._model_files) and env._model_files[episode_idx]:
        xml = env._model_files[episode_idx]
        if hasattr(env._env, "edit_model_xml"):
            xml = env._env.edit_model_xml(xml)
        else:
            from robosuite.utils.mjcf_utils import postprocess_model_xml
            xml = postprocess_model_xml(xml)
        env._env.reset_from_xml_string(xml)
        env._env.sim.reset()
    state_vec = env._init_states[episode_idx].numpy()
    env._env.sim.set_state_from_flattened(state_vec)
    env._env.sim.forward()
    raw_obs = env._env._get_observations()

    # Initial-state parity (frame 0 of the dataset).
    s_env0 = _state_from_robosuite(raw_obs)
    s_ds0 = states_recorded[0]
    init_mae = float(np.abs(s_env0 - s_ds0).mean())
    print(f"  init state MAE (env vs dataset frame 0): {init_mae:.5f}")
    labels = ["eef_x", "eef_y", "eef_z", "aa_x", "aa_y", "aa_z", "grip_l", "grip_r"]
    print("  per-dim init delta (env - dataset):")
    for i, lab in enumerate(labels):
        print(f"    {lab:7s}  env={s_env0[i]:+.5f}  ds={s_ds0[i]:+.5f}  delta={s_env0[i] - s_ds0[i]:+.5f}")
    # Also dump the raw quaternion for debugging axis-angle sign-flip
    print(f"  raw eef_quat from env: {np.asarray(raw_obs['robot0_eef_quat'])}")

    # Per-step replay.
    success_step = None
    state_mae_running = 0.0
    state_mae_count = 0
    state_mae_max = 0.0
    for t, a in enumerate(actions):
        raw_obs, _, _, _ = env._env.step(a)

        if hasattr(env._env, "is_success"):
            sd = env._env.is_success()
            ok = sd.get("task", False) if isinstance(sd, dict) else bool(sd)
        else:
            ok = env._env._check_success()

        if t + 1 < len(actions):
            s_env = _state_from_robosuite(raw_obs)
            s_ds = states_recorded[t + 1]
            mae = float(np.abs(s_env - s_ds).mean())
            state_mae_running += mae
            state_mae_count += 1
            state_mae_max = max(state_mae_max, mae)

        if ok and success_step is None:
            success_step = t + 1

    avg_mae = state_mae_running / max(state_mae_count, 1)
    print(f"  per-step state MAE  avg={avg_mae:.5f}  max={state_mae_max:.5f}  (atol={state_atol})")
    if success_step is not None:
        print(f"  [ OK ] success at step {success_step}/{len(actions)}")
    else:
        print(f"  [FAIL] no success after {len(actions)} steps")
    env.close()


# ---------------------------------------------------------------------------
# Part 5: reproducibility (5a reset / 5b step / 5c seed diversity)
# ---------------------------------------------------------------------------

def part5_reproducibility(
    task: str,
    init_states_path: str | None,
    seed: int = 42,
    n_steps: int = 20,
    n_seeds: int = 10,
) -> None:
    _section("PART 5 — reproducibility")
    cfg = make_env_config("mimicgen", task=task, init_states_path=init_states_path)
    vec_envs = make_env(cfg, n_envs=1)
    vec_env = vec_envs[task][0]

    # ---- 5a: reset determinism ----
    print(f"\n  -- 5a: reset determinism (seed={seed}, two consecutive resets) --")
    obs_a, _ = vec_env.reset(seed=seed)
    obs_b, _ = vec_env.reset(seed=seed)
    a_proc = preprocess_observation(obs_a)
    b_proc = preprocess_observation(obs_b)
    for k in sorted(a_proc.keys()):
        a, b = a_proc[k], b_proc[k]
        if torch.equal(a, b):
            print(f"    [ OK ] {k:<35s} bit-equal")
        else:
            mx = (a - b).abs().float().max().item()
            nd = (a != b).sum().item()
            print(f"    [FAIL] {k:<35s} max|Δ|={mx:.6e}  diffs={nd}/{a.numel()}")

    # ---- 5b: step determinism ----
    print(f"\n  -- 5b: step determinism (seed={seed}, {n_steps} fixed actions, two rollouts) --")
    rng = np.random.default_rng(0)
    actions = np.clip(
        rng.normal(0.0, 0.1, size=(n_steps, 7)).astype(np.float32), -1.0, 1.0
    )

    def _rollout() -> list[dict]:
        obs, _ = vec_env.reset(seed=seed)
        traj = [preprocess_observation(obs)]
        for a in actions:
            obs, _, _, _, _ = vec_env.step(a[None, :])
            traj.append(preprocess_observation(obs))
        return traj

    traj_a = _rollout()
    traj_b = _rollout()

    n_diff_steps = 0
    first_div = None
    for t, (a, b) in enumerate(zip(traj_a, traj_b)):
        for k in a:
            if not torch.equal(a[k], b[k]):
                if first_div is None:
                    first_div = (t, k, (a[k] - b[k]).abs().float().max().item())
                n_diff_steps += 1
                break
    if n_diff_steps == 0:
        print(f"    [ OK ] all {len(traj_a)} timesteps bit-equal across both rollouts")
    else:
        t, k, m = first_div
        print(f"    [FAIL] {n_diff_steps}/{len(traj_a)} timesteps differ;"
              f" first divergence at step {t} on '{k}' max|Δ|={m:.6e}")

    # ---- 5c: seed diversity ----
    print(f"\n  -- 5c: seed diversity (seeds 0..{n_seeds - 1}) --")
    eef = []
    for s in range(n_seeds):
        obs, _ = vec_env.reset(seed=s)
        eef.append(np.asarray(obs["agent_pos"]).reshape(-1, 8)[0, :3].copy())
    eef = np.stack(eef)
    for i, lab in enumerate(["x", "y", "z"]):
        col = eef[:, i]
        print(f"    eef_{lab}: range=[{col.min():+.4f}, {col.max():+.4f}]  std={col.std():.4f}")
    rounded = np.round(eef, 4)
    unique = len({tuple(r) for r in rounded})
    threshold = max(3, n_seeds // 2)
    flag = " OK " if unique >= threshold else "WARN"
    print(f"    [{flag}] {unique}/{n_seeds} distinct init eef positions (threshold≥{threshold})")

    vec_env.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", required=True)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--task", default="Coffee_D0")
    p.add_argument("--init_states_path", default=None)
    p.add_argument("--episode_idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--video_backend", default="pyav",
                   help="Video decoder backend (default 'pyav' avoids torchcodec/ffmpeg issues).")
    p.add_argument("--skip_replay", action="store_true",
                   help="Skip part 4a (env replay) — useful if robosuite isn't installed.")
    p.add_argument("--skip_repro", action="store_true",
                   help="Skip part 5 (reproducibility checks).")
    p.add_argument("--repro_seed", type=int, default=42)
    p.add_argument("--repro_n_steps", type=int, default=20)
    p.add_argument("--repro_n_seeds", type=int, default=10)
    args = p.parse_args()

    if args.init_states_path is None:
        args.init_states_path = str(Path(args.dataset_root) / "meta" / "init_states.pt")
    if not Path(args.init_states_path).exists():
        print(f"  [WARN] init_states_path does not exist: {args.init_states_path}")
        args.init_states_path = None

    ds, batch = part1_dataset(args.repo_id, args.dataset_root, batch_size=args.batch_size, video_backend=args.video_backend)
    env_obs = part2_env_obs(args.task, args.init_states_path, seed=args.seed)
    part3_parity(batch, env_obs)
    if not args.skip_replay and args.init_states_path is not None:
        part4a_replay(ds, args.task, args.init_states_path, args.episode_idx)
    if not args.skip_repro:
        part5_reproducibility(
            args.task,
            args.init_states_path,
            seed=args.repro_seed,
            n_steps=args.repro_n_steps,
            n_seeds=args.repro_n_seeds,
        )


if __name__ == "__main__":
    main()
