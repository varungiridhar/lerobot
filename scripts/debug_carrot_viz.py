"""Debug visualization: BC eval rollout (left) vs dataset carrot frame at t+H+1 (right).

Steps:
  1. Load a lerobot/pusht dataset episode (episode_idx = seed % n_episodes)
  2. Decode its video frames
  3. Infer initial state from frame 0: agent pos from observation.state col (exact),
     block centroid+angle via color detection on the 96x96 image
  4. Initialize eval env to that state
  5. Run BC policy (no planning); at step t show dataset frame min(t+H+1, ep_len-1) on right

Usage:
    python debug_carrot_viz.py --seed 0 --out outputs/carrot_debug.mp4

    --seed    : selects dataset episode (episode_idx = seed % 206) and env seed
    --out     : output video path
    --fps     : output fps (default 10)
    --horizon : carrot look-ahead H (default 16, = chunk_size)
"""
import argparse
import os

import imageio
import numpy as np
import torch

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

DATASET_ROOT = "/storage/home/hcoda1/7/igeorgiev3/.cache/huggingface/lerobot/lerobot/pusht"
POLICY_PATH = (
    "/storage/project/r-agarg35-0/shared/awm/"
    "act_simple_awm_pusht_wm1.0_l2norm_truly_deterministic/"
    "checkpoints/100000/pretrained_model"
)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_dataset_meta(root: str):
    import pyarrow.parquet as pq
    data = pq.read_table(f"{root}/data/chunk-000/file-000.parquet")
    eps  = pq.read_table(f"{root}/meta/episodes/chunk-000/file-000.parquet")
    return data, eps


def _decode_episode_frames(root: str, eps_table, ep_idx: int) -> list[np.ndarray]:
    """Return list of (96,96,3) uint8 frames for the given episode."""
    from lerobot.datasets.video_utils import decode_video_frames

    chunk  = eps_table["videos/observation.image/chunk_index"][ep_idx].as_py()
    fidx   = eps_table["videos/observation.image/file_index"][ep_idx].as_py()
    t_from = eps_table["videos/observation.image/from_timestamp"][ep_idx].as_py()
    length = eps_table["length"][ep_idx].as_py()
    fps    = 10.0

    video_path = f"{root}/videos/observation.image/chunk-{chunk:03d}/file-{fidx:03d}.mp4"
    timestamps = [t_from + i / fps for i in range(length)]
    tensors = decode_video_frames(video_path, timestamps, tolerance_s=0.12, backend="pyav")
    return [(t.permute(1, 2, 0).mul(255).byte().numpy()) for t in tensors]


def _get_episode_agent_pos(data_table, eps_table, ep_idx: int) -> np.ndarray:
    """Return initial agent pos [x, y] from observation.state col (exact)."""
    from_idx = eps_table["dataset_from_index"][ep_idx].as_py()
    return np.array(data_table["observation.state"][from_idx].as_py(), dtype=np.float32)


# ---------------------------------------------------------------------------
# Initial-state inference from image
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# T-shape geometry (from gym_pusht add_tee, scale=30)
# Local frame vertices (y-up, body origin at (0,0), NOT COG-centered)
# ---------------------------------------------------------------------------
_T_CROSSBAR = np.array([[-60.0, 0.0], [60.0, 0.0], [60.0, 30.0], [-60.0, 30.0]])
_T_STEM     = np.array([[-15.0, 30.0], [15.0, 30.0], [15.0, 120.0], [-15.0, 120.0]])
# Area-weighted centroid of the combined T in local frame (y-up)
_T_AREA1 = 120.0 * 30.0   # crossbar
_T_AREA2 =  30.0 * 90.0   # stem
_T_LOCAL_CY = (_T_AREA1 * 15.0 + _T_AREA2 * 75.0) / (_T_AREA1 + _T_AREA2)  # ≈ 40.714
_T_LOCAL_CX = 0.0


def _block_mask(img: np.ndarray) -> np.ndarray:
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return (
        (r > 80) & (r < 160) &
        (g > 100) & (g < 175) &
        (b > 120) & (b < 195) &
        ~(g > b + 40)
    )


def _render_t_mask(img_size: int, cx_img: float, cy_img: float, theta: float, env_size: int = 512) -> np.ndarray:
    """Binary mask of T with its area-centroid at (cx_img, cy_img) in the image, rotated by theta.

    Coordinate system: pymunk uses y-DOWN (no flip), so img_x = wx*s, img_y = wy*s.
    T area-centroid is at local (0, _T_LOCAL_CY) ≈ (0, 40.71) relative to body origin.

    Body position that places T centroid at (cx_img, cy_img) for rotation theta:
      bx = cx_img/s + _T_LOCAL_CY * sin(theta)   [CCW rotation of centroid offset]
      by = cy_img/s - _T_LOCAL_CY * cos(theta)
    """
    import cv2
    s = img_size / env_size
    c, sn = np.cos(theta), np.sin(theta)
    bx = cx_img / s + _T_LOCAL_CY * sn
    by = cy_img / s - _T_LOCAL_CY * c

    canvas = np.zeros((img_size, img_size), dtype=np.uint8)
    for verts in (_T_CROSSBAR, _T_STEM):
        wx = bx + verts[:, 0] * c - verts[:, 1] * sn
        wy = by + verts[:, 0] * sn + verts[:, 1] * c
        ix = (wx * s).round().astype(np.int32)
        iy = (wy * s).round().astype(np.int32)   # y-down: no flip
        cv2.fillPoly(canvas, [np.stack([ix, iy], axis=1)], 1)
    return canvas.astype(bool)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union > 0 else 0.0


def _infer_block_state_from_image(img: np.ndarray, env_size: int = 512) -> tuple[float, float, float]:
    """Find block (body_x, body_y, angle) by template-matching the known T polygon to the image mask.

    Returns body position in pymunk world coordinates and angle in radians.
    Coordinate mapping: img↔world via img_x=wx*(96/512), img_y=(512-wy)*(96/512).
    """
    img_size = img.shape[0]
    s = img_size / env_size

    obs_mask = _block_mask(img)
    if obs_mask.sum() < 10:
        return env_size / 2, env_size / 2, 0.0

    ys, xs = np.where(obs_mask)
    cx_img, cy_img = float(xs.mean()), float(ys.mean())

    # Coarse sweep: 2° steps
    best_theta, best_iou_val = 0.0, -1.0
    for deg in range(0, 360, 2):
        theta = np.deg2rad(deg)
        iou = _iou(obs_mask, _render_t_mask(img_size, cx_img, cy_img, theta, env_size))
        if iou > best_iou_val:
            best_iou_val, best_theta = iou, theta

    # Fine sweep: 0.25° steps around winner
    for ddeg in np.arange(-2.0, 2.25, 0.25):
        theta = best_theta + np.deg2rad(ddeg)
        iou = _iou(obs_mask, _render_t_mask(img_size, cx_img, cy_img, theta, env_size))
        if iou > best_iou_val:
            best_iou_val, best_theta = iou, theta

    # Recover body position from centroid + best angle (y-down, no flip)
    c, sn = np.cos(best_theta), np.sin(best_theta)
    bx = cx_img / s + _T_LOCAL_CY * sn
    by = cy_img / s - _T_LOCAL_CY * c
    return bx, by, best_theta


def _set_env_state(base, agent_pos: np.ndarray, block_x: float, block_y: float, block_angle: float):
    """Force pymunk bodies to the given state and settle physics.

    Angle must be set before position because pymunk's position setter is angle-dependent
    (it accounts for the body's center_of_gravity offset using the current angle).
    """
    base.block.angle = block_angle           # set angle FIRST
    base.block.position = (block_x, block_y)
    base.block.velocity = (0.0, 0.0)
    base.block.angular_velocity = 0.0
    base.agent.position = (float(agent_pos[0]), float(agent_pos[1]))
    base.agent.velocity = (0.0, 0.0)
    base.space.step(1e-6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0,
                        help="Selects dataset episode (seed %% n_episodes) and env seed")
    parser.add_argument("--out", default="outputs/carrot_debug.mp4")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=16, help="Carrot look-ahead H = chunk_size")
    args = parser.parse_args()

    import gymnasium as gym
    import gym_pusht  # noqa

    from lerobot import envs as lerobot_envs
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import add_envs_task, preprocess_observation
    from lerobot.utils.utils import get_safe_torch_device
    from lerobot.utils.random_utils import set_seed
    from lerobot.utils.constants import ACTION as _ACTION

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    set_seed(args.seed)

    device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu", log=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Loading dataset…")
    data_table, eps_table = _load_dataset_meta(DATASET_ROOT)
    n_episodes = len(eps_table)
    ep_idx = args.seed % n_episodes
    ep_len = eps_table["length"][ep_idx].as_py()
    print(f"Using dataset episode {ep_idx} (seed={args.seed}, ep_len={ep_len})")

    print("Decoding episode video frames…")
    ds_frames = _decode_episode_frames(DATASET_ROOT, eps_table, ep_idx)  # list of (96,96,3) uint8
    assert len(ds_frames) == ep_len, f"Frame count mismatch: {len(ds_frames)} vs {ep_len}"

    init_agent_pos = _get_episode_agent_pos(data_table, eps_table, ep_idx)
    block_cx, block_cy, block_angle = _infer_block_state_from_image(ds_frames[0])
    print(f"Inferred initial state — agent: {init_agent_pos}, block: ({block_cx:.1f}, {block_cy:.1f}), angle: {block_angle:.3f}")

    # ── Policy ────────────────────────────────────────────────────────────────
    print("Loading policy (BC only, no planning)…")
    policy_cfg = PreTrainedConfig.from_pretrained(POLICY_PATH)
    policy_cfg.pretrained_path = POLICY_PATH
    env_cfg = lerobot_envs.PushtEnv()
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy = policy.to(device)
    policy.eval()

    # ── Env ───────────────────────────────────────────────────────────────────
    print("Making env…")
    env = gym.make_vec(
        "gym_pusht/PushT-v0",
        num_envs=1,
        vectorization_mode="sync",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
    )
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=POLICY_PATH,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )
    max_steps = env.call("_max_episode_steps")[0]

    # ── Reset and force initial state ────────────────────────────────────────
    policy.reset()
    observation, _ = env.reset(seed=[args.seed])
    base = env.envs[0].unwrapped
    _set_env_state(base, init_agent_pos, block_cx, block_cy, block_angle)
    # Get observation in the correct 96x96 format via the env's own getter
    single_obs = base.get_obs()
    observation = {
        "pixels": np.stack([single_obs["pixels"]]),
        "agent_pos": np.stack([single_obs["agent_pos"]]),
    }
    print(f"Env initialized — agent: {observation['agent_pos'][0]}")

    # ── Side-by-side helper ──────────────────────────────────────────────────
    def side_by_side(eval_img: np.ndarray, ds_img: np.ndarray) -> np.ndarray:
        h, w = eval_img.shape[:2]
        if ds_img.shape[:2] != (h, w):
            from PIL import Image as _PIL
            ds_img = np.array(_PIL.fromarray(ds_img).resize((w, h)))
        return np.concatenate([eval_img, ds_img], axis=1)

    # ── Rollout ───────────────────────────────────────────────────────────────
    print("Running BC rollout…")
    video_frames = []
    done = False
    step = 0

    carrot_t = min(args.horizon + 1, ep_len - 1)
    video_frames.append(side_by_side(observation["pixels"][0], ds_frames[carrot_t]))

    while not done and step < max_steps:
        obs_proc = preprocess_observation(observation)
        obs_proc = add_envs_task(env, obs_proc)
        obs_proc = env_preprocessor(obs_proc)
        obs_proc = preprocessor(obs_proc)
        with torch.no_grad():
            action = policy.select_action(obs_proc)
        action = postprocessor(action)
        action_t = {_ACTION: action}
        action_t = env_postprocessor(action_t)
        action_np = action_t[_ACTION].cpu().numpy()  # (1, action_dim)

        observation, reward, terminated, truncated, _ = env.step(action_np)
        step += 1
        done = bool(terminated[0] or truncated[0])

        carrot_t = min(step + args.horizon + 1, ep_len - 1)
        video_frames.append(side_by_side(observation["pixels"][0], ds_frames[carrot_t]))

        if step % 50 == 0:
            print(f"  step={step}, reward={reward[0]:.3f}, done={done}")

    success = bool(terminated[0] and reward[0] > 0.9)
    print(f"Episode done at step {step}. Success: {success}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"Saving {len(video_frames)} frames → {args.out}")
    imageio.mimwrite(args.out, video_frames, fps=args.fps)
    print(f"Done. Video: {args.out}")


if __name__ == "__main__":
    main()
