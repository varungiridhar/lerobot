"""Environment-specific goal observation extraction for latent-space planning.

Each provider returns a goal observation dict in the same format that
`preprocess_observation` expects (i.e. the raw numpy arrays keyed by the
env's native keys).  The eval rollout can then pass the result through the
same preprocessor pipeline before calling ``policy.set_goal``.
"""
from __future__ import annotations

import glob
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import gymnasium as gym
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PushT T-shape template-matching helpers (for block-state inference from image)
# ---------------------------------------------------------------------------
# T geometry in local body frame (y-DOWN, pymunk convention, scale=30)
_T_CROSSBAR = np.array([[-60.0, 0.0], [60.0, 0.0], [60.0, 30.0], [-60.0, 30.0]])
_T_STEM = np.array([[-15.0, 30.0], [15.0, 30.0], [15.0, 120.0], [-15.0, 120.0]])
_T_AREA1 = 120.0 * 30.0
_T_AREA2 = 30.0 * 90.0
_T_LOCAL_CY = (_T_AREA1 * 15.0 + _T_AREA2 * 75.0) / (_T_AREA1 + _T_AREA2)  # ≈ 40.714


def _block_mask(img: np.ndarray) -> np.ndarray:
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return (
        (r > 80) & (r < 160) & (g > 100) & (g < 175) & (b > 120) & (b < 195) & ~(g > b + 40)
    )


def _render_t_mask(img_size: int, cx_img: float, cy_img: float, theta: float, env_size: int = 512) -> np.ndarray:
    """Binary mask of T with area-centroid at (cx_img, cy_img), rotated by theta (y-DOWN)."""
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
        iy = (wy * s).round().astype(np.int32)
        cv2.fillPoly(canvas, [np.stack([ix, iy], axis=1)], 1)
    return canvas.astype(bool)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union > 0 else 0.0


def _infer_block_state_from_image(img: np.ndarray, env_size: int = 512) -> tuple[float, float, float]:
    """Return pymunk body (pos_x, pos_y, angle) by template-matching the T polygon to the image mask.

    Uses coarse 2° sweep followed by fine 0.25° sweep around the best match.
    Coordinate system: y-DOWN (img_x = wx*s, img_y = wy*s).
    """
    img_size = img.shape[0]
    s = img_size / env_size
    obs_mask = _block_mask(img)
    if obs_mask.sum() < 10:
        return env_size / 2, env_size / 2, 0.0
    ys, xs = np.where(obs_mask)
    cx_img, cy_img = float(xs.mean()), float(ys.mean())
    best_theta, best_iou_val = 0.0, -1.0
    for deg in range(0, 360, 2):
        theta = np.deg2rad(deg)
        iou = _iou(obs_mask, _render_t_mask(img_size, cx_img, cy_img, theta, env_size))
        if iou > best_iou_val:
            best_iou_val, best_theta = iou, theta
    for ddeg in np.arange(-2.0, 2.25, 0.25):
        theta = best_theta + np.deg2rad(ddeg)
        iou = _iou(obs_mask, _render_t_mask(img_size, cx_img, cy_img, theta, env_size))
        if iou > best_iou_val:
            best_iou_val, best_theta = iou, theta
    c, sn = np.cos(best_theta), np.sin(best_theta)
    bx = cx_img / s + _T_LOCAL_CY * sn
    by = cy_img / s - _T_LOCAL_CY * c
    return bx, by, best_theta


class BaseGoalProvider(ABC):
    """Return a goal observation dict for each env in a VectorEnv."""

    @abstractmethod
    def get_goal_obs(self, vec_env: gym.vector.VectorEnv) -> dict[str, np.ndarray]:
        """Produce goal observations matching the env's native observation format.

        Args:
            vec_env: A Gymnasium VectorEnv whose underlying single envs support
                the provider's goal extraction logic.

        Returns:
            Dict of numpy arrays with the same keys and shapes as ``env.reset()``
            but batched: each array has shape ``(num_envs, *feature_shape)``.
        """
        ...

    def initialize_envs(
        self,
        seeds: list[int] | None,
        vec_env: gym.vector.VectorEnv,
    ) -> dict[str, np.ndarray] | None:
        """Called after env.reset() to do any provider-specific initialization.

        Override in subclasses that need to modify env state or assign reference
        episodes.

        Returns:
            Updated observation dict (same format as env.reset()) if the env state
            was modified, or None to keep the existing observation.
        """
        return None

    def get_goal_obs_at_step(
        self,
        step: int,
        vec_env: gym.vector.VectorEnv,
    ) -> dict[str, np.ndarray] | None:
        """Return a dynamic goal observation for the given rollout step.

        Override in subclasses that update the goal as the episode progresses
        (e.g. carrot-on-a-stick).  The rollout calls this at every planning
        chunk boundary (i.e. every ``n_action_steps`` steps).

        Returns:
            Dict of numpy arrays in env-native format, or None to keep the
            current goal unchanged.
        """
        return None


# ---------------------------------------------------------------------------
# Carrot-on-a-stick providers
# ---------------------------------------------------------------------------

class CarrotGoalProviderMixin:
    """Mixin that implements moving-carrot goal logic given stored reference frames.

    Subclasses or the eval rollout must call ``set_reference_episodes`` before
    the first ``get_goal_obs`` / ``get_goal_obs_at_step`` call.
    """

    _horizon: int
    _ref_frames: list[list[np.ndarray]] | None = None   # [env_i][t] = (H,W,C) uint8
    _ref_states: list[np.ndarray] | None = None          # [env_i] = (T, state_dim) float32

    def set_reference_episodes(
        self,
        frames: list[list[np.ndarray]],
        states: list[np.ndarray] | None = None,
    ) -> None:
        """Store reference episode frames (and optionally states) for all envs.

        Args:
            frames: List of length ``num_envs``.  Each element is a list of
                ``(H, W, C)`` uint8 numpy arrays — one per timestep.
            states: Optional list of length ``num_envs``.  Each element is a
                ``(T, state_dim)`` float32 array of agent positions.
        """
        self._ref_frames = frames
        self._ref_states = states
        logger.info(
            "CarrotGoalProvider: stored reference episodes for %d envs, "
            "episode lengths %s",
            len(frames),
            [len(f) for f in frames],
        )

    def _goal_index(self, step: int, ep_len: int) -> int:
        return min(step + self._horizon + 1, ep_len - 1)


# ---------------------------------------------------------------------------
# Dataset-based carrot provider
# ---------------------------------------------------------------------------

class DatasetCarrotGoalProvider(CarrotGoalProviderMixin, BaseGoalProvider, ABC):
    """Loads reference episodes from a LeRobot dataset parquet+video store.

    No BC reference rollout is needed — episodes come directly from the dataset.
    Uses pyarrow to read per-frame metadata and ``decode_video_frames`` for
    images, bypassing the LeRobotDataset HuggingFace wrapper.

    Subclasses implement ``_build_goal_dict`` to assemble the env-native
    observation dict (e.g. adding ``agent_pos`` for PushT).
    """

    needs_bc_reference: bool = False  # dataset already has the reference trajectories

    def __init__(
        self,
        dataset_root: str | Path,
        horizon: int,
        fps: float = 10.0,
        video_key: str = "observation.image",
    ) -> None:
        self._root = Path(dataset_root)
        self._horizon = horizon
        self._fps = fps
        self._video_key = video_key

        self._ref_ep_indices: list[int] | None = None
        self._frames_cache: dict[int, list[np.ndarray]] = {}
        self._states_cache: dict[int, np.ndarray] = {}

        self._episodes_meta = self._load_episodes_meta()
        self._data_table = self._load_data_table()
        logger.info(
            "DatasetCarrotGoalProvider: loaded %d episodes from %s",
            len(self._episodes_meta),
            self._root,
        )

    def _load_episodes_meta(self) -> pa.Table:
        files = sorted(glob.glob(str(self._root / "meta/episodes/**/*.parquet"), recursive=True))
        if not files:
            raise FileNotFoundError(f"No episode parquet files found under {self._root}/meta/episodes/")
        tables = [pq.read_table(f) for f in files]
        return pa.concat_tables(tables) if len(tables) > 1 else tables[0]

    def _load_data_table(self) -> pa.Table:
        files = sorted(glob.glob(str(self._root / "data/**/*.parquet"), recursive=True))
        if not files:
            raise FileNotFoundError(f"No data parquet files found under {self._root}/data/")
        tables = [pq.read_table(f) for f in files]
        return pa.concat_tables(tables) if len(tables) > 1 else tables[0]

    def _n_episodes(self) -> int:
        return len(self._episodes_meta)

    def _episode_length(self, ep_idx: int) -> int:
        return self._episodes_meta["length"][ep_idx].as_py()

    def _video_from_timestamp(self, ep_idx: int) -> float:
        return self._episodes_meta[f"videos/{self._video_key}/from_timestamp"][ep_idx].as_py()

    def _video_file_path(self, ep_idx: int) -> Path:
        chunk_idx = self._episodes_meta[f"videos/{self._video_key}/chunk_index"][ep_idx].as_py()
        file_idx = self._episodes_meta[f"videos/{self._video_key}/file_index"][ep_idx].as_py()
        info = json.loads((self._root / "meta/info.json").read_text())
        template = info["video_path"]
        rel_path = template.format(
            video_key=self._video_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        return self._root / rel_path

    def _load_episode_frames(self, ep_idx: int) -> list[np.ndarray]:
        from lerobot.datasets.video_utils import decode_video_frames

        length = self._episode_length(ep_idx)
        from_ts = self._video_from_timestamp(ep_idx)
        video_path = self._video_file_path(ep_idx)
        timestamps = [from_ts + i / self._fps for i in range(length)]
        frames_tensor = decode_video_frames(
            video_path, timestamps, tolerance_s=1.0 / self._fps, backend="pyav"
        )
        return [(f.permute(1, 2, 0).mul(255).byte().numpy()) for f in frames_tensor]

    def _load_episode_states(self, ep_idx: int) -> np.ndarray:
        from_idx = self._episodes_meta["dataset_from_index"][ep_idx].as_py()
        to_idx = self._episodes_meta["dataset_to_index"][ep_idx].as_py()
        states = self._data_table["observation.state"][from_idx:to_idx].to_pylist()
        return np.array(states, dtype=np.float32)

    def _ensure_episode_loaded(self, ep_idx: int) -> None:
        if ep_idx not in self._frames_cache:
            logger.debug("DatasetCarrotGoalProvider: loading episode %d", ep_idx)
            self._frames_cache[ep_idx] = self._load_episode_frames(ep_idx)
            self._states_cache[ep_idx] = self._load_episode_states(ep_idx)

    def initialize_envs(
        self,
        seeds: list[int] | None,
        vec_env: gym.vector.VectorEnv,
    ) -> dict[str, np.ndarray] | None:
        n_envs = vec_env.num_envs
        n_eps = self._n_episodes()

        if seeds is not None:
            self._ref_ep_indices = [int(s) % n_eps for s in seeds]
        else:
            self._ref_ep_indices = [i % n_eps for i in range(n_envs)]

        for ep_idx in set(self._ref_ep_indices):
            self._ensure_episode_loaded(ep_idx)

        # Build unified frame/state lists and call the mixin setter
        frames = [self._frames_cache[ep_idx] for ep_idx in self._ref_ep_indices]
        states = [self._states_cache[ep_idx] for ep_idx in self._ref_ep_indices]
        self.set_reference_episodes(frames, states)
        return None  # don't override env observation

    def get_goal_obs(self, vec_env: gym.vector.VectorEnv) -> dict[str, np.ndarray]:
        result = self.get_goal_obs_at_step(0, vec_env)
        if result is None:
            raise RuntimeError("initialize_envs() must be called before get_goal_obs()")
        return result

    def get_goal_obs_at_step(
        self,
        step: int,
        vec_env: gym.vector.VectorEnv,
    ) -> dict[str, np.ndarray] | None:
        if self._ref_frames is None:
            return None
        return self._build_goal_dict(step)

    @abstractmethod
    def _build_goal_dict(self, step: int) -> dict[str, np.ndarray]: ...


class PushTDatasetCarrotGoalProvider(DatasetCarrotGoalProvider):
    """Carrot-on-a-stick goal provider for PushT backed by the lerobot/pusht dataset.

    Episode assignment: ``ep_idx = seed % n_episodes`` (deterministic per eval seed).
    Initialization: template-matches video frame 0 to recover block (pos, angle), then
    sets the pymunk physics state to exactly match the dataset episode start.
    Goal at step t: the dataset video frame at ``min(t + H + 1, ep_len - 1)``.
    """

    def __init__(self, dataset_root: str | Path, horizon: int) -> None:
        super().__init__(dataset_root=dataset_root, horizon=horizon, video_key="observation.image")
        self._vis_w: int = 96
        self._vis_h: int = 96

    def initialize_envs(
        self,
        seeds: list[int] | None,
        vec_env: gym.vector.VectorEnv,
    ) -> dict[str, np.ndarray] | None:
        """Assign dataset episodes, load frames, init env physics from dataset frame 0."""
        n_envs = vec_env.num_envs
        n_eps = self._n_episodes()

        if seeds is not None:
            self._ref_ep_indices = [int(s) % n_eps for s in seeds]
        else:
            self._ref_ep_indices = [i % n_eps for i in range(n_envs)]

        for ep_idx in set(self._ref_ep_indices):
            self._ensure_episode_loaded(ep_idx)

        frames = [self._frames_cache[ep_idx] for ep_idx in self._ref_ep_indices]
        states = [self._states_cache[ep_idx] for ep_idx in self._ref_ep_indices]
        self.set_reference_episodes(frames, states)

        logger.info(
            "PushTDatasetCarrotGoalProvider: assigned episodes %s (seeds %s)",
            self._ref_ep_indices,
            seeds,
        )

        if not hasattr(vec_env, "envs"):
            return None

        # Infer visualization resolution from the first env.
        base0 = vec_env.envs[0].unwrapped
        self._vis_w = getattr(base0, "visualization_width", getattr(base0, "observation_width", 96))
        self._vis_h = getattr(base0, "visualization_height", getattr(base0, "observation_height", 96))

        images: list[np.ndarray] = []
        agent_positions: list[np.ndarray] = []

        for i, sub_env in enumerate(vec_env.envs):
            base = sub_env.unwrapped
            ep_idx = self._ref_ep_indices[i]
            frame0 = self._frames_cache[ep_idx][0]          # (96, 96, 3) uint8
            agent_state0 = self._states_cache[ep_idx][0]    # [agent_x, agent_y]

            # Template-match the first frame to recover block body (pos, angle).
            bx, by, ba = _infer_block_state_from_image(frame0)
            logger.debug(
                "env %d: ep=%d  block=(%.1f, %.1f, %.3f rad)  agent=(%.1f, %.1f)",
                i, ep_idx, bx, by, ba, agent_state0[0], agent_state0[1],
            )

            # Set angle FIRST — pymunk's position setter is angle-dependent due to CoG offset.
            base.block.angle = float(ba)
            base.block.position = (float(bx), float(by))
            base.block.velocity = (0.0, 0.0)
            base.block.angular_velocity = 0.0
            base.agent.position = (float(agent_state0[0]), float(agent_state0[1]))
            base.agent.velocity = (0.0, 0.0)
            base.space.step(1e-6)

            obs = base.get_obs()
            images.append(obs["pixels"].copy())
            agent_positions.append(np.array(base.agent.position, dtype=np.float32))

        return {
            "pixels": np.stack(images),
            "agent_pos": np.stack(agent_positions, axis=0),
        }

    def _build_goal_dict(self, step: int) -> dict[str, np.ndarray]:
        """Return the dataset video frame at the carrot index for each env."""
        from PIL import Image as _PIL

        images: list[np.ndarray] = []
        images_vis: list[np.ndarray] = []
        states: list[np.ndarray] = []

        for i, frames in enumerate(self._ref_frames):
            ep_len = len(frames)
            idx = self._goal_index(step, ep_len)
            img = frames[idx]  # (96, 96, 3) uint8 — dataset observation resolution
            images.append(img)
            # Resize to env visualization resolution for the side-by-side video.
            if (self._vis_w, self._vis_h) != (img.shape[1], img.shape[0]):
                img_vis = np.array(_PIL.fromarray(img).resize((self._vis_w, self._vis_h), _PIL.BILINEAR))
            else:
                img_vis = img
            images_vis.append(img_vis)
            if self._ref_states is not None:
                states.append(self._ref_states[i][idx])

        result: dict[str, np.ndarray] = {
            "pixels": np.stack(images, axis=0),
            "pixels_vis": np.stack(images_vis, axis=0),
        }
        if states:
            result["agent_pos"] = np.stack(states, axis=0)
        return result


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def make_goal_provider(env_type: str) -> BaseGoalProvider:
    """Return the static (final-goal) BaseGoalProvider for the given env type."""
    if env_type == "pusht":
        return PushTGoalProvider()
    raise ValueError(
        f"No goal provider implemented for env_type={env_type!r}. "
        "Supported: ['pusht']"
    )


def make_carrot_goal_provider(
    env_type: str,
    horizon: int,
    max_env_steps: int = 300,
    dataset_root: str | Path | None = None,
) -> BaseGoalProvider:
    """Return a carrot-on-a-stick goal provider for the given env type.

    When ``dataset_root`` is provided the provider loads reference episodes
    directly from the dataset (no BC reference rollout needed).  Otherwise falls
    back to on-the-fly BC reference generation.

    Args:
        env_type: String env type (e.g. ``"pusht"``).
        horizon: Planning horizon H (typically ``policy.config.chunk_size``).
        max_env_steps: Unused; kept for API compatibility.
        dataset_root: Local path to the LeRobot dataset root directory.
    """
    if env_type == "pusht":
        if dataset_root is None:
            raise ValueError(
                "dataset_root must be provided for the PushT carrot goal provider. "
                "Pass --carrot_goal.dataset_root=<path/to/lerobot/pusht>."
            )
        logger.info("Using dataset-based carrot provider from %s", dataset_root)
        return PushTDatasetCarrotGoalProvider(dataset_root=dataset_root, horizon=horizon)
    raise ValueError(
        f"No carrot goal provider implemented for env_type={env_type!r}. "
        "Supported: ['pusht']"
    )


# ---------------------------------------------------------------------------
# Static final-goal providers (existing behaviour)
# ---------------------------------------------------------------------------

def _render_pusht_goal_scene(base) -> np.ndarray:
    """Render a goal-achieved image for a single PushT env without touching physics."""
    import pygame
    import pymunk.pygame_util

    _BLOCK_COLOR = (119, 136, 153, 255)
    _AGENT_COLOR = (65, 105, 225, 255)
    _AGENT_RADIUS = 15

    screen = pygame.Surface((512, 512))
    screen.fill((255, 255, 255))

    goal_body = base.get_goal_pose_body(base.goal_pose)

    for shape in base.block.shapes:
        pts = [goal_body.local_to_world(v) for v in shape.get_vertices()]
        pts = [pymunk.pygame_util.to_pygame(p, screen) for p in pts]
        pygame.draw.polygon(screen, pygame.Color("LightGreen"), pts + [pts[0]])

    for shape in base.block.shapes:
        pts = [goal_body.local_to_world(v) for v in shape.get_vertices()]
        pts = [pymunk.pygame_util.to_pygame(p, screen) for p in pts]
        pygame.draw.polygon(screen, pygame.Color(*_BLOCK_COLOR), pts + [pts[0]])

    # Agent is not part of the goal — omit it so the WM cost only captures block placement.
    return base._get_img(screen, base.observation_width, base.observation_height)


class PushTGoalProvider(BaseGoalProvider):
    """Renders the fully-solved state (block at goal pose) as the planning goal."""

    def get_goal_obs(self, vec_env: gym.vector.VectorEnv) -> dict[str, np.ndarray]:
        images: list[np.ndarray] = []
        states: list[np.ndarray] = []

        if not hasattr(vec_env, "envs"):
            raise RuntimeError(
                "PushTGoalProvider requires a SyncVectorEnv with an .envs attribute."
            )

        for env in vec_env.envs:
            base = env.unwrapped
            img = _render_pusht_goal_scene(base)
            images.append(img)
            # Agent position is irrelevant to the goal; use workspace centre so the
            # WM cost signal focuses purely on block placement.
            states.append(np.array([256.0, 256.0], dtype=np.float64))

        return {
            "pixels": np.stack(images, axis=0),
            "agent_pos": np.stack(states, axis=0),
        }
