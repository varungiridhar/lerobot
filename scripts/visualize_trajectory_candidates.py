"""Visualize MPPI vs BC-diffusion trajectory candidates scored by Q function.

Supports Libero and RoboTwin environments (--env libero|robotwin).

For each planning step produces 3 conditions × 2 figures (plot_a + plot_b):
  plot_a — per-action-dim line plots colored by Q value (viridis)
  plot_b — PCA 2D scatter colored by Q + spaghetti on selected dims

Conditions: MPPI (no smooth), MPPI (smooth σ=2), BC diffusion (--diffusion_steps)

Usage:
    # Libero
    python scripts/visualize_trajectory_candidates.py \\
        --env libero --task_id 0 --n_samples 64 --n_steps 5 --out_dir ./traj_vis_libero

    # RoboTwin
    python scripts/visualize_trajectory_candidates.py \\
        --env robotwin --task beat_block_hammer --n_samples 64 --n_steps 5 --out_dir ./traj_vis_robotwin
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without installing.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

# FastWAM sets DIFFSYNTH_MODEL_BASE_PATH inside its module; import it first.
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy  # noqa: F401 (sets env var on import)
from lerobot.policies.act_simple.planning import (
    PlannerContext,
    _sample_noise,
    _score_candidates_fast,
)
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.constants import (
    ACTION,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


LIBERO_FASTWAM_CKPT = "/storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint"
LIBERO_Q_CKPT = (
    "/storage/scratch1/6/vgiridhar6/lerobot/outputs/train"
    "/2026-05-19/23-54-42_qf_libero_ddp2_bsz48_bc_h200/checkpoints/last/pretrained_model"
)
ROBOTWIN_FASTWAM_CKPT = "/storage/project/r-agarg35-0/shared/fastwam/hf_checkpoint_robotwin"
ROBOTWIN_Q_CKPT = (
    "/storage/project/r-agarg35-0/vgiridhar6/robotwin/outputs/train"
    "/qf_robotwin_ddp_20260526_220319/checkpoints/030000/pretrained_model"
)
ROBOTWIN_ROOT = "/storage/project/r-agarg35-0/vgiridhar6/robotwin/RoboTwin"

# Libero: 7-dim (6 arm + gripper), plot arm dims only
LIBERO_DIM_NAMES = ["eef_x", "eef_y", "eef_z", "roll", "pitch", "yaw", "gripper"]
LIBERO_PLOT_DIMS = list(range(6))
LIBERO_MPPI_NOISE_STD_PER_DIM = [0.179, 0.202, 0.237, 0.066, 0.085, 0.106, 0.0]

# RoboTwin: 14-dim bimanual (left arm 0-5, left gripper 6, right arm 7-12, right gripper 13)
ROBOTWIN_DIM_NAMES = [
    "L_eef_x", "L_eef_y", "L_eef_z", "L_roll", "L_pitch", "L_yaw", "L_gripper",
    "R_eef_x", "R_eef_y", "R_eef_z", "R_roll", "R_pitch", "R_yaw", "R_gripper",
]
ROBOTWIN_PLOT_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]  # skip grippers 6 and 13
ROBOTWIN_MPPI_NOISE_STD_PER_DIM = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.0]


# ── Env helpers ──────────────────────────────────────────────────────────────

def make_libero_env(task_id: int = 0):
    """Instantiate a single LiberoEnv (pixels_agent_pos) for the given task."""
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_10"]()
    env = LiberoEnv(
        task_suite=task_suite,
        task_id=task_id,
        task_suite_name="libero_10",
        obs_type="pixels_agent_pos",
        observation_width=224,
        observation_height=224,
        episode_index=0,
    )
    return env


def _add_batch_dim_nested(d: dict) -> dict:
    """Recursively unsqueeze dim-0 on all Tensors in a nested dict (to add B=1 batch dim)."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _add_batch_dim_nested(v)
        elif isinstance(v, torch.Tensor):
            result[k] = v.unsqueeze(0) if v.dim() < 2 else v
        else:
            result[k] = v
    return result


def obs_to_batch(obs: dict, device: torch.device) -> dict:
    """Convert raw LiberoEnv observation dict to a float32 tensor batch (B=1).

    Replicates the logic of preprocess_observation() + LiberoProcessorStep.
    """
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep

    # Step 1: raw numpy dict → LeRobot tensor format (images as (B, C, H, W) float [0,1]).
    tensor_obs = preprocess_observation(obs)

    # preprocess_observation adds B dim to images but not to robot_state tensors.
    # Add B=1 dim to robot_state nested tensors so LiberoProcessorStep gets (B, D).
    robot_state_key = "observation.robot_state"
    if robot_state_key in tensor_obs:
        tensor_obs[robot_state_key] = _add_batch_dim_nested(tensor_obs[robot_state_key])

    # Step 2: LiberoProcessorStep — flip images 180°, flatten robot_state → observation.state.
    libero_proc = LiberoProcessorStep()
    tensor_obs = libero_proc.observation(tensor_obs)

    # Move everything to device.
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tensor_obs.items()}


def make_robotwin_env(task: str, robotwin_root: str):
    """Instantiate a single RoboTwinEnv for the given task."""
    import os
    from lerobot.envs.robotwin import RoboTwinEnv
    curobo_src = os.path.join(robotwin_root, "envs", "curobo", "src")
    for p in (curobo_src, robotwin_root):
        if p not in sys.path:
            sys.path.insert(0, p)
    return RoboTwinEnv(task_name=task, robotwin_root=robotwin_root)


def robotwin_obs_to_batch(obs: dict, task_str: str, device: torch.device) -> dict:
    """Convert raw RoboTwinEnv observation dict to a float32 tensor batch (B=1)."""
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import RoboTwinProcessorStep

    tensor_obs = preprocess_observation(obs)
    robotwin_proc = RoboTwinProcessorStep()
    tensor_obs = robotwin_proc.observation(tensor_obs)
    tensor_obs["task"] = [task_str]
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tensor_obs.items()}


def render_env_frame(env) -> np.ndarray:
    """Return (H, W, 3) uint8 image for the current env state."""
    try:
        return env.render()
    except Exception:
        return np.zeros((224, 224, 3), dtype=np.uint8)


# ── Model loading ────────────────────────────────────────────────────────────

def load_models(fastwam_ckpt: str, q_ckpt: str, device: torch.device):
    """Load FastWAM policy, Q-function, and all processor pipelines."""
    from lerobot.processor.converters import (
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )

    print("Loading FastWAM policy…")
    fastwam_policy = FastWAMPolicy.from_pretrained(fastwam_ckpt).to(device).eval()

    print("Loading FastWAM postprocessor (bc_post) and preprocessor…")
    bc_post = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=fastwam_ckpt,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=fastwam_ckpt,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )

    print("Loading Q-function and Q preprocessor…")
    from lerobot.policies.q_function.modeling_q_function import QFunctionPolicy
    q_policy = QFunctionPolicy.from_pretrained(q_ckpt).to(device).eval()
    q_pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=q_ckpt,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )

    ctx = PlannerContext(
        q_policy=q_policy,
        q_pre=q_pre,
        bc_post=bc_post,
        q_camera_keys=tuple(q_policy.config.camera_keys),
        horizon=int(q_policy.config.h),
    )

    return fastwam_policy, preprocessor, ctx


# ── Candidate scoring ────────────────────────────────────────────────────────

@torch.no_grad()
def score_candidates(
    candidates_fastwam_norm: torch.Tensor,  # (N, h, A)
    batch: dict,
    ctx: PlannerContext,
    device: torch.device,
) -> torch.Tensor:
    """Score N candidates in FastWAM-norm space. Returns (N,) Q values."""
    N = candidates_fastwam_norm.shape[0]
    img_feats = {k: batch[k] for k in ctx.q_camera_keys if k in batch}
    if "task" in batch:
        img_feats["task"] = batch["task"]

    single_batch = {ACTION: candidates_fastwam_norm[:1], **img_feats}
    single_pre = ctx.q_pre(single_batch)
    obs_context = ctx.q_policy.encode_obs_context(single_pre)  # (S, 1, D)

    q_vals = _score_candidates_fast(candidates_fastwam_norm, obs_context, ctx, img_feats)
    return q_vals.to(device=device, dtype=torch.float32)


@torch.no_grad()
def candidates_to_raw(
    candidates_fastwam_norm: torch.Tensor,  # (N, h, A)
    ctx: PlannerContext,
) -> np.ndarray:
    """Unnorm from FastWAM-norm space to raw env space. Returns (N, h, A) float32 numpy."""
    N, h, A = candidates_fastwam_norm.shape
    flat = candidates_fastwam_norm.reshape(N * h, A)
    raw_flat = ctx.bc_post(flat)
    return raw_flat.reshape(N, h, A).float().cpu().numpy()


# ── Plotting ─────────────────────────────────────────────────────────────────

def _colormap_lines(ax, xs, ys_list, q_vals, cmap="viridis", alpha=0.35, lw=0.6):
    """Draw each candidate as a colored line; return the ScalarMappable for colorbar."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    vmin, vmax = float(q_vals.min()), float(q_vals.max())
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = cm.get_cmap(cmap)
    for i, ys in enumerate(ys_list):
        color = cmap_obj(norm(float(q_vals[i])))
        ax.plot(xs, ys, color=color, alpha=alpha, linewidth=lw)
    sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    return sm


def plot_a_6dim(
    candidates_raw: np.ndarray,  # (N, h, A)
    bc_mean_raw: np.ndarray,     # (h, A)
    q_vals: np.ndarray,          # (N,)
    title: str,
    out_path: Path,
    plot_dims: list[int] = LIBERO_PLOT_DIMS,
    dim_names: list[str] = LIBERO_DIM_NAMES,
    overlay_lines: list[dict] | None = None,
) -> None:
    """Plot A: one panel per plotted action dim (no gripper), lines colored by Q value.

    overlay_lines draws additional bold lines on top (e.g. per-iteration MPPI means,
    top-K weighted mean). Each entry: {data, label, lw, color, linestyle}.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N, h, A = candidates_raw.shape
    n_dims = len(plot_dims)
    n_cols = 3
    n_rows = (n_dims + n_cols - 1) // n_cols
    xs = np.arange(h)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = np.array(axes).flatten()

    sm = None
    for panel_i, d in enumerate(plot_dims):
        ax = axes[panel_i]
        ys_list = [candidates_raw[i, :, d] for i in range(N)]
        sm = _colormap_lines(ax, xs, ys_list, q_vals)
        ax.plot(xs, bc_mean_raw[:, d], color="black", linewidth=1.5, linestyle="--",
                label="BC mean", zorder=5)
        if overlay_lines:
            for ol in overlay_lines:
                ax.plot(xs, ol["data"][:, d],
                        color=ol.get("color", "red"),
                        linewidth=ol.get("lw", 2.5),
                        linestyle=ol.get("linestyle", "-"),
                        label=ol.get("label", ""),
                        zorder=6)
        ax.set_title(dim_names[d] if d < len(dim_names) else f"dim{d}", fontsize=9)
        ax.set_xlabel("Action step", fontsize=7)
        ax.tick_params(labelsize=7)
        if panel_i == 0:
            ax.legend(fontsize=7)

    for ax in axes[n_dims:]:
        ax.set_visible(False)

    if sm is not None:
        fig.colorbar(sm, ax=axes[:n_dims].tolist(), label="Q value", fraction=0.02, pad=0.04)
    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_b_pca(
    candidates_raw: np.ndarray,  # (N, h, A)
    bc_mean_raw: np.ndarray,     # (h, A)
    q_vals: np.ndarray,          # (N,)
    title: str,
    out_path: Path,
    spaghetti_dims: tuple[int, int] = (0, 4),
    dim_names: list[str] = LIBERO_DIM_NAMES,
    overlay_lines: list[dict] | None = None,
) -> None:
    """Plot B: PCA scatter + 2 spaghetti panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from sklearn.decomposition import PCA

    N, h, A = candidates_raw.shape
    xs = np.arange(h)

    vmin, vmax = float(q_vals.min()), float(q_vals.max())
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("viridis")
    colors = [cmap(norm(float(q_vals[i]))) for i in range(N)]

    # PCA: flatten (N, h*A) → (N, 2)
    flat = candidates_raw.reshape(N, h * A)
    pca = PCA(n_components=2)
    proj = pca.fit_transform(flat)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Left: PCA scatter
    ax_pca = axes[0]
    sc = ax_pca.scatter(proj[:, 0], proj[:, 1], c=q_vals, cmap="viridis",
                        vmin=vmin, vmax=vmax, s=20, alpha=0.8)
    bc_proj = pca.transform(bc_mean_raw.flatten().reshape(1, -1))
    ax_pca.scatter(bc_proj[:, 0], bc_proj[:, 1], c="black", s=80, marker="*",
                   zorder=10, label="BC mean")
    ax_pca.set_xlabel("PC1", fontsize=8)
    ax_pca.set_ylabel("PC2", fontsize=8)
    ax_pca.set_title("PCA of trajectory candidates", fontsize=9)
    ax_pca.legend(fontsize=7)
    fig.colorbar(sc, ax=ax_pca, label="Q value", fraction=0.04)

    # Middle + Right: spaghetti on selected dims
    for panel_i, dim in enumerate(spaghetti_dims[:2]):
        ax = axes[1 + panel_i]
        for i in range(N):
            ax.plot(xs, candidates_raw[i, :, dim], color=colors[i], alpha=0.35, linewidth=0.6)
        ax.plot(xs, bc_mean_raw[:, dim], color="black", linewidth=2.0, linestyle="--",
                label="BC mean", zorder=5)
        if overlay_lines:
            for ol in overlay_lines:
                ax.plot(xs, ol["data"][:, dim],
                        color=ol.get("color", "red"),
                        linewidth=ol.get("lw", 2.5),
                        linestyle=ol.get("linestyle", "-"),
                        label=ol.get("label", ""),
                        zorder=6)
        dim_name = dim_names[dim] if dim < len(dim_names) else f"dim{dim}"
        ax.set_title(f"Spaghetti — {dim_name}", fontsize=9)
        ax.set_xlabel("Action step", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def save_step_data(
    step_idx: int,
    mppi_candidates_raw: np.ndarray,       # (N, h, A) final MPPI iter candidates
    mppi_q: np.ndarray,                     # (N,) final iter Q values
    mppi_iter_candidates_raw: list[np.ndarray],  # list of (N, h, A) per MPPI iter
    mppi_iter_q: list[np.ndarray],          # list of (N,) per MPPI iter
    mppi_iter_means_raw: list[np.ndarray],  # list of (h, A) per MPPI iter
    bc_diff_candidates_raw: np.ndarray,     # (N, h, A)
    bc_diff_q: np.ndarray,                  # (N,)
    bc_mean_raw: np.ndarray,                # (h, A)
    step_dir: Path,
    pca_rows: list,                         # accumulator for pca_data.csv
) -> None:
    """Save dense array data as NPZ and PCA scatter data into pca_rows list."""
    from sklearn.decomposition import PCA

    N, h, A = mppi_candidates_raw.shape

    # ── NPZ: all raw trajectories + Q values ─────────────────────────────────
    npz_data = {
        "bc_mean_raw": bc_mean_raw,
        "mppi_candidates_raw": mppi_candidates_raw,
        "mppi_q_values": mppi_q,
        "bc_diff_candidates_raw": bc_diff_candidates_raw,
        "bc_diff_q_values": bc_diff_q,
    }
    for i, (cands, q_vals, mean) in enumerate(
        zip(mppi_iter_candidates_raw, mppi_iter_q, mppi_iter_means_raw)
    ):
        npz_data[f"mppi_iter{i}_candidates_raw"] = cands
        npz_data[f"mppi_iter{i}_q_values"] = q_vals
        npz_data[f"mppi_iter{i}_mean_raw"] = mean

    npz_path = step_dir / "data.npz"
    np.savez_compressed(str(npz_path), **npz_data)
    print(f"  Saved {npz_path}")

    # ── PCA projections → accumulate rows for pca_data.csv ───────────────────
    # Fit PCA on union of all candidates so projections are comparable within step.
    all_cands = np.concatenate([mppi_candidates_raw, bc_diff_candidates_raw], axis=0)
    flat_all = all_cands.reshape(len(all_cands), h * A)
    pca = PCA(n_components=2)
    pca.fit(flat_all)

    for condition, cands, q_vals in [
        ("mppi", mppi_candidates_raw, mppi_q),
        ("bc_diff", bc_diff_candidates_raw, bc_diff_q),
    ]:
        proj = pca.transform(cands.reshape(len(cands), h * A))
        for i in range(len(cands)):
            pca_rows.append({
                "step": step_idx,
                "condition": condition,
                "sample_idx": i,
                "pc1": float(proj[i, 0]),
                "pc2": float(proj[i, 1]),
                "q_value": float(q_vals[i]),
            })

    # Also add BC mean projection as reference
    bc_proj = pca.transform(bc_mean_raw.flatten().reshape(1, h * A))
    pca_rows.append({
        "step": step_idx,
        "condition": "bc_mean",
        "sample_idx": 0,
        "pc1": float(bc_proj[0, 0]),
        "pc2": float(bc_proj[0, 1]),
        "q_value": float("nan"),
    })


def save_obs_frame(frame: np.ndarray, out_path: Path) -> None:
    """Save env frame as a PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(frame)
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize MPPI vs BC-diffusion trajectory candidates")
    parser.add_argument("--env", default="libero", choices=["libero", "robotwin"],
                        help="Environment type")
    parser.add_argument("--fastwam_ckpt", default=None, help="FastWAM checkpoint (default: env-specific)")
    parser.add_argument("--q_ckpt", default=None, help="Q checkpoint (default: env-specific)")
    parser.add_argument("--task_id", type=int, default=0, help="Libero-10 task index (0–9)")
    parser.add_argument("--task", default="beat_block_hammer", help="RoboTwin task name")
    parser.add_argument("--robotwin_root", default=ROBOTWIN_ROOT, help="Path to RoboTwin repo")
    parser.add_argument("--n_samples", type=int, default=64, help="Candidates per condition")
    parser.add_argument("--n_mppi_iters", type=int, default=3, help="MPPI iterations")
    parser.add_argument("--n_elites", type=int, default=16, help="Top-K for weighted mean")
    parser.add_argument("--temperature", type=float, default=1.0, help="MPPI softmax temperature")
    parser.add_argument("--noise_decay", type=float, default=0.5,
                        help="Multiply noise std by this factor each MPPI iteration (1.0=no decay)")
    parser.add_argument("--noise_smooth_sigma_t", type=float, default=0.0,
                        help="Gaussian smoothing sigma along time axis for MPPI noise (0=IID, 2.0=smooth)")
    parser.add_argument("--diffusion_steps", type=int, default=3,
                        help="Number of denoising steps for BC diffusion candidates")
    parser.add_argument("--n_steps", type=int, default=10, help="Number of planning steps to visualize")
    parser.add_argument("--out_dir", default="./traj_vis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Resolve env-specific defaults ────────────────────────────────────────
    is_robotwin = args.env == "robotwin"
    fastwam_ckpt = args.fastwam_ckpt or (ROBOTWIN_FASTWAM_CKPT if is_robotwin else LIBERO_FASTWAM_CKPT)
    q_ckpt = args.q_ckpt or (ROBOTWIN_Q_CKPT if is_robotwin else LIBERO_Q_CKPT)
    dim_names = ROBOTWIN_DIM_NAMES if is_robotwin else LIBERO_DIM_NAMES
    plot_dims = ROBOTWIN_PLOT_DIMS if is_robotwin else LIBERO_PLOT_DIMS
    mppi_noise_per_dim = ROBOTWIN_MPPI_NOISE_STD_PER_DIM if is_robotwin else LIBERO_MPPI_NOISE_STD_PER_DIM

    # ── Load models ──────────────────────────────────────────────────────────
    fastwam_policy, preprocessor, ctx = load_models(fastwam_ckpt, q_ckpt, device)
    h = ctx.horizon  # 32

    # ── Load env ─────────────────────────────────────────────────────────────
    if is_robotwin:
        import os
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
        print(f"\nBuilding RoboTwin env, task={args.task}…")
        env = make_robotwin_env(args.task, args.robotwin_root)
        obs, _ = env.reset(seed=args.seed)
        task_str = env.task
        n_action_steps = 24  # FastWAM RoboTwin n_action_steps
    else:
        print(f"\nBuilding Libero-10 env, task_id={args.task_id}…")
        env = make_libero_env(task_id=args.task_id)
        obs, _ = env.reset(seed=args.seed)
        task_str = getattr(env, "task_description", "robot manipulation task")
        n_action_steps = 10  # FastWAM Libero n_action_steps
    print(f"  Task: {task_str[:80]}")

    all_q_data = []
    pca_rows = []  # accumulates across all steps for pca_data.csv

    # ── Per-step loop ─────────────────────────────────────────────────────────
    for step_idx in range(args.n_steps):
        print(f"\n{'='*60}")
        print(f"Planning step {step_idx}/{args.n_steps - 1}")

        step_dir = out_dir / f"step_{step_idx:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Build batch from current obs
        if is_robotwin:
            batch = robotwin_obs_to_batch(obs, task_str, device)
        else:
            batch = obs_to_batch(obs, device)
            batch["task"] = [task_str]

        # Preprocess for FastWAM (normalizes images, state)
        preprocessed = preprocessor(batch)

        # Render frame for reference
        frame = render_env_frame(env)
        save_obs_frame(frame, step_dir / "obs_frame.pdf")

        # ── BC mean (shared anchor) ───────────────────────────────────────
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            bc_mean_norm = fastwam_policy.predict_action_chunk(preprocessed).to(device)  # (1, h, A)

        A = bc_mean_norm.shape[-1]
        noise_std_per_dim = torch.tensor(
            mppi_noise_per_dim[:A], device=device, dtype=torch.float32
        )
        iter_colors = ["#f4a261", "#e76f51", "#9b2226"]  # orange→red→darkred

        # ── Helper: run MPPI for a given smoothing sigma ──────────────────
        def run_mppi(smooth_sigma_val):
            smooth_sigma = smooth_sigma_val if smooth_sigma_val > 0 else None
            mean = bc_mean_norm.float().clone()
            iter_means_norm, iter_cands_norm, iter_q_vals = [], [], []
            candidates, q_vals = None, None
            with torch.no_grad():
                for iter_i in range(args.n_mppi_iters):
                    decay = args.noise_decay ** iter_i
                    noise = _sample_noise(
                        (args.n_samples, h, A), noise_std_per_dim * decay,
                        clip_to=None, device=device, dtype=torch.float32,
                        generator=None, smooth_sigma_t=smooth_sigma,
                    )
                    candidates = mean.expand(args.n_samples, h, -1) + noise
                    q_vals = score_candidates(candidates, preprocessed, ctx, device)
                    iter_cands_norm.append(candidates.clone())
                    iter_q_vals.append(q_vals.clone())
                    K = min(args.n_elites, args.n_samples)
                    topk_idx = torch.topk(q_vals, K).indices
                    topk_q = q_vals[topk_idx]
                    topk_cands = candidates[topk_idx]
                    weights = torch.softmax((topk_q - topk_q.max()) / args.temperature, dim=0)
                    mean = (weights.view(K, 1, 1) * topk_cands).sum(dim=0, keepdim=True)
                    iter_means_norm.append(mean.clone())
                    q_np = q_vals.cpu().numpy()
                    tag = f"smooth{smooth_sigma_val}" if smooth_sigma_val > 0 else "no_smooth"
                    print(f"  MPPI[{tag}] iter {iter_i+1} (noise×{decay:.2f}) Q: "
                          f"min={q_np.min():.4f}  max={q_np.max():.4f}  "
                          f"mean={q_np.mean():.4f}  std={q_np.std():.4f}")
            return candidates, q_vals, iter_cands_norm, iter_q_vals, iter_means_norm

        # ── Run all 3 conditions ──────────────────────────────────────────
        mppi_cands, mppi_q, mppi_iter_cands, mppi_iter_qs, mppi_iter_means = run_mppi(0.0)
        mppi_s2_cands, mppi_s2_q, mppi_s2_iter_cands, mppi_s2_iter_qs, mppi_s2_iter_means = run_mppi(2.0)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            bc_diff_cands = fastwam_policy.predict_n_action_chunks(
                preprocessed, n_samples=args.n_samples, num_inference_steps=args.diffusion_steps,
            ).to(device=device, dtype=torch.float32)
        if bc_diff_cands.dim() == 2:
            bc_diff_cands = bc_diff_cands.unsqueeze(0)
        with torch.no_grad():
            bc_diff_q = score_candidates(bc_diff_cands, preprocessed, ctx, device)
        bc_diff_q_np = bc_diff_q.cpu().numpy()
        print(f"  BC-diffusion ({args.diffusion_steps} steps) Q: "
              f"min={bc_diff_q_np.min():.4f}  max={bc_diff_q_np.max():.4f}  "
              f"mean={bc_diff_q_np.mean():.4f}  std={bc_diff_q_np.std():.4f}")

        # ── Unnorm to raw env space ───────────────────────────────────────
        with torch.no_grad():
            bc_mean_raw = candidates_to_raw(bc_mean_norm.float(), ctx)[0]   # (h, A)

            def to_raw_condition(cands, iter_cands_n, iter_qs_t, iter_means_n):
                return (
                    candidates_to_raw(cands.float(), ctx),
                    cands.float().cpu().numpy() if False else None,  # placeholder
                    [candidates_to_raw(c.float(), ctx) for c in iter_cands_n],
                    [q.cpu().numpy() for q in iter_qs_t],
                    [candidates_to_raw(m.float(), ctx)[0] for m in iter_means_n],
                )

            mppi_raw, _, mppi_iter_cands_raw, mppi_iter_q_np, mppi_iter_means_raw = \
                to_raw_condition(mppi_cands, mppi_iter_cands, mppi_iter_qs, mppi_iter_means)
            mppi_s2_raw, _, mppi_s2_iter_cands_raw, mppi_s2_iter_q_np, mppi_s2_iter_means_raw = \
                to_raw_condition(mppi_s2_cands, mppi_s2_iter_cands, mppi_s2_iter_qs, mppi_s2_iter_means)
            bc_diff_raw = candidates_to_raw(bc_diff_cands, ctx)

        mppi_q_np = mppi_q.cpu().numpy()
        mppi_s2_q_np = mppi_s2_q.cpu().numpy()

        # ── Weighted means for overlay ────────────────────────────────────
        def weighted_mean_raw(cands_norm, q_vals_t):
            K = min(args.n_elites, args.n_samples)
            topk_idx = torch.topk(q_vals_t, K).indices
            topk_q = q_vals_t[topk_idx]; topk_c = cands_norm[topk_idx]
            w = torch.softmax((topk_q - topk_q.max()) / args.temperature, dim=0)
            mean_norm = (w.view(K, 1, 1) * topk_c).sum(dim=0, keepdim=True)
            with torch.no_grad():
                return candidates_to_raw(mean_norm.float(), ctx)[0]

        mppi_wmean_raw = weighted_mean_raw(mppi_cands, mppi_q)
        mppi_s2_wmean_raw = weighted_mean_raw(mppi_s2_cands, mppi_s2_q)
        with torch.no_grad():
            K_bcd = min(args.n_elites, args.n_samples)
            bcd_topk_idx = torch.topk(bc_diff_q, K_bcd).indices
            bcd_weights = torch.softmax(
                (bc_diff_q[bcd_topk_idx] - bc_diff_q[bcd_topk_idx].max()) / args.temperature, dim=0
            )
            bcd_wmean_norm = (bcd_weights.view(K_bcd, 1, 1) * bc_diff_cands[bcd_topk_idx]).sum(dim=0)
            bcd_wmean_raw = candidates_to_raw(bcd_wmean_norm.unsqueeze(0).float(), ctx)[0]

        # ── Save raw data ─────────────────────────────────────────────────
        save_step_data(
            step_idx=step_idx,
            mppi_candidates_raw=mppi_raw,
            mppi_q=mppi_q_np,
            mppi_iter_candidates_raw=mppi_iter_cands_raw,
            mppi_iter_q=mppi_iter_q_np,
            mppi_iter_means_raw=mppi_iter_means_raw,
            bc_diff_candidates_raw=bc_diff_raw,
            bc_diff_q=bc_diff_q_np,
            bc_mean_raw=bc_mean_raw,
            step_dir=step_dir,
            pca_rows=pca_rows,
        )
        # Also save mppi_smooth2 into the NPZ
        npz_path = step_dir / "data.npz"
        existing = dict(np.load(str(npz_path)))
        extra = {
            "mppi_smooth2_candidates_raw": mppi_s2_raw,
            "mppi_smooth2_q_values": mppi_s2_q_np,
        }
        for i, (cands, q_vals, mean) in enumerate(
            zip(mppi_s2_iter_cands_raw, mppi_s2_iter_q_np, mppi_s2_iter_means_raw)
        ):
            extra[f"mppi_smooth2_iter{i}_candidates_raw"] = cands
            extra[f"mppi_smooth2_iter{i}_q_values"] = q_vals
            extra[f"mppi_smooth2_iter{i}_mean_raw"] = mean
        np.savez_compressed(str(npz_path), **{**existing, **extra})

        # Add mppi_smooth2 rows to PCA CSV (same PCA fit as in save_step_data)
        from sklearn.decomposition import PCA as _PCA
        all_for_pca = np.concatenate([mppi_raw, bc_diff_raw, mppi_s2_raw], axis=0)
        N_all, h_all, A_all = all_for_pca.shape
        pca_fit = _PCA(n_components=2).fit(all_for_pca.reshape(N_all, h_all * A_all))
        for cond_name, cands_r, q_np in [
            ("mppi_smooth2", mppi_s2_raw, mppi_s2_q_np),
        ]:
            proj = pca_fit.transform(cands_r.reshape(len(cands_r), h_all * A_all))
            for i in range(len(cands_r)):
                pca_rows.append({
                    "step": step_idx, "condition": cond_name, "sample_idx": i,
                    "pc1": float(proj[i, 0]), "pc2": float(proj[i, 1]),
                    "q_value": float(q_np[i]),
                })

        # ── Overlays ──────────────────────────────────────────────────────
        mppi_overlays = [
            {"data": mppi_iter_means_raw[i], "label": f"mean iter {i+1}",
             "lw": 2.0 + 0.5 * i, "color": iter_colors[i], "linestyle": "-"}
            for i in range(args.n_mppi_iters)
        ]
        mppi_s2_overlays = [
            {"data": mppi_s2_iter_means_raw[i], "label": f"mean iter {i+1}",
             "lw": 2.0 + 0.5 * i, "color": iter_colors[i], "linestyle": "-"}
            for i in range(args.n_mppi_iters)
        ]
        bcd_overlay = [{"data": bcd_wmean_raw, "label": f"top-{args.n_elites} mean",
                        "lw": 2.5, "color": "#2a9d8f", "linestyle": "-"}]

        # ── Plots ─────────────────────────────────────────────────────────
        task_tag = args.task if is_robotwin else f"task_id={args.task_id}"

        plot_a_6dim(mppi_raw, bc_mean_raw, mppi_q_np,
            title=f"MPPI {args.n_mppi_iters}-iter (no smooth) — step {step_idx}  {task_tag}",
            out_path=step_dir / "mppi_plot_a.pdf", overlay_lines=mppi_overlays,
            plot_dims=plot_dims, dim_names=dim_names)
        plot_b_pca(mppi_raw, bc_mean_raw, mppi_q_np,
            title=f"MPPI {args.n_mppi_iters}-iter (no smooth) PCA — step {step_idx}  {task_tag}",
            out_path=step_dir / "mppi_plot_b.pdf", overlay_lines=mppi_overlays,
            spaghetti_dims=(plot_dims[0], plot_dims[1]), dim_names=dim_names)

        plot_a_6dim(mppi_s2_raw, bc_mean_raw, mppi_s2_q_np,
            title=f"MPPI {args.n_mppi_iters}-iter (smooth σ=2) — step {step_idx}  {task_tag}",
            out_path=step_dir / "mppi_smooth2_plot_a.pdf", overlay_lines=mppi_s2_overlays,
            plot_dims=plot_dims, dim_names=dim_names)
        plot_b_pca(mppi_s2_raw, bc_mean_raw, mppi_s2_q_np,
            title=f"MPPI {args.n_mppi_iters}-iter (smooth σ=2) PCA — step {step_idx}  {task_tag}",
            out_path=step_dir / "mppi_smooth2_plot_b.pdf", overlay_lines=mppi_s2_overlays,
            spaghetti_dims=(plot_dims[0], plot_dims[1]), dim_names=dim_names)

        plot_a_6dim(bc_diff_raw, bc_mean_raw, bc_diff_q_np,
            title=f"BC diffusion ({args.diffusion_steps} steps) — step {step_idx}  {task_tag}",
            out_path=step_dir / "bc_diff_plot_a.pdf", overlay_lines=bcd_overlay,
            plot_dims=plot_dims, dim_names=dim_names)
        plot_b_pca(bc_diff_raw, bc_mean_raw, bc_diff_q_np,
            title=f"BC diffusion ({args.diffusion_steps} steps) PCA — step {step_idx}  {task_tag}",
            out_path=step_dir / "bc_diff_plot_b.pdf", overlay_lines=bcd_overlay,
            spaghetti_dims=(plot_dims[0], plot_dims[1]), dim_names=dim_names)

        # ── Record Q data ─────────────────────────────────────────────────
        step_q = {
            "step": step_idx,
            "mppi": {"min": float(mppi_q_np.min()), "max": float(mppi_q_np.max()),
                     "mean": float(mppi_q_np.mean()), "std": float(mppi_q_np.std()),
                     "values": mppi_q_np.tolist()},
            "mppi_smooth2": {"min": float(mppi_s2_q_np.min()), "max": float(mppi_s2_q_np.max()),
                             "mean": float(mppi_s2_q_np.mean()), "std": float(mppi_s2_q_np.std()),
                             "values": mppi_s2_q_np.tolist()},
            "bc_diffusion": {"min": float(bc_diff_q_np.min()), "max": float(bc_diff_q_np.max()),
                             "mean": float(bc_diff_q_np.mean()), "std": float(bc_diff_q_np.std()),
                             "values": bc_diff_q_np.tolist()},
        }
        all_q_data.append(step_q)

        # ── Advance env using BC mean actions ─────────────────────────────
        if is_robotwin:
            # RoboTwin actions are raw joint targets — no gripper remap needed.
            for t in range(min(n_action_steps, h)):
                action_env = bc_mean_raw[t].astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action_env)
                if terminated or truncated:
                    print(f"  Episode ended at env step {t} of planning step {step_idx}.")
                    obs, _ = env.reset(seed=args.seed)
                    break
        else:
            # Libero: FastWAM trains gripper in [0,1]; env expects sign convention.
            from lerobot.processor.env_processor import FastWAMGripperRemapStep
            gripper_remap = FastWAMGripperRemapStep()
            for t in range(min(n_action_steps, h)):
                action_tensor = torch.from_numpy(bc_mean_raw[t]).float().unsqueeze(0)
                action_env = gripper_remap.action(action_tensor).squeeze(0).numpy()
                obs, reward, terminated, truncated, info = env.step(action_env)
                if terminated or truncated:
                    print(f"  Episode ended at env step {t} of planning step {step_idx}.")
                    obs, _ = env.reset(seed=args.seed)
                    break

    # ── Save Q value summary ─────────────────────────────────────────────────
    q_json_path = out_dir / "q_values.json"
    with open(q_json_path, "w") as f:
        json.dump({"task_id": args.task_id, "task": task_str, "steps": all_q_data}, f, indent=2)
    print(f"\nSaved Q-value summary to {q_json_path}")

    # ── Save PCA CSV ─────────────────────────────────────────────────────────
    import csv
    pca_csv_path = out_dir / "pca_data.csv"
    with open(pca_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "condition", "sample_idx", "pc1", "pc2", "q_value"])
        writer.writeheader()
        writer.writerows(pca_rows)
    print(f"Saved PCA data to {pca_csv_path}")

    env.close()
    print(f"\nDone. PDFs in {out_dir}/")


if __name__ == "__main__":
    main()
