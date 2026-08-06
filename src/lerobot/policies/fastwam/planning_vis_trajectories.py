"""XY-trajectory visualization for bc_diffusion planning candidates.

For each planning chunk in an episode, plots the N candidate action trajectories
in the XY translation plane, color-coded by Q value. The selected (weighted mean)
trajectory is overlaid in white. Produces a multi-panel figure per episode.

Usage (standalone):
    python -m lerobot.policies.fastwam.planning_vis_trajectories \
        --policy_path /path/to/fastwam_ckpt \
        --q_ckpt /path/to/q_ckpt \
        --output_dir outputs/vis/traj_debug \
        --n_episodes 3 \
        --n_samples 16 \
        --n_elites 4 \
        --diffusion_steps 3
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

logger = logging.getLogger(__name__)

# Action dim indices (FastWAM libero: [x, y, z, roll, pitch, yaw, gripper])
_DIM_X, _DIM_Y, _DIM_Z = 0, 1, 2
_DIM_GRIPPER = -1

# Dark theme: foreground must be set explicitly or matplotlib defaults to black
# text, which is unreadable on these backgrounds.
_FG = "white"
_BG_FIG = "#0d0d1a"
_BG_AX = "#1a1a2e"


def _q_to_colors(q_vals: np.ndarray, cmap_name: str = "plasma") -> np.ndarray:
    """Map (N,) Q values to (N, 4) RGBA colors via a colormap."""
    cmap = cm.get_cmap(cmap_name)
    q_min, q_max = q_vals.min(), q_vals.max()
    norm = (q_vals - q_min) / (q_max - q_min + 1e-8)
    return cmap(norm)


def plot_episode_trajectories(
    episode_chunks: list[dict],
    output_path: str | Path,
    episode_idx: int = 0,
    diffusion_steps: int | None = None,
    n_cols: int = 5,
) -> None:
    """Render multi-panel trajectory figure for one episode.

    Each panel = one planning chunk. Shows N candidate XY trajectories
    colour-coded by Q value (low=dark, high=bright), plus the selected
    (weighted mean) trajectory in white with higher z-order.

    Args:
        episode_chunks: list of dicts with keys:
            - ``action_candidates``: (N, h, A) float32 numpy array
            - ``action_selected``:  (h, A) float32 numpy array
            - ``q_candidates``:     (N,) float32 numpy array
            - ``frame``:            (H, W, 3) uint8 obs image or None
        output_path: path to save the figure (PNG)
        episode_idx: episode index for title
        diffusion_steps: number of diffusion steps used (for title)
        n_cols: number of columns in the grid
    """
    chunks_with_actions = [c for c in episode_chunks if "action_candidates" in c]
    n_panels = len(chunks_with_actions)
    if n_panels == 0:
        logger.warning("No action_candidates found in episode chunks — nothing to plot.")
        return

    n_rows = max(1, (n_panels + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.2, n_rows * 3.0),
        squeeze=False,
        constrained_layout=True,
    )
    steps_label = f"{diffusion_steps} steps" if diffusion_steps is not None else "full"
    fig.suptitle(
        f"BC Diffusion Trajectories — episode {episode_idx}, {steps_label}",
        fontsize=13, color=_FG,
    )

    for panel_idx, chunk in enumerate(chunks_with_actions):
        row, col = divmod(panel_idx, n_cols)
        ax = axes[row][col]

        cands = chunk["action_candidates"]   # (N, h, A)
        selected = chunk["action_selected"]   # (h, A)
        q_vals = chunk["q_candidates"]        # (N,)
        N, h, A = cands.shape

        colors = _q_to_colors(q_vals)

        # Plot candidate XY trajectories
        for i in range(N):
            xs = cands[i, :, _DIM_X]
            ys = cands[i, :, _DIM_Y]
            ax.plot(xs, ys, color=colors[i], linewidth=0.8, alpha=0.75)
            ax.scatter(xs[0], ys[0], color=colors[i], s=10, zorder=3)

        # Plot selected trajectory (white, thick)
        ax.plot(
            selected[:, _DIM_X], selected[:, _DIM_Y],
            color="white", linewidth=2.0, zorder=5, label="selected",
        )
        ax.scatter(
            selected[0, _DIM_X], selected[0, _DIM_Y],
            color="white", s=30, zorder=6, marker="*",
        )

        q_min, q_max = float(q_vals.min()), float(q_vals.max())
        ax.set_title(f"chunk {panel_idx}\nQ [{q_min:.3f}, {q_max:.3f}]", fontsize=7, color=_FG)
        ax.set_xlabel("x", fontsize=7, color=_FG)
        ax.set_ylabel("y", fontsize=7, color=_FG)
        ax.tick_params(labelsize=6, colors=_FG)
        ax.set_facecolor(_BG_AX)
        for spine in ax.spines.values():
            spine.set_color(_FG)
        ax.grid(True, color="gray", alpha=0.2, linewidth=0.5)

    # Add colorbar to last used axis
    sm = plt.cm.ScalarMappable(cmap="plasma")
    sm.set_array(q_vals)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), label="Q value", shrink=0.6, pad=0.02)
    cbar.set_label("Q value", color=_FG)
    cbar.ax.tick_params(colors=_FG)
    cbar.outline.set_edgecolor(_FG)

    # Hide unused panels
    for idx in range(n_panels, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, facecolor=_BG_FIG)
    plt.close(fig)
    logger.info("Saved trajectory vis: %s", output_path)


def run_vis_eval(
    policy_path: str,
    q_ckpt: str,
    output_dir: str,
    n_episodes: int = 3,
    n_samples: int = 16,
    n_elites: int = 4,
    diffusion_steps: int | None = None,
    temperature: float = 1.0,
    task: str = "libero_10",
    seed: int = 42,
) -> None:
    """Run a short eval with bc_diffusion_mppi and save trajectory figures."""
    import torch
    from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
    from lerobot.policies.fastwam.planning import FastWAMPlanner
    from lerobot.policies.act_simple.planning import PlanningConfig

    os.environ.setdefault(
        "DIFFSYNTH_MODEL_BASE_PATH",
        "/storage/project/r-agarg35-0/shared/awm/fastwam_wan22_weights",
    )
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading FastWAM from %s ...", policy_path)
    policy = FastWAMPolicy.from_pretrained(policy_path).to(device).eval()

    cfg = PlanningConfig(
        q_checkpoint_path=q_ckpt,
        planner_type="bc_diffusion_mppi",
        n_samples=n_samples,
        n_elites=n_elites,
        temperature=temperature,
        num_diffusion_steps=diffusion_steps,
    )
    planner = FastWAMPlanner.from_checkpoints(
        cfg=cfg,
        bc_post=policy.config,
        bc_chunk_size=policy.config.chunk_size,
        device=device,
    )
    policy.attach_planner(planner)

    # Import eval utilities
    from lerobot.envs.libero import make_libero_env
    from lerobot.processor import build_policy_processor

    processor = build_policy_processor(policy_path)
    env = make_libero_env(task=task, n_envs=1, seed=seed, obs_height=224, obs_width=224)

    steps_label = f"steps{diffusion_steps}" if diffusion_steps is not None else "stepsfull"
    for ep_idx in range(n_episodes):
        policy.reset()
        planner.start_episode()
        obs, _ = env.reset()

        done = False
        while not done:
            batch = processor(obs)
            with torch.no_grad():
                action = policy.select_action(batch)
            obs, _, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = bool(terminated[0] or truncated[0])

        ep_chunks = planner.end_episode()
        if ep_chunks:
            out_path = output_dir / f"ep{ep_idx:02d}_{steps_label}_trajectories.png"
            plot_episode_trajectories(
                ep_chunks,
                output_path=out_path,
                episode_idx=ep_idx,
                diffusion_steps=diffusion_steps,
            )

    env.close()
    logger.info("Done. Figures saved to %s", output_dir)


def _run_via_lerobot_eval(
    policy_path: str,
    q_ckpt: str,
    output_dir: str,
    n_episodes: int,
    n_samples: int,
    n_elites: int,
    diffusion_steps: int | None,
    temperature: float,
    task: str,
    seed: int,
) -> None:
    """Run via lerobot-eval subprocess, then generate figures from saved vis data."""
    import subprocess, sys, json, pickle

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    diffusion_steps_arg = (
        f"--policy.planning.num_diffusion_steps={diffusion_steps}"
        if diffusion_steps is not None else ""
    )

    cmd = [
        sys.executable, "-m", "lerobot.scripts.eval",
        f"--policy.path={policy_path}",
        "--policy.device=cuda",
        f"--env.type=libero",
        f"--env.task={task}",
        "--env.observation_height=224",
        "--env.observation_width=224",
        "--eval.batch_size=1",
        f"--eval.n_episodes={n_episodes}",
        "--policy.use_planning=true",
        f"--policy.planning.q_checkpoint_path={q_ckpt}",
        "--policy.planning.planner_type=bc_diffusion_mppi",
        f"--policy.planning.n_samples={n_samples}",
        f"--policy.planning.n_elites={n_elites}",
        f"--policy.planning.temperature={temperature}",
        f"--output_dir={output_dir}",
        f"--seed={seed}",
    ]
    if diffusion_steps is not None:
        cmd.append(f"--policy.planning.num_diffusion_steps={diffusion_steps}")

    subprocess.run(["echo", "N"], check=True)
    subprocess.run(cmd, input=b"N\n", check=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="BC diffusion trajectory visualizer")
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--q_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--n_elites", type=int, default=4)
    parser.add_argument("--diffusion_steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--task", default="libero_10")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_vis_eval(
        policy_path=args.policy_path,
        q_ckpt=args.q_ckpt,
        output_dir=args.output_dir,
        n_episodes=args.n_episodes,
        n_samples=args.n_samples,
        n_elites=args.n_elites,
        diffusion_steps=args.diffusion_steps,
        temperature=args.temperature,
        task=args.task,
        seed=args.seed,
    )
