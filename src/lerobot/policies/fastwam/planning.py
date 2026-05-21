"""Online Q-planning for FastWAM: perturb BC chunk, score with Q, aggregate.

FastWAM's diffusion runs once to produce a BC mean chunk; we then sample N noisy
variants, score each with a trained Q-function, and aggregate via MPPI/CEM/argmax.

Key difference from act_simple planning:
- BC mean comes from ``predict_action_chunk()`` (20-step diffusion) not ``model()``
- No backbone precomputation: Q's DINOv2 encodes raw camera images internally,
  so we pass ``{cam_key: batch[cam_key]}`` directly as ``img_feats``
- Actions round-trip via FastWAM postprocessor (MIN_MAX→raw) then Q preprocessor
  (raw→MEAN_STD), same as act_simple but with different normalization modes
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

from lerobot.policies.act_simple.planning import (
    PlannerContext,
    PlanningConfig,
    _sample_noise,
    _score_candidates,
)

if TYPE_CHECKING:
    from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
    from lerobot.processor import PolicyProcessorPipeline


class FastWAMPlanner:
    """Q-scored planner attached to a ``FastWAMPolicy`` at eval time.

    Mirrors ``act_simple.planning.Planner`` but uses ``predict_action_chunk``
    (diffusion) for the BC prior and passes raw images to the DINOv2 Q-function.
    """

    def __init__(
        self,
        cfg: PlanningConfig,
        ctx: PlannerContext,
        generator: torch.Generator | None = None,
    ):
        self.cfg = cfg
        self.ctx = ctx
        self.generator = generator
        self.last_q_spread: tuple[float, float, float, float] | None = None
        # Vis recording state (populated when start_episode() is called)
        self._vis_chunks: list[dict] | None = None   # None = not recording
        self._completed_episodes: list[list[dict]] = []

    @classmethod
    def from_checkpoints(
        cls,
        cfg: PlanningConfig,
        bc_post: "PolicyProcessorPipeline",
        bc_chunk_size: int,
        device: torch.device,
    ) -> "FastWAMPlanner":
        """Load Q + Q preprocessor from ``cfg.q_checkpoint_path`` and return a ready planner."""
        from lerobot.policies.q_function.modeling_q_function import QFunctionPolicy
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import batch_to_transition, transition_to_batch
        from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME

        if not cfg.q_checkpoint_path:
            raise ValueError(
                "PlanningConfig.q_checkpoint_path must be set "
                "(via --policy.planning.q_checkpoint_path=...)"
            )
        q_policy = QFunctionPolicy.from_pretrained(cfg.q_checkpoint_path).to(device).eval()
        q_h = int(q_policy.config.h)
        if q_h != bc_chunk_size:
            raise ValueError(
                f"Q horizon (h={q_h}) must equal FastWAM chunk_size ({bc_chunk_size}). "
                "Retrain Q with --policy.h matching FastWAM chunk_size."
            )
        q_pre = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=cfg.q_checkpoint_path,
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        # FastWAM action normalization mismatch fix:
        # FastWAM bc_post (MIN_MAX) maps gripper from norm [-1,1] → raw [0,1], but Q was
        # trained with gripper in [-1,1].  FastWAM normalized space and Q's raw space are both
        # in [-1,1] for ALL dims, so the correct bc_post for scoring is identity — pass
        # FastWAM-normalized candidates directly to q_pre without unnormalizing first.
        ctx = PlannerContext(
            q_policy=q_policy,
            q_pre=q_pre,
            bc_post=lambda x: x,  # identity: FastWAM-norm ≈ Q-raw (both in [-1,1])
            q_camera_keys=tuple(q_policy.config.camera_keys),
            horizon=bc_chunk_size,
        )
        gen = (
            torch.Generator(device=device).manual_seed(cfg.seed)
            if cfg.seed is not None
            else None
        )
        return cls(cfg=cfg, ctx=ctx, generator=gen)

    def start_episode(self) -> None:
        """Begin accumulating vis data for a new episode."""
        self._vis_chunks = []

    def end_episode(self) -> list[dict] | None:
        """Finalise the current episode and return its chunk data (or None if not recording)."""
        chunks = self._vis_chunks
        if chunks is not None and chunks:
            self._completed_episodes.append(chunks)
        self._vis_chunks = None
        return chunks

    def pop_completed_episode(self) -> list[dict] | None:
        """Return and remove the oldest completed episode's vis data."""
        return self._completed_episodes.pop(0) if self._completed_episodes else None

    @torch.no_grad()
    def plan(self, bc_policy: "FastWAMPolicy", batch: dict[str, Tensor]) -> Tensor:
        """Return planned chunk ``(1, h, A)`` in FastWAM-normalized space."""
        result, spread, q_vals_raw = plan_chunk_fastwam(
            bc_policy, batch, self.ctx, self.cfg, self.generator
        )
        self.last_q_spread = spread
        if spread is not None:
            logger.info(
                "Q-planning chunk: planner=%s  q_min=%.4f  q_max=%.4f  q_mean=%.4f  q_std=%.4f",
                self.cfg.planner_type, spread[0], spread[1], spread[2], spread[3],
            )

        # Accumulate vis data when recording
        if self._vis_chunks is not None and q_vals_raw is not None:
            from lerobot.policies.fastwam.planning_vis import _obs_frame_to_uint8
            from lerobot.utils.constants import OBS_IMAGES
            cam_key = f"{OBS_IMAGES}.image"
            frame = _obs_frame_to_uint8(batch[cam_key]) if cam_key in batch else None
            q_np = q_vals_raw.cpu().float().numpy()
            N = len(q_np)
            # MPPI weighted mean of Q values ≈ Q of the selected action
            if self.cfg.planner_type == "mppi" and spread is not None:
                weights = torch.softmax(
                    (q_vals_raw - q_vals_raw.max()) / self.cfg.temperature, dim=0
                )
                q_sel = float((weights * q_vals_raw).sum())
            else:
                q_sel = float(q_np.max())  # argmax / CEM: use best candidate Q
            self._vis_chunks.append({"frame": frame, "q_candidates": q_np, "q_selected": q_sel})

        return result


@torch.no_grad()
def plan_chunk_fastwam(
    bc_policy: "FastWAMPolicy",
    batch: dict[str, Tensor],
    ctx: PlannerContext,
    cfg: PlanningConfig,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, tuple[float, float, float, float] | None, Tensor | None]:
    """Return ``(planned_chunk, q_spread, q_values)`` for one chunk in FastWAM-norm space.

    ``planned_chunk`` shape: ``(1, h, A)`` — ready to drop into FastWAM's action queue.
    ``q_spread``: ``(q_min, q_max, q_mean, q_std)`` across candidate scores.
    ``q_values``: raw ``(N,)`` Q scores for all candidates (for vis / logging).
    """
    device = next(bc_policy.parameters()).device
    # Run FastWAM diffusion once → MIN_MAX-normalized actions, returned on CPU.
    bc_mean = bc_policy.predict_action_chunk(batch).to(device=device)
    if bc_mean.shape[0] != 1:
        raise NotImplementedError(
            f"Q-planning supports batch_size=1 only (got {bc_mean.shape[0]}). "
            "Run lerobot-eval with --eval.batch_size=1 when use_planning=true."
        )
    _, h, A = bc_mean.shape
    N = cfg.n_samples

    # Raw camera images + task text — Q uses DINOv2 + T5 text conditioning.
    img_feats = {cam_key: batch[cam_key] for cam_key in ctx.q_camera_keys}
    if "task" in batch:
        img_feats["task"] = batch["task"]

    def _spread(q: Tensor) -> tuple[float, float, float, float]:
        return (float(q.min()), float(q.max()), float(q.mean()), float(q.std()))

    if cfg.planner_type == "mppi":
        # Iterative MPPI: re-center on weighted mean each iteration for convergence.
        # n_iters=1 reproduces standard single-pass MPPI.
        mean = bc_mean.clone()
        q_values = None
        for _ in range(max(1, cfg.n_iters)):
            noise = _sample_noise(
                (N, h, A), cfg.noise_std, cfg.clip_to, device, mean.dtype,
                generator, smooth_sigma_t=cfg.noise_smooth_sigma_t,
            )
            candidates = mean.expand(N, h, A) + noise
            q_values = _score_candidates(candidates, img_feats, ctx).to(device=device, dtype=mean.dtype)
            weights = torch.softmax((q_values - q_values.max()) / cfg.temperature, dim=0)
            mean = (weights.view(N, 1, 1) * candidates).sum(dim=0, keepdim=True)
        return mean, _spread(q_values), q_values

    noise = _sample_noise(
        (N, h, A), cfg.noise_std, cfg.clip_to, device, bc_mean.dtype,
        generator, smooth_sigma_t=cfg.noise_smooth_sigma_t,
    )
    candidates = bc_mean.expand(N, h, A) + noise

    if cfg.planner_type == "argmax":
        q_values = _score_candidates(candidates, img_feats, ctx).to(device=device, dtype=bc_mean.dtype)
        best = int(torch.argmax(q_values))
        return candidates[best : best + 1], _spread(q_values), q_values

    # CEM — return q_values from last iteration
    mean = bc_mean.clone()
    std = torch.full_like(mean, cfg.noise_std).clamp_min(1e-3)
    spread = None
    q_values = None
    for _ in range(cfg.n_iters):
        noise_it = _sample_noise(
            (N, h, A), 1.0, cfg.clip_to, device, bc_mean.dtype, generator,
            smooth_sigma_t=cfg.noise_smooth_sigma_t,
        )
        cands = mean.expand(N, h, A) + std.expand(N, h, A) * noise_it
        q_values = _score_candidates(cands, img_feats, ctx).to(device=device, dtype=bc_mean.dtype)
        spread = _spread(q_values)
        elite_idx = torch.topk(q_values, cfg.n_elites).indices
        elites = cands.index_select(0, elite_idx)
        mean = elites.mean(dim=0, keepdim=True)
        std = elites.std(dim=0, keepdim=True).clamp_min(1e-3)
    return mean, spread, q_values
