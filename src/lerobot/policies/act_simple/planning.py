"""Online planning for ACT-Simple with a Q-function critic.

At each chunk boundary, BC produces a deterministic 10-step action chunk in its
normalized action space. We sample N noisy variants of that chunk, score each
with a trained Q-function (h must equal BC.chunk_size), and either softmax-weight
the elite candidates (MPPI) or iteratively refit a Gaussian over top-k (CEM).
The resulting normalized chunk is fed into BC's action queue and consumed one
step at a time via the usual ``select_action`` path.

Why a round-trip ``bc_post → q_pre`` on candidates: BC and Q were trained on
overlapping but distinct dataset mixtures, so their action mean/std may differ.
Unnormalizing via BC's postprocessor and re-normalizing via Q's preprocessor
matches the action statistics Q saw at training time exactly.

The vision backbone is shared: we run ``bc_policy.model.backbone(image)`` once
per chunk boundary to produce the ``(B, 512, 3, 3)`` features Q's cached encoder
expects (those weights are FrozenBatchNorm2d and identical to the offline
precache's). Features are tiled along the candidate dim, so the cost per chunk
is one BC backbone forward + N Q-decoder forwards.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lerobot.utils.constants import ACTION

if TYPE_CHECKING:
    from lerobot.policies.act_simple.modeling_act_simple import ACTSimplePolicy
    from lerobot.policies.q_function.modeling_q_function import QFunctionPolicy
    from lerobot.processor import PolicyProcessorPipeline


@dataclass
class PlanningConfig:
    """Hyperparameters for MPPI / CEM / argmax planning around BC's chunk output."""

    # Path to the Q-function checkpoint to use for scoring candidate action chunks.
    # Required when ``ACTSimpleConfig.use_planning=True``; loaded at eval time.
    q_checkpoint_path: str | None = None

    # ``str`` (not ``Literal``) so draccus can parse this from a CLI string. We
    # validate the choice explicitly in ``__post_init__``.
    planner_type: str = "mppi"
    n_samples: int = 64
    noise_std: float = 0.3
    temperature: float = 1.0
    n_elites: int = 16
    n_iters: int = 3
    clip_to: float | None = None
    seed: int | None = None
    # When set, IID Gaussian noise is low-pass filtered along the time axis with
    # a 1D Gaussian kernel of this bandwidth (in timesteps). Keeps perturbed
    # candidate chunks temporally smooth — closer to the BC training distribution
    # than IID per-step jitter.
    noise_smooth_sigma_t: float | None = None

    def __post_init__(self):
        if self.planner_type not in ("mppi", "cem", "argmax"):
            raise ValueError(
                f"planner_type must be one of {{mppi, cem, argmax}}, got {self.planner_type!r}"
            )
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be > 0, got {self.n_samples}")
        if self.noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {self.noise_std}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.planner_type == "cem":
            if self.n_elites <= 0 or self.n_elites > self.n_samples:
                raise ValueError(
                    f"n_elites must be in (0, n_samples={self.n_samples}], got {self.n_elites}"
                )
            if self.n_iters < 1:
                raise ValueError(f"n_iters must be >= 1, got {self.n_iters}")


@dataclass
class PlannerContext:
    """External references the planner needs at each chunk: the Q-function, Q's
    preprocessor (renormalizes ACTION via Q's stats), BC's postprocessor
    (unnormalizes BC-space candidates to raw), the Q camera keys to pull from the
    BC batch, and the chunk horizon (must equal both BC.chunk_size and Q.h)."""

    q_policy: "QFunctionPolicy"
    q_pre: "PolicyProcessorPipeline"
    bc_post: "PolicyProcessorPipeline"
    q_camera_keys: tuple[str, ...]
    horizon: int


class Planner:
    """Q-scored planner attached to an ``ACTSimplePolicy`` at eval time.

    The policy's only knowledge of planning is "if I have a planner, call its
    ``plan(self, batch)`` instead of the deterministic BC chunk". Everything
    else — loading the Q checkpoint, building Q's preprocessor, sanity-checking
    the horizon match, RNG state, per-chunk diagnostics — lives here.
    """

    def __init__(
        self,
        cfg: PlanningConfig,
        ctx: PlannerContext,
        generator: "torch.Generator | None" = None,
    ):
        self.cfg = cfg
        self.ctx = ctx
        self.generator = generator
        # (q_min, q_max, q_mean, q_std) of the most recent chunk's candidate scores.
        self.last_q_spread: tuple[float, float, float, float] | None = None

    @classmethod
    def from_checkpoints(
        cls,
        cfg: PlanningConfig,
        bc_post: "PolicyProcessorPipeline",
        bc_chunk_size: int,
        device: "torch.device",
    ) -> "Planner":
        """Load Q + Q's preprocessor from ``cfg.q_checkpoint_path`` and assemble a ready planner."""
        from lerobot.policies.q_function.modeling_q_function import QFunctionPolicy
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import batch_to_transition, transition_to_batch
        from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME

        if not cfg.q_checkpoint_path:
            raise ValueError(
                "PlanningConfig.q_checkpoint_path must be set "
                "(via --policy.planning.q_checkpoint_path=...)"
            )
        q_checkpoint_path = cfg.q_checkpoint_path
        q_policy = QFunctionPolicy.from_pretrained(q_checkpoint_path).to(device).eval()
        q_h = int(q_policy.config.h)
        if q_h != int(bc_chunk_size):
            raise ValueError(
                f"Q horizon (h={q_h}) must equal BC chunk_size ({bc_chunk_size}) "
                "for direct planning. Retrain Q with --policy.h matching the BC chunk_size."
            )
        q_pre = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=q_checkpoint_path,
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        ctx = PlannerContext(
            q_policy=q_policy,
            q_pre=q_pre,
            bc_post=bc_post,
            q_camera_keys=tuple(q_policy.config.camera_keys),
            horizon=int(bc_chunk_size),
        )
        gen = (
            torch.Generator(device=device).manual_seed(cfg.seed)
            if cfg.seed is not None else None
        )
        return cls(cfg=cfg, ctx=ctx, generator=gen)

    @torch.no_grad()
    def plan(self, bc_policy: "ACTSimplePolicy", batch: dict[str, Tensor]) -> Tensor:
        """Score noisy variants of BC's chunk and return the aggregated plan in BC-norm space."""
        result, last_q_spread = plan_chunk(bc_policy, batch, self.ctx, self.cfg, self.generator)
        self.last_q_spread = last_q_spread
        return result


def _backbone_features(bc_policy: "ACTSimplePolicy", batch: dict[str, Tensor],
                       camera_keys: tuple[str, ...]) -> dict[str, Tensor]:
    """Run BC's ResNet18 backbone on the current observation, keyed for Q.

    Returns a dict mapping ``{cam}_preencoded`` → ``(1, 512, 3, 3)``.
    """
    feats: dict[str, Tensor] = {}
    for cam_key in camera_keys:
        img = batch[cam_key]
        if img.dim() == 3:
            img = img.unsqueeze(0)
        feats[f"{cam_key}_preencoded"] = bc_policy.model.backbone(img)["feature_map"]
    return feats


def _score_candidates(
    candidates_norm: Tensor,
    img_feats: dict[str, Tensor],
    ctx: PlannerContext,
) -> Tensor:
    """Unnormalize → renormalize → Q-score N candidate chunks. Returns (N,)."""
    N, h, A = candidates_norm.shape
    flat_norm = candidates_norm.reshape(N * h, A)
    flat_raw = ctx.bc_post(flat_norm)
    candidates_raw = flat_raw.reshape(N, h, A).to(candidates_norm.device)

    q_batch: dict[str, Tensor] = {ACTION: candidates_raw}
    for cam_key, feat in img_feats.items():
        q_batch[cam_key] = feat.expand(N, *feat.shape[1:]).contiguous()

    q_batch = ctx.q_pre(q_batch)
    return ctx.q_policy.predict_value(q_batch)


def _gaussian_kernel_1d(sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    """1D Gaussian kernel, length ~ 6*sigma+1, normalized to sum to 1."""
    radius = max(1, int(round(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (xs / max(sigma, 1e-6)) ** 2)
    return k / k.sum()


def _smooth_time(noise: Tensor, sigma: float) -> Tensor:
    """Low-pass filter noise along the time axis (dim=1) with a 1D Gaussian kernel."""
    import torch.nn.functional as F  # noqa: N812
    N, h, A = noise.shape
    if sigma <= 0:
        return noise
    kernel = _gaussian_kernel_1d(sigma, noise.device, noise.dtype)  # (K,)
    radius = (kernel.shape[0] - 1) // 2
    x = noise.permute(0, 2, 1).reshape(N * A, 1, h)
    x = F.pad(x, (radius, radius), mode="replicate")
    y = F.conv1d(x, kernel.view(1, 1, -1))
    y = y.reshape(N, A, h).permute(0, 2, 1).contiguous()
    # Rescale to preserve marginal std (the filter shrinks variance).
    scale = noise.std(unbiased=False) / (y.std(unbiased=False) + 1e-8)
    return y * scale


def _sample_noise(shape: tuple[int, int, int], std: float, clip_to: float | None,
                  device: torch.device, dtype: torch.dtype,
                  generator: torch.Generator | None,
                  smooth_sigma_t: float | None = None) -> Tensor:
    if std == 0.0:
        return torch.zeros(shape, device=device, dtype=dtype)
    if generator is not None:
        noise = torch.randn(shape, device=device, dtype=dtype, generator=generator) * std
    else:
        noise = torch.randn(shape, device=device, dtype=dtype) * std
    if smooth_sigma_t is not None and smooth_sigma_t > 0:
        noise = _smooth_time(noise, smooth_sigma_t)
    if clip_to is not None:
        noise = noise.clamp(-clip_to, clip_to)
    return noise


@torch.no_grad()
def plan_chunk(
    bc_policy: "ACTSimplePolicy",
    batch: dict[str, Tensor],
    ctx: PlannerContext,
    cfg: PlanningConfig,
    generator: torch.Generator | None = None,
) -> "tuple[Tensor, tuple[float, float, float, float] | None]":
    """Return ``(planned_chunk, q_spread)`` for one chunk.

    ``planned_chunk`` has shape ``(1, h, A)`` in BC-normalized space — ready to
    feed into BC's action queue. ``q_spread`` is ``(q_min, q_max, q_mean, q_std)``
    across the candidate set on the most recent inner pass, or None if no scoring
    happened (unreachable for current planner types but reserved for future).

    ``batch`` must already be through BC's preprocessor (images normalized,
    OBS_IMAGES list constructed by ``predict_action_chunk``). Currently supports
    batch size B=1 only; for ``lerobot-eval``, pass ``--eval.batch_size=1``.
    """
    bc_mean = bc_policy.model(batch)
    if bc_mean.shape[0] != 1:
        raise NotImplementedError(
            f"Q-planning currently supports batch_size=1 only (got {bc_mean.shape[0]}). "
            "Run lerobot-eval with --eval.batch_size=1 when --policy.use_q_planning=true."
        )
    if bc_mean.shape[1] != ctx.horizon:
        raise RuntimeError(
            f"BC chunk length {bc_mean.shape[1]} != planner horizon {ctx.horizon}"
        )
    _, h, A = bc_mean.shape
    device, dtype = bc_mean.device, bc_mean.dtype
    N = cfg.n_samples

    img_feats = _backbone_features(bc_policy, batch, ctx.q_camera_keys)

    noise = _sample_noise((N, h, A), cfg.noise_std, cfg.clip_to, device, dtype,
                          generator, smooth_sigma_t=cfg.noise_smooth_sigma_t)
    candidates = bc_mean.expand(N, h, A) + noise

    def _spread(q: Tensor) -> tuple[float, float, float, float]:
        return (
            float(q.min().item()), float(q.max().item()),
            float(q.mean().item()), float(q.std().item()),
        )

    if cfg.planner_type == "mppi":
        q_values = _score_candidates(candidates, img_feats, ctx).to(device=device, dtype=dtype)
        spread = _spread(q_values)
        weights = torch.softmax((q_values - q_values.max()) / cfg.temperature, dim=0)
        planned = (weights.view(N, 1, 1) * candidates).sum(dim=0, keepdim=True)
        return planned, spread

    if cfg.planner_type == "argmax":
        q_values = _score_candidates(candidates, img_feats, ctx).to(device=device, dtype=dtype)
        spread = _spread(q_values)
        best = int(torch.argmax(q_values).item())
        return candidates[best : best + 1], spread

    # CEM
    mean = bc_mean.clone()
    std = torch.full_like(mean, cfg.noise_std).clamp_min(1e-3)
    spread = None
    for _ in range(cfg.n_iters):
        noise_it = _sample_noise((N, h, A), 1.0, cfg.clip_to, device, dtype, generator,
                                 smooth_sigma_t=cfg.noise_smooth_sigma_t)
        candidates_it = mean.expand(N, h, A) + std.expand(N, h, A) * noise_it
        q_values = _score_candidates(candidates_it, img_feats, ctx).to(device=device, dtype=dtype)
        spread = _spread(q_values)
        elite_idx = torch.topk(q_values, cfg.n_elites, dim=0).indices
        elites = candidates_it.index_select(0, elite_idx)
        mean = elites.mean(dim=0, keepdim=True)
        std = elites.std(dim=0, keepdim=True).clamp_min(1e-3)
    return mean, spread
