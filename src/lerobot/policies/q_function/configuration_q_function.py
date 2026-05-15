"""Configuration for the Q-Function policy.

A categorical h-step TD critic for offline BC data, inspired by Q-chunking
(https://arxiv.org/abs/2507.07969). Scores an action chunk a_{t:t+h} at state s_t
using camera views encoded by DINOv2-small (finetuned end-to-end).

Output head is an HL-Gauss categorical critic (num_bins logits over [v_min, v_max]).
Loss is h-step TD with Polyak target network: see modeling_q_function.py.

This is the LeRobot port of the original ``imitation/policies/q_function/`` config,
adapted for the v1 MimicGen sim dataset:
  * 2 camera views (agentview + robot0_eye_in_hand) instead of the real-robot's 3
  * 84x84 images (DINOv2-patch-aligned at p=14 → 6x6 = 36 tokens / view)
  * 7-dim actions, no tail to drop (real-world had a 3-dim mobile base trailer)
  * 3 buckets in our v1 dataset: q5, q3_termjitter, play
"""

import logging
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig

logger = logging.getLogger(__name__)


# Default 2-camera set for the MimicGen LeRobot dataset format.
# These match what scripts/convert_mimicgen_to_lerobot.py emits:
#   agentview        → observation.images.image
#   robot0_eye_in_hand → observation.images.image2
_DEFAULT_CAMERA_KEYS = (
    "observation.images.image",
    "observation.images.image2",
)


@PreTrainedConfig.register_subclass("q_function")
@dataclass
class QFunctionConfig(PreTrainedConfig):
    """Q-function for h-step TD on offline BC data (sim, MimicGen v1 dataset)."""

    # ── Problem shapes ─────────────────────────────────────────────────────
    n_obs_steps: int = 1
    h: int = 20                       # action chunk length / TD horizon
    gamma: float = 0.99               # TD discount
    drop_action_tail: int = 0         # sim has 7-d EEF action, no mobile base trailer
    camera_keys: tuple[str, ...] = _DEFAULT_CAMERA_KEYS

    # ── Image preprocessing (handled inside the model, on-device) ──────────
    # MG v1 frames are 84x84. DINOv2 patch_size = 14, and 84 = 14*6, so the
    # native resolution is already patch-aligned. Each view yields 6*6 = 36
    # patch tokens (well below the 312/view the real-robot config used).
    image_resize_h: int = 84
    image_resize_w: int = 84

    # Normalisation for inputs (only visual is used; state is ignored).
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,  # loaded but unused
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # ── Vision front-end selection ────────────────────────────────────────
    # Either run DINOv2 live per step, or consume pre-cached ResNet18 features.
    vision_backbone: str = "dinov2"

    # ── DINOv2 backbone settings (used when vision_backbone == "dinov2") ───
    dino_model_name: str = "facebook/dinov2-small"  # 384-dim, patch 14, ~22M params
    freeze_backbone: bool = False  # finetune end-to-end

    # ── Cached-feature settings (used when vision_backbone == "resnet18_cached") ──
    preencoded_feature_channels: int = 512
    # Location of pre-encoded feature caches. When set, the QValueLabelDataset
    # wrapper looks for ``<precache_root>/<sub_repo_id>/meta.json`` (one cache
    # per dataset). When None, no cached features are loaded — the wrapper
    # serves whatever the underlying dataset returns.
    # Convention: ``<policy_checkpoint>/encoded_backbone`` (the precache writer's
    # default), so caches are namespaced by the BC checkpoint that produced them.
    precache_root: str | None = None

    # ── Action normalisation (toggle) ──────────────────────────────────────
    # When True, q_dataset.py applies (a - mean) / (std + eps) to action chunks
    # using stats loaded from the policy preprocessor (see policy_checkpoint_path
    # in the train driver). Effect is small (sim actions live in [-1, 1] already),
    # but matches the reference Q-function repo's pattern and stabilises action_proj.
    normalize_actions: bool = True

    # Optional path to an external safetensors file containing ``action.mean`` and
    # ``action.std`` to OVERRIDE the dataset-derived action stats inside Q's
    # NormalizerProcessorStep. Use case: when BC was trained on a subset of the
    # dataset (e.g. only q5 demos) and Q is trained on a superset (q5+q3+play),
    # BC's chunk-output sits off-center in Q's frame at eval. Pointing this at
    # BC's saved unnormalizer aligns the two frames so the planner can skip the
    # bc_post → q_pre round-trip distortion at inference.
    action_stats_path: str | None = None

    # ── Transformer decoder body ───────────────────────────────────────────
    dim_model: int = 384
    n_heads: int = 6
    dim_feedforward: int = 1536
    feedforward_activation: str = "gelu"
    n_decoder_layers: int = 4
    dropout: float = 0.1
    pre_norm: bool = True

    # ── Pool strategy for decoder output ───────────────────────────────────
    pool: str = "cls"                 # "cls" or "mean"

    # ── HL-Gauss categorical critic ────────────────────────────────────────
    # Worst-case sparse return on a 250-step ep with the most-negative bucket bonus:
    #   sum_t step_reward + terminal_bonus = -250 + -500 = -750.
    # v_min=-800 leaves a small margin; bin width = 800/100 = 8.
    v_min: float = -800.0
    v_max: float = 0.0
    num_bins: int = 101
    hl_gauss_sigma: float = 6.0       # ~0.75 * bin_width

    # ── Terminal-reward bonuses by quality bucket ──────────────────────────
    # Applied at the terminal step, unscaled, in both reward modes.
    # Keys must match the buckets emitted by experiments/mg_dataset_v1/q_dataset.py.
    terminal_bonuses: dict[str, float] = field(
        default_factory=lambda: {
            "q5":            0.0,
            "q3_termjitter": -100.0,
            "play":          -500.0,
        }
    )
    step_reward: float = -1.0

    # ── Reward shaping mode ────────────────────────────────────────────────
    # "sparse"     — per-step reward = step_reward everywhere + terminal bonus.
    # "time_to_go" — for quality buckets, per-step reward = quality_scalars[bucket]
    #                * -(T - t); play bucket is always sparsely labelled.
    # We default to sparse for v1 (the user's pick); time_to_go retained as an option.
    reward_mode: str = "sparse"
    quality_scalars: dict[str, float] = field(
        default_factory=lambda: {
            "q5":            1.0,
            "q3_termjitter": 2.0,
            "play":          5.0,  # unused when play is forced to sparse
        }
    )
    # Conservative upper bound on episode length, used for v_min auto-sizing
    # in time_to_go mode. v1 caps episodes at 200 (q5/q3_term mean ~150-220, play ~200).
    max_ep_length_hint: int = 250

    # ── Target network (Polyak) ────────────────────────────────────────────
    target_tau: float = 0.005

    # ── Training preset ────────────────────────────────────────────────────
    optimizer_lr: float = 1e-4
    optimizer_lr_backbone: float = 3e-5   # lower LR for DINOv2 weights
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if self.num_bins < 3:
            raise ValueError(f"num_bins must be >= 3, got {self.num_bins}")
        if self.pool not in ("cls", "mean"):
            raise ValueError(f"pool must be 'cls' or 'mean', got {self.pool}")
        if self.h <= 0:
            raise ValueError(f"h must be > 0, got {self.h}")
        if self.vision_backbone not in ("dinov2", "resnet18_cached"):
            raise ValueError(
                f"vision_backbone must be 'dinov2' or 'resnet18_cached', got {self.vision_backbone!r}"
            )
        if self.reward_mode not in ("sparse", "time_to_go"):
            raise ValueError(
                f"reward_mode must be 'sparse' or 'time_to_go', got {self.reward_mode!r}"
            )
        if self.max_ep_length_hint <= 0:
            raise ValueError(f"max_ep_length_hint must be > 0, got {self.max_ep_length_hint}")
        missing_scalars = set(self.terminal_bonuses) - set(self.quality_scalars)
        if missing_scalars:
            raise ValueError(
                f"quality_scalars missing entries for buckets: {sorted(missing_scalars)}"
            )

        # Auto-size v_min for time_to_go mode when the user-supplied value is too tight.
        if self.reward_mode == "time_to_go":
            worst = self._worst_case_return_bound()
            if self.v_min > worst:
                logger.info(
                    "reward_mode=time_to_go: overriding v_min=%.1f → %.1f "
                    "(worst-case discounted return at T=%d, gamma=%.4f). "
                    "Set v_min explicitly below this to silence.",
                    self.v_min, worst, self.max_ep_length_hint, self.gamma,
                )
                self.v_min = worst

        if not (self.v_min < self.v_max):
            raise ValueError(f"Expected v_min < v_max, got v_min={self.v_min}, v_max={self.v_max}")

    def _worst_case_return_bound(self) -> float:
        T = int(self.max_ep_length_hint)
        gamma = float(self.gamma)
        worst = float("inf")
        for bucket, bonus in self.terminal_bonuses.items():
            if bucket == "play":
                ret = 0.0
                for t in range(T):
                    ret += (gamma ** t) * self.step_reward
                ret += (gamma ** (T - 1)) * bonus
            else:
                scalar = float(self.quality_scalars.get(bucket, 1.0))
                ret = 0.0
                for t in range(T):
                    ret += (gamma ** t) * scalar * -(T - t)
                ret += (gamma ** (T - 1)) * bonus
            worst = min(worst, ret)
        return float(int(worst * 1.05))

    # ── PreTrainedConfig abstract API ──────────────────────────────────────

    @property
    def observation_delta_indices(self) -> list[int]:
        # Load s_t and s_{t+h} for each camera.
        return [0, self.h]

    @property
    def action_delta_indices(self) -> list[int]:
        # Load a_{t:t+2h} to cover the current chunk + the bootstrap chunk.
        return list(range(0, 2 * self.h))

    @property
    def reward_delta_indices(self) -> None:
        # Rewards are synthesised at dataload time (q_dataset.py) — not stored on disk.
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("Q-function requires at least one image feature.")
