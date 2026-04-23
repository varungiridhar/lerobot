#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.policies.act_simple_with_awm_head.planning import PlanningConfig


@PreTrainedConfig.register_subclass("act_simple_with_awm_head")
@dataclass
class ACTSimpleWithAWMHeadConfig(PreTrainedConfig):
    """Configuration for ACT Simple + World Model Head policy.

    Combines a diffusion-policy action head with the world model decoder from AWM.
    The action generator uses the same diffusion U-Net setup as the diffusion policy,
    while the world model decoder keeps the existing transformer-based latent prediction path.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy.
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation.
        vision_backbone: Name of the torchvision resnet backbone for encoding images.
        pretrained_backbone_weights: Pretrained weights to initialize the backbone.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final stride with dilation.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads in multi-head attention.
        dim_feedforward: The feed-forward expansion dimension.
        feedforward_activation: Activation function for feed-forward layers.
        n_encoder_layers: Number of transformer encoder layers.
        n_decoder_layers: Number of transformer decoder layers (action decoder).
        dropout: Dropout rate for transformer layers.
        wm_loss_weight: Weight on world model loss relative to action prediction loss.
        wm_warmup_steps: Number of steps to linearly ramp wm_loss_weight from 0 to target.
        detach_encoder_from_wm: Detach encoder outputs before WM cross-attention.
        n_wm_decoder_layers: Number of layers in the world model decoder.
        use_ema_target: Use an EMA copy of the encoder to compute z_target.
        ema_momentum: EMA decay coefficient.
        ema_momentum_end: Final EMA momentum after annealing.
        ema_anneal_steps: Steps over which to anneal EMA momentum.
        decoder_loss_weight: Weight on image reconstruction loss.
        n_image_viz_pairs: Number of GT/decoded image pairs to log.
    """

    # Input / output structure.
    n_obs_steps: int = 2
    chunk_size: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # The diffusion-policy loader drops the final frames to avoid excessive action padding.
    drop_n_last_frames: int = 7  # chunk_size - n_action_steps - n_obs_steps + 1

    # Action diffusion branch: shared full-frame vision backbone + spatial-softmax preprocessing.
    action_vision_backbone: str = "resnet18"
    action_crop_shape: tuple[int, int] | None = None
    action_crop_is_random: bool = False
    action_pretrained_backbone_weights: str | None = None
    action_use_group_norm: bool = True
    action_spatial_softmax_num_keypoints: int = 32
    action_use_separate_rgb_encoder_per_camera: bool = False

    # Action diffusion branch: U-Net + scheduler.
    action_down_dims: tuple[int, ...] = (512, 1024, 2048)
    action_kernel_size: int = 5
    action_n_groups: int = 8
    action_diffusion_step_embed_dim: int = 128
    action_use_film_scale_modulation: bool = True
    action_noise_scheduler_type: str = "DDPM"
    action_num_train_timesteps: int = 100
    action_beta_schedule: str = "squaredcos_cap_v2"
    action_beta_start: float = 0.0001
    action_beta_end: float = 0.02
    action_prediction_type: str = "epsilon"
    action_clip_sample: bool = True
    action_clip_sample_range: float = 1.0
    action_num_inference_steps: int | None = None
    action_do_mask_loss_for_padding: bool = False

    # World-model architecture.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False

    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4

    # Training and loss computation.
    dropout: float = 0.1

    # World model head.
    wm_loss_weight: float = 0.2
    wm_warmup_steps: int = 0
    detach_encoder_from_wm: bool = False
    n_wm_decoder_layers: int = 4
    use_ema_target: bool = False
    ema_momentum: float = 0.996
    ema_momentum_end: float = 0.999
    ema_anneal_steps: int = 50_000
    normalize_wm_representations: bool = False  # L2-normalize z_pred and z_target to unit sphere before WM loss and image decoding
    decoder_loss_weight: float = 0.1
    n_image_viz_pairs: int = 12

    # Training preset.
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    # Test-time planning.
    use_planning: bool = False
    planning: PlanningConfig = field(default_factory=PlanningConfig)

    # Deprecated — kept for checkpoint compatibility with older configs.
    image_resize: int | None = None
    wm_visual_pool: bool = False
    wm_pool_size: int = 9
    log_wm_action_sensitivity: bool = False

    def __post_init__(self):
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if not self.action_vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`action_vision_backbone` must be one of the ResNet variants. Got {self.action_vision_backbone}."
            )
        if self.vision_backbone != self.action_vision_backbone:
            raise ValueError(
                "The WM and action branches now share a single visual backbone, so "
                f"`vision_backbone` and `action_vision_backbone` must match. Got "
                f"{self.vision_backbone=} and {self.action_vision_backbone=}."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps <= 0:
            raise ValueError(
                f"`n_obs_steps` must be positive. Got {self.n_obs_steps}."
            )
        supported_prediction_types = ["epsilon", "sample"]
        if self.action_prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`action_prediction_type` must be one of {supported_prediction_types}. "
                f"Got {self.action_prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.action_noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`action_noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.action_noise_scheduler_type}."
            )
        downsampling_factor = 2 ** len(self.action_down_dims)
        if self.chunk_size % downsampling_factor != 0:
            raise ValueError(
                "The chunk size should be an integer multiple of the diffusion downsampling factor. "
                f"Got {self.chunk_size=} and {self.action_down_dims=}."
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if self.robot_state_feature is None:
            raise ValueError("You must provide `observation.state` for the diffusion action branch.")
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")
        # Deprecated crop fields are accepted for checkpoint compatibility, but the action and WM
        # branches now both consume full-frame shared ResNet feature maps.

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1)) + [self.chunk_size]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
