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
"""ACT Simple + AWM head with an optional diffusion action branch."""

from collections import deque
from copy import deepcopy
from itertools import chain
from types import SimpleNamespace

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act_simple.modeling_act_simple import (
    ACTDecoder,
    ACTEncoder,
    ACTLearnedPositionEmbedding2d,
    get_activation_fn,
)
from lerobot.policies.act_simple_with_awm_head.configuration_act_simple_with_awm_head import (
    ACTSimpleWithAWMHeadConfig,
)
from lerobot.policies.act_simple_with_awm_head.planning import make_planner
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    SpatialSoftmax,
    _make_noise_scheduler,
    _replace_submodules,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_output_shape, populate_queues
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


# ---------------------------------------------------------------------------
# Helpers (shared with AWM)
# ---------------------------------------------------------------------------

class ResBlock2d(nn.Module):
    """Conv2d residual block: two 3×3 convs with a skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(x + self.block(x))


class WMImageDecoder(nn.Module):
    """Debug image decoder: (S_img, B, dim_model) -> (B, C, H, W)."""

    def __init__(self, dim_model: int, image_shape: tuple[int, int, int], replace_final_stride_with_dilation: bool = False):
        super().__init__()
        C, H, W = image_shape
        stride = 16 if replace_final_stride_with_dilation else 32
        h0, w0 = H // stride, W // stride
        base_ch = 128

        self.h0 = h0
        self.w0 = w0
        self.chan_proj = nn.Conv2d(dim_model, base_ch, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_ch, 64, 4, stride=2, padding=1), nn.ReLU(),
            ResBlock2d(64),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            ResBlock2d(32),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            ResBlock2d(16),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1), nn.ReLU(),
            ResBlock2d(8),
            nn.ConvTranspose2d(8, C, 4, stride=2, padding=1),
        )

    def forward(self, z: Tensor) -> Tensor:
        S_img, B, D = z.shape
        x = z.permute(1, 2, 0).view(B, D, self.h0, self.w0)
        x = self.chan_proj(x)
        return self.decoder(x)


def _n_encoder_tokens(config: ACTSimpleWithAWMHeadConfig) -> int:
    """Compute the total number of encoder output tokens S from config."""
    n = sum([bool(config.robot_state_feature), bool(config.env_state_feature)])
    if config.image_features:
        for feat in config.image_features.values():
            C, H, W = feat.shape
            stride = 16 if config.replace_final_stride_with_dilation else 32
            n += (H // stride) * (W // stride)
    return n


def _slice_obs_batch(batch: dict[str, Tensor], idx: int) -> dict[str, Tensor]:
    """Return a batch dict with observation tensors sliced to a single temporal index."""
    result = {}
    for key, val in batch.items():
        if key.startswith("observation.") and isinstance(val, Tensor) and val.ndim >= 2:
            result[key] = val[:, idx]
        else:
            result[key] = val
    return result


def _compute_wm_loss(z_pred: Tensor, z_target: Tensor, valid_wm: Tensor) -> Tensor:
    """Cosine similarity world-model loss, masked at episode boundaries."""
    valid_wm_f = valid_wm.to(dtype=z_pred.dtype)
    valid_count = valid_wm_f.sum()
    cos_sim = F.cosine_similarity(z_pred, z_target, dim=-1).mean(dim=0)  # (B,)
    wm_loss = 1 - (cos_sim * valid_wm_f).sum() / valid_count.clamp(min=1.0)
    return wm_loss


def _compute_image_reconstruction_metrics(
    pred: Tensor, target: Tensor, prefix: str, valid_mask: Tensor | None = None,
) -> dict[str, float]:
    if valid_mask is not None:
        if not valid_mask.any():
            return {}
        pred = pred[valid_mask]
        target = target[valid_mask]
    mse = F.mse_loss(pred, target)
    psnr = -10.0 * torch.log10(mse.clamp(min=1e-8))
    return {f"{prefix}/mse": float(mse.item()), f"{prefix}/psnr": float(psnr.item())}


# ---------------------------------------------------------------------------
# WM Decoder (non-causal, bidirectional — reused from AWM)
# ---------------------------------------------------------------------------

class WMDecoder(nn.Module):
    """Stack of WMDecoderLayer modules. Non-causal (bidirectional self-attention)."""

    def __init__(self, config: ACTSimpleWithAWMHeadConfig):
        super().__init__()
        self.layers = nn.ModuleList([WMDecoderLayer(config) for _ in range(config.n_wm_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        cross_kv: Tensor,
        cross_pos: Tensor,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, cross_kv, cross_pos)
        return self.norm(x)


class WMDecoderLayer(nn.Module):
    """Single WM decoder layer: bidirectional self-attention + cross-attention + FFN."""

    def __init__(self, config: ACTSimpleWithAWMHeadConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(
            config.dim_model, config.n_heads, dropout=config.dropout,
        )

        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def _add_pos(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        cross_kv: Tensor,
        cross_pos: Tensor,
    ) -> Tensor:
        # Bidirectional self-attention (no causal mask).
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x
        x = self.self_attn(q, k, value=x, need_weights=False)[0]
        x = skip + self.dropout1(x)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        # Cross-attention on encoder tokens.
        x = self.multihead_attn(
            query=x,
            key=self._add_pos(cross_kv, cross_pos),
            value=cross_kv,
            need_weights=False,
        )[0]
        x = skip + self.dropout2(x)

        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x

        # Feed-forward.
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)

        return x


class SharedResNet18Backbone(nn.Module):
    """Shared ResNet-18 trunk used by both the action and WM branches."""

    def __init__(self, config: ACTSimpleWithAWMHeadConfig):
        super().__init__()
        backbone_model = getattr(torchvision.models, config.action_vision_backbone)(
            weights=config.action_pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.action_use_group_norm:
            if config.action_pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained action backbone without invalidating the weights."
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )
        self.out_channels = backbone_model.fc.in_features

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)


class ActionVisionHead(nn.Module):
    """Diffusion-policy visual head on top of a shared ResNet feature map."""

    def __init__(
        self,
        config: ACTSimpleWithAWMHeadConfig,
        feature_map_shape: tuple[int, int, int],
        shared_backbone: SharedResNet18Backbone | None = None,
    ):
        super().__init__()
        self.config = config
        # Kept as a module alias for older checkpoints; forward consumes precomputed feature maps.
        self.shared_backbone = shared_backbone
        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.action_spatial_softmax_num_keypoints)
        self.feature_dim = config.action_spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(self.feature_dim, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, feature_map: Tensor) -> Tensor:
        x = torch.flatten(self.pool(feature_map), start_dim=1)
        x = self.relu(self.out(x))
        return x


class ActionDiffusionModel(nn.Module):
    """Diffusion-policy action generator conditioned on shared visual features."""

    def __init__(
        self,
        config: ACTSimpleWithAWMHeadConfig,
        feature_map_shape: tuple[int, int, int] | None,
        shared_backbone: SharedResNet18Backbone | None = None,
    ):
        super().__init__()
        self.config = config

        global_cond_dim = config.robot_state_feature.shape[0]
        if config.image_features:
            num_images = len(config.image_features)
            if feature_map_shape is None:
                raise ValueError("Feature map shape is required when image features are configured.")
            if config.action_use_separate_rgb_encoder_per_camera:
                encoders = [
                    ActionVisionHead(config, feature_map_shape, shared_backbone) for _ in range(num_images)
                ]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = ActionVisionHead(config, feature_map_shape, shared_backbone)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if config.env_state_feature:
            global_cond_dim += config.env_state_feature.shape[0]

        self.unet = DiffusionConditionalUnet1d(
            self._make_unet_config(),
            global_cond_dim=global_cond_dim * config.n_obs_steps,
        )
        self.noise_scheduler = _make_noise_scheduler(
            config.action_noise_scheduler_type,
            num_train_timesteps=config.action_num_train_timesteps,
            beta_start=config.action_beta_start,
            beta_end=config.action_beta_end,
            beta_schedule=config.action_beta_schedule,
            clip_sample=config.action_clip_sample,
            clip_sample_range=config.action_clip_sample_range,
            prediction_type=config.action_prediction_type,
        )
        self.num_inference_steps = (
            config.action_num_inference_steps or self.noise_scheduler.config.num_train_timesteps
        )

    def _make_unet_config(self) -> SimpleNamespace:
        return SimpleNamespace(
            action_feature=self.config.action_feature,
            down_dims=self.config.action_down_dims,
            kernel_size=self.config.action_kernel_size,
            n_groups=self.config.action_n_groups,
            diffusion_step_embed_dim=self.config.action_diffusion_step_embed_dim,
            use_film_scale_modulation=self.config.action_use_film_scale_modulation,
        )

    def _prepare_global_conditioning(
        self,
        batch: dict[str, Tensor],
        image_feature_maps: Tensor | None = None,
    ) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]

        if self.config.image_features:
            if image_feature_maps is None:
                raise ValueError("Precomputed shared ResNet feature maps are required for image conditioning.")
            if self.config.action_use_separate_rgb_encoder_per_camera:
                features_per_camera = einops.rearrange(image_feature_maps, "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(features)
                        for encoder, features in zip(self.rgb_encoder, features_per_camera, strict=True)
                    ]
                )
                img_features = einops.rearrange(
                    img_features_list,
                    "(n b s) ... -> b s (n ...)",
                    b=batch_size,
                    s=n_obs_steps,
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(image_feature_maps, "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features,
                    "(b s n) ... -> b s (n ...)",
                    b=batch_size,
                    s=n_obs_steps,
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def compute_loss(self, batch: dict[str, Tensor], image_feature_maps: Tensor | None = None) -> Tensor:
        global_cond = self._prepare_global_conditioning(batch, image_feature_maps)
        trajectory = batch[ACTION]
        eps = torch.randn_like(trajectory)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        if self.config.action_prediction_type == "epsilon":
            target = eps
        else:
            target = trajectory

        loss = F.mse_loss(pred, target, reduction="none")
        if self.config.action_do_mask_loss_for_padding:
            loss = loss * (~batch["action_is_pad"]).unsqueeze(-1)
        return loss.mean()

    @torch.no_grad()
    def sample(
        self,
        batch: dict[str, Tensor],
        image_feature_maps: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        batch_size = batch[OBS_STATE].shape[0]
        global_cond = self._prepare_global_conditioning(batch, image_feature_maps)
        sample = (
            noise
            if noise is not None
            else torch.randn(
                batch_size,
                self.config.chunk_size,
                self.config.action_feature.shape[0],
                device=global_cond.device,
                dtype=global_cond.dtype,
            )
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            timesteps = torch.full((batch_size,), t, device=sample.device, dtype=torch.long)
            pred_noise = self.unet(sample, timesteps, global_cond=global_cond)
            sample = self.noise_scheduler.step(pred_noise, t, sample).prev_sample
        return sample


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class ACTSimpleWithAWMHeadPolicy(PreTrainedPolicy):
    """ACT Simple + AWM head policy with an optional diffusion action branch."""

    config_class = ACTSimpleWithAWMHeadConfig
    name = "act_simple_with_awm_head"

    def __init__(self, config: ACTSimpleWithAWMHeadConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACTSimpleWithAWMHead(config)
        self._train_step = 0
        self._ema_step = 0
        self._pending_ema_momentum = None

        if config.use_planning:
            self._planner = make_planner(config.planning)
        else:
            self._planner = None

        self.reset()

    def get_optim_params(self) -> dict:
        if self.config.use_diffusion_action_head:
            return self.model.parameters()
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.use_diffusion_action_head:
            self._queues = {
                OBS_STATE: deque(maxlen=self.config.n_obs_steps),
                ACTION: deque(maxlen=self.config.n_action_steps),
            }
            if self.config.image_features:
                self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
            if self.config.env_state_feature:
                self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)
        else:
            self._queues = None
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self._z_goal: Tensor | None = None            # (S, B, dim_model)
        self._encoder_pos_cache: Tensor | None = None  # (S, 1, dim_model)

    def update(self):
        if self._pending_ema_momentum is not None:
            self.model.update_ema(self._pending_ema_momentum)
            self._ema_step += 1
            self._pending_ema_momentum = None

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        if self.config.use_diffusion_action_head:
            if ACTION in batch:
                batch = dict(batch)
                batch.pop(ACTION)
            if self.config.image_features:
                batch = dict(batch)
                batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
            self._queues = populate_queues(self._queues, batch)

        if len(self._action_queue) == 0:
            if self.config.use_planning and self._z_goal is not None:
                actions = self._plan_action_chunk(batch)[:, : self.config.n_action_steps]
            else:
                actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if self.config.use_diffusion_action_head:
            queued_batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in self._queues if k != ACTION}
            return self.model.predict_action(queued_batch)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return self.model.predict_action(batch)

    def _prepare_wm_observation(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Format a single-step observation batch for the WM encoder."""
        wm_batch = dict(batch)
        if self.config.image_features:
            wm_batch[OBS_IMAGES] = [wm_batch[key] for key in self.config.image_features]
        return wm_batch

    @torch.no_grad()
    def set_goal(self, batch: dict[str, Tensor]) -> None:
        """Encode a goal observation and store its latent tokens for planning.

        Args:
            batch: Preprocessed observation dict (same format as ``select_action``).
                Should correspond to a single time-step (n_obs_steps=1).
        """
        self.eval()
        batch = self._prepare_wm_observation(batch)
        _, _, encoder_pos, encoder_in = self.model._encode(batch)
        # encoder_in: (S, B, dim_model) — store as-is, index per batch element during planning.
        self._z_goal = encoder_in
        self._encoder_pos_cache = encoder_pos

    def _plan_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Use MPPI planning to produce an optimized action chunk.

        Uses BC action chunk as warm start and minimizes
        ``1 - cosine_similarity(z_goal, z_pred)`` over the full chunk.

        Args:
            batch: Preprocessed observation dict.

        Returns:
            (B, chunk_size, action_dim) tensor of planned actions.
        """
        if self._planner is None:
            return self.predict_action_chunk(batch)
        if self._z_goal is None:
            raise RuntimeError("Planning requested before calling set_goal().")

        if self.config.use_diffusion_action_head:
            queued_batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in self._queues if k != ACTION}
            initial_actions = self.model.predict_action(queued_batch)

            wm_batch = self._prepare_wm_observation(batch)
            _, _, encoder_pos, encoder_in = self.model._encode(wm_batch)
            encoder_pos = self._encoder_pos_cache if self._encoder_pos_cache is not None else encoder_pos

            batch_size = initial_actions.shape[0]
            goal_batch_size = self._z_goal.shape[1]
            planned_actions = []
            for batch_idx in range(batch_size):
                goal_idx = batch_idx if goal_batch_size > 1 else 0
                planned_actions.append(
                    self._planner.optimize(
                        z_start=encoder_in[:, batch_idx].detach(),
                        encoder_pos=encoder_pos,
                        z_goal=self._z_goal[:, goal_idx].detach(),
                        initial_actions=initial_actions[batch_idx].detach(),
                        wm_predict_fn=self.model.run_wm_decoder,
                    )
                )
            return torch.stack(planned_actions, dim=0)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        initial_actions = self.model.predict_action(batch)
        _, _, encoder_pos, encoder_in = self.model._encode(batch)
        encoder_pos = self._encoder_pos_cache if self._encoder_pos_cache is not None else encoder_pos

        batch_size = initial_actions.shape[0]
        goal_batch_size = self._z_goal.shape[1]
        planned_actions = []
        for batch_idx in range(batch_size):
            goal_idx = batch_idx if goal_batch_size > 1 else 0
            planned_actions.append(
                self._planner.optimize(
                    z_start=encoder_in[:, batch_idx].detach(),
                    encoder_pos=encoder_pos,
                    z_goal=self._z_goal[:, goal_idx].detach(),
                    initial_actions=initial_actions[batch_idx].detach(),
                    wm_predict_fn=self.model.run_wm_decoder,
                )
            )

        return torch.stack(planned_actions, dim=0)

    @torch.no_grad()
    def visualize(self, batch: dict[str, Tensor], n_pairs: int = 12) -> dict[str, Tensor] | None:
        """Generate WM image reconstruction pairs for debugging.

        Returns a dict with keys "curr" and "next", each (N, C, H, 2W) float in [0, 1]:
          curr: GT (left) | decoded current encoder tokens (right)
          next: GT (left) | decoded WM future prediction (right)

        Returns None when no image features are configured.
        """
        if not self.config.image_features or not hasattr(self.model, "wm_image_decoder"):
            return None

        was_training = self.training
        self.eval()

        n = min(n_pairs, batch[ACTION].shape[0])

        def _prep(raw_batch: dict, idx: int) -> dict:
            sliced = _slice_obs_batch(raw_batch, idx)
            d = {k: v[:n] if isinstance(v, Tensor) else v for k, v in sliced.items()}
            d = dict(d)
            d[OBS_IMAGES] = [d[k][:n] for k in self.config.image_features]
            return d

        curr_obs_idx = self.config.n_obs_steps - 1
        next_obs_idx = self.config.n_obs_steps
        curr_batch = _prep(batch, curr_obs_idx)
        next_batch = _prep(batch, next_obs_idx)

        n_1d = self.model.n_1d_tokens
        s_img = self.model.img_tokens_per_cam

        # Encode current obs.
        batch_size, encoder_out, encoder_pos, curr_encoder_in = self.model._encode(curr_batch)

        # Current observation: decode from pre-transformer encoder input tokens.
        curr_img_z = curr_encoder_in[n_1d : n_1d + s_img]
        if self.config.normalize_wm_representations:
            curr_img_z = F.normalize(curr_img_z, dim=-1)
        decoded_curr = self.model.wm_image_decoder(curr_img_z)
        gt_curr = curr_batch[OBS_IMAGES][0]

        # Run WM decoder to get future state prediction.
        actions = batch[ACTION][:n]
        T = actions.shape[1]
        action_embeds = self.model.wm_action_proj(actions).transpose(0, 1)
        wm_action_pos = self.model.wm_action_pos_embed.weight[:T].unsqueeze(1)
        S = self.model.n_encoder_tokens
        query_pos = self.model.wm_query_pos_embed.weight.unsqueeze(1)
        queries = (self.model.wm_query_tokens + query_pos).expand(-1, batch_size, -1)
        wm_in = torch.cat([queries, action_embeds + wm_action_pos], dim=0)
        wm_encoder_in = curr_encoder_in.detach() if self.config.detach_encoder_from_wm else curr_encoder_in
        wm_cross_kv = self.model.wm_cross_attn_proj(wm_encoder_in)
        wm_cross_pos = encoder_pos
        wm_out = self.model.wm_decoder(wm_in, wm_cross_kv, wm_cross_pos)
        z_pred = self.model.wm_proj_head(wm_out[:S])

        # Future observation: decode from WM predicted tokens.
        next_img_z = z_pred[n_1d : n_1d + s_img]
        if self.config.normalize_wm_representations:
            next_img_z = F.normalize(next_img_z, dim=-1)
        decoded_next = self.model.wm_image_decoder(next_img_z)
        gt_next = _prep(batch, next_obs_idx)[OBS_IMAGES][0]

        def _to_01(t: Tensor) -> Tensor:
            B = t.shape[0]
            t_flat = t.view(B, -1)
            lo = t_flat.min(dim=1).values.view(B, 1, 1, 1)
            hi = t_flat.max(dim=1).values.view(B, 1, 1, 1)
            return ((t - lo) / (hi - lo + 1e-8)).clamp(0, 1)

        if was_training:
            self.train()

        curr = torch.cat([_to_01(gt_curr), _to_01(decoded_curr)], dim=3)
        next_ = torch.cat([_to_01(gt_next), _to_01(decoded_next)], dim=3)
        return {"curr": curr.cpu(), "next": next_.cpu()}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass: action loss + WM loss."""
        if self.config.use_diffusion_action_head:
            curr_obs_idx = self.config.n_obs_steps - 1
            next_obs_idx = self.config.n_obs_steps

            action_batch = _slice_obs_batch(batch, slice(0, self.config.n_obs_steps))
            curr_batch = _slice_obs_batch(batch, curr_obs_idx)
            next_batch = _slice_obs_batch(batch, next_obs_idx)

            if self.config.image_features:
                action_batch = dict(action_batch)
                action_batch[OBS_IMAGES] = torch.stack(
                    [action_batch[key] for key in self.config.image_features],
                    dim=2,
                )
                curr_batch = dict(curr_batch)
                curr_batch[OBS_IMAGES] = [curr_batch[key] for key in self.config.image_features]
                next_batch = dict(next_batch)
                next_batch[OBS_IMAGES] = [next_batch[key] for key in self.config.image_features]
            action_loss, wm_tensors = self.model(action_batch, curr_batch, next_batch)
        else:
            curr_obs_idx = 0
            next_obs_idx = 1
            curr_batch = _slice_obs_batch(batch, curr_obs_idx)
            next_batch = _slice_obs_batch(batch, next_obs_idx)

            if self.config.image_features:
                curr_batch = dict(curr_batch)
                curr_batch[OBS_IMAGES] = [curr_batch[key] for key in self.config.image_features]
                next_batch = dict(next_batch)
                next_batch[OBS_IMAGES] = [next_batch[key] for key in self.config.image_features]
            bc_loss_mask = batch.get("bc_loss_mask")
            if bc_loss_mask is not None:
                curr_batch = dict(curr_batch)
                curr_batch["bc_loss_mask"] = bc_loss_mask
            action_loss, wm_tensors = self.model(curr_batch, None, next_batch)

        # Episode-boundary mask: True where t+H is beyond the episode end.
        next_obs_is_pad = batch.get(
            "observation.state_is_pad",
            batch.get("observation.environment_state_is_pad"),
        )

        # WM loss.
        z_pred, z_target, decoded_curr, gt_curr_img = wm_tensors
        valid_wm = ~next_obs_is_pad[:, next_obs_idx]  # (B,)
        wm_loss = _compute_wm_loss(z_pred, z_target, valid_wm)

        if self.config.wm_warmup_steps > 0 and self.training:
            warmup_frac = min(self._train_step / self.config.wm_warmup_steps, 1.0)
            effective_wm_weight = self.config.wm_loss_weight * warmup_frac
        else:
            effective_wm_weight = self.config.wm_loss_weight

        loss = action_loss + effective_wm_weight * wm_loss
        info = {
            "action_loss": action_loss.item(),
            "wm_loss": wm_loss.item(),
            "effective_wm_loss_weight": effective_wm_weight,
            "z_target_norm": z_target.norm(dim=-1).mean().item(),
            "z_pred_norm": z_pred.norm(dim=-1).mean().item(),
            "z_pred_batch_std": z_pred.std(dim=1).mean().item(),
            "z_target_batch_std": z_target.std(dim=1).mean().item(),
        }

        with torch.no_grad():
            info["wm_cosine_sim"] = F.cosine_similarity(z_pred, z_target, dim=-1).mean().item()
            info["z_pred_target_norm_ratio"] = (
                z_pred.norm(dim=-1).mean() / z_target.norm(dim=-1).mean().clamp(min=1e-8)
            ).item()

        # Image reconstruction loss on current obs.
        if decoded_curr is not None and gt_curr_img is not None:
            decoder_loss = F.mse_loss(decoded_curr, gt_curr_img)
            loss = loss + self.config.decoder_loss_weight * decoder_loss
            info["decoder_loss"] = decoder_loss.item()

            with torch.no_grad():
                info.update(
                    _compute_image_reconstruction_metrics(
                        decoded_curr.detach(), gt_curr_img.detach(), prefix="wm_curr",
                    )
                )

                next_img_z = z_pred[
                    self.model.n_1d_tokens : self.model.n_1d_tokens + self.model.img_tokens_per_cam
                ]
                decoded_next = self.model.wm_image_decoder(next_img_z.detach())
                gt_next_img = next_batch[OBS_IMAGES][0].detach()
                info.update(
                    _compute_image_reconstruction_metrics(
                        decoded_next, gt_next_img, prefix="wm_next", valid_mask=valid_wm,
                    )
                )

        if self.config.use_ema_target and self.training:
            t = min(self._ema_step / max(self.config.ema_anneal_steps, 1), 1.0)
            momentum = self.config.ema_momentum + t * (
                self.config.ema_momentum_end - self.config.ema_momentum
            )
            self._pending_ema_momentum = momentum
            info["ema_momentum"] = momentum

        if self.training:
            self._train_step += 1

        info["loss"] = loss.item()
        return loss, info


# ---------------------------------------------------------------------------
# Core network
# ---------------------------------------------------------------------------

class ACTSimpleWithAWMHead(nn.Module):
    """Core network: main-branch ACT Simple path or optional diffusion path."""

    def __init__(self, config: ACTSimpleWithAWMHeadConfig):
        super().__init__()
        self.config = config

        feature_map_shape = None
        if config.image_features:
            if config.use_diffusion_action_head:
                self.shared_backbone = SharedResNet18Backbone(config)
                images_shape = next(iter(config.image_features.values())).shape
                dummy_shape = (1, *images_shape)
                feature_map_shape = get_output_shape(self.shared_backbone, dummy_shape)[1:]
            else:
                backbone_model = getattr(torchvision.models, config.vision_backbone)(
                    replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                    weights=config.pretrained_backbone_weights,
                    norm_layer=FrozenBatchNorm2d,
                )
                self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        if config.use_diffusion_action_head:
            self.action_diffusion = ActionDiffusionModel(
                config,
                feature_map_shape,
                getattr(self, "shared_backbone", None),
            )

        # ------------------------------------------------------------------
        # Transformer encoder for the WM branch
        # ------------------------------------------------------------------
        self.encoder = ACTEncoder(config)

        if not config.use_diffusion_action_head:
            self.action_decoder = ACTDecoder(config)

        # ------------------------------------------------------------------
        # Encoder input projections
        # ------------------------------------------------------------------
        if config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.robot_state_feature.shape[0], config.dim_model
            )
        if config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                config.env_state_feature.shape[0], config.dim_model
            )
        if config.image_features:
            backbone_out_channels = self.shared_backbone.out_channels if config.use_diffusion_action_head else backbone_model.fc.in_features
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_out_channels, config.dim_model, kernel_size=1
            )

        # ------------------------------------------------------------------
        # Encoder positional embeddings
        # ------------------------------------------------------------------
        n_1d_tokens = sum([bool(config.robot_state_feature), bool(config.env_state_feature)])
        self.n_1d_tokens = n_1d_tokens
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if config.image_features:
            C, H, W = config.image_features["observation.image"].shape
            self.encoder_cam_feat_pos_embed = ACTLearnedPositionEmbedding2d(H, W, config.dim_model)

        action_dim = config.action_feature.shape[0]
        if not config.use_diffusion_action_head:
            self.action_decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
            self.action_head = nn.Linear(config.dim_model, action_dim)

        # ------------------------------------------------------------------
        # World model decoder
        # ------------------------------------------------------------------
        self.wm_decoder = WMDecoder(config)

        # WM action conditioning: project continuous action chunks to dim_model.
        self.wm_action_proj = nn.Linear(action_dim, config.dim_model)

        # Positional embeddings for WM action tokens.
        self.wm_action_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # S learnable query tokens — one per encoder output token.
        n_enc = _n_encoder_tokens(config)
        self.n_encoder_tokens = n_enc
        self.wm_query_tokens = nn.Parameter(torch.zeros(n_enc, 1, config.dim_model))
        nn.init.trunc_normal_(self.wm_query_tokens, std=0.02)
        self.wm_query_pos_embed = nn.Embedding(n_enc, config.dim_model)

        if config.image_features:
            stride = 16 if config.replace_final_stride_with_dilation else 32
            C, H, W = next(iter(config.image_features.values())).shape
            self.img_tokens_per_cam = (H // stride) * (W // stride)

        # WM projection head: maps query outputs to predicted next-state latent tokens.
        self.wm_proj_head = nn.Sequential(
            nn.Linear(config.dim_model, config.dim_model),
            nn.ReLU(),
            nn.Linear(config.dim_model, config.dim_model),
        )

        # WM cross-attention projection for encoder input tokens.
        self.wm_cross_attn_proj = nn.Sequential(
            nn.Linear(config.dim_model, config.dim_model),
            nn.ReLU(),
            nn.Linear(config.dim_model, config.dim_model),
        )

        # ------------------------------------------------------------------
        # Image decoder (debug only)
        # ------------------------------------------------------------------
        if config.image_features:
            first_feat = next(iter(config.image_features.values()))
            self.wm_image_decoder = WMImageDecoder(
                config.dim_model, tuple(first_feat.shape), config.replace_final_stride_with_dilation
            )

        self._reset_parameters()

        if config.use_ema_target:
            self._build_ema_encoder()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.use_ema_target:
            self._set_ema_eval_mode()
        return self

    def _reset_parameters(self):
        modules = [self.encoder.parameters(), self.wm_decoder.parameters(), self.wm_proj_head.parameters(), self.wm_cross_attn_proj.parameters(), self.wm_query_pos_embed.parameters()]
        if self.config.use_diffusion_action_head:
            modules.insert(0, self.action_diffusion.parameters())
        else:
            modules.insert(0, self.action_decoder.parameters())
        if hasattr(self, "wm_image_decoder"):
            modules.append(self.wm_image_decoder.parameters())
        for p in chain(*modules):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_ema_encoder(self):
        if hasattr(self, "shared_backbone"):
            self.ema_shared_backbone = deepcopy(self.shared_backbone)
        if hasattr(self, "backbone"):
            self.ema_backbone = deepcopy(self.backbone)
        self.ema_encoder = deepcopy(self.encoder)
        if hasattr(self, "encoder_robot_state_input_proj"):
            self.ema_encoder_robot_state_input_proj = deepcopy(self.encoder_robot_state_input_proj)
        if hasattr(self, "encoder_env_state_input_proj"):
            self.ema_encoder_env_state_input_proj = deepcopy(self.encoder_env_state_input_proj)
        if hasattr(self, "encoder_img_feat_input_proj"):
            self.ema_encoder_img_feat_input_proj = deepcopy(self.encoder_img_feat_input_proj)
        self.ema_encoder_1d_feature_pos_embed = deepcopy(self.encoder_1d_feature_pos_embed)
        if hasattr(self, "encoder_cam_feat_pos_embed"):
            self.ema_encoder_cam_feat_pos_embed = deepcopy(self.encoder_cam_feat_pos_embed)

        for name, param in self.named_parameters():
            if name.startswith("ema_"):
                param.requires_grad = False
        self._set_ema_eval_mode()

    def _set_ema_eval_mode(self):
        for name, module in self.named_children():
            if name.startswith("ema_"):
                module.eval()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_stacked_image_features(self, images: Tensor) -> Tensor:
        """Run all observation images through the shared ResNet in one flattened pass."""
        batch_size, n_obs_steps, n_images = images.shape[:3]
        flat_images = einops.rearrange(images, "b s n c h w -> (b s n) c h w")
        flat_features = self.shared_backbone(flat_images)
        return einops.rearrange(
            flat_features,
            "(b s n) c h w -> b s n c h w",
            b=batch_size,
            s=n_obs_steps,
            n=n_images,
        )

    def _encode_image_list_features(self, images: list[Tensor]) -> list[Tensor]:
        """Run a list of camera images through the shared ResNet with one concatenated call."""
        if len(images) == 1:
            return [self.shared_backbone(images[0])]
        batch_size = images[0].shape[0]
        flat_features = self.shared_backbone(torch.cat(images, dim=0))
        return list(flat_features.split(batch_size, dim=0))

    def _encode(
        self,
        batch: dict[str, Tensor],
        image_feature_maps: list[Tensor] | None = None,
    ) -> tuple[int, Tensor, Tensor, Tensor]:
        """Run the encoder.

        Returns:
            batch_size, encoder_out (S, B, dim_model), encoder_pos (S, 1, dim_model),
            encoder_in (S, B, dim_model) — pre-transformer input tokens.
        """
        if OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES][0].shape[0]
        elif OBS_ENV_STATE in batch:
            batch_size = batch[OBS_ENV_STATE].shape[0]
        else:
            batch_size = batch[OBS_STATE].shape[0]

        encoder_in_tokens: list[Tensor] = []
        encoder_in_pos_embed: list[Tensor] = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            if self.config.use_diffusion_action_head:
                if image_feature_maps is None:
                    image_feature_maps = self._encode_image_list_features(batch[OBS_IMAGES])
                feature_iter = image_feature_maps
            else:
                feature_iter = [self.backbone(img)["feature_map"] for img in batch[OBS_IMAGES]]
            for cam_features in feature_iter:
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        return batch_size, encoder_out, encoder_in_pos_embed, encoder_in_tokens

    @torch.no_grad()
    def _encode_ema(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode observations with EMA modules and return pre-transformer encoder tokens."""
        encoder_in_tokens: list[Tensor] = []
        encoder_in_pos_embed: list[Tensor] = list(self.ema_encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.ema_encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.ema_encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                if self.config.use_diffusion_action_head:
                    cam_features = self.ema_shared_backbone(img)
                else:
                    cam_features = self.ema_backbone(img)["feature_map"]
                cam_pos_embed = self.ema_encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.ema_encoder_img_feat_input_proj(cam_features)

                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)

        _ = self.ema_encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        return encoder_in_tokens

    @torch.no_grad()
    def update_ema(self, momentum: float):
        if not self.config.use_ema_target:
            return

        ema_pairs = []
        if hasattr(self, "ema_shared_backbone"):
            ema_pairs.extend(zip(self.shared_backbone.parameters(), self.ema_shared_backbone.parameters()))
        if hasattr(self, "ema_backbone"):
            ema_pairs.extend(zip(self.backbone.parameters(), self.ema_backbone.parameters()))
        ema_pairs.extend(zip(self.encoder.parameters(), self.ema_encoder.parameters()))
        if hasattr(self, "ema_encoder_robot_state_input_proj"):
            ema_pairs.extend(
                zip(self.encoder_robot_state_input_proj.parameters(),
                    self.ema_encoder_robot_state_input_proj.parameters())
            )
        if hasattr(self, "ema_encoder_env_state_input_proj"):
            ema_pairs.extend(
                zip(self.encoder_env_state_input_proj.parameters(),
                    self.ema_encoder_env_state_input_proj.parameters())
            )
        if hasattr(self, "ema_encoder_img_feat_input_proj"):
            ema_pairs.extend(
                zip(self.encoder_img_feat_input_proj.parameters(),
                    self.ema_encoder_img_feat_input_proj.parameters())
            )
        ema_pairs.extend(
            zip(self.encoder_1d_feature_pos_embed.parameters(),
                self.ema_encoder_1d_feature_pos_embed.parameters())
        )
        if hasattr(self, "ema_encoder_cam_feat_pos_embed"):
            ema_pairs.extend(
                zip(self.encoder_cam_feat_pos_embed.parameters(),
                    self.ema_encoder_cam_feat_pos_embed.parameters())
            )

        for online_p, ema_p in ema_pairs:
            ema_p.data.mul_(momentum).add_(online_p.data, alpha=1.0 - momentum)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(
        self,
        action_batch: dict[str, Tensor],
        curr_batch: dict[str, Tensor],
        next_batch: dict[str, Tensor],
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor | None, Tensor | None]]:
        """Training forward: action loss + WM loss."""
        if self.config.use_diffusion_action_head:
            action_feature_maps = None
            curr_feature_maps = None
            if self.config.image_features:
                action_feature_maps = self._encode_stacked_image_features(action_batch[OBS_IMAGES])
                curr_feature_maps = [
                    action_feature_maps[:, self.config.n_obs_steps - 1, cam_idx]
                    for cam_idx in range(action_feature_maps.shape[2])
                ]
            action_loss = self.action_diffusion.compute_loss(action_batch, action_feature_maps)
            batch_size, _, encoder_pos, encoder_in = self._encode(curr_batch, curr_feature_maps)
            actions = action_batch[ACTION]
        else:
            batch_size, encoder_out, encoder_pos, encoder_in = self._encode(action_batch)
            decoder_in = torch.zeros(
                (self.config.chunk_size, batch_size, self.config.dim_model),
                dtype=encoder_pos.dtype,
                device=encoder_pos.device,
            )
            decoder_out = self.action_decoder(
                decoder_in,
                encoder_out,
                encoder_pos_embed=encoder_pos,
                decoder_pos_embed=self.action_decoder_pos_embed.weight.unsqueeze(1),
            )
            decoder_out = decoder_out.transpose(0, 1)
            actions = self.action_head(decoder_out)
            action_loss_unreduced = (
                F.l1_loss(action_batch[ACTION], actions, reduction="none")
                * ~action_batch["action_is_pad"].unsqueeze(-1)
            )
            bc_loss_mask = action_batch.get("bc_loss_mask")
            if bc_loss_mask is not None:
                action_loss_unreduced = action_loss_unreduced * bc_loss_mask.view(-1, 1, 1)
            action_loss = action_loss_unreduced.mean()

        # === World model decoder ===
        # Target: encoder input tokens of the next observation (pre-transformer, stop-gradient).
        if self.config.use_ema_target:
            z_target = self._encode_ema(next_batch)
        else:
            _, _, _, next_encoder_in = self._encode(next_batch)
            z_target = next_encoder_in.detach()  # (S, B, dim_model)
        if self.config.normalize_wm_representations:
            z_target = F.normalize(z_target, dim=-1)

        # WM input: [S query tokens, T continuous action tokens].
        T = actions.shape[1]
        action_embeds = self.wm_action_proj(actions).transpose(0, 1)  # (T, B, dim_model)
        wm_action_pos = self.wm_action_pos_embed.weight[:T].unsqueeze(1)  # (T, 1, dim_model)
        query_pos = self.wm_query_pos_embed.weight.unsqueeze(1)  # (S, 1, dim_model)
        queries = (self.wm_query_tokens + query_pos).expand(-1, batch_size, -1)  # (S, B, dim_model)
        wm_in = torch.cat([queries, action_embeds + wm_action_pos], dim=0)  # (S+T, B, dim_model)

        # WM cross-attends to encoder INPUT tokens (pre-transformer) — same space as target.
        S = self.n_encoder_tokens
        wm_encoder_in = encoder_in.detach() if self.config.detach_encoder_from_wm else encoder_in
        wm_cross_kv = self.wm_cross_attn_proj(wm_encoder_in)  # (S, B, dim_model)
        wm_cross_pos = encoder_pos  # (S, 1, dim_model)
        wm_out = self.wm_decoder(wm_in, wm_cross_kv, wm_cross_pos)  # (S+T, B, dim_model)
        z_pred = self.wm_proj_head(wm_out[:S])  # (S, B, dim_model)
        if self.config.normalize_wm_representations:
            z_pred = F.normalize(z_pred, dim=-1)

        # Image decoder.
        decoded_curr, gt_curr_img = None, None
        image_batch = curr_batch if self.config.use_diffusion_action_head else action_batch
        if hasattr(self, "wm_image_decoder") and OBS_IMAGES in image_batch:
            curr_img_z = encoder_in[self.n_1d_tokens : self.n_1d_tokens + self.img_tokens_per_cam]
            if self.config.normalize_wm_representations:
                curr_img_z = F.normalize(curr_img_z, dim=-1)
            decoded_curr = self.wm_image_decoder(curr_img_z.detach())
            gt_curr_img = image_batch[OBS_IMAGES][0].detach()

        wm_tensors = (z_pred, z_target, decoded_curr, gt_curr_img)
        return action_loss, wm_tensors

    def predict_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Inference: predict action chunk (no WM needed)."""
        if self.config.use_diffusion_action_head:
            image_feature_maps = None
            if self.config.image_features:
                image_feature_maps = self._encode_stacked_image_features(batch[OBS_IMAGES])
            return self.action_diffusion.sample(batch, image_feature_maps)

        batch_size, encoder_out, encoder_pos, _ = self._encode(batch)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_pos.dtype,
            device=encoder_pos.device,
        )
        decoder_out = self.action_decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_pos,
            decoder_pos_embed=self.action_decoder_pos_embed.weight.unsqueeze(1),
        )
        decoder_out = decoder_out.transpose(0, 1)
        return self.action_head(decoder_out)

    def run_wm_decoder(
        self,
        encoder_in: Tensor,
        encoder_pos: Tensor,
        actions: Tensor,
    ) -> Tensor:
        """Run the world-model decoder to predict next latent tokens.

        Args:
            encoder_in: (S, N, dim_model) — encoder input tokens for N samples.
            encoder_pos: (S, 1, dim_model) — positional embeddings from encoder.
            actions: (T, N, action_dim) — action sequences for N samples.

        Returns:
            z_pred: (S, N, dim_model) — predicted next encoder input tokens.
        """
        S = self.n_encoder_tokens
        T, N, _ = actions.shape

        action_embeds = self.wm_action_proj(actions)  # (T, N, dim_model)
        wm_action_pos = self.wm_action_pos_embed.weight[:T].unsqueeze(1)  # (T, 1, dim_model)
        query_pos = self.wm_query_pos_embed.weight.unsqueeze(1)  # (S, 1, dim_model)
        queries = (self.wm_query_tokens + query_pos).expand(-1, N, -1)  # (S, N, dim_model)
        wm_in = torch.cat([queries, action_embeds + wm_action_pos], dim=0)  # (S+T, N, dim_model)

        wm_cross_kv = self.wm_cross_attn_proj(encoder_in)  # (S, N, dim_model)
        wm_out = self.wm_decoder(wm_in, wm_cross_kv, encoder_pos)  # (S+T, N, dim_model)
        z_pred = self.wm_proj_head(wm_out[:S])  # (S, N, dim_model)

        if self.config.normalize_wm_representations:
            z_pred = F.normalize(z_pred, dim=-1)

        return z_pred
