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
"""AWM Policy — Autoregressive ACT Decoder with Discrete Action Tokens and Online Planning.

Two key differences from ACTSimple:
  1. Autoregressive decoder with causal self-attention: teacher-forcing at training time
     (shifted-right ground-truth token embeddings), step-by-step greedy decoding at inference.
  2. Discrete action tokens: continuous actions are quantised via a tokenizer into discrete tokens.

Online planning (optional):
  - At inference, the BC trajectory from predict_ar() is used as a warm start.
  - An MPPI planner then refines it by rolling proposed trajectories through the world model and
    minimising cosine distance to a cached goal latent.
  - Call AWMPolicy.set_planning_goal(batch) once per episode before select_action().
"""

from __future__ import annotations

from collections import deque
from copy import copy
from itertools import chain
from typing import Callable

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act_simple.modeling_act_simple import (
    ACTEncoder,
    ACTLearnedPositionEmbedding2d,
    get_activation_fn,
)
from lerobot.policies.awm.configuration_awm import AWMConfig
from lerobot.policies.awm.tokenizer_awm import FactoredActionTokenizer
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class WMImageDecoder(nn.Module):
    """Lightweight debug image decoder: (S_img, B, dim_model) → (B, C, H, W).

    Trained on pre-transformer encoder *input* tokens (direct ResNet spatial projections
    with positional embeddings). These have clear per-patch spatial correspondence, so the
    decoder can learn to reconstruct images rather than collapsing to the dataset mean (which
    happens when training on post-attention encoder output tokens whose spatial structure has
    been globally mixed by self-attention).

    At visualisation time, the same decoder is applied to WM-predicted tokens (z_pred) to
    assess whether the world model is predicting spatially coherent future representations.

    Architecture: 1×1 Conv2d channel projection → 5 × stride-2 ConvTranspose2d.

    For 96×96 images (h0=w0=3, S_img=9): ~160K parameters.
    """

    def __init__(
        self,
        dim_model: int,
        image_shape: tuple[int, int, int],
        replace_final_stride_with_dilation: bool = False,
    ):
        super().__init__()
        C, H, W = image_shape
        stride = 16 if replace_final_stride_with_dilation else 32
        h0, w0 = H // stride, W // stride   # ResNet layer4 spatial resolution
        base_ch = 32

        self.h0 = h0
        self.w0 = w0
        self.base_ch = base_ch
        # 1×1 conv projects from dim_model channels to base_ch, preserving spatial layout.
        self.chan_proj = nn.Conv2d(dim_model, base_ch, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_ch, 16, 4, stride=2, padding=1), nn.ReLU(),  # ×2
            nn.ConvTranspose2d(16,       8, 4, stride=2, padding=1), nn.ReLU(),  # ×2
            nn.ConvTranspose2d( 8,       4, 4, stride=2, padding=1), nn.ReLU(),  # ×2
            nn.ConvTranspose2d( 4,       2, 4, stride=2, padding=1), nn.ReLU(),  # ×2
            nn.ConvTranspose2d( 2,       C, 4, stride=2, padding=1),             # ×2 → H×W
        )

    def forward(self, z: Tensor) -> Tensor:
        # z: (S_img, B, dim_model) where S_img = h0 * w0
        S_img, B, D = z.shape
        # Rearrange spatial tokens back to a feature map grid.
        x = z.permute(1, 2, 0).view(B, D, self.h0, self.w0)  # (B, D, h0, w0)
        x = self.chan_proj(x)                                  # (B, base_ch, h0, w0)
        return self.decoder(x)


def _n_encoder_tokens(config: AWMConfig) -> int:
    """Compute the total number of encoder output tokens S from config."""
    n = sum([bool(config.robot_state_feature), bool(config.env_state_feature)])
    if config.image_features:
        for feat in config.image_features.values():
            C, H, W = feat.shape
            stride = 16 if config.replace_final_stride_with_dilation else 32
            n += (H // stride) * (W // stride)
    return n


def _slice_obs_batch(batch: dict[str, Tensor], idx: int) -> dict[str, Tensor]:
    """Return a batch dict with observation tensors sliced to a single temporal index.

    When observation_delta_indices loads multiple time steps, obs tensors gain an extra
    leading temporal dimension: (B, num_steps, ...). This helper extracts one step.
    Non-observation keys (action, action_is_pad, etc.) are passed through unchanged.
    """
    result = {}
    for key, val in batch.items():
        if key.startswith("observation.") and isinstance(val, Tensor) and val.ndim >= 2:
            result[key] = val[:, idx]
        else:
            result[key] = val
    return result


class AWMPolicy(PreTrainedPolicy):
    """AWM: Autoregressive Action Chunking Transformer with discrete token prediction.

    At training time the decoder is teacher-forced with the (shifted-right) embeddings of the
    ground-truth token indices and trained with cross-entropy loss.  At inference time tokens
    are generated greedily one step at a time and decoded back to continuous actions.

    Optionally, a WM-guided MPPI planner refines the BC trajectory at inference time.
    Call ``set_planning_goal(batch)`` once per episode to activate planning.
    """

    config_class = AWMConfig
    name = "awm"

    def __init__(self, config: AWMConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = AWM(config)

        # Planning: instantiate planner if enabled; goal must be set separately.
        if config.use_planning:
            from lerobot.policies.awm.planning_awm import build_planner
            self._planner = build_planner(config)
        else:
            self._planner = None
        self._goal_latent: Tensor | None = None  # set via set_planning_goal()

        self.reset()

    # ------------------------------------------------------------------
    # Planning API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def set_planning_goal(self, batch: dict[str, Tensor]) -> None:
        """Encode a goal observation and cache its latent for the planner.

        Must be called once per episode (or whenever the goal changes) before
        ``select_action()`` so that planning has a target to optimise toward.

        Args:
            batch: A single-timestep observation batch for the goal state.  The batch
                should contain the same keys as a normal observation (e.g. images, state).
        """
        self.eval()
        goal_batch = dict(batch)
        if self.config.image_features:
            goal_batch[OBS_IMAGES] = [goal_batch[key] for key in self.config.image_features]
        _, _, _, goal_encoder_in = self.model._encode(goal_batch)
        # goal_encoder_in: (S, B, dim_model) — take first (and only) batch element.
        self._goal_latent = goal_encoder_in[:, 0, :].detach()  # (S, dim_model)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_optim_params(self) -> dict:
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
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions; optionally refined via WM-guided MPPI.

        Returns:
            ``(B, chunk_size, action_dim)`` continuous action tensor.
        """
        self.eval()

        # Assemble image list expected by _encode / predict_ar.
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        # Encode once and share between BC decoder and planner.
        batch_size, cross_kv, cross_pos, encoder_in = self.model._encode(batch)

        # BC warm start: greedy autoregressive decoding.  (B, T, D)
        actions = self.model._predict_ar_from_encoded(cross_kv, cross_pos, batch_size)

        # WM-guided MPPI refinement (if planner is configured and a goal is set).
        if self._planner is not None and self._goal_latent is not None:
            # Move goal to the same device as the model.
            goal = self._goal_latent.to(encoder_in.device)
            lows = self.model.tokenizer.lows    # (D,)
            highs = self.model.tokenizer.highs  # (D,)
            # Run planning for each batch element independently (typically B=1 at inference).
            refined = []
            for b in range(batch_size):
                enc_b = encoder_in[:, b : b + 1, :]     # (S, 1, dim_model)
                cost_fn_b = self.model.make_wm_cost_fn(enc_b, cross_pos, goal)
                refined.append(self._planner.optimize(actions[b], cost_fn_b, lows, highs))
            actions = torch.stack(refined, dim=0)  # (B, T, D)

        return actions  # (B, T, D)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Teacher-forced training forward pass; returns combined action + world model loss."""
        # Split the temporally-stacked obs (B, 2, ...) into current (t) and next (t+H).
        curr_batch = _slice_obs_batch(batch, 0)
        next_batch = _slice_obs_batch(batch, 1)

        # OBS_IMAGES is a list, not a Tensor — assemble it after slicing so images are (B, C, H, W).
        if self.config.image_features:
            curr_batch = dict(curr_batch)
            curr_batch[OBS_IMAGES] = [curr_batch[key] for key in self.config.image_features]
            next_batch = dict(next_batch)
            next_batch[OBS_IMAGES] = [next_batch[key] for key in self.config.image_features]

        # Episode-boundary mask: True where t+H is beyond the episode end.
        next_obs_is_pad = batch.get(
            "observation.state_is_pad",
            batch.get("observation.environment_state_is_pad"),
        )

        logits, token_ids, wm_tensors = self.model(curr_batch, next_batch)

        total_V = self.model.tokenizer.total_vocab_size
        loss_per_tok = F.cross_entropy(
            logits.reshape(-1, total_V),
            token_ids.reshape(-1),
            reduction="none",
            label_smoothing=0.1,
        )  # (B*T*D,)

        # Zero out padded timesteps; divide only by valid (non-padding) count.
        # action_is_pad: (B, T) → expand to (B, T, D) → flatten to match (B*T*D,) logits
        action_dim = self.config.action_feature.shape[0]
        valid = ~(
            curr_batch["action_is_pad"]
            .unsqueeze(-1)
            .expand(-1, -1, action_dim)
            .reshape(-1)
        )
        action_loss = (loss_per_tok * valid).sum() / valid.sum().clamp(min=1)

        # World model cosine similarity loss — masked at episode boundaries.
        z_pred, z_target, decoded_curr, gt_curr_img = wm_tensors
        valid_wm = ~next_obs_is_pad[:, 1]  # (B,)
        cos_sim = F.cosine_similarity(z_pred, z_target, dim=-1).mean(dim=0)  # mean over S → (B,)
        wm_loss = 1 - (cos_sim * valid_wm).sum() / valid_wm.sum().clamp(min=1)

        loss = action_loss + self.config.wm_loss_weight * wm_loss
        info = {
            "action_loss": action_loss.item(),
            "wm_loss": wm_loss.item(),
        }

        # Image reconstruction loss on current obs — MSE in normalised pixel space.
        if decoded_curr is not None and gt_curr_img is not None:
            decoder_loss = F.mse_loss(decoded_curr, gt_curr_img)
            loss = loss + self.config.decoder_loss_weight * decoder_loss
            info["decoder_loss"] = decoder_loss.item()

        info["loss"] = loss.item()
        return loss, info

    @torch.no_grad()
    def visualize(self, batch: dict[str, Tensor], n_pairs: int = 12) -> dict[str, Tensor] | None:
        """Generate WM image reconstruction pairs for debugging.

        Returns a dict with two keys, each ``(N, C, H, 2W)`` float in ``[0, 1]``:
          - ``"curr"`` — GT current observation (left) | decoded from encoder tokens (right)
          - ``"next"`` — GT next observation (left) | decoded from WM prediction (right)

        Returns ``None`` when no image features are configured.
        """
        if not self.config.image_features or not hasattr(self.model, "wm_image_decoder"):
            return None

        was_training = self.training
        self.eval()

        n = min(n_pairs, batch["action"].shape[0])

        def _prep(raw_batch: dict, idx: int) -> dict:
            sliced = _slice_obs_batch(raw_batch, idx)
            d = {k: v[:n] if isinstance(v, Tensor) else v for k, v in sliced.items()}
            d = dict(d)
            d[OBS_IMAGES] = [d[k][:n] for k in self.config.image_features]
            return d

        curr_batch = _prep(batch, 0)
        next_batch = _prep(batch, 1)

        n_1d = self.model.n_1d_tokens
        s_img = self.model.img_tokens_per_cam

        # Encode current obs.
        batch_size, cross_kv, cross_pos, curr_encoder_in = self.model._encode(curr_batch)

        # Current decoded image — from pre-transformer encoder input tokens.
        curr_img_z = curr_encoder_in[n_1d : n_1d + s_img]           # (S_img, N, D)
        decoded_curr = self.model.wm_image_decoder(curr_img_z)       # (N, C, H, W)
        gt_curr = curr_batch[OBS_IMAGES][0]                          # (N, C, H, W)

        # WM-predicted future latent from ground-truth action tokens.
        # token_ids: (N, T, D) — one token per action dim per step.
        token_ids = self.model.tokenizer.encode(batch[ACTION][:n])
        T = token_ids.shape[1]
        action_embeds = self.model.wm_token_embed(token_ids).mean(dim=2).transpose(0, 1)  # (T, N, d)

        wm_pos = self.model.wm_decoder_pos_embed.weight[:T].unsqueeze(1)
        S = self.model.n_encoder_tokens
        query_pos = self.model.wm_query_pos_embed.weight.unsqueeze(1)
        queries = (self.model.wm_query_tokens + query_pos).expand(-1, batch_size, -1)
        wm_in = torch.cat([queries, action_embeds + wm_pos], dim=0)
        wm_cross_kv = self.model.wm_cross_attn_proj(curr_encoder_in)
        wm_out = self.model.wm_decoder(wm_in, wm_cross_kv, cross_pos, None)
        z_pred = self.model.wm_proj_head(wm_out[:S])

        # Decode WM prediction to image.
        next_img_z = z_pred[n_1d : n_1d + s_img]                    # (S_img, N, D)
        decoded_next = self.model.wm_image_decoder(next_img_z)       # (N, C, H, W)
        gt_next = next_batch[OBS_IMAGES][0]                          # (N, C, H, W)

        def _to_01(t: Tensor) -> Tensor:
            B = t.shape[0]
            t_flat = t.view(B, -1)
            lo = t_flat.min(dim=1).values.view(B, 1, 1, 1)
            hi = t_flat.max(dim=1).values.view(B, 1, 1, 1)
            return ((t - lo) / (hi - lo + 1e-8)).clamp(0, 1)

        if was_training:
            self.train()

        curr = torch.cat([_to_01(gt_curr), _to_01(decoded_curr)], dim=3)   # (N, C, H, 2W)
        next_ = torch.cat([_to_01(gt_next), _to_01(decoded_next)], dim=3)  # (N, C, H, 2W)
        return {"curr": curr.cpu(), "next": next_.cpu()}


class AWM(nn.Module):
    """Core network for AWMPolicy.

    Encoder: identical to ACT (ResNet backbone + transformer encoder).
    Decoder: autoregressive transformer with
        * causal self-attention
        * cross-attention on (optionally compressed) encoder outputs
        * discrete token prediction head
    World model: bidirectional transformer that predicts next-state encoder tokens
        given current encoder tokens and the action chunk.
    """

    def __init__(self, config: AWMConfig):
        super().__init__()
        self.config = config

        # ------------------------------------------------------------------
        # Action tokenizer
        # ------------------------------------------------------------------
        action_dim = config.action_feature.shape[0]
        self.action_dim = action_dim
        action_ranges = (
            config.action_ranges if config.action_ranges is not None else [[-1.0, 1.0]] * action_dim
        )
        self.tokenizer = FactoredActionTokenizer(action_ranges, config.action_token_vocab_size)
        total_V = self.tokenizer.total_vocab_size
        decoder_seq_len = config.chunk_size * action_dim  # T*D positions (one token per dim per step)

        # ------------------------------------------------------------------
        # Vision backbone (optional)
        # ------------------------------------------------------------------
        if config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # ------------------------------------------------------------------
        # Transformer encoder and decoder
        # ------------------------------------------------------------------
        self.encoder = ACTEncoder(config)
        self.decoder = AWMDecoder(config)

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
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )

        # ------------------------------------------------------------------
        # Encoder positional embeddings
        # ------------------------------------------------------------------
        n_1d_tokens = sum([bool(config.robot_state_feature), bool(config.env_state_feature)])
        self.n_1d_tokens = n_1d_tokens  # stored so image decoder knows which tokens are spatial
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if config.image_features:
            C, H, W = config.image_features["observation.image"].shape
            self.encoder_cam_feat_pos_embed = ACTLearnedPositionEmbedding2d(H, W, config.dim_model)

        # ------------------------------------------------------------------
        # Cross-attention dimension reduction
        # ------------------------------------------------------------------
        self.cross_attn_proj = nn.Sequential(
            nn.Linear(config.dim_model, config.cross_attn_dim),
            nn.ReLU(),
            nn.Linear(config.cross_attn_dim, config.cross_attn_dim),
        )
        self.cross_attn_pos_proj = nn.Linear(config.dim_model, config.cross_attn_dim)
        # Separate projection for WM decoder cross-attention (encoder INPUT tokens as KV).
        self.wm_cross_attn_proj = nn.Sequential(
            nn.Linear(config.dim_model, config.cross_attn_dim),
            nn.ReLU(),
            nn.Linear(config.cross_attn_dim, config.cross_attn_dim),
        )

        # ------------------------------------------------------------------
        # Decoder inputs: BOS token + discrete token embedding table
        # ------------------------------------------------------------------
        self.bos_embed = nn.Embedding(1, config.dim_model)
        # Action decoder embedding table.
        self.token_embed = nn.Embedding(total_V, config.dim_model)
        # Separate embedding table for WM action conditioning — prevents competing gradients with
        # the action decoder's token_embed.
        self.wm_token_embed = nn.Embedding(total_V, config.dim_model)

        # ------------------------------------------------------------------
        # Decoder positional embeddings — chunk_size*action_dim positions (T*D).
        # ------------------------------------------------------------------
        self.decoder_pos_embed = nn.Embedding(decoder_seq_len, config.dim_model)

        # ------------------------------------------------------------------
        # Action head: predicts logits over the token vocabulary
        # ------------------------------------------------------------------
        self.action_head = nn.Linear(config.dim_model, total_V)

        # ------------------------------------------------------------------
        # World model decoder — bidirectional (no causal mask), shallower.
        # ------------------------------------------------------------------
        wm_cfg = copy(config)
        wm_cfg.n_decoder_layers = config.n_wm_decoder_layers
        self.wm_decoder = AWMDecoder(wm_cfg)

        # Positional embeddings for WM decoder action-token inputs (length T, not T*D,
        # because WM uses per-timestep embeddings after mean-pooling over dims if factored).
        self.wm_decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

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

        # MLP projection head: query output → predicted next-state latent token.
        self.wm_proj_head = nn.Sequential(
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

    def _reset_parameters(self):
        """Xavier-uniform initialisation for transformer and projection weights."""
        modules = [
            self.encoder.parameters(),
            self.decoder.parameters(),
            self.wm_decoder.parameters(),
            self.wm_proj_head.parameters(),
            self.cross_attn_proj.parameters(),
            self.cross_attn_pos_proj.parameters(),
            self.wm_cross_attn_proj.parameters(),
            self.wm_query_pos_embed.parameters(),
        ]
        if hasattr(self, "wm_image_decoder"):
            modules.append(self.wm_image_decoder.parameters())
        for p in chain(*modules):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, batch: dict[str, Tensor]) -> tuple[int, Tensor, Tensor, Tensor]:
        """Run the encoder and project its output for decoder cross-attention.

        Returns:
            batch_size:    int
            cross_kv:      (S, B, cross_attn_dim)  — keys/values for cross-attention
            cross_pos:     (S, 1, cross_attn_dim)  — positional bias added to cross-attn keys
            encoder_in:    (S, B, dim_model)  — projected input tokens (pre-transformer)
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
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        cross_kv = self.cross_attn_proj(encoder_out)                    # (S, B, cross_attn_dim)
        cross_pos = self.cross_attn_pos_proj(encoder_in_pos_embed)      # (S, 1, cross_attn_dim)

        return batch_size, cross_kv, cross_pos, encoder_in_tokens

    def _predict_ar_from_encoded(
        self,
        cross_kv: Tensor,
        cross_pos: Tensor,
        batch_size: int,
    ) -> Tensor:
        """Autoregressive greedy decoding given pre-computed encoder outputs.

        Avoids re-running the encoder when it has already been called (e.g. by the planner).

        Returns:
            ``(B, chunk_size, action_dim)`` continuous action chunk.
        """
        device = cross_kv.device
        decoder_seq = self.bos_embed.weight.unsqueeze(1).expand(1, batch_size, -1).contiguous()

        D = self.action_dim
        n_steps = self.config.chunk_size * D
        predicted_flat: list[Tensor] = []
        for i in range(n_steps):
            T_cur = i + 1
            causal_mask = _make_causal_mask(T_cur, device=device)
            pos_embed_i = self.decoder_pos_embed.weight[:T_cur].unsqueeze(1)
            out = self.decoder(
                decoder_seq, cross_kv, cross_pos, causal_mask, decoder_pos_embed=pos_embed_i
            )
            tok_i = self.action_head(out[-1]).argmax(dim=-1)   # (B,)
            predicted_flat.append(tok_i)
            if i < n_steps - 1:
                embed = self.token_embed(tok_i).unsqueeze(0)
                decoder_seq = torch.cat([decoder_seq, embed], dim=0)
        flat_ids = torch.stack(predicted_flat, dim=1)                       # (B, T*D)
        token_ids = flat_ids.reshape(batch_size, self.config.chunk_size, D) # (B, T, D)
        return self.tokenizer.decode(token_ids)                             # (B, T, D)

    def make_wm_cost_fn(
        self,
        encoder_in: Tensor,
        cross_pos: Tensor,
        goal_latent: Tensor,
    ) -> Callable[[Tensor], Tensor]:
        """Build a WM-based cost function for use by the planner.

        The returned callable evaluates N candidate action trajectories by rolling them
        through the world model and measuring cosine distance to a goal latent — consistent
        with the WM training objective.

        Args:
            encoder_in:  ``(S, 1, dim_model)`` pre-transformer encoder input tokens for the
                         current observation (batch size 1 expected at inference).
            cross_pos:   ``(S, 1, cross_attn_dim)`` positional biases for cross-attention.
            goal_latent: ``(S, dim_model)`` encoder-in tokens of the goal observation (obtained
                         by calling ``AWMPolicy.set_planning_goal()``).

        Returns:
            ``cost_fn(actions: (N, T, D)) → (N,)`` — cost per candidate trajectory.
            Lower cost = trajectory predicted to end closer to the goal.
        """
        S = self.n_encoder_tokens
        # Pre-compute WM cross-attention keys once; amortised across all planner iterations.
        wm_cross_kv1 = self.wm_cross_attn_proj(encoder_in)  # (S, 1, cross_attn_dim)

        @torch.no_grad()
        def cost_fn(actions: Tensor) -> Tensor:
            """Evaluate N candidate trajectories; returns ``(N,)`` costs."""
            N, T, _ = actions.shape

            # 1. Tokenise continuous actions and embed via WM embedding table.
            token_ids = self.tokenizer.encode(actions)                            # (N, T, D)
            action_emb = self.wm_token_embed(token_ids).mean(dim=2).permute(1, 0, 2)  # (T, N, d)

            # 2. Build WM decoder input: [S query tokens ‖ T action tokens].
            wm_pos    = self.wm_decoder_pos_embed.weight[:T].unsqueeze(1)         # (T, 1, d)
            q_pos     = self.wm_query_pos_embed.weight.unsqueeze(1)               # (S, 1, d)
            queries   = (self.wm_query_tokens + q_pos).expand(-1, N, -1)         # (S, N, d)
            wm_in     = torch.cat([queries, action_emb + wm_pos], dim=0)         # (S+T, N, d)

            # 3. Run WM decoder (bidirectional).
            kv_n      = wm_cross_kv1.expand(-1, N, -1)                           # (S, N, cross_dim)
            wm_out    = self.wm_decoder(wm_in, kv_n, cross_pos, None)            # (S+T, N, d)
            z_pred    = self.wm_proj_head(wm_out[:S])                            # (S, N, d)

            # 4. Cosine distance to goal (lower = closer to goal = better trajectory).
            goal_n    = goal_latent.unsqueeze(1).expand(-1, N, -1)               # (S, N, d)
            cos_sim   = F.cosine_similarity(z_pred, goal_n, dim=-1).mean(dim=0)  # (N,)
            return 1.0 - cos_sim                                                  # ∈ [0, 2]

        return cost_fn

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: dict[str, Tensor],
        next_batch: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor] | None]:
        """Teacher-forced training forward pass.

        Args:
            batch:      Must contain ``ACTION`` with shape ``(B, chunk_size, action_dim)``.
            next_batch: Batch dict for the next observation (t+H). When provided, the world
                        model head is computed and WM tensors are returned.

        Returns:
            logits:       joint:    ``(B, T, V^D)``
                          factored: ``(B, T*D, V)``
            token_ids:    joint:    ``(B, T)``
                          factored: ``(B, T*D)``
            wm_tensors:   ``(z_pred, z_target, decoded_curr, gt_curr_img)``
        """
        batch_size, cross_kv, cross_pos, encoder_in = self._encode(batch)

        actions = batch[ACTION]  # (B, T, action_dim)
        T = actions.shape[1]

        # One token per action dimension → flat sequence of length T*D.
        token_ids = self.tokenizer.encode(actions)              # (B, T, D)
        D = self.action_dim
        flat_ids = token_ids.reshape(batch_size, T * D)         # (B, T*D)

        bos = self.bos_embed.weight.unsqueeze(1).expand(1, batch_size, -1)  # (1, B, d)
        prev_embeds = self.token_embed(flat_ids[:, :-1]).transpose(0, 1)    # (T*D-1, B, d)
        decoder_in = torch.cat([bos, prev_embeds], dim=0)                   # (T*D, B, d)

        L = T * D
        causal_mask = _make_causal_mask(L, device=decoder_in.device)
        decoder_pos_embed = self.decoder_pos_embed.weight[:L].unsqueeze(1)
        decoder_out = self.decoder(decoder_in, cross_kv, cross_pos, causal_mask,
                                   decoder_pos_embed=decoder_pos_embed)

        logits = self.action_head(decoder_out.transpose(0, 1))  # (B, T*D, V)

        # ------------------------------------------------------------------
        # World model forward.
        # ------------------------------------------------------------------
        _, _, _, next_encoder_in = self._encode(next_batch)
        z_target = next_encoder_in.detach()  # (S, B, dim_model)

        # WM action conditioning: embed per-dim tokens, mean-pool over D → (T, B, d).
        action_embeds = self.wm_token_embed(token_ids).mean(dim=2).transpose(0, 1)  # (T, B, d)

        wm_pos    = self.wm_decoder_pos_embed.weight[:T].unsqueeze(1)           # (T, 1, d)
        query_pos = self.wm_query_pos_embed.weight.unsqueeze(1)                 # (S, 1, d)
        queries   = (self.wm_query_tokens + query_pos).expand(-1, batch_size, -1)  # (S, B, d)
        wm_in     = torch.cat([queries, action_embeds + wm_pos], dim=0)         # (S+T, B, d)

        S = self.n_encoder_tokens
        wm_cross_kv = self.wm_cross_attn_proj(encoder_in)               # (S, B, cross_attn_dim)
        wm_out = self.wm_decoder(wm_in, wm_cross_kv, cross_pos, None)   # (S+T, B, d)
        z_pred = self.wm_proj_head(wm_out[:S])                           # (S, B, d)

        decoded_curr, gt_curr_img = None, None
        if hasattr(self, "wm_image_decoder") and OBS_IMAGES in batch:
            curr_img_z = encoder_in[self.n_1d_tokens : self.n_1d_tokens + self.img_tokens_per_cam]
            decoded_curr = self.wm_image_decoder(curr_img_z.detach())    # (B, C, H, W)
            gt_curr_img = batch[OBS_IMAGES][0].detach()                  # (B, C, H, W)

        wm_tensors = (z_pred, z_target, decoded_curr, gt_curr_img)
        return logits, flat_ids, wm_tensors

    def predict_ar(self, batch: dict[str, Tensor]) -> Tensor:
        """Autoregressive greedy inference (encodes internally).

        Returns:
            ``(B, chunk_size, action_dim)`` continuous action chunk.
        """
        batch_size, cross_kv, cross_pos, _ = self._encode(batch)
        return self._predict_ar_from_encoded(cross_kv, cross_pos, batch_size)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_causal_mask(seq_len: int, device: torch.device) -> Tensor:
    """Boolean causal mask of shape ``(seq_len, seq_len)``.

    ``True`` entries are masked out (ignored); position ``i`` attends only to positions ``0..i``.
    """
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()


# ---------------------------------------------------------------------------
# Decoder modules
# ---------------------------------------------------------------------------


class AWMDecoder(nn.Module):
    """Stack of AWMDecoderLayer modules followed by optional layer norm."""

    def __init__(self, config: AWMConfig):
        super().__init__()
        self.layers = nn.ModuleList([AWMDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self,
        x: Tensor,
        cross_kv: Tensor,
        cross_pos: Tensor,
        causal_mask: Tensor | None,
        decoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x:                (T, B, dim_model)
            cross_kv:         (S, B, cross_attn_dim)
            cross_pos:        (S, 1, cross_attn_dim)
            causal_mask:      (T, T) bool — True = ignored; None = bidirectional (no masking)
            decoder_pos_embed:(T, 1, dim_model) or None

        Returns:
            (T, B, dim_model)
        """
        for layer in self.layers:
            x = layer(x, cross_kv, cross_pos, causal_mask, decoder_pos_embed=decoder_pos_embed)
        return self.norm(x)


class AWMDecoderLayer(nn.Module):
    """Single AWM decoder layer: causal self-attention + compressed cross-attention + FFN."""

    def __init__(self, config: AWMConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(
            config.dim_model,
            config.n_heads,
            dropout=config.dropout,
            kdim=config.cross_attn_dim,
            vdim=config.cross_attn_dim,
        )

        # Feed-forward sublayer.
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
        causal_mask: Tensor | None,
        decoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x:                (T, B, dim_model)
            cross_kv:         (S, B, cross_attn_dim)
            cross_pos:        (S, 1, cross_attn_dim)
            causal_mask:      (T, T) bool or None — None gives bidirectional self-attention
            decoder_pos_embed:(T, 1, dim_model) or None

        Returns:
            (T, B, dim_model)
        """
        # --- Causal self-attention ---
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self._add_pos(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x, attn_mask=causal_mask, need_weights=False)[0]
        x = skip + self.dropout1(x)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        # --- Cross-attention (encoder keys/values are compressed to cross_attn_dim) ---
        x = self.multihead_attn(
            query=self._add_pos(x, decoder_pos_embed),
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

        # --- Feed-forward ---
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)

        return x
