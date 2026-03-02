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
"""AWM Policy — Autoregressive ACT Decoder with Discrete Action Tokens

Two key differences from ACTSimple:
  1. Autoregressive decoder with causal self-attention: teacher-forcing at training time
     (shifted-right ground-truth token embeddings), step-by-step greedy decoding at inference.
  2. Discrete action tokens: continuous actions are quantised via UniformActionTokenizer into
     joint discrete tokens.  The model predicts a categorical distribution over the joint
     vocabulary (cross-entropy loss) and decodes the argmax token back to a continuous action
     for environment interaction.

Optional:
  - Cross-attention dimension reduction MLP: projects encoder outputs from `dim_model` to
    `cross_attn_dim` before decoder cross-attention (default: no compression).
"""

from collections import deque
from itertools import chain

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
from lerobot.policies.awm.tokenizer_awm import UniformActionTokenizer
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class AWMPolicy(PreTrainedPolicy):
    """AWM: Autoregressive Action Chunking Transformer with discrete token prediction.

    At training time the decoder is teacher-forced with the (shifted-right) embeddings of the
    ground-truth token indices and trained with cross-entropy loss.  At inference time tokens
    are generated greedily one step at a time and decoded back to continuous actions.
    """

    config_class = AWMConfig
    name = "awm"

    def __init__(self, config: AWMConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = AWM(config)

        self.reset()

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
        """Predict a chunk of actions autoregressively; returns continuous actions."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        return self.model.predict_ar(batch)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Teacher-forced training forward pass; returns cross-entropy loss."""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        logits, token_ids = self.model(batch)
        # logits:    (B, T, total_vocab_size)
        # token_ids: (B, T)

        total_V = self.model.tokenizer.total_vocab_size
        loss_per_tok = F.cross_entropy(
            logits.reshape(-1, total_V),
            token_ids.reshape(-1),
            reduction="none",
        )  # (B*T,)

        # Zero out padded timesteps; divide only by valid (non-padding) count.
        valid = ~batch["action_is_pad"].reshape(-1)  # (B*T,)
        loss = (loss_per_tok * valid).sum() / valid.sum().clamp(min=1)

        return loss, {"loss": loss.item()}


class AWM(nn.Module):
    """Core network for AWMPolicy.

    Encoder: identical to ACT (ResNet backbone + transformer encoder).
    Decoder: autoregressive transformer with
        * causal self-attention
        * cross-attention on (optionally compressed) encoder outputs
        * discrete token prediction head (vocab = vocab_size^action_dim)
    """

    def __init__(self, config: AWMConfig):
        super().__init__()
        self.config = config

        # ------------------------------------------------------------------
        # Action tokenizer
        # ------------------------------------------------------------------
        action_dim = config.action_feature.shape[0]
        action_ranges = config.action_ranges if config.action_ranges is not None else [[-1.0, 1.0]] * action_dim
        self.tokenizer = UniformActionTokenizer(action_ranges, config.action_token_vocab_size)
        total_V = self.tokenizer.total_vocab_size

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

        # ------------------------------------------------------------------
        # Decoder inputs: BOS token + discrete token embedding table
        # ------------------------------------------------------------------
        self.bos_embed = nn.Embedding(1, config.dim_model)
        # Embed the previous step's discrete token as the decoder input for the next step.
        self.token_embed = nn.Embedding(total_V, config.dim_model)

        # ------------------------------------------------------------------
        # Decoder positional embeddings (used during AR inference)
        # ------------------------------------------------------------------
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # ------------------------------------------------------------------
        # Action head: predicts logits over the joint token vocabulary
        # ------------------------------------------------------------------
        self.action_head = nn.Linear(config.dim_model, total_V)

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialisation for transformer and projection weights."""
        for p in chain(
            self.encoder.parameters(),
            self.decoder.parameters(),
            self.cross_attn_proj.parameters(),
            self.cross_attn_pos_proj.parameters(),
        ):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, batch: dict[str, Tensor]) -> tuple[int, Tensor, Tensor, Tensor]:
        """Run the encoder and project its output for decoder cross-attention.

        Returns:
            batch_size: int
            encoder_out: (S, B, dim_model)
            cross_kv:    (S, B, cross_attn_dim)  — keys/values for cross-attention
            cross_pos:   (S, 1, cross_attn_dim)  — positional bias added to cross-attn keys
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

        cross_kv = self.cross_attn_proj(encoder_out)            # (S, B, cross_attn_dim)
        cross_pos = self.cross_attn_pos_proj(encoder_in_pos_embed)  # (S, 1, cross_attn_dim)

        return batch_size, encoder_out, cross_kv, cross_pos

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Teacher-forced training forward pass.

        Args:
            batch: Must contain ``ACTION`` with shape ``(B, chunk_size, action_dim)``.

        Returns:
            logits:    ``(B, T, total_vocab_size)`` — unnormalised log-probabilities.
            token_ids: ``(B, T)`` — ground-truth joint token indices (for cross-entropy).
        """
        batch_size, _, cross_kv, cross_pos = self._encode(batch)

        actions = batch[ACTION]  # (B, T, action_dim)
        T = actions.shape[1]

        # Tokenise ground-truth actions → joint token indices.
        token_ids = self.tokenizer.encode(actions)  # (B, T)

        # Build shifted-right decoder input: [BOS, embed(tok_0), …, embed(tok_{T-2})].
        bos = self.bos_embed.weight.unsqueeze(1).expand(1, batch_size, -1)  # (1, B, dim_model)
        prev_embeds = self.token_embed(token_ids[:, :-1]).transpose(0, 1)   # (T-1, B, dim_model)
        decoder_in = torch.cat([bos, prev_embeds], dim=0)                   # (T, B, dim_model)

        causal_mask = _make_causal_mask(T, device=decoder_in.device)
        decoder_pos_embed = self.decoder_pos_embed.weight[:T].unsqueeze(1)  # (T, 1, dim_model)
        decoder_out = self.decoder(decoder_in, cross_kv, cross_pos, causal_mask,
                                   decoder_pos_embed=decoder_pos_embed)

        logits = self.action_head(decoder_out.transpose(0, 1))  # (B, T, total_vocab_size)
        return logits, token_ids

    def predict_ar(self, batch: dict[str, Tensor]) -> Tensor:
        """Autoregressive greedy inference.

        At each step the highest-probability token is selected (argmax), embedded, and fed as
        input to the next decoder step.  All generated token indices are decoded back to
        continuous actions via the tokenizer.

        Returns:
            ``(B, chunk_size, action_dim)`` continuous action chunk.
        """
        batch_size, _, cross_kv, cross_pos = self._encode(batch)

        decoder_seq = self.bos_embed.weight.unsqueeze(1).expand(1, batch_size, -1).contiguous()

        predicted_ids: list[Tensor] = []
        for t in range(self.config.chunk_size):
            T = t + 1
            causal_mask = _make_causal_mask(T, device=decoder_seq.device)
            pos_embed_t = self.decoder_pos_embed.weight[:T].unsqueeze(1)  # (T, 1, dim_model)

            out = self.decoder(
                decoder_seq, cross_kv, cross_pos, causal_mask, decoder_pos_embed=pos_embed_t
            )  # (T, B, dim_model)

            logits_t = self.action_head(out[-1])          # (B, total_vocab_size)
            token_id_t = logits_t.argmax(dim=-1)          # (B,)  greedy
            predicted_ids.append(token_id_t)

            if t < self.config.chunk_size - 1:
                embed = self.token_embed(token_id_t).unsqueeze(0)  # (1, B, dim_model)
                decoder_seq = torch.cat([decoder_seq, embed], dim=0)

        token_ids = torch.stack(predicted_ids, dim=1)  # (B, chunk_size)
        return self.tokenizer.decode(token_ids)         # (B, chunk_size, action_dim)


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
        causal_mask: Tensor,
        decoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x:                (T, B, dim_model)
            cross_kv:         (S, B, cross_attn_dim)
            cross_pos:        (S, 1, cross_attn_dim)
            causal_mask:      (T, T) bool — True = ignored
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
        causal_mask: Tensor,
        decoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x:                (T, B, dim_model)
            cross_kv:         (S, B, cross_attn_dim)
            cross_pos:        (S, 1, cross_attn_dim)
            causal_mask:      (T, T) bool
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
