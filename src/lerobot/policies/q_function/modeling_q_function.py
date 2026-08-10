"""Categorical h-step TD Q-function for offline BC data (Q-chunking-inspired).

Overview
========

A **Q(s_t, a_{t:t+h}, l)** critic that maps (multi-view image state, h-step action
chunk, language instruction) to a scalar value via a categorical HL-Gauss head.
Trained with the h-step TD loss from the Q-chunking paper (Eq. 4):

    L(θ) = E_D [ ( Q_θ(s_t, a_{t:t+h}, l)
                   - Σ_{t'=1..h} γ^{t'-1} r_{t+t'-1}
                   - γ^h · Q_{θ̄}(s_{t+h}, a_{t+h:t+2h}, l) )² ]

with the following twists:

* Targets are embedded into a two-hot (HL-Gauss) histogram over ``num_bins`` bins
  in [v_min, v_max]. Loss is cross-entropy of predicted logits against the target.
* Target network θ̄ is a Polyak-averaged copy of θ.
* Reward labeling happens at dataload time (QValueLabelDataset);
  this module just reads ``q_reward_chunk_first`` / ``q_bootstrap_valid`` from the batch.

The vision backbone is **DINOv2-large, finetuned end-to-end**. Patch tokens from the
V views are concatenated with projected T5 text tokens as cross-attention context for
the h-step action-chunk queries.

Note (LeRobot port): unlike the original ``imitation/policies/q_function/`` source,
this module does NOT do internal normalization. LeRobot routes normalization through
external processors built via ``make_pre_post_processors``; callers are responsible
for normalizing inputs before invoking ``forward``.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.q_function.configuration_q_function import QFunctionConfig
from lerobot.policies.q_function.processor_q_function import DINOv2ImagePreprocessor
from lerobot.utils.constants import ACTION

logger = logging.getLogger(__name__)


# Batch keys produced by experiments/mg_dataset_v1/q_dataset.py (must stay in sync).
REWARD_CHUNK_FIRST = "q_reward_chunk_first"   # (B, h) float
REWARD_PAD_FIRST = "q_reward_pad_first"       # (B, h) bool — True where step is beyond episode
BOOTSTRAP_VALID = "q_bootstrap_valid"         # (B,) bool — s_{t+h} exists and is non-terminal
BUCKET_INDEX = "q_bucket_index"               # (B,) long — used for eval/logging


# ── DINOv2 backbone ──────────────────────────────────────────────────────────

class _DINOv2FinetunableBackbone(nn.Module):
    """HF DINOv2 wrapper that returns patch tokens with autograd on.

    Input preprocessing (resize + ImageNet normalisation) is expected to be
    applied *before* this module on the GPU — the HF AutoImageProcessor is NOT
    used since it goes through PIL and breaks the autograd path.
    """

    def __init__(self, model_name: str, freeze: bool = False):
        super().__init__()
        try:
            from transformers import AutoModel  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "transformers is required for DINOv2FinetunableBackbone; install 'transformers'."
            ) from e
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = int(self.model.config.hidden_size)
        self.patch_size = int(self.model.config.patch_size)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, pixel_values: Tensor) -> Tensor:
        """(B*V, 3, H, W) → (B*V, N_patches, D_hidden) — CLS token stripped."""
        outs = self.model(pixel_values=pixel_values)
        return outs.last_hidden_state[:, 1:, :]


# ── T5 text encoder (frozen) ──────────────────────────────────────────────────

class T5TextEncoder(nn.Module):
    """Frozen T5 encoder that maps task strings to token embeddings.

    Uses a lazy per-string cache: with ~130 unique LIBERO tasks the cache
    fills quickly and all subsequent lookups are free.

    Input  : list[str] of length B (one task description per batch element)
    Output : (B, seq_len, hidden_size) float tensor on the requested device
    """

    def __init__(self, model_name: str):
        super().__init__()
        try:
            from transformers import T5EncoderModel, T5Tokenizer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "transformers is required for T5TextEncoder; install 'transformers'."
            ) from e
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.hidden_size: int = int(self.model.config.d_model)
        for p in self.model.parameters():
            p.requires_grad = False
        # Cache: text string → (seq_len, hidden_size) CPU tensor
        self._cache: dict[str, Tensor] = {}

    @torch.no_grad()
    def encode(self, texts: list[str], device: torch.device | str) -> Tensor:
        """Encode a batch of strings, using the cache where possible.

        Returns (B, seq_len, hidden_size) on ``device``.
        """
        unique = list(dict.fromkeys(texts))   # preserve order, deduplicate
        missing = [t for t in unique if t not in self._cache]
        if missing:
            enc = self.tokenizer(
                missing, return_tensors="pt", padding=True, truncation=True, max_length=128
            )
            input_ids = enc["input_ids"].to(next(self.model.parameters()).device)
            attention_mask = enc["attention_mask"].to(input_ids.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = out.last_hidden_state.cpu()   # (N_missing, seq_len, D)
            for i, text in enumerate(missing):
                self._cache[text] = embeddings[i]

        # Gather batch in original order; pad to common seq_len
        seqs = [self._cache[t] for t in texts]
        max_len = max(s.shape[0] for s in seqs)
        padded = torch.zeros(len(texts), max_len, self.hidden_size)
        for i, s in enumerate(seqs):
            padded[i, : s.shape[0]] = s
        return padded.to(device=device)

    def forward(self, texts: list[str], device: torch.device | str) -> Tensor:
        return self.encode(texts, device)


# ── HL-Gauss two-hot target + decode ─────────────────────────────────────────

def _two_hot_target(values: Tensor, bin_centers: Tensor, sigma: float) -> Tensor:
    """Gaussian-smoothed two-hot encoding of scalar targets over a bin grid."""
    values = values.clamp(bin_centers[0], bin_centers[-1]).unsqueeze(-1)  # (B, 1)
    d = (bin_centers.unsqueeze(0) - values) / sigma
    logits = -0.5 * d.pow(2)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    probs = torch.exp(logits)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _expected_value(logits: Tensor, bin_centers: Tensor) -> Tensor:
    """E[bins · softmax(logits)] — scalar expectation of the categorical head."""
    probs = F.softmax(logits, dim=-1)
    return (probs * bin_centers).sum(dim=-1)


# ── Transformer decoder body (self-contained) ────────────────────────────────

def _get_activation(name: str):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    raise ValueError(f"unknown activation: {name}")


class _DecoderLayer(nn.Module):
    def __init__(self, config: QFunctionConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.cross_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout_ffn = nn.Dropout(config.dropout)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        self.pre_norm = config.pre_norm
        self.activation = _get_activation(config.feedforward_activation)

    @staticmethod
    def _maybe_add(x: Tensor, p: Tensor | None) -> Tensor:
        return x if p is None else x + p

    def forward(self, queries, context, query_pos=None, context_pos=None):
        # Self-attention
        x = queries
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self._maybe_add(x, query_pos)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        if not self.pre_norm:
            x = self.norm1(x)

        # Cross-attention
        skip = x
        if self.pre_norm:
            x = self.norm2(x)
        q = self._maybe_add(x, query_pos)
        k = self._maybe_add(context, context_pos)
        x = self.cross_attn(query=q, key=k, value=context)[0]
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)

        # FFN
        skip = x
        if self.pre_norm:
            x = self.norm3(x)
        x = self.linear2(self.dropout_ffn(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


class _SinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embedding for feature maps."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        not_mask = torch.ones_like(x[0, :1])
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi
        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )
        x_enc = x_range.unsqueeze(3) / inverse_frequency
        y_enc = y_range.unsqueeze(3) / inverse_frequency
        x_enc = torch.stack([x_enc[..., 0::2].sin(), x_enc[..., 1::2].cos()], dim=4).flatten(3)
        y_enc = torch.stack([y_enc[..., 0::2].sin(), y_enc[..., 1::2].cos()], dim=4).flatten(3)
        pos = torch.cat([y_enc, x_enc], dim=3).permute(0, 3, 1, 2)
        return pos.to(dtype=x.dtype)


# ── Image encoders ───────────────────────────────────────────────────────────

class DINOv2ImageEncoder(nn.Module):
    """Live DINOv2 front-end: raw images → cross-attention context tokens.

    Input  : (B, V, 3, H, W) float images in [0, 1]-ish range
    Output : (V * N_patches, B, dim_model) context for the decoder
    """

    def __init__(self, config: QFunctionConfig):
        super().__init__()
        self.config = config
        self.image_preprocessor = DINOv2ImagePreprocessor(config.image_resize_h, config.image_resize_w)
        self.backbone = _DINOv2FinetunableBackbone(config.dino_model_name, freeze=config.freeze_backbone)
        if self.backbone.hidden_size != config.dim_model:
            self.backbone_proj = nn.Linear(self.backbone.hidden_size, config.dim_model)
        else:
            self.backbone_proj = nn.Identity()
        self.view_embed = nn.Embedding(len(config.camera_keys), config.dim_model)
        # Pre-register patch positional embedding with known shape so it survives checkpoint
        # save/load. DINOv2 patch_size=14; N_patches = (H/14)*(W/14).
        V = len(config.camera_keys)
        N_patches = (config.image_resize_h // 14) * (config.image_resize_w // 14)
        pe = nn.Parameter(torch.zeros(V, N_patches, config.dim_model))
        nn.init.normal_(pe, std=0.02)
        self._patch_pos_embed = pe

    def forward(self, images: Tensor) -> Tensor:
        B, V, C, H, W = images.shape
        flat = images.reshape(B * V, C, H, W)
        flat = self.image_preprocessor(flat)
        tokens = self.backbone(flat)                 # (B*V, N_patches, D_hidden)
        tokens = self.backbone_proj(tokens)
        N_patches = tokens.shape[1]

        # Fall back to dynamic re-init if image size changed (e.g. different eval resolution).
        if self._patch_pos_embed.shape[1] != N_patches:
            V_cfg = len(self.config.camera_keys)
            pe = nn.Parameter(torch.zeros(V_cfg, N_patches, self.config.dim_model,
                                          device=tokens.device, dtype=tokens.dtype))
            nn.init.normal_(pe, std=0.02)
            self._patch_pos_embed = pe

        view_ids = torch.arange(V, device=tokens.device)
        view_vec = self.view_embed(view_ids).unsqueeze(1).expand(V, N_patches, -1)
        pos = self._patch_pos_embed + view_vec

        tokens = tokens.view(B, V, N_patches, -1)
        tokens = tokens + pos.unsqueeze(0)
        tokens = tokens.reshape(B, V * N_patches, -1)
        return tokens.transpose(0, 1).contiguous()


class ResNet18CachedImageEncoder(nn.Module):
    """Cached-feature front-end: ResNet18 feature maps → cross-attention context.

    Input  : (B, V, C, H_f, W_f)
    Output : (V * H_f * W_f, B, dim_model) context for the decoder
    """

    def __init__(self, config: QFunctionConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Conv2d(config.preencoded_feature_channels, config.dim_model, kernel_size=1)
        self.view_embed = nn.Embedding(len(config.camera_keys), config.dim_model)
        self.pos_embed_2d = _SinusoidalPositionEmbedding2d(config.dim_model // 2)

    def forward(self, feats: Tensor) -> Tensor:
        B, V, C, H, W = feats.shape
        flat = feats.reshape(B * V, C, H, W).to(dtype=self.proj.weight.dtype)
        flat = self.proj(flat)
        pos = self.pos_embed_2d(flat)
        view_ids = torch.arange(V, device=flat.device)
        view_vec = self.view_embed(view_ids)
        tokens = flat.view(B, V, self.config.dim_model, H, W)
        tokens = tokens + pos.unsqueeze(0) + view_vec.view(1, V, -1, 1, 1)
        tokens = tokens.permute(0, 1, 3, 4, 2).reshape(B, V * H * W, -1)
        return tokens.transpose(0, 1).contiguous()


def _make_image_encoder(config: QFunctionConfig) -> nn.Module:
    if config.vision_backbone == "dinov2":
        return DINOv2ImageEncoder(config)
    if config.vision_backbone == "resnet18_cached":
        return ResNet18CachedImageEncoder(config)
    raise ValueError(f"Unknown vision_backbone: {config.vision_backbone!r}")


# ── Inner Q-network ──────────────────────────────────────────────────────────

class QFunction(nn.Module):
    """Q_θ(s_t, a_{t:t+h}, l) as a categorical histogram over ``num_bins`` bins."""

    def __init__(self, config: QFunctionConfig, action_dim_effective: int, t5_hidden_size: int = 0):
        super().__init__()
        self.config = config
        self.action_dim_effective = action_dim_effective

        self.image_encoder = _make_image_encoder(config)

        # Optional text projection: T5 hidden_size → dim_model
        self.use_text = config.use_text_conditioning and t5_hidden_size > 0
        if self.use_text:
            self.text_proj = nn.Linear(t5_hidden_size, config.dim_model)

        self.action_proj = nn.Linear(action_dim_effective, config.dim_model)
        self.query_pos_embed = nn.Parameter(torch.zeros(config.h, config.dim_model))
        nn.init.normal_(self.query_pos_embed, std=0.02)

        if config.pool == "cls":
            self.cls_query = nn.Parameter(torch.zeros(1, config.dim_model))
            nn.init.normal_(self.cls_query, std=0.02)

        self.layers = nn.ModuleList([_DecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.out_norm = nn.LayerNorm(config.dim_model)

        self.head = nn.Sequential(
            nn.Linear(config.dim_model, config.dim_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_model, config.num_bins),
        )

        for p in self.layers.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_context(self, images: Tensor, text_tokens: Tensor | None = None) -> Tensor:
        """Encode image (and optional text) into cross-attention context tokens.

        Returns (S, B, D) context for use with forward_with_context().
        Separating encoding from action scoring enables reuse across N planning candidates.
        """
        context = self.image_encoder(images)                       # (S_img, B, D)
        if self.use_text and text_tokens is not None:
            text_ctx = self.text_proj(text_tokens)                 # (B, T_text, D)
            text_ctx = text_ctx.transpose(0, 1).contiguous()      # (T_text, B, D)
            context = torch.cat([text_ctx, context], dim=0)        # (T_text + S_img, B, D)
        return context

    def forward_with_context(self, context: Tensor, action_chunk: Tensor) -> Tensor:
        """Score action chunks using pre-encoded context — skips image encoder.

        context      : (S, B, D) from encode_context(); B must match action_chunk B
        action_chunk : (B, h, action_dim)
        Returns      : (B, num_bins) logits
        """
        B, h, _ = action_chunk.shape
        queries = self.action_proj(action_chunk)                   # (B, h, D)
        queries = queries.transpose(0, 1).contiguous()             # (h, B, D)
        query_pos = self.query_pos_embed.unsqueeze(1)              # (h, 1, D)
        if self.config.pool == "cls":
            cls = self.cls_query.unsqueeze(1).expand(-1, B, -1)
            queries = torch.cat([cls, queries], dim=0)
            cls_pos = torch.zeros(1, 1, self.config.dim_model,
                                   device=queries.device, dtype=queries.dtype)
            query_pos = torch.cat([cls_pos, query_pos], dim=0)
        x = queries
        for layer in self.layers:
            x = layer(x, context, query_pos=query_pos, context_pos=None)
        x = self.out_norm(x)
        pooled = x[0] if self.config.pool == "cls" else x.mean(dim=0)
        return self.head(pooled)

    def forward(self, images: Tensor, action_chunk: Tensor, text_tokens: Tensor | None = None) -> Tensor:
        """
        images       : (B, V, 3, H, W)
        action_chunk : (B, h, action_dim)
        text_tokens  : (B, T_text, t5_hidden_size) or None
        Returns      : (B, num_bins) logits
        """
        B, h, _ = action_chunk.shape
        if h != self.config.h:
            raise ValueError(f"Expected action chunk length {self.config.h}, got {h}.")

        context = self.encode_context(images, text_tokens)         # (S, B, D)
        return self.forward_with_context(context, action_chunk)


# ── Policy wrapper (LeRobot PreTrainedPolicy) ────────────────────────────────

class QFunctionPolicy(PreTrainedPolicy):
    """LeRobot PreTrainedPolicy holding the online + Polyak-target Q-networks."""

    config_class = QFunctionConfig
    name = "q_function"

    def __init__(self, config: QFunctionConfig, **kwargs):
        super().__init__(config, **kwargs)

        action_ft = config.action_feature
        if action_ft is None:
            raise ValueError("Q-function requires an ACTION feature on the config.")
        full_action_dim = int(action_ft.shape[-1])
        self.full_action_dim = full_action_dim
        self.action_dim_effective = full_action_dim - config.drop_action_tail
        if self.action_dim_effective <= 0:
            raise ValueError(
                f"Dropping tail {config.drop_action_tail} from action_dim {full_action_dim} "
                f"leaves non-positive effective dim."
            )

        # Frozen text encoder (lives on same device as the policy).
        self.text_encoder: T5TextEncoder | None = None
        t5_hidden_size = 0
        if config.use_text_conditioning:
            self.text_encoder = T5TextEncoder(config.text_encoder_model)
            t5_hidden_size = self.text_encoder.hidden_size

        self.q_online = QFunction(config, self.action_dim_effective, t5_hidden_size=t5_hidden_size)
        self.q_target = copy.deepcopy(self.q_online)
        for p in self.q_target.parameters():
            p.requires_grad = False

        centers = torch.linspace(config.v_min, config.v_max, config.num_bins, dtype=torch.float32)
        self.register_buffer("bin_centers", centers)

        self._updates = 0

        num_params = sum(p.numel() for p in self.parameters())
        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_text = (
            sum(p.numel() for p in self.text_encoder.parameters())
            if self.text_encoder is not None else 0
        )
        logger.info(
            f"Q-function: {num_params:,} total, {num_trainable:,} trainable, "
            f"{num_text:,} frozen text encoder "
            f"(target network duplicated, non-trainable)"
        )

    # ── Param groups: lower LR for DINOv2 backbone; text encoder excluded (frozen) ───
    def get_optim_params(self) -> list[dict]:
        backbone_prefix = "q_online.image_encoder.backbone."
        text_prefix = "text_encoder."
        head_params, backbone_params = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith(text_prefix):
                continue   # frozen, never trained
            if n.startswith(backbone_prefix):
                backbone_params.append(p)
            else:
                head_params.append(p)
        return [
            {"params": head_params},
            {"params": backbone_params, "lr": self.config.optimizer_lr_backbone},
        ]

    def reset(self):
        pass

    @torch.no_grad()
    def update(self) -> None:
        """Polyak average online → target. Call after each optimizer.step()."""
        tau = self.config.target_tau
        for p_online, p_target in zip(self.q_online.parameters(), self.q_target.parameters()):
            p_target.data.mul_(1.0 - tau).add_(p_online.data, alpha=tau)
        for b_online, b_target in zip(self.q_online.buffers(), self.q_target.buffers()):
            b_target.data.copy_(b_online.data)
        self._updates += 1

    # ── Helpers ──────────────────────────────────────────────────────────
    def _truncate_action(self, actions: Tensor) -> Tensor:
        d = self.config.drop_action_tail
        return actions if d == 0 else actions[..., : -d]

    def _encode_language(self, batch: dict[str, Any]) -> Tensor | None:
        """Return (B, T_text, D_t5) text tokens, or None if conditioning is off."""
        if self.text_encoder is None:
            return None
        key = self.config.language_key
        if key not in batch:
            logger.warning("language_key=%r not found in batch; skipping text conditioning.", key)
            return None
        raw = batch[key]
        # batch[key] may be a list of strings or a single string
        if isinstance(raw, str):
            texts = [raw]
        else:
            texts = list(raw)
        device = next(self.q_online.parameters()).device
        return self.text_encoder.encode(texts, device=device)

    def _stack_camera_views(self, batch: dict[str, Tensor], delta_idx: int) -> Tensor:
        use_cached = self.config.vision_backbone == "resnet18_cached"
        views = []
        for key in self.config.camera_keys:
            lookup_key = f"{key}_preencoded" if use_cached else key
            if lookup_key not in batch:
                raise KeyError(
                    f"Missing camera key {lookup_key!r} in batch "
                    f"(vision_backbone={self.config.vision_backbone!r})."
                )
            imgs = batch[lookup_key]
            if imgs.dim() == 5:         # (B, n_delta, *per_view)
                imgs = imgs[:, delta_idx]
            elif imgs.dim() == 4:
                if delta_idx != 0:
                    raise RuntimeError(
                        f"Requested delta index {delta_idx} for {lookup_key} but batch has no delta dim."
                    )
            else:
                raise RuntimeError(f"Unexpected shape {imgs.shape} for {lookup_key}")
            views.append(imgs)
        return torch.stack(views, dim=1)    # (B, V, ...)

    # ── Main training forward ────────────────────────────────────────────
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any]]:
        """h-step categorical TD loss. Caller must pre-normalise the batch."""
        actions = batch[ACTION]
        if actions.dim() != 3 or actions.shape[1] != 2 * self.config.h:
            raise RuntimeError(
                f"Expected action shape (B, 2h={2 * self.config.h}, D), got {tuple(actions.shape)}"
            )
        a_first = self._truncate_action(actions[:, : self.config.h, :])
        a_second = self._truncate_action(actions[:, self.config.h :, :])

        imgs_t = self._stack_camera_views(batch, delta_idx=0)
        imgs_tph = self._stack_camera_views(batch, delta_idx=1)

        text_tokens = self._encode_language(batch)

        # Encode the obs context once; reused below for synthetic negatives
        # (identical to q_online(imgs, a) — forward() is exactly this two-step).
        context = self.q_online.encode_context(imgs_t, text_tokens)
        logits_online = self.q_online.forward_with_context(context, a_first)

        with torch.no_grad():
            target_logits = self.q_target(imgs_tph, a_second, text_tokens=text_tokens)
            q_next = _expected_value(target_logits, self.bin_centers)

        reward_chunk = batch[REWARD_CHUNK_FIRST].to(dtype=q_next.dtype)  # (B, h)
        discounts = torch.pow(
            torch.full((self.config.h,), self.config.gamma, device=reward_chunk.device,
                        dtype=reward_chunk.dtype),
            torch.arange(self.config.h, device=reward_chunk.device, dtype=reward_chunk.dtype),
        )
        discounted_rewards = (reward_chunk * discounts.unsqueeze(0)).sum(dim=-1)

        bootstrap_valid = batch[BOOTSTRAP_VALID].to(dtype=q_next.dtype)
        gamma_h = self.config.gamma ** self.config.h
        y = discounted_rewards + gamma_h * bootstrap_valid * q_next

        with torch.no_grad():
            y_min = y.min().item()
            y_max = y.max().item()
            oob = y_min < self.config.v_min or y_max > self.config.v_max
            if oob and (self._updates % 500 == 0):
                logger.warning(
                    "Q target outside HL-Gauss support: y∈[%.2f, %.2f] vs "
                    "[v_min=%.2f, v_max=%.2f] (step %d). Targets will be clamped "
                    "into the support; widen [v_min, v_max] or reduce reward magnitudes.",
                    y_min, y_max, self.config.v_min, self.config.v_max, self._updates,
                )

        target_probs = _two_hot_target(y, self.bin_centers, self.config.hl_gauss_sigma)
        log_probs = F.log_softmax(logits_online, dim=-1)
        td_ce_per_sample = -(target_probs * log_probs).sum(dim=-1)
        loss = td_ce_per_sample.mean()

        with torch.no_grad():
            q_pred = _expected_value(logits_online, self.bin_centers)
            td_error = (q_pred - y).abs()
            # Return Python floats (not 0-dim tensors) so LeRobot's WandB wrapper
            # actually logs them — its scalar-type check rejects torch.Tensor.
            loss_dict = {
                "td_ce_loss": loss.detach().item(),
                "q_pred_mean": q_pred.mean().item(),
                "q_pred_min": q_pred.min().item(),
                "q_pred_max": q_pred.max().item(),
                "q_target_mean": y.mean().item(),
                "q_target_min": y.min().item(),
                "q_target_max": y.max().item(),
                "td_abs_error_mean": td_error.mean().item(),
                "bootstrap_frac": bootstrap_valid.mean().item(),
                "reward_sum_mean": discounted_rewards.mean().item(),
                "tau": float(self.config.target_tau),
            }
            # Per-bucket diagnostics: without these, a shortcut (e.g. play-style
            # detection) is invisible in the aggregate metrics.
            bucket_idx = batch.get(BUCKET_INDEX)
            if bucket_idx is not None and self.config.reward_mode != "all_success":
                from lerobot.policies.q_function.q_value_labels import DEFAULT_BUCKET_ORDER

                for ordinal, name in enumerate(DEFAULT_BUCKET_ORDER):
                    m = bucket_idx.view(-1) == ordinal
                    if m.any():
                        loss_dict[f"bucket_{name}/td_ce_loss"] = td_ce_per_sample[m].mean().item()
                        loss_dict[f"bucket_{name}/q_pred_mean"] = q_pred[m].mean().item()
                        loss_dict[f"bucket_{name}/q_target_mean"] = y[m].mean().item()
                        loss_dict[f"bucket_{name}/frac"] = m.float().mean().item()

        # ── Ranking-margin loss on batch-internal synthetic negatives ────────
        if self.config.neg_margin_weight > 0:
            margin_loss, neg_metrics = self._negatives_margin_loss(
                context, a_first, logits_online, batch.get(BUCKET_INDEX),
                a_full=self._truncate_action(actions),
                episode_frac=batch.get("q_episode_frac"),
                episode_id=batch.get("q_episode_id"),
                bootstrap_valid=batch.get("q_bootstrap_valid"),
            )
            if margin_loss is not None:
                loss = loss + self.config.neg_margin_weight * margin_loss
                loss_dict["margin_loss"] = float(margin_loss.detach().item())
                loss_dict.update(neg_metrics)

        return loss, loss_dict

    def _negatives_margin_loss(
        self,
        context: Tensor,
        a_first: Tensor,
        logits_online: Tensor,
        bucket_idx: Tensor | None,
        a_full: Tensor | None = None,
        episode_frac: Tensor | None = None,
        episode_id: Tensor | None = None,
        bootstrap_valid: Tensor | None = None,
    ) -> tuple[Tensor | None, dict[str, float]]:
        """Ranking margin: relu(δ + Q(s, a_neg) − Q(s, a_true)), demo buckets only.

        Negative chunks are built batch-internally in the (normalized) action
        space and scored by reusing ``context`` — one extra decoder pass, no
        extra image encoding. Returns (None, {}) when no eligible samples.
        """
        cfg = self.config
        B = a_first.shape[0]

        if bucket_idx is not None and cfg.reward_mode != "all_success":
            from lerobot.policies.q_function.q_value_labels import DEFAULT_BUCKET_ORDER

            allowed = {
                ordinal for ordinal, name in enumerate(DEFAULT_BUCKET_ORDER) if name in cfg.neg_buckets
            }
            mask = torch.tensor(
                [int(b) in allowed for b in bucket_idx.view(-1).tolist()], device=a_first.device
            )
        else:
            mask = torch.ones(B, dtype=torch.bool, device=a_first.device)
        if not mask.any():
            return None, {}

        a_pos = a_first[mask]                                # (M, h, A)
        M = a_pos.shape[0]
        negs: list[Tensor] = []
        names: list[str] = []

        if cfg.neg_use_swap and M > 1:
            # Wrong-chunk negative: another sample's true chunk (≠ episode/task/
            # state w.h.p. under a shuffled loader) at THIS state.
            negs.append(torch.roll(a_pos, shifts=1, dims=0))
            names.append("swap")

        if cfg.neg_use_matched_swap and M > 1 and episode_frac is not None:
            # Fraction-MATCHED swap: for each anchor, the batch chunk at the
            # nearest episode fraction from a DIFFERENT episode — same task
            # phase, wrong state. The hardest wrong-chunk negative; directly
            # targets the G2 frame-matched swap probe.
            frac = episode_frac.view(-1)[mask].to(a_pos.device)          # (M,)
            dist = (frac.unsqueeze(0) - frac.unsqueeze(1)).abs()         # (M, M)
            dist.fill_diagonal_(torch.inf)
            if episode_id is not None:
                eid = episode_id.view(-1)[mask].to(a_pos.device)
                same_ep = eid.unsqueeze(0) == eid.unsqueeze(1)
                same_ep.fill_diagonal_(False)
                dist_cross = dist.masked_fill(same_ep, torch.inf)
                # Fall back to same-episode pairing only for rows with no
                # cross-episode partner in the batch (rare at batch >= 8).
                has_cross = torch.isfinite(dist_cross).any(dim=1)
                dist = torch.where(has_cross.unsqueeze(1), dist_cross, dist)
            j = dist.argmin(dim=1)                                       # (M,)
            negs.append(a_pos[j])
            names.append("mswap")

        if cfg.neg_tube_sigmas:
            from lerobot.policies.act_simple.planning import _smooth_time

            for s in cfg.neg_tube_sigmas:
                noise = torch.randn_like(a_pos) * float(s)
                if cfg.neg_tube_smooth_sigma_t > 0:
                    noise = _smooth_time(noise, cfg.neg_tube_smooth_sigma_t)
                noise[..., -1] = 0.0  # hold the gripper dim
                negs.append(a_pos + noise)
                names.append(f"tube{s:g}s")

        if cfg.neg_use_temporal:
            negs.append(torch.flip(a_pos, dims=[1]))
            names.append("trev")

        if cfg.neg_use_shift and a_full is not None:
            # "Right actions, wrong time": the true trajectory's chunk starting k
            # frames later, free from the 2h action window. Where the shifted
            # window would cross the episode end (≈ bootstrap_valid False, so the
            # window is padding-heavy), fall back to the true chunk — that pair
            # contributes ~0 margin instead of a corrupt negative.
            h = self.config.h
            a_full_m = a_full[mask]                                      # (M, 2h, A')
            if bootstrap_valid is not None:
                ok = bootstrap_valid.view(-1)[mask].to(a_pos.device).bool()
            else:
                ok = torch.ones(M, dtype=torch.bool, device=a_pos.device)
            for k in cfg.neg_shift_frames:
                k = int(k)
                if k <= 0 or h + k > a_full_m.shape[1]:
                    continue
                shifted = a_full_m[:, k : k + h, :]
                shifted = torch.where(ok.view(-1, 1, 1), shifted, a_pos)
                negs.append(shifted)
                names.append(f"shift{k}")

        if not negs:
            return None, {}

        K = len(negs)
        ctx_pos = context[:, mask]                           # (S, M, D)
        ctx_rep = torch.cat([ctx_pos] * K, dim=1)            # (S, K·M, D)
        logits_neg = self.q_online.forward_with_context(ctx_rep, torch.cat(negs, dim=0))
        q_neg = _expected_value(logits_neg, self.bin_centers).view(K, M)
        q_pos = _expected_value(logits_online, self.bin_centers)[mask]   # differentiable

        margin = F.relu(cfg.neg_margin_delta + q_neg - q_pos.unsqueeze(0))
        metrics: dict[str, float] = {}
        with torch.no_grad():
            for k, name in enumerate(names):
                metrics[f"neg_{name}/q_mean"] = q_neg[k].mean().item()
                metrics[f"neg_{name}/rank_acc"] = (q_pos > q_neg[k]).float().mean().item()
        return margin.mean(), metrics

    @torch.no_grad()
    def encode_obs_context(self, preprocessed_batch: dict[str, Tensor]) -> Tensor:
        """Encode observation to context tokens (S, 1, D) for reuse across N planning candidates.

        preprocessed_batch must have already been through q_pre (NormalizerProcessorStep runs on
        images too, so raw images give different DINOv2 inputs than q_pre-normalized images).
        Caller should run q_pre on a single-item obs batch before calling this.
        """
        self.eval()
        imgs = self._stack_camera_views(preprocessed_batch, delta_idx=0)  # (1, V, 3, H, W)
        text_tokens = self._encode_language(preprocessed_batch)
        return self.q_online.encode_context(imgs, text_tokens)             # (S, 1, D)

    @torch.no_grad()
    def predict_value(self, batch: dict[str, Tensor]) -> Tensor:
        """Return the scalar Q(s_t, a_{t:t+h}, l) prediction. Shape (B,)."""
        self.eval()
        actions = batch[ACTION]
        if actions.dim() == 3 and actions.shape[1] >= self.config.h:
            a_first = self._truncate_action(actions[:, : self.config.h, :])
        elif actions.dim() == 3 and actions.shape[1] == self.config.h:
            a_first = self._truncate_action(actions)
        else:
            raise RuntimeError(f"predict_value: bad action shape {tuple(actions.shape)}")
        imgs_t = self._stack_camera_views(batch, delta_idx=0)
        text_tokens = self._encode_language(batch)
        logits = self.q_online(imgs_t, a_first, text_tokens=text_tokens)
        return _expected_value(logits, self.bin_centers)

    # ── Inference interfaces required by PreTrainedPolicy ────────────────
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Q-function doesn't produce actions; return Q as a degenerate (B, 1, 1) chunk."""
        v = self.predict_value(batch)
        return v.view(-1, 1, 1)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        return self.predict_value(batch)
