from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class WMLatentImageDecoder(nn.Module):
    """Detached debug decoder: WAN VAE image latents -> RGB image."""

    def __init__(self, latent_channels: int, image_shape: tuple[int, int, int], latent_hw: tuple[int, int]):
        super().__init__()
        out_channels, height, width = image_shape
        self.latent_hw = (int(latent_hw[0]), int(latent_hw[1]))
        hidden_channels = [128, 64, 32]
        layers: list[nn.Module] = [
            nn.Conv2d(latent_channels, hidden_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            ResBlock2d(hidden_channels[0]),
        ]
        for in_ch, out_ch in zip(hidden_channels[:-1], hidden_channels[1:]):
            layers.extend(
                [
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    ResBlock2d(out_ch),
                ]
            )
        layers.append(nn.Conv2d(hidden_channels[-1], out_channels, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*layers)
        self.image_shape = (out_channels, height, width)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        image = self.decoder(latents)
        if image.shape[-2:] != self.image_shape[1:]:
            image = F.interpolate(
                image,
                size=self.image_shape[1:],
                mode="bilinear",
                align_corners=False,
            )
        return image


class WMDecoderLayer(nn.Module):
    def __init__(self, dim_model: int, n_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, cross_kv: torch.Tensor, cross_pos: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.norm1(x)
        x = self.self_attn(x, x, value=x, need_weights=False)[0]
        x = skip + self.dropout1(x)

        skip = x
        x = self.norm2(x)
        x = self.cross_attn(
            query=x,
            key=cross_kv + cross_pos,
            value=cross_kv,
            need_weights=False,
        )[0]
        x = skip + self.dropout2(x)

        skip = x
        x = self.norm3(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return skip + self.dropout3(x)


class WMDecoder(nn.Module):
    def __init__(self, dim_model: int, n_heads: int, dim_feedforward: int, n_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [WMDecoderLayer(dim_model, n_heads, dim_feedforward, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, x: torch.Tensor, cross_kv: torch.Tensor, cross_pos: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cross_kv, cross_pos)
        return self.norm(x)


class FastWAMLatentWMHead(nn.Module):
    """AWM-style transformer head over WAN VAE image latents."""

    def __init__(
        self,
        latent_channels: int,
        latent_hw: tuple[int, int],
        action_dim: int,
        action_horizon: int,
        patch_size: tuple[int, int],
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        n_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        latent_h, latent_w = latent_hw
        patch_h, patch_w = patch_size
        if latent_h % patch_h != 0 or latent_w % patch_w != 0:
            raise ValueError(
                "Latent spatial shape must be divisible by WM patch size, "
                f"got ({latent_h}, {latent_w}) vs ({patch_h}, {patch_w})."
            )

        self.latent_channels = int(latent_channels)
        self.latent_hw = (int(latent_h), int(latent_w))
        self.patch_size = (int(patch_h), int(patch_w))
        self.tokens_per_image = (latent_h // patch_h) * (latent_w // patch_w)
        self.patch_dim = self.latent_channels * patch_h * patch_w

        self.latent_in_proj = nn.Linear(self.patch_dim, dim_model)
        self.action_proj = nn.Linear(action_dim, dim_model)
        self.decoder = WMDecoder(
            dim_model=dim_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.latent_out_proj = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.GELU(),
            nn.Linear(dim_model, self.patch_dim),
        )

        self.action_pos_embed = nn.Embedding(action_horizon, dim_model)
        self.cross_pos_embed = nn.Embedding(self.tokens_per_image, dim_model)
        self.query_pos_embed = nn.Embedding(self.tokens_per_image, dim_model)
        self.query_tokens = nn.Parameter(torch.zeros(self.tokens_per_image, 1, dim_model))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

    def _patchify(self, latents: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = latents.shape
        if channels != self.latent_channels or (height, width) != self.latent_hw:
            raise ValueError(
                "WM head latent shape mismatch: "
                f"expected [B, {self.latent_channels}, {self.latent_hw[0]}, {self.latent_hw[1]}], "
                f"got {tuple(latents.shape)}."
            )
        patch_h, patch_w = self.patch_size
        latents = latents.view(
            bsz,
            channels,
            height // patch_h,
            patch_h,
            width // patch_w,
            patch_w,
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5).contiguous()
        return latents.view(bsz, self.tokens_per_image, self.patch_dim)

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz = tokens.shape[0]
        latent_h, latent_w = self.latent_hw
        patch_h, patch_w = self.patch_size
        tokens = tokens.view(
            bsz,
            latent_h // patch_h,
            latent_w // patch_w,
            self.latent_channels,
            patch_h,
            patch_w,
        )
        tokens = tokens.permute(0, 3, 1, 4, 2, 5).contiguous()
        return tokens.view(bsz, self.latent_channels, latent_h, latent_w)

    def forward(
        self,
        current_latents: torch.Tensor,
        actions: torch.Tensor,
        action_is_pad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if current_latents.ndim != 4:
            raise ValueError(
                f"`current_latents` must be [B, C, H, W], got {tuple(current_latents.shape)}."
            )
        if actions.ndim != 3:
            raise ValueError(f"`actions` must be [B, T, A], got {tuple(actions.shape)}.")
        if actions.shape[1] > self.action_pos_embed.num_embeddings:
            raise ValueError(
                "Action horizon exceeds WM head positional capacity: "
                f"{actions.shape[1]} > {self.action_pos_embed.num_embeddings}."
            )

        if action_is_pad is not None:
            actions = actions.masked_fill(action_is_pad.unsqueeze(-1), 0)

        batch_size = current_latents.shape[0]
        latent_tokens = self._patchify(current_latents)
        cross_kv = self.latent_in_proj(latent_tokens).transpose(0, 1)
        cross_pos = self.cross_pos_embed.weight[: self.tokens_per_image].unsqueeze(1)

        action_tokens = self.action_proj(actions).transpose(0, 1)
        action_tokens = action_tokens + self.action_pos_embed.weight[: actions.shape[1]].unsqueeze(1)

        query_pos = self.query_pos_embed.weight[: self.tokens_per_image].unsqueeze(1)
        query_tokens = (self.query_tokens[: self.tokens_per_image] + query_pos).expand(-1, batch_size, -1)
        wm_in = torch.cat([query_tokens, action_tokens], dim=0)

        wm_out = self.decoder(wm_in, cross_kv, cross_pos)
        pred_tokens = self.latent_out_proj(wm_out[: self.tokens_per_image].transpose(0, 1))
        return self._unpatchify(pred_tokens)


def compute_image_reconstruction_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    prefix: str,
    valid_mask: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    if valid_mask is not None:
        if not valid_mask.any():
            return {}
        pred = pred[valid_mask]
        target = target[valid_mask]
    mse = F.mse_loss(pred.float(), target.float())
    psnr = -10.0 * torch.log10(mse.clamp(min=1e-8))
    return {f"{prefix}/mse": float(mse.item()), f"{prefix}/psnr": float(psnr.item())}


def q_two_hot_target(values: torch.Tensor, bin_centers: torch.Tensor, sigma: float) -> torch.Tensor:
    values = values.clamp(bin_centers[0], bin_centers[-1]).unsqueeze(-1)
    d = (bin_centers.unsqueeze(0) - values) / sigma
    logits = -0.5 * d.pow(2)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    probs = torch.exp(logits)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def q_expected_value(logits: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return (probs * bin_centers).sum(dim=-1)


class FastWAMLatentQFunction(nn.Module):
    """Chunk-value critic over WAN VAE latents and action chunks."""

    def __init__(
        self,
        latent_channels: int,
        latent_hw: tuple[int, int],
        action_dim: int,
        action_horizon: int,
        patch_size: tuple[int, int],
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        n_layers: int,
        num_bins: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        latent_h, latent_w = latent_hw
        patch_h, patch_w = patch_size
        if latent_h % patch_h != 0 or latent_w % patch_w != 0:
            raise ValueError(
                "Latent spatial shape must be divisible by Q-function patch size, "
                f"got ({latent_h}, {latent_w}) vs ({patch_h}, {patch_w})."
            )

        self.latent_channels = int(latent_channels)
        self.latent_hw = (int(latent_h), int(latent_w))
        self.patch_size = (int(patch_h), int(patch_w))
        self.tokens_per_image = (latent_h // patch_h) * (latent_w // patch_w)
        self.patch_dim = self.latent_channels * patch_h * patch_w

        self.latent_in_proj = nn.Linear(self.patch_dim, dim_model)
        self.action_proj = nn.Linear(action_dim, dim_model)
        self.decoder = WMDecoder(
            dim_model=dim_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.action_pos_embed = nn.Embedding(action_horizon, dim_model)
        self.cross_pos_embed = nn.Embedding(self.tokens_per_image, dim_model)
        self.cls_query = nn.Parameter(torch.zeros(1, 1, dim_model))
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, dim_model))
        self.head = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.GELU(),
            nn.Linear(dim_model, num_bins),
        )
        nn.init.trunc_normal_(self.cls_query, std=0.02)
        nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)

    def _patchify(self, latents: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = latents.shape
        if channels != self.latent_channels or (height, width) != self.latent_hw:
            raise ValueError(
                "Q-function latent shape mismatch: "
                f"expected [B, {self.latent_channels}, {self.latent_hw[0]}, {self.latent_hw[1]}], "
                f"got {tuple(latents.shape)}."
            )
        patch_h, patch_w = self.patch_size
        latents = latents.view(
            bsz,
            channels,
            height // patch_h,
            patch_h,
            width // patch_w,
            patch_w,
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5).contiguous()
        return latents.view(bsz, self.tokens_per_image, self.patch_dim)

    def forward(
        self,
        current_latents: torch.Tensor,
        actions: torch.Tensor,
        action_is_pad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if current_latents.ndim != 4:
            raise ValueError(
                f"`current_latents` must be [B, C, H, W], got {tuple(current_latents.shape)}."
            )
        if actions.ndim != 3:
            raise ValueError(f"`actions` must be [B, T, A], got {tuple(actions.shape)}.")
        if actions.shape[1] > self.action_pos_embed.num_embeddings:
            raise ValueError(
                "Action horizon exceeds Q-function positional capacity: "
                f"{actions.shape[1]} > {self.action_pos_embed.num_embeddings}."
            )

        if action_is_pad is not None:
            actions = actions.masked_fill(action_is_pad.unsqueeze(-1), 0)

        batch_size = current_latents.shape[0]
        latent_tokens = self._patchify(current_latents)
        cross_kv = self.latent_in_proj(latent_tokens).transpose(0, 1)
        cross_pos = self.cross_pos_embed.weight[: self.tokens_per_image].unsqueeze(1)

        action_tokens = self.action_proj(actions).transpose(0, 1)
        action_tokens = action_tokens + self.action_pos_embed.weight[: actions.shape[1]].unsqueeze(1)

        cls_query = self.cls_query.expand(-1, batch_size, -1)
        cls_pos = self.cls_pos_embed.expand(-1, batch_size, -1)
        q_in = torch.cat([cls_query + cls_pos, action_tokens], dim=0)
        q_out = self.decoder(q_in, cross_kv, cross_pos)
        return self.head(q_out[0])
