"""Fresh state-value critic and TD(lambda) targets for FastWAM DAWR."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class FrozenDINOEncoder(nn.Module):
    """Frozen generic visual features for a newly initialized DAWR value head.

    This loads only the public DINOv2 base model.  It never loads a Q checkpoint
    or any weights trained on the RL task.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(
            model_name,
            local_files_only=True,
        ).to(device=device, dtype=dtype)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.output_dim = int(self.backbone.config.hidden_size)
        self.device = device
        self.dtype = dtype
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, V, 3, H, W)`` images into ``(B, V, D)`` CLS features."""
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"Expected images shaped (B, V, 3, H, W), got {tuple(images.shape)}")
        batch_size, n_views = images.shape[:2]
        pixels = images.flatten(0, 1).to(device=self.device, dtype=torch.float32)
        if tuple(pixels.shape[-2:]) != (224, 224):
            pixels = F.interpolate(
                pixels,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        pixels = ((pixels - self.pixel_mean) / self.pixel_std).to(dtype=self.dtype)
        output = self.backbone(pixel_values=pixels)
        features = output.last_hidden_state[:, 0].float()
        return features.reshape(batch_size, n_views, self.output_dim)

    def train(self, mode: bool = True):
        """Keep the frozen backbone deterministic when a parent module trains."""
        super().train(False)
        self.backbone.eval()
        return self


class DAWRValueCritic(nn.Module):
    """Task-conditioned state-value head over frozen multi-camera features."""

    def __init__(
        self,
        *,
        vision_dim: int,
        n_cameras: int,
        state_dim: int,
        n_tasks: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_cameras <= 0 or n_tasks <= 0:
            raise ValueError("DAWRValueCritic requires at least one camera and one task.")
        self.vision_dim = int(vision_dim)
        self.n_cameras = int(n_cameras)
        self.state_dim = int(state_dim)
        self.n_tasks = int(n_tasks)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        feature_dim = 256
        task_dim = 32
        state_feature_dim = 32
        self.vision_projection = nn.Sequential(
            nn.LayerNorm(self.vision_dim),
            nn.Linear(self.vision_dim, feature_dim),
            nn.GELU(),
        )
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, state_feature_dim),
            nn.GELU(),
        )
        self.task_embedding = nn.Embedding(self.n_tasks, task_dim)
        joint_dim = self.n_cameras * feature_dim + state_feature_dim + task_dim
        self.value_head = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        state: torch.Tensor,
        task_index: torch.Tensor,
    ) -> torch.Tensor:
        if vision_features.ndim != 3:
            raise ValueError(
                f"Expected vision features shaped (B, V, D), got {tuple(vision_features.shape)}"
            )
        if tuple(vision_features.shape[1:]) != (self.n_cameras, self.vision_dim):
            raise ValueError(
                "Value-critic vision shape mismatch: "
                f"expected (*, {self.n_cameras}, {self.vision_dim}), "
                f"got {tuple(vision_features.shape)}"
            )
        vision = self.vision_projection(vision_features).flatten(1)
        state_features = self.state_encoder(state)
        task_features = self.task_embedding(task_index.long())
        return self.value_head(
            torch.cat([vision, state_features, task_features], dim=-1)
        ).squeeze(-1)

    def checkpoint_config(self) -> dict[str, int | float]:
        return {
            "vision_dim": self.vision_dim,
            "n_cameras": self.n_cameras,
            "state_dim": self.state_dim,
            "n_tasks": self.n_tasks,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }


@dataclass
class EncodedDAWRReplay:
    """CPU-resident frozen observations and trajectory labels."""

    vision_features: torch.Tensor
    state: torch.Tensor
    task_index: torch.Tensor
    reward: torch.Tensor
    terminal: torch.Tensor
    trajectory_index: torch.Tensor

    def __len__(self) -> int:
        return len(self.reward)


@torch.no_grad()
def td_lambda_returns(
    rewards: torch.Tensor,
    terminals: torch.Tensor,
    trajectory_indices: torch.Tensor,
    values: torch.Tensor,
    *,
    gamma: float = 0.99,
    td_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute standard GAE advantages and TD(lambda) value targets.

    Inputs must be in chronological replay order.  A trajectory boundary or a
    terminal transition resets the recursion, preventing return leakage between
    independently reset LIBERO episodes.
    """
    rewards = rewards.detach().cpu().float().flatten()
    terminals = terminals.detach().cpu().bool().flatten()
    trajectory_indices = trajectory_indices.detach().cpu().long().flatten()
    values = values.detach().cpu().float().flatten()
    n_steps = len(rewards)
    if not (len(terminals) == len(trajectory_indices) == len(values) == n_steps):
        raise ValueError("TD(lambda) inputs must have identical lengths.")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= td_lambda <= 1.0:
        raise ValueError("gamma and td_lambda must lie in [0, 1].")

    advantages = torch.zeros_like(rewards)
    next_advantage = torch.tensor(0.0)
    for index in range(n_steps - 1, -1, -1):
        has_same_next = (
            index + 1 < n_steps
            and trajectory_indices[index + 1] == trajectory_indices[index]
            and not terminals[index]
        )
        if has_same_next:
            next_value = values[index + 1]
            continuation = 1.0
        else:
            next_value = torch.tensor(0.0)
            next_advantage = torch.tensor(0.0)
            continuation = 0.0
        delta = rewards[index] + gamma * next_value * continuation - values[index]
        advantages[index] = delta + gamma * td_lambda * continuation * next_advantage
        next_advantage = advantages[index]
    return advantages, advantages + values
