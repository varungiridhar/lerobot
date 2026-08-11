"""Minimal TD3 core for imitation-bootstrapped reinforcement learning.

The behavior-cloned policy is deliberately external to this module.  It stays
frozen and supplies one action proposal per observation.  TD3 supplies the
other proposal, and target twin-Q values choose which proposal is executed and
which proposal bootstraps the critic target.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def freeze_behavior_policy(policy: nn.Module) -> nn.Module:
    """Freeze a behavior policy used only to generate IBRL proposals."""
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return policy


def _small_last_layer(module: nn.Sequential) -> None:
    last = module[-1]
    if not isinstance(last, nn.Linear):
        raise TypeError("Expected the final MLP layer to be linear.")
    nn.init.uniform_(last.weight, -3e-3, 3e-3)
    nn.init.uniform_(last.bias, -3e-3, 3e-3)


@contextmanager
def _temporary_training_mode(module: nn.Module, mode: bool) -> Iterator[None]:
    """Temporarily set a module's training mode without leaking rollout state."""
    previous_mode = module.training
    module.train(mode)
    try:
        yield
    finally:
        module.train(previous_mode)


class TD3Actor(nn.Module):
    """Tanh actor with dropout regularization that can be disabled for rollout."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        actor_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Actor dimensions must be positive.")
        if not 0.0 <= actor_dropout < 1.0:
            raise ValueError("actor_dropout must lie in [0, 1).")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.actor_dropout = float(actor_dropout)
        self.network = nn.Sequential(
            nn.LayerNorm(self.observation_dim),
            nn.Linear(self.observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.actor_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.actor_dropout),
            nn.Linear(hidden_dim, self.action_dim),
        )
        _small_last_layer(self.network)

    def forward(self, observation: Tensor) -> Tensor:
        return torch.tanh(self.network(observation))


class _QNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.observation_norm = nn.LayerNorm(observation_dim)
        self.network = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: Tensor, action: Tensor) -> Tensor:
        normalized_observation = self.observation_norm(observation)
        return self.network(torch.cat((normalized_observation, action), dim=-1))


class TwinQCritic(nn.Module):
    """Independent twin action-value functions used conservatively via ``min``."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Critic dimensions must be positive.")
        self.q1 = _QNetwork(observation_dim, action_dim, hidden_dim)
        self.q2 = _QNetwork(observation_dim, action_dim, hidden_dim)

    def forward(self, observation: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        return self.q1(observation, action), self.q2(observation, action)

    def conservative(self, observation: Tensor, action: Tensor) -> Tensor:
        q1, q2 = self(observation, action)
        return torch.minimum(q1, q2)


@dataclass(frozen=True)
class IBRLBatch:
    observations: Tensor
    actions: Tensor
    rewards: Tensor
    next_observations: Tensor
    dones: Tensor
    discounts: Tensor
    next_bc_actions: Tensor

    def __post_init__(self) -> None:
        for name in ("observations", "actions", "next_observations", "next_bc_actions"):
            tensor = getattr(self, name)
            if tensor.ndim != 2:
                raise ValueError(f"{name} must be a rank-2 batched tensor, got {tuple(tensor.shape)}.")
        batch_size = self.observations.shape[0]
        for name in ("actions", "next_observations", "next_bc_actions"):
            if getattr(self, name).shape[0] != batch_size:
                raise ValueError(f"{name} has a different batch size from observations.")
        for name in ("rewards", "dones", "discounts"):
            tensor = getattr(self, name)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1)
                object.__setattr__(self, name, tensor)
            if tensor.shape != (batch_size, 1):
                raise ValueError(
                    f"{name} must have shape ({batch_size}, 1), got {tuple(tensor.shape)}."
                )

    def to(self, device: torch.device | str) -> IBRLBatch:
        return IBRLBatch(
            observations=self.observations.to(device=device, dtype=torch.float32),
            actions=self.actions.to(device=device, dtype=torch.float32),
            rewards=self.rewards.to(device=device, dtype=torch.float32),
            next_observations=self.next_observations.to(device=device, dtype=torch.float32),
            dones=self.dones.to(device=device, dtype=torch.float32),
            discounts=self.discounts.to(device=device, dtype=torch.float32),
            next_bc_actions=self.next_bc_actions.to(device=device, dtype=torch.float32),
        )


class EncodedReplayBuffer:
    """Fixed-size CPU replay for frozen visual features and cached BC proposals."""

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        *,
        storage_dtype: torch.dtype = torch.float16,
    ) -> None:
        if capacity <= 0 or observation_dim <= 0 or action_dim <= 0:
            raise ValueError("Replay dimensions and capacity must be positive.")
        if storage_dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise ValueError("Replay storage_dtype must be a floating-point tensor dtype.")
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.observations = torch.empty(capacity, observation_dim, dtype=storage_dtype)
        self.actions = torch.empty(capacity, action_dim, dtype=storage_dtype)
        self.rewards = torch.empty(capacity, 1, dtype=torch.float32)
        self.next_observations = torch.empty(capacity, observation_dim, dtype=storage_dtype)
        self.dones = torch.empty(capacity, 1, dtype=torch.bool)
        self.discounts = torch.empty(capacity, 1, dtype=torch.float32)
        self.next_bc_actions = torch.empty(capacity, action_dim, dtype=storage_dtype)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    @torch.no_grad()
    def add(
        self,
        observation: Tensor,
        action: Tensor,
        reward: float,
        next_observation: Tensor,
        done: bool,
        discount: float,
        next_bc_action: Tensor,
    ) -> None:
        observation = observation.detach().cpu().flatten()
        action = action.detach().cpu().flatten()
        next_observation = next_observation.detach().cpu().flatten()
        next_bc_action = next_bc_action.detach().cpu().flatten()
        if observation.numel() != self.observation_dim:
            raise ValueError(
                f"Observation has {observation.numel()} elements; expected {self.observation_dim}."
            )
        if next_observation.numel() != self.observation_dim:
            raise ValueError(
                "Next observation has "
                f"{next_observation.numel()} elements; expected {self.observation_dim}."
            )
        if action.numel() != self.action_dim or next_bc_action.numel() != self.action_dim:
            raise ValueError(f"Replay actions must contain {self.action_dim} elements.")

        index = self.position
        self.observations[index].copy_(observation)
        self.actions[index].copy_(action)
        self.rewards[index, 0] = float(reward)
        self.next_observations[index].copy_(next_observation)
        self.dones[index, 0] = bool(done)
        self.discounts[index, 0] = float(discount)
        self.next_bc_actions[index].copy_(next_bc_action)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, *, generator: torch.Generator | None = None) -> IBRLBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.size < batch_size:
            raise ValueError(f"Replay contains {self.size} items, fewer than batch_size={batch_size}.")
        indices = torch.randint(self.size, (batch_size,), generator=generator)
        return IBRLBatch(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            dones=self.dones[indices],
            discounts=self.discounts[indices],
            next_bc_actions=self.next_bc_actions[indices],
        )

    def state_dict(self) -> dict[str, Any]:
        """Return a compact, exact replay snapshot suitable for phase resume.

        Only initialized rows are serialized.  Until the first wrap, replay
        writes occupy ``[:size]``; once full, ``size == capacity`` and the
        complete physical ring is stored together with its write position.
        Keeping the physical order is important because deterministic replay
        sampling addresses rows by index.
        """
        used = self.size
        return {
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "position": self.position,
            "size": self.size,
            "storage_dtype": self.observations.dtype,
            "observations": self.observations[:used].clone(),
            "actions": self.actions[:used].clone(),
            "rewards": self.rewards[:used].clone(),
            "next_observations": self.next_observations[:used].clone(),
            "dones": self.dones[:used].clone(),
            "discounts": self.discounts[:used].clone(),
            "next_bc_actions": self.next_bc_actions[:used].clone(),
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`state_dict` in-place."""
        for name, expected in (
            ("capacity", self.capacity),
            ("observation_dim", self.observation_dim),
            ("action_dim", self.action_dim),
        ):
            actual = int(state[name])
            if actual != expected:
                raise ValueError(
                    f"Replay {name} mismatch while resuming: "
                    f"checkpoint={actual}, current={expected}."
                )
        if state.get("storage_dtype", self.observations.dtype) != self.observations.dtype:
            raise ValueError(
                "Replay storage dtype mismatch while resuming: "
                f"checkpoint={state.get('storage_dtype')}, current={self.observations.dtype}."
            )
        size = int(state["size"])
        position = int(state["position"])
        if not 0 <= size <= self.capacity:
            raise ValueError(f"Invalid replay size in checkpoint: {size}.")
        if not 0 <= position < self.capacity:
            raise ValueError(f"Invalid replay position in checkpoint: {position}.")
        if size < self.capacity and position != size:
            raise ValueError(
                "A partially filled replay must have position == size; "
                f"got position={position}, size={size}."
            )
        tensors = {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "dones": self.dones,
            "discounts": self.discounts,
            "next_bc_actions": self.next_bc_actions,
        }
        for name, destination in tensors.items():
            source = state[name]
            expected_shape = (size, *destination.shape[1:])
            if tuple(source.shape) != expected_shape:
                raise ValueError(
                    f"Replay tensor {name!r} has shape {tuple(source.shape)}; "
                    f"expected {expected_shape}."
                )
            destination[:size].copy_(source)
        self.position = position
        self.size = size


@dataclass(frozen=True)
class HybridActionSelection:
    action: Tensor
    rl_action: Tensor
    bc_selected: Tensor
    bc_q: Tensor
    rl_q: Tensor


class IBRLTD3(nn.Module):
    """TD3 with IBRL proposal selection and IBRL critic bootstrapping."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 512,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        tau: float = 0.01,
        policy_delay: int = 2,
        target_noise: float = 0.1,
        target_noise_clip: float = 0.3,
        actor_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must lie in (0, 1].")
        if policy_delay <= 0:
            raise ValueError("policy_delay must be positive.")
        if actor_lr <= 0.0 or critic_lr <= 0.0:
            raise ValueError("Actor and critic learning rates must be positive.")
        if target_noise < 0.0 or target_noise_clip < 0.0:
            raise ValueError("Target noise parameters cannot be negative.")
        if not 0.0 <= actor_dropout < 1.0:
            raise ValueError("actor_dropout must lie in [0, 1).")

        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.tau = float(tau)
        self.policy_delay = int(policy_delay)
        self.target_noise = float(target_noise)
        self.target_noise_clip = float(target_noise_clip)
        self.actor_dropout = float(actor_dropout)

        self.actor = TD3Actor(
            observation_dim,
            action_dim,
            hidden_dim,
            actor_dropout=self.actor_dropout,
        )
        self.critic = TwinQCritic(observation_dim, action_dim, hidden_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        for target in (self.actor_target, self.critic_target):
            target.requires_grad_(False)
            target.eval()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.total_updates = 0

    def train(self, mode: bool = True):
        """Keep frozen target networks in eval mode when the learner trains."""
        super().train(mode)
        self.actor_target.eval()
        self.critic_target.eval()
        return self

    @torch.no_grad()
    def select_hybrid_action(
        self,
        observation: Tensor,
        bc_action: Tensor,
        *,
        exploration_noise: float = 0.0,
        noise: Tensor | None = None,
    ) -> HybridActionSelection:
        """Choose BC or exploratory RL proposal using conservative target-Q values."""
        # Rollout proposals must be deterministic with respect to actor dropout.
        # Exploration, when requested, is injected explicitly below.
        with _temporary_training_mode(self.actor, False):
            rl_action = self.actor(observation)
        if noise is None and exploration_noise > 0.0:
            noise = torch.randn_like(rl_action) * exploration_noise
        if noise is not None:
            rl_action = rl_action + noise.to(device=rl_action.device, dtype=rl_action.dtype)
        rl_action = rl_action.clamp(-1.0, 1.0)
        bc_action = bc_action.clamp(-1.0, 1.0)

        bc_q = self.critic_target.conservative(observation, bc_action)
        rl_q = self.critic_target.conservative(observation, rl_action)
        # Released hard IBRL stacks [RL, BC] before argmax, so exact ties choose
        # RL. Using a strict comparison reproduces that deterministic ordering.
        bc_selected = bc_q > rl_q
        action = torch.where(bc_selected, bc_action, rl_action)
        return HybridActionSelection(
            action=action,
            rl_action=rl_action,
            bc_selected=bc_selected,
            bc_q=bc_q,
            rl_q=rl_q,
        )

    @torch.no_grad()
    def bellman_target(self, batch: IBRLBatch, *, noise: Tensor | None = None) -> Tensor:
        """Compute the IBRL target: max over BC/RL proposals after twin-Q min."""
        # IBRL's actor regularization also applies to the target proposal used
        # for critic bootstrapping.  Keep the target actor in eval mode outside
        # this narrowly scoped forward pass so rollout state cannot leak.
        with _temporary_training_mode(self.actor_target, True):
            next_rl_action = self.actor_target(batch.next_observations)
        if noise is None:
            noise = torch.randn_like(next_rl_action) * self.target_noise
        noise = noise.to(device=next_rl_action.device, dtype=next_rl_action.dtype)
        noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
        next_rl_action = (next_rl_action + noise).clamp(-1.0, 1.0)
        next_bc_action = batch.next_bc_actions.clamp(-1.0, 1.0)

        bc_value = self.critic_target.conservative(batch.next_observations, next_bc_action)
        rl_value = self.critic_target.conservative(batch.next_observations, next_rl_action)
        next_value = torch.maximum(bc_value, rl_value)
        return batch.rewards + (1.0 - batch.dones) * batch.discounts * next_value

    def update(self, batch: IBRLBatch, *, max_grad_norm: float | None = None) -> dict[str, Any]:
        """Run one critic update and, at TD3's delay, one actor/target update."""
        if batch.observations.shape[-1] != self.observation_dim:
            raise ValueError("IBRL batch observation dimension does not match the agent.")
        if batch.actions.shape[-1] != self.action_dim:
            raise ValueError("IBRL batch action dimension does not match the agent.")

        target = self.bellman_target(batch)
        q1, q2 = self.critic(batch.observations, batch.actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self.critic_optimizer.step()
        self.soft_update_critic_target()

        self.total_updates += 1
        actor_loss_value: float | None = None
        if self.total_updates % self.policy_delay == 0:
            self.critic.requires_grad_(False)
            with _temporary_training_mode(self.actor, True):
                actor_action = self.actor(batch.observations)
            actor_loss = -self.critic.conservative(batch.observations, actor_action).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
            self.actor_optimizer.step()
            self.critic.requires_grad_(True)
            self.soft_update_actor_target()
            actor_loss_value = float(actor_loss.detach().item())

        return {
            "critic_loss": float(critic_loss.detach().item()),
            "actor_loss": actor_loss_value,
            "target_mean": float(target.mean().item()),
            "q1_mean": float(q1.detach().mean().item()),
            "q2_mean": float(q2.detach().mean().item()),
            "updates": self.total_updates,
        }

    @torch.no_grad()
    def _soft_update(self, online: nn.Module, target: nn.Module) -> None:
        for online_parameter, target_parameter in zip(
            online.parameters(), target.parameters(), strict=True
        ):
            target_parameter.lerp_(online_parameter, self.tau)

    @torch.no_grad()
    def soft_update_actor_target(self) -> None:
        self._soft_update(self.actor, self.actor_target)

    @torch.no_grad()
    def soft_update_critic_target(self) -> None:
        self._soft_update(self.critic, self.critic_target)

    @torch.no_grad()
    def soft_update_targets(self) -> None:
        """Update both targets, primarily for explicit synchronization/checks."""
        self.soft_update_actor_target()
        self.soft_update_critic_target()

    def checkpoint(self) -> dict[str, Any]:
        return {
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "tau": self.tau,
            "policy_delay": self.policy_delay,
            "target_noise": self.target_noise,
            "target_noise_clip": self.target_noise_clip,
            "actor_dropout": self.actor_dropout,
            "model_state_dict": self.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "total_updates": self.total_updates,
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore model, target, optimizer, and delayed-update state."""
        for name, expected in (
            ("observation_dim", self.observation_dim),
            ("action_dim", self.action_dim),
            ("hidden_dim", self.hidden_dim),
            ("policy_delay", self.policy_delay),
        ):
            actual = int(checkpoint[name])
            if actual != expected:
                raise ValueError(
                    f"TD3 {name} mismatch while resuming: checkpoint={actual}, current={expected}."
                )
        for name, expected in (
            ("tau", self.tau),
            ("target_noise", self.target_noise),
            ("target_noise_clip", self.target_noise_clip),
            ("actor_dropout", self.actor_dropout),
        ):
            actual = float(checkpoint[name])
            if actual != expected:
                raise ValueError(
                    f"TD3 {name} mismatch while resuming: checkpoint={actual}, current={expected}."
                )
        self.load_state_dict(checkpoint["model_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.total_updates = int(checkpoint["total_updates"])
        # Preserve the invariant enforced by train(): target nets never enter
        # rollout/train mode even if a checkpoint was captured from agent.train().
        self.actor_target.requires_grad_(False)
        self.critic_target.requires_grad_(False)
        self.actor_target.eval()
        self.critic_target.eval()
