#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Online planning algorithms for AWM.

A planner refines an initial (BC warm-start) action trajectory by repeatedly
evaluating sampled perturbations through a world-model cost function.

Interface
---------
BasePlanner.optimize(initial_actions, cost_fn, action_lows, action_highs) → Tensor

``cost_fn`` is a callable produced by ``AWM.make_wm_cost_fn`` that takes
``(N, T, D)`` candidate action trajectories and returns an ``(N,)`` cost tensor.
Lower cost = better trajectory.

Adding a new algorithm
----------------------
1. Subclass ``BasePlanner`` and implement ``optimize()``.
2. Add a branch in ``build_planner()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

import torch
from torch import Tensor

if TYPE_CHECKING:
    from lerobot.policies.awm.configuration_awm import AWMConfig


class BasePlanner(ABC):
    """Abstract interface for trajectory planners.

    Args:
        initial_actions: ``(T, D)`` BC warm-start action trajectory.
        cost_fn: ``(N, T, D) → (N,)`` — evaluates N candidate trajectories, returns
            scalar cost per trajectory (lower is better).
        action_lows:  ``(D,)`` per-dimension lower bounds for clamping.
        action_highs: ``(D,)`` per-dimension upper bounds for clamping.

    Returns:
        ``(T, D)`` refined action trajectory.
    """

    @abstractmethod
    def optimize(
        self,
        initial_actions: Tensor,
        cost_fn: Callable[[Tensor], Tensor],
        action_lows: Tensor,
        action_highs: Tensor,
    ) -> Tensor:
        ...


class MPPIPlanner(BasePlanner):
    """Model Predictive Path Integral (MPPI) trajectory optimiser.

    At each iteration, ``n_samples`` Gaussian-perturbed copies of the current
    best trajectory are evaluated by ``cost_fn``.  A softmin-weighted average
    (lower cost → higher weight) produces the updated trajectory.

    Args:
        n_samples:    Number of perturbed trajectories sampled per iteration.
        n_iters:      Number of refinement iterations.
        noise_sigma:  Standard deviation of Gaussian action perturbations in
                      continuous action space.
        temperature:  Inverse temperature λ.  Higher values → sharper selection
                      (winner-takes-most); lower → more averaging.
    """

    def __init__(self, n_samples: int, n_iters: int, noise_sigma: float, temperature: float):
        self.n_samples = n_samples
        self.n_iters = n_iters
        self.noise_sigma = noise_sigma
        self.temperature = temperature

    def optimize(
        self,
        initial_actions: Tensor,
        cost_fn: Callable[[Tensor], Tensor],
        action_lows: Tensor,
        action_highs: Tensor,
    ) -> Tensor:
        """Run MPPI and return the refined trajectory.

        Args:
            initial_actions: ``(T, D)`` warm-start trajectory.
            cost_fn:         ``(N, T, D) → (N,)`` world-model cost.
            action_lows:     ``(D,)`` lower bounds.
            action_highs:    ``(D,)`` upper bounds.

        Returns:
            ``(T, D)`` refined trajectory, clamped to ``[action_lows, action_highs]``.
        """
        actions = initial_actions.clone()  # (T, D)

        for _ in range(self.n_iters):
            # Sample N Gaussian perturbations around the current best trajectory.
            noise = torch.randn(self.n_samples, *actions.shape, device=actions.device) * self.noise_sigma
            perturbed = (actions.unsqueeze(0) + noise).clamp(action_lows, action_highs)  # (N, T, D)

            costs = cost_fn(perturbed)  # (N,)

            # Numerically stable softmin: shift by min before exponentiation.
            costs_shifted = costs - costs.min()
            weights = torch.exp(-self.temperature * costs_shifted)
            weights = weights / weights.sum()  # (N,)

            # Weighted mean over sampled trajectories.
            actions = (weights[:, None, None] * perturbed).sum(dim=0)  # (T, D)
            actions = actions.clamp(action_lows, action_highs)

        return actions


def build_planner(config: AWMConfig) -> BasePlanner:
    """Factory: construct a planner from an ``AWMConfig``.

    Extend this function when adding new planning algorithms.
    """
    if config.planning_type == "mppi":
        return MPPIPlanner(
            n_samples=config.planning_n_samples,
            n_iters=config.planning_n_iters,
            noise_sigma=config.planning_noise_sigma,
            temperature=config.planning_temperature,
        )
    raise ValueError(
        f"Unknown planning_type {config.planning_type!r}. "
        "Add a branch in planning_awm.build_planner() to support it."
    )
