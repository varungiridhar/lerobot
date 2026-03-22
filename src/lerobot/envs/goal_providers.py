"""Goal observation providers for planning-enabled policies.

Adding support for a new environment
-------------------------------------
1. Subclass ``GoalObservationProvider`` and implement ``get_goal_raw_obs()``.
2. Add a detection branch in ``make_goal_provider()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np


class GoalObservationProvider(ABC):
    """Returns a raw (numpy, B=1) observation dict representing the goal state.

    The dict must contain the same keys as the environment's normal reset observation
    (e.g. ``"pixels"``, ``"agent_pos"``), with a leading batch dimension of 1.
    """

    @abstractmethod
    def get_goal_raw_obs(self, env: gym.vector.VectorEnv) -> dict[str, np.ndarray]:
        ...


class PushTGoalProvider(GoalObservationProvider):
    """Goal provider for PushT: temporarily moves the block to goal_pose and renders."""

    def get_goal_raw_obs(self, env: gym.vector.VectorEnv) -> dict[str, np.ndarray]:
        pusht = env.envs[0].unwrapped  # PushTEnv instance
        goal_pose = pusht.goal_pose  # [x, y, angle] — fixed [256, 256, π/4]

        # Save current physics state (positions + velocities).
        saved_state = np.array([
            pusht.agent.position[0],
            pusht.agent.position[1],
            pusht.block.position[0],
            pusht.block.position[1],
            pusht.block.angle,
        ])
        saved_vel_agent = (pusht.agent.velocity[0], pusht.agent.velocity[1])
        saved_vel_block = (pusht.block.velocity[0], pusht.block.velocity[1])
        saved_avel_block = pusht.block.angular_velocity

        # Zero velocities so that space.step() inside _set_state does not drift
        # the bodies away from the positions we set (residual velocities from PD control).
        pusht.agent.velocity = (0.0, 0.0)
        pusht.block.velocity = (0.0, 0.0)
        pusht.block.angular_velocity = 0.0

        # Set block to goal pose; keep agent at default reset position (256, 400).
        pusht._set_state(np.array([256.0, 400.0, goal_pose[0], goal_pose[1], goal_pose[2]]))
        raw_obs = pusht.get_obs()  # {"pixels": (H,W,3), "agent_pos": (2,)}

        # Zero velocities again before restoring so the restore step doesn't drift.
        pusht.agent.velocity = (0.0, 0.0)
        pusht.block.velocity = (0.0, 0.0)
        pusht.block.angular_velocity = 0.0

        # Restore original state and velocities.
        pusht._set_state(saved_state)
        pusht.agent.velocity = saved_vel_agent
        pusht.block.velocity = saved_vel_block
        pusht.block.angular_velocity = saved_avel_block

        return {
            "pixels": raw_obs["pixels"][None],  # (1, H, W, 3)
            "agent_pos": raw_obs["agent_pos"][None],  # (1, 2)
        }


def make_goal_provider(env: gym.vector.VectorEnv) -> GoalObservationProvider | None:
    """Return the appropriate GoalObservationProvider for env, or None if unsupported.

    Detection is duck-typed so it works without importing env-specific packages at the
    top level.  Add new environments here as additional elif branches.
    """
    unwrapped = env.envs[0].unwrapped
    if hasattr(unwrapped, "goal_pose"):  # PushT
        return PushTGoalProvider()
    # elif hasattr(unwrapped, "..."):    # Future envs go here
    return None
