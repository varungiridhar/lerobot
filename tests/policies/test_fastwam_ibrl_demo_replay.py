from __future__ import annotations

import pytest
import torch

from lerobot.policies.fastwam.ibrl_demo_replay import (
    build_demo_macro_transitions,
    select_demo_episode_indices_from_metadata,
)


def test_select_demo_episodes_is_exact_deterministic_and_not_task_index_based():
    episodes = [
        {"episode_index": 9, "tasks": ["task b"]},
        {"episode_index": 4, "tasks": ["task a"]},
        {"episode_index": 1, "tasks": ["task b"]},
        {"episode_index": 7, "tasks": ["task b"]},
        {"episode_index": 2, "tasks": ["task b extra"]},
    ]

    selected = select_demo_episode_indices_from_metadata(
        episodes,
        task_description="task b",
        n_episodes=2,
    )

    assert selected == (1, 7)


def test_select_demo_episodes_fails_when_task_has_too_few_demos():
    with pytest.raises(ValueError, match="contains 1 demonstrations"):
        select_demo_episode_indices_from_metadata(
            [{"episode_index": 0, "tasks": ["task"]}],
            task_description="task",
            n_episodes=2,
        )


def test_demo_macro_chunks_use_stride_h_sparse_smdp_labels_and_terminal_padding():
    gamma = 0.9
    actions = torch.arange(23 * 2, dtype=torch.float32).reshape(23, 2)

    transitions = build_demo_macro_transitions(
        actions,
        episode_index=17,
        action_horizon=10,
        gamma=gamma,
    )

    assert [transition.start_frame for transition in transitions] == [0, 10, 20]
    assert [transition.next_frame for transition in transitions] == [10, 20, None]
    assert [transition.executed_steps for transition in transitions] == [10, 10, 3]
    assert [transition.done for transition in transitions] == [False, False, True]
    assert [transition.reward for transition in transitions] == pytest.approx([0.0, 0.0, gamma**2])
    assert [transition.discount for transition in transitions] == pytest.approx(
        [gamma**10, gamma**10, gamma**3]
    )
    assert all(transition.action.shape == (10, 2) for transition in transitions)
    torch.testing.assert_close(transitions[0].action, actions[:10])
    torch.testing.assert_close(transitions[1].action, actions[10:20])
    torch.testing.assert_close(transitions[2].action[:3], actions[20:23])
    torch.testing.assert_close(
        transitions[2].action[3:],
        actions[22:].expand(7, -1),
    )


def test_demo_actions_remain_in_executed_libero_gripper_convention():
    actions = torch.tensor(
        [
            [0.1, -1.0],
            [0.2, 1.0],
            [0.3, -1.0],
        ]
    )

    transition = build_demo_macro_transitions(
        actions,
        episode_index=0,
        action_horizon=4,
        gamma=0.99,
    )[0]

    assert transition.action[:, -1].tolist() == [-1.0, 1.0, -1.0, -1.0]
