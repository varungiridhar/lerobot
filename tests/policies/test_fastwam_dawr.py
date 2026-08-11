from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.fastwam.dawr_dataset import (
    DAWR_REWARD,
    DAWR_TERMINAL,
    DAWR_TRAJECTORY_INDEX,
    DAWRDecisionWriter,
    LossWeightedDataset,
    exponential_advantage_weights,
    fastwam_action_to_libero,
    libero_action_to_fastwam,
    load_growing_dawr_dataset,
)
from lerobot.policies.fastwam.dawr_value import DAWRValueCritic, td_lambda_returns
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
from lerobot.policies.fastwam.wan22.fastwam import _apply_action_sample_weights
from lerobot.utils.constants import ACTION, OBS_STATE


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "self_improvement_dawr_loop.py"
LAUNCHER_PATH = REPO_ROOT / "scripts" / "run_dawr_self_improvement.sh"


def _load_loop_module():
    spec = importlib.util.spec_from_file_location("self_improvement_dawr_loop", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gripper_round_trip_matches_libero_execution():
    fastwam = torch.tensor([[0.2, 0.0], [0.4, 1.0], [0.6, 0.49]])
    original = fastwam.clone()
    libero = fastwam_action_to_libero(fastwam)
    torch.testing.assert_close(fastwam, original)
    torch.testing.assert_close(libero[:, -1], torch.tensor([1.0, -1.0, 1.0]))
    torch.testing.assert_close(
        libero_action_to_fastwam(libero)[:, -1],
        torch.tensor([0.0, 1.0, 0.0]),
    )


def test_td_lambda_propagates_reward_and_resets_at_trajectory_boundary():
    advantages, returns = td_lambda_returns(
        rewards=torch.tensor([0.0, 1.0, 0.0, 2.0]),
        terminals=torch.tensor([False, True, False, True]),
        trajectory_indices=torch.tensor([0, 0, 1, 1]),
        values=torch.zeros(4),
        gamma=0.99,
        td_lambda=0.95,
    )
    expected = torch.tensor([0.99 * 0.95, 1.0, 2.0 * 0.99 * 0.95, 2.0])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


def test_advantage_weights_are_standardized_clipped_and_nontrivial():
    weights, stats = exponential_advantage_weights(
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]),
        beta=10.0,
        max_weight=100.0,
    )
    assert weights.shape == (5,)
    assert torch.all(weights >= 0)
    assert float(weights.max()) == pytest.approx(100.0)
    assert weights[0] < weights[1] < weights[2] < weights[3]
    assert stats["standardized_advantage_mean"] == pytest.approx(0.0, abs=1e-6)
    assert 0 < stats["effective_sample_size"] <= len(weights)


def test_value_critic_is_a_fresh_state_value_not_an_action_value():
    critic = DAWRValueCritic(
        vision_dim=8,
        n_cameras=2,
        state_dim=8,
        n_tasks=3,
        hidden_dim=32,
        dropout=0.0,
    )
    values = critic(
        torch.randn(4, 2, 8),
        torch.randn(4, 8),
        torch.tensor([0, 1, 2, 0]),
    )
    assert values.shape == (4,)
    values.sum().backward()
    assert all(parameter.grad is not None for parameter in critic.parameters())


def test_weighted_action_loss_does_not_cancel_at_batch_size_one():
    weighted, unweighted, mean_weight, max_weight = _apply_action_sample_weights(
        torch.tensor([2.0]),
        torch.tensor([7.0]),
    )
    assert float(unweighted) == pytest.approx(2.0)
    assert float(weighted) == pytest.approx(14.0)
    assert mean_weight == pytest.approx(7.0)
    assert max_weight == pytest.approx(7.0)


def test_policy_forwards_explicit_sample_weights_to_fastwam_model():
    captured = {}

    class _Model:
        device = torch.device("cpu")
        torch_dtype = torch.float32

        def training_loss(self, sample, sample_weights=None):
            captured["sample"] = sample
            captured["weights"] = sample_weights
            return torch.tensor(1.0), {"loss_action": torch.tensor(1.0)}

    dummy_policy = SimpleNamespace(
        model=_Model(),
        _prepare_video_for_training=lambda _batch: torch.zeros(1, 3, 1, 2, 2),
        _encode_text=lambda _batch: (torch.zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool)),
        _get_proprio=lambda _batch: torch.zeros(1, 8),
    )
    weights = torch.tensor([3.0])
    loss, _ = FastWAMPolicy.forward(
        dummy_policy,
        {ACTION: torch.zeros(1, 4, 7)},
        sample_weights=weights,
    )
    assert float(loss) == pytest.approx(1.0)
    assert captured["weights"] is weights


def test_decision_writer_keeps_failure_trajectory_rewards_and_masks(tmp_path):
    camera_keys = ("observation.images.image", "observation.images.image2")
    horizon = 4
    root = tmp_path / "iter_000" / "dawr_actor_episodes"
    writer = DAWRDecisionWriter(
        root,
        camera_keys=camera_keys,
        image_size=(16, 16),
        action_dim=7,
        state_dim=8,
        horizon=horizon,
    )
    decisions = []
    for index in range(2):
        decision = {
            ACTION: torch.full((horizon, 7), float(index)),
            f"{ACTION}_is_pad": torch.tensor([False, False, True, True]),
            OBS_STATE: torch.arange(8).float(),
            DAWR_REWARD: float(index),
            DAWR_TERMINAL: index == 1,
        }
        for key in camera_keys:
            decision[key] = torch.rand(3, 16, 16)
        decisions.append(decision)
    writer.add_episode(decisions, task="test task", success=False)
    writer.finalize()

    dataset = load_growing_dawr_dataset(tmp_path, camera_keys=camera_keys)
    assert dataset is not None
    assert len(dataset) == 2
    first, last = dataset[0], dataset[1]
    assert float(first[DAWR_REWARD]) == pytest.approx(0.0)
    assert not bool(first[DAWR_TERMINAL])
    assert bool(last[DAWR_TERMINAL])
    assert int(first[DAWR_TRAJECTORY_INDEX]) == int(last[DAWR_TRAJECTORY_INDEX])
    torch.testing.assert_close(
        first[f"{ACTION}_is_pad"],
        torch.tensor([False, False, True, True]),
    )

    weighted = LossWeightedDataset(dataset, torch.tensor([0.25, 2.0]))
    actor_item = weighted[0]
    assert DAWR_REWARD not in actor_item
    assert DAWR_TERMINAL not in actor_item
    assert DAWR_TRAJECTORY_INDEX not in actor_item


def test_dawr_defaults_and_launch_contract_have_no_q_or_planning(monkeypatch):
    loop = _load_loop_module()
    monkeypatch.setattr(
        "sys.argv",
        [str(SCRIPT_PATH), "--fastwam_ckpt", "fastwam", "--output_dir", "out"],
    )
    args = loop.parse_args()
    assert args.gamma == pytest.approx(0.99)
    assert args.td_lambda == pytest.approx(0.95)
    assert args.beta == pytest.approx(10.0)
    assert args.max_adv_weight == pytest.approx(100.0)
    assert args.episode_length == 520
    assert args.collection_task_id is None
    assert not args.trace_rollouts
    assert not args.normalize_weight_mean
    assert args.critic_warmup_iterations == 2
    assert args.actor_lr == pytest.approx(1e-5)
    assert not hasattr(args, "q_ckpt")

    script = SCRIPT_PATH.read_text()
    launcher = LAUNCHER_PATH.read_text()
    assert "lerobot.policies.q_function" not in script
    assert "q_checkpoint" not in script.lower()
    assert "q_ckpt" not in launcher.lower()
    assert "planning:         disabled" in launcher.lower()
    assert "EPISODE_LENGTH=${EPISODE_LENGTH:-520}" in launcher
    assert '--episode_length "$EPISODE_LENGTH"' in launcher
    assert '--collection_task_id "$COLLECTION_TASK_ID"' in launcher
    assert "CUDA_LAUNCH_BLOCKING=1" in launcher
    assert "PYTHONFAULTHANDLER=1" in launcher
    assert "SEED=${SEED:-42}" in launcher
