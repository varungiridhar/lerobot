from __future__ import annotations

import ast
import copy
import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch
from torch import nn

from lerobot.policies.fastwam.ibrl_td3 import (
    EncodedReplayBuffer,
    IBRLBatch,
    IBRLTD3,
    freeze_behavior_policy,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "self_improvement_ibrl_td3_loop.py"
LAUNCHER_PATH = REPO_ROOT / "scripts" / "run_ibrl_self_improvement.sh"
PIPELINE_PATH = REPO_ROOT / "scripts" / "submit_ibrl_h1_pipeline.sh"


class _FixedActor(nn.Module):
    def __init__(self, action: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("action", action.float().flatten())

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.action.expand(observation.shape[0], -1)


class _ModeRecordingActor(_FixedActor):
    def __init__(self, action: torch.Tensor) -> None:
        super().__init__(action)
        self.forward_training_modes: list[bool] = []

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        self.forward_training_modes.append(self.training)
        return super().forward(observation)


class _TableTwinCritic(nn.Module):
    """Return known twin-Q tables for BC action 0 and RL action 1."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("bc_q1", torch.tensor([4.0, 1.0]))
        self.register_buffer("bc_q2", torch.tensor([3.0, 5.0]))
        self.register_buffer("rl_q1", torch.tensor([2.0, 7.0]))
        self.register_buffer("rl_q2", torch.tensor([5.0, 6.0]))

    def forward(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = observation[:, 0].long()
        is_rl = action[:, :1] > 0.5
        bc_q1 = self.bc_q1[indices].unsqueeze(-1)
        bc_q2 = self.bc_q2[indices].unsqueeze(-1)
        rl_q1 = self.rl_q1[indices].unsqueeze(-1)
        rl_q2 = self.rl_q2[indices].unsqueeze(-1)
        return torch.where(is_rl, rl_q1, bc_q1), torch.where(is_rl, rl_q2, bc_q2)

    def conservative(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(observation, action)
        return torch.minimum(q1, q2)


class _RaisingCritic(nn.Module):
    def conservative(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raise AssertionError("Hybrid rollout selection must not use the online critic.")


def _batch(batch_size: int = 4, observation_dim: int = 3, action_dim: int = 2) -> IBRLBatch:
    return IBRLBatch(
        observations=torch.randn(batch_size, observation_dim),
        actions=torch.zeros(batch_size, action_dim),
        rewards=torch.zeros(batch_size, 1),
        next_observations=torch.randn(batch_size, observation_dim),
        dones=torch.zeros(batch_size, 1),
        discounts=torch.full((batch_size, 1), 0.99),
        next_bc_actions=torch.zeros(batch_size, action_dim),
    )


def _load_loop_module():
    spec = importlib.util.spec_from_file_location("self_improvement_ibrl_td3_loop", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hybrid_action_uses_target_twin_q_min_per_sample():
    agent = IBRLTD3(observation_dim=1, action_dim=1, hidden_dim=8)
    agent.actor = _FixedActor(torch.ones(1))
    agent.critic = _RaisingCritic()
    agent.critic_target = _TableTwinCritic()
    observation = torch.tensor([[0.0], [1.0]])
    bc_action = torch.zeros(2, 1)

    selection = agent.select_hybrid_action(observation, bc_action)

    torch.testing.assert_close(selection.bc_q, torch.tensor([[3.0], [1.0]]))
    torch.testing.assert_close(selection.rl_q, torch.tensor([[2.0], [6.0]]))
    torch.testing.assert_close(selection.action, torch.tensor([[0.0], [1.0]]))
    assert selection.bc_selected.tolist() == [[True], [False]]


def test_bootstrap_target_maxes_bc_and_target_actor_conservative_values():
    agent = IBRLTD3(observation_dim=1, action_dim=1, hidden_dim=8)
    agent.actor_target = _FixedActor(torch.ones(1))
    agent.critic_target = _TableTwinCritic()
    batch = IBRLBatch(
        observations=torch.zeros(2, 1),
        actions=torch.zeros(2, 1),
        rewards=torch.tensor([[1.0], [2.0]]),
        next_observations=torch.tensor([[0.0], [1.0]]),
        dones=torch.tensor([[0.0], [1.0]]),
        discounts=torch.full((2, 1), 0.9),
        next_bc_actions=torch.zeros(2, 1),
    )

    target = agent.bellman_target(batch, noise=torch.zeros(2, 1))

    torch.testing.assert_close(target, torch.tensor([[3.7], [2.0]]))


def test_actor_dropout_modes_are_scoped_to_learning_and_checkpointed():
    agent = IBRLTD3(
        observation_dim=1,
        action_dim=1,
        hidden_dim=8,
        actor_dropout=0.35,
        policy_delay=1,
    )
    assert agent.actor_dropout == pytest.approx(0.35)
    assert [module.p for module in agent.actor.modules() if isinstance(module, nn.Dropout)] == [
        pytest.approx(0.35),
        pytest.approx(0.35),
    ]
    assert agent.checkpoint()["actor_dropout"] == pytest.approx(0.35)

    rollout_actor = _ModeRecordingActor(torch.ones(1))
    rollout_actor.train()
    agent.actor = rollout_actor
    agent.critic_target = _TableTwinCritic()
    agent.select_hybrid_action(torch.zeros(1, 1), torch.zeros(1, 1))
    assert rollout_actor.forward_training_modes == [False]
    assert rollout_actor.training

    bootstrap_actor = _ModeRecordingActor(torch.ones(1))
    bootstrap_actor.eval()
    agent.actor_target = bootstrap_actor
    agent.bellman_target(
        IBRLBatch(
            observations=torch.zeros(1, 1),
            actions=torch.zeros(1, 1),
            rewards=torch.zeros(1, 1),
            next_observations=torch.zeros(1, 1),
            dones=torch.zeros(1, 1),
            discounts=torch.full((1, 1), 0.99),
            next_bc_actions=torch.zeros(1, 1),
        ),
        noise=torch.zeros(1, 1),
    )
    assert bootstrap_actor.forward_training_modes == [True]
    assert not bootstrap_actor.training


def test_actor_optimization_enables_dropout_even_if_agent_was_in_eval_mode():
    agent = IBRLTD3(
        observation_dim=3,
        action_dim=2,
        hidden_dim=8,
        actor_dropout=0.5,
        policy_delay=1,
    )
    forward_training_modes: list[bool] = []
    hook = agent.actor.register_forward_pre_hook(
        lambda module, inputs: forward_training_modes.append(module.training)
    )
    agent.eval()

    agent.update(_batch())

    hook.remove()
    assert forward_training_modes == [True]
    assert not agent.actor.training


def test_batch_canonicalizes_scalar_columns_without_cross_batch_broadcasting():
    batch = IBRLBatch(
        observations=torch.zeros(3, 2),
        actions=torch.zeros(3, 1),
        rewards=torch.tensor([1.0, 2.0, 3.0]),
        next_observations=torch.zeros(3, 2),
        dones=torch.tensor([False, False, True]),
        discounts=torch.tensor([0.9, 0.9, 0.9]),
        next_bc_actions=torch.zeros(3, 1),
    )

    assert batch.rewards.shape == (3, 1)
    assert batch.dones.shape == (3, 1)
    assert batch.discounts.shape == (3, 1)


def test_targets_are_independent_frozen_and_polyak_updated():
    agent = IBRLTD3(observation_dim=3, action_dim=2, hidden_dim=8, tau=0.25)
    for online, target in (
        (agent.actor, agent.actor_target),
        (agent.critic, agent.critic_target),
    ):
        for online_parameter, target_parameter in zip(
            online.parameters(), target.parameters(), strict=True
        ):
            torch.testing.assert_close(online_parameter, target_parameter)
            assert online_parameter.data_ptr() != target_parameter.data_ptr()
            assert not target_parameter.requires_grad
            online_parameter.data.fill_(2.0)
            target_parameter.data.zero_()

    agent.soft_update_targets()

    for target in (agent.actor_target, agent.critic_target):
        for parameter in target.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, 0.5))
    agent.train()
    assert not agent.actor_target.training
    assert not agent.critic_target.training


def test_frozen_bc_is_not_optimized_or_changed_by_td3_update():
    bc_policy = nn.Sequential(nn.Linear(3, 4), nn.Dropout(0.5), nn.Linear(4, 2))
    freeze_behavior_policy(bc_policy)
    before = [parameter.detach().clone() for parameter in bc_policy.parameters()]
    agent = IBRLTD3(
        observation_dim=3,
        action_dim=2,
        hidden_dim=8,
        policy_delay=1,
    )

    agent.update(_batch(), max_grad_norm=10.0)

    optimizer_parameters = {
        id(parameter)
        for optimizer in (agent.actor_optimizer, agent.critic_optimizer)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert not bc_policy.training
    for old, parameter in zip(before, bc_policy.parameters(), strict=True):
        assert not parameter.requires_grad
        assert parameter.grad is None
        assert id(parameter) not in optimizer_parameters
        torch.testing.assert_close(parameter, old, rtol=0, atol=0)


def test_critic_target_updates_before_delayed_actor_target():
    agent = IBRLTD3(
        observation_dim=3,
        action_dim=2,
        hidden_dim=8,
        tau=0.5,
        policy_delay=2,
    )
    actor_target_before = [parameter.clone() for parameter in agent.actor_target.parameters()]
    critic_target_before = [parameter.clone() for parameter in agent.critic_target.parameters()]
    with torch.no_grad():
        for parameter in agent.critic.parameters():
            parameter.add_(0.25)

    metrics = agent.update(_batch())

    assert metrics["actor_loss"] is None
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            critic_target_before, agent.critic_target.parameters(), strict=True
        )
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(actor_target_before, agent.actor_target.parameters(), strict=True)
    )


def test_macro_step_uses_discounted_rewards_and_actual_duration():
    loop = _load_loop_module()

    class _FakeEnv:
        def __init__(self) -> None:
            self.step_index = 0

        def step(self, action):
            assert action.shape == (1, 7)
            self.step_index += 1
            reward = float(self.step_index == 2)
            done = self.step_index == 2
            info = {"final_info": {"is_success": torch.tensor([done])}} if done else {}
            return {"step": self.step_index}, [reward], [done], [False], info

    observation, reward, raw_reward, discount, executed, done, success = loop._execute_chunk(
        _FakeEnv(),
        torch.zeros(3, 7),
        env_step=0,
        max_env_steps=10,
        gamma=0.9,
    )

    assert observation == {"step": 2}
    assert reward == pytest.approx(0.9)
    assert raw_reward == pytest.approx(1.0)
    assert discount == pytest.approx(0.9**2)
    assert executed == 2
    assert done
    assert success


def test_horizon_one_executes_one_primitive_action_and_replans():
    loop = _load_loop_module()

    class _FakeEnv:
        def __init__(self) -> None:
            self.actions: list[torch.Tensor] = []

        def step(self, action):
            self.actions.append(torch.from_numpy(action.copy()))
            return {"step": len(self.actions)}, [0.25], [False], [False], {}

    env = _FakeEnv()
    action = torch.arange(7, dtype=torch.float32).reshape(1, 7)
    observation, reward, raw_reward, discount, executed, done, success = loop._execute_chunk(
        env,
        action,
        env_step=0,
        max_env_steps=520,
        gamma=0.99,
    )

    assert observation == {"step": 1}
    assert len(env.actions) == 1
    torch.testing.assert_close(env.actions[0], action)
    assert reward == pytest.approx(0.25)
    assert raw_reward == pytest.approx(0.25)
    assert discount == pytest.approx(0.99)
    assert executed == 1
    assert not done
    assert not success


def test_update_cadence_is_counted_in_executed_primitive_actions():
    loop = _load_loop_module()

    # A complete H=10 decision must reproduce the released update_freq=2
    # cadence: five learner updates, not one update per macro decision.
    updates_due, credit = loop._consume_update_credit(0, 10, 2)
    assert (updates_due, credit) == (5, 0)

    # Early termination only credits actions that actually reached the env,
    # while odd remainders carry into the next decision without being lost.
    updates_due, credit = loop._consume_update_credit(0, 3, 2)
    assert (updates_due, credit) == (1, 1)
    updates_due, credit = loop._consume_update_credit(credit, 1, 2)
    assert (updates_due, credit) == (1, 0)


@pytest.mark.parametrize(
    ("carried", "executed", "frequency"),
    [(-1, 1, 2), (0, -1, 2), (0, 1, 0)],
)
def test_update_cadence_rejects_invalid_step_accounting(carried, executed, frequency):
    loop = _load_loop_module()

    with pytest.raises(ValueError):
        loop._consume_update_credit(carried, executed, frequency)


def test_paired_evaluation_reports_hybrid_delta_on_identical_trials():
    loop = _load_loop_module()
    episodes = [
        {"task": "suite/0", "seed": 100, "init_state_index": 0, "success": False},
        {"task": "suite/0", "seed": 101, "init_state_index": 1, "success": True},
        {"task": "suite/0", "seed": 102, "init_state_index": 2, "success": True},
    ]
    reference = {"episodes": episodes, "pc_success": 200 / 3}
    hybrid = {
        "episodes": [
            {**episodes[0], "success": True},
            {**episodes[1], "success": True},
            {**episodes[2], "success": False},
        ],
        "pc_success": 200 / 3,
    }

    paired = loop._paired_evaluation_summary(reference, hybrid)

    assert paired == {
        "n_paired_episodes": 3,
        "hybrid_wins": 1,
        "ties": 1,
        "hybrid_losses": 1,
        "hybrid_minus_reference_success_pp": 0.0,
    }


def test_ibrl_defaults_and_launcher_contract(monkeypatch):
    loop = _load_loop_module()
    monkeypatch.setattr(
        "sys.argv",
        [str(SCRIPT_PATH), "--fastwam_ckpt", "fastwam", "--output_dir", "out"],
    )
    args = loop.parse_args()
    assert args.gamma == pytest.approx(0.99)
    assert args.tau == pytest.approx(0.01)
    assert args.policy_delay == 2
    assert args.exploration_noise == pytest.approx(0.1)
    assert args.target_noise == pytest.approx(0.1)
    assert args.target_noise_clip == pytest.approx(0.3)
    assert args.action_horizon == 1
    assert args.phase == "all"
    assert args.online_episode_end is None
    assert args.resume_state is None
    assert args.n_demo_episodes == 10
    assert args.bc_warmup_episodes == 40
    assert args.update_every_env_steps == 2
    assert args.learner_warmstart_updates == 0
    assert args.actor_dropout == pytest.approx(0.5)
    assert args.hidden_dim == 1024
    assert args.buffer_size == 100_000
    assert args.eval_n_episodes == 50
    assert args.eval_seed == 100_000

    launcher = LAUNCHER_PATH.read_text()
    assert "ACTION_HORIZON=${ACTION_HORIZON:-1}" in launcher
    assert "ACTION_HORIZON=${ACTION_HORIZON:-10}" not in launcher
    assert "#SBATCH --array=0-9%10" in launcher
    assert "COLLECTION_TASK_ID=${COLLECTION_TASK_ID-${SLURM_ARRAY_TASK_ID:-0}}" in launcher
    assert "frozen FastWAM" in launcher
    assert "max over min target-Q" in launcher
    assert "N_DEMO_EPISODES=${N_DEMO_EPISODES:-10}" in launcher
    assert "BC_WARMUP_EPISODES=${BC_WARMUP_EPISODES:-40}" in launcher
    assert "EVAL_N_EPISODES=${EVAL_N_EPISODES:-50}" in launcher
    assert "UPDATE_EVERY_ENV_STEPS=${UPDATE_EVERY_ENV_STEPS:-2}" in launcher
    assert "LEARNER_WARMSTART_UPDATES=${LEARNER_WARMSTART_UPDATES:-0}" in launcher
    assert "HuggingFaceVLA/libero" in launcher
    assert "export PYOPENGL_PLATFORM=egl" in launcher
    assert '--phase "$PHASE"' in launcher
    loop_source = SCRIPT_PATH.read_text()
    assert "episode_index_offset" in loop_source
    assert "def finish_warmup()" in loop_source
    assert 'trainer["bc_warmup"]' in loop_source
    assert "hybrid=False" in loop_source
    assert "learn=False" in loop_source
    assert loop_source.index("def finish_warmup()") < loop_source.index("def finish_online(")

    pipeline = PIPELINE_PATH.read_text()
    assert 'ONLINE_SHARD_ENDS=${ONLINE_SHARD_ENDS:-"25 50 75 100"}' in pipeline
    assert '--dependency="aftercorr:$dependency"' in pipeline
    assert "submit_phase warmup" in pipeline
    assert "submit_phase reference_eval" in pipeline
    assert "submit_phase hybrid_eval" in pipeline
    assert "unset OUTPUT_DIR COLLECTION_TASK_ID PHASE ONLINE_EPISODE_END RESUME_STATE" in pipeline


def test_replay_state_roundtrip_preserves_ring_and_sampling():
    replay = EncodedReplayBuffer(capacity=3, observation_dim=2, action_dim=1)
    for index in range(5):
        replay.add(
            torch.tensor([index, index + 0.5]),
            torch.tensor([index / 10]),
            float(index),
            torch.tensor([index + 1, index + 1.5]),
            index == 4,
            0.99,
            torch.tensor([(index + 1) / 10]),
        )
    state = replay.state_dict()
    restored = EncodedReplayBuffer(capacity=3, observation_dim=2, action_dim=1)
    restored.load_state_dict(state)

    assert restored.position == replay.position
    assert len(restored) == len(replay)
    for name in (
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "dones",
        "discounts",
        "next_bc_actions",
    ):
        torch.testing.assert_close(getattr(restored, name), getattr(replay, name))
    left_generator = torch.Generator().manual_seed(123)
    right_generator = torch.Generator().manual_seed(123)
    left = replay.sample(3, generator=left_generator)
    right = restored.sample(3, generator=right_generator)
    for field in left.__dataclass_fields__:
        torch.testing.assert_close(getattr(left, field), getattr(right, field))


def test_agent_checkpoint_roundtrip_preserves_optimizers_and_next_update():
    torch.manual_seed(7)
    agent = IBRLTD3(
        observation_dim=3,
        action_dim=2,
        hidden_dim=8,
        actor_dropout=0.25,
        policy_delay=1,
    )
    batch = _batch()
    torch.manual_seed(8)
    agent.update(batch)

    restored = IBRLTD3(
        observation_dim=3,
        action_dim=2,
        hidden_dim=8,
        actor_dropout=0.25,
        policy_delay=1,
    )
    restored.load_checkpoint(copy.deepcopy(agent.checkpoint()))
    assert restored.total_updates == agent.total_updates
    for left, right in zip(agent.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)

    torch.manual_seed(9)
    left_metrics = agent.update(batch)
    torch.manual_seed(9)
    right_metrics = restored.update(batch)
    assert left_metrics == pytest.approx(right_metrics)
    for left, right in zip(agent.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_phase_rng_roundtrip_and_evaluation_merge():
    loop = _load_loop_module()
    replay_generator = torch.Generator().manual_seed(31)
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    state = loop._capture_rng_state(replay_generator)
    expected = (
        random.random(),
        np.random.random(),
        torch.rand(1),
        torch.rand(1, generator=replay_generator),
    )
    loop._restore_rng_state(state, replay_generator)
    actual = (
        random.random(),
        np.random.random(),
        torch.rand(1),
        torch.rand(1, generator=replay_generator),
    )
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])
    torch.testing.assert_close(actual[3], expected[3])

    common = {
        "pc_success": 100.0,
        "mean_return": 1.0,
        "bc_selected": 3,
        "rl_selected": 0,
        "bc_fraction": 1.0,
        "per_task": {"suite/0": {"episodes": 1, "successes": 1}},
        "policy": "frozen_fastwam",
        "exploration_noise": 0.0,
        "parameter_updates": 0,
    }
    first = {
        **common,
        "n_episodes": 1,
        "n_success": 1,
        "episodes": [{"episode": 0}],
    }
    second = {
        **common,
        "n_episodes": 1,
        "n_success": 0,
        "pc_success": 0.0,
        "mean_return": 0.0,
        "per_task": {"suite/0": {"episodes": 1, "successes": 0}},
        "episodes": [{"episode": 1}],
    }
    merged = loop._merge_evaluation_results(first, second)
    assert merged["n_episodes"] == 2
    assert merged["n_success"] == 1
    assert merged["pc_success"] == pytest.approx(50.0)
    assert merged["mean_return"] == pytest.approx(0.5)
    assert [episode["episode"] for episode in merged["episodes"]] == [0, 1]


def test_evaluation_shards_preserve_global_seed_and_task_init_indices():
    loop = _load_loop_module()
    task_envs = [("suite", 0, object()), ("suite", 1, object())]
    unsharded = loop._evaluation_trials(
        task_envs,
        total_episodes=7,
        episode_start=0,
        n_episodes=7,
    )
    sharded = [
        *loop._evaluation_trials(
            task_envs,
            total_episodes=7,
            episode_start=0,
            n_episodes=3,
        ),
        *loop._evaluation_trials(
            task_envs,
            total_episodes=7,
            episode_start=3,
            n_episodes=4,
        ),
    ]

    assert sharded == unsharded
    # Trial tuple: global episode, group, task id, env, task-local init offset.
    assert [(trial[0], trial[2], trial[4]) for trial in unsharded] == [
        (0, 0, 0),
        (1, 1, 0),
        (2, 0, 1),
        (3, 1, 1),
        (4, 0, 2),
        (5, 1, 2),
        (6, 0, 3),
    ]


def test_training_phase_calls_enforce_reference_then_hybrid_barrier():
    tree = ast.parse(SCRIPT_PATH.read_text())
    calls = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "collect_until"
        ),
        key=lambda node: node.lineno,
    )
    assert len(calls) == 2

    warmup, online = calls
    assert isinstance(warmup.args[0], ast.Subscript)
    assert ast.literal_eval(warmup.args[0].slice) == "bc_warmup"
    assert isinstance(warmup.args[1], ast.Attribute)
    assert warmup.args[1].attr == "bc_warmup_episodes"
    assert {keyword.arg: ast.literal_eval(keyword.value) for keyword in warmup.keywords} == {
        "hybrid": False,
        "learn": False,
    }

    assert isinstance(online.args[0], ast.Subscript)
    assert ast.literal_eval(online.args[0].slice) == "online_training"
    assert isinstance(online.args[1], ast.Name)
    assert online.args[1].id == "target"
    assert {keyword.arg: ast.literal_eval(keyword.value) for keyword in online.keywords} == {
        "hybrid": True,
        "learn": True,
    }
    assert warmup.lineno < online.lineno


def test_setup_restricts_libero_creation_to_collection_task(monkeypatch):
    """Select the task before make_env creates any MuJoCo/EGL contexts."""
    loop = _load_loop_module()
    captured = {}

    class _MakeEnvObserved(Exception):
        pass

    def fake_make_env(env_cfg, *, n_envs, use_async_envs):
        captured["env_cfg"] = env_cfg
        captured["n_envs"] = n_envs
        captured["use_async_envs"] = use_async_envs
        raise _MakeEnvObserved

    monkeypatch.setattr("lerobot.envs.factory.make_env", fake_make_env)
    args = SimpleNamespace(
        seed=42,
        task="libero_10",
        collection_task_id=6,
        episode_length=520,
    )

    with pytest.raises(_MakeEnvObserved):
        loop._setup(args, torch.device("cpu"))

    assert captured["env_cfg"].task_ids == [6]
    assert captured["env_cfg"].gym_kwargs["task_ids"] == [6]
    assert captured["n_envs"] == 1
    assert captured["use_async_envs"] is False
