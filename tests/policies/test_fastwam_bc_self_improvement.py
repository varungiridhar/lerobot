from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from lerobot.policies.fastwam.online_bc_dataset import (
    ROBOTWIN_ONLINE_BC_DATASET_FPS,
    ROBOTWIN_SOURCE_CAMERA_KEYS,
    FastWAMBCDataset,
    SuccessfulEpisodeWriter,
    _frame_ids_for_episodes,
    libero_gripper_to_fastwam,
    load_growing_online_success_dataset,
)
from lerobot.utils.constants import ACTION, OBS_STATE


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "self_improvement_bc_loop.py"
LAUNCHER_PATH = REPO_ROOT / "scripts" / "run_bc_self_improvement.sh"
ROBOTWIN_LAUNCHER_PATH = REPO_ROOT / "scripts" / "run_robotwin_bc_self_improvement.sh"


def _load_loop_module():
    spec = importlib.util.spec_from_file_location("self_improvement_bc_loop", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _OneSampleDataset(Dataset):
    def __init__(self, sample: dict):
        self.sample = sample

    def __len__(self):
        return 1

    def __getitem__(self, _idx):
        return self.sample


def test_libero_gripper_conversion_does_not_modify_input():
    action = torch.tensor([[0.2, -1.0], [0.4, 1.0]])
    original = action.clone()
    converted = libero_gripper_to_fastwam(action)
    torch.testing.assert_close(action, original)
    torch.testing.assert_close(converted[:, -1], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(converted[:, 0], action[:, 0])


def test_bc_dataset_resizes_and_converts_original_libero_actions():
    camera = "observation.images.image"
    sample = {
        camera: torch.rand(3, 256, 256),
        OBS_STATE: torch.arange(8).float(),
        ACTION: torch.tensor([[0.1, -1.0], [0.2, 1.0]]),
        f"{ACTION}_is_pad": torch.tensor([False, True]),
        "task": "test task",
    }
    dataset = FastWAMBCDataset(
        _OneSampleDataset(sample),
        camera_keys=(camera,),
        image_size=(224, 224),
        libero_action_convention=True,
    )
    item = dataset[0]
    assert item[camera].shape == (3, 224, 224)
    torch.testing.assert_close(item[ACTION][:, -1], torch.tensor([1.0, 0.0]))
    assert item[f"{ACTION}_is_pad"].dtype == torch.bool
    assert item["task"] == "test task"


def test_online_bc_dataset_rejects_wrong_gripper_convention():
    camera = "observation.images.image"
    sample = {
        camera: torch.rand(3, 16, 16),
        OBS_STATE: torch.zeros(2),
        ACTION: torch.tensor([[0.1, -1.0]]),
        "task": "test task",
    }
    dataset = FastWAMBCDataset(
        _OneSampleDataset(sample),
        camera_keys=(camera,),
        image_size=(16, 16),
        libero_action_convention=False,
    )
    with pytest.raises(ValueError, match="gripper targets"):
        dataset[0]


def test_bc_dataset_preserves_robotwin_qpos_actions():
    camera = "observation.images.image"
    action = torch.linspace(-2.0, 2.0, 28).reshape(2, 14)
    sample = {
        camera: torch.rand(3, 384, 320),
        OBS_STATE: torch.zeros(14),
        ACTION: action,
        "task": "Beat the block with the hammer",
    }
    dataset = FastWAMBCDataset(
        _OneSampleDataset(sample),
        camera_keys=(camera,),
        image_size=(384, 320),
        libero_action_convention=False,
        validate_gripper_targets=False,
    )
    item = dataset[0]
    torch.testing.assert_close(item[ACTION], action)
    assert ROBOTWIN_ONLINE_BC_DATASET_FPS == 50.0


def test_bc_dataset_lazily_concatenates_full_robotwin_replay_cameras():
    target_camera = "observation.images.image"
    sample = {
        ROBOTWIN_SOURCE_CAMERA_KEYS[0]: torch.full((3, 480, 640), 0.25),
        ROBOTWIN_SOURCE_CAMERA_KEYS[1]: torch.full((3, 480, 640), 0.50),
        ROBOTWIN_SOURCE_CAMERA_KEYS[2]: torch.full((3, 480, 640), 0.75),
        OBS_STATE: torch.zeros(14),
        ACTION: torch.zeros(32, 14),
        "task": "RoboTwin task",
    }
    dataset = FastWAMBCDataset(
        _OneSampleDataset(sample),
        camera_keys=(target_camera,),
        image_size=(384, 320),
        libero_action_convention=False,
        validate_gripper_targets=False,
        robotwin_source_camera_keys=ROBOTWIN_SOURCE_CAMERA_KEYS,
    )
    item = dataset[0]
    assert item[target_camera].shape == (3, 384, 320)
    torch.testing.assert_close(item[target_camera][:, :256], torch.full((3, 256, 320), 0.25))
    torch.testing.assert_close(item[target_camera][:, 256:, :160], torch.full((3, 128, 160), 0.50))
    torch.testing.assert_close(item[target_camera][:, 256:, 160:], torch.full((3, 128, 160), 0.75))


def test_task_replay_uses_frame_subset_without_copying_images():
    metadata = SimpleNamespace(
        episodes=[
            {"episode_index": 0, "dataset_from_index": 0, "dataset_to_index": 3},
            {"episode_index": 1, "dataset_from_index": 3, "dataset_to_index": 5},
            {"episode_index": 2, "dataset_from_index": 5, "dataset_to_index": 9},
        ]
    )
    assert _frame_ids_for_episodes(metadata, [0, 2]) == [0, 1, 2, 5, 6, 7, 8]
    with pytest.raises(ValueError, match="missing requested episode"):
        _frame_ids_for_episodes(metadata, [3])


def test_writer_never_accepts_failure(tmp_path):
    writer = SuccessfulEpisodeWriter(
        tmp_path / "successful_episodes",
        camera_keys=("observation.images.image",),
        image_size=(16, 16),
        action_dim=2,
        state_dim=2,
    )
    with pytest.raises(ValueError, match="only accepts successful"):
        writer.add_episode({"success": False})
    assert not writer.root.exists()


def test_successful_episode_round_trip(tmp_path):
    camera = "observation.images.image"
    root = tmp_path / "iter_000" / "successful_episodes"
    writer = SuccessfulEpisodeWriter(
        root,
        camera_keys=(camera,),
        image_size=(16, 16),
        action_dim=2,
        state_dim=2,
        fps=10.0,
    )
    writer.add_episode(
        {
            "success": True,
            "task": "round trip",
            camera: torch.rand(3, 3, 16, 16),
            OBS_STATE: torch.rand(3, 2),
            ACTION: torch.tensor([[0.1, 0.0], [0.2, 1.0], [0.3, 1.0]]),
        }
    )
    writer.finalize()

    config = SimpleNamespace(
        image_features={camera: None},
        image_size=(16, 16),
        action_delta_indices=[0, 1],
        observation_delta_indices=None,
        reward_delta_indices=None,
    )
    dataset = load_growing_online_success_dataset(tmp_path, config)
    assert dataset is not None
    assert len(dataset) == 3
    item = dataset[0]
    assert item[camera].shape == (3, 16, 16)
    assert item[ACTION].shape == (2, 2)
    assert item[f"{ACTION}_is_pad"].shape == (2,)
    assert item["task"] == "round trip"


class _ToyFastWAMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_expert = nn.Linear(3, 3)
        self.action_expert.use_gradient_checkpointing = False
        self.video_expert = nn.Linear(3, 3)
        self.vae = nn.Linear(3, 3)
        self.text_encoder = nn.Linear(3, 3)
        self.proprio_encoder = nn.Linear(2, 3)
        self.mot = SimpleNamespace(mot_checkpoint_mixed_attn=False)
        self.loss_lambda_video = 1.0


class _ToyFastWAMPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _ToyFastWAMModel()
        self.config = SimpleNamespace(
            freeze_video_dit=False,
            loss_lambda_video=1.0,
            action_dit_use_gradient_checkpointing=False,
            mot_checkpoint_mixed_attn=False,
        )


def test_only_behavior_path_is_trainable():
    loop = _load_loop_module()
    policy = _ToyFastWAMPolicy()
    trainable = loop.configure_behavior_finetune(policy)
    trainable_ids = {id(parameter) for parameter in trainable}
    expected = {
        id(parameter)
        for module in (policy.model.action_expert, policy.model.proprio_encoder)
        for parameter in module.parameters()
    }
    assert trainable_ids == expected
    assert all(not parameter.requires_grad for parameter in policy.model.video_expert.parameters())
    assert all(parameter.requires_grad for parameter in policy.model.proprio_encoder.parameters())
    assert policy.model.loss_lambda_video == 0.0


def test_rollout_actions_use_environment_specific_training_convention():
    loop = _load_loop_module()
    robotwin_policy_action = torch.full((1, 14), 99.0)
    robotwin_executed_action = torch.linspace(-1.5, 1.5, 14).unsqueeze(0)
    robotwin_target = loop.executed_action_to_training(
        "robotwin", robotwin_policy_action, robotwin_executed_action
    )
    torch.testing.assert_close(robotwin_target, robotwin_executed_action)

    libero_policy_action = torch.tensor([[0.25, 0.75]])
    libero_executed_action = torch.tensor([[0.25, -1.0]])
    libero_target = loop.executed_action_to_training(
        "libero", libero_policy_action, libero_executed_action
    )
    torch.testing.assert_close(libero_target, torch.tensor([[0.25, 1.0]]))


def test_plain_fastwam_loop_has_no_q_or_planner():
    launcher = LAUNCHER_PATH.read_text()
    assert "NEXT_FASTWAM_CKPT" in launcher
    assert "Q_CKPT" not in launcher
    assert "--q_ckpt" not in launcher
    assert "PLANNER_TYPE" not in launcher
    assert "NUM_INFERENCE_STEPS" in launcher
    assert "INTERMEDIATE_EVAL_N_EPISODES=${INTERMEDIATE_EVAL_N_EPISODES:-20}" in launcher
    assert "FINAL_EVAL_N_EPISODES=${FINAL_EVAL_N_EPISODES:-50}" in launcher
    assert "ITERATION == MAX_ITERATIONS - 1" in launcher
    assert '--eval_n_episodes "$CURRENT_EVAL_N_EPISODES"' in launcher

    loop_source = SCRIPT_PATH.read_text()
    assert "finetune_q" not in loop_source
    assert "q_ckpt" not in loop_source
    assert "FastWAMPlanner" not in loop_source
    assert "attach_planner" not in loop_source


def test_robotwin_launcher_is_plain_success_filtered_sft():
    launcher = ROBOTWIN_LAUNCHER_PATH.read_text()
    assert "--env robotwin" in launcher
    assert "--online_dataset_fps \"$ONLINE_DATASET_FPS\"" in launcher
    assert "ONLINE_FRACTION=${ONLINE_FRACTION:-0.5}" in launcher
    assert "COLLECT_SHARDS=${COLLECT_SHARDS:-4}" in launcher
    assert "EVAL_SHARDS=${EVAL_SHARDS:-4}" in launcher
    assert "EVAL_N_EPISODES=${EVAL_N_EPISODES:-20}" in launcher
    assert "MAX_ITERATIONS=${MAX_ITERATIONS:-5}" in launcher
    assert "N_EPISODES=${N_EPISODES:-100}" in launcher
    assert "open_laptop" not in launcher.split("TASKS=${TASKS:-", 1)[1].split("}", 1)[0]
    assert "place_object_scale" not in launcher.split("TASKS=${TASKS:-", 1)[1].split("}", 1)[0]
    assert "put_object_cabinet" not in launcher.split("TASKS=${TASKS:-", 1)[1].split("}", 1)[0]
    assert "--collect_only" in launcher
    assert "--skip_collect" in launcher
    assert "--expected_collect_shards" in launcher
    assert "scripts/check_robotwin_timing_contract.py" in launcher
    assert "/storage/project/r-agarg35-0/shared/robotwin2.0" in launcher
    assert "Q_CKPT" not in launcher
    assert "--q_ckpt" not in launcher
    assert "PLANNER_TYPE" not in launcher
