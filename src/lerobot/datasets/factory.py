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
import logging
from pprint import pformat

import numpy as np

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, OBS_PREFIX, REWARD

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    # Decide single- vs multi-dataset path. ``repo_ids`` (list) takes precedence
    # when non-empty; otherwise we use the single ``repo_id``.
    repo_ids_list = cfg.dataset.repo_ids
    repo_id_single = cfg.dataset.repo_id
    if (repo_ids_list is None or len(repo_ids_list) == 0) and repo_id_single is None:
        raise ValueError("DatasetConfig: must provide either `repo_id` (str) or `repo_ids` (list[str]).")

    if (repo_ids_list is None or len(repo_ids_list) == 0):
        ds_meta = LeRobotDatasetMetadata(
            repo_id_single, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        if not cfg.dataset.streaming:
            dataset = LeRobotDataset(
                repo_id_single,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                tolerance_s=cfg.tolerance_s,
            )
        else:
            dataset = StreamingLeRobotDataset(
                repo_id_single,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
            )
    else:
        # Multi-dataset path: resolve delta_timestamps from the first sub-dataset's
        # meta (assumes all sub-datasets share fps and feature schema, which is the
        # standard MultiLeRobotDataset invariant).
        if cfg.dataset.streaming:
            raise NotImplementedError("StreamingLeRobotDataset is not supported with multiple repo_ids.")
        from pathlib import Path as _Path
        first_root = (
            str(_Path(cfg.dataset.root) / repo_ids_list[0]) if cfg.dataset.root else None
        )
        first_meta = LeRobotDatasetMetadata(repo_ids_list[0], root=first_root)
        delta_timestamps = resolve_delta_timestamps(cfg.policy, first_meta) if cfg.policy is not None else None
        dataset = MultiLeRobotDataset(
            repo_ids_list,
            root=cfg.dataset.root,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )
        if cfg.dataset.use_imagenet_stats:
            for sub in dataset._datasets:
                for key in sub.meta.camera_keys:
                    for stats_type, stats in IMAGENET_STATS.items():
                        sub.meta.stats[key][stats_type] = np.array(stats, dtype=np.float32)
            # Re-aggregate after mutation so dataset.stats reflects the override.
            from lerobot.datasets.compute_stats import aggregate_stats
            dataset.stats = aggregate_stats([sub.meta.stats for sub in dataset._datasets])
        return _maybe_wrap_for_policy(dataset, cfg)

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = np.array(stats, dtype=np.float32)

    return _maybe_wrap_for_policy(dataset, cfg)


def _maybe_wrap_for_policy(dataset, cfg: TrainPipelineConfig):
    """Apply policy-specific dataset wrappers (e.g. Q-function reward labelling).

    For ``q_function``: wrap with ``QValueLabelDataset`` so each sample carries
    the four Q keys (``q_reward_chunk_first``, ``q_reward_pad_first``,
    ``q_bootstrap_valid``, ``q_bucket_index``). Reads required parameters
    (``h``, ``step_reward``, ``terminal_bonuses``, ``reward_mode``,
    ``quality_scalars``) directly off ``cfg.policy``.
    """
    if cfg.policy is None or getattr(cfg.policy, "type", None) != "q_function":
        return dataset
    from lerobot.policies.q_function.q_value_labels import QValueLabelDataset
    wrapped = QValueLabelDataset(
        dataset,
        h=cfg.policy.h,
        step_reward=cfg.policy.step_reward,
        terminal_bonuses=cfg.policy.terminal_bonuses,
        reward_mode=getattr(cfg.policy, "reward_mode", "sparse"),
        quality_scalars=getattr(cfg.policy, "quality_scalars", None),
        precache_root=getattr(cfg.policy, "precache_root", None),
        terminal_bonus_uniform=getattr(cfg.policy, "terminal_bonus_uniform", 1.0),
        bucket_overrides=getattr(cfg.policy, "bucket_overrides", None),
    )
    logging.info(f"Wrapped dataset with QValueLabelDataset; counts: {wrapped.bucket_counts()}")
    return wrapped
