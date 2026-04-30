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
from collections.abc import Iterator

import torch


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


class WeightedEpisodeAwareSampler:
    """Episode-aware sampler that draws frames from two buckets (primary / online)
    at a target online ratio, with replacement.

    Used for RLPD-style mixing during self-improvement finetuning: each yielded
    index is independently from the online bucket with probability
    ``online_sample_ratio`` and from the primary bucket otherwise. This decouples
    the per-batch online share from the dataset-size ratio, so a small amount of
    on-policy data can be upsampled without growing it on disk.
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        n_primary_episodes: int,
        online_sample_ratio: float,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        num_samples: int | None = None,
    ):
        if not 0.0 < online_sample_ratio < 1.0:
            raise ValueError(
                f"online_sample_ratio must be in (0, 1); got {online_sample_ratio}"
            )

        primary_indices: list[int] = []
        online_indices: list[int] = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            frames = list(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))
            if episode_idx < n_primary_episodes:
                primary_indices.extend(frames)
            else:
                online_indices.extend(frames)

        if not primary_indices:
            raise ValueError("WeightedEpisodeAwareSampler: primary bucket is empty")
        if not online_indices:
            raise ValueError("WeightedEpisodeAwareSampler: online bucket is empty")

        self.primary_indices = torch.as_tensor(primary_indices, dtype=torch.long)
        self.online_indices = torch.as_tensor(online_indices, dtype=torch.long)
        self.online_sample_ratio = float(online_sample_ratio)
        self._num_samples = (
            num_samples if num_samples is not None else len(primary_indices) + len(online_indices)
        )

    def __iter__(self) -> Iterator[int]:
        n = self._num_samples
        is_online = torch.rand(n) < self.online_sample_ratio
        primary_picks = torch.randint(len(self.primary_indices), (n,))
        online_picks = torch.randint(len(self.online_indices), (n,))
        for i in range(n):
            if bool(is_online[i]):
                yield int(self.online_indices[online_picks[i]])
            else:
                yield int(self.primary_indices[primary_picks[i]])

    def __len__(self) -> int:
        return self._num_samples
