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
"""Action tokenizers for AWM.

Two tokenizer variants are provided:

UniformActionTokenizer (default)
    Maps a D-dimensional action to a **single joint token** via mixed-radix encoding.
    Total vocabulary = V^D.  Compact representation; impractical for D ≥ 4 with V ≥ 32.

FactoredActionTokenizer
    Maps each action dimension independently to its own token in [0, V-1].
    Total vocabulary = V.  Decoder sequence grows from T to T*D tokens, but the
    embedding tables and action head remain small regardless of dimensionality.

AWMPolicy uses ``FactoredActionTokenizer`` by default.
"""

import torch
from torch import Tensor, nn


class UniformActionTokenizer(nn.Module):
    """Per-dimension uniform tokenizer with joint mixed-radix token encoding.

    Registered buffers (``lows``, ``highs``, ``radix``) follow the parent module to whatever
    device it is moved to, so encode/decode work transparently on CPU and GPU.

    Args:
        action_ranges: List of ``[lo, hi]`` pairs, one per action dimension.
        vocab_size: Number of bins per dimension.  Total vocabulary = ``vocab_size ** action_dim``.
    """

    def __init__(self, action_ranges: list[list[float]], vocab_size: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.action_dim = len(action_ranges)
        self.total_vocab_size = vocab_size**self.action_dim

        lows = torch.tensor([r[0] for r in action_ranges], dtype=torch.float32)
        highs = torch.tensor([r[1] for r in action_ranges], dtype=torch.float32)
        # Mixed-radix multipliers: [1, V, V^2, …]
        radix = torch.tensor([vocab_size**d for d in range(self.action_dim)], dtype=torch.long)

        self.register_buffer("lows", lows)
        self.register_buffer("highs", highs)
        self.register_buffer("radix", radix)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, actions: Tensor) -> Tensor:
        """Quantise continuous actions to joint token indices.

        Args:
            actions: ``(…, action_dim)`` float tensor of continuous actions.

        Returns:
            ``(…,)`` long tensor of joint token indices in ``[0, total_vocab_size - 1]``.
        """
        # Normalise each dimension to [0, 1], then scale to [0, vocab_size).
        # Clamp to keep bin indices in-range even for out-of-bound actions.
        span = self.highs - self.lows  # (D,)
        normalised = (actions - self.lows) / span  # (…, D)  in [0, 1]
        bin_indices = (normalised * self.vocab_size).long().clamp(0, self.vocab_size - 1)  # (…, D)

        # Mixed-radix encoding: sum b_d * V^d over d.
        joint = (bin_indices * self.radix).sum(dim=-1)  # (…,)
        return joint

    def decode(self, token_ids: Tensor) -> Tensor:
        """Convert joint token indices back to continuous action values (bin centres).

        Args:
            token_ids: ``(…,)`` long tensor of joint token indices.

        Returns:
            ``(…, action_dim)`` float tensor of continuous actions.
        """
        bin_width = (self.highs - self.lows) / self.vocab_size  # (D,)

        parts = []
        remaining = token_ids.clone()
        for d in range(self.action_dim):
            bin_idx = remaining % self.vocab_size  # (…,)
            remaining = remaining // self.vocab_size
            # Map bin index to the centre of that bin.
            value = self.lows[d] + (bin_idx.float() + 0.5) * bin_width[d]
            parts.append(value)

        return torch.stack(parts, dim=-1)  # (…, D)

    # nn.Module forward is unused; raise to catch accidental calls.
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Call encode() or decode() directly.")


class FactoredActionTokenizer(nn.Module):
    """Per-dimension uniform tokenizer — one token per action dimension.

    Each action dimension d is independently quantised into ``vocab_size`` uniformly-spaced bins
    within its configured range [lo, hi].  Unlike :class:`UniformActionTokenizer`, the
    per-dimension bin indices are **not** combined into a joint token; each dimension produces its
    own token in ``[0, vocab_size-1]``.

    This keeps the vocabulary at size ``V`` regardless of the action dimensionality D, at the cost
    of expanding the decoder sequence from length T to T*D (one sub-token per dimension per step).

    Registered buffers (``lows``, ``highs``) follow the parent module to whatever device it is
    moved to, so encode/decode work transparently on CPU and GPU.

    Args:
        action_ranges: List of ``[lo, hi]`` pairs, one per action dimension.
        vocab_size: Number of bins per dimension.  Total vocabulary = ``vocab_size`` (not V^D).
    """

    def __init__(self, action_ranges: list[list[float]], vocab_size: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.action_dim = len(action_ranges)
        # API compatibility with UniformActionTokenizer: total_vocab_size == vocab_size here.
        self.total_vocab_size = vocab_size

        lows = torch.tensor([r[0] for r in action_ranges], dtype=torch.float32)
        highs = torch.tensor([r[1] for r in action_ranges], dtype=torch.float32)

        self.register_buffer("lows", lows)
        self.register_buffer("highs", highs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, actions: Tensor) -> Tensor:
        """Quantise continuous actions to per-dimension token indices.

        Args:
            actions: ``(…, action_dim)`` float tensor of continuous actions.

        Returns:
            ``(…, action_dim)`` long tensor of per-dimension token indices in
            ``[0, vocab_size - 1]``.
        """
        span = self.highs - self.lows  # (D,)
        normalised = (actions - self.lows) / span  # (…, D) in [0, 1]
        bin_indices = (normalised * self.vocab_size).long().clamp(0, self.vocab_size - 1)  # (…, D)
        return bin_indices

    def decode(self, token_ids: Tensor) -> Tensor:
        """Convert per-dimension token indices back to continuous action values (bin centres).

        Args:
            token_ids: ``(…, action_dim)`` long tensor of per-dimension token indices.

        Returns:
            ``(…, action_dim)`` float tensor of continuous actions.
        """
        bin_width = (self.highs - self.lows) / self.vocab_size  # (D,)
        # Map each bin index to the centre of that bin.
        return self.lows + (token_ids.float() + 0.5) * bin_width  # (…, D)

    # nn.Module forward is unused; raise to catch accidental calls.
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Call encode() or decode() directly.")
