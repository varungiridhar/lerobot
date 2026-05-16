"""Dataset wrapper that attaches per-sample reward labels for the Q-function.

LeRobot port of ``imitation/dataset/utils/q_value_labels.py`` from the reference
repo (``varun_onsite_imitation``). Wraps a ``LeRobotDataset`` or
``MultiLeRobotDataset`` and injects, for each frame at global index ``i``, the
keys that ``QFunctionPolicy.forward`` expects:

* ``q_reward_chunk_first`` (h,) float — synthesized r_t … r_{t+h-1}
* ``q_reward_pad_first``   (h,) bool  — True where slot is past the end of the episode
* ``q_bootstrap_valid``    () bool    — True iff s_{t+h} exists and is non-terminal
* ``q_bucket_index``       () long    — ordinal index into ``bucket_order`` (for logging)

Reward modes
------------
* ``"sparse"`` (default for sim v1): every in-episode frame gets ``step_reward``;
  the terminal frame additionally gets ``terminal_bonuses[bucket]``.
* ``"time_to_go"``: for non-play buckets, each in-episode frame at position
  ``t`` of an episode of length ``T`` receives ``quality_scalars[bucket] * -(T - t)``;
  terminal bonus is added unscaled. Play always falls back to sparse.

Bucket inference
----------------
Bucket is parsed from the source dataset's ``repo_id`` at wrapper-construction
time. The MimicGen v1 convention used by this repo is:

    mg_<task>_<bucket>     # e.g. mg_coffee_q5, mg_coffee_q3_termjitter, mg_coffee_play

The wrapper maps ``dataset_index`` (provided by ``MultiLeRobotDataset.__getitem__``)
to a bucket via this parse, so each frame's reward synthesis uses the right
terminal bonus and quality scalar.

Multi-dataset episode boundaries
--------------------------------
``MultiLeRobotDataset`` does not expose a global ``episode_data_index``. We
build one at init by walking each sub-dataset's ``episode_data_index`` and
adding the cumulative frame offset.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset

log = logging.getLogger(__name__)

# v1 sim-data buckets, in the canonical order used for ``q_bucket_index``.
DEFAULT_BUCKET_ORDER: tuple[str, ...] = ("q5", "q3_termjitter", "play")

# Repo_id pattern: "mg_<task>_<bucket>" where bucket ∈ DEFAULT_BUCKET_ORDER.
# Match the LONGEST suffix that's a known bucket so e.g. "mg_coffee_q3_termjitter"
# resolves to bucket="q3_termjitter" and not "q3" (we don't even have a q3 bucket
# but someone might add one later).
_BUCKETS_BY_LENGTH = sorted(DEFAULT_BUCKET_ORDER, key=len, reverse=True)
_REPO_ID_BUCKET_RE = re.compile(
    r"_(" + "|".join(re.escape(b) for b in _BUCKETS_BY_LENGTH) + r")$"
)


def parse_bucket_from_repo_id(repo_id: str) -> str:
    """Return the bucket label implied by the trailing portion of ``repo_id``.

    Accepts e.g. ``mg_coffee_q5`` → ``q5``, ``mg_coffee_q3_termjitter`` →
    ``q3_termjitter``, ``mg_coffee_play`` → ``play``.
    """
    stripped = repo_id.rstrip("/").split("/")[-1]
    m = _REPO_ID_BUCKET_RE.search(stripped)
    if m:
        return m.group(1)
    raise ValueError(
        f"Cannot infer bucket from repo_id {repo_id!r}. "
        f"Expected suffix to be one of {DEFAULT_BUCKET_ORDER}."
    )


def _episode_bounds(sub_dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sub-dataset (ep_from, ep_to) longs from ``meta.episodes`` columns."""
    eps = sub_dataset.meta.episodes
    ep_from = torch.tensor(list(eps["dataset_from_index"]), dtype=torch.long)
    ep_to = torch.tensor(list(eps["dataset_to_index"]), dtype=torch.long)
    return ep_from, ep_to


def _build_global_episode_index(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (ep_from, ep_to) tensors over the full (possibly multi-) dataset.

    Reads from ``meta.episodes["dataset_{from,to}_index"]`` (the canonical
    LeRobot v3 layout) and adds cumulative frame offsets for multi-dataset.
    """
    if isinstance(dataset, MultiLeRobotDataset):
        froms, tos = [], []
        offset = 0
        for sub in dataset._datasets:
            f, t = _episode_bounds(sub)
            froms.append(f + offset)
            tos.append(t + offset)
            offset += int(sub.num_frames)
        return torch.cat(froms), torch.cat(tos)
    return _episode_bounds(dataset)


class _MultiDatasetMetaProxy:
    """Proxies the first sub-dataset's meta but overrides ``stats`` with the
    aggregated multi-dataset stats. Used when the wrapped dataset is a
    ``MultiLeRobotDataset`` and the rest of lerobot-train expects ``.meta``.
    """

    def __init__(self, multi: MultiLeRobotDataset):
        self._first = multi._datasets[0].meta
        self._stats = multi.stats

    @property
    def stats(self):
        return self._stats

    def __getattr__(self, name):
        return getattr(self._first, name)


def _repo_ids_of(dataset) -> list[str]:
    if isinstance(dataset, MultiLeRobotDataset):
        return list(dataset.repo_ids)
    if isinstance(dataset, LeRobotDataset):
        return [dataset.repo_id]
    return list(getattr(dataset, "repo_ids", None) or [getattr(dataset, "repo_id")])


class QValueLabelDataset(Dataset):
    """Wrap a LeRobot dataset to inject Q-function reward labels per sample."""

    def __init__(
        self,
        dataset,
        h: int,
        step_reward: float,
        terminal_bonuses: dict[str, float],
        bucket_order: Sequence[str] = DEFAULT_BUCKET_ORDER,
        reward_mode: str = "sparse",
        quality_scalars: dict[str, float] | None = None,
        load_preencoded: bool = True,
        precache_root: str | Path | None = None,
        terminal_bonus_uniform: float = 1.0,
    ):
        if h <= 0:
            raise ValueError(f"h must be positive, got {h}")
        if reward_mode not in ("sparse", "time_to_go", "all_success"):
            raise ValueError(
                f"reward_mode must be 'sparse', 'time_to_go', or 'all_success', got {reward_mode!r}"
            )

        self.dataset = dataset
        self.h = int(h)
        self.step_reward = float(step_reward)
        self.terminal_bonuses = dict(terminal_bonuses)
        self.bucket_order = tuple(bucket_order)
        self._bucket_to_ordinal = {b: i for i, b in enumerate(self.bucket_order)}
        self.reward_mode = reward_mode
        self.quality_scalars = dict(quality_scalars) if quality_scalars is not None else {}
        self.terminal_bonus_uniform = float(terminal_bonus_uniform)
        self._all_success = reward_mode == "all_success"

        if not self._all_success:
            repo_ids = _repo_ids_of(dataset)
            self._dataset_index_to_bucket: list[str] = [parse_bucket_from_repo_id(r) for r in repo_ids]

            missing_bonuses = set(self._dataset_index_to_bucket) - set(self.terminal_bonuses.keys())
            if missing_bonuses:
                raise ValueError(
                    f"terminal_bonuses missing entries for buckets present in the data: "
                    f"{sorted(missing_bonuses)}"
                )
            if self.reward_mode == "time_to_go":
                needed = {b for b in self._dataset_index_to_bucket if b != "play"}
                missing_scalars = needed - set(self.quality_scalars.keys())
                if missing_scalars:
                    raise ValueError(
                        "reward_mode='time_to_go' requires quality_scalars for all non-play "
                        f"buckets; missing: {sorted(missing_scalars)}"
                    )

        # Episode boundaries over the combined (possibly multi-) dataset.
        self._ep_from, self._ep_to = _build_global_episode_index(dataset)
        num_episodes = int(self._ep_from.shape[0])
        num_frames = int(dataset.num_frames)

        # frame_idx → episode_position lookup.
        ep_pos_by_frame = torch.empty(num_frames, dtype=torch.int32)
        for ep_pos in range(num_episodes):
            ep_pos_by_frame[int(self._ep_from[ep_pos]) : int(self._ep_to[ep_pos])] = ep_pos
        self._ep_pos_by_frame = ep_pos_by_frame

        # frame_idx → dataset_index lookup (all 0s for single-dataset).
        # Plus per-sub-dataset cumulative frame offset for converting global → local idx.
        dataset_idx_by_frame = torch.zeros(num_frames, dtype=torch.int32)
        cum_offsets = [0]
        if isinstance(dataset, MultiLeRobotDataset):
            cursor = 0
            for ds_idx, sub in enumerate(dataset._datasets):
                n = int(sub.num_frames)
                dataset_idx_by_frame[cursor : cursor + n] = ds_idx
                cursor += n
                cum_offsets.append(cursor)
            if cursor != num_frames:
                raise RuntimeError(
                    f"Multi-dataset frame accounting mismatch: cursor={cursor}, num_frames={num_frames}"
                )
        else:
            cum_offsets.append(num_frames)
        self._dataset_idx_by_frame = dataset_idx_by_frame
        self._cum_offsets = cum_offsets   # length = num_sub_datasets + 1

        # ── Optional pre-encoded feature cache ─────────────────────────────
        # When ``precache_root`` is provided, look for
        # ``<precache_root>/<sub_repo_id>/meta.json`` per sub-dataset. All
        # sub-datasets must have a cache or we raise (mixed coverage is
        # rejected — would be confusing). Layout matches what
        # ``src/lerobot/policies/act_simple/precache_features.py`` writes.
        self._preencoded_mmaps: list[dict[str, np.memmap]] | None = None
        self._preencoded_shape: tuple[int, ...] | None = None
        self._precache_root = Path(precache_root) if precache_root is not None else None
        if load_preencoded and self._precache_root is not None:
            self._preencoded_mmaps = self._maybe_load_preencoded(dataset)

    # ── Pre-encoded feature cache ──────────────────────────────────────────

    def _maybe_load_preencoded(self, dataset) -> list[dict[str, np.memmap]] | None:
        """Per-sub-dataset memmaps under ``self._precache_root/<repo_id>/``.

        Requires ALL sub-datasets to have a cache (no partial coverage), with
        matching camera_keys, feature_shape, and dtype across them.
        """
        subs = dataset._datasets if isinstance(dataset, MultiLeRobotDataset) else [dataset]
        sub_repo_ids = [getattr(s, "repo_id", None) for s in subs]
        if any(rid is None for rid in sub_repo_ids):
            raise RuntimeError("All sub-datasets must expose .repo_id for precache lookup.")
        meta_paths = [self._precache_root / rid / "meta.json" for rid in sub_repo_ids]
        present = [p.exists() for p in meta_paths]
        if not any(present):
            log.info(
                "QValueLabelDataset: precache_root=%s configured but no sub-dataset has a "
                "cache (looked for %s). Falling back to no cached features.",
                self._precache_root, [str(p) for p in meta_paths],
            )
            return None
        if not all(present):
            missing = [str(p) for p, ok in zip(meta_paths, present) if not ok]
            raise FileNotFoundError(
                "Partial precache coverage across sub-datasets; "
                f"missing: {missing}. Run precache for all sub-datasets or none."
            )
        metas = [json.loads(p.read_text()) for p in meta_paths]
        # Reference values from the first meta; require the rest to match.
        ref_shape = tuple(metas[0]["feature_shape"])
        ref_dtype = metas[0]["dtype"]
        ref_cams = sorted(metas[0]["camera_keys"])
        for i, m in enumerate(metas[1:], start=1):
            if tuple(m["feature_shape"]) != ref_shape:
                raise ValueError(f"feature_shape mismatch at sub-dataset {i}: {m['feature_shape']} vs {ref_shape}")
            if m["dtype"] != ref_dtype:
                raise ValueError(f"dtype mismatch at sub-dataset {i}: {m['dtype']} vs {ref_dtype}")
            if sorted(m["camera_keys"]) != ref_cams:
                raise ValueError(f"camera_keys mismatch at sub-dataset {i}: {m['camera_keys']} vs {ref_cams}")
        if ref_dtype != "float16":
            raise ValueError(f"only float16 caches are supported; got {ref_dtype!r}")
        self._preencoded_shape = ref_shape

        per_sub_mmaps: list[dict[str, np.memmap]] = []
        for sub, meta_path, m in zip(subs, meta_paths, metas):
            n = int(sub.num_frames)
            if int(m["n_frames"]) != n:
                raise ValueError(
                    f"cache n_frames={m['n_frames']} != sub.num_frames={n} at {meta_path}"
                )
            sub_dir = meta_path.parent
            mm_for_sub: dict[str, np.memmap] = {}
            for cam in ref_cams:
                fpath = sub_dir / m["files"][cam]
                if not fpath.exists():
                    raise FileNotFoundError(f"missing cache file {fpath}")
                mm_for_sub[cam] = np.memmap(str(fpath), dtype="float16", mode="r", shape=(n, *ref_shape))
            per_sub_mmaps.append(mm_for_sub)

        log.info(
            "QValueLabelDataset: loaded encoded_backbone caches for %d sub-datasets, "
            "cams=%s, feature_shape=%s",
            len(per_sub_mmaps), ref_cams, ref_shape,
        )
        return per_sub_mmaps

    # ── Size / indexing ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        ep_pos = int(self._ep_pos_by_frame[idx].item())
        ep_start = int(self._ep_from[ep_pos].item())
        ep_end = int(self._ep_to[ep_pos].item())
        ep_length = ep_end - ep_start
        frame_in_ep = idx - ep_start
        terminal_idx_in_ep = ep_length - 1

        reward_chunk = torch.zeros(self.h, dtype=torch.float32)
        reward_pad = torch.zeros(self.h, dtype=torch.bool)

        if self._all_success:
            for i in range(self.h):
                f = frame_in_ep + i
                if f < ep_length:
                    reward_chunk[i] = self.terminal_bonus_uniform if f == terminal_idx_in_ep else 0.0
                else:
                    reward_pad[i] = True
            bucket_index = 0
        else:
            ds_idx = int(self._dataset_idx_by_frame[idx].item())
            bucket = self._dataset_index_to_bucket[ds_idx]
            terminal_bonus = float(self.terminal_bonuses[bucket])
            use_time_to_go = (self.reward_mode == "time_to_go") and (bucket != "play")
            scalar = self.quality_scalars.get(bucket, 1.0) if use_time_to_go else 0.0
            for i in range(self.h):
                f = frame_in_ep + i
                if f < ep_length:
                    r = scalar * -(ep_length - f) if use_time_to_go else self.step_reward
                    if f == terminal_idx_in_ep:
                        r += terminal_bonus
                    reward_chunk[i] = r
                else:
                    reward_pad[i] = True
            bucket_index = self._bucket_to_ordinal.get(bucket, -1)

        bootstrap_valid = (frame_in_ep + self.h) < (ep_length - 1)

        item["q_reward_chunk_first"] = reward_chunk
        item["q_reward_pad_first"] = reward_pad
        item["q_bootstrap_valid"] = torch.tensor(bool(bootstrap_valid))
        item["q_bucket_index"] = torch.tensor(bucket_index, dtype=torch.long)

        # ── Pre-encoded features at delta indices [0, h] ───────────────────
        # We mirror the underlying dataset's observation_delta_indices=[0, h]
        # by indexing the per-sub-dataset mmap at the same absolute frames and
        # stacking. For frames where t+h would cross the episode boundary, we
        # clamp to the last in-episode frame; bootstrap_valid=False masks that
        # contribution out at the loss level, so the cached feature value
        # there is don't-care.
        if self._preencoded_mmaps is not None:
            ds_idx = int(self._dataset_idx_by_frame[idx].item())
            sub_offset = self._cum_offsets[ds_idx]
            local_idx_t = idx - sub_offset
            ep_end_global = ep_end - 1                         # last valid in-ep frame
            idx_tph_clamped = min(idx + self.h, ep_end_global)
            local_idx_tph = idx_tph_clamped - sub_offset
            sub_mm = self._preencoded_mmaps[ds_idx]
            for cam, mm in sub_mm.items():
                feat_t = np.asarray(mm[local_idx_t], dtype=np.float32)        # (C, Hf, Wf)
                feat_tph = np.asarray(mm[local_idx_tph], dtype=np.float32)
                stack = np.stack([feat_t, feat_tph], axis=0)                  # (2, C, Hf, Wf)
                item[f"{cam}_preencoded"] = torch.from_numpy(stack)

        return item

    # ── Forwarded attributes ───────────────────────────────────────────────

    @property
    def meta(self):
        if isinstance(self.dataset, MultiLeRobotDataset):
            if not hasattr(self, "_meta_proxy"):
                self._meta_proxy = _MultiDatasetMetaProxy(self.dataset)
            return self._meta_proxy
        return getattr(self.dataset, "meta", None)

    @property
    def num_frames(self) -> int:
        return self.dataset.num_frames

    @property
    def num_episodes(self) -> int:
        return self.dataset.num_episodes

    @property
    def episode_data_index(self):
        # Old-style API; not present in current LeRobot v3 — use ``meta.episodes``.
        return getattr(self.dataset, "episode_data_index", None)

    @property
    def episodes(self):
        return getattr(self.dataset, "episodes", None)

    @property
    def repo_id(self):
        return getattr(self.dataset, "repo_id", None)

    @property
    def repo_ids(self):
        return getattr(self.dataset, "repo_ids", None)

    @property
    def stats(self):
        return getattr(self.dataset, "stats", None)

    @property
    def image_transforms(self):
        return self.dataset.image_transforms

    @image_transforms.setter
    def image_transforms(self, value):
        self.dataset.image_transforms = value

    @property
    def features(self):
        return self.dataset.features

    @property
    def fps(self):
        return self.dataset.fps

    @property
    def camera_keys(self):
        return getattr(self.dataset, "camera_keys", None)

    def __getattr__(self, name):
        # Delegate any other attribute access to the wrapped dataset.
        return getattr(self.dataset, name)

    # ── Diagnostics ────────────────────────────────────────────────────────

    def bucket_counts(self) -> dict[str, int]:
        if self._all_success:
            return {"success": len(self)}
        counts: dict[str, int] = {b: 0 for b in self.bucket_order}
        for ds_idx, bucket in enumerate(self._dataset_index_to_bucket):
            n = int((self._dataset_idx_by_frame == ds_idx).sum().item())
            counts[bucket] = counts.get(bucket, 0) + n
        return counts
