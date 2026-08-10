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

Bucket assignment
-----------------
Bucket is assigned **explicitly** via the ``bucket_overrides`` dict passed at
wrapper-construction time (sourced from ``policy.bucket_overrides`` in the
config). Every sub-dataset's ``repo_id`` must appear as a key in
``bucket_overrides``; the wrapper maps ``dataset_index`` to its bucket via
this dict, so each frame's reward synthesis uses the right terminal bonus
and quality scalar. Missing entries raise at construction time.

No implicit bucket inference from repo_id — pass everything explicitly so
the assignment is auditable from the launch script alone.

Per-episode success labels (real-robot datasets)
------------------------------------------------
A repo may map to the sentinel bucket ``"episode_labels"`` instead of a real
bucket name. The wrapper then reads ``<root>/meta/episode_labels.json`` (the
leLab operator sidecar, ``{"<ep_idx>": {"success": bool, "outcome": str}}``)
and assigns buckets per *episode*: success → ``EPISODE_LABEL_SUCCESS_BUCKET``
(``"q5"``), failure → ``EPISODE_LABEL_FAILURE_BUCKET`` (``"play"``). Failed
episodes thus become zero-return TD trajectories (real negatives) under the
usual ``terminal_bonuses={"q5": 1.0, "play": 0.0}``. The train/test holdout
is stratified per (repo, resolved bucket), so both success and failure
episodes appear in the held-out split. Every episode index must be present
in the sidecar; missing entries raise at construction time.

Multi-dataset episode boundaries
--------------------------------
``MultiLeRobotDataset`` does not expose a global ``episode_data_index``. We
build one at init by walking each sub-dataset's ``episode_data_index`` and
adding the cumulative frame offset.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset

log = logging.getLogger(__name__)

# Canonical bucket-name ordering for the ``q_bucket_index`` field logged with
# each batch. New bucket labels can be added at the tail.
DEFAULT_BUCKET_ORDER: tuple[str, ...] = ("q5", "q3_termjitter", "play")


# Sentinel bucket_overrides value: resolve buckets per episode from the
# dataset's ``meta/episode_labels.json`` sidecar instead of per repo.
# Defined in the (import-light) configuration module so the deployment fork's
# inference-only port can validate checkpoints without this training module.
from lerobot.policies.q_function.configuration_q_function import EPISODE_LABELS_SENTINEL  # noqa: E402
EPISODE_LABEL_SUCCESS_BUCKET = "q5"
EPISODE_LABEL_FAILURE_BUCKET = "play"


def _load_episode_label_buckets(sub_dataset) -> list[str]:
    """Per-local-episode bucket list from the leLab ``episode_labels.json`` sidecar."""
    labels_path = Path(sub_dataset.root) / "meta" / "episode_labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"bucket_overrides maps {sub_dataset.repo_id!r} to {EPISODE_LABELS_SENTINEL!r} "
            f"but {labels_path} does not exist."
        )
    labels = json.loads(labels_path.read_text())
    n_eps = int(sub_dataset.meta.total_episodes)
    buckets: list[str] = []
    missing = [ep for ep in range(n_eps) if str(ep) not in labels]
    if missing:
        raise ValueError(
            f"{labels_path} is missing entries for episodes {missing[:10]}"
            f"{'…' if len(missing) > 10 else ''} (dataset has {n_eps} episodes)."
        )
    for ep in range(n_eps):
        entry = labels[str(ep)]
        buckets.append(
            EPISODE_LABEL_SUCCESS_BUCKET if bool(entry["success"]) else EPISODE_LABEL_FAILURE_BUCKET
        )
    n_fail = sum(b == EPISODE_LABEL_FAILURE_BUCKET for b in buckets)
    log.info(
        f"[q-labels] {sub_dataset.repo_id}: per-episode buckets from sidecar — "
        f"{n_eps - n_fail} success ({EPISODE_LABEL_SUCCESS_BUCKET}), "
        f"{n_fail} failure ({EPISODE_LABEL_FAILURE_BUCKET})"
    )
    return buckets


def resolve_bucket(repo_id: str, bucket_overrides: dict[str, str]) -> str:
    """Look up the bucket label for ``repo_id`` in ``bucket_overrides``.

    Explicit-only: every repo_id must have an entry. There is no implicit
    inference from the repo_id suffix anymore — pass everything explicitly
    via ``policy.bucket_overrides`` so the assignment is auditable from the
    launch script.
    """
    if repo_id not in bucket_overrides:
        raise ValueError(
            f"Repo {repo_id!r} has no bucket assignment. Add it to "
            f"policy.bucket_overrides. Currently set: "
            f"{sorted(bucket_overrides.keys())}."
        )
    return bucket_overrides[repo_id]


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
        bucket_overrides: dict[str, str] | None = None,
        holdout_fraction: float = 0.0,
        holdout_seed: int | None = None,
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

        self.bucket_overrides = dict(bucket_overrides) if bucket_overrides else {}

        self.holdout_fraction = float(holdout_fraction)
        self.holdout_seed = holdout_seed
        if self.holdout_fraction < 0.0 or self.holdout_fraction >= 1.0:
            raise ValueError(
                f"holdout_fraction must be in [0.0, 1.0), got {self.holdout_fraction!r}"
            )

        repo_ids_for_split: list[str] | None = None
        self._bucket_by_global_ep: list[str] | None = None
        self._local_ep_buckets_by_ds: list[list[str] | None] | None = None
        if not self._all_success:
            repo_ids = _repo_ids_of(dataset)
            repo_ids_for_split = repo_ids
            self._dataset_index_to_bucket: list[str] = [
                resolve_bucket(r, self.bucket_overrides) for r in repo_ids
            ]

            # Resolve the per-episode sidecar sentinel (see module docstring).
            # _bucket_by_global_ep is the single source of truth for a frame's
            # bucket; for plain per-repo assignments it is just the repo bucket
            # repeated over that repo's episodes.
            sub_datasets = (
                list(dataset._datasets) if isinstance(dataset, MultiLeRobotDataset) else [dataset]
            )
            self._local_ep_buckets_by_ds = []
            bucket_by_global_ep: list[str] = []
            for ds_idx, sub in enumerate(sub_datasets):
                repo_bucket = self._dataset_index_to_bucket[ds_idx]
                if repo_bucket == EPISODE_LABELS_SENTINEL:
                    local_buckets = _load_episode_label_buckets(sub)
                else:
                    local_buckets = [repo_bucket] * int(sub.meta.total_episodes)
                self._local_ep_buckets_by_ds.append(local_buckets)
                bucket_by_global_ep.extend(local_buckets)
            self._bucket_by_global_ep = bucket_by_global_ep

            resolved_buckets = set(bucket_by_global_ep)
            missing_bonuses = resolved_buckets - set(self.terminal_bonuses.keys())
            if missing_bonuses:
                raise ValueError(
                    f"terminal_bonuses missing entries for buckets present in the data: "
                    f"{sorted(missing_bonuses)}"
                )
            if self.reward_mode == "time_to_go":
                needed = {b for b in resolved_buckets if b != "play"}
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

        # ── Optional episode-level train/test split ───────────────────────────
        # Stratified per (repo_id, bucket): the same fraction of episodes is
        # carved out from every sub-dataset, deterministically seeded so the
        # split survives resumes and re-launches with the same cfg.seed.
        # Train/test selection happens at the sampler layer (see
        # ``train_frame_indices`` / ``test_frame_indices``); __getitem__ is
        # unchanged so a single wrapper instance serves both loaders.
        self._frame_indices_train: list[int] = []
        self._frame_indices_test: list[int] = []
        self._train_episode_global_ids: list[int] = []
        self._test_episode_global_ids: list[int] = []
        if self.holdout_fraction > 0.0:
            if self._all_success or repo_ids_for_split is None:
                raise ValueError(
                    "holdout_fraction>0 requires non-all_success mode (the wrapper "
                    "needs per-repo bucket assignments to stratify the split). "
                    "Disable the holdout or switch reward_mode."
                )
            import random as _random
            sub_iter = (
                dataset._datasets if isinstance(dataset, MultiLeRobotDataset) else [dataset]
            )
            offset = 0
            global_ep_idx = 0
            for ds_idx, sub in enumerate(sub_iter):
                repo_id = repo_ids_for_split[ds_idx]
                ep_from_local, ep_to_local = _episode_bounds(sub)
                local_n = int(ep_from_local.shape[0])
                local_buckets = self._local_ep_buckets_by_ds[ds_idx]
                # Stratify per (repo, resolved bucket). For plain per-repo
                # assignments there is exactly one group and this reduces to
                # the original single-shuffle behavior (identical rng seed).
                test_local: set[int] = set()
                groups: dict[str, list[int]] = {}
                for local_ep_id, b in enumerate(local_buckets):
                    groups.setdefault(b, []).append(local_ep_id)
                for bucket, ep_ids in groups.items():
                    rng = _random.Random(hash((self.holdout_seed, repo_id, bucket)) & 0xFFFFFFFF)
                    order = ep_ids.copy()
                    rng.shuffle(order)
                    n_test = max(1, int(round(len(ep_ids) * self.holdout_fraction)))
                    n_test = min(n_test, len(ep_ids) - 1)   # keep >=1 train ep per group
                    if n_test <= 0:
                        log.warning(
                            f"[q-split] repo={repo_id} bucket={bucket} has only "
                            f"{len(ep_ids)} episode(s); nothing held out for this group."
                        )
                        continue
                    test_local.update(order[:n_test])
                    log.info(
                        f"[q-split] repo={repo_id} bucket={bucket} "
                        f"train_eps={len(ep_ids) - n_test} test_eps={n_test} "
                        f"(holdout={self.holdout_fraction:.0%}, seed={self.holdout_seed})"
                    )
                for local_ep_id in range(local_n):
                    f = int(ep_from_local[local_ep_id].item())
                    t = int(ep_to_local[local_ep_id].item())
                    gids = list(range(offset + f, offset + t))
                    if local_ep_id in test_local:
                        self._test_episode_global_ids.append(global_ep_idx)
                        self._frame_indices_test.extend(gids)
                    else:
                        self._train_episode_global_ids.append(global_ep_idx)
                        self._frame_indices_train.extend(gids)
                    global_ep_idx += 1
                offset += int(sub.num_frames)

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

    # ── Train/test split accessors ─────────────────────────────────────────

    @property
    def train_frame_indices(self) -> list[int]:
        """Global frame indices belonging to train episodes (empty if no holdout)."""
        return self._frame_indices_train

    @property
    def test_frame_indices(self) -> list[int]:
        """Global frame indices belonging to held-out test episodes."""
        return self._frame_indices_test

    @property
    def test_episode_ids(self) -> list[int]:
        """Global episode indices held out as the test split (empty if no holdout).

        Index into ``self._ep_from`` / ``self._ep_to`` to get a test episode's
        global frame range — used by the test-set Q-value visualization.
        """
        return self._test_episode_global_ids

    @property
    def has_holdout(self) -> bool:
        """True if a non-empty episode-level holdout was carved at construction."""
        return self.holdout_fraction > 0.0 and len(self._frame_indices_test) > 0

    def balanced_train_frame_indices(
        self, weights: dict[str, float], seed: int | None = None
    ) -> list[int]:
        """Static per-bucket resampling of the train split (weight 1.0 = natural).

        Integer part of the weight duplicates an index; the fractional part is a
        Bernoulli draw (seeded — deterministic across resumes/ranks). Weights < 1
        subsample. Returns a shuffled index list for a plain ``Subset`` +
        ``shuffle=True`` DataLoader — keeping accelerate's even-batches DDP path
        (a custom sampler would desync ranks; see lerobot_train.py).
        """
        if self._all_success:
            raise ValueError("bucket_sample_weights requires per-repo bucket assignments.")
        import random as _random

        rng = _random.Random(seed if seed is not None else self.holdout_seed)
        base = self._frame_indices_train if self._frame_indices_train else list(range(len(self)))
        out: list[int] = []
        for idx in base:
            bucket = self._bucket_by_global_ep[int(self._ep_pos_by_frame[idx].item())]
            w = float(weights.get(bucket, 1.0))
            n = int(w) + (1 if rng.random() < (w - int(w)) else 0)
            out.extend([idx] * n)
        rng.shuffle(out)
        return out

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
            bucket = self._bucket_by_global_ep[ep_pos]
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
        # Episode phase + identity, consumed by the fraction-matched swap negative
        # (modeling_q_function._negatives_margin_loss): pair each anchor with the
        # batch sample at the nearest episode fraction from a DIFFERENT episode.
        item["q_episode_frac"] = torch.tensor(
            frame_in_ep / max(1, ep_length - 1), dtype=torch.float32
        )
        item["q_episode_id"] = torch.tensor(ep_pos, dtype=torch.long)

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
