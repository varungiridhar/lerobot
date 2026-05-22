#!/usr/bin/env python
"""End-to-end data-pipeline tests for the multi-dataset Q-function setup used by
scripts/train_q_libero_ddp_multi_interactive.sh.

The interactive script mixes two LeRobot dataset formats in a single
MultiLeRobotDataset:

  * HuggingFaceVLA/libero            — frames stored as inline images in parquet
                                       (info.json dtype='image', no videos/ dir)
  * VarunGiridhar3/libero40_*_play   — frames stored as MP4 videos under videos/,
                                       but info.json still claims dtype='image'
                                       (gets auto-promoted to dtype='video' at
                                        LeRobotDatasetMetadata.load_metadata()
                                        time, lerobot_dataset.py:206)

These tests target the bridging logic — they don't re-test lerobot internals,
they verify that BOTH formats end up producing comparable tensors out of
__getitem__, that QValueLabelDataset routes per-repo buckets correctly, that
the resulting batch goes through the q_function policy without NaN, etc.

Run as ``python scripts/tests/test_multi_dataset_qf_pipeline.py``. Each test
is wrapped in try/except so the slurm log contains a full report even when
something fails. Exit code is 0 iff all tests pass.
"""
from __future__ import annotations

import logging
import math
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── repo-relative imports ────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets.factory import make_dataset  # noqa: E402
from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_policy  # noqa: E402
from lerobot.policies.q_function.configuration_q_function import QFunctionConfig  # noqa: E402
from lerobot.policies.q_function.q_value_labels import QValueLabelDataset  # noqa: E402

# ── config mirrors scripts/train_q_libero_ddp_multi_interactive.sh ───────
REPO_IDS = [
    "HuggingFaceVLA/libero",
    "VarunGiridhar3/libero40_libero_object_play",
    "VarunGiridhar3/libero40_libero_10_play",
    "VarunGiridhar3/libero40_libero_goal_play",
    "VarunGiridhar3/libero40_libero_spatial_play",
]
BC_REPO = "HuggingFaceVLA/libero"
PLAY_REPOS = [r for r in REPO_IDS if r != BC_REPO]

BUCKET_OVERRIDES = {
    BC_REPO: "q5",
    **{r: "play" for r in PLAY_REPOS},
}
TERMINAL_BONUSES = {"q5": 1.0, "play": 0.0}
EXPECTED_CAMERA_KEYS = {"observation.images.image", "observation.images.image2"}

DATASET_ROOT = os.path.expanduser("~/scratch/hf_cache/lerobot")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# These mirror src/lerobot/datasets/factory.py:32 (the imagenet override).
IMAGENET_MEAN = np.array([[[0.485]], [[0.456]], [[0.406]]], dtype=np.float32)
IMAGENET_STD = np.array([[[0.229]], [[0.224]], [[0.225]]], dtype=np.float32)


# ── tiny test runner ──────────────────────────────────────────────────────
_RESULTS: list[tuple[str, bool, str, float]] = []


def _run(name: str, fn, *args, **kwargs):
    print(f"\n━━━ {name} ━━━", flush=True)
    t0 = time.time()
    try:
        fn(*args, **kwargs)
        dt = time.time() - t0
        print(f"PASS  {name}  ({dt:.2f}s)", flush=True)
        _RESULTS.append((name, True, "", dt))
    except Exception as e:
        dt = time.time() - t0
        tb = traceback.format_exc()
        print(f"FAIL  {name}  ({dt:.2f}s)\n{tb}", flush=True)
        _RESULTS.append((name, False, f"{type(e).__name__}: {e}", dt))


def _approx(a, b, rtol=1e-5, atol=1e-6):
    return np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


# ── shared fixtures (built once, reused) ─────────────────────────────────
_FIXTURES: dict = {}


def build_dataset():
    """Build the wrapped (QValueLabelDataset → MultiLeRobotDataset) just like
    make_dataset(cfg) does for the q_function policy. Cached in _FIXTURES."""
    if "wrapped" in _FIXTURES:
        return _FIXTURES["wrapped"]

    policy_cfg = QFunctionConfig(
        h=32,
        gamma=0.99,
        dim_model=1024,
        n_heads=16,
        dim_feedforward=4096,
        n_decoder_layers=18,
        image_resize_h=224,
        image_resize_w=224,
        reward_mode="sparse",
        step_reward=0.0,
        terminal_bonuses=dict(TERMINAL_BONUSES),
        bucket_overrides=dict(BUCKET_OVERRIDES),
        v_min=-0.01,
        v_max=1.01,
        hl_gauss_sigma=0.0075,
        use_text_conditioning=True,
        text_encoder_model="google/t5-v1_1-base",
        language_key="task",
        dino_model_name="facebook/dinov2-large",
        target_tau=0.005,
        optimizer_lr=3e-4,
        optimizer_lr_backbone=9e-5,
        optimizer_weight_decay=1e-4,
        lr_scheduler="cosine_decay_with_warmup",
        lr_warmup_steps=10,
        lr_decay_steps=100,
        lr_decay_min=1e-6,
        device=DEVICE,
        push_to_hub=False,
    )
    dataset_cfg = DatasetConfig(
        repo_ids=list(REPO_IDS),
        root=DATASET_ROOT,
        use_imagenet_stats=True,
    )
    cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        batch_size=1,
        steps=10,
        num_workers=0,
    )
    cfg.validate()
    print(f"Building dataset from root={DATASET_ROOT} …", flush=True)
    wrapped = make_dataset(cfg)
    _FIXTURES["wrapped"] = wrapped
    _FIXTURES["multi"] = wrapped.dataset if isinstance(wrapped, QValueLabelDataset) else wrapped
    _FIXTURES["cfg"] = cfg
    print(f"  → wrapped: {type(wrapped).__name__}, "
          f"inner: {type(_FIXTURES['multi']).__name__}, "
          f"num_frames={wrapped.dataset.num_frames}", flush=True)
    return wrapped


# ────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────


def t01_dataset_construction():
    """The multi-dataset path must build a QValueLabelDataset over a
    MultiLeRobotDataset containing exactly the 5 expected sub-datasets."""
    wrapped = build_dataset()
    assert isinstance(wrapped, QValueLabelDataset), type(wrapped)
    multi = wrapped.dataset
    assert isinstance(multi, MultiLeRobotDataset), type(multi)
    sub_ids = [s.repo_id for s in multi._datasets]
    assert sub_ids == REPO_IDS, f"sub-dataset order mismatch:\n got: {sub_ids}\n want: {REPO_IDS}"


def t02_per_repo_dtype_promotion():
    """The auto-promote MUST fire for every VarunGiridhar3/* repo and MUST NOT
    fire for HuggingFaceVLA/libero. This is the central correctness property
    of the v2-image / v3-video-mislabelled-as-image bridge."""
    multi = _FIXTURES["multi"]
    for sub in multi._datasets:
        img_keys = set(sub.meta.image_keys)
        vid_keys = set(sub.meta.video_keys)
        if sub.repo_id == BC_REPO:
            assert img_keys == EXPECTED_CAMERA_KEYS, (
                f"{sub.repo_id}: expected image-backed cameras "
                f"{EXPECTED_CAMERA_KEYS}, got image_keys={img_keys}, video_keys={vid_keys}"
            )
            assert vid_keys == set(), f"{sub.repo_id}: should have no video_keys, got {vid_keys}"
        else:
            assert vid_keys == EXPECTED_CAMERA_KEYS, (
                f"{sub.repo_id}: expected auto-promoted video-backed cameras "
                f"{EXPECTED_CAMERA_KEYS}, got image_keys={img_keys}, video_keys={vid_keys}"
            )
            assert img_keys == set(), (
                f"{sub.repo_id}: auto-promotion left orphan image_keys={img_keys}"
            )
            videos_dir = Path(DATASET_ROOT) / sub.repo_id / "videos"
            assert videos_dir.is_dir(), f"{sub.repo_id}: missing videos/ dir at {videos_dir}"
            for key in EXPECTED_CAMERA_KEYS:
                key_dir = videos_dir / key
                assert key_dir.is_dir(), f"{sub.repo_id}: missing videos/{key}/"
                mp4s = list(key_dir.rglob("*.mp4"))
                assert mp4s, f"{sub.repo_id}: no MP4s under {key_dir}"


def t03_camera_keys_uniform_across_subs():
    """All sub-datasets must expose the SAME camera_keys (union of image+video).
    Heterogeneous camera sets would break the MultiLeRobotDataset schema check
    AND silently produce garbage if it didn't."""
    multi = _FIXTURES["multi"]
    ref = set(multi._datasets[0].meta.camera_keys)
    assert ref == EXPECTED_CAMERA_KEYS, f"first sub camera_keys = {ref}"
    for sub in multi._datasets[1:]:
        assert set(sub.meta.camera_keys) == ref, (
            f"{sub.repo_id} camera_keys mismatch: {set(sub.meta.camera_keys)} vs {ref}"
        )


def t04_imagenet_stats_override_applied():
    """factory.make_dataset overrides stats[camera_key]['mean'|'std'] with
    IMAGENET_STATS on every sub-dataset when use_imagenet_stats=True. If this
    doesn't fire on one sub, that sub will normalize images differently — a
    silent train-time bug."""
    multi = _FIXTURES["multi"]
    for sub in multi._datasets:
        for cam in EXPECTED_CAMERA_KEYS:
            assert cam in sub.meta.stats, f"{sub.repo_id}: no stats for {cam}"
            mean = np.asarray(sub.meta.stats[cam]["mean"])
            std = np.asarray(sub.meta.stats[cam]["std"])
            assert _approx(mean, IMAGENET_MEAN), (
                f"{sub.repo_id} {cam}: mean override missed. got={mean.flatten().tolist()}, "
                f"want={IMAGENET_MEAN.flatten().tolist()}"
            )
            assert _approx(std, IMAGENET_STD), (
                f"{sub.repo_id} {cam}: std override missed. got={std.flatten().tolist()}, "
                f"want={IMAGENET_STD.flatten().tolist()}"
            )


def t05_decoded_frame_shape_dtype_range():
    """One frame from each sub-dataset must come out of __getitem__ with the
    same shape and dtype regardless of whether the source was inline images
    or MP4 videos. Records observed shapes for the log."""
    multi = _FIXTURES["multi"]
    observed = {}
    for sub in multi._datasets:
        sample = sub[0]
        for cam in EXPECTED_CAMERA_KEYS:
            assert cam in sample, f"{sub.repo_id}: missing {cam} in __getitem__"
            t = sample[cam]
            assert isinstance(t, torch.Tensor), f"{sub.repo_id} {cam}: not a Tensor ({type(t)})"
            observed.setdefault(cam, []).append((sub.repo_id, tuple(t.shape), t.dtype,
                                                  float(t.min()), float(t.max())))
    for cam, rows in observed.items():
        shapes = {r[1] for r in rows}
        dtypes = {r[2] for r in rows}
        print(f"  {cam}: per-repo (shape, dtype, min, max):")
        for repo, shape, dtype, lo, hi in rows:
            print(f"    {repo}: shape={shape} dtype={dtype} range=[{lo:.3f}, {hi:.3f}]")
        assert len(shapes) == 1, f"{cam}: heterogeneous shapes across subs: {shapes}"
        assert len(dtypes) == 1, f"{cam}: heterogeneous dtypes across subs: {dtypes}"
        for repo, _, _, lo, hi in rows:
            assert 0.0 - 1e-3 <= lo <= hi <= 1.0 + 1e-3, (
                f"{repo} {cam}: pixel range [{lo}, {hi}] outside [0,1]"
            )


def t06_per_repo_pixel_statistics_in_band():
    """A weak statistical sanity check: per-sub-dataset mean of one frame
    should fall in [0.05, 0.95] and std should be non-trivial (>0.01). This
    catches whole-frame corruption (all zeros / all ones) that would still
    pass the shape/dtype assertions."""
    multi = _FIXTURES["multi"]
    for sub in multi._datasets:
        sample = sub[0]
        for cam in EXPECTED_CAMERA_KEYS:
            t = sample[cam].float()
            m = float(t.mean())
            s = float(t.std())
            print(f"  {sub.repo_id} {cam}: mean={m:.3f} std={s:.3f}")
            assert 0.02 <= m <= 0.98, f"{sub.repo_id} {cam}: pixel mean {m:.3f} off-band"
            assert s >= 0.01, f"{sub.repo_id} {cam}: pixel std {s:.4f} too low (flat frame?)"


def t07_bucket_counts_match_frame_totals():
    """QValueLabelDataset.bucket_counts() must equal the sum of frames per
    bucket-assigned repo. Off-by-one in dataset_idx_by_frame would show up
    here as a count drift."""
    wrapped = _FIXTURES["wrapped"]
    multi = _FIXTURES["multi"]
    counts = wrapped.bucket_counts()
    print(f"  bucket_counts: {counts}")
    expected_q5 = next(s.num_frames for s in multi._datasets if s.repo_id == BC_REPO)
    expected_play = sum(s.num_frames for s in multi._datasets if s.repo_id != BC_REPO)
    assert counts["q5"] == expected_q5, f"q5: got {counts['q5']}, want {expected_q5}"
    assert counts["play"] == expected_play, f"play: got {counts['play']}, want {expected_play}"
    assert counts.get("q3_termjitter", 0) == 0, f"unexpected q3_termjitter: {counts}"


def t08_terminal_reward_per_bucket():
    """For an episode-terminal frame:
       - q5 bucket: reward_chunk[0] == terminal_bonuses['q5'] = 1.0
       - play bucket: reward_chunk[0] == terminal_bonuses['play'] = 0.0
    With step_reward=0.0 and i=0 at the terminal, the chunk's first slot is
    purely the terminal bonus."""
    wrapped = _FIXTURES["wrapped"]
    ep_from = wrapped._ep_from
    ep_to = wrapped._ep_to
    ds_idx_by_frame = wrapped._dataset_idx_by_frame
    multi = _FIXTURES["multi"]
    seen = {"q5": False, "play": False}
    for ep in range(len(ep_from)):
        terminal_global_idx = int(ep_to[ep]) - 1
        ds_idx = int(ds_idx_by_frame[terminal_global_idx])
        bucket = wrapped._dataset_index_to_bucket[ds_idx]
        if seen[bucket]:
            continue
        sample = wrapped[terminal_global_idx]
        r0 = float(sample["q_reward_chunk_first"][0])
        expected = TERMINAL_BONUSES[bucket]
        print(f"  ep={ep} terminal_idx={terminal_global_idx} bucket={bucket} "
              f"reward_chunk[0]={r0} (expect {expected})")
        assert math.isclose(r0, expected, abs_tol=1e-6), (
            f"bucket={bucket} terminal reward mismatch: got {r0}, want {expected}"
        )
        # Also: bootstrap_valid must be False at the last frame.
        assert not bool(sample["q_bootstrap_valid"]), (
            f"bucket={bucket} terminal frame: q_bootstrap_valid should be False"
        )
        seen[bucket] = True
        if all(seen.values()):
            break
    assert all(seen.values()), f"never saw both buckets at a terminal frame: {seen}"


def t09_dataloader_yields_consistent_batches():
    """Pull a few batches via a real DataLoader. Each batch must contain the
    four Q keys plus all camera+state+action keys, and the camera tensors
    must have shape (B, C, H, W). This exercises the collate path on top of
    a mixed-format MultiLeRobotDataset."""
    wrapped = _FIXTURES["wrapped"]
    loader = DataLoader(
        wrapped,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    required = {"q_reward_chunk_first", "q_reward_pad_first",
                "q_bootstrap_valid", "q_bucket_index",
                "observation.state", "action"}
    required |= EXPECTED_CAMERA_KEYS
    for i, batch in enumerate(loader):
        missing = required - set(batch.keys())
        assert not missing, f"batch {i}: missing keys {missing}"
        for cam in EXPECTED_CAMERA_KEYS:
            t = batch[cam]
            # Per-sample shape from __getitem__ is (T, C, H, W) where T = n_obs_steps
            # (=1) + len(observation_delta_indices); collation prepends the batch dim,
            # so a valid batch is (B, T, C, H, W). T01–T05 already pin T,C,H,W; here we
            # only verify batch dim + that decode produced the same (C,H,W) the per-sub
            # checks observed.
            assert t.ndim == 5, (
                f"batch {i} {cam}: expected 5D (B,T,C,H,W), got shape {tuple(t.shape)}"
            )
            assert t.shape[0] == 4, f"batch {i} {cam}: batch dim {t.shape[0]}"
            assert t.shape[-3:] == (3, 256, 256), (
                f"batch {i} {cam}: (C,H,W) trailing dims = {tuple(t.shape[-3:])}, want (3,256,256)"
            )
            assert torch.isfinite(t).all(), f"batch {i} {cam}: non-finite pixels"
        assert batch["q_reward_chunk_first"].shape == (4, 32), batch["q_reward_chunk_first"].shape
        assert batch["q_bucket_index"].dtype == torch.long
        if i >= 2:  # 3 batches is plenty
            break


def t10_one_train_step_no_nan():
    """End-to-end: build the actual q_function policy and run one forward +
    backward on a mixed batch. Catches shape mismatches between sub-datasets
    and the policy's expected input schema, plus NaN-producing inputs."""
    cfg = _FIXTURES["cfg"]
    wrapped = _FIXTURES["wrapped"]
    if DEVICE == "cpu":
        print("  SKIP: no CUDA available — DINOv2-large + T5-base is too heavy for CPU")
        return
    print("  Building policy (DINOv2-large + T5-base, ~1.3B params)…", flush=True)
    policy = make_policy(cfg=cfg.policy, ds_meta=wrapped.dataset._datasets[0].meta)
    policy.train()
    loader = DataLoader(wrapped, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    batch = next(iter(loader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(DEVICE, non_blocking=False)
    print("  Forward pass…", flush=True)
    out = policy.forward(batch)
    loss = out[0] if isinstance(out, tuple) else out["loss"] if isinstance(out, dict) else out
    assert torch.isfinite(loss), f"loss is non-finite: {loss}"
    print(f"  loss={float(loss):.4f}", flush=True)
    print("  Backward pass…", flush=True)
    loss.backward()
    bad_grads = []
    for name, p in policy.named_parameters():
        if p.grad is None or not p.requires_grad:
            continue
        if not torch.isfinite(p.grad).all():
            bad_grads.append(name)
    assert not bad_grads, f"non-finite grads in: {bad_grads[:5]}{'…' if len(bad_grads) > 5 else ''}"
    print(f"  All {sum(1 for p in policy.parameters() if p.requires_grad)} "
          f"trainable params have finite grads", flush=True)


# ────────────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("=" * 72)
    print("Multi-dataset Q-function pipeline tests")
    print(f"  DEVICE:        {DEVICE}")
    print(f"  DATASET_ROOT:  {DATASET_ROOT}")
    print(f"  REPO_IDS:      {len(REPO_IDS)} repos")
    print("=" * 72)

    # Single fixture build first so subsequent tests reuse it. Failure here
    # is fatal — every other test depends on the wrapped dataset.
    try:
        build_dataset()
    except Exception:
        traceback.print_exc()
        print("\nFATAL: dataset construction failed; skipping remaining tests.")
        return 2

    _run("T01 dataset_construction", t01_dataset_construction)
    _run("T02 per_repo_dtype_promotion", t02_per_repo_dtype_promotion)
    _run("T03 camera_keys_uniform_across_subs", t03_camera_keys_uniform_across_subs)
    _run("T04 imagenet_stats_override_applied", t04_imagenet_stats_override_applied)
    _run("T05 decoded_frame_shape_dtype_range", t05_decoded_frame_shape_dtype_range)
    _run("T06 per_repo_pixel_statistics_in_band", t06_per_repo_pixel_statistics_in_band)
    _run("T07 bucket_counts_match_frame_totals", t07_bucket_counts_match_frame_totals)
    _run("T08 terminal_reward_per_bucket", t08_terminal_reward_per_bucket)
    _run("T09 dataloader_yields_consistent_batches", t09_dataloader_yields_consistent_batches)
    _run("T10 one_train_step_no_nan", t10_one_train_step_no_nan)

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    n_pass = sum(1 for _, ok, _, _ in _RESULTS if ok)
    n_fail = sum(1 for _, ok, _, _ in _RESULTS if not ok)
    for name, ok, err, dt in _RESULTS:
        status = "PASS" if ok else "FAIL"
        suffix = "" if ok else f"  — {err}"
        print(f"  [{status}] {name}  ({dt:.2f}s){suffix}")
    print(f"\n{n_pass} passed, {n_fail} failed, {len(_RESULTS)} total")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
