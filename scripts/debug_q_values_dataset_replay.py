"""Replay a dataset episode through the Q function's planning-time code path.

This diagnostic checks whether Q values on dataset observations (known ground truth)
give the expected 0.2→1.0 arc, or the same compressed values (0.1→0.2) seen at eval time.

Usage:
    python scripts/debug_q_values_dataset_replay.py \
        --q_ckpt /path/to/q_checkpoint \
        --dataset_root /storage/project/r-agarg35-0/vgiridhar6/robotwin/dataset/robotwin2.0_multicam \
        --episode_index 0 \
        --chunk_stride 24
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def load_video_frames(video_path: Path, global_frame_indices: list[int]) -> dict[int, np.ndarray]:
    """Decode specific frames from an mp4 by global frame index.

    Returns dict mapping global_frame_index → (H, W, 3) uint8 array.
    """
    import imageio.v3 as iio
    frames = {}
    # imageio reads all frames; we index by position
    reader = iio.imopen(str(video_path), "r", plugin="pyav")
    for i, frame in enumerate(reader.iter()):
        if i in global_frame_indices:
            frames[i] = frame  # (H, W, 3) uint8
        if i > max(global_frame_indices):
            break
    reader.close()
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q_ckpt", required=True, help="Path to Q function checkpoint")
    parser.add_argument("--dataset_root", required=True, help="Path to robotwin2.0_multicam dataset")
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--chunk_stride", type=int, default=24, help="Evaluate Q every N frames (= n_action_steps)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset_root = Path(args.dataset_root)
    q_ckpt = Path(args.q_ckpt)

    # ── Load Q function ──────────────────────────────────────────────────────
    print("Loading Q function...")
    from lerobot.policies.q_function.modeling_q_function import QFunctionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch

    q_policy = QFunctionPolicy.from_pretrained(str(q_ckpt)).to(device).eval()
    q_pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(q_ckpt),
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    h = int(q_policy.config.h)
    print(f"  Q horizon h={h}, camera_keys={q_policy.config.camera_keys}")

    # ── Load episode from parquet ────────────────────────────────────────────
    print(f"\nLoading episode {args.episode_index} from dataset...")
    parquet_path = dataset_root / "data" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(parquet_path)
    ep_df = df[df["episode_index"] == args.episode_index].reset_index(drop=True)
    print(f"  Episode length: {len(ep_df)} frames")

    # Load task description
    tasks_df = pd.read_parquet(dataset_root / "meta" / "tasks.parquet").reset_index()
    tasks_df.columns = ["task", "task_index"]
    task_idx = int(ep_df["task_index"].iloc[0])
    task_str = tasks_df[tasks_df["task_index"] == task_idx]["task"].iloc[0]
    print(f"  Task: {task_str[:80]}")

    # Global frame indices for this episode
    global_indices = ep_df["index"].tolist()

    # ── Decode video frames for all three cameras ────────────────────────────
    print("\nDecoding video frames...")
    cam_keys = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
    cam_frames: dict[str, dict[int, np.ndarray]] = {}
    for cam_key in cam_keys:
        cam_name = cam_key.split(".")[-1]
        vid_path = dataset_root / "videos" / cam_key / "chunk-000" / "file-000.mp4"
        print(f"  {cam_name}: {vid_path}")
        cam_frames[cam_key] = load_video_frames(vid_path, global_indices)

    # Helper to get frame as float32 [0,1] tensor (1, 3, H, W)
    def get_frame_tensor(cam_key: str, ep_frame_idx: int) -> torch.Tensor:
        global_idx = global_indices[ep_frame_idx]
        img = cam_frames[cam_key][global_idx]  # (H, W, 3) uint8
        t = torch.from_numpy(img).float() / 255.0  # (H, W, 3) float32 [0,1]
        t = t.permute(2, 0, 1).unsqueeze(0)         # (1, 3, H, W)
        return t.to(device)

    # ── Evaluate Q at each chunk boundary ────────────────────────────────────
    print(f"\nEvaluating Q every {args.chunk_stride} frames...")
    print(f"{'Frame':>6}  {'t/T':>6}  {'Q_value':>8}  {'path'}")
    print("-" * 45)

    actions_np = np.stack(ep_df["action"].tolist())  # (T, action_dim)
    T = len(ep_df)
    q_values_out = []

    for t in range(0, T - h, args.chunk_stride):
        # Build batch exactly as planning does:
        #   - raw camera images (float32 [0,1])
        #   - action chunk [t : t+h] in raw env space
        #   - task description
        batch: dict = {}
        for cam_key in cam_keys:
            batch[cam_key] = get_frame_tensor(cam_key, t)

        action_chunk = torch.from_numpy(actions_np[t : t + h]).float().unsqueeze(0).to(device)  # (1, h, A)
        batch["action"] = action_chunk
        batch["task"] = [task_str]

        # ── PATH A: planning-time path ─────────────────────────────────────
        # Same as plan_chunk_fastwam: q_pre → encode_obs_context → predict_value
        preprocessed = q_pre(batch)
        q_val = float(q_policy.predict_value(preprocessed).item())

        q_values_out.append((t, q_val))
        frac = t / max(T - 1, 1)
        print(f"{t:6d}  {frac:6.2f}  {q_val:8.4f}")

    print("-" * 45)
    if q_values_out:
        qs = [q for _, q in q_values_out]
        print(f"  Q range: {min(qs):.4f} → {max(qs):.4f}  (mean={sum(qs)/len(qs):.4f})")

    # ── PATH B: also evaluate using QValueLabelDataset (training path) ─────
    print("\nNow evaluating via QValueLabelDataset (training-time path) for comparison...")
    try:
        from lerobot.policies.q_function.dataset import QValueLabelDataset
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        ds_meta = LeRobotDatasetMetadata.from_json(
            str(dataset_root / "meta" / "info.json"),
            dataset_root=str(dataset_root),
        )
        from lerobot.policies.q_function.modeling_q_function import resolve_delta_timestamps
        delta_ts = resolve_delta_timestamps(q_policy.config, ds_meta)

        q_ds = QValueLabelDataset(
            dataset_root=str(dataset_root),
            delta_timestamps=delta_ts,
            h=h,
            camera_keys=list(q_policy.config.camera_keys),
            task_keys=None,
            episodes=[args.episode_index],
        )
        print(f"  Dataset size for ep {args.episode_index}: {len(q_ds)} samples")

        # Evaluate every chunk_stride samples
        print(f"{'Sample':>6}  {'Q_value':>8}")
        print("-" * 20)
        q_ds_vals = []
        for i in range(0, len(q_ds), max(1, args.chunk_stride)):
            sample = q_ds[i]
            b = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else [v]
                 for k, v in sample.items()}
            preprocessed = q_pre(b)
            q_val = float(q_policy.predict_value(preprocessed).item())
            q_ds_vals.append(q_val)
            print(f"{i:6d}  {q_val:8.4f}")

        if q_ds_vals:
            print(f"  Q range: {min(q_ds_vals):.4f} → {max(q_ds_vals):.4f}")
    except Exception as e:
        print(f"  Skipped (error): {e}")


if __name__ == "__main__":
    main()
