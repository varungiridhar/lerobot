#!/usr/bin/env python
"""Is the Q success-aligned, or just a demo-memorizer?

Controlled test at the SAME held-out states: does the Q rate FastWAM's OWN
(successful, ~91%) action chunk far below the dataset demo chunk? If
Q(demo) >> Q(FastWAM-action) at the same state, the Q credits exact demos but
penalizes the good policy it's meant to guide — so MPPI following the Q steers
away from FastWAM toward chunks the Q likes but that don't succeed (explains why
the fixed planner scored <= BC).

For each held-out frame:
  q_demo  = Q(dataset demo chunk)                       (the probe path, ~0.45 expected)
  q_bc    = Q(FastWAM BC chunk -> bc_post -> q_pre)     (deployment center, raw)
  q_bccl  = same but hold_gripper + clamp to [-1,1]     (what the fixed planner feeds)
All actions are scored through the SAME q_pre, at the SAME state, so any gap is
purely the action content (demo vs FastWAM), not the state.

Usage (GPU node):
  python scripts/debug_q_success_align.py \
    --q-ckpt <Q .../checkpoints/last/pretrained_model> \
    --fastwam-ckpt /storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint \
    --dataset-config <BC+play train_config.json> \
    --n-frames 40
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("q_success_align")

GRIP_DIM = -1


def _load_q(q_ckpt, device):
    from lerobot.policies.q_function.modeling_q_function import QFunctionPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch
    from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME

    q = QFunctionPolicy.from_pretrained(str(q_ckpt)).to(device).eval()
    q_pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(q_ckpt),
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    return q, q_pre


def _load_fastwam(fastwam_ckpt, device):
    from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    bc = FastWAMPolicy.from_pretrained(str(fastwam_ckpt)).to(device).eval()
    bc_pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(fastwam_ckpt),
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    # bc_post: FastWAM-norm -> raw, then gripper [0,1]->[-1,1] (the planner's mapping).
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

    bc_post_raw = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(fastwam_ckpt),
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    def bc_post(x):
        r = bc_post_raw(x).clone()
        r[..., GRIP_DIM] = r[..., GRIP_DIM] * 2.0 - 1.0
        return r

    return bc, bc_pre, bc_post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q-ckpt", required=True)
    ap.add_argument("--fastwam-ckpt", default="/storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--n-frames", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import draccus

    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.q_function.configuration_q_function import QFunctionConfig  # noqa: F401
    from lerobot.policies.q_function.q_vis import _compute_chunk_q_values
    from lerobot.utils.constants import ACTION

    device = torch.device(args.device)
    q, q_pre = _load_q(args.q_ckpt, device)
    bc, bc_pre, bc_post = _load_fastwam(args.fastwam_ckpt, device)
    h = int(q.config.h)
    cam_keys = tuple(q.config.camera_keys)
    log.info("Q h=%d cams=%s; FastWAM chunk=%s", h, cam_keys, getattr(bc.config, "chunk_size", "?"))

    cfg = draccus.parse(TrainPipelineConfig, config_path=str(args.dataset_config), args=[])
    dataset = make_dataset(cfg)
    test_ids = list(getattr(dataset, "test_episode_ids", []) or [])
    # Sample held-out frames spread across the BC (q5) test episodes.
    frames = []
    for ep in test_ids:
        f0 = int(dataset._ep_from[ep].item())
        ds_idx = int(dataset._dataset_idx_by_frame[f0].item())
        if dataset._dataset_index_to_bucket[ds_idx] != "q5":
            continue
        f1 = int(dataset._ep_to[ep].item())
        frames += [f0 + (f1 - f0) // 3, f0 + (f1 - f0) // 2]  # a couple mid-episode states
        if len(frames) >= args.n_frames:
            break
    frames = frames[: args.n_frames]
    log.info("Scoring %d held-out q5 frames", len(frames))

    rows = {"q_demo": [], "q_bc": [], "q_bccl": [], "adist": []}
    for fi in frames:
        item = dataset[fi]
        L, A = item[ACTION].shape
        # FastWAM obs batch: current-frame images + state + task, normalized by bc_pre.
        obs = {}
        for ck in cam_keys:
            t = item[ck]
            obs[ck] = (t[0] if t.dim() == 4 else t).unsqueeze(0).to(device)
        if "observation.state" in item:
            s = item["observation.state"]
            obs["observation.state"] = (s[0] if s.dim() == 2 else s).unsqueeze(0).to(device)
        ts = item.get("task", "")
        obs["task"] = [ts[0] if isinstance(ts, list) else str(ts)]
        with torch.no_grad():
            bc_norm = bc.predict_action_chunk(bc_pre(obs)).to(device)  # (1, h, A) FastWAM-norm
            bc_raw = bc_post(bc_norm.reshape(-1, A)).reshape(1, -1, A)  # (1, h, A) Q-raw
            bc_cl = bc_norm.clone()
            bc_cl[..., GRIP_DIM] = 0.0  # NOT used directly; clamp norm then post
            bc_norm_cl = bc_norm.clamp(-1.0, 1.0)
            bc_raw_cl = bc_post(bc_norm_cl.reshape(-1, A)).reshape(1, -1, A)

        # Pad FastWAM chunk (length h) to L=2h so the helper's shape check passes;
        # predict_value only uses the first h, so the tail is don't-care.
        def _pad(chunk):  # (1, h, A) -> (1, 1, L, A)
            reps = (L + chunk.shape[1] - 1) // chunk.shape[1]
            return chunk.repeat(1, reps, 1)[:, :L, :].unsqueeze(1)

        candidates = torch.cat([_pad(bc_raw), _pad(bc_raw_cl)], dim=1).cpu()  # (1, 2, L, A)
        qv = _compute_chunk_q_values(q, q_pre, [item], 0, 0.0, device, candidates=candidates)
        rows["q_demo"].append(float(qv[0, 0]))
        rows["q_bc"].append(float(qv[0, 1]))
        rows["q_bccl"].append(float(qv[0, 2]))
        rows["adist"].append(float((bc_raw_cl[0, :h].cpu() - item[ACTION][:h]).abs().mean()))

    qd = np.array(rows["q_demo"]); qb = np.array(rows["q_bc"]); qc = np.array(rows["q_bccl"])
    print("\n==== Q success-alignment diagnosis ====")
    print(f"frames: {len(qd)}")
    print(f"Q(demo)        mean={qd.mean():.4f}  median={np.median(qd):.4f}")
    print(f"Q(FastWAM raw) mean={qb.mean():.4f}  median={np.median(qb):.4f}")
    print(f"Q(FastWAM clmp)mean={qc.mean():.4f}  median={np.median(qc):.4f}")
    print(f"frac Q(demo) > Q(FastWAM clmp): {float((qd > qc).mean()):.3f}")
    print(f"mean |demo - FastWAM| (raw): {np.mean(rows['adist']):.4f}")
    print("INTERPRETATION: if Q(demo) >> Q(FastWAM) at the same state, the Q is a")
    print("demo-memorizer (penalizes the good policy) -> not success-aligned ->")
    print("planning that follows it degrades. If Q(demo) ~= Q(FastWAM), the Q does")
    print("credit FastWAM's actions and the flat result is a different issue.")


if __name__ == "__main__":
    main()
