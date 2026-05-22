#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Convert a released FastWAM RoboTwin checkpoint to a LeRobot PreTrainedPolicy dir.

The released checkpoint (``robotwin_uncond_3cam_384.pt``) is a ``torch.save`` dict::

    {"mot": <MoT.state_dict()>, "proprio_encoder": <Linear.state_dict()>, "step": int, ...}

The matching ``robotwin_uncond_3cam_384_dataset_stats.json`` holds FastWAM's z-score
(mean/std) normalization stats. This script:

  1. Builds a ``FastWAMPolicy`` with the RoboTwin config preset (single 384x320
     concatenated camera, 14-D state/action, z-score normalization).
  2. Loads the fine-tuned MoT + proprio-encoder weights into it (the VAE and T5
     text encoder come from the Wan2.2 pretrained weights via ``__init__``).
  3. Builds the LeRobot pre/post processors with the released stats.
  4. Writes a standard LeRobot checkpoint directory (config.json,
     model.safetensors, processor configs) consumable by ``lerobot-eval``.

Usage:
    python scripts/convert_fastwam_robotwin_checkpoint.py \
        --pt    /path/robotwin_uncond_3cam_384.pt \
        --stats /path/robotwin_uncond_3cam_384_dataset_stats.json \
        --out   /path/output_checkpoint_dir
"""
import argparse
import json
from pathlib import Path

import torch

from lerobot.configs.types import NormalizationMode
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
from lerobot.policies.fastwam.processor_fastwam import make_fastwam_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE


def build_robotwin_config(device: str) -> FastWAMConfig:
    """FastWAM config preset for the released RoboTwin 3-cam checkpoint.

    Values match FastWAM's robotwin configs:
      - single pre-concatenated [3,384,320] frame  -> num_cameras=1
      - 14-D bimanual qpos action + 14-D joint state
      - chunk_size=32 (action_horizon), n_action_steps=24 (replan_steps),
        num_inference_steps=10  (configs/sim_robotwin.yaml + deploy_policy.yml)
      - "uncond" video DiT -> video_dit_action_conditioned=False
      - z-score (MEAN_STD) normalization for state + action
    """
    return FastWAMConfig(
        model_variant="fastwam",
        n_obs_steps=1,
        chunk_size=32,
        n_action_steps=24,
        image_size=(384, 320),
        num_cameras=1,
        action_dim=14,
        state_dim=14,
        max_state_dim=14,
        max_action_dim=14,
        num_inference_steps=10,
        video_dit_action_conditioned=False,
        mot_checkpoint_mixed_attn=False,
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        device=device,
        push_to_hub=False,
    )


def build_dataset_stats(stats_json_path: str) -> dict[str, dict[str, torch.Tensor]]:
    """Convert FastWAM's dataset_stats.json to LeRobot's {key: {mean, std}} format.

    Uses the global (not stepwise) mean/std — FastWAM's robotwin config sets
    ``use_stepwise_action_norm: False``.
    """
    with open(stats_json_path, encoding="utf-8") as f:
        raw = json.load(f)

    def _mean_std(field: str) -> dict[str, torch.Tensor]:
        d = raw[field]["default"]
        return {
            "mean": torch.tensor(d["global_mean"], dtype=torch.float32),
            "std": torch.tensor(d["global_std"], dtype=torch.float32),
        }

    return {OBS_STATE: _mean_std("state"), ACTION: _mean_std("action")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="Released FastWAM RoboTwin .pt checkpoint")
    ap.add_argument("--stats", required=True, help="Released *_dataset_stats.json")
    ap.add_argument("--out", required=True, help="Output LeRobot checkpoint directory")
    ap.add_argument("--device", default="cuda", help="config.device baked into the checkpoint")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"### Building FastWAM policy (RoboTwin preset) — downloads Wan2.2 VAE + T5 ...")
    config = build_robotwin_config(a.device)
    policy = FastWAMPolicy(config)
    policy.eval()
    config.validate_features()

    print(f"### Loading released checkpoint: {a.pt}")
    ckpt = torch.load(a.pt, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "mot" not in ckpt:
        raise ValueError(
            f"Unexpected checkpoint structure; top-level keys = "
            f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )
    print(f"  checkpoint top-level keys: {list(ckpt.keys())}")

    # MoT (video + action experts). Load non-strict + report any key mismatch.
    mot_missing, mot_unexpected = policy.model.mot.load_state_dict(ckpt["mot"], strict=False)
    print(f"  mot: loaded; missing={len(mot_missing)} unexpected={len(mot_unexpected)}")
    for k in list(mot_missing)[:10]:
        print(f"    [missing]    {k}")
    for k in list(mot_unexpected)[:10]:
        print(f"    [unexpected] {k}")
    if mot_missing or mot_unexpected:
        print("  WARNING: MoT key mismatch — verify the lerobot port matches the released arch.")

    # Proprio encoder (nn.Linear state_dim -> text_dim).
    if "proprio_encoder" in ckpt and policy.model.proprio_encoder is not None:
        policy.model.proprio_encoder.load_state_dict(ckpt["proprio_encoder"], strict=True)
        print("  proprio_encoder: loaded (strict)")
    else:
        print(f"  WARNING: proprio_encoder not loaded "
              f"(in_ckpt={'proprio_encoder' in ckpt}, "
              f"model_has={policy.model.proprio_encoder is not None})")

    print(f"### Building pre/post processors with z-score stats from {a.stats}")
    dataset_stats = build_dataset_stats(a.stats)
    for k, v in dataset_stats.items():
        print(f"  {k}: mean[:3]={v['mean'][:3].tolist()} std[:3]={v['std'][:3].tolist()}")
    preprocessor, postprocessor = make_fastwam_pre_post_processors(config, dataset_stats)

    print(f"### Saving LeRobot checkpoint to {out}")
    policy.save_pretrained(out)
    preprocessor.save_pretrained(out)
    postprocessor.save_pretrained(out)

    print("### Contents:")
    for p in sorted(out.iterdir()):
        size = p.stat().st_size / 1e6 if p.is_file() else 0
        print(f"  {p.name}  ({size:.1f} MB)" if p.is_file() else f"  {p.name}/")
    print("### convert_fastwam_robotwin_checkpoint.py complete")


if __name__ == "__main__":
    main()
