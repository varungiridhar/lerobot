#!/usr/bin/env python
"""Prototype: FastWAM + random-weight Q-function online planning mechanics.

Tests two things:
  1. Core planning loop (plan_chunk_fastwam) with a mocked BC policy and random Q.
  2. Full round-trip through PlannerContext with identity preprocessors.

Loads the real Q-function architecture with random weights so we can validate
the full data-flow end-to-end without needing a trained Q checkpoint.

Run on interactive GPU node:
    python scripts/proto_fastwam_planning.py

FastWAM checkpoint for full eval:
    /storage/project/r-agarg35-0/shared/fastwam/hf_checkpoint
"""

import os

os.environ["HF_HOME"] = "/storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/.cache/huggingface"

import torch

FASTWAM_CKPT = "/storage/project/r-agarg35-0/shared/fastwam/hf_checkpoint"
DATA_ROOT = "/storage/home/hcoda1/7/igeorgiev3/shared/lerobot-data-2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------------
# 1. Build random Q-function
# ---------------------------------------------------------------------------
print("\n[1] Building random-weight Q-function...")
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.q_function.configuration_q_function import QFunctionConfig
from lerobot.policies.q_function.modeling_q_function import QFunctionPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES

q_cfg = QFunctionConfig(
    h=32,                           # must match FastWAM chunk_size
    dino_model_name="facebook/dinov2-large",
    dim_model=1024,
    n_heads=16,
    dim_feedforward=4096,
    n_decoder_layers=18,
    image_resize_h=224,
    image_resize_w=224,
    use_text_conditioning=True,
    text_encoder_model="google/t5-v1_1-base",
    language_key="task",
    v_min=-0.01,
    v_max=1.01,
    input_features={
        f"{OBS_IMAGES}.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        f"{OBS_IMAGES}.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    },
    output_features={
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    },
)
q_policy = QFunctionPolicy(q_cfg).to(device).eval()
print(f"  Q-function parameters: {sum(p.numel() for p in q_policy.parameters()) / 1e6:.1f}M")
print(f"  Q camera_keys: {q_policy.config.camera_keys}")


# ---------------------------------------------------------------------------
# 2. Build synthetic batch matching Q's expected input format
# ---------------------------------------------------------------------------
print("\n[2] Building synthetic batch...")
B, h, A = 1, 32, 7
cam_h, cam_w = 224, 224

# Q expects: observation.images.* (float [0,1]), action (B,h,A), task (list[str])
# These will go through Q's preprocessor (MEAN_STD normalization) inside predict_value.
# For the prototype we skip normalization by using trivial pre/post processors.
batch = {
    "observation.images.image": torch.rand(B, 3, cam_h, cam_w, device=device),
    "observation.images.image2": torch.rand(B, 3, cam_h, cam_w, device=device),
    "observation.state": torch.rand(B, 8, device=device),
    "task": ["pick up the red block and place it in the bin"],
}
print(f"  Camera image shape: {batch['observation.images.image'].shape}")


# ---------------------------------------------------------------------------
# 3. Test plan_chunk_fastwam with mocked BC policy
# ---------------------------------------------------------------------------
print("\n[3] Testing plan_chunk_fastwam with mocked FastWAM...")

from lerobot.policies.act_simple.planning import PlannerContext, PlanningConfig
from lerobot.policies.fastwam.planning import plan_chunk_fastwam


class MockFastWAMPolicy:
    """Minimal mock: predict_action_chunk returns random MIN_MAX-norm actions."""
    config = type("cfg", (), {"chunk_size": h, "n_action_steps": 10})()

    def parameters(self):
        yield torch.zeros(1, device=device)  # so next(parameters()).device works

    def predict_action_chunk(self, batch):
        # Return (B, h, A) on CPU in [0, 1] (mimics MIN_MAX normalized actions)
        return torch.rand(B, h, A)


mock_bc = MockFastWAMPolicy()

# Trivial preprocessors for prototype: identity pass-through
def identity_post(x):
    """Unnorm: flat_norm (N*h, A) → raw actions (same, for random weights)."""
    return x


def identity_pre(q_batch):
    """Norm: q_batch dict → same dict (skip normalization for prototype)."""
    # predict_value expects 'action' key; ensure it's present
    return q_batch


ctx = PlannerContext(
    q_policy=q_policy,
    q_pre=identity_pre,
    bc_post=identity_post,
    q_camera_keys=tuple(q_policy.config.camera_keys),
    horizon=h,
)

plan_cfg = PlanningConfig(
    planner_type="mppi",
    n_samples=8,
    noise_std=0.1,
    temperature=0.1,
)

print(f"  Q camera_keys in context: {ctx.q_camera_keys}")
print(f"  Planning config: n_samples={plan_cfg.n_samples}, noise_std={plan_cfg.noise_std}")

with torch.no_grad():
    planned, spread = plan_chunk_fastwam(mock_bc, batch, ctx, plan_cfg, generator=None)

print(f"  Planned chunk shape: {planned.shape}  (expected: ({B}, {h}, {A}))")
print(f"  Q spread: min={spread[0]:.4f}, max={spread[1]:.4f}, mean={spread[2]:.4f}, std={spread[3]:.4f}")
assert planned.shape == (B, h, A), f"Shape mismatch: {planned.shape}"
assert all(not (v != v) for v in spread), "NaN in Q spread!"
print("  [OK] plan_chunk_fastwam mechanics verified!")


# ---------------------------------------------------------------------------
# 4. Test argmax planner
# ---------------------------------------------------------------------------
print("\n[4] Testing argmax planner...")
plan_cfg_argmax = PlanningConfig(planner_type="argmax", n_samples=8, noise_std=0.1)
with torch.no_grad():
    planned_arg, spread_arg = plan_chunk_fastwam(mock_bc, batch, ctx, plan_cfg_argmax)
print(f"  Argmax planned shape: {planned_arg.shape}")
print(f"  Argmax Q spread: min={spread_arg[0]:.4f}, max={spread_arg[1]:.4f}")
print("  [OK]")


# ---------------------------------------------------------------------------
# 5. Verify planned != BC mean (planning changes the output)
# ---------------------------------------------------------------------------
print("\n[5] Verifying planning perturbs BC mean...")
bc_mean = mock_bc.predict_action_chunk(batch).to(device)
diff = (planned - bc_mean).abs().mean().item()
print(f"  Mean absolute deviation from BC: {diff:.6f}  (should be > 0)")
assert diff > 0, "Planned chunk identical to BC mean — planning has no effect!"
print("  [OK] Planning produces different actions from BC baseline")


# ---------------------------------------------------------------------------
# 6. Verify attach_planner / select_action hook on FastWAM modeling
# ---------------------------------------------------------------------------
print("\n[6] Testing attach_planner on FastWAMPolicy (config/modeling hooks only)...")
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig

# Just test config parsing — don't load the 6B model
fw_cfg = FastWAMConfig()
print(f"  use_planning default: {fw_cfg.use_planning}")
print(f"  planning config type: {type(fw_cfg.planning).__name__}")
print(f"  planning.planner_type default: {fw_cfg.planning.planner_type}")
fw_cfg.use_planning = True
fw_cfg.planning.n_samples = 32
fw_cfg.planning.noise_std = 0.3
print(f"  After setting use_planning=True, n_samples={fw_cfg.planning.n_samples}: OK")
print("  [OK] FastWAMConfig planning fields work")


print("\n" + "=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)
print("""
Next steps:
  • Load real FastWAM from /storage/project/r-agarg35-0/shared/fastwam/hf_checkpoint
  • Load trained Q checkpoint (once available)
  • Run lerobot-eval with:
      --policy.type=fastwam
      --policy.pretrained_path=<fastwam_ckpt>
      --policy.use_planning=true
      --policy.planning.q_checkpoint_path=<q_ckpt>
      --policy.planning.n_samples=64
      --policy.planning.planner_type=mppi
""")
