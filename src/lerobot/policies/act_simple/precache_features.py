"""Pre-cache vision features from a trained act_simple policy backbone.

What it does
============
For every frame of a target ``LeRobotDataset``, runs the trained policy's
ResNet18 backbone over the camera images (with the policy's training-time
image normalisation), and writes the resulting feature maps to a memmap on
disk. This lets a downstream policy (e.g. the Q-function via
``vision_backbone="resnet18_cached"``) consume frozen, policy-aligned vision
features without re-running the backbone per training step.

Output layout
-------------
The cache lives next to the policy that produced it (one cache per (policy
checkpoint, dataset) pair). Default location::

    <policy_checkpoint>/encoded_backbone/<repo_id>/
        meta.json                       # provenance + feature shape
        <camera_key>.mmap               # float16, shape (N_frames, 512, Hf, Wf)

This makes the cache "checkpoint-aware": re-training the BC policy → fresh
checkpoint dir → fresh cache, no risk of silently consuming a stale cache.
Override the location with ``--cache_root <dir>``; the per-repo subdirectory
is always appended.

Idempotency
-----------
If ``<dataset_root>/encoded_backbone/meta.json`` already exists, the script
exits with a friendly message unless ``--overwrite`` is set.

Usage
-----
::

    python -u src/lerobot/policies/act_simple/precache_features.py \\
        --policy_checkpoint outputs/train/mimicgen_coffee_d0_act_simple/checkpoints/100000/pretrained_model \\
        --dataset_root /storage/.../v1_lerobot/mg_coffee_q5 \\
        --repo_id mg_coffee_q5

Notes
-----
* The script does NOT mutate the dataset's parquet/PNGs — it only writes the
  ``encoded_backbone/`` subdir.
* Image preprocessing matches the policy's training: PIL → uint8 → /255 →
  per-camera ``(img - mean) / std`` using stats loaded directly from the
  policy's preprocessor safetensors. (No `AutoImageProcessor` involvement.)
* ResNet18 layer4 on 84x84 inputs yields (512, 3, 3); the script asserts this.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from safetensors.torch import load_file
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [precache_features] %(message)s")
log = logging.getLogger("precache_features")


# ──────────────────────────────────────────────────────────────────────────────
# Backbone + normalization stats from the policy checkpoint
# ──────────────────────────────────────────────────────────────────────────────

def load_policy_backbone(checkpoint_dir: Path, device: torch.device) -> nn.Module:
    """Rebuild the act_simple ResNet18 backbone and load weights from safetensors.

    Mirrors ACTSimplePolicy's backbone construction (same torchvision model,
    same FrozenBatchNorm2d, same IntermediateLayerGetter at layer4) and copies
    the trained weights from the checkpoint.
    """
    cfg_path = checkpoint_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing {cfg_path}; expected an act_simple pretrained_model dir")
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("type") != "act_simple":
        raise ValueError(f"expected type=act_simple in {cfg_path}, got {cfg.get('type')!r}")
    if not cfg.get("vision_backbone", "").startswith("resnet"):
        raise ValueError(f"expected resnet backbone, got {cfg.get('vision_backbone')!r}")

    arch = cfg["vision_backbone"]
    dilate_last = bool(cfg.get("replace_final_stride_with_dilation", False))
    backbone_model = getattr(torchvision.models, arch)(
        replace_stride_with_dilation=[False, False, dilate_last],
        weights=None,
        norm_layer=FrozenBatchNorm2d,
    )
    backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

    sd_full = load_file(str(checkpoint_dir / "model.safetensors"))
    backbone_sd = {k[len("model.backbone."):]: v for k, v in sd_full.items() if k.startswith("model.backbone.")}
    if not backbone_sd:
        raise RuntimeError("no model.backbone.* keys found in policy state dict")
    missing, unexpected = backbone.load_state_dict(backbone_sd, strict=False)
    if missing or unexpected:
        log.warning("backbone load: missing=%d unexpected=%d (head 5 each: %s | %s)",
                    len(missing), len(unexpected), missing[:5], unexpected[:5])

    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    return backbone.to(device)


def load_camera_norm_stats(checkpoint_dir: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Return {lerobot_key: (mean, std)} as (3,1,1) tensors from the policy preprocessor.

    The policy's normalizer keys match the LeRobot input feature keys
    (``observation.images.image``, ``observation.images.image2``, …), so we
    return whatever VISUAL keys are present.
    """
    norm_path = checkpoint_dir / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    sd = load_file(str(norm_path))
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for k in sd.keys():
        if not k.startswith("observation.images."):
            continue
        if not (k.endswith(".mean") or k.endswith(".std")):
            continue
        cam_key = k.rsplit(".", 1)[0]
        if cam_key in out:
            continue
        m_key, s_key = f"{cam_key}.mean", f"{cam_key}.std"
        out[cam_key] = (sd[m_key].float(), sd[s_key].float())
    if not out:
        raise KeyError(f"no observation.images.*.mean/std keys in {norm_path}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Per-frame encoding
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _encode_chunk(
    backbone: nn.Module,
    images_uint8: torch.Tensor,         # (B, 3, H, W) uint8 OR (B, H, W, 3) uint8
    mean_chw: torch.Tensor,             # (3, 1, 1)
    std_chw: torch.Tensor,              # (3, 1, 1)
    device: torch.device,
) -> torch.Tensor:
    """Run a chunk of frames through the backbone. Return (B, C_out, Hf, Wf) float16 on CPU."""
    if images_uint8.dim() != 4:
        raise RuntimeError(f"expected 4D tensor, got shape {tuple(images_uint8.shape)}")
    # LeRobotDataset returns images as either (B, 3, H, W) float in [0,1]
    # or (B, 3, H, W) uint8, depending on transforms. We accept both and
    # also the (B, H, W, 3) uint8 layout for paranoia.
    eps = 1e-8
    mean = mean_chw.to(device).view(1, 3, 1, 1)
    std = std_chw.to(device).view(1, 3, 1, 1).clamp_min(eps)
    x = images_uint8.to(device, non_blocking=True)
    if x.dtype == torch.uint8:
        if x.shape[1] != 3 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.float() / 255.0
    elif x.dtype == torch.float32 and x.shape[1] != 3 and x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2)
    else:
        x = x.float()
    x = (x - mean) / std
    feat = backbone(x)["feature_map"]
    return feat.to(torch.float16).cpu()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_checkpoint", required=True, type=Path,
                    help="path to .../pretrained_model dir of a trained act_simple policy")
    ap.add_argument("--dataset_root", required=True, type=Path,
                    help="root of a LeRobotDataset (contains meta/info.json)")
    ap.add_argument("--repo_id", required=True, type=str)
    ap.add_argument("--cache_root", default=None, type=Path,
                    help="override the cache parent dir. Default: "
                         "<policy_checkpoint>/encoded_backbone. The per-repo subdir "
                         "(<repo_id>/) is always appended.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=64, help="frames per backbone forward pass")
    ap.add_argument("--overwrite", action="store_true",
                    help="delete and re-encode existing cache dir for this repo_id")
    args = ap.parse_args()

    device = torch.device(args.device)
    log.info("device=%s repo_id=%s", device, args.repo_id)

    cache_parent = args.cache_root if args.cache_root is not None else (args.policy_checkpoint / "encoded_backbone")
    encoded_dir = cache_parent / args.repo_id
    meta_path = encoded_dir / "meta.json"
    if meta_path.exists() and not args.overwrite:
        log.info("cache already exists at %s. Pass --overwrite to redo.", meta_path)
        return
    if encoded_dir.exists() and args.overwrite:
        import shutil
        log.info("--overwrite: removing %s", encoded_dir)
        shutil.rmtree(encoded_dir)
    encoded_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model + stats ────────────────────────────────────────────────
    log.info("loading policy backbone from %s", args.policy_checkpoint)
    backbone = load_policy_backbone(args.policy_checkpoint, device)
    norm_stats = load_camera_norm_stats(args.policy_checkpoint)
    log.info("loaded image norm stats for %s", sorted(norm_stats))

    # ── Open the LeRobotDataset (no delta_timestamps; we want raw per-frame access) ──
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    cam_keys_in_data = list(dataset.meta.camera_keys)
    cam_keys_to_encode = [k for k in cam_keys_in_data if k in norm_stats]
    if not cam_keys_to_encode:
        raise RuntimeError(
            f"No camera keys in dataset ({cam_keys_in_data}) overlap with policy preprocessor "
            f"VISUAL keys ({sorted(norm_stats)}). Check the --policy_checkpoint."
        )
    log.info("camera keys to encode: %s", cam_keys_to_encode)

    N = int(dataset.num_frames)

    # ── Probe one frame to confirm feature shape ─────────────────────────
    sample = dataset[0]
    probe_img = sample[cam_keys_to_encode[0]]
    if probe_img.dim() == 3:
        probe_img = probe_img.unsqueeze(0)
    probe_feat = _encode_chunk(backbone, probe_img, *norm_stats[cam_keys_to_encode[0]], device)
    feat_shape = tuple(probe_feat.shape[1:])  # (C, Hf, Wf)
    log.info("feature_shape=%s, total_frames=%d", feat_shape, N)
    if feat_shape[0] != 512 or feat_shape[1] != 3 or feat_shape[2] != 3:
        log.warning("unexpected feature_shape %s (expected (512,3,3) for ResNet18 @ 84x84)", feat_shape)

    # ── Allocate memmaps ─────────────────────────────────────────────────
    memmaps: dict[str, np.memmap] = {}
    for cam in cam_keys_to_encode:
        path = encoded_dir / f"{cam}.mmap"
        memmaps[cam] = np.memmap(str(path), dtype="float16", mode="w+", shape=(N, *feat_shape))
        log.info("created %s shape=(%d, %s)", path, N, feat_shape)

    # ── Encode every frame in chunks ─────────────────────────────────────
    t0 = time.time()
    cursor = 0
    while cursor < N:
        end = min(cursor + args.chunk, N)
        # Fetch a batch by gathering individual __getitem__ calls (LeRobotDataset
        # has no native batched __getitem__; the cost here is dominated by PNG decode).
        batch_imgs: dict[str, list[torch.Tensor]] = {cam: [] for cam in cam_keys_to_encode}
        for fi in range(cursor, end):
            sample = dataset[fi]
            for cam in cam_keys_to_encode:
                batch_imgs[cam].append(sample[cam])
        for cam in cam_keys_to_encode:
            stacked = torch.stack(batch_imgs[cam], dim=0)
            mean, std = norm_stats[cam]
            feats = _encode_chunk(backbone, stacked, mean, std, device)
            memmaps[cam][cursor:end] = feats.numpy()
        cursor = end
        if (cursor // args.chunk) % 25 == 0 or cursor == N:
            elapsed = time.time() - t0
            log.info("  encoded %d / %d frames  (%.1fs, %.2f frames/s)",
                     cursor, N, elapsed, cursor / max(elapsed, 1e-9))

    # ── Flush memmaps and write meta.json ────────────────────────────────
    for cam, mm in memmaps.items():
        mm.flush()
        del mm

    meta = {
        "version": 1,
        "repo_id": args.repo_id,
        "policy_checkpoint": str(args.policy_checkpoint),
        "feature_shape": list(feat_shape),
        "dtype": "float16",
        "n_frames": N,
        "camera_keys": cam_keys_to_encode,
        "files": {cam: f"{cam}.mmap" for cam in cam_keys_to_encode},
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("wrote %s", meta_path)
    log.info("done.")


if __name__ == "__main__":
    main()
