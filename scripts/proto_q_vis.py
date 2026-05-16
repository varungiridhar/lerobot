#!/usr/bin/env python
"""Prototype: run Q-function visualization with random weights on LIBERO data.

Usage (on an interactive GPU node):
    cd /storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/lerobot
    conda activate lerobot
    python scripts/proto_q_vis.py

Outputs MP4s to outputs/q_vis/ (created automatically).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = "/storage/home/hcoda1/7/igeorgiev3/shared/lerobot-data-2"
REPO_ID = "HuggingFaceVLA/libero"

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_DATASETS_CACHE",
    "/storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/.cache/huggingface/datasets")


def main():
    import draccus
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.q_function.configuration_q_function import QFunctionConfig
    from lerobot.policies.factory import make_policy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    H = 32

    # Load dataset first to discover FPS, then reload with delta_timestamps
    logger.info("Loading dataset %s from %s (metadata only first)", REPO_ID, DATA_ROOT)
    _ds_meta_only = LeRobotDataset(repo_id=REPO_ID, root=f"{DATA_ROOT}/{REPO_ID}")
    FPS = _ds_meta_only.fps
    logger.info("Dataset FPS: %d", FPS)
    del _ds_meta_only

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=f"{DATA_ROOT}/{REPO_ID}",
        delta_timestamps={
            "observation.images.image": [0.0, H / FPS],
            "observation.images.image2": [0.0, H / FPS],
            "action": [i / FPS for i in range(2 * H)],
        },
    )
    logger.info("Dataset: %d episodes, %d frames", dataset.meta.total_episodes, dataset.num_frames)

    cfg = QFunctionConfig(
        h=H,
        dino_model_name="facebook/dinov2-large",
        dim_model=1024,
        n_heads=16,
        dim_feedforward=4096,
        n_decoder_layers=18,
        image_resize_h=224,
        image_resize_w=224,
        reward_mode="all_success",
        terminal_bonus_uniform=1.0,
        step_reward=0.0,
        v_min=-0.01,
        v_max=1.01,
        hl_gauss_sigma=0.0075,
        use_text_conditioning=True,
        text_encoder_model="google/t5-v1_1-base",
        language_key="task",
        gamma=0.99,
        target_tau=0.005,
        optimizer_lr=1e-4,
        optimizer_lr_backbone=3e-5,
        optimizer_weight_decay=1e-4,
    )

    logger.info("Building Q-function policy with random weights")
    policy = make_policy(cfg=cfg, ds_meta=dataset.meta).to(device)
    logger.info("Policy ready (random weights, fp32)")
    torch.cuda.empty_cache()

    from lerobot.policies.q_function.q_vis import log_q_visualizations
    logger.info("Running Q-function visualization (3 episodes)...")
    log_q_visualizations(
        policy=policy,
        raw_dataset=dataset,
        step=0,
        wandb_logger=None,
        num_episodes=3,
        device=device,
    )
    logger.info("Done. Videos saved to outputs/q_vis/")


if __name__ == "__main__":
    main()
