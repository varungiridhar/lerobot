#!/usr/bin/env python
"""Convert an official FastWAM .pt checkpoint to HuggingFace format for use with lerobot-eval.

Downloads Wan2.2 base weights (~20 GB) on first run, loads the fine-tuned weights on top,
then saves the full model to a local directory in safetensors format.

Usage:
    python convert_fastwam_checkpoint.py

After conversion, evaluate with:
    lerobot-eval \\
        --policy.path=<OUTPUT_DIR> \\
        --policy.device=cuda \\
        --env.type=libero \\
        --env.task=libero_10 \\
        --env.observation_height=224 \\
        --env.observation_width=224 \\
        --eval.batch_size=1 \\
        --eval.n_episodes=50
"""

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Where DiffSynth-based loader stores/looks for Wan2.2 weights.
WAN22_WEIGHTS_DIR = "/storage/project/r-agarg35-0/shared/fastwam/wan22_weights"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        default="/storage/project/r-agarg35-0/shared/fastwam/libero_uncond_2cam224.pt",
        help="Path to the official FastWAM .pt checkpoint file.",
    )
    parser.add_argument(
        "--dataset-stats",
        default="/storage/project/r-agarg35-0/shared/fastwam/libero_uncond_2cam224_dataset_stats.json",
        help="Path to the dataset stats JSON file (from the same HuggingFace release).",
    )
    parser.add_argument(
        "--output-dir",
        default="/storage/project/r-agarg35-0/shared/fastwam/hf_checkpoint",
        help="Directory to save the converted HuggingFace-format checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to load the model on during conversion (default: cuda).",
    )
    parser.add_argument(
        "--wan22-weights-dir",
        default=WAN22_WEIGHTS_DIR,
        help="Local directory where Wan2.2 weights are stored/downloaded.",
    )
    args = parser.parse_args()

    # Tell the DiffSynth-based loader to use HuggingFace (not ModelScope) and where to save weights.
    os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = args.wan22_weights_dir

    checkpoint = Path(args.checkpoint)
    dataset_stats = Path(args.dataset_stats)
    output_dir = Path(args.output_dir)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not dataset_stats.exists():
        raise FileNotFoundError(f"Dataset stats not found: {dataset_stats}")

    logger.info(f"Loading FastWAM checkpoint from: {checkpoint}")
    logger.info(f"Using dataset stats from:         {dataset_stats}")
    logger.info(f"Wan2.2 weights dir:               {args.wan22_weights_dir}")
    logger.info("(Wan2.2 base weights will be downloaded from HuggingFace on first run ~20 GB)")

    from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
    from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy

    # libero_uncond checkpoint: unconditional, no text encoder needed (saves ~8 GB download).
    # redirect_common_files=False: use original Wan-AI/Wan2.2-TI2V-5B directly (DiffSynth mirror is private).
    config = FastWAMConfig(
        load_wan22_weights=True,
        load_text_encoder=False,
        redirect_common_files=False,
        device=args.device,
    )

    policy = FastWAMPolicy.from_fastwam_checkpoint(
        checkpoint_path=str(checkpoint),
        dataset_stats_path=str(dataset_stats),
        config=config,
        device=args.device,
    )

    logger.info(f"Saving converted checkpoint to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(output_dir))
    logger.info("Done.")
    logger.info("")
    logger.info("To evaluate, run:")
    logger.info(
        f"  lerobot-eval \\\n"
        f"      --policy.path={output_dir} \\\n"
        f"      --policy.device=cuda \\\n"
        f"      --env.type=libero \\\n"
        f"      --env.task=libero_10 \\\n"
        f"      --env.observation_height=224 \\\n"
        f"      --env.observation_width=224 \\\n"
        f"      --eval.batch_size=1 \\\n"
        f"      --eval.n_episodes=50"
    )


if __name__ == "__main__":
    main()
