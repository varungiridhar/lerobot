#!/usr/bin/env python

"""
Randomly sample N episodes from a LeRobotDataset and save as a new dataset.

Usage:
    python -m lerobot.scripts.sample_episodes \
        --repo_id lerobot/pusht \
        --new_repo_id VarunGiridhar3/pusht_100ep \
        --n_episodes 100 \
        --seed 42 \
        --push_to_hub
"""

import argparse
import logging
import random
from pathlib import Path

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--new_repo_id", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    init_logging()

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    total = dataset.meta.total_episodes
    logging.info(f"Source dataset has {total} episodes")

    if args.n_episodes > total:
        raise ValueError(f"Requested {args.n_episodes} episodes but source only has {total}")

    rng = random.Random(args.seed)
    keep = sorted(rng.sample(range(total), args.n_episodes))
    delete = sorted(set(range(total)) - set(keep))
    logging.info(f"Keeping {len(keep)} episodes, deleting {len(delete)} (seed={args.seed})")

    root = Path(args.root) if args.root else HF_LEROBOT_HOME
    output_dir = root / args.new_repo_id

    new_dataset = delete_episodes(
        dataset,
        episode_indices=delete,
        output_dir=output_dir,
        repo_id=args.new_repo_id,
    )
    logging.info(
        f"Saved {new_dataset.meta.total_episodes} episodes "
        f"({new_dataset.meta.total_frames} frames) to {output_dir}"
    )

    if args.push_to_hub:
        logging.info(f"Pushing to hub as {args.new_repo_id}")
        LeRobotDataset(args.new_repo_id, root=output_dir).push_to_hub()


if __name__ == "__main__":
    main()
