#!/bin/bash
# Build the policy.json + critic.json manifests for the v1 dataset.
# Light-weight (just walks HDF5 trees), but submitted via slurm to chain
# after the play-rollout jobs finish.
set -euxo pipefail

PYBIN="/storage/home/hcoda1/6/vgiridhar6/.conda/envs/lerobot-mg-datagen/bin/python"

cd /storage/home/hcoda1/6/vgiridhar6/forks/lerobot
"${PYBIN}" -u experiments/mg_dataset_v1/build_manifests.py
