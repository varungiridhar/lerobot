#!/bin/bash
# Q-function-on-RoboTwin probe-scale training run — l40s.
#     sbatch scripts/train_q_robotwin_probe_sbatch.sh
# Override:  STEPS=10000 BATCH=8 sbatch scripts/train_q_robotwin_probe_sbatch.sh
#SBATCH --job-name=qf-robotwin-probe
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH --gres=gpu:L40s:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=80G
#SBATCH -q embers
#SBATCH -t 8:00:00
#SBATCH --output=slurm_out/Report-%j.out

export MODE=probe
exec bash /storage/home/hcoda1/6/vgiridhar6/forks/lerobot-qfunction-robotwin/scripts/run_robotwin_qfunction_train.sh
