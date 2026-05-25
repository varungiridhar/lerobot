#!/bin/bash
# Q-function-on-RoboTwin smoke train (pipeline check) — rtx6000.
#     sbatch scripts/train_q_robotwin_smoke_sbatch.sh
#SBATCH --job-name=qf-robotwin-smoke
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=80G
#SBATCH -q embers
#SBATCH -t 2:00:00
#SBATCH --output=slurm_out/Report-%j.out

export MODE=smoke
exec bash /storage/home/hcoda1/6/vgiridhar6/forks/lerobot-qfunction-robotwin/scripts/run_robotwin_qfunction_train.sh
