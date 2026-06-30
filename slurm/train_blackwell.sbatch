#!/bin/bash
#SBATCH --job-name=rtdetr_bw
# Blackwell GPUs live in the ice-bw-gpu partition, NOT ice-gpu.
#SBATCH --partition=ice-bw-gpu
#SBATCH --gres=gpu:1
# torch here is a CUDA-13 build; Blackwell nodes are cuda13. This pins onto an
# RTX Pro 6000 Blackwell node for a head-to-head against the H200.
#SBATCH --constraint="cuda13&RTX-Pro-Blackwell"
# Kept identical to the H200 sbatch (8 cpus / 64G / workers=4 in the adapter) so
# the only variable in the comparison is the GPU. Node has 96 cores, 16 GPUs.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=slurm/logs/train_%x_%j.out
#SBATCH --error=slurm/logs/train_%x_%j.err

# Submit one model:
#   sbatch slurm/train_blackwell.sbatch configs/rtdetr_h23_full.yaml
set -euo pipefail

CONFIG="${1:?Usage: sbatch slurm/train_blackwell.sbatch <config_yaml>}"
PYTHON=/home/hice1/abrown693/scratch/.conda/envs/freeman/bin/python

# If the GPU compute node has no outbound internet, uncomment to log W&B
# offline, then run `wandb sync slurm/logs/wandb/<run>` from the login node:
# export WANDB_MODE=offline

echo "Job $SLURM_JOB_ID on $(hostname) — config=$CONFIG — $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PYTHON" slurm/run_experiment.py "$CONFIG"
echo "Finished at $(date)"
