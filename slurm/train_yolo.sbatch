#!/bin/bash
#SBATCH --job-name=yolo_train
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
# torch here is a CUDA-13 build, so only schedule onto GPUs with a CUDA-13 driver
# (e.g. A100 nodes). Without this, jobs landing on CUDA-12 V100 nodes crash with
# torch.cuda.is_available()==False.
#SBATCH --constraint=cuda13
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=slurm/logs/train_%x_%j.out
#SBATCH --error=slurm/logs/train_%x_%j.err

# Submit one model:
#   sbatch slurm/train_yolo.sbatch configs/yolo11_h23_full.yaml
#   sbatch slurm/train_yolo.sbatch configs/yolo26_h23_full.yaml
set -euo pipefail

CONFIG="${1:?Usage: sbatch slurm/train_yolo.sbatch <config_yaml>}"
PYTHON=/home/hice1/abrown693/scratch/.conda/envs/freeman/bin/python

# If the GPU compute node has no outbound internet, uncomment to log W&B
# offline, then run `wandb sync slurm/logs/wandb/<run>` from the login node:
# export WANDB_MODE=offline

echo "Job $SLURM_JOB_ID on $(hostname) — config=$CONFIG — $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PYTHON" slurm/run_experiment.py "$CONFIG"
echo "Finished at $(date)"
