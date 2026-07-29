#!/bin/bash
#SBATCH --job-name=h23_hpo
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="cuda13&H200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=slurm/logs/hpo_%x_%j.out
#SBATCH --error=slurm/logs/hpo_%x_%j.err

set -euo pipefail

CONFIG="${1:?Usage: sbatch slurm/run_hpo.sbatch <hpo_config_yaml>}"
PYTHON=/home/hice1/abrown693/scratch/.conda/envs/freeman/bin/python

echo "Job $SLURM_JOB_ID on $(hostname) — config=$CONFIG — $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PYTHON" run_hpo.py "$CONFIG"
echo "Finished at $(date)"
