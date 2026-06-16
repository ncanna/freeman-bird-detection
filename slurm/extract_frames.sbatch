#!/bin/bash
#SBATCH --job-name=extract_frames
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/extract_frames_%j.out
#SBATCH --error=slurm/logs/extract_frames_%j.err

# Run from the repo root: sbatch slurm/extract_frames.sbatch
set -euo pipefail

PYTHON=/home/hice1/abrown693/scratch/.conda/envs/freeman/bin/python

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
"$PYTHON" slurm/extract_frames.py
echo "Finished at $(date)"
