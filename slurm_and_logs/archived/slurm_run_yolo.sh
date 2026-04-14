#!/bin/bash
#SBATCH --job-name=yolo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/slurm_logs/yolo.%j.out

module load anaconda3/2023.03
conda activate bird_behavior

nvidia-smi
python /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/models/yolo.py \
  --train_data /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/train.yaml \
  --test_data  /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/test/IMG_0165.MP4 \
  --output_dir /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/output \
  --epochs 5
