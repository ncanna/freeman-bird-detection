#!/bin/bash
#SBATCH --job-name=yolo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/slurm_and_logs/logs/yolo.train_and_val.%j.out

module load anaconda3/2023.03
conda activate bird_behavior

# Test run: prediction only
nvidia-smi
python /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/models/yolo.py \
  --train_data /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/train.yaml \
  --val_data  /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/train.yaml \
  --output_dir /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/output/train_and_val \
  --draw_bounding_box \
  --epochs 1
