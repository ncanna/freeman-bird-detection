'''
Extract frames from a video file
YOLO needs to run on each frame (it is an "image detector") for training.
May be able to run on video directly for prediction.

Example usage:
python /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/utilities/extract_frames.py \
--input_video /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/train/IMG_0169.MP4 \
--output_dir /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/frames/train
'''

import cv2
import argparse
import os

parser = argparse.ArgumentParser(description='Extract frames from a video file')
parser.add_argument('--input_video', type=str,
                    help='Input video to extract frames from')
parser.add_argument('--output_dir', type=str, default='data/frames',
                    help='Path to output directory')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)
# Get prefix for the output file names
output_prefix = os.path.splitext(os.path.split(args.input_video)[1])[0]

cap = cv2.VideoCapture(args.input_video)
i = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Save frame to output directory
    # Use %05d padded with leading zeros if the number has <5 digits.
    cv2.imwrite(f"{args.output_dir}/{output_prefix}_frame_{i:05d}.jpg", frame)
    i += 1
    if i % 100 == 0:
        print(f"\r# - Extracted {i} frames", end="", flush=True)
print(f"\r# - Extracted {i} frames")

cap.release()

