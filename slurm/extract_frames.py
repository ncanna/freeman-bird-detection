"""Standalone frame extraction — mirrors the 'Extract Frames' cell in prepare_dataset.ipynb.

Extracts all frames from every .mp4 in data/<site>/videos into data/<site>/images.
Safe to re-run: frame filenames are deterministic ({stem}_frame_{idx:06d}.png),
so existing frames are overwritten, never duplicated.
"""

import sys
from pathlib import Path

# Repo root is the parent of this slurm/ directory
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "utilities"))

from video_dataset_prep_tools import extract_frames_from_dir

SITES = ["h23"]

if __name__ == "__main__":
    for site in SITES:
        video_dir = REPO_ROOT / "data" / site / "videos"
        out_dir = REPO_ROOT / "data" / site / "images"
        print(f"[{site}] extracting {video_dir} -> {out_dir}", flush=True)
        extract_frames_from_dir(video_dir, out_dir, workers=8)
    print("All sites done.", flush=True)
