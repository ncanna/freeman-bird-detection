"""Pick a random video sample from a camera-trap directory and record provenance.

Part of the H55 experiment (see ``system_one_detector/h55_experiment/README.md``).
The sample is reproducible: the seed, the candidate list, and the draw are all
written to the output JSON, so any later reader can verify the selection without
trusting the summary.

Usage (from the repo root):

    python -m system_one_detector.tools.sample_videos \
        --video-dir data/H55/Download1_28072024 \
        --num 5 --seed 42 \
        --output data/H55/video_sample.json
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def list_videos(video_dir: Path) -> list[Path]:
    """Every video file in the directory, sorted, so the draw is stable."""
    return sorted(
        p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES and p.is_file()
    )


def probe_video(path: Path) -> dict:
    """Stream metadata via ffprobe. Missing fields stay ``None`` — metadata only,
    the file is never decoded."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,codec_name",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    info: dict = {}
    try:
        raw = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        payload = json.loads(raw.stdout)
        stream = (payload.get("streams") or [{}])[0]
        fps_raw = stream.get("r_frame_rate") or ""
        if "/" in fps_raw:
            num, den = fps_raw.split("/", 1)
            fps = round(float(num) / float(den), 4) if float(den) else None
        else:
            fps = float(fps_raw) if fps_raw else None
        info = {
            "codec": stream.get("codec_name"),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "fps": fps,
            "n_frames": int(stream["nb_frames"]) if stream.get("nb_frames") else None,
            "duration_s": round(float(payload["format"]["duration"]), 3)
            if payload.get("format", {}).get("duration")
            else None,
        }
    except (subprocess.SubprocessError, json.JSONDecodeError, KeyError, ValueError) as exc:
        info = {"probe_error": f"{type(exc).__name__}: {exc}"}
    return info


def build_sample(
    video_dir: Path,
    *,
    num: int,
    seed: int,
) -> dict:
    """Draw ``num`` videos uniformly at random and describe each one."""
    candidates = list_videos(video_dir)
    if not candidates:
        raise FileNotFoundError(f"no video files under {video_dir}")
    if num > len(candidates):
        raise ValueError(f"asked for {num} videos but only {len(candidates)} exist")

    rng = random.Random(seed)
    chosen = sorted(rng.sample(candidates, num))

    videos = []
    for path in chosen:
        entry = {
            "file_name": path.name,
            "video_stem": path.stem,
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }
        entry.update(probe_video(path))
        videos.append(entry)

    return {
        "tool": "system_one_detector.tools.sample_videos",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": (
            "uniform draw without replacement: "
            "random.Random(seed).sample(sorted(candidate_paths), num), then sorted()"
        ),
        "video_dir": str(video_dir),
        "seed": seed,
        "num_requested": num,
        "num_candidates": len(candidates),
        "candidate_suffixes": list(VIDEO_SUFFIXES),
        "videos": videos,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, required=True, help="Where to write the sample provenance JSON."
    )
    args = parser.parse_args(argv)

    sample = build_sample(args.video_dir, num=args.num, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sample, indent=2))

    print(f"Drew {len(sample['videos'])} of {sample['num_candidates']} videos "
          f"(seed {args.seed}) from {args.video_dir}")
    for entry in sample["videos"]:
        size_mb = entry["size_bytes"] / 1e6
        print(
            f"  {entry['file_name']:16s} {entry.get('width', '?')}x{entry.get('height', '?')} "
            f"{entry.get('n_frames', '?'):>4} frames {entry.get('duration_s', '?'):>6}s "
            f"{size_mb:5.1f} MB"
        )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
