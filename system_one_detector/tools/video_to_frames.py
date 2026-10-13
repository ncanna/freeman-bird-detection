"""Convert camera-trap videos to image frames, keeping an explicit frame map.

Part of the H55 experiment (see ``system_one_detector/h55_experiment/README.md``).
Frames are named ``<video_stem>_frame_<original_index:06d>.png`` — the same
convention as ``utilities/video_dataset_prep_tools.py`` and the h23 dataset, so
``system_one_detector.datasets.frames`` can group them by video stem unchanged.

``--stride N`` keeps every Nth frame (index 0 is always kept), so a 303-frame
10 s clip at stride 10 yields 31 frames. Original indices — not renumbered
positions — go into the filenames, so a frame can always be traced back to its
position in the source video.

Usage (from the repo root):

    python -m system_one_detector.tools.video_to_frames \
        --sample-json data/H55/video_sample.json \
        --out-dir data/H55/frames_sample5_stride10 \
        --stride 10

or with explicit videos instead of a sample JSON:

    python -m system_one_detector.tools.video_to_frames \
        --videos data/H55/Download1_28072024/IMG_0432.MP4 [...] --out-dir ...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

FRAME_PREFIX = "_frame_"


def extract_one_video(video_path: Path, out_dir: Path, *, stride: int, max_frames: int | None) -> list[dict]:
    """Decode one video and write every stride-th frame as PNG.

    Returns one record per written frame. Runs in a worker process; the records
    are plain dicts because they cross a process boundary.
    """
    import cv2  # deferred: only needed in the workers and the driver

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [{"error": f"could not open {video_path}", "video": video_path.name}]

    stem = video_path.stem
    written: list[dict] = []
    original_index = 0
    kept = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if original_index % stride == 0:
                if max_frames is not None and kept >= max_frames:
                    break
                name = f"{stem}{FRAME_PREFIX}{original_index:06d}.png"
                if not cv2.imwrite(str(out_dir / name), frame):
                    raise OSError(f"cv2.imwrite failed for {name}")
                written.append({"frame": name, "video": video_path.name, "frame_index": original_index})
                kept += 1
            original_index += 1
    finally:
        cap.release()
    return written


def _videos_from_args(args: argparse.Namespace) -> list[Path]:
    if args.sample_json:
        payload = json.loads(args.sample_json.read_text())
        return [Path(entry["path"]) for entry in payload["videos"]]
    if args.videos:
        return [Path(v) for v in args.videos]
    if args.video_dir:
        return sorted(p for p in args.video_dir.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".avi"))
    raise SystemExit("one of --sample-json, --videos, or --video-dir is required")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample-json", type=Path, help="video_sample.json from sample_videos")
    group.add_argument("--videos", nargs="+", type=Path, help="explicit video paths")
    group.add_argument("--video-dir", type=Path, help="use every video in this directory")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=10, help="keep every Nth frame (1 = all)")
    parser.add_argument("--max-frames", type=int, default=None, help="cap frames written per video")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--overwrite-ok", action="store_true",
        help="write into a non-empty out-dir (existing frames are left in place)",
    )
    args = parser.parse_args(argv)

    if args.stride < 1:
        parser.error("--stride must be >= 1")

    videos = _videos_from_args(args)
    missing = [v for v in videos if not v.exists()]
    if missing:
        raise SystemExit(f"video(s) not found: {missing}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in out_dir.iterdir() if p.suffix == ".png"]
    if existing and not args.overwrite_ok:
        raise SystemExit(
            f"{out_dir} already holds {len(existing)} frame(s). Move them or pass --overwrite-ok."
        )

    print(f"Extracting from {len(videos)} video(s), stride {args.stride}, {args.workers} worker(s) → {out_dir}")
    records: list[dict] = []
    failures: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                extract_one_video, video, out_dir,
                stride=args.stride, max_frames=args.max_frames,
            ): video
            for video in videos
        }
        for future in as_completed(futures):
            video = futures[future]
            result = future.result()
            errors = [r for r in result if "error" in r]
            written = [r for r in result if "error" not in r]
            records.extend(written)
            failures.extend(errors)
            note = f" ({errors[0]['error']})" if errors else ""
            print(f"  {video.name:24s} {len(written):4d} frames{note}")

    records.sort(key=lambda r: (r["video"], r["frame_index"]))

    map_path = out_dir / "frame_map.csv"
    with open(map_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "video", "frame_index"])
        writer.writeheader()
        writer.writerows(records)

    provenance = {
        "tool": "system_one_detector.tools.video_to_frames",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample_json": str(args.sample_json) if args.sample_json else None,
        "videos": [str(v) for v in videos],
        "stride": args.stride,
        "max_frames": args.max_frames,
        "n_frames_total": len(records),
        "n_frames_per_video": {
            video.name: sum(1 for r in records if r["video"] == video.name) for video in videos
        },
        "failures": failures,
        "frame_map": str(map_path),
        "naming": "<video_stem>_frame_<original_index:06d>.png",
    }
    prov_path = out_dir / "extraction_provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2))

    print(f"Done. {len(records)} frames from {len(videos)} video(s). Map → {map_path}")
    if failures:
        print(f"{len(failures)} video(s) had errors — see {prov_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
