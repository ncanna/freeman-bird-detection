import csv
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def extract_single_video(
    video_path: Path, out_dir: Path
) -> list[tuple[str, str]]:
    """Extract all frames from one video, indexing from 0 per video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: could not open {video_path.name}, skipping.")
        return []

    stem = video_path.stem
    mapping = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = f"{stem}_frame_{frame_idx:06d}.png"
        cv2.imwrite(str(out_dir / frame_name), frame)
        mapping.append((frame_name, video_path.name))
        frame_idx += 1

    cap.release()
    return mapping


def extract_frames_from_dir(video_dir: Path, out_dir: Path, workers: int = 4) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(
        p for p in Path(video_dir).iterdir() if p.suffix.lower() == ".mp4"
    )
    if not video_paths:
        print(f"No .mp4 files found in {video_dir}")
        return

    all_mappings: list[tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(extract_single_video, vp, out_dir): vp.name
            for vp in video_paths
        }
        for future in as_completed(futures):
            result = future.result()
            all_mappings.extend(result)
            print(f"Finished {futures[future]} ({len(result)} frames)")

    all_mappings.sort(key=lambda x: x[0])

    map_path = out_dir / "frame_map.csv"
    with open(map_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "video"])
        writer.writerows(all_mappings)

    total = len(all_mappings)
    print(f"Done. {total} frames from {len(video_paths)} video(s). Map → {map_path}")