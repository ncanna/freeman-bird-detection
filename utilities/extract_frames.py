import csv
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def _extract_single_video(
    video_path: Path, out_dir: Path, start_idx: int
) -> list[tuple[str, str]]:
    """Extract all frames from one video, starting global index at start_idx."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: could not open {video_path.name}, skipping.")
        return []

    mapping = []
    frame_idx = start_idx
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_dir / frame_name), frame)
        mapping.append((frame_name, video_path.name))
        frame_idx += 1

    cap.release()
    return mapping


def extract_frames(video_dir: Path, out_dir: Path, workers: int = 4) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(
        p for p in Path(video_dir).iterdir() if p.suffix.lower() == ".mp4"
    )
    if not video_paths:
        print(f"No .mp4 files found in {video_dir}")
        return

    # Pre-compute per-video frame counts to assign non-overlapping global indices
    frame_counts = []
    for vp in video_paths:
        cap = cv2.VideoCapture(str(vp))
        frame_counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()

    start_indices = [0] + list(__import__("itertools").accumulate(frame_counts))

    all_mappings: list[tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_extract_single_video, vp, out_dir, start): vp.name
            for vp, start in zip(video_paths, start_indices)
        }
        for future in as_completed(futures):
            result = future.result()
            all_mappings.extend(result)
            print(f"Finished {futures[future]} ({len(result)} frames)")

    # Sort by frame name to restore global order in the CSV
    all_mappings.sort(key=lambda x: x[0])

    map_path = out_dir / "frame_map.csv"
    with open(map_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "video"])
        writer.writerows(all_mappings)

    total = sum(frame_counts)
    print(f"Done. {total} frames from {len(video_paths)} video(s). Map → {map_path}")