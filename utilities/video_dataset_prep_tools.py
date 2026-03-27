import json
import re
import csv
import cv2
from pathlib import Path
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
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


def compute_split_statistics(coco_json_path: str, bird_category_name: str = "Bird") -> pd.DataFrame:
    """
    Compute per-video statistics from a COCO annotation file for use in
    stratified train/val/test splitting.

    Frame file names are expected to encode the source video, e.g.:
        IMG_0073_frame_000000.png  →  video "IMG_0073"

    Parameters
    ----------
    coco_json_path : str
        Path to the COCO-format JSON annotation file.
    bird_category_name : str
        Category name to treat as "bird". Case-insensitive. Default "Bird".

    Returns
    -------
    pd.DataFrame
        One row per video with columns:
          - video          : video identifier parsed from the frame filename
          - n_total_frames : total annotated frames in this video
          - n_bird_frames  : frames that contain at least one bird annotation
          - prevalence     : n_bird_frames / n_total_frames
          - stratum        : quartile bin [0, 3] assigned by pd.qcut on prevalence
    """
    with open(coco_json_path) as f:
        data = json.load(f)

    # Resolve bird category id
    bird_cat_id = None
    for cat in data["categories"]:
        if cat["name"].lower() == bird_category_name.lower():
            bird_cat_id = cat["id"]
            break
    if bird_cat_id is None:
        raise ValueError(
            f"Category '{bird_category_name}' not found. "
            f"Available: {[c['name'] for c in data['categories']]}"
        )

    # Map image_id → video name
    image_to_video: dict[int, str] = {}
    for img in data["images"]:
        fname = img["file_name"]
        # Extract video prefix: everything before "_frame_"
        match = re.match(r"^(.+)_frame_\d+", fname)
        video = match.group(1) if match else fname
        image_to_video[img["id"]] = video

    # Collect frames that have at least one bird annotation
    bird_image_ids: set[int] = set()
    for ann in data["annotations"]:
        if ann["category_id"] == bird_cat_id:
            bird_image_ids.add(ann["image_id"])

    # Aggregate per video
    video_total: dict[str, set[int]] = defaultdict(set)
    video_bird: dict[str, set[int]] = defaultdict(set)

    for image_id, video in image_to_video.items():
        video_total[video].add(image_id)
        if image_id in bird_image_ids:
            video_bird[video].add(image_id)

    rows = []
    for video, frame_ids in video_total.items():
        n_total = len(frame_ids)
        n_bird = len(video_bird[video])
        rows.append(
            {
                "video": video,
                "n_total_frames": n_total,
                "n_bird_frames": n_bird,
                "prevalence": n_bird / n_total,
            }
        )

    df = pd.DataFrame(rows).sort_values("video").reset_index(drop=True)

    # Bin each video by bird prevalence quartile → use as stratum label
    prevalence = df["prevalence"].tolist()
    df["stratum"] = pd.qcut(prevalence, q=4, labels=False, duplicates="drop")

    return df


def stratified_video_split(df, train_frac=0.70, val_frac=0.20, test_frac=0.10, random_state=42, save_dir=None):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    train_videos, val_videos, test_videos = [], [], []

    for stratum_id, group in df.groupby("stratum"):
        videos = group["video"].tolist()

        # First cut: train vs (val+test)
        train, val_test = train_test_split(
            videos,
            test_size=(val_frac + test_frac),
            random_state=random_state,
        )

        # Second cut: val vs test
        relative_test_frac = test_frac / (val_frac + test_frac)
        val, test = train_test_split(
            val_test,
            test_size=relative_test_frac,
            random_state=random_state,
        )

        train_videos.extend(train)
        val_videos.extend(val)
        test_videos.extend(test)

    if save_dir is not None:
        save_path = Path(save_dir) / "split.json"
        with open(save_path, "w") as f:
            json.dump({"train": sorted(train_videos), "val": sorted(val_videos), "test": sorted(test_videos)}, f, indent=2)
        print(f"Split saved to {save_path}")

    return train_videos, val_videos, test_videos


def split_report(df, videos, name):
    sub = df[df["video"].isin(videos)]
    print(
        f"{name:6s} | {len(videos):3d} videos | "
        f"{sub['n_total_frames'].sum():6d} frames | "
        f"bird prevalence: {sub['prevalence'].mean():.3f} ± {sub['prevalence'].std():.3f}"
    )


def extract_frames_by_split(split_json, video_dir, out_dir):
    with open(split_json, "r") as f:
            splits: dict[str, list[str]] = json.load(f)

    for split_name, video_names in splits.items():
        out_dir = out_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for video_name in video_names:
            video_path = video_dir / f"{video_name}.MP4"
            if not video_path.exists():
                video_path = video_dir / f"{video_name}.mp4"
            if not video_path.exists():
                print(f"WARNING: video not found for {video_name}, skipping.")
                continue

            frames = extract_single_video(video_path, out_dir)
            print(f"[{split_name:5s}] {video_name}: {len(frames)} frames → {out_dir}")