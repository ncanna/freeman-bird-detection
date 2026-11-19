import json
import os
import re
import csv
import cv2
from pathlib import Path
import pandas as pd
from collections import defaultdict
import random
from concurrent.futures import ProcessPoolExecutor, as_completed


_FRAME_RE = re.compile(r"^(?P<prefix>.+)_frame_(?P<idx>\d{6})\.png$")


def _index_existing_frames(out_dir: Path) -> dict[str, tuple[int, int]]:
    """Scan out_dir once: frame-name prefix (SITE_VIDEO) -> (count, max_index)."""
    found: dict[str, tuple[int, int]] = {}
    with os.scandir(out_dir) as it:
        for entry in it:
            m = _FRAME_RE.match(entry.name)
            if m:
                count, max_idx = found.get(m["prefix"], (0, -1))
                found[m["prefix"]] = (count + 1, max(max_idx, int(m["idx"])))
    return found


def extract_single_video(
    site_name: str,
    video_path: Path,
    out_dir: Path,
    existing: tuple[int, int] = (0, -1),
) -> tuple[list[tuple[str, str]], bool]:
    """Extract all frames from one video, indexing from 0 per video.

    Idempotent: if out_dir already holds a complete, contiguous set of frames for
    this video (frame_000000 .. frame_{N-1}, N = the video's reported frame count),
    nothing is decoded or rewritten. Otherwise the video is re-extracted from
    scratch. Frames are written to a temp name and renamed, so an interrupted run
    never leaves a truncated PNG that a later run would mistake for a finished frame.

    `existing` is (count, max_index) of frames already on disk for this video.
    Returns (mapping, skipped).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: could not open {video_path.name}, skipping.")
        return [], False

    stem = video_path.stem
    prefix = f"{site_name}_{stem}"

    n_expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_expected > 0 and existing == (n_expected, n_expected - 1):
        cap.release()
        mapping = [(f"{prefix}_frame_{i:06d}.png", video_path.name) for i in range(n_expected)]
        return mapping, True

    mapping = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = f"{prefix}_frame_{frame_idx:06d}.png"
        tmp_path = out_dir / f".tmp_{frame_name}"
        cv2.imwrite(str(tmp_path), frame)
        os.replace(tmp_path, out_dir / frame_name)
        mapping.append((frame_name, video_path.name))
        frame_idx += 1

    cap.release()

    # Drop stale frames left over from an earlier, longer extraction of this video.
    for i in range(frame_idx, existing[1] + 1):
        (out_dir / f"{prefix}_frame_{i:06d}.png").unlink(missing_ok=True)

    return mapping, False


def extract_frames_from_dir(site_name: str, video_dir: Path, out_dir: Path, workers: int = 4) -> None:
    """Extract frames for every .mp4 in video_dir into the flat out_dir. Safe to re-run.

    Videos whose frames are already fully extracted are skipped. frame_map.csv is
    shared by all sites, so it is merged rather than overwritten: rows for the
    videos handled here are replaced and rows for every other site/video are kept.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(
        p for p in Path(video_dir).iterdir() if p.suffix.lower() == ".mp4"
    )
    if not video_paths:
        print(f"No .mp4 files found in {video_dir}")
        return

    existing = _index_existing_frames(out_dir)

    all_mappings: list[tuple[str, str]] = []
    n_skipped = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                extract_single_video, site_name, vp, out_dir,
                existing.get(f"{site_name}_{vp.stem}", (0, -1)),
            ): vp.name
            for vp in video_paths
        }
        for future in as_completed(futures):
            result, skipped = future.result()
            all_mappings.extend(result)
            n_skipped += skipped
            status = "already extracted" if skipped else "extracted"
            print(f"Finished {futures[future]} ({len(result)} frames, {status})")

    map_path = out_dir / "frame_map.csv"

    # Merge with the existing map: keep rows for other sites/videos.
    handled_prefixes = tuple(f"{site_name}_{vp.stem}_frame_" for vp in video_paths)
    kept: list[tuple[str, str]] = []
    if map_path.exists():
        with open(map_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            kept = [(r[0], r[1]) for r in reader
                    if len(r) >= 2 and not r[0].startswith(handled_prefixes)]

    merged = sorted(kept + all_mappings, key=lambda x: x[0])
    tmp_map = map_path.with_name(map_path.name + ".tmp")
    with open(tmp_map, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "video"])
        writer.writerows(merged)
    os.replace(tmp_map, map_path)

    print(
        f"Done. {len(all_mappings)} frames from {len(video_paths)} video(s) "
        f"({n_skipped} already extracted). Map → {map_path} ({len(merged)} rows total)"
    )


def extract_frames_by_split(site_name, split_json, video_dir, out_dir):
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

            frames = extract_single_video(site_name, video_path, out_dir)
            print(f"[{split_name:5s}] {video_name}: {len(frames)} frames → {out_dir}")


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


def site_from_video(video: str) -> str:
    """Site id from a video stem, e.g. 'H23_IMG_0050' -> 'H23'."""
    match = re.match(r"^([A-Za-z]+\d+)_", video)
    if not match:
        raise ValueError(
            f"Cannot parse site from video name {video!r}; expected '<SITE>_<VIDEO>' (e.g. H23_IMG_0050)"
        )
    return match.group(1)


def site_video_split(df, train_frac=0.70, val_frac=0.15, test_frac=0.15, random_state=42, save_dir=None,
                     n_trials=2000, density_weight=0.0):
    """
    Split videos into train/val/test so that every site lives in exactly one split.

    Because a site is the unit of assignment, all of its videos (and therefore all of its
    frames) land in the same split. Sites are assigned to splits by searching random site
    orderings (plus a largest-first greedy pass) for the assignment whose frame fractions are
    closest to the requested fractions. A split with a fraction of 0 receives no sites; every
    split with a positive fraction receives at least one.

    With few sites the achieved fractions are only approximate -- see split_report.

    density_weight
        Trades split-size accuracy against annotation-density balance. Each site's density is
        n_bird_frames / n_total_frames (summed over its videos from compute_split_statistics).
        The search cost is  sum|actual_share - target_share| + density_weight * (max - min of the
        splits' % annotated), so 0 ignores density and larger values accept bigger size errors in
        exchange for more similar % annotated across splits. No frames are discarded.
    """
    fracs = {"train": train_frac, "val": val_frac, "test": test_frac}
    if any(f < 0 for f in fracs.values()) or abs(sum(fracs.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must be non-negative and sum to 1, got {fracs}")
    if density_weight < 0:
        raise ValueError("density_weight must be >= 0")
    active = [k for k, f in fracs.items() if f > 0]

    per_site = df.assign(site=df["video"].map(site_from_video)).groupby("site")[["n_total_frames", "n_bird_frames"]].sum()
    site_frames = per_site["n_total_frames"]
    site_ann = per_site["n_bird_frames"]
    sites = site_frames.index.tolist()
    if len(sites) < len(active):
        raise ValueError(f"{len(sites)} site(s) cannot fill {len(active)} non-empty splits ({active})")
    total = float(site_frames.sum())

    def assign(order):
        # Give each site to the split with the largest remaining frame deficit.
        got = {k: 0.0 for k in active}
        out = {}
        for site in order:
            best = max(active, key=lambda k: fracs[k] * total - got[k])
            out[site] = best
            got[best] += site_frames[site]
        return out, got

    def cost(out, got):
        if len(set(out.values())) < len(active):
            return float("inf")  # a split ended up empty
        c = sum(abs(got[k] / total - fracs[k]) for k in active)
        if density_weight:
            ann = {k: sum(site_ann[s] for s, v in out.items() if v == k) for k in active}
            dens = [ann[k] / got[k] for k in active]
            c += density_weight * (max(dens) - min(dens))
        return c

    rng = random.Random(random_state)
    candidates = [sorted(sites, key=lambda s: -site_frames[s])]
    for _ in range(n_trials):
        order = sites[:]
        rng.shuffle(order)
        candidates.append(order)

    best_out, best_cost = None, float("inf")
    for order in candidates:
        out, got = assign(order)
        c = cost(out, got)
        if c < best_cost:
            best_out, best_cost = out, c
    if best_out is None:
        raise RuntimeError("Could not find a site assignment that fills every split")

    split_videos = {k: [] for k in fracs}
    for video in df["video"]:
        split_videos[best_out[site_from_video(video)]].append(video)

    if save_dir is not None:
        save_path = Path(save_dir) / "split.json"
        with open(save_path, "w") as f:
            json.dump({k: sorted(v) for k, v in split_videos.items()}, f, indent=2)
        print(f"Split saved to {save_path}\n")

    return split_videos["train"], split_videos["val"], split_videos["test"]


def split_report(df, splits, targets=None):
    """
    Print one table summarising a split.

    Parameters
    ----------
    df      : per-video stats from compute_split_statistics.
    splits  : {name: [video, ...]}, e.g. {"Train": train_videos, "Val": val_videos, "Test": test_videos}.
    targets : optional {name: fraction} of the requested split shares (same keys as `splits`).
    """
    grand_total = int(df[df["video"].isin([v for vids in splits.values() for v in vids])]["n_total_frames"].sum())

    rows = []
    for name, videos in splits.items():
        sub = df[df["video"].isin(videos)]
        total = int(sub["n_total_frames"].sum())
        annotated = int(sub["n_bird_frames"].sum())
        rows.append(
            (
                name,
                f"{targets[name] * 100:.1f}%" if targets and name in targets else "-",
                f"{total / grand_total * 100:.1f}%" if grand_total else "-",
                f"{total:,}",
                f"{annotated:,}",
                f"{annotated / total * 100:.1f}%" if total else "-",
            )
        )

    header = ("Split", "Target %", "Actual %", "Total frames", "Annotated frames", "% annotated")
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    fmt = lambda r: "  ".join(str(c).ljust(widths[0]) if i == 0 else str(c).rjust(widths[i]) for i, c in enumerate(r))
    print(fmt(header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt(r))


def remove_multi_bird_frames(coco_json_path: str, output_path: str, bird_category_name: str = "Bird") -> None:
    """
    Remove all frames containing more than one bird annotation from a COCO dataset
    and write the filtered result to output_path.

    Parameters
    ----------
    coco_json_path : str
        Path to the input COCO-format JSON annotation file.
    output_path : str
        Path where the filtered COCO JSON will be written.
    bird_category_name : str
        Category name to treat as "bird". Case-insensitive. Default "Bird".
    """
    with open(coco_json_path) as f:
        data = json.load(f)

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

    # Count bird annotations per image
    bird_count: dict[int, int] = defaultdict(int)
    for ann in data["annotations"]:
        if ann["category_id"] == bird_cat_id:
            bird_count[ann["image_id"]] += 1

    multi_bird_ids: set[int] = {img_id for img_id, count in bird_count.items() if count > 1}

    kept_images = [img for img in data["images"] if img["id"] not in multi_bird_ids]
    kept_image_ids = {img["id"] for img in kept_images}
    kept_annotations = [ann for ann in data["annotations"] if ann["image_id"] in kept_image_ids]

    filtered = {**data, "images": kept_images, "annotations": kept_annotations}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(filtered, f)

    removed = len(data["images"]) - len(kept_images)
    print(
        f"Removed {removed} frames with multiple bird detections. "
        f"{len(kept_images)} frames remaining → {output_path}"
    )
