"""Convert System One per-frame decisions into a COCO-format annotation file.

Part of the H55 experiment (see ``system_one_detector/h55_experiment/README.md``).
System One models answer typed questions, not boxes. This tool maps those
answers onto COCO the only way that does not invent information:

- every frame becomes an ``images`` entry (an empty annotation list means the
  model predicted no bird);
- when the model says a bird is present (``bird_present`` noul >= threshold) AND
  names a grid cell for ``largest_bird_cell``, one annotation is emitted whose
  ``bbox`` is the predicted 3x3 cell rectangle — a third of the frame in each
  dimension, per the cell definition in ``datasets/questions.py``;
- ``score`` (detector confidence in pycocotools terms) is the bird-present
  probability; the count level and cell distribution ride along in
  ``attributes`` so nothing is thrown away.

Two input shapes are understood and auto-detected:

- an ``engine`` decisions.json written by ``system_one_detector.engine``;
- a ``poc`` results JSON written by ``system_one_detector.jev_omni_mlx_poc``
  (its cell options are prose labels and carry no "none" option).

Usage (from the repo root):

    python -m system_one_detector.tools.coco_from_decisions \
        --input outputs/h55/h55_poc_20261004_120000/poc_results.json \
        --frames-dir data/H55/frames_sample5_stride10 \
        --output outputs/h55/h55_poc_20261004_120000/coco_annotations.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from system_one_detector.datasets.questions import CELL_NAMES, COUNT_CRITERIA

# Prose cell labels used by the MLX POC -> 3x3 cell codes from questions.py.
POC_CELL_LABEL_TO_CODE = {
    "top-left of the frame": "tl",
    "top-center of the frame": "tc",
    "top-right of the frame": "tr",
    "middle-left of the frame": "ml",
    "center of the frame": "mc",
    "middle-right of the frame": "mr",
    "bottom-left of the frame": "bl",
    "bottom-center of the frame": "bc",
    "bottom-right of the frame": "br",
}

POC_PRESENT_OPTION = "At least one bird is visible"


# ---------------------------------------------------------------------------
# input normalization
# ---------------------------------------------------------------------------


def detect_source(payload: dict[str, Any]) -> str:
    frames = payload.get("frames")
    if isinstance(frames, list) and frames:
        first = frames[0]
        if isinstance(first, dict):
            if "answers" in first:
                return "engine"
            if "decisions" in first:
                return "poc"
    raise ValueError("input is neither an engine decisions.json nor a POC results JSON")


def _count_label(level: int) -> str:
    return COUNT_CRITERIA[level] if 0 <= level < len(COUNT_CRITERIA) else f"level {level}"


def parse_engine_frame(row: dict[str, Any]) -> dict[str, Any]:
    answers = row.get("answers") or {}
    noul_answer = answers.get("bird_present") or {}
    count_answer = answers.get("bird_count") or {}
    cell_answer = answers.get("largest_bird_cell") or {}
    score = count_answer.get("score")
    level = int(round(float(score))) if score is not None else None
    if level is not None:
        level = max(0, min(3, level))
    return {
        "file_name": row["file_name"],
        "video_stem": row.get("video_stem") or row["file_name"].split("_frame_")[0],
        "media_path": row.get("media_path"),
        "error": row.get("error"),
        "model": row.get("model") or "",
        "noul": noul_answer.get("noul"),
        "count_level": level,
        "count_score": score,
        "count_probabilities": count_answer.get("probabilities") or {},
        "cell": cell_answer.get("choice"),
        "cell_probabilities": cell_answer.get("probabilities") or {},
        "cell_confidence": cell_answer.get("confidence"),
    }


def parse_poc_frame(row: dict[str, Any]) -> dict[str, Any]:
    by_id = {d.get("question_id"): d for d in row.get("decisions") or []}
    present = by_id.get("bird_present") or {}
    count = by_id.get("bird_count") or {}
    cell = by_id.get("largest_bird_cell") or {}

    noul = (present.get("probabilities") or {}).get(POC_PRESENT_OPTION)
    count_options = count.get("options") or []
    count_prediction = count.get("prediction")
    level = count_options.index(count_prediction) if count_prediction in count_options else None
    cell_prediction = cell.get("prediction")
    cell_code = POC_CELL_LABEL_TO_CODE.get(cell_prediction)
    if cell_prediction is not None and cell_code is None:
        cell_code = cell_prediction  # already a cell code, or something unknown; keep verbatim
    return {
        "file_name": row["file_name"],
        "video_stem": row["file_name"].split("_frame_")[0],
        "media_path": row.get("path"),
        "error": row.get("error"),
        "model": "jev-omni-mlx-4bit",
        "noul": noul,
        "count_level": level,
        "count_score": None,
        "count_probabilities": count.get("probabilities") or {},
        "cell": cell_code,
        "cell_probabilities": cell.get("probabilities") or {},
        "cell_confidence": cell.get("confidence"),
    }


# ---------------------------------------------------------------------------
# image geometry
# ---------------------------------------------------------------------------


def png_size(path: Path) -> tuple[int, int] | None:
    """Width/height straight from the PNG IHDR chunk, no image library."""
    try:
        with open(path, "rb") as handle:
            header = handle.read(24)
        if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
            width, height = struct.unpack(">II", header[16:24])
            return int(width), int(height)
    except OSError:
        pass
    return None


def image_size(path: Path | None) -> tuple[int, int] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    size = png_size(path)
    if size:
        return size
    try:
        import cv2  # deferred: only needed for non-PNG frames

        image = cv2.imread(str(path))
        if image is not None:
            height, width = image.shape[:2]
            return int(width), int(height)
    except ImportError:
        pass
    return None


# ---------------------------------------------------------------------------
# COCO assembly
# ---------------------------------------------------------------------------


def cell_bbox(cell: str, width: int, height: int) -> tuple[float, float, float, float]:
    """The 3x3 cell rectangle as COCO ``[x, y, w, h]`` (columns then rows, tl origin)."""
    row = "tmb".index(cell[0])
    col = "lcr".index(cell[1])
    x = col * width / 3.0
    y = row * height / 3.0
    return round(x, 2), round(y, 2), round(width / 3.0, 2), round(height / 3.0, 2)


def build_coco(
    frames: list[dict[str, Any]],
    *,
    source: str,
    input_path: Path,
    noul_threshold: float,
    category_name: str = "bird",
    model_hint: str | None = None,
) -> dict[str, Any]:
    known_cells = set(CELL_NAMES)
    records = []
    for frame in frames:
        media = frame.get("media_path")
        size = image_size(Path(media) if media else None)
        records.append((frame, size))
    records.sort(key=lambda item: (item[0]["video_stem"], item[0]["file_name"]))

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    n_without_size = 0
    next_image_id = 1
    next_ann_id = 1
    for frame, size in records:
        if size is None:
            n_without_size += 1
            continue
        width, height = size
        image_id = next_image_id
        next_image_id += 1
        images.append({
            "id": image_id,
            "file_name": frame["file_name"],
            "width": width,
            "height": height,
            "video_stem": frame["video_stem"],
            "attributes": {
                "noul_probability": frame.get("noul"),
                "predicted_bird_count_level": frame.get("count_level"),
                "predicted_bird_cell": frame.get("cell"),
            },
        })

        noul = frame.get("noul")
        cell = frame.get("cell")
        if noul is None or float(noul) < noul_threshold:
            continue  # model says: no bird -> no annotation
        if cell is None or cell not in known_cells:
            continue  # present but no usable location -> documented, not invented
        x, y, w, h = cell_bbox(cell, width, height)
        annotations.append({
            "id": next_ann_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [x, y, w, h],
            "area": round(w * h, 2),
            "iscrowd": 0,
            "score": round(float(noul), 4),
            "attributes": {
                "cell": cell,
                "cell_probability": frame.get("cell_confidence"),
                "cell_probabilities": frame.get("cell_probabilities") or {},
                "bird_count_level": frame.get("count_level"),
                "bird_count_label": _count_label(frame["count_level"]) if frame.get("count_level") is not None else None,
                "bird_count_score": frame.get("count_score"),
                "bird_count_probabilities": frame.get("count_probabilities") or {},
            },
        })
        next_ann_id += 1

    models = sorted({f.get("model") for f in frames if f.get("model")})
    if model_hint:
        models.append(model_hint)
    input_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()

    coco = {
        "info": {
            "description": (
                "System One decisions mapped to COCO. Each bbox is the predicted "
                "3x3 grid cell (third of the frame) for the largest bird, not a "
                "tight box. Empty annotation lists mean 'no bird predicted'."
            ),
            "url": "https://github.com/ncanna/freeman-bird-detection",
            "version": "1.0",
            "year": datetime.now(timezone.utc).year,
            "contributor": "system_one_detector.tools.coco_from_decisions",
            "date_created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "licenses": [],
        "categories": [{"id": 1, "name": category_name, "supercategory": "animal"}],
        "images": images,
        "annotations": annotations,
        "source": {
            "input_json": str(input_path),
            "input_sha256": input_sha,
            "input_format": source,
            "models": models,
            "noul_threshold": noul_threshold,
            "n_frames_input": len(frames),
            "n_images_with_size": len(images),
            "n_frames_missing_size": n_without_size,
        },
    }
    return coco


def write_summary_csv(coco: dict[str, Any], frames: list[dict[str, Any]], path: Path) -> None:
    """One row per frame for quick eyeballing, regardless of COCO output."""
    anns_by_image: dict[int, int] = {}
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]] = anns_by_image.get(ann["image_id"], 0) + 1
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "file_name", "video_stem", "noul_probability", "predicted_count_level",
            "predicted_count_label", "predicted_cell", "cell_confidence",
            "n_coco_annotations", "error",
        ])
        by_name = {img["file_name"]: img for img in coco["images"]}
        for frame in frames:
            image = by_name.get(frame["file_name"])
            writer.writerow([
                frame["file_name"],
                frame["video_stem"],
                frame.get("noul"),
                frame.get("count_level"),
                _count_label(frame["count_level"]) if frame.get("count_level") is not None else "",
                frame.get("cell"),
                frame.get("cell_confidence"),
                anns_by_image.get(image["id"], 0) if image else "",
                frame.get("error") or "",
            ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True, help="engine decisions.json or POC results JSON")
    parser.add_argument("--output", type=Path, default=None, help="default: coco_annotations.json next to --input")
    parser.add_argument("--summary-csv", type=Path, default=None, help="default: per_frame_summary.csv next to --output")
    parser.add_argument("--frames-dir", type=Path, default=None, help="fallback image dir when records lack media_path")
    parser.add_argument("--noul-threshold", type=float, default=0.5)
    parser.add_argument("--category-name", default="bird")
    parser.add_argument("--model", default=None, help="override/annotate the model name in source metadata")
    args = parser.parse_args(argv)

    payload = json.loads(args.input.read_text())
    source = detect_source(payload)
    parsers = {"engine": parse_engine_frame, "poc": parse_poc_frame}
    frames = [parsers[source](row) for row in payload["frames"]]

    if args.frames_dir:
        for frame in frames:
            if not frame.get("media_path"):
                frame["media_path"] = str(args.frames_dir / frame["file_name"])

    coco = build_coco(
        frames,
        source=source,
        input_path=args.input,
        noul_threshold=args.noul_threshold,
        category_name=args.category_name,
        model_hint=args.model,
    )

    output = args.output or args.input.parent / "coco_annotations.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(coco, indent=2))

    summary = args.summary_csv or output.parent / "per_frame_summary.csv"
    write_summary_csv(coco, frames, summary)

    meta = coco["source"]
    errors = sum(1 for f in frames if f.get("error"))
    print(
        f"{source} input: {meta['n_frames_input']} frame(s) -> "
        f"{len(coco['images'])} image entries, {len(coco['annotations'])} annotation(s) "
        f"at noul >= {args.noul_threshold}"
    )
    if errors:
        print(f"{errors} frame(s) had decision errors and carry no answers")
    if meta["n_frames_missing_size"]:
        print(f"{meta['n_frames_missing_size']} frame(s) skipped: image size unknown (pass --frames-dir)")
    print(f"Wrote {output}")
    print(f"Wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
