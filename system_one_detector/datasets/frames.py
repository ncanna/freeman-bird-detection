"""Build frame records from the repo's flat images dir, split.json, and COCO JSON."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from system_one_detector.datasets.questions import build_questions, labels_for_frame, text_state
from system_one_detector.util.config import SystemOneConfig

_FRAME_INDEX = re.compile(r"_frame_(\d+)")
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class FrameExample:
    frame_id: str
    file_name: str
    image_path: str | None
    video_stem: str
    split: str
    state: dict[str, Any]
    questions: dict[str, dict]
    labels: dict[str, dict]
    modality: str
    media_path: str | None = None
    width: int | None = None
    height: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "file_name": self.file_name,
            "image_path": self.image_path,
            "video_stem": self.video_stem,
            "split": self.split,
            "modality": self.modality,
            "media_path": self.media_path,
            "labels": self.labels,
        }


def video_stem_for(file_name: str, stems: list[str]) -> str | None:
    """Match the longest split stem that is a filename prefix, bounded by ``_`` or ``.``."""
    matches = []
    for stem in stems:
        if file_name == stem or file_name.startswith(stem + "_") or file_name.startswith(stem + "."):
            matches.append(stem)
    if not matches:
        return None
    return max(matches, key=len)


def frame_index_of(file_name: str) -> int | None:
    match = _FRAME_INDEX.search(file_name)
    return int(match.group(1)) if match else None


def _load_json(path: str) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def _category_names(categories: list[dict]) -> dict[int, str]:
    return {int(cat["id"]): str(cat.get("name") or cat["id"]) for cat in categories if "id" in cat}


def _annotations_by_image(annotations: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for ann in annotations:
        grouped.setdefault(int(ann["image_id"]), []).append(ann)
    return grouped


def _scan_images(images_dir: Path) -> list[dict]:
    records = []
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() in _IMAGE_SUFFIXES and path.is_file():
            records.append({"file_name": path.name, "id": path.name, "width": None, "height": None})
    return records


def _media_path(config: SystemOneConfig, file_name: str, video_stem: str, image_path: Path | None) -> str | None:
    if config.modality == "text":
        return None
    if config.modality == "image":
        return str(image_path) if image_path is not None else None
    if config.videos_dir:
        video_dir = Path(config.videos_dir)
        for suffix in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            candidate = video_dir / f"{video_stem}{suffix}"
            if candidate.exists():
                return str(candidate)
    return str(image_path) if image_path is not None else None


def load_frames(config: SystemOneConfig) -> tuple[list[FrameExample], dict[str, dict]]:
    """Load one split of frames and the question map shared by every frame."""
    stems: list[str] | None = None
    if config.split_json:
        split_path = Path(config.split_json)
        if not split_path.exists():
            raise FileNotFoundError(f"split_json not found: {split_path}")
        split_data = _load_json(str(split_path))
        stems = list(split_data.get(config.split, []))
        if not stems:
            raise ValueError(f"split {config.split!r} is empty in {split_path}")

    categories: list[dict] = []
    ann_by_image: dict[Any, list[dict]] = {}
    if config.coco_json:
        coco_path = Path(config.coco_json)
        if not coco_path.exists():
            raise FileNotFoundError(f"coco_json not found: {coco_path}")
        coco = _load_json(str(coco_path))
        categories = list(coco.get("categories") or [])
        images = list(coco.get("images") or [])
        ann_by_image = _annotations_by_image(list(coco.get("annotations") or []))
    elif config.images_dir:
        images_root = Path(config.images_dir)
        if not images_root.exists():
            raise FileNotFoundError(f"images_dir not found: {images_root}")
        images = _scan_images(images_root)
    else:
        raise ValueError("set coco_json or images_dir")

    questions = build_questions(
        config.question_ids,
        categories=categories,
        extra=config.extra_questions,
    )
    names = _category_names(categories)
    images_root = Path(config.images_dir) if config.images_dir else None
    examples: list[FrameExample] = []

    for image in images:
        file_name = str(image.get("file_name") or "")
        if not file_name:
            continue
        if stems is not None:
            video_stem = video_stem_for(file_name, stems)
            if video_stem is None:
                continue
        else:
            video_stem = file_name.split("_frame_")[0]
        image_path = (images_root / file_name) if images_root is not None else None
        if image_path is not None and not image_path.exists():
            image_path = None
        width = image.get("width")
        height = image.get("height")
        if config.coco_json:
            anns = ann_by_image.get(image.get("id"), [])
            labels = labels_for_frame(
                list(questions),
                anns,
                width=float(width) if width else None,
                height=float(height) if height else None,
                category_names=names if names else None,
            )
        else:
            labels = {}
        media = _media_path(config, file_name, video_stem, image_path)
        examples.append(
            FrameExample(
                frame_id=Path(file_name).stem,
                file_name=file_name,
                image_path=str(image_path) if image_path is not None else None,
                video_stem=video_stem,
                split=config.split,
                state=text_state(file_name, video_stem, frame_index_of(file_name)),
                questions=questions,
                labels=labels,
                modality=config.modality,
                media_path=media,
                width=int(width) if width else None,
                height=int(height) if height else None,
            )
        )

    examples.sort(key=lambda item: (item.video_stem, item.file_name))
    if config.limit is not None:
        examples = examples[: config.limit]
    if not examples:
        raise FileNotFoundError(
            f"no frames for split {config.split!r}. "
            "Check split_json stems against coco file_name prefixes."
        )
    return examples, questions
