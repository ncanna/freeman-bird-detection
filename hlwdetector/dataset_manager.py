"""DatasetManager and SplitView — loads COCO JSONs filtered by split.json."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hlwdetector.config import ExperimentConfig

logger = logging.getLogger(__name__)


@dataclass
class SplitView:
    split: str
    coco_json_path: str
    video_stems: list[str]       # from split.json
    images_dir: str              # directory containing extracted frames from all videos
    images: list[dict]           # COCO image dicts filtered to this split
    annotations: list[dict]      # COCO annotation dicts for filtered images
    categories: list[dict]

    @property
    def image_paths(self) -> list[Path]:
        """Absolute paths to the image files belonging to this split."""
        base = Path(self.images_dir)
        return [base / img["file_name"] for img in self.images]


class DatasetManager:
    """Loads COCO annotation files and produces per-split SplitView objects."""

    def __init__(self, config: "ExperimentConfig") -> None:
        self._config = config
        self._split_views: dict[str, SplitView] = {}

        # Discover split.json
        split_json_path = self._resolve_split_json()

        with open(split_json_path, "r") as f:
            split_data: dict[str, list[str]] = json.load(f)

        images_base = Path(config.images_dir)

        if not images_base.exists():
            raise FileNotFoundError(f"images_dir not found: {images_base}")

        for split_name in ("train", "val", "test"):
            video_stems: list[str] = split_data.get(split_name, [])

            images, annotations, categories = self._load_coco_split(
                self._config.coco_json, video_stems
            )

            self._split_views[split_name] = SplitView(
                split=split_name,
                coco_json_path=self._config.coco_json,
                video_stems=video_stems,
                images_dir=str(images_base),
                images=images,
                annotations=annotations,
                categories=categories,
            )

            logger.info(
                "Split '%s': %d images, %d annotations from %d video(s)",
                split_name,
                len(images),
                len(annotations),
                len(video_stems),
            )

    def _resolve_split_json(self) -> str:
        """Return path to split.json, auto-discovering if not set in config."""
        if self._config.split_json is not None:
            p = Path(self._config.split_json)
            if not p.exists():
                raise FileNotFoundError(f"split_json not found: {p}")
            return str(p)

        # Auto-discover: look in same directory as coco_train
        candidate = Path(self._config.coco_train).parent / "split.json"
        if candidate.exists():
            logger.debug("Auto-discovered split.json at %s", candidate)
            return str(candidate)

        raise FileNotFoundError(
            f"split.json not found. Expected at {candidate} or set split_json in config."
        )

    def _load_coco_split(
        self,
        coco_json_path: str,
        video_stems: list[str],
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Load COCO JSON and filter images/annotations to those matching video_stems."""
        with open(coco_json_path, "r") as f:
            coco_data = json.load(f)

        categories = coco_data.get("categories", [])

        if not video_stems:
            # Empty split — return nothing
            return [], [], categories

        stems_set = set(video_stems)

        # Filter images: keep those whose file_name stem starts with a listed video stem
        filtered_images = [
            img for img in coco_data["images"]
            if any(img["file_name"].startswith(s) for s in stems_set)
        ]
        filtered_image_ids = {img["id"] for img in filtered_images}

        # Filter annotations to matching images
        filtered_annotations = [
            ann for ann in coco_data.get("annotations", [])
            if ann["image_id"] in filtered_image_ids
        ]

        return filtered_images, filtered_annotations, categories

    def get_split(self, split: str) -> SplitView:
        """Return SplitView for the given split name (train/val/test)."""
        if split not in self._split_views:
            raise KeyError(f"Unknown split: {split!r}. Available: {list(self._split_views)}")
        return self._split_views[split]
