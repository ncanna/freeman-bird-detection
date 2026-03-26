"""VisualizationPipeline — orchestrates VideoAnnotator using framework data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import supervision as sv

from hlwdetector.visualization.annotator import VideoAnnotator

if TYPE_CHECKING:
    from hlwdetector.adapters.base import DetectionResult
    from hlwdetector.artifact_manager import ArtifactManager
    from hlwdetector.config import ExperimentConfig
    from hlwdetector.dataset_manager import DatasetManager, SplitView

logger = logging.getLogger(__name__)


class VisualizationPipeline:
    """Convert COCO GT annotations → sv.Detections and delegate to VideoAnnotator."""

    def __init__(
        self,
        config: "ExperimentConfig",
        artifact_manager: "ArtifactManager",
        dataset_manager: "DatasetManager",
    ) -> None:
        self._config = config
        self._artifact_manager = artifact_manager
        self._dataset_manager = dataset_manager

    def run(self, detections: "DetectionResult") -> None:
        """Annotate test split frames with GT + predictions and write video(s)."""
        split_view = self._dataset_manager.get_split(self._config.visualize_split)
        class_names = [c["name"] for c in split_view.categories]

        gt_detections = self._coco_to_sv_detections(split_view)

        annotator = VideoAnnotator(
            images_dir=split_view.images_split_dir,
            gt_detections=gt_detections,
            predictions=detections,
            class_names=class_names,
        )

        output_path = str(self._artifact_manager.visualizations_dir)
        annotator.annotate_video(
            output_path=output_path,
            fps=self._config.visualization_fps,
        )
        logger.info("Visualization written to: %s", output_path)

    def _coco_to_sv_detections(self, split_view: "SplitView") -> dict[str, sv.Detections]:
        """Convert COCO [x,y,w,h] annotations → sv.Detections keyed by frame stem."""
        # Build image_id → stem mapping
        id_to_stem = {
            img["id"]: Path(img["file_name"]).stem
            for img in split_view.images
        }

        # Build category_id → class_index mapping
        sorted_cats = sorted(split_view.categories, key=lambda c: c["id"])
        cat_id_to_idx = {cat["id"]: i for i, cat in enumerate(sorted_cats)}

        # Group annotations by image stem
        stem_to_boxes: dict[str, list[list[float]]] = {}
        stem_to_cls: dict[str, list[int]] = {}

        for ann in split_view.annotations:
            stem = id_to_stem.get(ann["image_id"])
            if stem is None:
                continue
            x, y, w, h = ann["bbox"]
            xyxy = [x, y, x + w, y + h]
            cat_idx = cat_id_to_idx.get(ann["category_id"], 0)
            stem_to_boxes.setdefault(stem, []).append(xyxy)
            stem_to_cls.setdefault(stem, []).append(cat_idx)

        result: dict[str, sv.Detections] = {}
        for stem in id_to_stem.values():
            boxes = stem_to_boxes.get(stem, [])
            cls = stem_to_cls.get(stem, [])
            if boxes:
                result[stem] = sv.Detections(
                    xyxy=np.array(boxes, dtype=np.float32),
                    class_id=np.array(cls, dtype=int),
                )
            else:
                result[stem] = sv.Detections.empty()

        return result
