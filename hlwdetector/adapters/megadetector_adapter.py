"""MegaDetectorAdapter — migrated from models/megadetector/megadetector.ipynb."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import supervision as sv

from hlwdetector.adapters.base import (
    BaseModelAdapter,
    DetectionResult,
    MetricsDict,
    TrainingResult,
)
from hlwdetector.registry import register_adapter

if TYPE_CHECKING:
    from hlwdetector.config import ExperimentConfig
    from hlwdetector.dataset_manager import DatasetManager, SplitView

logger = logging.getLogger(__name__)


@register_adapter("megadetector")
class MegaDetectorAdapter(BaseModelAdapter):
    """MegaDetector V6 (pretrained) adapter via PytorchWildlife.

    Note: animal detections are treated as bird detections until a species
    classifier step is added (see TODO comment in predict()).
    """

    def __init__(self, artifact_manager, tracker) -> None:
        super().__init__(artifact_manager, tracker)
        self._detection_model = None
        self._split_views: dict[str, "SplitView"] = {}
        self._cached_predictions: DetectionResult | None = None

    # ------------------------------------------------------------------ #
    # prepare_data
    # ------------------------------------------------------------------ #

    def prepare_data(
        self,
        dataset_manager: "DatasetManager",
        config: "ExperimentConfig",
        work_dir: str,
    ) -> None:
        """Validate that the images directory exists and store split views."""
        images_dir = Path(config.images_dir)
        if not images_dir.exists():
            raise FileNotFoundError(f"images_dir not found: {images_dir}")

        for split_name in ("train", "val", "test"):
            self._split_views[split_name] = dataset_manager.get_split(split_name)

    # ------------------------------------------------------------------ #
    # train (stub — MegaDetector is pretrained)
    # ------------------------------------------------------------------ #

    def train(self, config: "ExperimentConfig") -> TrainingResult:
        """Load pretrained MDV6; no fine-tuning performed."""
        import torch
        from PytorchWildlife.models import detection as pw_detection  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        version = config.hyperparameters.get("version", "MDV6-yolov10-e")

        cache_dir = Path(config.output_dir) / "megadetector_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading MegaDetector %s on %s ...", version, device)
        self._detection_model = pw_detection.MegaDetectorV6(
            device=device, pretrained=True, version=version
        )
        logger.info("MegaDetector loaded.")

        cache_path = str(cache_dir / f"megadetector_{version}.marker")
        Path(cache_path).write_text(f"version={version}\ndevice={device}\n")

        return TrainingResult(
            run_dir=str(cache_dir),
            best_weights_path=cache_path,
            last_weights_path=None,
            training_metrics={},
        )

    # ------------------------------------------------------------------ #
    # evaluate
    # ------------------------------------------------------------------ #

    def evaluate(self, config: "ExperimentConfig") -> MetricsDict:
        """Evaluate on test split using pycocotools COCOeval."""
        from pycocotools.coco import COCO  # type: ignore
        from pycocotools.cocoeval import COCOeval  # type: ignore

        test_view = self._split_views.get("test")
        if test_view is None:
            raise RuntimeError("Call prepare_data() before evaluate().")

        if self._cached_predictions is None:
            self._cached_predictions = self.predict(config)

        # Build COCO GT object from SplitView annotations
        import json
        import tempfile

        categories = test_view.categories
        # Map category names to id=1 (bird) for evaluation
        gt_coco_dict = {
            "images": test_view.images,
            "annotations": test_view.annotations,
            "categories": categories,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(gt_coco_dict, tf)
            gt_path = tf.name

        coco_gt = COCO(gt_path)

        # Build predictions in COCO result format
        # MD maps animals → treat as bird (category_id=1, the only category)
        bird_cat_id = categories[0]["id"] if categories else 1

        image_stem_to_id = {
            Path(img["file_name"]).stem: img["id"]
            for img in test_view.images
        }

        coco_results = []
        for stem, dets in self._cached_predictions.items():
            image_id = image_stem_to_id.get(stem)
            if image_id is None:
                continue
            for i in range(len(dets)):
                x1, y1, x2, y2 = dets.xyxy[i]
                w = float(x2 - x1)
                h = float(y2 - y1)
                conf = float(dets.confidence[i]) if dets.confidence is not None else 1.0
                coco_results.append({
                    "image_id": image_id,
                    "category_id": bird_cat_id,
                    "bbox": [float(x1), float(y1), w, h],
                    "score": conf,
                })

        if not coco_results:
            logger.warning("No predictions — returning zero metrics.")
            return MetricsDict(precision=0.0, recall=0.0, f1=0.0, map50=0.0, map50_95=0.0)

        coco_dt = coco_gt.loadRes(coco_results)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        # stats[0] = mAP@[.5:.95], stats[1] = mAP@.5
        map50_95 = float(evaluator.stats[0])
        map50 = float(evaluator.stats[1])
        precision = float(evaluator.stats[0])  # approximate; COCOeval doesn't give scalar P/R
        recall = float(evaluator.stats[8])     # AR@maxDets=10 as proxy for recall
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        import os
        os.unlink(gt_path)

        return MetricsDict(
            precision=precision,
            recall=recall,
            f1=f1,
            map50=map50,
            map50_95=map50_95,
            raw={
                "coco_stats": list(evaluator.stats),
            },
        )

    # ------------------------------------------------------------------ #
    # predict
    # ------------------------------------------------------------------ #

    def predict(self, config: "ExperimentConfig") -> DetectionResult:
        """Run MDV6 inference on test images; return per-frame sv.Detections."""
        if self._detection_model is None:
            raise RuntimeError("Call train() before predict() to load the model.")

        test_view = self._split_views.get("test")
        if test_view is None:
            raise RuntimeError("Call prepare_data() before predict().")

        import cv2

        image_files = sorted(
            p for p in test_view.image_paths
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        predictions: DetectionResult = {}

        for img_path in image_files:
            frame = cv2.imread(str(img_path))
            if frame is None:
                logger.warning("Could not read frame: %s", img_path)
                continue

            try:
                results_det = self._detection_model.single_image_detection(
                    frame, img_path=str(img_path)
                )
                dets: sv.Detections = results_det["detections"]

                # TODO: add species classifier step here (see megadetector.ipynb)
                # Until the species classifier is added, all "animal" detections are
                # treated as "bird" detections. This is a known approximation — see paper.

                # Remap class_ids to 0 (bird)
                if dets.class_id is not None and len(dets) > 0:
                    dets.class_id = np.zeros(len(dets), dtype=int)

                predictions[img_path.stem] = dets

            except Exception as exc:
                logger.warning("Error running MegaDetector on %s: %s", img_path, exc)
                predictions[img_path.stem] = sv.Detections.empty()

        self._cached_predictions = predictions
        return predictions
