"""ConfusionMatrixVisualizer — frame-level binary confusion matrix with frame sampling."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv

from hlwdetector.visualization.video_annotator import VideoAnnotator

if TYPE_CHECKING:
    from hlwdetector.adapters.base import DetectionResult
    from hlwdetector.artifact_manager import ArtifactManager
    from hlwdetector.config import ExperimentConfig
    from hlwdetector.dataset_manager import DatasetManager, SplitView

logger = logging.getLogger(__name__)

_VALID_CATEGORIES = frozenset({"tp", "fp", "tn", "fn"})


@dataclass
class ConfusionMatrixResult:
    """Frame stems classified into the four confusion matrix categories."""

    tp: list[str] = field(default_factory=list)
    fp: list[str] = field(default_factory=list)
    tn: list[str] = field(default_factory=list)
    fn: list[str] = field(default_factory=list)
    split: str = "test"
    confidence_threshold: float = 0.25

    @property
    def counts(self) -> dict[str, int]:
        return {"tp": len(self.tp), "fp": len(self.fp), "tn": len(self.tn), "fn": len(self.fn)}

    def __repr__(self) -> str:
        c = self.counts
        total = sum(c.values())
        return (
            f"ConfusionMatrixResult(split={self.split!r}, "
            f"threshold={self.confidence_threshold}, "
            f"TP={c['tp']}, FP={c['fp']}, TN={c['tn']}, FN={c['fn']}, total={total})"
        )


class ConfusionMatrixVisualizer:
    """Frame-level binary confusion matrix and per-category frame sampling.

    A frame is positive if it has any GT annotation (bird present), negative
    otherwise. A frame is predicted-positive if any detection exceeds the
    confidence threshold.
    """

    def __init__(
        self,
        config: "ExperimentConfig",
        artifact_manager: "ArtifactManager",
        dataset_manager: "DatasetManager",
    ) -> None:
        self._config = config
        self._artifact_manager = artifact_manager
        self._dataset_manager = dataset_manager

    def compute(
        self,
        detections: "DetectionResult",
        split: str = "test",
        confidence_threshold: float = 0.25,
    ) -> ConfusionMatrixResult:
        """Classify each frame in the split into TP / FP / TN / FN."""
        split_view = self._dataset_manager.get_split(split)
        id_to_stem = {img["id"]: Path(img["file_name"]).stem for img in split_view.images}

        stems_with_gt: set[str] = set()
        for ann in split_view.annotations:
            stem = id_to_stem.get(ann["image_id"])
            if stem is not None:
                stems_with_gt.add(stem)

        result = ConfusionMatrixResult(split=split, confidence_threshold=confidence_threshold)
        for stem in id_to_stem.values():
            has_gt = stem in stems_with_gt
            has_pred = self._has_prediction(detections, stem, confidence_threshold)
            if has_gt and has_pred:
                result.tp.append(stem)
            elif not has_gt and has_pred:
                result.fp.append(stem)
            elif not has_gt and not has_pred:
                result.tn.append(stem)
            else:
                result.fn.append(stem)

        logger.info("Confusion matrix: %s", result)
        return result

    def plot(
        self,
        result: ConfusionMatrixResult,
        output_path: str | None = None,
    ) -> None:
        """Render a 2×2 confusion matrix heatmap.

        Args:
            result: Output of compute().
            output_path: If given, saves PNG to this path; otherwise calls plt.show().
        """
        counts = np.array([
            [len(result.tn), len(result.fp)],
            [len(result.fn), len(result.tp)],
        ])
        total = int(counts.sum())
        pcts = counts / total * 100 if total > 0 else np.zeros_like(counts, dtype=float)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(counts, cmap="Blues")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted\nNo Bird", "Predicted\nBird"], fontsize=11)
        ax.set_yticklabels(["Actual\nNo Bird", "Actual\nBird"], fontsize=11)
        ax.set_title(
            f"Confusion Matrix — {result.split} split  "
            f"(confidence ≥ {result.confidence_threshold})",
            fontsize=12,
            pad=12,
        )

        cell_labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                text_color = "white" if counts[i, j] > counts.max() * 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{cell_labels[i][j]}\n{counts[i, j]}\n({pcts[i, j]:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=text_color,
                )

        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Confusion matrix plot saved to %s", output_path)
            plt.close(fig)
        else:
            plt.show()

    def sample_frames(
        self,
        result: ConfusionMatrixResult,
        category: str,
        n: int = 16,
        output_dir: str | None = None,
        detections: "DetectionResult | None" = None,
    ) -> list[Path]:
        """Sample frames from a confusion matrix category and save annotated PNGs.

        Args:
            result: Output of compute().
            category: One of 'tp', 'fp', 'tn', 'fn' (case-insensitive).
            n: Maximum number of frames to sample.
            output_dir: Root save directory; a {category}/ subdir is created inside.
                        Defaults to visualizations_dir/confusion_samples/.
            detections: Full DetectionResult for overlaying predictions. Pass None
                        to render GT annotations only.

        Returns:
            List of saved PNG file paths.
        """
        cat = category.lower()
        if cat not in _VALID_CATEGORIES:
            raise ValueError(
                f"category must be one of {sorted(_VALID_CATEGORIES)}, got {cat!r}"
            )

        stems: list[str] = getattr(result, cat)
        if not stems:
            logger.warning("Category %r has no frames — nothing to sample.", cat)
            return []

        sampled = random.sample(stems, min(n, len(stems)))

        if output_dir is None:
            save_dir = self._artifact_manager.visualizations_dir / "confusion_samples" / cat
        else:
            save_dir = Path(output_dir) / cat
        save_dir.mkdir(parents=True, exist_ok=True)

        split_view = self._dataset_manager.get_split(result.split)
        class_names = [c["name"] for c in split_view.categories]
        gt_detections = self._coco_to_sv_detections(split_view)
        images_dir = Path(self._config.images_dir)

        annotator = VideoAnnotator(
            images_dir=str(images_dir),
            gt_detections=gt_detections,
            predictions=detections,
            class_names=class_names,
        )

        saved: list[Path] = []
        for stem in sampled:
            frame_path = self._find_frame(images_dir, stem)
            if frame_path is None:
                logger.warning("Frame not found for stem %r — skipping.", stem)
                continue
            frame = annotator.annotate_single_frame(str(frame_path))
            out_path = save_dir / f"{stem}.png"
            cv2.imwrite(str(out_path), frame)
            saved.append(out_path)

        logger.info(
            "Saved %d/%d %s samples to %s",
            len(saved), len(sampled), cat.upper(), save_dir,
        )
        return saved

    @staticmethod
    def _has_prediction(
        detections: "DetectionResult",
        stem: str,
        threshold: float,
    ) -> bool:
        dets = detections.get(stem)
        if dets is None:
            return False
        if dets.confidence is not None and len(dets.confidence) > 0:
            return bool(np.any(dets.confidence >= threshold))
        return len(dets) > 0

    @staticmethod
    def _find_frame(images_dir: Path, stem: str) -> Path | None:
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _coco_to_sv_detections(split_view: "SplitView") -> dict[str, sv.Detections]:
        """Convert COCO [x,y,w,h] annotations to sv.Detections keyed by frame stem."""
        id_to_stem = {img["id"]: Path(img["file_name"]).stem for img in split_view.images}
        sorted_cats = sorted(split_view.categories, key=lambda c: c["id"])
        cat_id_to_idx = {c["id"]: i for i, c in enumerate(sorted_cats)}

        stem_boxes: dict[str, list] = {}
        stem_cls: dict[str, list] = {}
        for ann in split_view.annotations:
            stem = id_to_stem.get(ann["image_id"])
            if stem is None:
                continue
            x, y, w, h = ann["bbox"]
            stem_boxes.setdefault(stem, []).append([x, y, x + w, y + h])
            stem_cls.setdefault(stem, []).append(cat_id_to_idx.get(ann["category_id"], 0))

        result: dict[str, sv.Detections] = {}
        for stem in id_to_stem.values():
            boxes = stem_boxes.get(stem, [])
            cls = stem_cls.get(stem, [])
            if boxes:
                result[stem] = sv.Detections(
                    xyxy=np.array(boxes, dtype=np.float32),
                    class_id=np.array(cls, dtype=int),
                )
            else:
                result[stem] = sv.Detections.empty()

        return result
