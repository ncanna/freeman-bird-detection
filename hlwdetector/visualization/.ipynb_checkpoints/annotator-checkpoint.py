"""
Roboflow Supervision-based visualization pipeline for bird detection.

Overlays ground truth and/or predicted bounding boxes on extracted frames
and writes an annotated video. No inference is run here.

Moved from utilities/visualization.py. Minimal modifications:
- Added gt_detections parameter (COCO-derived GT as sv.Detections dict)
- Added class_names parameter (used when data_yaml_path is not given)
"""

import csv
import logging
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import yaml

logger = logging.getLogger(__name__)


def load_gt_detections(
    label_path: str,
    class_names: list[str],
    image_shape: tuple[int, int],
) -> sv.Detections:
    """Load YOLO-format ground truth labels and return sv.Detections in absolute xyxy.

    Args:
        label_path: Path to a YOLO .txt label file.
        class_names: List of class names (index == class_id).
        image_shape: (height, width) of the corresponding image in pixels.

    Returns:
        sv.Detections with absolute pixel coordinates, or sv.Detections.empty()
        if the file is missing or empty.
    """
    img_h, img_w = image_shape
    path = Path(label_path)

    if not path.exists():
        return sv.Detections.empty()

    lines = path.read_text().strip().splitlines()
    if not lines:
        return sv.Detections.empty()

    boxes = []
    class_ids = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id, x_c, y_c, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (x_c - w / 2) * img_w
        x2 = (x_c + w / 2) * img_w
        y1 = (y_c - h / 2) * img_h
        y2 = (y_c + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
        class_ids.append(cls_id)

    if not boxes:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )


class VideoAnnotator:
    """Overlay GT and/or predicted bounding boxes on extracted frames and write a video.

    Args:
        images_dir: Directory containing extracted frame images (jpg/png).
        data_yaml_path: Path to a YOLO data YAML file. Used only to read class names.
                        Optional when gt_detections is provided together with class_names.
        labels_dir: Directory containing YOLO .txt label files (GT mode). Optional.
        gt_detections: Mapping of frame stem -> sv.Detections (COCO-derived GT). Optional.
                       When provided, labels_dir is not needed for GT boxes.
        predictions: Mapping of frame stem -> sv.Detections (pred mode). Optional.
        class_names: List of class names used when data_yaml_path is not given. Optional.
        gt_color: Supervision Color for GT boxes/labels.
        pred_color: Supervision Color for prediction boxes/labels.

    Raises:
        ValueError: If neither labels_dir, gt_detections nor predictions is provided,
                    or images_dir is empty.
        FileNotFoundError: If images_dir or labels_dir does not exist.
    """

    _IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(
        self,
        images_dir: str,
        data_yaml_path: str | None = None,
        labels_dir: str | None = None,
        gt_detections: dict[str, sv.Detections] | None = None,
        predictions: dict[str, sv.Detections] | None = None,
        class_names: list[str] | None = None,
        gt_color: sv.Color = sv.Color.GREEN,
        pred_color: sv.Color = sv.Color.BLUE,
        frame_map_path: str | None = None,
        image_files: list[Path] | None = None,
    ) -> None:
        if labels_dir is None and gt_detections is None and predictions is None:
            raise ValueError(
                "At least one of labels_dir, gt_detections, or predictions must be provided."
            )

        self._images_dir = Path(images_dir)
        if not self._images_dir.exists():
            raise FileNotFoundError(f"images_dir does not exist: {images_dir}")

        self._labels_dir: Path | None = None
        if labels_dir is not None:
            self._labels_dir = Path(labels_dir)
            if not self._labels_dir.exists():
                raise FileNotFoundError(f"labels_dir does not exist: {labels_dir}")

        self._gt_detections = gt_detections
        self._predictions = predictions

        # Determine mode
        has_gt = self._labels_dir is not None or self._gt_detections is not None
        has_pred = self._predictions is not None
        if has_gt and has_pred:
            self._mode = "both"
        elif has_gt:
            self._mode = "gt"
        else:
            self._mode = "pred"

        # Load class names
        if data_yaml_path is not None:
            with open(data_yaml_path, "r") as f:
                data_yaml = yaml.safe_load(f)
            names_raw = data_yaml["names"]
            if isinstance(names_raw, dict):
                self._class_names: list[str] = [names_raw[i] for i in sorted(names_raw)]
            else:
                self._class_names = list(names_raw)
        elif class_names is not None:
            self._class_names = list(class_names)
        else:
            self._class_names = []

        # Collect sorted image files — use provided list if given, otherwise scan directory
        if image_files is not None:
            self._image_files = sorted(
                p for p in image_files
                if p.suffix.lower() in self._IMAGE_EXTENSIONS
            )
        else:
            self._image_files = sorted(
                p for p in self._images_dir.iterdir()
                if p.suffix.lower() in self._IMAGE_EXTENSIONS
            )
        if not self._image_files:
            raise ValueError(f"No image files found in images_dir: {images_dir}")

        # Build per-source-video frame grouping from frame_map.csv if provided
        self._frame_map_loaded = False
        self._video_to_frames: dict[str, list[Path]] = {}
        if frame_map_path is not None:
            frame_stem_to_path = {p.stem: p for p in self._image_files}
            with open(frame_map_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    frame_stem = Path(row["frame"]).stem
                    video_stem = Path(row["video"]).stem
                    img_path = frame_stem_to_path.get(frame_stem)
                    if img_path is not None:
                        self._video_to_frames.setdefault(video_stem, []).append(img_path)
            self._frame_map_loaded = True

        # Build annotators
        self._gt_box_ann = sv.BoundingBoxAnnotator(color=gt_color, thickness=3)
        self._gt_label_ann = sv.LabelAnnotator(color=gt_color)
        self._pred_box_ann = sv.BoundingBoxAnnotator(color=pred_color, thickness=2)
        self._pred_label_ann = sv.LabelAnnotator(color=pred_color)

        self._gt_color = gt_color
        self._pred_color = pred_color

    def annotate_video(self, output_path: str, fps: float) -> None:
        """Write annotated video(s) to output_path.

        When a frame_map_path was provided at construction, output_path is treated as
        an output directory and one ``<video_stem>_annotated.mp4`` is written per
        source video. Otherwise, output_path is a single destination .mp4 file.

        Args:
            output_path: Destination .mp4 file path (single-video mode) or directory
                (per-source-video mode, when frame_map_path was given).
            fps: Frames per second for the output video(s).
        """
        if self._frame_map_loaded:
            self._annotate_video_per_source(output_path, fps)
        else:
            self._annotate_video_single(output_path, fps)

    def _annotate_video_single(self, output_path: str, fps: float) -> None:
        """Original behavior: write all frames into one video at output_path."""
        first_frame = cv2.imread(str(self._image_files[0]))
        if first_frame is None:
            raise RuntimeError(f"Cannot read first frame: {self._image_files[0]}")
        h, w = first_frame.shape[:2]

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        for img_path in self._image_files:
            stem = img_path.stem
            try:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    logger.warning("Could not read frame: %s", img_path)
                    continue

                gt_dets = self._load_gt_for_stem(stem, (h, w))
                pred_dets = self._load_pred_for_stem(stem)
                annotated = self._annotate_frame(frame, gt_dets, pred_dets)
                writer.write(annotated)
            except Exception as exc:
                logger.warning("Error processing frame %s: %s", img_path, exc)
                continue

        writer.release()
        logger.info("Annotated video saved to %s", out_path)

    def _annotate_video_per_source(self, output_dir: str, fps: float) -> None:
        """Write one annotated video per source video into output_dir."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        for video_stem, frames in self._video_to_frames.items():
            first_frame = cv2.imread(str(frames[0]))
            if first_frame is None:
                logger.warning("Cannot read first frame for %s, skipping", video_stem)
                continue
            h, w = first_frame.shape[:2]

            out_path = out_dir / f"{video_stem}_annotated.mp4"
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

            for img_path in frames:
                stem = img_path.stem
                try:
                    frame = cv2.imread(str(img_path))
                    if frame is None:
                        logger.warning("Could not read frame: %s", img_path)
                        continue

                    gt_dets = self._load_gt_for_stem(stem, (h, w))
                    pred_dets = self._load_pred_for_stem(stem)
                    annotated = self._annotate_frame(frame, gt_dets, pred_dets)
                    writer.write(annotated)
                except Exception as exc:
                    logger.warning("Error processing frame %s: %s", img_path, exc)
                    continue

            writer.release()
            logger.info("Annotated video saved to %s", out_path)

    def annotate_single_frame(self, frame_path: str) -> np.ndarray:
        """Load a single frame, overlay annotations, and return the result.

        Args:
            frame_path: Path to the frame image file.

        Returns:
            Annotated frame as a BGR numpy array.
        """
        path = Path(frame_path)
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read frame: {frame_path}")

        h, w = frame.shape[:2]
        stem = path.stem
        gt_dets = self._load_gt_for_stem(stem, (h, w))
        pred_dets = self._load_pred_for_stem(stem)
        return self._annotate_frame(frame, gt_dets, pred_dets)

    def _load_gt_for_stem(self, stem: str, image_shape: tuple[int, int]) -> sv.Detections | None:
        # Prefer in-memory gt_detections (COCO-derived) over disk labels
        if self._gt_detections is not None:
            dets = self._gt_detections.get(stem)
            if dets is None:
                logger.debug("No GT detections for stem '%s'", stem)
                return sv.Detections.empty()
            return dets

        if self._labels_dir is None:
            return None
        label_file = self._labels_dir / f"{stem}.txt"
        if not label_file.exists():
            logger.debug("No GT annotations for stem '%s' (empty frame)", stem)
            return sv.Detections.empty()
        return load_gt_detections(str(label_file), self._class_names, image_shape)

    def _load_pred_for_stem(self, stem: str) -> sv.Detections | None:
        if self._predictions is None:
            return None
        dets = self._predictions.get(stem)
        if dets is None:
            logger.warning("No predictions found for stem '%s'", stem)
            return sv.Detections.empty()
        return dets

    def _build_gt_labels(self, dets: sv.Detections) -> list[str]:
        if dets.class_id is None:
            return []
        return [
            f"GT: {self._class_names[cid] if cid < len(self._class_names) else str(cid)}"
            for cid in dets.class_id
        ]

    def _build_pred_labels(self, dets: sv.Detections) -> list[str]:
        labels = []
        for i, cid in enumerate(dets.class_id if dets.class_id is not None else []):
            name = self._class_names[cid] if cid < len(self._class_names) else str(cid)
            if dets.confidence is not None and i < len(dets.confidence):
                conf = dets.confidence[i]
                labels.append(f"PRED: {name} {conf:.2f}")
            else:
                labels.append(f"PRED: {name}")
        return labels

    def _annotate_frame(
        self,
        frame: np.ndarray,
        gt_dets: sv.Detections | None,
        pred_dets: sv.Detections | None,
    ) -> np.ndarray:
        """Draw GT (green) then pred (blue) boxes on a copy of the frame."""
        out = frame.copy()

        if self._mode in ("gt", "both") and gt_dets is not None and len(gt_dets) > 0:
            out = self._gt_box_ann.annotate(scene=out, detections=gt_dets)
            out = self._gt_label_ann.annotate(
                scene=out,
                detections=gt_dets,
                labels=self._build_gt_labels(gt_dets),
            )

        if self._mode in ("pred", "both") and pred_dets is not None and len(pred_dets) > 0:
            out = self._pred_box_ann.annotate(scene=out, detections=pred_dets)
            out = self._pred_label_ann.annotate(
                scene=out,
                detections=pred_dets,
                labels=self._build_pred_labels(pred_dets),
            )

        if self._mode == "both":
            out = self._draw_legend(out)

        return out

    def _draw_legend(self, frame: np.ndarray) -> np.ndarray:
        """Draw a small legend in the top-left corner showing GT and PRED colors."""
        legend_x, legend_y = 10, 10
        box_size = 16
        row_h = 26
        padding = 8
        legend_w = 140
        legend_h = 2 * row_h + 2 * padding

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (legend_x, legend_y),
            (legend_x + legend_w, legend_y + legend_h),
            (30, 30, 30),
            -1,
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        def _sv_color_to_bgr(color: sv.Color) -> tuple[int, int, int]:
            return (color.b, color.g, color.r)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        row1_y = legend_y + padding
        cv2.rectangle(
            frame,
            (legend_x + padding, row1_y),
            (legend_x + padding + box_size, row1_y + box_size),
            _sv_color_to_bgr(self._gt_color),
            -1,
        )
        cv2.putText(
            frame,
            "GT",
            (legend_x + padding + box_size + 6, row1_y + box_size - 2),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

        row2_y = legend_y + padding + row_h
        cv2.rectangle(
            frame,
            (legend_x + padding, row2_y),
            (legend_x + padding + box_size, row2_y + box_size),
            _sv_color_to_bgr(self._pred_color),
            -1,
        )
        cv2.putText(
            frame,
            "PRED",
            (legend_x + padding + box_size + 6, row2_y + box_size - 2),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

        return frame
