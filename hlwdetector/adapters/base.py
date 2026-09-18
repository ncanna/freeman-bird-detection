"""Base model adapter ABC and shared result types.

The result types themselves live in :mod:`hlwdetector.results` so that modules
below the adapter layer can build them; they are re-exported here because that is
where every adapter already imports them from.
"""

from __future__ import annotations

import logging
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml

from hlwdetector import paths
from hlwdetector.results import DetectionResult, MetricsDict, TrainingResult

# Put the repo root on sys.path before importing utilities, which lives beside the
# package rather than inside it. Done here because base.py is imported first by every
# adapter module, so one insert covers them all.
_PROJECT_ROOT = str(paths.REPO_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utilities.annotation_converter import AnnotationConverter  # noqa: E402

if TYPE_CHECKING:
    from hlwdetector.config.experiment_config import ExperimentConfig
    from hlwdetector.dataset_manager import DatasetManager
    from hlwdetector.artifact_manager import ArtifactManager
    from hlwdetector.tracker import ExperimentTracker

logger = logging.getLogger(__name__)

__all__ = [
    "BaseModelAdapter",
    "DetectionResult",
    "MetricsDict",
    "TrainingResult",
    "TORCH_EPOCH_METRIC_KEYS",
    "ULTRALYTICS_EPOCH_METRIC_KEYS",
    "build_ultralytics_dataset",
    "resolve_device",
]

# MetricsDict field name -> the per-epoch metric key that framework reports.
# Assigned to an adapter's EPOCH_METRIC_KEYS so HPOptimizer can pull the value it
# is optimizing out of whatever dict the adapter hands to report_epoch_to_hpo().

# Ultralytics' on_fit_epoch_end keys (yolo, rtdetr). "f1" has no per-epoch key of
# its own; epoch_metric_value() derives it from precision/recall.
ULTRALYTICS_EPOCH_METRIC_KEYS = {
    "precision": "metrics/precision(B)",
    "recall":    "metrics/recall(B)",
    "map50":     "metrics/mAP50(B)",
    "map50_95":  "metrics/mAP50-95(B)",
}

# The PyTorch-native adapters (swin, detr) run their own validation pass and log
# it under the same keys ExperimentRunner.evaluate() uses — keep the two in sync.
TORCH_EPOCH_METRIC_KEYS = {
    "precision": "val/precision",
    "recall":    "val/recall",
    "f1":        "val/f1",
    "map50":     "val/mAP50",
    "map50_95":  "val/mAP50_95",
}


def resolve_device(device_str: str | None) -> torch.device:
    """Parse a config device string into a torch.device.

    Shared by the PyTorch-native adapters (swin, detr); the Ultralytics-backed
    ones pass the string straight through to the framework instead.
    """
    if device_str is None or device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    # Handle "0", "1" → "cuda:0", "cuda:1"
    if device_str.isdigit():
        return torch.device(f"cuda:{device_str}")
    return torch.device(device_str)


def _expected_label_path(image_path: str) -> Path:
    """Where Ultralytics will look for one image's label file.

    Mirrors ultralytics.data.utils.img2label_paths: the *last* ``/images/`` path
    segment is swapped for ``/labels/`` and the extension becomes ``.txt``.
    Restated here rather than imported so the guard below stays honest — if the
    framework ever changes the rule, the check fails loudly instead of silently
    agreeing with whatever Ultralytics now does.
    """
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return Path(sb.join(image_path.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt")


def build_ultralytics_dataset(
    dataset_manager: "DatasetManager",
    config: "ExperimentConfig",
    work_dir: "str | Path",
    yaml_name: str,
) -> str:
    """Build a self-contained Ultralytics dataset root under work_dir; return the yaml path.

    Shared by the two Ultralytics-backed adapters (yolo, rtdetr), which otherwise
    held identical copies of this code.

    Ultralytics is never told where labels live: it derives each label path from the
    image path by swapping the ``/images/`` segment for ``/labels/`` (see
    _expected_label_path). The convention is therefore a property of the paths handed
    to it, so this builds a layout that satisfies it no matter how ``data/`` is
    arranged:

        work/
          images -> <config.images_dir>   symlink, so 27k frames cost one inode
          labels/                         generated .txt files
          train.txt val.txt test.txt      lines: <work>/images/<file_name>
          <yaml_name>                     path: <work>

    The symlink is load-bearing. Listing frames at their real
    ``data/frames/<dataset>/`` location leaves no ``/images/`` segment to swap, so
    every derived label path misses; Ultralytics then only warns and trains on images
    it believes are empty. _verify_label_discovery turns that into an exception.
    """
    work_path = Path(work_dir)
    images_target = paths.resolve(config.images_dir)

    # Refresh rather than assume: prepare_data may run again over an existing work dir.
    images_link = work_path / "images"
    if images_link.is_symlink():
        images_link.unlink()
    elif images_link.exists():
        raise FileExistsError(
            f"{images_link} exists and is not a symlink; refusing to replace it."
        )
    images_link.symlink_to(images_target, target_is_directory=True)

    labels_dir = work_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    converter = AnnotationConverter(class_mapping={"bird": 0})
    split_image_paths: dict[str, list[str]] = {}

    for split_name in ("train", "val", "test"):
        split_view = dataset_manager.get_split(split_name)

        # All splits share one flat labels dir; video_filter keeps each pass to its own frames.
        converter.coco_to_yolo(
            coco_json_path=split_view.coco_json_path,
            output_dir=str(labels_dir),
            use_filename=True,
            video_filter=split_view.video_stems,
        )

        image_paths = split_view.image_paths
        missing = [p for p in image_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Split '{split_name}': {len(missing)}/{len(image_paths)} image files are missing "
                f"from {images_target}. Extract frames first using extract_frames_from_dir(). "
                f"First missing: {missing[0]}"
            )

        # Route through the symlink so the /images/ -> /labels/ swap lands in work/labels.
        linked_paths = [str(images_link / p.name) for p in image_paths]
        split_image_paths[split_name] = linked_paths
        (work_path / f"{split_name}.txt").write_text("\n".join(linked_paths) + "\n")

    yaml_data = {
        "path": str(work_path),
        "train": str(work_path / "train.txt"),
        "val": str(work_path / "val.txt"),
        "test": str(work_path / "test.txt"),
        "nc": 1,
        "names": {0: "bird"},
    }
    yaml_path = work_path / yaml_name
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    _verify_label_discovery(dataset_manager, split_image_paths)
    return str(yaml_path)


def _verify_label_discovery(
    dataset_manager: "DatasetManager",
    split_image_paths: dict[str, list[str]],
) -> None:
    """Fail loudly if Ultralytics would find no labels for an annotated split.

    Missing labels are only a warning inside Ultralytics ("training may not work
    correctly"), so a broken path convention otherwise yields a full run that scores
    ~0 mAP with nothing in the log to explain it. Checking costs one pass over paths
    already in memory.
    """
    for split_name, image_paths in split_image_paths.items():
        split_view = dataset_manager.get_split(split_name)
        if not split_view.annotations or not image_paths:
            continue  # legitimately empty split — nothing to discover

        found = sum(1 for p in image_paths if _expected_label_path(p).exists())
        if found == 0:
            raise RuntimeError(
                f"Split '{split_name}' has {len(split_view.annotations)} annotations but "
                f"Ultralytics would find 0 label files. It derives label paths from image paths "
                f"by swapping the last '/images/' segment for '/labels/'; for {image_paths[0]} "
                f"that gives {_expected_label_path(image_paths[0])}, which does not exist. "
                f"The work/images symlink or the labels dir is not where it is expected."
            )
        logger.info(
            "Split '%s': %d/%d images have discoverable labels",
            split_name,
            found,
            len(image_paths),
        )


class BaseModelAdapter(ABC):
    """Abstract base for all model adapters."""

    # Set True once an adapter reports per-epoch metrics via report_epoch_to_hpo();
    # HPOConfig.validate() rejects a pruner configured against an adapter that does not.
    supports_pruning: bool = False

    # MetricsDict field name -> this framework's per-epoch metric key. See the
    # ULTRALYTICS_/TORCH_EPOCH_METRIC_KEYS constants above.
    EPOCH_METRIC_KEYS: dict[str, str] = {}

    def __init__(
        self,
        artifact_manager: "ArtifactManager",
        tracker: "ExperimentTracker",
    ) -> None:
        self.experiment_dir = artifact_manager.experiment_dir
        self.work_dir = artifact_manager.work_dir
        self._tracker = tracker
        # Optional (epoch, metrics) -> None hook, assigned by HPOptimizer when a
        # study is running. Read only through report_epoch_to_hpo().
        self._hpo_pruning_callback = None

    def log_epoch(self, epoch: int, metrics: dict) -> None:
        """Log per-epoch metrics. Call from framework-specific callbacks in subclasses."""
        if self._tracker is not None:
            self._tracker.log(metrics, step=epoch)

    def report_epoch_to_hpo(self, epoch: int, metrics: dict) -> None:
        """Hand one epoch's metrics to the HPO pruning hook, if a study set one.

        Call at the end of every epoch, after checkpoints are written, so a pruned
        trial still leaves weights on disk. `metrics` is the adapter's own per-epoch
        dict; the keys it must contain are declared in EPOCH_METRIC_KEYS.

        Raises:
            optuna.TrialPruned: when Optuna decides the trial should stop. It is
                meant to propagate out of train() to HPOptimizer._objective.
        """
        if self._hpo_pruning_callback is not None:
            self._hpo_pruning_callback(epoch, metrics)

    @classmethod
    def epoch_metric_value(cls, metric: str, metrics: dict) -> float | None:
        """Pull one MetricsDict field out of a per-epoch metrics dict.

        Frameworks name their per-epoch metrics differently, so the translation
        lives on the adapter that produces them rather than in HPOptimizer.
        "f1" is derived from precision/recall when EPOCH_METRIC_KEYS has no direct
        key for it (the Ultralytics case). Returns None if unavailable.
        """
        key = cls.EPOCH_METRIC_KEYS.get(metric)
        if key is not None:
            return metrics.get(key)
        if metric != "f1":
            return None
        precision = metrics.get(cls.EPOCH_METRIC_KEYS.get("precision", ""))
        recall = metrics.get(cls.EPOCH_METRIC_KEYS.get("recall", ""))
        if precision is None or recall is None or (precision + recall) == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    @abstractmethod
    def prepare_data(
        self,
        dataset_manager: "DatasetManager",
        config: "ExperimentConfig",
    ) -> None:
        """Convert raw dataset into model-native format under work_dir."""
        ...

    @abstractmethod
    def train(self, config: "ExperimentConfig") -> TrainingResult:
        """Train (or load pretrained) model; return paths + metrics."""
        ...

    @abstractmethod
    def evaluate(self, config: "ExperimentConfig") -> MetricsDict:
        """Evaluate on test split and return standardized metrics."""
        ...

    @abstractmethod
    def predict(self, config: "ExperimentConfig") -> DetectionResult:
        """Run inference on test split; return per-frame sv.Detections."""
        ...
