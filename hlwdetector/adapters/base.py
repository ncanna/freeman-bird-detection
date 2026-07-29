"""Base model adapter ABC and shared result types."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import supervision as sv

if TYPE_CHECKING:
    from hlwdetector.config.experiment_config import ExperimentConfig
    from hlwdetector.dataset_manager import DatasetManager
    from hlwdetector.artifact_manager import ArtifactManager
    from hlwdetector.tracker import ExperimentTracker

# Mapping from frame stem → sv.Detections
DetectionResult = dict[str, sv.Detections]


@dataclass
class TrainingResult:
    run_dir: str
    best_weights_path: str | None
    last_weights_path: str | None
    training_metrics: dict


@dataclass
class MetricsDict:
    precision: float
    recall: float
    f1: float
    map50: float
    map50_95: float
    accuracy: float | None = None  # not all models report it
    raw: dict = field(default_factory=dict)


class BaseModelAdapter(ABC):
    """Abstract base for all model adapters."""
    def __init__(
        self,
        artifact_manager: "ArtifactManager",
        tracker: "ExperimentTracker",
    ) -> None:
        self.experiment_dir = artifact_manager.experiment_dir
        self.work_dir = artifact_manager.work_dir
        self._tracker = tracker

    def _prepare_ultralytics_dataset_root(
        self,
        source_images_dir: str | Path,
    ) -> tuple[Path, Path, Path]:
        """Create run-local image/label paths without copying source images.

        Ultralytics discovers labels by replacing the final ``/images/`` path
        component with ``/labels/``. A per-run image-directory symlink keeps
        that convention while isolating generated labels and ``labels.cache``
        from other SLURM jobs that may start concurrently.
        """
        source_images_dir = Path(source_images_dir).resolve()
        if not source_images_dir.is_dir():
            raise FileNotFoundError(f"images_dir not found: {source_images_dir}")

        dataset_root = Path(self.work_dir) / "dataset"
        linked_images_dir = dataset_root / "images"
        labels_dir = dataset_root / "labels"
        dataset_root.mkdir(parents=True, exist_ok=True)

        if linked_images_dir.is_symlink():
            if linked_images_dir.resolve() != source_images_dir:
                linked_images_dir.unlink()
        elif linked_images_dir.exists():
            raise FileExistsError(
                f"Expected a dataset image symlink, found a real path: {linked_images_dir}"
            )

        if not linked_images_dir.is_symlink():
            relative_target = os.path.relpath(source_images_dir, start=dataset_root)
            linked_images_dir.symlink_to(relative_target, target_is_directory=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        return dataset_root, linked_images_dir, labels_dir

    def log_epoch(self, epoch: int, metrics: dict) -> None:
        """Log per-epoch metrics. Call from framework-specific callbacks in subclasses."""
        if self._tracker is not None:
            self._tracker.log(metrics, step=epoch)

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
