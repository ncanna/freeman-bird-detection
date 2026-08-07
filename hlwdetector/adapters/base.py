"""Base model adapter ABC and shared result types.

The result types themselves live in :mod:`hlwdetector.results` so that modules
below the adapter layer can build them; they are re-exported here because that is
where every adapter already imports them from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from hlwdetector.results import DetectionResult, MetricsDict, TrainingResult

if TYPE_CHECKING:
    from hlwdetector.config.experiment_config import ExperimentConfig
    from hlwdetector.dataset_manager import DatasetManager
    from hlwdetector.artifact_manager import ArtifactManager
    from hlwdetector.tracker import ExperimentTracker

__all__ = [
    "BaseModelAdapter",
    "DetectionResult",
    "MetricsDict",
    "TrainingResult",
    "resolve_device",
]


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
