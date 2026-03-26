"""Base model adapter ABC and shared result types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import supervision as sv

if TYPE_CHECKING:
    from hlwdetector.config import ExperimentConfig
    from hlwdetector.dataset_manager import DatasetManager
    from hlwdetector.artifact_manager import ArtifactManager

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
    def __init__(self, artifact_manager: ArtifactManager) -> None:
        self.experiment_dir = artifact_manager.experiment_dir
        self.work_dir = artifact_manager.work_dir

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
