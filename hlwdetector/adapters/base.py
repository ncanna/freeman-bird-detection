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
    "TORCH_EPOCH_METRIC_KEYS",
    "ULTRALYTICS_EPOCH_METRIC_KEYS",
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
