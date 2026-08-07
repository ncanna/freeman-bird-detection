"""Shared result types returned by adapters.

These live below the adapter layer so that modules the adapters depend on (e.g.
``hlwdetector.metrics``) can produce them without importing the adapters package
back — importing ``hlwdetector.adapters.base`` runs ``adapters/__init__.py``,
which eagerly imports every adapter, so a shared type defined there would form an
import cycle.

``hlwdetector.adapters.base`` re-exports all three names, so
``from hlwdetector.adapters.base import MetricsDict`` keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import supervision as sv

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
