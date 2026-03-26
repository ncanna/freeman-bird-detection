"""ArtifactManager — canonical output paths and serialization helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import supervision as sv

if TYPE_CHECKING:
    from hlwdetector.adapters.base import MetricsDict, TrainingResult
    from hlwdetector.config import ExperimentConfig

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Manages all output paths for an experiment run."""

    def __init__(self, config: "ExperimentConfig") -> None:
        if config.resume_from is not None:
            # Resume: use existing directory as-is
            self.experiment_dir = Path(config.resume_from).resolve()
            if not self.experiment_dir.exists():
                raise FileNotFoundError(
                    f"resume_from directory not found: {self.experiment_dir}"
                )
            logger.info("Resuming experiment from: %s", self.experiment_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_dir = (
                Path(config.output_dir) / f"{config.experiment_name}_{timestamp}"
            ).resolve()
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Experiment directory: %s", self.experiment_dir)

        self.work_dir = self.experiment_dir / "work"
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Ensure visualizations subdirectory exists
        self.visualizations_dir = self.experiment_dir / "visualizations"
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #

    def save_config(self, config: "ExperimentConfig", wandb_run_id: str | None = None) -> None:
        """Write config.json (includes wandb_run_id if W&B is active)."""
        import dataclasses
        data = dataclasses.asdict(config)
        if wandb_run_id is not None:
            data["wandb_run_id"] = wandb_run_id
        self._write_json("config.json", data)

    def save_model_info(self, result: "TrainingResult") -> None:
        """Write model.json with best/last weights paths."""
        import dataclasses
        self._write_json("model.json", dataclasses.asdict(result))

    def save_metrics(self, metrics: "MetricsDict") -> None:
        """Write metrics.json with evaluation results."""
        import dataclasses
        self._write_json("metrics.json", dataclasses.asdict(metrics))

    def save_detections(self, detections: dict[str, sv.Detections]) -> None:
        """Serialize detections to detections.json (same schema as YOLODetector.save_detections)."""
        serialisable: dict[str, dict] = {}
        for stem, det in detections.items():
            serialisable[stem] = {
                "xyxy": det.xyxy.tolist() if det.xyxy is not None else [],
                "confidence": det.confidence.tolist() if det.confidence is not None else [],
                "class_id": det.class_id.tolist() if det.class_id is not None else [],
            }
        self._write_json("detections.json", serialisable)
        logger.info("Detections saved (%d frames)", len(detections))

    def load_detections(self) -> dict[str, sv.Detections]:
        """Load detections.json and return per-frame sv.Detections."""
        det_path = self.experiment_dir / "detections.json"
        if not det_path.exists():
            raise FileNotFoundError(f"No detections.json in {self.experiment_dir}")
        raw = json.loads(det_path.read_text())
        return {
            stem: sv.Detections(
                xyxy=np.array(d["xyxy"], dtype=np.float32),
                confidence=np.array(d["confidence"], dtype=np.float32),
                class_id=np.array(d["class_id"], dtype=int),
            )
            for stem, d in raw.items()
        }

    def load_config_json(self) -> dict:
        """Load the raw config.json dict (used by ExperimentTracker for resume)."""
        p = self.experiment_dir / "config.json"
        if not p.exists():
            return {}
        return json.loads(p.read_text())

    def _write_json(self, filename: str, data: dict) -> None:
        path = self.experiment_dir / filename
        path.write_text(json.dumps(data, indent=2))
        logger.debug("Written: %s", path)
