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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{config.config_name}_{timestamp}"
        if config.resume_from is not None:
            original_dir = (Path(config.output_dir) / config.resume_experiment).resolve()
            if not original_dir.exists():
                raise FileNotFoundError(
                    f"Original experiment dir not found: {original_dir}"
                )
            self.experiment_dir = (
                Path(config.output_dir) / self.experiment_name
            ).resolve()
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Resuming %s -> new dir: %s", original_dir.name, self.experiment_dir)
            self._stamp_resumed_in(original_dir, self.experiment_dir)
        else:
            self.experiment_dir = (
                Path(config.output_dir) / self.experiment_name
            ).resolve()
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Experiment directory: %s", self.experiment_dir)

        self.work_dir = self.experiment_dir / "work"
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Ensure visualizations subdirectory exists
        self.visualizations_dir = self.experiment_dir / "visualizations"
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_existing_dir(cls, experiment_dir: "str | Path") -> "ArtifactManager":
        """Attach to an existing experiment directory — no new timestamp, no mkdir."""
        experiment_dir = Path(experiment_dir).resolve()
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

        instance = cls.__new__(cls)
        instance.experiment_name = experiment_dir.name
        instance.experiment_dir = experiment_dir
        instance.work_dir = experiment_dir / "work"
        instance.visualizations_dir = experiment_dir / "visualizations"

        if not instance.work_dir.exists():
            raise FileNotFoundError(f"work/ subdirectory not found in: {experiment_dir}")
        # visualizations/ may not exist in older runs — create it like __init__ does
        instance.visualizations_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Attached to existing experiment: %s", experiment_dir)
        return instance

    def _stamp_resumed_in(self, original_dir: Path, new_dir: Path) -> None:
        """Append resumed_in field to the original experiment's config.json."""
        config_path = original_dir / "config.json"
        try:
            data = json.loads(config_path.read_text())
            data["resumed_in"] = str(new_dir)
            config_path.write_text(json.dumps(data, indent=2))
            logger.info("Stamped resumed_in on: %s", config_path)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning("Could not stamp resumed_in on original config.json: %s", exc)

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
        result = {}
        for stem, d in raw.items():
            if not d["xyxy"]:
                result[stem] = sv.Detections.empty()
            else:
                result[stem] = sv.Detections(
                    xyxy=np.array(d["xyxy"], dtype=np.float32),
                    confidence=np.array(d["confidence"], dtype=np.float32),
                    class_id=np.array(d["class_id"], dtype=int),
                )
        return result

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
