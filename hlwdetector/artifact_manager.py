"""ArtifactManager — canonical output paths and serialization helpers."""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import supervision as sv

from hlwdetector import paths

if TYPE_CHECKING:
    from hlwdetector.adapters.base import MetricsDict, TrainingResult
    from hlwdetector.config.experiment_config import ExperimentConfig
    from hlwdetector.config.hpo_config import HPOConfig

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Manages all output paths for an experiment or HPO study run."""

    def __init__(self, config: "ExperimentConfig" | "HPOConfig") -> None:
        # Local import avoids a load-time circular import; HPOConfig lacks the
        # config_name/resume_* fields the experiment path relies on.
        from hlwdetector.config.hpo_config import HPOConfig

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.is_hpo = isinstance(config, HPOConfig)

        if self.is_hpo:  # HPO study: <output_dir>/hpo/<study_name>_<timestamp>, no work/viz dirs.
            self.study_name = config.study_args.study_name
            self.experiment_name = f"{self.study_name}_{timestamp}"
            self.experiment_dir = (
                Path(config.output_dir) / "hpo" / self.experiment_name
            ).resolve()
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            self.optuna_journal_path = self.experiment_dir / "optuna_journal.log"
            self.trials_csv_path = self.experiment_dir / "trials.csv"
            self._trials_csv_fieldnames: list[str] | None = None
            logger.info("HPO study directory: %s", self.experiment_dir)
            return

        else:  # Experiment run: <output_dir>/experiments/<config_name>_<timestamp>.
            self.experiment_name = (
                config.run_name
                if config.run_name is not None
                else f"{config.config_name}_{timestamp}"
            )
            experiments_root = Path(config.output_dir) / "experiments"

            # Validate the resume target before creating the new dir, so a bad
            # resume doesn't leave a stray empty directory behind.
            original_dir = None
            if config.resume_from is not None:
                resume_candidates = (
                    experiments_root / config.resume_experiment,
                    Path(config.output_dir) / config.resume_experiment,
                )
                original_dir = next(
                    (path.resolve() for path in resume_candidates if path.exists()),
                    None,
                )
                if original_dir is None:
                    raise FileNotFoundError(
                        "Original experiment dir not found; checked: "
                        + ", ".join(str(path.resolve()) for path in resume_candidates)
                    )

            self.experiment_dir = (experiments_root / self.experiment_name).resolve()
            if config.run_name is not None and self.experiment_dir.exists():
                raise FileExistsError(
                    f"Explicit run_name already exists: {self.experiment_dir}. "
                    "Choose a new run number to avoid mixing experiment artifacts."
                )
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            if original_dir is not None:
                logger.info("Resuming %s -> new dir: %s", original_dir.name, self.experiment_dir)
                self._stamp_resumed_in(original_dir, self.experiment_dir)
            else:
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
        instance.is_hpo = False
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

    def attach_log_file(self, mode: str = "w") -> None:
        """Add a FileHandler to the root logger writing to experiment.log.

        Removes any stale FileHandlers first so notebook reruns don't stack handlers.
        Use mode='a' when attaching to an existing experiment dir.
        """
        log_path = self.experiment_dir / "experiment.log"
        root = logging.getLogger()
        for h in root.handlers[:]:
            if isinstance(h, logging.FileHandler):
                h.close()
                root.removeHandler(h)
        handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s")
        )
        root.addHandler(handler)
        self._log_handler = handler
        logger.info("Logging to file: %s", log_path)

    # ------------------------------------------------------------------ #
    # HPO study artifacts
    # ------------------------------------------------------------------ #

    def attach_hpo_log_file(self, mode: str = "w") -> None:
        """Add a FileHandler writing hpo.log to the hp_optimizer module logger.

        Attaches to the ``hlwdetector.hp_optimizer`` logger rather than the root
        logger on purpose: each trial's ExperimentRunner calls attach_log_file(),
        which strips FileHandlers off the *root* logger. Keeping the HPO handler on
        the module logger lets it survive across trials. Records still propagate to
        the console.
        """
        log_path = self.experiment_dir / "hpo.log"
        hpo_logger = logging.getLogger("hlwdetector.hp_optimizer")
        # Ensure INFO records reach the handler even if the root level is higher.
        hpo_logger.setLevel(logging.INFO)
        for h in hpo_logger.handlers[:]:
            if isinstance(h, logging.FileHandler):
                h.close()
                hpo_logger.removeHandler(h)
        handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s")
        )
        hpo_logger.addHandler(handler)
        self._hpo_log_handler = handler
        logger.info("HPO logging to file: %s", log_path)

    def record_trial(
        self,
        trial_number: int,
        sampled_hparams: dict,
        metric_value: float | None,
        state: str,
        duration_s: float,
    ) -> None:
        """Append one trial's row to trials.csv (header written on first call)."""
        row = {
            "trial_number": trial_number,
            **sampled_hparams,
            "metric_value": "" if metric_value is None else metric_value,
            "state": state,
            "duration_s": duration_s,
        }
        write_header = not self.trials_csv_path.exists()
        if self._trials_csv_fieldnames is None:
            self._trials_csv_fieldnames = [
                "trial_number",
                *sampled_hparams.keys(),
                "metric_value",
                "state",
                "duration_s",
            ]
        with self.trials_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=self._trials_csv_fieldnames, extrasaction="ignore"
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        logger.debug("Recorded trial %d (%s) to %s", trial_number, state, self.trials_csv_path)

    def save_study_summary(
        self,
        best_trial_number: int,
        best_config_name: str,
        best_value: float,
        best_params: dict,
        n_trials: int,
        direction: str,
        total_time_s: float,
    ) -> None:
        """Write study_summary.json describing the best trial and study size."""
        total_time = int(total_time_s)
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        self._write_json(
            "study_summary.json",
            {
                "study_name": self.study_name,
                "best_trial_number": best_trial_number,
                "best_config_name": best_config_name,
                "best_value": best_value,
                "best_params": best_params,
                "n_trials": n_trials,
                "direction": direction,
                "total_time": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            },
        )

    def _stamp_resumed_in(self, original_dir: Path, new_dir: Path) -> None:
        """Append resumed_in field to the original experiment's config.json."""
        config_path = original_dir / "config.json"
        try:
            data = json.loads(config_path.read_text())
            data["resumed_in"] = paths.to_repo_rel(new_dir)
            config_path.write_text(json.dumps(data, indent=2))
            logger.info("Stamped resumed_in on: %s", config_path)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning("Could not stamp resumed_in on original config.json: %s", exc)

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #

    def save_config(self, config: "ExperimentConfig", wandb_run_id: str | None = None) -> None:
        """Write config.json with repo-relative paths (includes wandb_run_id if W&B is active)."""
        data = config.to_serializable_dict()
        if wandb_run_id is not None:
            data["wandb_run_id"] = wandb_run_id
        self._write_json("config.json", data)

    def save_model_info(self, result: "TrainingResult") -> None:
        """Write model.json with run/weights paths stored relative to the repo root."""
        data = dataclasses.asdict(result)
        for key in ("run_dir", "best_weights_path", "last_weights_path"):
            if data.get(key) is not None:
                data[key] = paths.to_repo_rel(data[key])
        self._write_json("model.json", data)

    def save_metrics(self, metrics: "MetricsDict") -> None:
        """Write metrics.json with evaluation results."""
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
