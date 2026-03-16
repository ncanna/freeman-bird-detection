"""ExperimentTracker — W&B (optional) + always-on local metrics.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiments.artifact_manager import ArtifactManager
    from experiments.config import ExperimentConfig

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Tracks metrics locally and optionally to W&B.

    W&B failures are non-fatal — the run continues with local-only tracking.
    metrics.json is written on every log() call so partial results survive
    PACE ICE job preemption.
    """

    def __init__(
        self,
        config: "ExperimentConfig",
        artifact_manager: "ArtifactManager",
    ) -> None:
        self._config = config
        self._artifact_manager = artifact_manager
        self._metrics: dict = {}
        self._metrics_path = artifact_manager.experiment_dir / "metrics.json"
        self._wandb_run = None
        self._wandb_enabled = False

        if config.wandb_project is not None:
            self._init_wandb(config, artifact_manager)

    def _init_wandb(
        self,
        config: "ExperimentConfig",
        artifact_manager: "ArtifactManager",
    ) -> None:
        try:
            import wandb  # type: ignore

            init_kwargs: dict = {
                "project": config.wandb_project,
                "name": config.wandb_run_name or config.experiment_name,
            }

            # Resume: reuse existing W&B run id from config.json
            if config.resume_from is not None:
                saved = artifact_manager.load_config_json()
                run_id = saved.get("wandb_run_id")
                if run_id:
                    init_kwargs["resume"] = "must"
                    init_kwargs["id"] = run_id

            self._wandb_run = wandb.init(**init_kwargs)
            self._wandb_enabled = True

            # Persist W&B run id so resume works later
            artifact_manager.save_config(config, wandb_run_id=self._wandb_run.id)
            logger.info("W&B run initialized: %s", self._wandb_run.id)

        except Exception as exc:
            logger.warning("W&B init failed (continuing local-only): %s", exc)
            self._wandb_enabled = False

    def log(self, metrics: dict, step: int | None = None) -> None:
        """Update internal metrics dict, flush metrics.json, optionally push to W&B."""
        self._metrics.update(metrics)
        self._flush()

        if self._wandb_enabled and self._wandb_run is not None:
            try:
                self._wandb_run.log(metrics, step=step)
            except Exception as exc:
                logger.warning("W&B log failed: %s", exc)

    def log_artifact(self, path: str, name: str, artifact_type: str = "result") -> None:
        """Log a file/directory as a W&B artifact (no-op if W&B disabled)."""
        if not self._wandb_enabled or self._wandb_run is None:
            return
        try:
            import wandb  # type: ignore
            artifact = wandb.Artifact(name=name, type=artifact_type)
            p = Path(path)
            if p.is_dir():
                artifact.add_dir(str(p))
            else:
                artifact.add_file(str(p))
            self._wandb_run.log_artifact(artifact)
        except Exception as exc:
            logger.warning("W&B log_artifact failed: %s", exc)

    def finish(self) -> None:
        """Final flush of metrics.json and close W&B run."""
        self._flush()
        if self._wandb_enabled and self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception as exc:
                logger.warning("W&B finish failed: %s", exc)

    def _flush(self) -> None:
        self._metrics_path.write_text(json.dumps(self._metrics, indent=2))
