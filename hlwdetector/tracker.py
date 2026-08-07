"""ExperimentTracker — W&B (optional) + always-on local metrics.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING
import wandb

if TYPE_CHECKING:
    from hlwdetector.artifact_manager import ArtifactManager
    from hlwdetector.config.experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)


def _stored_wandb_run_id(experiment_dir: "str | Path") -> str | None:
    """Read wandb_run_id out of an experiment dir's config.json (None if unavailable)."""
    config_path = Path(experiment_dir) / "config.json"
    try:
        return json.loads(config_path.read_text()).get("wandb_run_id")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.warning("Could not read wandb_run_id from %s: %s", config_path, exc)
        return None


class ExperimentTracker:
    """Tracks metrics locally and optionally to W&B.

    W&B failures are non-fatal — the run continues and falls back on native model metrics tracking
    """

    def __init__(
        self,
        config: "ExperimentConfig",
        artifact_manager: "ArtifactManager",
        wandb_run_id: str | None = None,
    ) -> None:
        self._config = config
        self._artifact_manager = artifact_manager
        self._metrics: dict = {}
        self._metrics_path = artifact_manager.experiment_dir / "metrics.json"
        self._wandb_run = None
        self._wandb_enabled = False

        if config.wandb_project is not None:
            self._init_wandb(config, artifact_manager, wandb_run_id)

    def _init_wandb(
        self,
        config: "ExperimentConfig",
        artifact_manager: "ArtifactManager",
        wandb_run_id: str | None = None,
    ) -> None:
        try:
            init_kwargs: dict = {
                "project": config.wandb_project,
                "name": artifact_manager.experiment_name,
                "dir": Path(artifact_manager.experiment_dir)
            }

            if config.wandb_group is not None:
                init_kwargs["group"] = config.wandb_group

            if wandb_run_id is not None:
                # Attaching to an existing experiment (e.g. eval after training)
                init_kwargs["resume"] = "must"
                init_kwargs["id"] = wandb_run_id
                logger.info("Resuming W&B run ID %s", wandb_run_id)
            elif config.resume_experiment_name is not None:
                # Training resume: continue the original run's W&B history. The id lives
                # in the *original* experiment's config.json — the new run dir's copy was
                # just written by save_config() and carries no id yet.
                run_id = _stored_wandb_run_id(config.resume_experiment_dir)
                if run_id:
                    init_kwargs["resume"] = "must"
                    init_kwargs["id"] = run_id
                    logger.info(
                        "Resuming W&B run ID %s from %s", run_id, config.resume_experiment_name
                    )
                else:
                    logger.warning(
                        "No wandb_run_id found in %s — starting a new W&B run.",
                        config.resume_experiment_name,
                    )
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

    def log_wandb_step(self, metrics: dict, step: int) -> None:
        """Push per-step metrics directly to W&B without touching _metrics or metrics.json."""
        if not self._wandb_enabled or self._wandb_run is None:
            return
        try:
            self._wandb_run.log(metrics, step=step)
        except Exception as exc:
            logger.warning("W&B log failed: %s", exc)

    def log_video(self, path: str, key: str = "visualization") -> None:
        """Upload a video file to W&B as a media object (no-op if W&B disabled)."""
        if not self._wandb_enabled or self._wandb_run is None:
            return
        self._wandb_run.log({key: wandb.Video(path, format="mp4")})

    def log_artifact(self, path: str, name: str, artifact_type: str = "result") -> None:
        """Log a file/directory as a W&B artifact (no-op if W&B disabled)."""
        if not self._wandb_enabled or self._wandb_run is None:
            return
        artifact = wandb.Artifact(name=name, type=artifact_type)
        p = Path(path)
        if p.is_dir():
            artifact.add_dir(str(p))
        else:
            artifact.add_file(str(p))
        logged = self._wandb_run.log_artifact(artifact)
        if logged is not None:
            logged.wait()

    def finish(self) -> None:
        """Final flush of metrics.json and close W&B run (idempotent)."""
        self._flush()
        if self._wandb_enabled and self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception as exc:
                logger.warning("W&B finish failed: %s", exc)
            finally:
                self._wandb_run = None
                self._wandb_enabled = False

    def _flush(self) -> None:
        self._metrics_path.write_text(json.dumps(self._metrics, indent=2))
