"""ExperimentRunner — single entry point for all experiment runs.

Usage:
    python -m experiments.runner configs/example_yolo11_h23.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path

from hlwdetector.artifact_manager import ArtifactManager
from hlwdetector.config import ExperimentConfig
from hlwdetector.dataset_manager import DatasetManager
from hlwdetector.registry import get_adapter
from hlwdetector.tracker import ExperimentTracker
from hlwdetector.visualization.pipeline import VisualizationPipeline

# Import adapters to trigger @register_adapter decorators
import hlwdetector.adapters  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, config_path: str) -> None:
        self.config = ExperimentConfig.from_yaml(config_path)
        self.config.validate()  # raises clear errors if prereqs missing
        self.artifact_manager = ArtifactManager(self.config)
        self.artifact_manager.attach_log_file()
        self.artifact_manager.save_config(self.config)
        self.tracker = ExperimentTracker(self.config, self.artifact_manager)
        self.AdapterClass = get_adapter(self.config.model_name)
        self.adapter = self.AdapterClass(self.artifact_manager, self.tracker)
        self.dataset_manager = DatasetManager(self.config)
        logger.info("Initializing ExperimentRunner with adapter %s", self.AdapterClass)

    def train(self):
        if self.config.resume_from is None:
            self.adapter.prepare_data(self.dataset_manager, self.config)

        training_result = self.adapter.train(self.config)
        self.artifact_manager.save_model_info(training_result)

    def evaluate(self):
        metrics = self.adapter.evaluate(self.config)
        self.artifact_manager.save_metrics(metrics)
        self.tracker.log({
            "val/precision": metrics.precision,
            "val/recall": metrics.recall,
            "val/f1": metrics.f1,
            "val/mAP50": metrics.map50,
            "val/mAP50_95": metrics.map50_95,
        })

    def predict(self):
        self.detections = self.adapter.predict(self.config)
        self.artifact_manager.save_detections(self.detections)

    def visualize_predictions(self):
        if not hasattr(self, "detections"):
            self.detections = self.artifact_manager.load_detections()
        viz = VisualizationPipeline(self.config, self.artifact_manager, self.dataset_manager)
        viz.run(self.detections)
        video_path = str(
            self.artifact_manager.visualizations_dir / f"{self.config.config_name}_annotated.mp4"
        )
        self.tracker.log_video(video_path)

    def run_pipeline(self) -> None:
        self.train()
        self.evaluate()
        self.predict()
        self.visualize_predictions()
        self.tracker.finish()
        print(f"Experiment complete: {self.artifact_manager.experiment_dir}")


    @classmethod
    def from_experiment_dir(cls, experiment_dir: str) -> "ExperimentRunner":
        """Reconstruct a runner from a completed experiment directory for eval/predict.

        Does not create a new output directory. All outputs land in the existing dir.
        Raises FileNotFoundError if config.json or model.json are missing.
        """
        import dataclasses
        import json as _json

        experiment_dir = Path(experiment_dir).resolve()

        config_json_path = experiment_dir / "config.json"
        model_json_path  = experiment_dir / "model.json"
        if not config_json_path.exists():
            raise FileNotFoundError(f"config.json not found in: {experiment_dir}")
        if not model_json_path.exists():
            raise FileNotFoundError(
                f"model.json not found in: {experiment_dir}\n"
                "This experiment may not have completed training."
            )

        # Load config, stripping extra keys (wandb_run_id, resumed_in, etc.)
        raw = _json.loads(config_json_path.read_text())
        wandb_run_id = raw.get("wandb_run_id")
        valid_fields = {f.name for f in dataclasses.fields(ExperimentConfig)}
        config = ExperimentConfig(**{k: v for k, v in raw.items() if k in valid_fields})

        artifact_manager = ArtifactManager.from_existing_dir(experiment_dir)
        artifact_manager.attach_log_file(mode="a")
        tracker          = ExperimentTracker(config, artifact_manager, wandb_run_id=wandb_run_id)
        AdapterClass     = get_adapter(config.model_name)
        adapter          = AdapterClass(artifact_manager, tracker)
        dataset_manager  = DatasetManager(config)

        runner = cls.__new__(cls)
        runner.config           = config
        runner.artifact_manager = artifact_manager
        runner.tracker          = tracker
        runner.AdapterClass     = AdapterClass
        runner.adapter          = adapter
        runner.dataset_manager  = dataset_manager
        logger.info(
            "ExperimentRunner attached to: %s (adapter: %s)",
            experiment_dir.name,
            AdapterClass,
        )
        return runner


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m experiments.runner <config.yaml>")
        sys.exit(1)

    ExperimentRunner(sys.argv[1]).run()
