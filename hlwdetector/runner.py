"""ExperimentRunner — single entry point for all experiment runs.

Usage:
    python -m experiments.runner configs/example_yolo11_h23.yaml
"""

from __future__ import annotations

import logging

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
        self.AdapterClass = get_adapter(self.config.model_name)
        self.dataset_manager = DatasetManager(self.config)
        logger.info("Initializing ExperimentRunner with adapter %s", self.AdapterClass)

    def run(self) -> None:
        config = self.config
        self.artifact_manager = ArtifactManager(config)
        self.artifact_manager.save_config(config)
        self.adapter = self.AdapterClass(self.artifact_manager)
        tracker = ExperimentTracker(config, self.artifact_manager)

        if config.resume_from is None:
            self.adapter.prepare_data(self.dataset_manager, config)
            training_result = self.adapter.train(config)
            self.artifact_manager.save_model_info(training_result)
            tracker.log({"training": training_result.training_metrics})
        else:
            logger.info(
                "Resuming from %s — skipping prepare_data and train.", config.resume_from
            )

        metrics = self.adapter.evaluate(config)
        self.artifact_manager.save_metrics(metrics)
        tracker.log({
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "mAP50": metrics.map50,
            "mAP50_95": metrics.map50_95,
        })

        detections = self.adapter.predict(config)
        self.artifact_manager.save_detections(detections)

        if config.visualize:
            viz = VisualizationPipeline(config, self.artifact_manager, self.dataset_manager)
            viz.run(detections)
            tracker.log_artifact(str(self.artifact_manager.visualizations_dir), "visualizations")

        tracker.finish()
        print(f"Experiment complete: {self.artifact_manager.experiment_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m experiments.runner <config.yaml>")
        sys.exit(1)

    ExperimentRunner(sys.argv[1]).run()
