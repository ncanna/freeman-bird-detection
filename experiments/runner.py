"""ExperimentRunner — single entry point for all experiment runs.

Usage:
    python -m experiments.runner configs/example_yolo11_h23.yaml
"""

from __future__ import annotations

import logging

from experiments.artifact_manager import ArtifactManager
from experiments.config import ExperimentConfig
from experiments.dataset_manager import DatasetManager
from experiments.registry import get_adapter
from experiments.tracker import ExperimentTracker
from experiments.visualization.pipeline import VisualizationPipeline

# Import adapters to trigger @register_adapter decorators
import experiments.adapters  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, config_path: str) -> None:
        self.config = ExperimentConfig.from_yaml(config_path)
        self.config.validate()  # raises clear errors if prereqs missing

    def run(self) -> None:
        config = self.config
        dataset_manager = DatasetManager(config)
        artifact_manager = ArtifactManager(config)
        artifact_manager.save_config(config)
        tracker = ExperimentTracker(config, artifact_manager)

        AdapterClass = get_adapter(config.model_name)
        adapter = AdapterClass()

        if config.resume_from is None:
            work_dir = str(artifact_manager.get_work_dir("adapter"))
            adapter.prepare_data(dataset_manager, config, work_dir)
            training_result = adapter.train(config)
            artifact_manager.save_model_info(training_result)
            tracker.log({"training": training_result.training_metrics})
        else:
            logger.info(
                "Resuming from %s — skipping prepare_data and train.", config.resume_from
            )

        metrics = adapter.evaluate(config)
        artifact_manager.save_metrics(metrics)
        tracker.log({
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "mAP50": metrics.map50,
            "mAP50_95": metrics.map50_95,
        })

        detections = adapter.predict(config)
        artifact_manager.save_detections(detections)

        if config.visualize:
            viz = VisualizationPipeline(config, artifact_manager, dataset_manager)
            viz.run(detections)
            tracker.log_artifact(str(artifact_manager.visualizations_dir), "visualizations")

        tracker.finish()
        print(f"Experiment complete: {artifact_manager.experiment_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m experiments.runner <config.yaml>")
        sys.exit(1)

    ExperimentRunner(sys.argv[1]).run()
