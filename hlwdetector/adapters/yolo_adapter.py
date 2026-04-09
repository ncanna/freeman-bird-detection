"""YOLOAdapter — migrated from models/yolo/yolo.ipynb."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import supervision as sv
from ultralytics import YOLO, settings

from hlwdetector.adapters.base import (
    BaseModelAdapter,
    DetectionResult,
    MetricsDict,
    TrainingResult,
)
from hlwdetector.registry import register_adapter

if TYPE_CHECKING:
    from hlwdetector.config import ExperimentConfig
    from hlwdetector.dataset_manager import DatasetManager

logger = logging.getLogger(__name__)

# Ensure project root is on sys.path so utilities imports work
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@register_adapter("yolo")
class YOLOAdapter(BaseModelAdapter):
    """YOLO adapter backed by Ultralytics.

    Internal state is preserved across sequential calls:
        prepare_data → train → evaluate → predict
    """

    def __init__(self, artifact_manager) -> None:
        super().__init__(artifact_manager)
        self._model = None
        self._data_yaml_path: str | None = None
        self._training_result: TrainingResult | None = None

    # ------------------------------------------------------------------ #
    # prepare_data
    # ------------------------------------------------------------------ #

    def prepare_data(
        self,
        dataset_manager: "DatasetManager",
        config: "ExperimentConfig",
    ) -> None:
        """Convert COCO annotations to YOLO format and create yolo.yaml."""
        from utilities.annotation_converter import AnnotationConverter

        work_path = Path(self.work_dir)

        for split_name in ("train", "val", "test"):
            split_view = dataset_manager.get_split(split_name)
            labels_out = work_path / split_name / "labels"
            labels_out.mkdir(parents=True, exist_ok=True)

            converter = AnnotationConverter(class_mapping={"bird": 0})
            converter.coco_to_yolo(
                coco_json_path=split_view.coco_json_path,
                output_dir=str(labels_out),
                use_filename=True,
                video_filter=split_view.video_stems,
            )

        # Create yolo.yaml using train split's images_dir as dataset root
        train_view = dataset_manager.get_split("train")
        val_view = dataset_manager.get_split("val")
        test_view = dataset_manager.get_split("test")

        # dataset_path = images base dir (parent of train/ val/ test/)
        images_base = str(Path(train_view.images_split_dir).parent)

        # Point YOLO at the actual image directories for each split
        converter = AnnotationConverter(class_mapping={"bird": 0})
        converter.create_yaml_config(
            output_dir=self.work_dir,
            dataset_path=images_base,
            train_dir="train",
            val_dir="val",
            test_dir="test",
        )

        self._data_yaml_path = str(work_path / "yolo.yaml")
        logger.info("YOLO data yaml written to: %s", self._data_yaml_path)

    # ------------------------------------------------------------------ #
    # train
    # ------------------------------------------------------------------ #

    def train(self, config: "ExperimentConfig") -> TrainingResult:
        """Fine-tune YOLO and return TrainingResult."""
        if self._data_yaml_path is None and config.resume_from is None:
            raise RuntimeError("Call prepare_data() before train().")

        hp = config.hyperparameters
        model_weights = hp.get("model_weights")
        epochs = hp.get("epochs")
        imgsz = hp.get("imgsz")
        batch = hp.get("batch")
        device = hp.get("device")

        # Point Ultralytics runs to outputs directory
        runs_dir = str(Path(self.work_dir) / "runs")
        #runs_dir = str(Path(config.output_dir) / "runs")
        settings.update({
            "runs_dir": runs_dir,
            "tensorboard": False,
            "wandb": False,
        })

        if config.resume_from is None:
            self._model = YOLO(model_weights)
            self._model.train(
                data=self._data_yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                project=runs_dir,
                name=f"{config.experiment_name}_train",
            )
        else:  # resume: load pretrained weights, train fresh with full hparam control
            self._discover_data_yaml(config)
            self._model = YOLO(config.resume_from)
            self._model.train(
                data=self._data_yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                project=runs_dir,
                name=f"{config.experiment_name}_train",
            )

        run_dir = Path(self._model.trainer.save_dir)
        best_pt = run_dir / "weights" / "best.pt"
        last_pt = run_dir / "weights" / "last.pt"

        # Load best weights for subsequent evaluate/predict calls
        if best_pt.exists():
            self._model = YOLO(str(best_pt))

        # Collect scalar training metrics if available
        training_metrics: dict = {}
        if hasattr(self._model.trainer, "metrics"):
            raw = self._model.trainer.metrics
            if isinstance(raw, dict):
                training_metrics = {k: float(v) for k, v in raw.items() if v is not None}

        self._training_result = TrainingResult(
            run_dir=str(run_dir),
            best_weights_path=str(best_pt) if best_pt.exists() else None,
            last_weights_path=str(last_pt) if last_pt.exists() else None,
            training_metrics=training_metrics,
        )
        return self._training_result

    # ------------------------------------------------------------------ #
    # evaluate
    # ------------------------------------------------------------------ #

    def evaluate(self, config: "ExperimentConfig") -> MetricsDict:
        """Validate on test split and return MetricsDict."""
        if self._model is None:
            self._load_model_from_artifacts(config)

        if self._data_yaml_path is None:
            self._discover_data_yaml(config)

        results = self._model.val(data=self._data_yaml_path, split="test")  # type: ignore

        box = results.box
        precision = float(box.mp)
        recall = float(box.mr)
        map50 = float(box.map50)
        map50_95 = float(box.map)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return MetricsDict(
            precision=precision,
            recall=recall,
            f1=f1,
            map50=map50,
            map50_95=map50_95,
            raw={
                "map50": map50,
                "map50_95": map50_95,
                "precision": precision,
                "recall": recall,
            },
        )

    # ------------------------------------------------------------------ #
    # predict
    # ------------------------------------------------------------------ #

    def predict(self, config: "ExperimentConfig") -> DetectionResult:
        """Run inference on test images; return per-frame sv.Detections."""
        if self._model is None:
            self._load_model_from_artifacts(config)

        test_images_dir = str(Path(config.images_dir) / "test")
        results = self._model.predict(test_images_dir)  # type: ignore

        predictions: DetectionResult = {}
        for result in results:
            stem = Path(result.path).stem
            xyxy = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            predictions[stem] = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)

        return predictions

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_model_from_artifacts(self, config: "ExperimentConfig") -> None:
        """Load weights from resume_from for evaluate/predict without a prior train()."""
        if config.resume_from is None:
            raise RuntimeError("No model loaded and resume_from is not set.")
        weights_path = Path(config.resume_from)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        self._model = YOLO(str(weights_path))
        logger.info("Loaded weights from: %s", weights_path)

    def _discover_data_yaml(self, config: "ExperimentConfig") -> None:
        """Locate yolo.yaml from the original experiment referenced by resume_experiment."""
        if config.resume_experiment is None:
            raise RuntimeError("resume_experiment is not set; cannot locate original yolo.yaml.")
        candidate = Path(config.output_dir) / config.resume_experiment / "work" / "yolo.yaml"
        if candidate.exists():
            self._data_yaml_path = str(candidate)
            logger.info("Found data yaml at: %s", self._data_yaml_path)
        else:
            raise FileNotFoundError(
                f"yolo.yaml not found at {candidate}. "
                f"Ensure prepare_data() was run in experiment '{config.resume_experiment}'."
            )
