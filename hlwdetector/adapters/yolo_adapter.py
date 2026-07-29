"""YOLOAdapter — migrated from models/yolo/yolo.ipynb."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict

import supervision as sv
from ultralytics import YOLO, settings

from hlwdetector import paths
from hlwdetector.adapters.base import (
    BaseModelAdapter,
    DetectionResult,
    MetricsDict,
    TrainingResult,
)
from hlwdetector.registry import register_adapter

if TYPE_CHECKING:
    from hlwdetector.config.experiment_config import ExperimentConfig
    from hlwdetector.dataset_manager import DatasetManager

logger = logging.getLogger(__name__)

# Ensure project root is on sys.path so utilities imports work
_PROJECT_ROOT = str(paths.REPO_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@register_adapter("yolo")
class YOLOAdapter(BaseModelAdapter):
    """YOLO adapter backed by Ultralytics.

    Internal state is preserved across sequential calls:
        prepare_data → train → evaluate → predict
    """

    def __init__(self, artifact_manager, tracker) -> None:
        super().__init__(artifact_manager, tracker)
        self._model = None
        self._data_yaml_path: str | None = None
        self._training_result: TrainingResult | None = None
        # Optional (epoch, metrics) -> None hook, set by HPO; may raise optuna.TrialPruned.
        self._hpo_pruning_callback = None

    # ------------------------------------------------------------------ #
    # prepare_data
    # ------------------------------------------------------------------ #

    def prepare_data(
        self,
        dataset_manager: "DatasetManager",
        config: "ExperimentConfig",
    ) -> None:
        """Convert COCO annotations to YOLO format and create yolo.yaml."""
        import yaml as _yaml
        from utilities.annotation_converter import AnnotationConverter

        work_path = Path(self.work_dir)
        source_images_dir = Path(config.images_dir).resolve()
        dataset_root, images_dir, labels_dir = self._prepare_ultralytics_dataset_root(
            source_images_dir
        )

        converter = AnnotationConverter(class_mapping={"bird": 0})

        for split_name in ("train", "val", "test"):
            split_view = dataset_manager.get_split(split_name)

            # Convert COCO annotations to YOLO label files (flat, all splits share labels_dir)
            converter.coco_to_yolo(
                coco_json_path=split_view.coco_json_path,
                output_dir=str(labels_dir),
                use_filename=True,
                video_filter=split_view.video_stems,
            )

            # Write a text file listing absolute image paths for this split
            source_image_paths = split_view.image_paths
            missing = [p for p in source_image_paths if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Split '{split_name}': {len(missing)}/{len(source_image_paths)} image files are missing from "
                    f"{source_images_dir}. Extract frames first using extract_frames_from_dir(). "
                    f"First missing: {missing[0]}"
                )
            # Resolve only to calculate a source-relative filename, then rebuild
            # the run-local symlink path that Ultralytics uses to find labels.
            image_paths = [
                images_dir / path.resolve().relative_to(source_images_dir)
                for path in source_image_paths
            ]
            txt_path = work_path / f"{split_name}.txt"
            txt_path.write_text(
                "\n".join(str(p) for p in image_paths) + "\n"
            )

        # Write yolo.yaml referencing the text files
        # YOLO label auto-discovery: replaces /images/ → /labels/ in each image path,
        # so labels land at {images_dir.parent}/labels/{filename}.txt ✓
        yaml_data = {
            "path": str(dataset_root),
            "train": str(work_path / "train.txt"),
            "val": str(work_path / "val.txt"),
            "test": str(work_path / "test.txt"),
            "nc": 1,
            "names": {0: "bird"},
        }
        yaml_path = work_path / "yolo.yaml"
        with open(yaml_path, "w") as f:
            _yaml.dump(yaml_data, f, default_flow_style=False)

        self._data_yaml_path = str(yaml_path)
        logger.info("YOLO data yaml written to: %s", self._data_yaml_path)

    # ------------------------------------------------------------------ #
    # train
    # ------------------------------------------------------------------ #

    def train(self, config: "ExperimentConfig") -> TrainingResult:
        """Fine-tune YOLO and return TrainingResult."""
        if self._data_yaml_path is None and config.resume_from is None:
            raise RuntimeError("Call prepare_data() before train().")

        hp = config.hyperparameters
        model_weights = config.model_weights
        train_kwargs = {k: v for k, v in hp.items()}

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
            self._register_epoch_callback()
            self._model.train(
                data=self._data_yaml_path,
                project=runs_dir,
                name="train",
                **train_kwargs,
            )
        else:  # resume: load pretrained weights, train fresh with full hparam control
            self._discover_data_yaml(config)
            self._model = YOLO(config.resume_from)
            self._register_epoch_callback()
            self._model.train(
                data=self._data_yaml_path,
                project=runs_dir,
                name="train",
                **train_kwargs,
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
        """Validate on val split and return MetricsDict."""
        if self._model is None:
            self._load_model_from_artifacts(config)

        if self._data_yaml_path is None:
            self._discover_data_yaml(config)

        results = self._model.val(data=self._data_yaml_path, split="val")  # type: ignore

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

        test_txt = Path(self.work_dir) / "test.txt"
        results = self._model.predict(str(test_txt))  # type: ignore

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

    def _register_epoch_callback(self) -> None:
        """Register an on_fit_epoch_end callback that routes per-epoch metrics to the tracker.

        Fires once per epoch after validation completes. Logs training losses,
        validation metrics, and learning rates as a single dict at step=epoch.
        Future adapters should follow this same pattern using their framework's
        equivalent hook, calling self.log_epoch(epoch, metrics).
        """
        adapter = self

        def on_fit_epoch_end(trainer) -> Dict:
            epoch = trainer.epoch + 1  # Ultralytics epochs are 0-indexed
            metrics: dict = {}
            if trainer.metrics:
                metrics.update({k: float(v) for k, v in trainer.metrics.items()})
            if hasattr(trainer, "label_loss_items") and trainer.tloss is not None:
                metrics.update(
                    {k: float(v) for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items()}
                )
            if trainer.lr:
                metrics.update({k: float(v) for k, v in trainer.lr.items()})
            if metrics:
                adapter._tracker.log_wandb_step(metrics, step=epoch)
            if adapter._hpo_pruning_callback is not None:
                adapter._hpo_pruning_callback(epoch, metrics)  # may raise optuna.TrialPruned

            return metrics

        self._model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    def _load_model_from_artifacts(self, config: "ExperimentConfig") -> None:
        """Load weights for evaluate/predict without a prior train() call.

        Priority:
          1. config.resume_from (resume-training flow)
          2. best_weights_path from model.json in experiment_dir (attach flow)
        """
        if config.resume_from is not None:
            weights_path = Path(config.resume_from)
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            self._model = YOLO(str(weights_path))
            logger.info("Loaded weights from config.resume_from: %s", weights_path)
            return

        # Attach flow: read best_weights_path from model.json
        model_json_path = Path(self.experiment_dir) / "model.json"
        if not model_json_path.exists():
            raise FileNotFoundError(
                f"No model loaded, resume_from is not set, and model.json not found in: "
                f"{self.experiment_dir}"
            )
        model_info = json.loads(model_json_path.read_text())
        weights_path_str = model_info.get("best_weights_path")
        if not weights_path_str:
            raise RuntimeError(
                f"model.json in {self.experiment_dir} has no best_weights_path"
            )
        weights_path = paths.resolve(weights_path_str)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"best_weights_path from model.json does not exist: {weights_path}"
            )
        self._model = YOLO(str(weights_path))
        logger.info("Loaded weights from model.json: %s", weights_path)

    def _discover_data_yaml(self, config: "ExperimentConfig") -> None:
        """Locate yolo.yaml for evaluate/predict without a prior prepare_data() call.

        Priority:
          1. self.work_dir/yolo.yaml — present when attached to an existing experiment
             (work_dir IS the original experiment's work dir in the attach flow)
          2. config.resume_experiment work dir — existing resume-training flow
        """
        direct_candidate = Path(self.work_dir) / "yolo.yaml"
        if direct_candidate.exists():
            self._data_yaml_path = str(direct_candidate)
            logger.info("Found data yaml in work_dir: %s", self._data_yaml_path)
            return

        if config.resume_experiment is None:
            raise RuntimeError(
                "yolo.yaml not found in work_dir and resume_experiment is not set; "
                "cannot locate original yolo.yaml."
            )
        work_dir  = Path(config.output_dir) / "experiments" / config.resume_experiment / "work"
        candidate = work_dir / "yolo.yaml"
        if candidate.exists():
            self._data_yaml_path = str(candidate)
            logger.info("Found data yaml via resume_experiment: %s", self._data_yaml_path)
        else:
            raise FileNotFoundError(
                f"yolo.yaml not found at {candidate}. "
                f"Ensure prepare_data() was run in experiment '{config.resume_experiment}'."
            )
