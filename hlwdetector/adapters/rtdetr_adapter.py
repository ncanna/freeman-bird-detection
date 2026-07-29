"""RTDETRAdapter — Ultralytics RT-DETR, mirrors YOLOAdapter exactly."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import supervision as sv
import torch
from ultralytics import RTDETR, settings
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.val import RTDETRDataset

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

DEFAULT_OPTIMIZER = "AdamW"
DEFAULT_LR0 = 1e-4
DEFAULT_MOMENTUM = 0.9
DEFAULT_WARMUP_BIAS_LR = 0.0

_PROJECT_ROOT = str(paths.REPO_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _collate_fractional_mosaic_batch(batch: list[dict]) -> dict:
    """Resize mixed RT-DETR image tensors before Ultralytics stacks them.

    RT-DETR's stretched augmentation pipeline emits mosaic samples at twice
    ``imgsz`` while non-mosaic samples remain at ``imgsz``. Endpoint mosaic
    probabilities therefore work, but a fractional probability can put both
    shapes in one batch. Stretch smaller tensors to the largest batch shape;
    normalized bounding boxes remain valid.
    """
    target_height = max(sample["img"].shape[-2] for sample in batch)
    target_width = max(sample["img"].shape[-1] for sample in batch)
    resized_batch = []
    for sample in batch:
        image = sample["img"]
        if image.shape[-2:] != (target_height, target_width):
            sample = dict(sample)
            sample["img"] = (
                torch.nn.functional.interpolate(
                    image.unsqueeze(0).float(),
                    size=(target_height, target_width),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .to(dtype=image.dtype)
            )
        resized_batch.append(sample)
    return RTDETRDataset.collate_fn(resized_batch)


class _FractionalMosaicRTDETRTrainer(RTDETRTrainer):
    """Use a shape-safe collator only when RT-DETR mosaic is fractional."""

    def build_dataset(
        self,
        img_path: str,
        mode: str = "val",
        batch: int | None = None,
    ):
        dataset = super().build_dataset(img_path, mode, batch)
        if mode == "train" and 0.0 < float(self.args.mosaic) < 1.0:
            dataset.collate_fn = _collate_fractional_mosaic_batch
        return dataset


@register_adapter("rtdetr")
class RTDETRAdapter(BaseModelAdapter):
    """RT-DETR adapter backed by Ultralytics.

    Hyperparameters (config.hyperparameters):
        model_weights: Ultralytics checkpoint name or path (e.g. "rtdetr-l.pt")
        epochs:        number of training epochs
        imgsz:         input image size (square, default 640)
        batch:         batch size
        device:        device string passed to Ultralytics (e.g. "0", "cpu")
        optimizer:     optimizer name (default AdamW for RT-DETR)
        lr0:           initial learning rate (default 1e-4 for RT-DETR)

    Internal state is preserved across sequential calls:
        prepare_data → train → evaluate → predict
    """

    def __init__(self, artifact_manager, tracker) -> None:
        super().__init__(artifact_manager, tracker)
        self._model = None
        self._data_yaml_path: str | None = None
        self._training_result: TrainingResult | None = None
        # Optional (epoch, metrics) -> None hook set by HPO.
        self._hpo_pruning_callback = None

    # ------------------------------------------------------------------ #
    # prepare_data
    # ------------------------------------------------------------------ #

    def prepare_data(
        self,
        dataset_manager: "DatasetManager",
        config: "ExperimentConfig",
    ) -> None:
        """Convert COCO annotations to YOLO format and create rtdetr.yaml."""
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

            converter.coco_to_yolo(
                coco_json_path=split_view.coco_json_path,
                output_dir=str(labels_dir),
                use_filename=True,
                video_filter=split_view.video_stems,
            )

            source_image_paths = split_view.image_paths
            missing = [p for p in source_image_paths if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Split '{split_name}': {len(missing)}/{len(source_image_paths)} image files are missing from "
                    f"{source_images_dir}. Extract frames first using extract_frames_from_dir(). "
                    f"First missing: {missing[0]}"
                )
            # Keep the final text-file paths under run-local ``dataset/images``.
            image_paths = [
                images_dir / path.resolve().relative_to(source_images_dir)
                for path in source_image_paths
            ]
            txt_path = work_path / f"{split_name}.txt"
            txt_path.write_text(
                "\n".join(str(p) for p in image_paths) + "\n"
            )

        yaml_data = {
            "path": str(dataset_root),
            "train": str(work_path / "train.txt"),
            "val": str(work_path / "val.txt"),
            "test": str(work_path / "test.txt"),
            "nc": 1,
            "names": {0: "bird"},
        }
        yaml_path = work_path / "rtdetr.yaml"
        with open(yaml_path, "w") as f:
            _yaml.dump(yaml_data, f, default_flow_style=False)

        self._data_yaml_path = str(yaml_path)
        logger.info("RT-DETR data yaml written to: %s", self._data_yaml_path)

    # ------------------------------------------------------------------ #
    # train
    # ------------------------------------------------------------------ #

    def train(self, config: "ExperimentConfig") -> TrainingResult:
        """Fine-tune RT-DETR and return TrainingResult."""
        if self._data_yaml_path is None and config.resume_from is None:
            raise RuntimeError("Call prepare_data() before train().")

        hp = config.hyperparameters
        model_weights = config.model_weights
        train_kwargs = {key: value for key, value in hp.items()}
        # Ultralytics' auto mode ignores lr0 and selects an SGD-family optimizer
        # for this study's iteration count. Pin transformer-appropriate defaults.
        train_kwargs.setdefault("optimizer", DEFAULT_OPTIMIZER)
        train_kwargs.setdefault("lr0", DEFAULT_LR0)
        train_kwargs.setdefault("momentum", DEFAULT_MOMENTUM)
        train_kwargs.setdefault("warmup_bias_lr", DEFAULT_WARMUP_BIAS_LR)
        # Keep below the 8 allocated cores or the GPU-feeding process starves.
        train_kwargs["workers"] = 4

        runs_dir = str(Path(self.work_dir) / "runs")
        settings.update({
            "runs_dir": runs_dir,
            "tensorboard": False,
            "wandb": False,
        })

        if config.resume_from is None:
            self._model = RTDETR(model_weights)
            self._register_epoch_callback()
            self._model.train(
                data=self._data_yaml_path,
                project=runs_dir,
                name="train",
                trainer=_FractionalMosaicRTDETRTrainer,
                **train_kwargs,
            )
        else:
            # True resume: Ultralytics reloads optimizer/EMA/LR-schedule/epoch and
            # the original train args (data, epochs, save_dir) from the checkpoint
            # and continues in the checkpoint's own run dir until its original
            # epoch target. Passing resume=True with no other args is what makes
            # this *continue* training rather than restart a fresh loop with a
            # reset LR schedule (which is why the old resume "started from scratch").
            self._model = RTDETR(config.resume_from)
            self._register_epoch_callback()
            self._model.train(resume=True)

        run_dir = Path(self._model.trainer.save_dir)
        best_pt = run_dir / "weights" / "best.pt"
        last_pt = run_dir / "weights" / "last.pt"

        if best_pt.exists():
            self._model = RTDETR(str(best_pt))

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
        """Register on_fit_epoch_end to route per-epoch metrics to the tracker."""
        adapter = self

        def on_fit_epoch_end(trainer) -> None:
            epoch = trainer.epoch + 1
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
                adapter._hpo_pruning_callback(epoch, metrics)

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
            self._model = RTDETR(str(weights_path))
            logger.info("Loaded weights from config.resume_from: %s", weights_path)
            return

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
        self._model = RTDETR(str(weights_path))
        logger.info("Loaded weights from model.json: %s", weights_path)

    def _discover_data_yaml(self, config: "ExperimentConfig") -> None:
        """Locate rtdetr.yaml for evaluate/predict without a prior prepare_data() call.

        Priority:
          1. self.work_dir/rtdetr.yaml — present in attach flow
          2. config.resume_experiment work dir — resume-training flow
        """
        direct_candidate = Path(self.work_dir) / "rtdetr.yaml"
        if direct_candidate.exists():
            self._data_yaml_path = str(direct_candidate)
            logger.info("Found data yaml in work_dir: %s", self._data_yaml_path)
            return

        if config.resume_experiment is None:
            raise RuntimeError(
                "rtdetr.yaml not found in work_dir and resume_experiment is not set; "
                "cannot locate original rtdetr.yaml."
            )
        work_dir = Path(config.output_dir) / "experiments" / config.resume_experiment / "work"
        candidate = work_dir / "rtdetr.yaml"
        if candidate.exists():
            self._data_yaml_path = str(candidate)
            logger.info("Found data yaml via resume_experiment: %s", self._data_yaml_path)
        else:
            raise FileNotFoundError(
                f"rtdetr.yaml not found at {candidate}. "
                f"Ensure prepare_data() was run in experiment '{config.resume_experiment}'."
            )
