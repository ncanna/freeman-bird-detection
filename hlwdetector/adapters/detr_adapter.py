"""DETRAdapter — DETR-family object detection via HuggingFace Transformers.

Checkpoint-driven: ``config.model_weights`` names any HF object-detection
checkpoint and ``AutoModelForObjectDetection`` / ``AutoImageProcessor`` pick the
right classes, so switching variants is a YAML edit rather than a new adapter.

The conventions this adapter depends on — who owns resize/normalize, what
``imgsz`` means here, and the COCO ``category_id`` remap — are documented in
CLAUDE.md ("DETR Adapter Constraints") and the README. They are load-bearing:
changing one silently breaks training or misplaces boxes. Read them before
editing this file.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import supervision as sv
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoImageProcessor, AutoModelForObjectDetection

from hlwdetector import paths
from hlwdetector.adapters.base import (
    BaseModelAdapter,
    DetectionResult,
    MetricsDict,
    TrainingResult,
    resolve_device,
)
from hlwdetector.metrics import EVAL_SCORE_THRESHOLD, category_ids, compute_coco_metrics
from hlwdetector.registry import register_adapter

if TYPE_CHECKING:
    from hlwdetector.config import ExperimentConfig
    from hlwdetector.dataset_manager import DatasetManager, SplitView

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "facebook/detr-resnet-50"

# Camera-trap footage is 16:9; the long side follows from the short side so that
# `imgsz` stays the single resolution knob. Matches the swin adapter's max_size.
_ASPECT_RATIO = 16 / 9


# ====================================================================== #
# Helper: COCO-format dataset producing HF image-processor inputs
# ====================================================================== #


class _COCODetectionDataset(Dataset):
    """COCO-format dataset yielding native-resolution images + COCO annotations.

    Returns ``(image, target, stem, (height, width))`` where ``target`` is the
    ``{"image_id": int, "annotations": [...]}`` structure the HF image processor
    expects with ``format="coco_detection"``.
    """

    def __init__(
        self,
        images: list[dict],
        annotations: list[dict],
        images_dir: str,
        cat_id_to_class_idx: dict[int, int],
        train: bool = False,
    ) -> None:
        self._images = images
        self._img_id_to_anns: dict[int, list[dict]] = {}
        for ann in annotations:
            self._img_id_to_anns.setdefault(ann["image_id"], []).append(ann)
        self._images_dir = Path(images_dir)
        self._cat_id_to_class_idx = cat_id_to_class_idx
        self._train = train

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int):
        img_info = self._images[idx]
        img_path = self._images_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        coco_anns = []
        for ann in self._img_id_to_anns.get(img_info["id"], []):
            x, y, w, h = ann["bbox"]  # COCO format: [x, y, width, height]
            if w <= 0 or h <= 0:
                continue
            coco_anns.append(
                {
                    "bbox": [float(x), float(y), float(w), float(h)],
                    # COCO category_id -> contiguous 0-based class index. The raw
                    # id would land on the no-object logit and train the box as
                    # background, silently. See CLAUDE.md, "DETR Adapter Constraints".
                    "category_id": self._cat_id_to_class_idx[ann["category_id"]],
                    # The processor reads both off every annotation and raises
                    # without them; h23 has them, other datasets may not.
                    "area": float(ann.get("area", w * h)),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                }
            )

        target = {"image_id": int(img_info["id"]), "annotations": coco_anns}

        if self._train:
            image, target = _augment(image, target)

        width, height = image.size
        return image, target, Path(img_info["file_name"]).stem, (height, width)


def _augment(image: "Image.Image", target: dict) -> tuple["Image.Image", dict]:
    """Training-time augmentation hook. Identity for now.

    Boxes are native-resolution COCO ``[x, y, w, h]`` at this point, so any
    transform added here must update ``target["annotations"]`` in the same space.
    """
    return image, target


class _DETRCollator:
    """Batches samples through the HF image processor.

    A callable class rather than a function because it needs to carry the
    processor. Returns ``(encoding, image_ids, stems, orig_sizes)``; ``encoding``
    holds ``pixel_values``, ``pixel_mask`` (the processor pads to the largest
    image in the batch) and ``labels`` with normalized cxcywh boxes.
    ``orig_sizes`` feeds ``post_process_object_detection(target_sizes=...)``.
    """

    def __init__(self, processor) -> None:
        self._processor = processor

    def __call__(self, batch):
        images, targets, stems, sizes = zip(*batch)
        encoding = self._processor(
            images=list(images), annotations=list(targets), return_tensors="pt"
        )
        image_ids = [t["image_id"] for t in targets]
        orig_sizes = torch.tensor([[h, w] for h, w in sizes], dtype=torch.int64)
        return encoding, image_ids, list(stems), orig_sizes


# ====================================================================== #
# Helper: build a DETR-family model from any HF checkpoint
# ====================================================================== #


def _build_detr_model(
    model_weights: str,
    num_labels: int,
    id2label: dict[int, str],
    pretrained: bool = True,
):
    """Instantiate a DETR-family detector with a head sized for this dataset."""
    label2id = {name: idx for idx, name in id2label.items()}
    if pretrained:
        return AutoModelForObjectDetection.from_pretrained(
            model_weights,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            # The hub checkpoints ship a 91-class COCO head; swap it for ours.
            ignore_mismatched_sizes=True,
        )
    # Resuming: build the architecture only — every weight comes from the
    # checkpoint, so downloading pretrained weights would be wasted work.
    cfg = AutoConfig.from_pretrained(
        model_weights, num_labels=num_labels, id2label=id2label, label2id=label2id
    )
    return AutoModelForObjectDetection.from_config(cfg)


def _clip_boxes(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    """Clip xyxy boxes to the image. DETR predicts in normalized space and can
    extrapolate slightly past the border after rescaling."""
    if len(boxes) == 0:
        return boxes
    boxes = boxes.copy()
    boxes[:, 0] = boxes[:, 0].clip(0, width)
    boxes[:, 1] = boxes[:, 1].clip(0, height)
    boxes[:, 2] = boxes[:, 2].clip(0, width)
    boxes[:, 3] = boxes[:, 3].clip(0, height)
    return boxes


# ====================================================================== #
# Main adapter class
# ====================================================================== #


@register_adapter("detr")
class DETRAdapter(BaseModelAdapter):
    """DETR-family adapter built on HuggingFace Transformers.

    Hyperparameters (config.hyperparameters):
        epochs:               total epochs — absolute, not additional, on resume
        batch:                batch size (default 4; DETR is memory-heavy)
        imgsz:                short side of the network input (default 640)
        max_size:             long side (default: imgsz * 16/9)
        lr:                   learning rate for transformer + heads (default 1e-4)
        lr_backbone:          learning rate for the CNN backbone (default 1e-5)
        weight_decay:         AdamW weight decay (default 1e-4)
        lr_step_size:         StepLR step size in epochs (default 40)
        lr_gamma:             StepLR decay factor (default 0.1)
        clip_max_norm:        gradient clipping (default 0.1, the DETR recipe)
        score_threshold:      min confidence for predict()/visualization (default 0.5)
        eval_score_threshold: min confidence for metrics (default 0.001 — keep low,
                              a high value truncates the PR curve and understates AP)
        device:               device string ("0", "cuda", "mps", "cpu", "auto")
        num_workers:          DataLoader workers (default 4)

    Internal state is preserved across sequential calls:
        prepare_data → train → evaluate → predict
    """

    def __init__(self, artifact_manager, tracker) -> None:
        super().__init__(artifact_manager, tracker)
        self._model = None
        self._processor = None
        self._collator: _DETRCollator | None = None
        self._device: torch.device | None = None
        self._train_dataset: _COCODetectionDataset | None = None
        self._val_dataset: _COCODetectionDataset | None = None
        self._test_dataset: _COCODetectionDataset | None = None
        self._val_split: "SplitView | None" = None
        self._test_split: "SplitView | None" = None
        self._cat_id_to_class_idx: dict[int, int] | None = None
        self._id2label: dict[int, str] | None = None
        self._num_labels: int | None = None
        self._training_result: TrainingResult | None = None
        # Set by HPOptimizer when a study is running; see hp_optimizer.py.
        self._hpo_pruning_callback = None

    # ------------------------------------------------------------------ #
    # prepare_data
    # ------------------------------------------------------------------ #

    def prepare_data(
        self,
        dataset_manager: "DatasetManager",
        config: "ExperimentConfig",
    ) -> None:
        """Build PyTorch datasets from COCO annotations for each split.

        Writes nothing to disk and is safe to re-run — ``ExperimentRunner
        .from_experiment_dir`` calls it again on attach.
        """
        self._ensure_processor(config)
        images_dir = config.images_dir

        for split_name in ("train", "val", "test"):
            split_view = dataset_manager.get_split(split_name)

            missing = [p for p in split_view.image_paths if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Split '{split_name}': {len(missing)}/{len(split_view.image_paths)} "
                    f"image files missing. First missing: {missing[0]}"
                )

            # Class space is derived from the dataset, not hardcoded.
            if self._cat_id_to_class_idx is None:
                self._init_class_mapping(split_view)

            dataset = _COCODetectionDataset(
                images=split_view.images,
                annotations=split_view.annotations,
                images_dir=images_dir,
                cat_id_to_class_idx=self._cat_id_to_class_idx,
                train=(split_name == "train"),
            )

            if split_name == "train":
                self._train_dataset = dataset
            elif split_name == "val":
                self._val_dataset = dataset
                self._val_split = split_view
            else:
                self._test_dataset = dataset
                self._test_split = split_view

        logger.info(
            "DETR data prepared: train=%d, val=%d, test=%d images (%d class(es): %s)",
            len(self._train_dataset),
            len(self._val_dataset),
            len(self._test_dataset),
            self._num_labels,
            ", ".join(self._id2label.values()),
        )

    # ------------------------------------------------------------------ #
    # train
    # ------------------------------------------------------------------ #

    def train(self, config: "ExperimentConfig") -> TrainingResult:
        """Fine-tune a DETR-family detector and return TrainingResult."""
        self._ensure_datasets(config)

        hp = config.hyperparameters
        epochs = hp.get("epochs", 50)
        batch = hp.get("batch", 4)
        lr = hp.get("lr", 1e-4)
        lr_backbone = hp.get("lr_backbone", 1e-5)
        weight_decay = hp.get("weight_decay", 1e-4)
        lr_step_size = hp.get("lr_step_size", 40)
        lr_gamma = hp.get("lr_gamma", 0.1)
        clip_max_norm = hp.get("clip_max_norm", 0.1)
        num_workers = hp.get("num_workers", 4)

        self._device = resolve_device(hp.get("device", "auto"))
        logger.info("Training on device: %s", self._device)

        checkpoint = None
        if config.resume_weights is not None:
            self._model = _build_detr_model(
                self._checkpoint_id(config), self._num_labels, self._id2label, pretrained=False
            )
            checkpoint = torch.load(
                config.resume_weights, map_location=self._device, weights_only=False
            )
            self._model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Resumed from checkpoint: %s", config.resume_weights)
        else:
            self._model = _build_detr_model(
                self._checkpoint_id(config), self._num_labels, self._id2label, pretrained=True
            )

        self._model.to(self._device)

        # Two param groups: the pretrained CNN backbone wants a 10x smaller LR
        # than the freshly-initialised transformer and heads (the DETR recipe).
        backbone_params, head_params = [], []
        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                continue
            (backbone_params if "backbone" in name else head_params).append(param)
        if not backbone_params:
            logger.warning(
                "No parameters matched 'backbone'; all %d params train at lr=%s",
                len(head_params), lr,
            )
        params = head_params + backbone_params
        optimizer = torch.optim.AdamW(
            [
                {"params": head_params, "lr": lr},
                {"params": backbone_params, "lr": lr_backbone},
            ],
            weight_decay=weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step_size, gamma=lr_gamma
        )

        # Resume optimizer state if available. The presence of the resume fields is
        # itself the signal to restore state — there is no separate opt-in flag.
        start_epoch = 0
        if checkpoint is not None:
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"]
            if "scheduler_state_dict" in checkpoint:
                lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("Resuming training from epoch %d", start_epoch)

            # `epochs` is an absolute target, not a count of additional epochs: the
            # checkpoint stores epochs *completed*, and the loop below runs
            # range(start_epoch, epochs). Without this the loop is a silent no-op.
            if start_epoch >= epochs:
                raise ValueError(
                    f"Nothing to train: checkpoint {config.resume_weights} already completed "
                    f"{start_epoch} epochs and hyperparameters.epochs is {epochs}. "
                    "'epochs' counts total epochs across the resumed run, so set it "
                    f"above {start_epoch} to continue training."
                )

        train_loader = DataLoader(
            self._train_dataset,
            batch_size=batch,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self._collator,
            pin_memory=(self._device.type == "cuda"),
        )

        run_dir = Path(self.work_dir) / "runs" / "train"
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        # Carry the best score across the resume boundary: a resumed run is a
        # continuation, so an epoch is only "best" if it beats the whole history.
        # The resumed checkpoint seeds best.pt so best_weights_path is populated
        # even when no new epoch improves on it.
        best_metric = -float("inf")
        best_epoch = -1
        if checkpoint is not None:
            best_metric = checkpoint.get("best_metric", -float("inf"))
            if best_metric is None or not np.isfinite(best_metric):
                best_metric = -float("inf")
            else:
                shutil.copy2(config.resume_weights, weights_dir / "best.pt")
                logger.info(
                    "Carrying best val mAP50-95=%.4f over from the resumed checkpoint",
                    best_metric,
                )
        avg_loss = float("nan")
        current_lr = optimizer.param_groups[0]["lr"]

        for epoch in range(start_epoch, epochs):
            self._model.train()
            epoch_loss = 0.0
            component_totals: dict[str, float] = {}
            num_batches = 0

            for encoding, _image_ids, _stems, _orig_sizes in train_loader:
                labels = [
                    {k: v.to(self._device) for k, v in target.items()}
                    for target in encoding["labels"]
                ]
                outputs = self._model(
                    pixel_values=encoding["pixel_values"].to(self._device),
                    pixel_mask=encoding["pixel_mask"].to(self._device),
                    labels=labels,
                )
                loss = outputs.loss  # already the weighted sum of the components

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=clip_max_norm)
                optimizer.step()

                epoch_loss += loss.item()
                # Accumulate every component so the logged value is an epoch mean,
                # not whatever the last batch happened to produce.
                for key, value in (outputs.loss_dict or {}).items():
                    # detach first: these still carry grad_fn, and float() on a
                    # grad-requiring tensor warns.
                    component_totals[key] = (
                        component_totals.get(key, 0.0) + value.detach().item()
                    )
                num_batches += 1

            lr_scheduler.step()

            denom = max(num_batches, 1)
            avg_loss = epoch_loss / denom
            current_lr = optimizer.param_groups[0]["lr"]

            epoch_metrics = {
                "train/loss": avg_loss,
                "train/lr": current_lr,
                "train/lr_backbone": optimizer.param_groups[1]["lr"],
            }
            for key, total in component_totals.items():
                epoch_metrics[f"train/{key}"] = total / denom

            # Validate every epoch: best.pt is selected on val mAP50-95, and this
            # is the per-epoch metric stream a future HPO pruner would read.
            val_metrics = self._run_validation(config)
            epoch_metrics.update(
                {
                    "val/precision": val_metrics.precision,
                    "val/recall": val_metrics.recall,
                    "val/f1": val_metrics.f1,
                    "val/mAP50": val_metrics.map50,
                    "val/mAP50_95": val_metrics.map50_95,
                }
            )

            self.log_epoch(epoch + 1, epoch_metrics)
            if self._tracker is not None:
                self._tracker.log_wandb_step(epoch_metrics, step=epoch + 1)
            if self._hpo_pruning_callback is not None:
                self._hpo_pruning_callback(epoch + 1, epoch_metrics)

            logger.info(
                "Epoch %d/%d — loss: %.4f, lr: %.6f, val mAP50-95: %.4f, val mAP50: %.4f",
                epoch + 1, epochs, avg_loss, current_lr,
                val_metrics.map50_95, val_metrics.map50,
            )

            checkpoint_data = {
                "epoch": epoch + 1,  # epochs COMPLETED
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "loss": avg_loss,
                "best_metric": best_metric,
                # Enough to rebuild the architecture without the experiment config.
                "model_weights": self._checkpoint_id(config),
                "id2label": self._id2label,
            }
            torch.save(checkpoint_data, weights_dir / "last.pt")

            if val_metrics.map50_95 > best_metric:
                best_metric = val_metrics.map50_95
                best_epoch = epoch + 1
                checkpoint_data["best_metric"] = best_metric
                torch.save(checkpoint_data, weights_dir / "best.pt")
                logger.info(
                    "New best model at epoch %d (val mAP50-95=%.4f)", best_epoch, best_metric
                )

        # Load best weights for subsequent evaluate/predict
        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"
        if best_pt.exists():
            best_ckpt = torch.load(best_pt, map_location=self._device, weights_only=False)
            self._model.load_state_dict(best_ckpt["model_state_dict"])

        self._training_result = TrainingResult(
            run_dir=str(run_dir),
            best_weights_path=str(best_pt) if best_pt.exists() else None,
            last_weights_path=str(last_pt) if last_pt.exists() else None,
            training_metrics={
                "best_map50_95": best_metric if np.isfinite(best_metric) else None,
                # -1 means no epoch in this run beat the checkpoint it resumed from.
                "best_epoch": best_epoch,
                "final_loss": avg_loss,
                "final_lr": current_lr,
            },
        )
        return self._training_result

    # ------------------------------------------------------------------ #
    # evaluate
    # ------------------------------------------------------------------ #

    def evaluate(self, config: "ExperimentConfig") -> MetricsDict:
        """Evaluate on the val split with pycocotools COCOeval."""
        self._ensure_datasets(config)
        if self._model is None:
            self._load_model_from_artifacts(config)
        return self._run_validation(config)

    # ------------------------------------------------------------------ #
    # predict
    # ------------------------------------------------------------------ #

    def predict(self, config: "ExperimentConfig") -> DetectionResult:
        """Run inference on test images; return per-frame sv.Detections."""
        self._ensure_datasets(config)
        if self._model is None:
            self._load_model_from_artifacts(config)

        hp = config.hyperparameters
        score_threshold = hp.get("score_threshold", 0.5)

        predictions: DetectionResult = {}
        for _image_ids, stems, sizes, results in self._infer(
            self._test_dataset, config, score_threshold
        ):
            for stem, (height, width), result in zip(stems, sizes, results):
                boxes = result["boxes"].cpu().numpy().astype(np.float32)
                boxes = _clip_boxes(boxes, height, width)
                if len(boxes) == 0:
                    # Still emit a key: VisualizationPipeline looks up every frame.
                    predictions[stem] = sv.Detections.empty()
                    continue
                predictions[stem] = sv.Detections(
                    xyxy=boxes,
                    confidence=result["scores"].cpu().numpy().astype(np.float32),
                    # Already 0-based — the dataset remapped category_id on the way in.
                    class_id=result["labels"].cpu().numpy().astype(int),
                )

        logger.info("DETR predictions for %d test frames", len(predictions))
        return predictions

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _init_class_mapping(self, split_view: "SplitView") -> None:
        """Derive the model's class space from the dataset's COCO categories."""
        ordered = sorted(split_view.categories, key=lambda c: c["id"])
        self._cat_id_to_class_idx = {cat["id"]: idx for idx, cat in enumerate(ordered)}
        self._id2label = {idx: cat["name"] for idx, cat in enumerate(ordered)}
        self._num_labels = len(ordered)

    def _checkpoint_id(self, config: "ExperimentConfig") -> str:
        return config.model_weights or DEFAULT_CHECKPOINT

    def _ensure_processor(self, config: "ExperimentConfig") -> None:
        """Build the image processor if it does not exist yet."""
        if self._processor is not None:
            return
        hp = config.hyperparameters
        imgsz = hp.get("imgsz", 640)
        max_size = hp.get("max_size", int(round(imgsz * _ASPECT_RATIO)))
        self._processor = AutoImageProcessor.from_pretrained(
            self._checkpoint_id(config),
            size={"shortest_edge": imgsz, "longest_edge": max_size},
        )
        self._collator = _DETRCollator(self._processor)
        logger.info(
            "Image processor from %s (short side %d, long side %d)",
            self._checkpoint_id(config), imgsz, max_size,
        )

    def _ensure_datasets(self, config: "ExperimentConfig") -> None:
        """Rebuild datasets if prepare_data has not run in this process.

        ``ExperimentRunner.train()`` skips ``prepare_data`` entirely when
        ``resume_weights`` is set, so a resumed run would otherwise reach the
        per-epoch validation pass with ``self._val_dataset is None``.
        """
        self._ensure_processor(config)
        if self._train_dataset is not None:
            return
        from hlwdetector.dataset_manager import DatasetManager

        logger.info("Datasets not prepared in this process; rebuilding from config")
        self.prepare_data(DatasetManager(config), config)

    def _infer(self, dataset, config: "ExperimentConfig", threshold: float):
        """Yield ``(image_ids, stems, sizes, results)`` per batch.

        Identifiers come from the collator alongside the samples they describe, so
        predictions cannot drift out of alignment with their source image.
        """
        hp = config.hyperparameters
        batch = hp.get("batch", 4)
        num_workers = hp.get("num_workers", 4)

        if self._device is None:
            self._device = resolve_device(hp.get("device", "auto"))
        self._model.to(self._device)
        self._model.eval()

        loader = DataLoader(
            dataset,
            batch_size=batch,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._collator,
            pin_memory=(self._device.type == "cuda"),
        )

        with torch.inference_mode():
            for encoding, image_ids, stems, orig_sizes in loader:
                outputs = self._model(
                    pixel_values=encoding["pixel_values"].to(self._device),
                    pixel_mask=encoding["pixel_mask"].to(self._device),
                )
                # target_sizes un-scales predictions from the network's resized
                # input back to native image coordinates.
                results = self._processor.post_process_object_detection(
                    outputs, threshold=threshold, target_sizes=orig_sizes
                )
                yield image_ids, stems, orig_sizes.tolist(), results

    def _run_validation(self, config: "ExperimentConfig") -> MetricsDict:
        """Score the val split with COCOeval. Shared by train() and evaluate()."""
        threshold = config.hyperparameters.get("eval_score_threshold", EVAL_SCORE_THRESHOLD)
        cat_ids = category_ids(self._val_split)

        detections: list[dict] = []
        for image_ids, _stems, sizes, results in self._infer(
            self._val_dataset, config, threshold
        ):
            for image_id, (height, width), result in zip(image_ids, sizes, results):
                boxes = _clip_boxes(result["boxes"].cpu().numpy(), height, width)
                scores = result["scores"].cpu().numpy()
                labels = result["labels"].cpu().numpy().astype(int)
                for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
                    detections.append(
                        {
                            "image_id": int(image_id),
                            "category_id": cat_ids[label],
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(score),
                        }
                    )

        return compute_coco_metrics(self._val_split, detections)

    def _load_model_from_artifacts(self, config: "ExperimentConfig") -> None:
        """Load weights for evaluate/predict without a prior train() call."""
        hp = config.hyperparameters
        self._device = resolve_device(hp.get("device", "auto"))

        if config.resume_weights is not None:
            weights_path = Path(config.resume_weights)
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            self._load_checkpoint(weights_path, config)
            logger.info("Loaded weights from config.resume_weights: %s", weights_path)
            return

        # Attach flow: read best_weights_path from model.json
        model_json_path = Path(self.experiment_dir) / "model.json"
        if not model_json_path.exists():
            raise FileNotFoundError(
                f"No model loaded, resume_weights is not set, and model.json not found in: "
                f"{self.experiment_dir}"
            )
        model_info = json.loads(model_json_path.read_text())
        weights_path_str = model_info.get("best_weights_path")
        if not weights_path_str:
            raise RuntimeError(f"model.json in {self.experiment_dir} has no best_weights_path")
        weights_path = paths.resolve(weights_path_str)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"best_weights_path from model.json does not exist: {weights_path}"
            )
        self._load_checkpoint(weights_path, config)
        logger.info("Loaded weights from model.json: %s", weights_path)

    def _load_checkpoint(self, weights_path: Path, config: "ExperimentConfig") -> None:
        """Rebuild the architecture from a checkpoint's metadata and load it."""
        checkpoint = torch.load(str(weights_path), map_location=self._device, weights_only=False)
        # Checkpoints carry their own class space so they stand alone; fall back to
        # whatever prepare_data derived for checkpoints written before that.
        id2label = checkpoint.get("id2label") or self._id2label
        if id2label is None:
            raise RuntimeError(
                f"Checkpoint {weights_path} has no id2label and no dataset has been "
                "prepared; cannot determine the number of classes."
            )
        id2label = {int(k): v for k, v in id2label.items()}
        self._id2label = id2label
        self._num_labels = len(id2label)
        self._model = _build_detr_model(
            checkpoint.get("model_weights") or self._checkpoint_id(config),
            self._num_labels,
            id2label,
            pretrained=False,
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self._device)
