"""SwinAdapter — Swin Transformer backbone + Faster R-CNN detection head.

Uses timm for the Swin-T/S/B backbone (with multi-scale feature extraction)
and torchvision's FasterRCNN detection head with FPN.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import supervision as sv
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

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

_PROJECT_ROOT = str(paths.REPO_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ====================================================================== #
# Helper: COCO-format dataset for PyTorch detection training
# ====================================================================== #


class _COCODetectionDataset(Dataset):
    """Minimal COCO-format dataset returning images + targets for Faster R-CNN.

    Images and boxes stay at native resolution; the detector's own transform
    handles resize + normalization.
    """

    def __init__(
        self,
        images: list[dict],
        annotations: list[dict],
        images_dir: str,
        transform: Callable | None = None,
    ) -> None:
        self._images = images
        self._img_id_to_anns: dict[int, list[dict]] = {}
        for ann in annotations:
            self._img_id_to_anns.setdefault(ann["image_id"], []).append(ann)
        self._images_dir = Path(images_dir)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int):
        img_info = self._images[idx]
        img_path = self._images_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        anns = self._img_id_to_anns.get(img_info["id"], [])

        boxes = []
        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO format: [x, y, width, height]
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])

        if boxes:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),  # class 1 = bird
            "image_id": torch.tensor([img_info["id"]]),
        }

        if self._transform:
            image = self._transform(image)

        return image, target


# ====================================================================== #
# Helper: Swin backbone wrapper for torchvision FasterRCNN
# ====================================================================== #


class _SwinBackboneWithFPN(nn.Module):
    """Wraps a timm Swin model (features_only=True) with torchvision FPN.

    Produces the `out_channels` attribute that FasterRCNN expects from a backbone.
    """

    def __init__(self, backbone_name: str = "swin_tiny_patch4_window7_224", pretrained: bool = True):
        super().__init__()
        self.body = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            # Accept the variable, non-square batches Faster R-CNN's transform emits
            strict_img_size=False,
        )
        # timm exposes channel dims for each stage
        in_channels_list = self.body.feature_info.channels()  # e.g. [96, 192, 384, 768]
        self._stage_channels = list(in_channels_list)
        self.out_channels = 256

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.body(x)
        # FPN expects an OrderedDict of NCHW; timm Swin returns a list of NHWC
        feat_dict = {}
        for i, f in enumerate(features):
            expected_c = self._stage_channels[i]
            if f.shape[-1] == expected_c and f.shape[1] != expected_c:
                f = f.permute(0, 3, 1, 2).contiguous()
            feat_dict[str(i)] = f
        return self.fpn(feat_dict)


# ====================================================================== #
# Helper: Build the full detection model
# ====================================================================== #


def _build_swin_fasterrcnn(
    backbone_name: str = "swin_tiny_patch4_window7_224",
    num_classes: int = 2,  # background + bird
    pretrained_backbone: bool = True,
    imgsz: int = 640,
) -> FasterRCNN:
    """Construct Faster R-CNN with a Swin Transformer backbone + FPN.

    `imgsz` sets the short side of the network input; aspect ratio is preserved.
    """
    backbone = _SwinBackboneWithFPN(backbone_name=backbone_name, pretrained=pretrained_backbone)

    # Anchor generator tuned for typical bird sizes in camera trap frames
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        # Owns resize + normalization; rescales GT boxes and un-scales predictions
        min_size=imgsz,
        max_size=int(round(imgsz * 16 / 9)),
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        # Use default ROI pooler settings (7x7, adaptive)
    )
    return model


# ====================================================================== #
# Helper: Collate function for variable-size targets
# ====================================================================== #


def _collate_fn(batch):
    """Custom collate that keeps targets as a list of dicts (Faster R-CNN requirement)."""
    return tuple(zip(*batch))


# ====================================================================== #
# Helper: Select best available device
# ====================================================================== #


# Lives in adapters.base so the detr adapter can share it; kept under the old
# private name here so the rest of this module reads unchanged.
_resolve_device = resolve_device


# ====================================================================== #
# Main adapter class
# ====================================================================== #


@register_adapter("swin")
class SwinAdapter(BaseModelAdapter):
    """Swin Transformer + Faster R-CNN adapter using timm + torchvision.

    Hyperparameters (config.hyperparameters):
        backbone:       timm model name (default: swin_tiny_patch4_window7_224)
        epochs:         number of training epochs
        imgsz:          short side of the network input, aspect preserved (default 640)
        batch:          batch size (default 8; SWIN is memory-heavy)
        device:         device string ("0", "cuda", "mps", "cpu", "auto")
        lr:             learning rate (default 0.0001)
        weight_decay:   AdamW weight decay (default 0.05)
        lr_step_size:   StepLR step size in epochs (default 20)
        lr_gamma:       StepLR decay factor (default 0.1)
        score_threshold: min confidence for predict()/visualization (default 0.5)
        eval_score_threshold: min confidence for metrics (default 0.001 — keep low,
                        a high value truncates the PR curve and understates AP)

    Internal state is preserved across sequential calls:
        prepare_data → train → evaluate → predict
    """

    def __init__(self, artifact_manager, tracker) -> None:
        super().__init__(artifact_manager, tracker)
        self._model: FasterRCNN | None = None
        self._device: torch.device | None = None
        self._train_dataset: _COCODetectionDataset | None = None
        self._val_dataset: _COCODetectionDataset | None = None
        self._test_dataset: _COCODetectionDataset | None = None
        self._test_images: list[dict] | None = None
        self._val_split: "SplitView | None" = None
        self._training_result: TrainingResult | None = None
        self._transform: Callable | None = None

    # ------------------------------------------------------------------ #
    # prepare_data
    # ------------------------------------------------------------------ #

    def prepare_data(
        self,
        dataset_manager: "DatasetManager",
        config: "ExperimentConfig",
    ) -> None:
        """Build PyTorch datasets from COCO annotations for each split."""
        # Resize + normalization belong to the model's transform, not here: doing
        # them on the image alone would leave the COCO boxes in native coordinates.
        self._transform = transforms.ToTensor()

        images_dir = config.images_dir

        for split_name in ("train", "val", "test"):
            split_view = dataset_manager.get_split(split_name)

            # Validate images exist
            missing = [p for p in split_view.image_paths if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Split '{split_name}': {len(missing)}/{len(split_view.image_paths)} "
                    f"image files missing. First missing: {missing[0]}"
                )

            dataset = _COCODetectionDataset(
                images=split_view.images,
                annotations=split_view.annotations,
                images_dir=images_dir,
                transform=self._transform,
            )

            if split_name == "train":
                self._train_dataset = dataset
            elif split_name == "val":
                self._val_dataset = dataset
                self._val_split = split_view  # ground truth for COCOeval
            else:
                self._test_dataset = dataset
                self._test_images = split_view.images

        logger.info(
            "SWIN data prepared: train=%d, val=%d, test=%d images",
            len(self._train_dataset),
            len(self._val_dataset),
            len(self._test_dataset),
        )

    # ------------------------------------------------------------------ #
    # train
    # ------------------------------------------------------------------ #

    def train(self, config: "ExperimentConfig") -> TrainingResult:
        """Fine-tune Swin + Faster R-CNN and return TrainingResult."""
        if self._train_dataset is None and config.resume_weights is None:
            raise RuntimeError("Call prepare_data() before train().")

        hp = config.hyperparameters
        backbone_name = hp.get("backbone", "swin_tiny_patch4_window7_224")
        epochs = hp.get("epochs", 50)
        batch = hp.get("batch", 8)
        lr = hp.get("lr", 0.0001)
        weight_decay = hp.get("weight_decay", 0.05)
        lr_step_size = hp.get("lr_step_size", 20)
        lr_gamma = hp.get("lr_gamma", 0.1)
        device_str = hp.get("device", "auto")
        imgsz = hp.get("imgsz", 640)

        self._device = _resolve_device(device_str)
        logger.info("Training on device: %s", self._device)

        # Build model
        if config.resume_weights is not None:
            # Load from checkpoint
            self._model = _build_swin_fasterrcnn(
                backbone_name=backbone_name, pretrained_backbone=False, imgsz=imgsz
            )
            checkpoint = torch.load(config.resume_weights, map_location=self._device, weights_only=False)
            self._model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Resumed from checkpoint: %s", config.resume_weights)
        else:
            self._model = _build_swin_fasterrcnn(
                backbone_name=backbone_name, pretrained_backbone=True, imgsz=imgsz
            )

        self._model.to(self._device)

        # Optimizer and scheduler
        params = [p for p in self._model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step_size, gamma=lr_gamma
        )

        # Resume optimizer state if available. The presence of the resume fields is
        # itself the signal to restore state — there is no separate opt-in flag.
        start_epoch = 0
        if config.resume_weights is not None:
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

        # DataLoader
        train_loader = DataLoader(
            self._train_dataset,
            batch_size=batch,
            shuffle=True,
            num_workers=4,
            collate_fn=_collate_fn,
            pin_memory=(self._device.type == "cuda"),
        )

        # Training loop
        run_dir = Path(self.work_dir) / "runs" / "train"
        run_dir.mkdir(parents=True, exist_ok=True)
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        best_loss = float("inf")
        best_epoch = -1
        current_lr = optimizer.param_groups[0]["lr"]  # defined even if the loop body never runs

        for epoch in range(start_epoch, epochs):
            self._model.train()
            epoch_loss = 0.0
            num_batches = 0

            for images, targets in train_loader:
                images = [img.to(self._device) for img in images]
                targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]

                loss_dict = self._model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                # Gradient clipping for stability with transformers
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                epoch_loss += losses.item()
                num_batches += 1

            lr_scheduler.step()

            avg_loss = epoch_loss / max(num_batches, 1)
            current_lr = optimizer.param_groups[0]["lr"]

            # Log epoch metrics
            epoch_metrics = {
                "train/loss": avg_loss,
                "train/lr": current_lr,
            }
            # Add individual loss components
            for k, v in loss_dict.items():
                epoch_metrics[f"train/{k}"] = v.item()

            self.log_epoch(epoch + 1, epoch_metrics)
            # Also push to W&B step tracking
            if self._tracker is not None:
                self._tracker.log_wandb_step(epoch_metrics, step=epoch + 1)

            logger.info(
                "Epoch %d/%d — loss: %.4f, lr: %.6f",
                epoch + 1, epochs, avg_loss, current_lr,
            )

            # Save checkpoint
            checkpoint_data = {
                "epoch": epoch + 1,
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "loss": avg_loss,
            }

            # Save last checkpoint every epoch
            last_path = weights_dir / "last.pt"
            torch.save(checkpoint_data, last_path)

            # Save best checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch + 1
                best_path = weights_dir / "best.pt"
                torch.save(checkpoint_data, best_path)
                logger.info("New best model at epoch %d (loss=%.4f)", best_epoch, best_loss)

        # Load best weights for subsequent evaluate/predict
        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"
        if best_pt.exists():
            best_ckpt = torch.load(best_pt, map_location=self._device, weights_only=False)
            self._model.load_state_dict(best_ckpt["model_state_dict"])

        training_metrics = {
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "final_lr": current_lr,
        }

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
        """Evaluate on val split with pycocotools COCOeval."""
        if self._model is None:
            self._load_model_from_artifacts(config)

        hp = config.hyperparameters
        # Metrics use their own low threshold, not the visualization one: AP is the
        # area under the full PR curve, and pre-filtering at 0.5 truncates it.
        threshold = hp.get("eval_score_threshold", EVAL_SCORE_THRESHOLD)
        batch = hp.get("batch", 8)

        self._model.to(self._device)
        self._model.eval()

        val_loader = DataLoader(
            self._val_dataset,
            batch_size=batch,
            shuffle=False,
            num_workers=4,
            collate_fn=_collate_fn,
            pin_memory=(self._device.type == "cuda"),
        )

        cat_ids = category_ids(self._val_split)
        detections: list[dict] = []

        with torch.inference_mode():
            for images, targets in val_loader:
                images = [img.to(self._device) for img in images]
                outputs = self._model(images)

                for output, target in zip(outputs, targets):
                    image_id = int(target["image_id"].item())
                    keep = output["scores"] >= threshold
                    boxes = output["boxes"][keep].cpu().numpy()
                    scores = output["scores"][keep].cpu().numpy()
                    labels = output["labels"][keep].cpu().numpy().astype(int)

                    for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
                        detections.append(
                            {
                                "image_id": image_id,
                                # torchvision labels are 1-based with 0 = background.
                                "category_id": cat_ids[label - 1],
                                "bbox": [
                                    float(x1), float(y1), float(x2 - x1), float(y2 - y1)
                                ],
                                "score": float(score),
                            }
                        )

        return compute_coco_metrics(self._val_split, detections)

    # ------------------------------------------------------------------ #
    # predict
    # ------------------------------------------------------------------ #

    def predict(self, config: "ExperimentConfig") -> DetectionResult:
        """Run inference on test images; return per-frame sv.Detections."""
        if self._model is None:
            self._load_model_from_artifacts(config)

        hp = config.hyperparameters
        score_threshold = hp.get("score_threshold", 0.5)
        batch = hp.get("batch", 8)

        self._model.to(self._device)
        self._model.eval()

        test_loader = DataLoader(
            self._test_dataset,
            batch_size=batch,
            shuffle=False,
            num_workers=4,
            collate_fn=_collate_fn,
            pin_memory=(self._device.type == "cuda"),
        )

        predictions: DetectionResult = {}
        img_idx = 0

        with torch.inference_mode():
            for images, _ in test_loader:
                images = [img.to(self._device) for img in images]
                outputs = self._model(images)

                for output in outputs:
                    keep = output["scores"] >= score_threshold
                    boxes = output["boxes"][keep].cpu().numpy()
                    scores = output["scores"][keep].cpu().numpy()
                    labels = output["labels"][keep].cpu().numpy().astype(int)

                    # Map label 1 (bird) → class_id 0 for consistency with other adapters
                    class_ids = np.zeros_like(labels)

                    stem = Path(self._test_images[img_idx]["file_name"]).stem
                    if len(boxes) > 0:
                        predictions[stem] = sv.Detections(
                            xyxy=boxes,
                            confidence=scores,
                            class_id=class_ids,
                        )
                    else:
                        predictions[stem] = sv.Detections.empty()

                    img_idx += 1

        return predictions

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_model_from_artifacts(self, config: "ExperimentConfig") -> None:
        """Load weights for evaluate/predict without a prior train() call."""
        hp = config.hyperparameters
        backbone_name = hp.get("backbone", "swin_tiny_patch4_window7_224")
        device_str = hp.get("device", "auto")
        imgsz = hp.get("imgsz", 640)
        self._device = _resolve_device(device_str)

        if config.resume_weights is not None:
            weights_path = Path(config.resume_weights)
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            self._model = _build_swin_fasterrcnn(
                backbone_name=backbone_name, pretrained_backbone=False, imgsz=imgsz
            )
            checkpoint = torch.load(str(weights_path), map_location=self._device, weights_only=False)
            self._model.load_state_dict(checkpoint["model_state_dict"])
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
            raise RuntimeError(
                f"model.json in {self.experiment_dir} has no best_weights_path"
            )
        weights_path = paths.resolve(weights_path_str)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"best_weights_path from model.json does not exist: {weights_path}"
            )
        self._model = _build_swin_fasterrcnn(
            backbone_name=backbone_name, pretrained_backbone=False, imgsz=imgsz
        )
        checkpoint = torch.load(str(weights_path), map_location=self._device, weights_only=False)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded weights from model.json: %s", weights_path)
