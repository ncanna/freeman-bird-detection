"""ExperimentConfig dataclass with YAML loading and validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    model_name: str                # "yolo11" | "megadetector"
    experiment_name: str
    hyperparameters: dict[str, Any]  # model-specific; adapter interprets

    # Canonical data inputs (COCO JSON + image paths)
    coco_json: str  # coco annotations for all frames
    images_dir: str  # base dir containing train/ val/ test/ subdirs

    split_json: str  # json defining train/val/text splits

    output_dir: str = "outputs"
    random_seed: int = 42

    wandb_project: str | None = None
    wandb_run_name: str | None = None
    resume_from: str | None = None  # path to existing experiment dir; skips to evaluate/predict

    visualize: bool = False
    visualize_split: str = "test"
    visualization_fps: float = 29.0

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load config from YAML, resolving all relative paths against the YAML's parent dir."""
        yaml_path = Path(path).resolve()
        base_dir = yaml_path.parent

        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        def resolve(val: str | None) -> str | None:
            if val is None:
                return None
            p = Path(val)
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            return str(p)

        # Resolve all path fields
        for key in ("coco_json", "images_dir", "split_json", "output_dir", "resume_from"):
            if key in raw and raw[key] is not None:
                raw[key] = resolve(raw[key])

        return cls(**raw)

    def validate(self) -> None:
        """Raise clear errors if prerequisites are missing."""
        from hlwdetector.registry import get_adapter  # avoid circular import at module level

        # Check model_name is registered
        get_adapter(self.model_name)  # raises KeyError with helpful message if unknown

        # Check dataset JSON files exist
        for attr in ("coco_json", "split_json"):
            p = Path(getattr(self, attr))
            if not p.exists():
                raise FileNotFoundError(
                    f"File not found: {p}\n"
                    f"Ensure COCO annotation and split JSON files exist"
                )

        # Check images split subdirectories exist
        images_base = Path(self.images_dir)
        for split in ("train", "val", "test"):
            split_dir = images_base / split
            if not split_dir.exists():
                raise FileNotFoundError(f"Extracted frames not found at {split_dir}")

        # Check visualize_split is valid
        if self.visualize_split not in ("train", "val", "test"):
            raise ValueError(
                f"visualize_split must be one of train/val/test, got: {self.visualize_split!r}"
            )
