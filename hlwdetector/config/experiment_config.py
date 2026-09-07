"""ExperimentConfig dataclass with YAML loading and validation."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hlwdetector.paths import REPO_ROOT, to_repo_rel, resolve

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    model_name: str                # "yolo11" | "rtdetr"
    config_name: str
    hyperparameters: dict[str, Any]  # model-specific; adapter interprets

    # Canonical data inputs (COCO JSON + image paths)
    coco_json: str  # coco annotations for all frames
    images_dir: str  # base dir containing extracted video frames

    split_json: str  # json defining train/val/text splits

    model_weights: str | None = None  # weights filename/path; adapter loads unless resuming
    output_dir: str = "outputs"
    random_seed: int = 42

    wandb_project: str | None = None
    wandb_group: str | None = None  # groups related runs in W&B (e.g. all trials of an HPO study)
    resume_experiment_name: str | None = None  # directory NAME of the experiment being resumed
    resume_weights: str | None = None  # path to model weights to load and resume training from

    visualize_split: str = "test"
    visualization_fps: float = 29.0

    # Fields holding filesystem paths — serialized repo-relative, resolved on load.
    # resume_experiment_name is deliberately absent: it is a directory *name*, not a path.
    PATH_FIELDS = ("coco_json", "images_dir", "split_json", "output_dir", "resume_weights")

    @property
    def resume_experiment_dir(self) -> Path | None:
        """Absolute dir of the experiment being resumed, or None if not resuming.

        resume_experiment_name holds a bare directory name (e.g. 'swin_h23_20260805_234436');
        the enclosing <output_dir>/experiments/ is supplied here so every caller builds
        the path the same way.
        """
        if self.resume_experiment_name is None:
            return None
        return (Path(self.output_dir) / "experiments" / self.resume_experiment_name).resolve()

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load config from YAML, resolving all relative paths against the YAML's parent dir."""
        yaml_path = Path(path).resolve()
        #base_dir = yaml_path.parent

        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        def resolve(val: str | None) -> str | None:
            if val is None:
                return None
            p = Path(val)
            if not p.is_absolute():
                p = (REPO_ROOT / p).resolve()
            return str(p)

        # Resolve all path fields
        for key in cls.PATH_FIELDS:
            if key in raw and raw[key] is not None:
                raw[key] = resolve(raw[key])

        return cls(**raw)

    def to_serializable_dict(self) -> dict[str, Any]:
        """Dict for config.json with path fields stored relative to the repo root."""
        data = dataclasses.asdict(self)
        for key in self.PATH_FIELDS:
            if data.get(key) is not None:
                data[key] = to_repo_rel(data[key])
        return data

    @classmethod
    def from_stored_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        """Reconstruct from a stored config.json dict, resolving repo-relative paths.

        Unknown keys (e.g. wandb_run_id, resumed_in) are ignored.
        """
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in raw.items() if k in valid_fields}
        for key in cls.PATH_FIELDS:
            if kwargs.get(key) is not None:
                kwargs[key] = str(resolve(kwargs[key]))
        return cls(**kwargs)

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

        # Check images directory exists
        images_base = Path(self.images_dir)
        if not images_base.exists():
            raise FileNotFoundError(f"images_dir not found: {images_base}")
            
        # Check resume fields are both set or both unset
        if (self.resume_weights is None) != (self.resume_experiment_name is None):
            raise ValueError(
                "resume_weights and resume_experiment_name must both be set or both be unset; "
                f"got resume_weights={self.resume_weights!r}, resume_experiment_name={self.resume_experiment_name!r}"
            )

        # resume_weights names the weights file to load
        if self.resume_weights is not None:
            weights_path = Path(self.resume_weights)
            if not weights_path.is_file():
                raise ValueError(f"resume_weights file not found at {weights_path}")

        # resume_experiment_name names the original experiment dir under <output_dir>/experiments
        if self.resume_experiment_name is not None:
            original_dir = self.resume_experiment_dir
            if not original_dir.is_dir():
                raise ValueError(
                    f"resume_experiment_name directory not found at {original_dir}. "
                    "resume_experiment_name must be an experiment directory *name* "
                    "(e.g. 'swin_h23_20260805_234436'), not a path to weights."
                )

        # Check visualize_split is valid
        if self.visualize_split not in ("train", "val", "test"):
            raise ValueError(
                f"visualize_split must be one of train/val/test, got: {self.visualize_split!r}"
            )
