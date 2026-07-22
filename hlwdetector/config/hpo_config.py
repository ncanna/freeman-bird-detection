"""HPOConfig dataclass with YAML loading and validation."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from hlwdetector.paths import REPO_ROOT, to_repo_rel, resolve

logger = logging.getLogger(__name__)

# Evaluation metrics that may be optimized (keys of adapters.base.MetricsDict).
VALID_METRICS = ("precision", "recall", "f1", "map50", "map50_95")

# All categories except "static" make up the Optuna study search space
HPARAM_CATEGORIES = ("static", "categorical", "int", "float")


@dataclass
class StudyArgs:
    """kwargs passed into ``optuna.create_study``."""
    study_name: str                  # used to name generated experiment configs
    direction: str = "maximize"      # "maximize" | "minimize"
    sampler: str = "TPE"             # "TPE" | "Random" | "Grid" | "CmaEs"
    pruner: str = "Hyperband"        # "Median" | "None" | "Hyperband" | ...
    storage: str | None = None       # Optuna storage URL (e.g. sqlite:///hpo.db); NOT a filesystem path


@dataclass
class OptimizeArgs:
    """kwargs passed into ``optuna.study.optimize``."""
    n_trials: int = 20
    timeout: int | None = None       # seconds; None = no wall-clock limit


@dataclass
class HPOConfig:
    """Configuration for an Optuna hyperparameter optimization study.

    Holds the base experiment inputs shared by every trial, the Optuna study
    parameters, and the model-specific hyperparameter search space. The search
    space is stored opaquely as ``hyperparameters`` (a dict with keys that align
    with Optuna's search space features categorical/int/float); ``HPOptimizer`` 
    interprets those keys into ``trial.suggest_*`` calls.
    """

    model_name: str                  # registered adapter: "yolo" | "rtdetr"
    hyperparameters: dict[str, Any]  # opaque search space: static/categorical/int/float keys

    # Canonical data inputs (COCO JSON + image paths), shared by every trial
    coco_json: str   # coco annotations for all frames
    images_dir: str  # base dir containing extracted video frames
    split_json: str  # json defining train/val/test splits

    # Optuna study parameters
    study_args: StudyArgs                 # optuna.create_study kwargs
    optimize_args: OptimizeArgs = field(default_factory=OptimizeArgs)  # optuna.study.optimize kwargs

    model_weights: str | None = None  # weights filename/path shared by every trial
    metric: str = "map50_95"         # key from MetricsDict to optimize
    output_dir: str = "outputs"
    wandb_project: str | None = None
    random_seed: int = 42

    # Fields holding filesystem paths — serialized repo-relative, resolved on load.
    # NOTE: `storage` is intentionally excluded (it is a DB URL, not a path).
    PATH_FIELDS = ("coco_json", "images_dir", "split_json", "output_dir")

    @classmethod
    def from_yaml(cls, path: str) -> "HPOConfig":
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

        raw["study_args"] = StudyArgs(**(raw.get("study_args") or {}))
        raw["optimize_args"] = OptimizeArgs(**(raw.get("optimize_args") or {}))

        return cls(**raw)

    def to_serializable_dict(self) -> dict[str, Any]:
        """Dict for study config with path fields stored relative to the repo root."""
        data = asdict(self)
        for key in self.PATH_FIELDS:
            if data.get(key) is not None:
                data[key] = to_repo_rel(data[key])
        return data

    @classmethod
    def from_stored_dict(cls, raw: dict[str, Any]) -> "HPOConfig":
        """Reconstruct from a stored config dict, resolving repo-relative paths.

        Unknown keys are ignored.
        """
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in raw.items() if k in valid_fields}
        for key in cls.PATH_FIELDS:
            if kwargs.get(key) is not None:
                kwargs[key] = str(resolve(kwargs[key]))
        # asdict() serializes study_args/optimize_args as nested dicts; rebuild the instances.
        if isinstance(kwargs.get("study_args"), dict):
            kwargs["study_args"] = StudyArgs(**kwargs["study_args"])
        if isinstance(kwargs.get("optimize_args"), dict):
            kwargs["optimize_args"] = OptimizeArgs(**kwargs["optimize_args"])
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

        # Check the optimization target is a real metric
        if self.metric not in VALID_METRICS:
            raise ValueError(
                f"metric must be one of {VALID_METRICS}, got: {self.metric!r}"
            )

        # Check the study direction is valid
        if self.study_args.direction not in ("maximize", "minimize"):
            raise ValueError(
                f"direction must be 'maximize' or 'minimize', got: {self.study_args.direction!r}"
            )

        # Check trial count is positive
        if self.optimize_args.n_trials <= 0:
            raise ValueError(f"n_trials must be positive, got: {self.optimize_args.n_trials}")

        # Light structural check of the search space; HPOptimizer validates range specs.
        if not isinstance(self.hyperparameters, dict):
            raise ValueError(
                f"hyperparameters must be a dict of search-space tiers, got: "
                f"{type(self.hyperparameters).__name__}"
            )
        unknown_tiers = set(self.hyperparameters) - set(HPARAM_CATEGORIES)
        if unknown_tiers:
            raise ValueError(
                f"hyperparameters has unknown tier(s) {sorted(unknown_tiers)}; "
                f"expected a subset of {HPARAM_CATEGORIES}"
            )
