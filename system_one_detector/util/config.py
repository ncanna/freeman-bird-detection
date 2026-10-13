"""YAML config for a System One frame experiment.

Paths are resolved relative to the YAML file, matching ``hlwdetector.config``.
This is not an ``ExperimentConfig`` and will not load in ``ExperimentRunner``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


logger = logging.getLogger(__name__)

_PATH_FIELDS = (
    "coco_json",
    "split_json",
    "images_dir",
    "videos_dir",
    "output_dir",
    "adapter_dir",
    "base_model_dir",
)

KNOWN_BACKENDS = {
    "jev",
    "laya",
    "jev_omni",
    "clef",
    "djev",
    "jevimage",
    "omni_jev",
    "openai_decisions",
    "systemone_http",
    "mock",
}

# Backends whose published schema has no audio or video input.
IMAGE_ONLY_BACKENDS = {"clef", "djev", "jevimage"}
AUDIO_CAPABLE_BACKENDS = {"jev_omni", "omni_jev", "systemone_http", "openai_decisions"}


@dataclass
class SystemOneConfig:
    config_name: str
    backend: str

    model: str | None = None
    coco_json: str | None = None
    split_json: str | None = None
    images_dir: str | None = None
    videos_dir: str | None = None
    output_dir: str = "outputs"
    split: str = "test"
    limit: int | None = None
    question_ids: list[str] = field(
        default_factory=lambda: [
            "bird_present",
            "bird_count",
            "largest_bird_cell",
            "bird_behavior",
        ]
    )
    extra_questions: dict[str, dict] = field(default_factory=dict)
    modality: str = "image"
    device: str | None = None
    api_key_env: str = "TYPESAFE_API_KEY"
    base_url: str | None = None
    endpoint: str | None = None
    send_media: bool = False
    media_style: str = "onejev"
    checkpoint: str | None = None
    subfolder: str | None = None
    use_router: bool = True
    timeout_s: float = 120.0
    noul_threshold: float = 0.5
    seed: int = 42

    # Clef / Workers AI
    account_id: str | None = None
    account_id_env: str = "CLOUDFLARE_ACCOUNT_ID"
    auth_token_env: str = "CLOUDFLARE_AUTH_TOKEN"

    # Djev diffusion sampling
    samples: int = 1
    isolation: str = "joint"
    score_mode: str = "categorical"

    # jevimage
    encoder: str | None = None

    # OmniJev (tinnel123)
    adapter_dir: str | None = None
    base_model_dir: str | None = None
    noul_bias: float | None = None
    no_panels: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "SystemOneConfig":
        yaml_path = Path(path).resolve()
        base_dir = yaml_path.parent
        with open(yaml_path, "r") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config must be a mapping: {yaml_path}")

        def resolve(val: str | None) -> str | None:
            if val is None:
                return None
            candidate = Path(val)
            if not candidate.is_absolute():
                candidate = (base_dir / candidate).resolve()
            return str(candidate)

        for key in _PATH_FIELDS:
            if key in raw and raw[key] is not None:
                raw[key] = resolve(raw[key])
        return cls(**raw)

    def validate(self) -> None:
        if self.backend not in KNOWN_BACKENDS:
            raise ValueError(
                f"unknown backend {self.backend!r}. Expected one of {sorted(KNOWN_BACKENDS)}."
            )
        if self.modality not in {"text", "image", "audio", "video"}:
            raise ValueError(f"modality must be text, image, audio, or video; got {self.modality!r}")
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train, val, or test; got {self.split!r}")
        if self.images_dir is None and self.coco_json is None:
            raise ValueError("set images_dir or coco_json so frames can be listed")
        if self.backend == "systemone_http" and not (self.base_url or self.endpoint):
            raise ValueError("systemone_http requires base_url or endpoint")
        if self.backend == "systemone_http" and self.media_style not in {"onejev", "ollama"}:
            raise ValueError("systemone_http media_style must be 'onejev' or 'ollama'")
        if self.backend == "systemone_http" and self.media_style == "ollama" and not self.send_media:
            raise ValueError("Ollama image requests require send_media: true")
        if self.backend == "clef" and self.model not in (None, "clef", "clef-flash"):
            raise ValueError(f"clef model must be clef or clef-flash; got {self.model!r}")
        if self.backend == "clef" and self.modality in {"audio", "video"}:
            raise ValueError(
                "Clef's published input schema has images only. Set modality: image and point "
                "images_dir at extracted frames."
            )
        if self.backend == "djev" and self.modality in {"audio", "video"}:
            raise ValueError("Djev takes images, not audio or video. Send frames as images.")
        if self.backend == "jevimage" and self.modality not in {"text", "image"}:
            raise ValueError(f"jevimage is image-only; got modality {self.modality!r}")
        if self.backend == "omni_jev" and not (self.adapter_dir and self.base_model_dir):
            raise ValueError("omni_jev requires adapter_dir and base_model_dir from the OmniJev repo")
        if self.backend == "djev" and self.isolation == "independent" and self.score_mode == "categorical":
            logger.warning(
                "djev: isolation=independent with score_mode=categorical re-sends the state per "
                "question, so input tokens scale with the question count."
            )
