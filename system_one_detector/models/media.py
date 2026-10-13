"""Encode a local frame the way OneJev's client documents ``media``."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path


def data_uri(path: str) -> str:
    file_path = Path(path)
    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def base64_image(path: str) -> str:
    """Return raw base64 bytes for Ollama's local ``/v1/systemone`` API."""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def onejev_media(path: str, modality: str) -> list[dict[str, str]]:
    """Media list from the OneJev README: ``[{"type": "image", "data": data_uri(...)}]``."""
    kind = modality if modality in {"image", "audio", "video"} else "image"
    return [{"type": kind, "data": data_uri(path)}]


CLEF_IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def clef_images(paths: list[str]) -> list[str]:
    """Clef takes up to 4 embedded PNG, JPEG or WebP images as data URLs.

    Remote URLs are not accepted, and video has no field in its input schema,
    so the caller must pass image paths.
    """
    images: list[str] = []
    for path in paths[:4]:
        suffix = Path(path).suffix.lower()
        if suffix not in CLEF_IMAGE_TYPES:
            raise ValueError(
                f"Clef accepts PNG, JPEG or WebP images; got {suffix or 'no extension'} for {path}"
            )
        images.append(data_uri(path))
    return images


def djev_images(path: str) -> list[str]:
    """Djev takes at most one static image, as a data URL."""
    suffix = Path(path).suffix.lower()
    if suffix not in CLEF_IMAGE_TYPES:
        raise ValueError(f"Djev accepts JPEG, PNG or WebP; got {suffix or 'no extension'} for {path}")
    return [data_uri(path)]


def state_with_image_placeholder(state, placeholder: str = "<image:1>"):
    """OneJev's example puts ``<image:1>`` in the state so the model can attend to media."""
    if isinstance(state, dict):
        updated = dict(state)
        updated.setdefault("frame", placeholder)
        return updated
    return {"text": state, "frame": placeholder}
