"""Run typed questions over a frame split and score them against COCO labels."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from system_one_detector.datasets.frames import FrameExample, load_frames
from system_one_detector.models.base import DecisionBackend
from system_one_detector.util.config import SystemOneConfig
from system_one_detector.util.metrics import aggregate

logger = logging.getLogger(__name__)


def _progress(iterable, total: int):
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc="decisions")


def evaluate_examples(
    examples: list[FrameExample],
    backend: DecisionBackend,
    *,
    noul_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Call the backend once per frame. A failed frame is recorded, not fatal."""
    if examples and not backend.sees_pixels and examples[0].modality != "text":
        logger.warning(
            "%s does not see pixels. modality=%s is ignored; only the text state is sent. "
            "Use jev_omni, or systemone_http with send_media, for the image.",
            backend.name,
            examples[0].modality,
        )
    rows: list[dict[str, Any]] = []
    for example in _progress(examples, len(examples)):
        row: dict[str, Any] = {
            "frame_id": example.frame_id,
            "file_name": example.file_name,
            "video_stem": example.video_stem,
            "media_path": example.media_path,
            "labels": example.labels,
            "error": None,
        }
        try:
            result = backend.decide(
                example.state,
                example.questions,
                media_path=example.media_path,
                modality=example.modality if backend.sees_pixels else "text",
            )
            row["model"] = result.model
            row["answers"] = {qid: answer.to_dict() for qid, answer in result.answers.items()}
            row["usage"] = result.usage
        except Exception as exc:  # noqa: BLE001 — one bad frame should not drop the split
            logger.exception("decision failed for %s", example.file_name)
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    metrics = aggregate(rows, noul_threshold=noul_threshold)
    return rows, metrics


def run_experiment(config: SystemOneConfig, *, dry_run: bool = False) -> Path:
    """Load a split, ask the configured backend, and write decisions plus metrics."""
    from system_one_detector.models import build_backend
    from system_one_detector.models.mock import MockBackend

    config.validate()
    examples, questions = load_frames(config)
    backend: DecisionBackend = MockBackend() if dry_run else build_backend(config)
    logger.info(
        "Running %s on %d %s frame(s) with questions %s",
        backend.name,
        len(examples),
        config.split,
        list(questions),
    )
    rows, metrics = evaluate_examples(
        examples,
        backend,
        noul_threshold=config.noul_threshold,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config.output_dir) / f"{config.config_name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_name": config.config_name,
        "backend": "mock" if dry_run else config.backend,
        "model": config.model,
        "split": config.split,
        "modality": config.modality,
        "sees_pixels": backend.sees_pixels,
        "questions": questions,
        "frames": rows,
    }
    (out_dir / "decisions.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Wrote %s", out_dir)
    return out_dir
